"""CLI for the isolated, resumable predictions JSON importer."""

# A CLI is expected to print progress and translates dynamically typed wire
# messages at its edge.
# ruff: noqa: ANN401, D103, EM101, EM102, PLR2004, T201, TRY003

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect

from volleyball_analysis_engine.predictions_import import (
    build_prediction_index,
    create_import_worker,
    create_plan,
    load_plan,
    load_prediction_index,
    save_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream a predictions JSON export into VollyAI")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="index the JSON and detect rally candidates")
    plan.add_argument("--predictions", type=Path, required=True)
    plan.add_argument("--index", type=Path, required=True)
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--smooth-seconds", type=float, default=0.5)
    plan.add_argument("--max-gap-seconds", type=float, default=1.0)
    plan.add_argument("--min-active-seconds", type=float, default=0.75)
    plan.add_argument("--padding-before-seconds", type=float, default=2.0)
    plan.add_argument("--padding-after-seconds", type=float, default=1.0)

    submit = sub.add_parser("submit", help="create and submit rallies through annotation WS")
    _add_submit_arguments(submit)

    worker = sub.add_parser("worker", help="serve imported analysis through Provider Work WS")
    _add_worker_arguments(worker)

    all_command = sub.add_parser("all", help="run the Provider Work client and submit rallies")
    _add_submit_arguments(all_command)
    _add_worker_arguments(all_command, include_plan=False)
    return parser


def _add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--capture-session-id", required=True)
    parser.add_argument("--server-http", default="http://127.0.0.1:4000")
    parser.add_argument("--annotation-ws", default="ws://127.0.0.1:4000/ws/annotations")
    parser.add_argument("--user-token", default=os.getenv("VOLLYAI_USER_TOKEN"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--media-wait-seconds", type=float, default=1800.0)


def _add_worker_arguments(parser: argparse.ArgumentParser, *, include_plan: bool = True) -> None:
    if include_plan:
        parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--server-ws", default="ws://127.0.0.1:4000/api/v2/ai/providers/ws")
    parser.add_argument("--token", default=os.getenv("VOLLYAI_TOKEN"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--completion-timeout-seconds", type=float, default=0.0)


def _headers(token: str | None) -> dict[str, str]:
    return {} if not token else {"Authorization": f"Bearer {token}"}


def _cursor(descriptor: dict[str, Any], capture_time_us: int) -> dict[str, Any]:
    origin = int(descriptor["presentation_origin_capture_us"])
    if capture_time_us < origin:
        raise ValueError("playback window does not cover requested capture time")
    return {
        "playback_window_id": descriptor["playback_window_id"],
        "mapping_version": descriptor["mapping_version"],
        "player_media_time_us": str(capture_time_us - origin),
        "observation_source": "current_time_fallback",
        "presented_frames": None,
        "seek_generation": 0,
        "cursor_status": "ready",
    }


async def _playback_window(
    client: httpx.AsyncClient,
    *,
    server_http: str,
    capture_session_id: str,
    start_time_us: int,
    end_time_us: int,
    wait_seconds: float,
) -> dict[str, Any]:
    target = (start_time_us + end_time_us) // 2
    back = max(5_000_000, target - start_time_us + 2_000_000)
    forward = max(5_000_000, end_time_us - target + 2_000_000)
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        response = await client.post(
            f"{server_http.rstrip('/')}/api/v1/media/playback-windows",
            json={
                "schema_version": "1.0.0",
                "capture_session_id": capture_session_id,
                "mode": "archive",
                "target_capture_time_us": str(target),
                "requested_back_us": str(back),
                "requested_forward_us": str(forward),
            },
        )
        if response.status_code < 400:
            return cast("dict[str, Any]", response.json())
        if response.status_code != 409 or asyncio.get_running_loop().time() >= deadline:
            response.raise_for_status()
        await asyncio.sleep(5)


def _ws_url(base: str, **query: str) -> str:
    parsed = urlparse(base)
    return urlunparse(parsed._replace(query=urlencode(query)))


async def _next_command_response(socket: Any, command_id: str) -> dict[str, Any]:
    while True:
        payload = json.loads(await socket.recv())
        if payload.get("command_id") != command_id:
            continue
        if payload.get("type") == "command_rejected":
            raise RuntimeError(
                f"annotation command rejected: {payload.get('code')}: {payload.get('message')}"
            )
        if payload.get("type") == "command_ack":
            return cast("dict[str, Any]", payload)


async def _send_command(socket: Any, payload: dict[str, Any]) -> dict[str, Any]:
    await socket.send(json.dumps(payload, separators=(",", ":")))
    return await _next_command_response(socket, str(payload["command_id"]))


async def _send_command_when_media_ready(
    socket: Any,
    payload: dict[str, Any],
    *,
    wait_seconds: float,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        attempt = {**payload, "command_id": str(uuid4())}
        try:
            return await _send_command(socket, attempt)
        except RuntimeError as error:
            if "MEDIA_NOT_READY" not in str(error) or asyncio.get_running_loop().time() >= deadline:
                raise
            await asyncio.sleep(2)


async def submit_rallies(args: argparse.Namespace) -> int:
    plan_path = cast("Path", args.plan).resolve()
    plan = load_plan(plan_path)
    room_id = f"match:{args.match_id}:capture:{args.capture_session_id}".lower()
    device_session_id = str(uuid4())
    annotation_url = _ws_url(
        args.annotation_ws,
        room_id=room_id,
        device_session_id=device_session_id,
        presence_nickname="predictions-importer",
    )
    headers = _headers(args.user_token)
    submitted = 0
    async with (
        httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30, read=30)) as client,
        connect(annotation_url, additional_headers=headers, ping_interval=None) as socket,
    ):
        open_snapshots: dict[str, dict[str, Any]] = {}
        while True:
            ready = json.loads(await socket.recv())
            if ready.get("type") == "rally_snapshot" and isinstance(ready.get("rally_id"), str):
                open_snapshots[str(ready["rally_id"])] = cast("dict[str, Any]", ready)
            if ready.get("type") == "connection_ready":
                break
        segments = cast("list[dict[str, Any]]", plan["segments"])
        for segment in segments:
            if segment.get("submission_id"):
                continue
            if args.limit is not None and submitted >= args.limit:
                break
            start_time_us = int(segment["start_time_us"])
            end_time_us = int(segment["end_time_us"])
            descriptor = await _playback_window(
                client,
                server_http=args.server_http,
                capture_session_id=args.capture_session_id,
                start_time_us=start_time_us,
                end_time_us=end_time_us,
                wait_seconds=args.media_wait_seconds,
            )
            rally_id = str(segment.get("rally_id") or uuid4())
            # Bind the rally before SUBMIT_RALLY can enqueue Provider Work.  The
            # worker reloads this plan for every offer, so persisting here closes
            # the small race between submission and the worker-side allow-list.
            segment["rally_id"] = rally_id
            save_plan(plan_path, plan)
            snapshot = open_snapshots.get(rally_id)
            snapshot_value = snapshot.get("snapshot") if snapshot else None
            snapshot_payload = (
                cast("dict[str, Any]", snapshot_value) if isinstance(snapshot_value, dict) else {}
            )
            boundaries_value = snapshot_payload.get("boundaries")
            boundaries = (
                [
                    cast("dict[str, Any]", item)
                    for item in cast("list[Any]", boundaries_value)
                    if isinstance(item, dict)
                ]
                if isinstance(boundaries_value, list)
                else []
            )
            has_start = any(item.get("kind") == "start" for item in boundaries)
            has_end = any(item.get("kind") == "end" for item in boundaries)
            revision = str(snapshot.get("revision", "0")) if snapshot else "0"
            if not has_start:
                start_ack = await _send_command_when_media_ready(
                    socket,
                    {
                        "schema_version": "4.0.0",
                        "room_id": room_id,
                        "base_revision": revision,
                        "rally_id": rally_id,
                        "kind": "START_RALLY",
                        "payload": {"playback_cursor": _cursor(descriptor, start_time_us)},
                    },
                    wait_seconds=args.media_wait_seconds,
                )
                revision = str(start_ack["result_revision"])
            if not has_end:
                end_ack = await _send_command_when_media_ready(
                    socket,
                    {
                        "schema_version": "4.0.0",
                        "room_id": room_id,
                        "base_revision": revision,
                        "rally_id": rally_id,
                        "kind": "END_RALLY",
                        "payload": {"playback_cursor": _cursor(descriptor, end_time_us)},
                    },
                    wait_seconds=args.media_wait_seconds,
                )
                revision = str(end_ack["result_revision"])
            submit_ack = await _send_command(
                socket,
                {
                    "schema_version": "4.0.0",
                    "command_id": str(uuid4()),
                    "room_id": room_id,
                    "base_revision": revision,
                    "rally_id": rally_id,
                    "kind": "SUBMIT_RALLY",
                    "payload": {},
                },
            )
            segment["submission_id"] = submit_ack["effects"]["submission_id"]
            save_plan(plan_path, plan)
            submitted += 1
            print(
                f"submitted segment {segment['segment_index']} rally={rally_id} "
                f"frames={segment['source_start_frame']}:{segment['source_end_frame_exclusive']}",
                flush=True,
            )
    return submitted


async def run_worker(args: argparse.Namespace) -> None:
    if not args.token:
        raise ValueError("AI worker token is required (--token or VOLLYAI_TOKEN)")
    client, handlers = create_import_worker(
        plan_path=cast("Path", args.plan).resolve(),
        server_ws_url=args.server_ws,
        token=args.token,
        workspace=cast("Path", args.workspace).resolve(),
        instance_id=args.instance_id,
    )
    await client.run_forever(cast("Any", handlers))


async def run_all(args: argparse.Namespace) -> None:
    if not args.token:
        raise ValueError("AI worker token is required (--token or VOLLYAI_TOKEN)")
    client, handlers = create_import_worker(
        plan_path=cast("Path", args.plan).resolve(),
        server_ws_url=args.server_ws,
        token=args.token,
        workspace=cast("Path", args.workspace).resolve(),
        instance_id=args.instance_id,
    )
    worker_task = asyncio.create_task(client.run_forever(cast("Any", handlers)))
    try:
        await asyncio.sleep(1)
        await submit_rallies(args)
        submitted_rallies = [
            str(segment["rally_id"])
            for segment in cast("list[dict[str, Any]]", load_plan(args.plan)["segments"])
            if segment.get("submission_id") and segment.get("rally_id")
        ]
        deadline = (
            None
            if args.completion_timeout_seconds <= 0
            else asyncio.get_running_loop().time() + args.completion_timeout_seconds
        )
        status_dir = cast("Path", args.workspace).resolve() / "status"
        while submitted_rallies:
            completed = 0
            for rally_id in submitted_rallies:
                completed += int((status_dir / f"{rally_id}.json").exists())
            print(f"completed {completed}/{len(submitted_rallies)} submitted rallies", flush=True)
            if completed == len(submitted_rallies):
                break
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("timed out waiting for imported analysis jobs")
            if worker_task.done():
                await worker_task
            await asyncio.sleep(5)
    finally:
        await client.stop()
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        if args.index.exists():
            index = load_prediction_index(args.index)
            if index.source_path != args.predictions.resolve():
                raise ValueError("existing index belongs to a different predictions source")
            print(f"reusing index with {index.metadata.frame_count} frames", flush=True)
        else:
            index = build_prediction_index(
                args.predictions,
                args.index,
                report=lambda count: print(f"indexed {count} frames", flush=True),
            )
        plan = create_plan(
            args.index,
            args.plan,
            smooth_seconds=args.smooth_seconds,
            max_gap_seconds=args.max_gap_seconds,
            min_active_seconds=args.min_active_seconds,
            padding_before_seconds=args.padding_before_seconds,
            padding_after_seconds=args.padding_after_seconds,
        )
        summary = (
            f"indexed {index.metadata.frame_count} frames; "
            f"detected {len(plan['segments'])} segments"
        )
        print(summary)
        return 0
    if args.command == "submit":
        asyncio.run(submit_rallies(args))
        return 0
    if args.command == "worker":
        asyncio.run(run_worker(args))
        return 0
    asyncio.run(run_all(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
