"""Reprocess imported rallies through the existing Provider Work pipeline.

This operational helper is intentionally separate from the annotation importer.
It never creates or edits rally drafts/submissions.  Each selected rally is sent
through the server's retryProcessing mutation, which creates a new versioned AI
job and AnalysisRun for the existing immutable submission.
"""

# This CLI prints bounded progress and validates dynamic HTTP responses at the edge.
# ruff: noqa: EM101, EM102, T201, TRY003

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

import httpx

from volleyball_analysis_engine.predictions_import import create_import_worker, load_plan

RETRY_PROCESSING_MUTATION = """
mutation RetryImportedRally($rallyId: ID!) {
  retryProcessing(input: { rallyId: $rallyId }) {
    rallyId
    submissionId
    status
    retriedStage
  }
}
"""
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
TOKEN_DELETE_ATTEMPTS = 10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reprocess predictions imports into new versioned AnalysisRuns"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--server-http", default="http://127.0.0.1:4000")
    parser.add_argument("--server-ws", default="ws://127.0.0.1:4000/api/v2/ai/providers/ws")
    parser.add_argument(
        "--match-id",
        help="scope Provider Work offers to this match UUID (required unless --queue-only)",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--user-token", default=os.getenv("VOLLYAI_USER_TOKEN"))
    parser.add_argument("--worker-token", default=os.getenv("VOLLYAI_TOKEN"))
    parser.add_argument("--worker-token-name", default="predictions-path-reprocessor")
    parser.add_argument("--instance-id")
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--rally-id", action="append", dest="rally_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="call retryProcessing without starting Provider Workers",
    )
    parser.add_argument(
        "--resume-queued",
        action="store_true",
        help="serve already queued retries without calling retryProcessing again",
    )
    parser.add_argument("--completion-timeout-seconds", type=float, default=7200.0)
    return parser


def _headers(token: str | None) -> dict[str, str]:
    return {} if not token else {"Authorization": f"Bearer {token}"}


def _scoped_server_ws(server_ws: str, match_id: str) -> str:
    normalized_match_id = str(UUID(match_id))
    parts = urlsplit(server_ws)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    existing = query.get("match_id")
    if existing is not None and existing != normalized_match_id:
        raise ValueError(
            f"server WS is already scoped to match {existing}, not {normalized_match_id}"
        )
    query["match_id"] = normalized_match_id
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _selected_rallies(plan: dict[str, Any], requested: list[str] | None) -> list[str]:
    bound = [
        str(segment["rally_id"])
        for segment in cast("list[dict[str, Any]]", plan["segments"])
        if segment.get("rally_id") and segment.get("submission_id")
    ]
    if not requested:
        return bound
    requested_set = set(requested)
    unknown = requested_set.difference(bound)
    if unknown:
        raise ValueError(f"rallies are not bound to this import plan: {sorted(unknown)}")
    return [rally_id for rally_id in bound if rally_id in requested_set]


async def _create_worker_token(
    client: httpx.AsyncClient,
    *,
    server_http: str,
    name: str,
) -> tuple[str, str]:
    response = await client.post(
        f"{server_http.rstrip('/')}/api/v1/operations/ai-worker-tokens",
        json={"name": name},
    )
    response.raise_for_status()
    payload = cast("dict[str, Any]", response.json())
    access_token = cast("dict[str, Any]", payload.get("access_token"))
    token_id = access_token.get("id") if access_token else None
    token = payload.get("token")
    if not isinstance(token_id, str) or not isinstance(token, str):
        raise TypeError("AI worker token response is invalid")
    return token_id, token


async def _delete_worker_token(
    client: httpx.AsyncClient,
    *,
    server_http: str,
    token_id: str,
) -> None:
    url = f"{server_http.rstrip('/')}/api/v1/operations/ai-worker-tokens/{token_id}"
    for attempt in range(TOKEN_DELETE_ATTEMPTS):
        response = await client.delete(url)
        if response.status_code < HTTP_BAD_REQUEST or response.status_code == HTTP_NOT_FOUND:
            return
        if response.status_code != HTTP_CONFLICT or attempt == TOKEN_DELETE_ATTEMPTS - 1:
            response.raise_for_status()
        await asyncio.sleep(1)


async def _retry_rally(
    client: httpx.AsyncClient,
    *,
    server_http: str,
    rally_id: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{server_http.rstrip('/')}/graphql",
        json={"query": RETRY_PROCESSING_MUTATION, "variables": {"rallyId": rally_id}},
    )
    response.raise_for_status()
    payload = cast("dict[str, Any]", response.json())
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"retryProcessing failed for rally {rally_id}: {json.dumps(errors)}")
    data = cast("dict[str, Any]", payload.get("data"))
    result = data.get("retryProcessing") if data else None
    if not isinstance(result, dict):
        raise TypeError(f"retryProcessing returned no state for rally {rally_id}")
    return cast("dict[str, Any]", result)


async def _queue_rallies(
    client: httpx.AsyncClient,
    *,
    server_http: str,
    rally_ids: list[str],
) -> None:
    for index, rally_id in enumerate(rally_ids, start=1):
        state = await _retry_rally(client, server_http=server_http, rally_id=rally_id)
        print(
            f"queued {index}/{len(rally_ids)} rally={rally_id} "
            f"status={state.get('status')} stage={state.get('retriedStage')}",
            flush=True,
        )


async def _wait_for_completion(
    *,
    workspace: Path,
    rally_ids: list[str],
    started_at_ns: int,
    timeout_seconds: float,
    worker_tasks: list[asyncio.Task[Any]],
) -> None:
    status_dir = workspace / "status"
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_completed = -1
    while True:
        completed = sum(
            (status_dir / f"{rally_id}.json").exists()
            and (status_dir / f"{rally_id}.json").stat().st_mtime_ns >= started_at_ns
            for rally_id in rally_ids
        )
        if completed != last_completed:
            print(f"completed {completed}/{len(rally_ids)} rallies", flush=True)
            last_completed = completed
        if completed == len(rally_ids):
            return
        for worker_task in worker_tasks:
            if worker_task.done():
                await worker_task
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("timed out waiting for reprocessed analysis jobs")
        await asyncio.sleep(5)


async def reprocess(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912
    """Reprocess selected imported rallies without changing their submissions."""
    plan_path = cast("Path", args.plan).resolve()
    workspace = cast("Path", args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    rally_ids = _selected_rallies(load_plan(plan_path), args.rally_ids)
    if args.limit is not None:
        rally_ids = rally_ids[: args.limit]
    if not rally_ids:
        raise ValueError("no submitted rallies were selected")
    if args.worker_count < 1:
        raise ValueError("worker count must be positive")
    if not args.queue_only and not args.match_id:
        raise ValueError("--match-id is required when starting Provider Workers")
    scoped_server_ws = (
        args.server_ws
        if args.queue_only
        else _scoped_server_ws(cast("str", args.server_ws), cast("str", args.match_id))
    )

    headers = _headers(args.user_token)
    timeout = httpx.Timeout(30, read=30)
    token_id: str | None = None
    worker_token = args.worker_token
    workers: list[Any] = []
    worker_tasks: list[asyncio.Task[Any]] = []
    started_at_ns = time.time_ns()
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        try:
            if args.queue_only:
                await _queue_rallies(client, server_http=args.server_http, rally_ids=rally_ids)
                return
            if not worker_token:
                token_id, worker_token = await _create_worker_token(
                    client,
                    server_http=args.server_http,
                    name=args.worker_token_name,
                )
                print(f"created temporary AI worker token id={token_id}", flush=True)

            for index in range(args.worker_count):
                instance_id = args.instance_id
                if instance_id and args.worker_count > 1:
                    instance_id = f"{instance_id}-{index + 1}"
                worker, handlers = create_import_worker(
                    plan_path=plan_path,
                    server_ws_url=scoped_server_ws,
                    token=worker_token,
                    workspace=workspace,
                    instance_id=instance_id,
                )
                workers.append(worker)
                worker_tasks.append(asyncio.create_task(worker.run_forever(cast("Any", handlers))))
            await asyncio.sleep(1)
            for worker_task in worker_tasks:
                if worker_task.done():
                    await worker_task

            if not args.resume_queued:
                await _queue_rallies(client, server_http=args.server_http, rally_ids=rally_ids)
            await _wait_for_completion(
                workspace=workspace,
                rally_ids=rally_ids,
                started_at_ns=started_at_ns,
                timeout_seconds=args.completion_timeout_seconds,
                worker_tasks=worker_tasks,
            )
        finally:
            for worker in workers:
                await worker.stop()
            for worker_task in worker_tasks:
                worker_task.cancel()
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            if token_id is not None:
                await _delete_worker_token(
                    client,
                    server_http=args.server_http,
                    token_id=token_id,
                )
                print(f"deleted temporary AI worker token id={token_id}", flush=True)


def main() -> int:
    """Run the reprocessor CLI."""
    asyncio.run(reprocess(_parser().parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
