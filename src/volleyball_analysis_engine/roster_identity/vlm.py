"""Candidate-constrained jersey reading.

The model is asked "which of these players is this?", never "who is this?".  Open-set
identification invites confabulation; a closed candidate list turns the task into a
multiple-choice question and makes an explicit "unknown" answer meaningful.

Runs a local checkpoint by default: the input is full match footage, so keeping it on the
machine is simpler than shipping frames to a hosted API, and latency is measurable without
network variance.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .records import JerseyReading, RosterCandidate

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

SYSTEM_PROMPT = (
    "You are a precise sports video analyst. You read jersey numbers from "
    "low-resolution volleyball broadcast frames. You never guess."
)

USER_PROMPT = """You are identifying one volleyball player.

All provided images show the SAME tracked player, taken from different frames of one rally.
The large tiles are full-body crops; the smaller strip below (if present) shows zoomed-in
torso regions where the jersey number would appear.

{team_line}

The only possible players are:

{candidate_lines}

Instructions:
- Inspect ALL images together. A number unreadable in one frame may be readable in another.
- Jersey numbers are printed on BOTH the front and the back, so either view is usable.
- Seen from the side the number is compressed and often unreadable; rely on frames where the
  torso faces toward or away from the camera.
- Use the visible jersey number as the strongest evidence. Jersey colour is supporting
  evidence only.
- Each candidate lists the kit that player actually wears. A libero wears a contrasting kit
  from the rest of the team, so colour alone never rules a candidate in or out.
- Do NOT invent a player outside the candidate list.
- If you cannot reliably determine the number, return decision "unknown". Returning
  "unknown" is CORRECT behaviour when the images are insufficient. Do not guess.

- Also list your next best alternatives. A later stage resolves clashes between players who
  appear in the same frame, and it needs somewhere to fall back to.

Return JSON only, no other text:

{{"roster_player_id": "<id from list or null>", "jersey_number": "<digits or null>", \
"decision": "candidate" or "unknown", "confidence": "high" or "medium" or "low", \
"evidence": ["jersey_number" and/or "jersey_color" and/or "elimination"], \
"alternatives": ["<second best id or omit>", "<third best id or omit>"]}}"""


def build_prompt(candidates: list[RosterCandidate], team_name: str | None) -> str:
    """Render the candidate-constrained question for one tracklet."""
    lines = [
        f"{position}. Player ID {candidate.roster_player_id} - jersey number "
        f"{candidate.jersey_number} - team {candidate.team_id} - "
        f"wears a {candidate.jersey_description} jersey"
        for position, candidate in enumerate(candidates, 1)
    ]
    team_line = (
        f"The player belongs to team {team_name}."
        if team_name
        else "The team is NOT given - determine it from the jersey colour as well as the number."
    )
    return USER_PROMPT.format(team_line=team_line, candidate_lines="\n".join(lines))


def parse_response(text: str, valid_ids: set[str]) -> JerseyReading:
    """Lenient extraction, strict validation.

    The model routinely wraps JSON in prose or code fences, so the object is pulled out with
    a regex; but a player outside the candidate list is always rejected.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return JerseyReading(None, None, "unknown", "low", raw_response=text)
    try:
        payload: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JerseyReading(None, None, "unknown", "low", raw_response=text)
    player_id = payload.get("roster_player_id") or payload.get("player_id")
    if isinstance(player_id, str):
        player_id = player_id.strip()
    if player_id not in valid_ids:
        player_id = None
    evidence: Any = payload.get("evidence") or []
    number: Any = payload.get("jersey_number")
    alternatives: list[Any] = payload.get("alternatives") or []
    ranking: list[str] = []
    for entry in [player_id, *alternatives]:
        if isinstance(entry, str) and entry.strip() in valid_ids and entry.strip() not in ranking:
            ranking.append(entry.strip())
    return JerseyReading(
        roster_player_id=player_id,
        jersey_number=str(number) if number is not None else None,
        decision="candidate" if player_id else "unknown",
        confidence=str(payload.get("confidence", "low")).lower(),  # type: ignore[arg-type]
        evidence=(
            tuple(str(item) for item in cast("list[Any]", evidence))
            if isinstance(evidence, list)
            else (str(evidence),)
        ),
        ranking=tuple(ranking),
        raw_response=text,
    )


@dataclass(frozen=True, slots=True)
class VlmSettings:
    """Which checkpoint to run and where."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 300


class JerseyIdentifier:
    """Reads one contact sheet and returns the chosen candidate.

    One GPU, one request at a time.
    """

    def __init__(self, settings: VlmSettings) -> None:
        """Load the checkpoint onto the configured device."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[settings.dtype]
        self._settings = settings
        # Transformers ships no type information for these classes.
        processor_factory = cast("Any", AutoProcessor)
        model_factory = cast("Any", Qwen3VLForConditionalGeneration)
        self._processor: Any = processor_factory.from_pretrained(settings.model_id)
        self._model: Any = (
            model_factory.from_pretrained(settings.model_id, dtype=dtype)
            .to(settings.device)
            .eval()
        )

    def identify(
        self,
        sheet: Image,
        candidates: list[RosterCandidate],
        team_name: str | None = None,
    ) -> JerseyReading:
        """Read one contact sheet and return the chosen candidate."""
        import torch

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_prompt(candidates, team_name)},
                ],
            },
        ]
        chat: Any = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs: Any = self._processor(text=[chat], images=[sheet], return_tensors="pt").to(
            self._settings.device
        )
        started = time.time()
        with torch.no_grad():
            generated: Any = self._model.generate(
                **inputs, max_new_tokens=self._settings.max_new_tokens, do_sample=False
            )
        elapsed = time.time() - started
        text = cast(
            "str",
            self._processor.decode(
                generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ),
        )
        reading = parse_response(text, {c.roster_player_id for c in candidates})
        return JerseyReading(
            roster_player_id=reading.roster_player_id,
            jersey_number=reading.jersey_number,
            decision=reading.decision,
            confidence=reading.confidence,
            evidence=reading.evidence,
            ranking=reading.ranking,
            raw_response=reading.raw_response,
            latency_s=round(elapsed, 2),
        )
