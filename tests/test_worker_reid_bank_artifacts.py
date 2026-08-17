"""Worker-side ReID bank descriptor resolution invariants."""

from __future__ import annotations

from typing import Any

import pytest

from volleyball_analysis_engine.worker import match_bank_descriptor_artifacts


def test_matches_stable_bank_artifact_ids_to_ephemeral_job_inputs_by_sha256() -> None:
    bank: dict[str, Any] = {
        "evidence_artifacts": [
            {"artifact_id": "stable-a", "sha256": "a" * 64},
            {"artifact_id": "stable-b", "sha256": "b" * 64},
        ]
    }

    assert match_bank_descriptor_artifacts(
        bank,
        [
            ("job-input-b", "b" * 64, b"second"),
            ("job-input-a", "a" * 64, b"first"),
        ],
    ) == {"stable-a": b"first", "stable-b": b"second"}


def test_rejects_missing_or_extra_historical_descriptor_inputs() -> None:
    bank: dict[str, Any] = {"evidence_artifacts": [{"artifact_id": "stable-a", "sha256": "a" * 64}]}

    with pytest.raises(ValueError, match="historical descriptor inputs"):
        match_bank_descriptor_artifacts(bank, [])
    with pytest.raises(ValueError, match="historical descriptor inputs"):
        match_bank_descriptor_artifacts(
            bank,
            [
                ("job-input-a", "a" * 64, b"first"),
                ("job-input-extra", "b" * 64, b"extra"),
            ],
        )
