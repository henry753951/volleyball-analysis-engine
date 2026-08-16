"""CLI override coverage for optional worker capabilities."""

import pytest

from volleyball_analysis_engine.cli import (
    _parser,  # pyright: ignore[reportPrivateUsage]
    _settings_from_arguments,  # pyright: ignore[reportPrivateUsage]
)


def test_worker_vlm_cli_override_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLLYAI_REID_VLM_ENABLED", "true")
    disabled = _settings_from_arguments(_parser().parse_args(["worker", "--disable-reid-vlm"]))
    assert disabled.reid_vlm_enabled is False

    monkeypatch.setenv("VOLLYAI_REID_VLM_ENABLED", "false")
    enabled = _settings_from_arguments(_parser().parse_args(["worker", "--enable-reid-vlm"]))
    assert enabled.reid_vlm_enabled is True


def test_worker_vlm_uses_environment_without_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLLYAI_REID_VLM_ENABLED", "true")
    settings = _settings_from_arguments(_parser().parse_args(["worker"]))
    assert settings.reid_vlm_enabled is True
