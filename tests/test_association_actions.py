"""Action taxonomy regression tests."""

from volleyball_analysis_engine.association import classify_action


def test_action_uses_review_taxonomy_for_cross_court_path() -> None:
    assert classify_action((0.25, 0.5), (0.75, 0.5)) == ("Spiking", True)


def test_action_uses_review_taxonomy_for_same_court_path() -> None:
    assert classify_action((0.25, 0.4), (0.3, 0.6)) == ("Waiting", False)
