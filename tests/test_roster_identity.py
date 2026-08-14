"""Roster identity service regression tests."""

from volleyball_analysis_engine.roster_identity.assign import (
    AssignmentSettings,
    assign_clip,
)
from volleyball_analysis_engine.roster_identity.records import (
    CanonicalTracklet,
    JerseyReading,
    TrackletObservation,
)
from volleyball_analysis_engine.roster_identity.roster import Roster, Player, Team, player_id
from volleyball_analysis_engine.roster_identity.vlm import build_prompt, parse_response


def _tracklet(
    canonical_track_id: int,
    *,
    frames: range,
    cannot_link: tuple[int, ...] = (),
) -> CanonicalTracklet:
    return CanonicalTracklet(
        clip_id="clip-1",
        canonical_track_id=canonical_track_id,
        track_ids=(canonical_track_id,),
        court_side="left",
        observations=tuple(
            TrackletObservation(
                frame_index=index,
                frame_bbox=(0.1, 0.1, 0.2, 0.4),
                court_pos=(0.3, 0.5),
                confidence=0.9,
            )
            for index in frames
        ),
        cannot_link_canonical_track_ids=cannot_link,
    )


def _reading(player: str, *alternatives: str) -> JerseyReading:
    return JerseyReading(
        roster_player_id=player,
        jersey_number=player.split("_")[1],
        decision="candidate",
        confidence="high",
        ranking=(player, *alternatives),
    )


def _roster() -> Roster:
    return Roster(
        match_id="m1",
        teams=(
            Team(
                team_id="TPE",
                display_name="Chinese Taipei",
                jersey_description="dark navy blue",
                libero_jersey_description="red",
                players=(
                    Player(jersey_number=3, role="libero"),
                    Player(jersey_number=11),
                    Player(jersey_number=13),
                ),
            ),
        ),
    )


def test_player_id_is_team_and_padded_number() -> None:
    assert player_id("TPE", 4) == "TPE_04"


def test_libero_candidate_gets_its_own_kit_description() -> None:
    candidates = {c.roster_player_id: c for c in _roster().candidates()}
    assert candidates["TPE_03"].jersey_description == "red"
    assert candidates["TPE_11"].jersey_description == "dark navy blue"


def test_on_court_narrows_the_candidate_list() -> None:
    candidates = _roster().candidates(on_court={"TPE": [11, 13]})
    assert [c.jersey_number for c in candidates] == [11, 13]


def test_prompt_lists_every_candidate_with_its_kit() -> None:
    prompt = build_prompt(_roster().candidates(), "Chinese Taipei")
    assert "Player ID TPE_03" in prompt
    assert "wears a red jersey" in prompt


def test_parse_response_rejects_a_player_outside_the_candidate_list() -> None:
    reading = parse_response('{"roster_player_id": "TPE_99"}', {"TPE_11"})
    assert reading.roster_player_id is None
    assert reading.decision == "unknown"


def test_parse_response_keeps_alternatives_in_rank_order() -> None:
    reading = parse_response(
        '{"roster_player_id": "TPE_11", "alternatives": ["TPE_13", "TPE_99"]}',
        {"TPE_11", "TPE_13"},
    )
    assert reading.ranking == ("TPE_11", "TPE_13")


def test_co_visible_tracklets_cannot_take_the_same_identity() -> None:
    """A player cannot be in two places at once, so the weaker claim falls back."""
    first = _tracklet(1, frames=range(0, 200), cannot_link=(2,))
    second = _tracklet(2, frames=range(0, 40), cannot_link=(1,))
    readings = {first.key: _reading("TPE_11"), second.key: _reading("TPE_11", "TPE_13")}
    resolved = assign_clip(
        [first, second],
        readings,
        {first.key: ["TPE_11", "TPE_13"], second.key: ["TPE_11", "TPE_13"]},
        AssignmentSettings(),
    )
    assert resolved[first.key][0] == "TPE_11"
    assert resolved[second.key][0] == "TPE_13"
    assert resolved[second.key][2] == "resolved_clash"


def test_a_clash_with_no_alternative_abstains_instead_of_guessing() -> None:
    first = _tracklet(1, frames=range(0, 200), cannot_link=(2,))
    second = _tracklet(2, frames=range(0, 40), cannot_link=(1,))
    readings = {first.key: _reading("TPE_11"), second.key: _reading("TPE_11")}
    resolved = assign_clip(
        [first, second],
        readings,
        {first.key: ["TPE_11"], second.key: ["TPE_11"]},
        AssignmentSettings(),
    )
    assert resolved[second.key][0] is None
    assert resolved[second.key][2] == "no_consistent_identity"


def test_non_co_visible_tracklets_may_share_an_identity() -> None:
    """One player is routinely split into several disjoint spans by the tracker."""
    first = _tracklet(1, frames=range(0, 100))
    second = _tracklet(2, frames=range(200, 300))
    readings = {first.key: _reading("TPE_11"), second.key: _reading("TPE_11")}
    resolved = assign_clip(
        [first, second],
        readings,
        {first.key: ["TPE_11"], second.key: ["TPE_11"]},
        AssignmentSettings(),
    )
    assert resolved[first.key][0] == "TPE_11"
    assert resolved[second.key][0] == "TPE_11"
