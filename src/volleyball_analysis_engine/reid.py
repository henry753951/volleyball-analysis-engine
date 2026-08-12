"""Analysis-run-local ReID using projected 2D court positions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .records import CourtSide, FrameObservation, PlayerObservation


@dataclass(slots=True)
class _Identity:
    track_id: int
    side: CourtSide
    last_frame: int
    last_court_pos: tuple[float, float] | None


class CourtPositionReidentifier:
    """Merge nearby enter/leave fragments while showing at most six players per side."""

    def __init__(self, *, max_per_side: int = 6, max_distance: float = 0.12) -> None:
        """Configure the per-side capacity and nearest-position threshold."""
        self.max_per_side = max_per_side
        self.max_distance = max_distance
        self._next_id = 1
        self._source_to_canonical: dict[int, int] = {}
        self._identities: dict[int, _Identity] = {}

    def apply(self, frames: list[FrameObservation]) -> list[FrameObservation]:
        """Return frames with canonical track IDs selected in court space."""
        output: list[FrameObservation] = []
        for frame in frames:
            selected_players = self._select_side_roster(frame.players)
            used_canonical: set[int] = set()
            players = tuple(
                self._assign(player, frame.frame_index, used_canonical)
                for player in sorted(selected_players, key=self._assignment_priority)
            )
            output.append(
                FrameObservation(
                    frame_index=frame.frame_index,
                    players=players,
                    homography_available=frame.homography_available,
                )
            )
        return output

    def _select_side_roster(
        self,
        players: tuple[PlayerObservation, ...],
    ) -> tuple[PlayerObservation, ...]:
        """Keep six player observations per side, preferring stable identities."""
        selected: list[PlayerObservation] = []
        for side in ("left", "right", "unknown"):
            side_players = [player for player in players if player.court_side == side]
            if side == "unknown":
                selected.extend(side_players)
                continue
            side_players.sort(
                key=lambda player: (
                    player.source_track_id not in self._source_to_canonical,
                    self._outside_court_distance(player.court_pos),
                    player.source_track_id,
                )
            )
            selected.extend(side_players[: self.max_per_side])
        return tuple(selected)

    def _assign(
        self,
        player: PlayerObservation,
        frame_index: int,
        used_canonical: set[int],
    ) -> PlayerObservation:
        known = self._source_to_canonical.get(player.source_track_id)
        if known is None or known in used_canonical:
            known = self._choose_identity(player, frame_index, used_canonical)
            self._source_to_canonical[player.source_track_id] = known
        identity = self._identities[known]
        identity.last_frame = frame_index
        identity.last_court_pos = player.court_pos
        used_canonical.add(known)
        return player.with_identity(known)

    def _choose_identity(
        self,
        player: PlayerObservation,
        frame_index: int,
        used_canonical: set[int],
    ) -> int:
        side = player.court_side
        side_identities = [
            identity for identity in self._identities.values() if identity.side == side
        ]
        candidates = [
            identity
            for identity in side_identities
            if identity.track_id not in used_canonical and identity.last_frame < frame_index
        ]
        nearest = min(
            candidates,
            key=lambda identity: self._distance(player.court_pos, identity.last_court_pos),
            default=None,
        )
        nearest_distance = (
            float("inf")
            if nearest is None
            else self._distance(player.court_pos, nearest.last_court_pos)
        )
        if nearest is not None and nearest_distance <= self.max_distance:
            return nearest.track_id
        identity = _Identity(
            track_id=self._next_id,
            side=side,
            last_frame=frame_index,
            last_court_pos=player.court_pos,
        )
        self._identities[identity.track_id] = identity
        self._next_id += 1
        return identity.track_id

    def _assignment_priority(self, player: PlayerObservation) -> tuple[bool, float, int]:
        """Let the visible source nearest its prior canonical position keep that identity."""
        canonical = self._source_to_canonical.get(player.source_track_id)
        identity = self._identities.get(canonical) if canonical is not None else None
        return (
            identity is None,
            (
                float("inf")
                if identity is None
                else self._distance(player.court_pos, identity.last_court_pos)
            ),
            player.source_track_id,
        )

    @staticmethod
    def _distance(
        first: tuple[float, float] | None,
        second: tuple[float, float] | None,
    ) -> float:
        if first is None or second is None:
            return float("inf")
        return hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _outside_court_distance(position: tuple[float, float] | None) -> float:
        """Rank projected observations by distance outside the canonical court."""
        if position is None:
            return float("inf")
        x, y = position
        dx = max(0.0, -x, x - 1.0)
        dy = max(0.0, -y, y - 1.0)
        return hypot(dx, dy)
