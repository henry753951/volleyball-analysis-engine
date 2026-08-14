"""Match roster and candidate selection.

Everything in here is **caller-supplied match metadata**, not something this service infers.
Squad numbers, team kit colours, libero kit colours and the six players currently on court
all come from the front end, which already holds the match record.  Hard-coding any of it
would silently break on the next match.

The candidate list is the single most valuable constraint available: it turns open-set
recognition into a multiple-choice question.  Narrower is better, so pass `on_court`
whenever the rally's line-up is known - six candidates beat a fourteen-player squad.

A libero wears a contrasting kit by rule.  Describing that player with the team colour is
actively harmful: the model is told to use colour as supporting evidence and would then rule
out the correct answer.  Kit description is therefore per player, not per team.

Roster JSON shape::

    {
      "match_id": "...",
      "teams": [
        {
          "team_id": "TPE",
          "display_name": "Chinese Taipei",
          "jersey_description": "dark navy blue",
          "libero_jersey_description": "red",
          "players": [
            {"jersey_number": 3, "name": null, "role": "libero"},
            {"jersey_number": 11, "name": null, "role": null}
          ]
        }
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .records import RosterCandidate

LIBERO_ROLE = "libero"


@dataclass(frozen=True, slots=True)
class Player:
    """One squad member as supplied by the caller."""

    jersey_number: int
    name: str | None = None
    role: str | None = None

    @property
    def is_libero(self) -> bool:
        """Whether this player wears the contrasting libero kit."""
        return (self.role or "").strip().lower() == LIBERO_ROLE


@dataclass(frozen=True, slots=True)
class Team:
    """One team's squad and kit description."""

    team_id: str
    display_name: str
    jersey_description: str
    players: tuple[Player, ...]
    libero_jersey_description: str | None = None

    def jersey_description_for(self, player: Player) -> str:
        """Kit the player actually wears, which differs for a libero."""
        if player.is_libero and self.libero_jersey_description:
            return self.libero_jersey_description
        return self.jersey_description


@dataclass(frozen=True, slots=True)
class Roster:
    """Both squads for one match."""

    match_id: str
    teams: tuple[Team, ...]

    @classmethod
    def load(cls, path: Path) -> Roster:
        """Read a roster JSON document."""
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            match_id=str(payload.get("match_id") or path.stem),
            teams=tuple(
                Team(
                    team_id=str(team["team_id"]),
                    display_name=str(team.get("display_name") or team["team_id"]),
                    jersey_description=str(team.get("jersey_description") or "unknown"),
                    libero_jersey_description=(
                        str(team["libero_jersey_description"])
                        if team.get("libero_jersey_description")
                        else None
                    ),
                    players=tuple(
                        Player(
                            jersey_number=int(player["jersey_number"]),
                            name=player.get("name"),
                            role=player.get("role"),
                        )
                        for player in sorted(
                            team["players"], key=lambda item: int(item["jersey_number"])
                        )
                    ),
                )
                for team in payload["teams"]
            ),
        )

    def candidates(
        self,
        team_id: str | None = None,
        on_court: dict[str, list[int]] | None = None,
    ) -> list[RosterCandidate]:
        """Candidates the VLM may choose from.

        `team_id` restricts to one team; `on_court` maps team_id to the jersey numbers
        currently on court, which is the narrowest and most useful constraint.
        """
        candidates: list[RosterCandidate] = []
        for team in self.teams:
            if team_id is not None and team.team_id != team_id:
                continue
            allowed = set((on_court or {}).get(team.team_id, []))
            for player in team.players:
                if allowed and player.jersey_number not in allowed:
                    continue
                candidates.append(
                    RosterCandidate(
                        roster_player_id=player_id(team.team_id, player.jersey_number),
                        jersey_number=player.jersey_number,
                        team_id=team.team_id,
                        jersey_description=team.jersey_description_for(player),
                    )
                )
        return candidates

    def team(self, team_id: str) -> Team:
        """Look a team up by id, raising when it is not in the roster."""
        for team in self.teams:
            if team.team_id == team_id:
                return team
        message = f"unknown team_id: {team_id}"
        raise KeyError(message)


def player_id(team_id: str, jersey_number: int) -> str:
    """Build a match-scoped identity from a team and a shirt number.

    Team plus number, never an anonymous sequence number: the statistics this feeds have to
    name a real player.
    """
    return f"{team_id}_{jersey_number:02d}"
