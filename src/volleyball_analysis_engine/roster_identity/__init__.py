"""Downstream match/roster identity service.

Consumes finished `AnalysisResult` artifacts and assigns a match-scoped roster identity
(team + jersey number) to each clip-local canonical track, without rewriting clip-local
track IDs.  Identity evidence is the jersey number read by a candidate-constrained VLM.

Only the pure records and roster types are re-exported here.  The frame-selection, VLM and
service modules pull in heavy optional dependencies (OpenCV, Torch, Ultralytics,
Transformers) and are imported directly by callers that need them, so importing the engine
package stays cheap.
"""

from .records import (
    CanonicalTracklet,
    CourtSide,
    IdentityAssignment,
    IdentityStatus,
    JerseyReading,
    RosterCandidate,
    SelectedFrame,
    TrackletObservation,
)
from .roster import Roster, Team, player_id

__all__ = [
    "CanonicalTracklet",
    "CourtSide",
    "IdentityAssignment",
    "IdentityStatus",
    "JerseyReading",
    "Roster",
    "RosterCandidate",
    "SelectedFrame",
    "Team",
    "TrackletObservation",
    "player_id",
]
