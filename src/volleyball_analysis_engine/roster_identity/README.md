# roster_identity

Downstream match/roster identity service: assigns a real player identity (team + jersey
number) to every clip-local canonical track, **without rewriting clip-local track IDs**.

This is the layer the engine README already anticipates:

> `AnalysisResult.extensions.reid_feature_bank` … so a **downstream match/roster identity
> service** can consolidate clips **without rewriting clip-local track IDs**.

## Why jersey numbers and not appearance

Same-team players wear identical kit, shorts and shoes. In a 1280x720 broadcast frame a
player is about 150 px tall, so apart from the printed number there is nothing in the image
that separates two teammates. Measured on the development match:

| Within-clip pair | Mean cosine similarity (Sports OSNet) |
| --- | ---: |
| Same person | 0.821 |
| **Different person, both on screen at the same moment** | **0.763** |
| Different person, different phase of the rally | 0.694 |

A *different* player who happens to be on screen at the same time scores higher than the
*same* player seen later in the rally. The embedding partly encodes **when the frame was
taken**, not who is in it. That is why identity here comes from reading the number.

Appearance is still useful, but for the easier question — "are these two the same person" —
which is what `cluster.py` uses it for.

## Where the inputs come from

Everything about the match is **caller-supplied metadata**, never inferred:

```jsonc
// roster.json — supplied by the front end, which holds the match record
{
  "match_id": "...",
  "teams": [
    {
      "team_id": "TPE",
      "display_name": "Chinese Taipei",
      "jersey_description": "dark navy blue",
      "libero_jersey_description": "red",          // a libero wears a contrasting kit by rule
      "players": [
        {"jersey_number": 3,  "name": null, "role": "libero"},
        {"jersey_number": 11, "name": null, "role": null}
      ]
    }
  ]
}
```

`Roster.candidates(team_id, on_court)` narrows the list. Narrower is strictly better:
pass the six players actually on court for the rally whenever the line-up is known.

Getting the libero kit wrong is not a cosmetic error. The prompt tells the model to use
colour as supporting evidence, so describing a libero with the team colour actively rules
out the correct answer. Adding libero descriptions moved Top-1 from 61.3% to 66.4% and cut
abstentions from 21.2% to 14.6%.

Tracklets come from a finished offline analysis directory, in priority order:

1. `AnalysisResult.extensions.fixed_roster_reid` — canonical IDs, court sides and
   cannot-link constraints, already computed by the engine.
2. `tracks.jsonl` alone — the equivalent structure is rebuilt (`inputs.py`) for artifacts
   produced before that bank existed. The bank's appearance descriptors are ignored either
   way; identity here is the number.

## Pipeline

```text
canonical tracklet
  -> frames.py    frame quality selection, frontality-weighted
  -> frames.py    pose-guided torso crops
  -> sheet.py     one upscaled contact sheet per tracklet
  -> vlm.py       candidate-constrained jersey reading
  -> cluster.py   group a clip's tracklets, vote the group's identity   (optional)
  -> assign.py    resolve clashes against co-visibility
  -> service.py   roster-identity.json
```

### frames.py — the dominant factor is orientation, not sharpness

A side-on broadcast camera means most frames show a player in profile, where the number is
compressed onto a near-vertical strip and is unreadable at any resolution. `frontality`
(shoulder span over torso height, from COCO-17 keypoints) is weighted highest of the five
quality terms. Only 28% of candidate frames have frontality >= 0.6; 71% of *selected*
frames do.

A failed pose is also a signal in its own right: it usually means the crop contains no whole
player because the tracker drifted onto empty court, hence `NO_POSE_PENALTY`.

### sheet.py — why 4x upscaling

A jersey number occupies roughly 20-25 px. A ViT patch is 14-16 px, so at native scale the
whole number falls inside one or two patches and carries no usable signal. Resampling does
not add information, it moves the information that is already there onto a scale the model
can see.

### assign.py — the co-visibility constraint is pairwise

Two tracklets visible in the same frame are provably different people. Measured: tracklets
in a clash were wrong 34% of the time, tracklets with no clash only 7%.

**The constraint is pairwise, not one-to-one.** A clip routinely holds far more tracklets
than players because the tracker fragments each player into disjoint spans — 32 tracklets
for 12 people in the development match. An earlier version solved the clip as a Hungarian
assignment and rejected most legal repeats: coverage 76% -> 46%, accuracy 61% -> 38%. The
correct model is a list colouring of the co-visibility graph, solved greedily in order of
evidence strength.

### cluster.py — group first, then name

One clear view of a number should name every fragment of that player, and one misread should
be outvoted. Grouping is where the headroom is: with oracle grouping, Top-1 rises from
64.2% to 83.9%.

Link score combines appearance with **re-entry in court space**, plus a kinematic gate (a
player cannot cross the court faster than a sprint). Measured pairwise AUC:

| Cue | AUC |
| --- | ---: |
| Court distance between exit and re-entry | **0.867** |
| Sports OSNet cosine | 0.826 |
| Time gap | 0.691 |
| Combined | 0.874 |

A pair with no court positions falls back to appearance alone. Scoring a missing cue as zero
penalises the pair for the court detector's gaps and silently suppresses most merges.

## Output

`roster-identity.json` is a standalone artifact. Clip-local track IDs are never rewritten, so
a clip can be re-analysed without invalidating identities a human already confirmed.

```jsonc
{
  "schema_version": "1.0.0",
  "scope": "match",
  "match_id": "...",
  "producer": {"name": "volleyball-analysis-engine.roster_identity", "build_id": "..."},
  "method": {"identity_evidence": "jersey_number", "vlm_model_id": "...", "candidate_scope": "all"},
  "identity_contract": "team_and_jersey_number",
  "assignments": [
    {
      "clip_id": "clip-1",
      "canonical_track_id": 1,
      "track_ids": [1],
      "roster_player_id": "TPE_04",
      "status": "auto_assigned",          // auto_assigned | human_review | unknown
      "method": "jersey_vlm+cannot_link",
      "rule": "top_choice",               // top_choice | resolved_clash | no_consistent_identity | ...
      "court_side": "left",
      "selected_frame_indices": ["4", "14", "26"],
      "jersey": {"roster_player_id": "TPE_04", "jersey_number": "4", "confidence": "high", "ranking": ["TPE_04"]},
      "quality_flags": []
    }
  ],
  "summary": {"assignment_count": 176, "auto_assigned_count": 102, "human_review_count": 74, "unknown_count": 0}
}
```

Frame indices are strings, matching `analysis-result.json`. Abstaining is a correct outcome:
a wrong identity silently corrupts every statistic derived from it, an abstention costs one
human click.

## Measured results

Development set: China vs Chinese Taipei, 7 clips, 176 canonical tracklets, 137 with a
human identity label, 15 players. Top-1 is forced-choice over all 137.

| Configuration | Top-1 | Auto coverage | Auto precision |
| --- | ---: | ---: | ---: |
| Per-tracklet, 15 candidates (no team given) | 61.3% | 75.9% | 80.8% |
| Per-tracklet, team-constrained | 64.2% | 76.6% | 83.8% |
| Team-constrained + co-visibility assignment | 56.9% | 64.2% | **88.6%** |
| **Cluster + group vote (appearance + re-entry)** | **69.3%** | 86.9% | 79.8% |
| Oracle grouping (ceiling) | 83.9% | — | — |

Latency: about 1.8 s per tracklet on one RTX 4090 with Qwen3-VL-8B, one call per tracklet.

## Things that were tried and did not work

Recorded so they are not retried blind.

| Attempt | Result |
| --- | ---: |
| Hungarian one-to-one assignment per clip | Top-1 61.3% -> 38.0%; the constraint is pairwise |
| Down-weighting low court-visibility tracks in the vote | 69.3% -> 67.9%; those tracks are right 55% of the time, not never |
| Hard-filtering low court-visibility tracks | 69.3% -> 60.6% over the same denominator; removes 22 real players to remove 37 close-ups |
| Clustering on appearance alone | 66.4%; re-entry position is the stronger cue |
| Asking the VLM for ranked alternatives | Returned `"alternatives": []` for 117 of 176; the fallback path is mostly starved |

## Known limitations

1. **The VLM's self-reported confidence is not discriminative.** 134 of 176 readings said
   `high` while precision was about 80%. Any rule that gates on it is gating on noise. A
   calibrated score — candidate log-probabilities, or agreement across differently sampled
   contact sheets — is the most valuable missing piece.
2. **Cluster purity caps the fused result** at 89.1%, against the 83.9% oracle ceiling.
   Better appearance features would help here; the bar to beat is AUC 0.826 on within-clip
   pairs, and it must clear roughly 0.9 to be worth the swap.
3. **Single match, single venue, single broadcaster.** Kit colours, number fonts and camera
   position have no variation in the development set.
4. **The team-constrained numbers assume the team is known per tracklet.** That is obtainable
   without annotation from `court_pos` plus which team is on which side, but it is not wired
   up; the 15-candidate figures are the zero-assumption baseline.
5. **Upstream footage is not guaranteed to be main-camera only.** 38 of 176 tracklets in the
   development set came from close-ups and replays. If the front end is expected to supply
   clean footage, confirm that layer actually exists.

## Development harness

The evaluation harness lives outside this repository, at
`my_player_activity/roster_reid/` on the development machine, because it depends on private
human annotations:

- `run_reid.py` — per-tracklet identification and evaluation against ground truth
- `fuse.py` — clustering plus group voting, threshold sweeps
- `make_video.py` — draws assignments back onto the source clips for visual review
- `roster.json` — the match roster in the shape above

Ground truth is `{clip_id}#{canonical_track_id} -> roster_player_id`. Any equivalent mapping
works; nothing in this package depends on that harness.
