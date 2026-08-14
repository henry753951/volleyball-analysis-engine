"""Tracking behavior required before court-side identity consolidation."""

import numpy as np

from volleyball_analysis_engine.inference import (
    HarmonicMeanTracker,
    normalize_frame_bbox,
)
from volleyball_analysis_engine.records import ReIdEmbeddingModel


def test_tracker_does_not_drop_people_before_court_side_filtering() -> None:
    tracker = HarmonicMeanTracker()
    boxes = np.asarray(
        [[index * 12.0, 10.0, index * 12.0 + 8.0, 50.0] for index in range(13)],
        dtype=np.float32,
    )
    scores = np.full((13,), 0.9, dtype=np.float32)
    embeddings = np.zeros((13, 512), dtype=np.float32)
    embeddings[np.arange(13), np.arange(13)] = 1.0

    output = tracker.update(0, boxes, scores, embeddings)

    assert len(output) == 13
    assert {item.track_id for item in output} == set(range(1, 14))


def test_tracker_uses_appearance_when_detection_order_changes() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    boxes = np.asarray(
        [[10.0, 10.0, 30.0, 60.0], [80.0, 10.0, 100.0, 60.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    embeddings = np.zeros((2, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    first = tracker.update(0, boxes, scores, embeddings)

    second = tracker.update(1, boxes[::-1].copy(), scores, embeddings[::-1].copy())

    first_by_detection = {item.detection_index: item.track_id for item in first}
    second_by_detection = {item.detection_index: item.track_id for item in second}
    assert second_by_detection[1] == first_by_detection[0]
    assert second_by_detection[0] == first_by_detection[1]


def test_tracker_keeps_identity_between_sparse_reid_frames() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    boxes = np.asarray(
        [[10.0, 10.0, 30.0, 60.0], [80.0, 10.0, 100.0, 60.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    embeddings = np.zeros((2, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    first = tracker.update(0, boxes, scores, embeddings)

    shifted = boxes + np.asarray([2.0, 0.0, 2.0, 0.0], dtype=np.float32)
    second = tracker.update(12, shifted, scores, None)

    assert [item.track_id for item in second] == [item.track_id for item in first]


def test_tracker_extrapolates_boxes_between_detector_frames() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    boxes = np.asarray([[10.0, 10.0, 30.0, 60.0]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    embeddings = np.zeros((1, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    tracker.update(0, boxes, scores, embeddings)
    shifted = boxes + np.asarray([8.0, 0.0, 8.0, 0.0], dtype=np.float32)
    tracker.update(4, shifted, scores, embeddings)

    prediction = tracker.predict(5)

    assert len(prediction) == 1
    assert prediction[0].track_id == 1
    assert prediction[0].bbox[0] > shifted[0, 0]


def test_tracker_bridges_short_miss_but_never_renders_long_lost_pool() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    boxes = np.asarray(
        [[10.0, 10.0, 30.0, 60.0], [80.0, 10.0, 100.0, 60.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    embeddings = np.zeros((2, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    first = tracker.update(0, boxes, scores, embeddings)

    short_gap = tracker.update(1, boxes[:1], scores[:1], embeddings[:1])
    tracker.update(2, boxes[:1], scores[:1], embeddings[:1])
    long_gap = tracker.update(3, boxes[:1], scores[:1], embeddings[:1])
    predicted = tracker.predict(4)

    assert {item.track_id for item in short_gap} == {item.track_id for item in first}
    assert [item.track_id for item in long_gap] == [first[0].track_id]
    assert [item.track_id for item in predicted] == [first[0].track_id]


def test_tracker_matches_against_motion_predicted_bbox() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    scores = np.asarray([0.9], dtype=np.float32)
    embedding = np.zeros((1, 512), dtype=np.float32)
    embedding[0, 0] = 1.0
    first_box = np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    first = tracker.update(0, first_box, scores, embedding)
    tracker.update(
        1,
        np.asarray([[5.0, 0.0, 15.0, 10.0]], dtype=np.float32),
        scores,
        embedding,
    )

    moved = tracker.update(
        5,
        np.asarray([[12.0, 0.0, 22.0, 10.0]], dtype=np.float32),
        scores,
        None,
    )

    assert moved[0].track_id == first[0].track_id


def test_tracker_recovers_long_gap_with_unambiguous_appearance() -> None:
    tracker = HarmonicMeanTracker(
        max_lost_frames=600,
        appearance_recovery_threshold=0.9,
        appearance_recovery_margin=0.02,
    )
    scores = np.asarray([0.9], dtype=np.float32)
    embedding = np.zeros((1, 512), dtype=np.float32)
    embedding[0, 0] = 1.0
    first = tracker.update(
        0,
        np.asarray([[0.0, 0.0, 10.0, 30.0]], dtype=np.float32),
        scores,
        embedding,
    )

    recovered = tracker.update(
        300,
        np.asarray([[200.0, 0.0, 210.0, 30.0]], dtype=np.float32),
        scores,
        embedding,
    )

    assert recovered[0].track_id == first[0].track_id


def test_tracker_rejects_ambiguous_long_gap_appearance() -> None:
    tracker = HarmonicMeanTracker(
        max_lost_frames=600,
        appearance_recovery_threshold=0.9,
        appearance_recovery_margin=0.02,
    )
    boxes = np.asarray(
        [[0.0, 0.0, 10.0, 30.0], [30.0, 0.0, 40.0, 30.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9], dtype=np.float32)
    embeddings = np.zeros((2, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.999
    embeddings[1, 1] = 0.0447
    first = tracker.update(0, boxes, scores, embeddings)

    recovered = tracker.update(
        300,
        np.asarray([[200.0, 0.0, 210.0, 30.0]], dtype=np.float32),
        scores[:1],
        embeddings[:1],
    )

    assert recovered[0].track_id not in {item.track_id for item in first}


def test_tracker_does_not_use_stale_geometry_after_recovery_window() -> None:
    tracker = HarmonicMeanTracker(
        max_lost_frames=600,
        max_geometry_lost_frames=100,
        appearance_recovery_threshold=0.9,
    )
    box = np.asarray([[0.0, 0.0, 10.0, 30.0]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    first_embedding = np.zeros((1, 512), dtype=np.float32)
    first_embedding[0, 0] = 1.0
    different_embedding = np.zeros((1, 512), dtype=np.float32)
    different_embedding[0, 1] = 1.0
    first = tracker.update(0, box, scores, first_embedding)

    returned_to_same_place = tracker.update(300, box, scores, different_embedding)

    assert returned_to_same_place[0].track_id != first[0].track_id


def test_tracker_aggregates_sparse_l2_prototypes_and_cannot_links() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2, reid_min_observations=2)
    boxes = np.asarray(
        [[10.0, 10.0, 30.0, 60.0], [80.0, 10.0, 100.0, 60.0]],
        dtype=np.float32,
    )
    first_scores = np.asarray([0.8, 0.6], dtype=np.float32)
    first_embeddings = np.zeros((2, 512), dtype=np.float32)
    first_embeddings[0, 0] = 1.0
    first_embeddings[1, 2] = 1.0
    tracker.update(0, boxes, first_scores, first_embeddings)
    tracker.update(1, boxes, np.asarray([0.5, 0.5], dtype=np.float32), None)

    last_scores = np.asarray([0.4, 0.2], dtype=np.float32)
    last_embeddings = np.zeros((2, 512), dtype=np.float32)
    last_embeddings[0, 1] = 1.0
    last_embeddings[1, 2] = 1.0
    tracker.update(2, boxes, last_scores, last_embeddings)
    snapshot = tracker.reid_feature_snapshot(
        ReIdEmbeddingModel(
            name="sports-osnet",
            checkpoint_sha256="a" * 64,
            preprocess_version="roi-align-rgb-imagenet-v1",
            dimension=512,
            distance="cosine",
        )
    )

    assert snapshot.schema_version == "1.0.0"
    assert len(snapshot.features) == 2
    first, second = snapshot.features
    assert (first.sample_count, first.first_frame_index, first.last_frame_index) == (2, 0, 2)
    assert np.isclose(first.mean_quality, 0.6)
    assert np.isclose(np.linalg.norm(first.prototype), 1.0)
    assert np.allclose(first.prototype[:3], [2**-0.5, 2**-0.5, 0.0])
    assert first.cannot_link_track_ids == (second.track_id,)
    assert second.cannot_link_track_ids == (first.track_id,)
    assert np.isclose(np.linalg.norm(second.prototype), 1.0)


def test_tracker_omits_short_or_tiny_tracks_from_reid_bank() -> None:
    tracker = HarmonicMeanTracker(match_threshold=0.2)
    large_box = np.asarray([[10.0, 10.0, 30.0, 60.0]], dtype=np.float32)
    tiny_box = np.asarray([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
    scores = np.asarray([0.8], dtype=np.float32)
    embeddings = np.zeros((1, 512), dtype=np.float32)
    embeddings[0, 0] = 1.0

    for frame_index in range(11):
        tracker.update(frame_index, large_box, scores, embeddings)
    tracker.update(11, tiny_box, scores, embeddings)
    snapshot = tracker.reid_feature_snapshot(
        ReIdEmbeddingModel(
            name="sports-osnet",
            checkpoint_sha256="a" * 64,
            preprocess_version="roi-align-rgb-imagenet-v1",
            dimension=512,
            distance="cosine",
        )
    )

    assert snapshot.features == ()


def test_detector_bbox_overshoot_is_clamped_to_video_coordinates() -> None:
    assert normalize_frame_bbox(
        (-4.75, 20.0, 1924.5, 1083.0),
        width=1920,
        height=1080,
    ) == (0.0, 20.0 / 1080.0, 1.0, 1.0)
