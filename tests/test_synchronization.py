from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from talkingfacekit.integrations.syncnet import measure_feature_offset
from talkingfacekit.synchronization import (
    SynchronizationPolicy,
    SynchronizationWindow,
    candidate_windows,
    decide_synchronization,
)


@pytest.mark.parametrize("lag", [-8, -4, 0, 2, 4, 8])
def test_recovers_signed_offset_with_equal_support(lag: int) -> None:
    visual = np.random.default_rng(12).normal(0, 4, (125, 32)).astype(np.float32)
    audio = np.roll(visual, lag, axis=0)
    window = measure_feature_offset(
        visual, audio, start_seconds=20, end_seconds=25, face_id=7, policy=SynchronizationPolicy()
    )
    assert window.estimated_offset_seconds == pytest.approx(lag / 25)
    assert window.status == ("offset_detected" if abs(lag) > 2 else "synchronized")
    assert window.confidence is not None and window.confidence > 3


def test_flat_similarity_and_search_boundary_cannot_pass() -> None:
    visual = np.random.default_rng(1).normal(0, 4, (125, 32)).astype(np.float32)
    edge = measure_feature_offset(
        visual,
        np.roll(visual, 25, axis=0),
        start_seconds=0,
        end_seconds=5,
        face_id=1,
        policy=SynchronizationPolicy(),
    )
    assert edge.status == "uncertain"
    assert "sync_search_boundary" in edge.reason_codes
    flat = measure_feature_offset(
        np.ones_like(visual),
        np.ones_like(visual),
        start_seconds=0,
        end_seconds=5,
        face_id=1,
        policy=SynchronizationPolicy(),
    )
    assert flat.status == "uncertain"
    assert "sync_low_confidence" in flat.reason_codes


def test_short_windows_report_the_actual_reduced_search_range() -> None:
    features = np.random.default_rng(1).normal(0, 4, (45, 32)).astype(np.float32)
    window = measure_feature_offset(
        features,
        features,
        start_seconds=0,
        end_seconds=2,
        face_id=1,
        policy=SynchronizationPolicy(),
    )
    assert window.search_limit_seconds == 0.4


def test_partition_covers_candidates_without_sampling_away_unmeasured_time() -> None:
    windows = candidate_windows(20, 31, SynchronizationPolicy())
    assert windows[0][0] == 20
    assert windows[-1][1] == 31
    assert all(a[1] == b[0] for a, b in pairwise(windows))
    assert all(end - start <= 5 for start, end in windows)


def test_confirmation_uncertainty_and_changing_offsets_remain_distinct() -> None:
    policy = SynchronizationPolicy()
    first = SynchronizationWindow(0, 5, 1, "synchronized", ("synchronization_passed",), 0, 8, 6, 1)
    second = replace(first, source_start_seconds=5, source_end_seconds=10)
    assert decide_synchronization(0, 10, 1, (first, second), policy).status == "candidate"
    incomplete = decide_synchronization(0, 10, 1, (first,), policy)
    assert incomplete.status == "uncertain" and incomplete.estimated_offset_seconds is None
    uncertain = replace(second, status="uncertain", reason_codes=("sync_low_confidence",))
    assert decide_synchronization(0, 10, 1, (first, uncertain), policy).status == "uncertain"
    late = replace(
        second,
        status="offset_detected",
        estimated_offset_seconds=0.32,
        reason_codes=("audio_video_offset",),
    )
    decision = decide_synchronization(0, 10, 1, (first, late), policy)
    assert decision.status == "rejected"
    assert decision.estimated_offset_seconds is None
    assert "inconsistent_audio_video_offsets" in decision.reason_codes
    assert decide_synchronization(0, 10, 1, (), policy).status == "uncertain"
    with pytest.raises(ValueError, match="same candidate face"):
        decide_synchronization(0, 10, 2, (first, second), policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tolerance_seconds", float("nan")),
        ("minimum_confidence", 0),
        ("maximum_offset_seconds", 0.01),
        ("window_seconds", 1),
    ],
)
def test_invalid_policy_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(SynchronizationPolicy(), **{field: value})
