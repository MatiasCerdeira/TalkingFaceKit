from dataclasses import replace

import pytest

from talkingfacekit.visual import (
    VisualObservation,
    VisualPolicy,
    VisualRegion,
    VisualResult,
    assess_visual_suitability,
)

STRICT_POLICY = VisualPolicy(
    maximum_absolute_yaw_degrees=25,
    maximum_absolute_pitch_degrees=20,
    minimum_face_size_pixels=96,
    minimum_face_width_fraction=0.08,
)


def observation(
    time: float,
    *,
    yaw: float = 0,
    pitch: float = 0,
    roll: float = 0,
    box: tuple[float, float, float, float] = (100, 100, 200, 200),
    face_id: int = 1,
) -> VisualObservation:
    return VisualObservation(time, face_id, box, (yaw, pitch, roll))


def assess(
    values: list[VisualObservation],
    *,
    end: float = 8,
    policy: VisualPolicy | None = None,
    intervals: tuple[VisualRegion, ...] | None = None,
) -> VisualResult:
    return assess_visual_suitability(
        intervals or (VisualRegion(0, end, 1, "candidate", ("structural_policy_passed",)),),
        values,
        image_width=1000,
        image_height=600,
        video_sample_rate_hz=25,
        policy=STRICT_POLICY if policy is None else policy,
    )


def test_sustained_turn_trims_run_and_preserves_raw_failures() -> None:
    result = assess([observation(i / 25, yaw=40 if 75 <= i < 100 else 0) for i in range(200)])
    assert [(r.source_start_seconds, r.source_end_seconds, r.status) for r in result.decisions] == [
        (0, 3, "candidate"),
        (3, 4, "rejected"),
        (4, 8, "candidate"),
    ]
    assert result.decisions[1].reason_codes == ("head_yaw_exceeded",)
    assert result.windows[1].reason_codes == ("head_yaw_exceeded",)
    assert result.decisions[0].metrics is not None
    assert result.decisions[0].metrics.passing_fraction == 1


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll", "reason"),
    [
        (26, 0, 0, "head_yaw_exceeded"),
        (-26, 0, 0, "head_yaw_exceeded"),
        (0, 21, 0, "head_pitch_exceeded"),
        (0, -21, 0, "head_pitch_exceeded"),
        (0, 0, 46, "head_roll_exceeded"),
        (0, 0, -46, "head_roll_exceeded"),
    ],
)
def test_pose_limits_are_absolute_and_axis_specific(
    yaw: float, pitch: float, roll: float, reason: str
) -> None:
    result = assess([observation(i / 25, yaw=yaw, pitch=pitch, roll=roll) for i in range(200)])
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reason_codes == (reason,)


def test_inclusive_pose_and_size_limits_and_independent_roll() -> None:
    result = assess(
        [observation(i / 25, yaw=-25, pitch=20, roll=45, box=(0, 0, 96, 96)) for i in range(200)]
    )
    assert result.decisions[0].status == "candidate"
    assert result.decisions[0].metrics is not None
    assert result.decisions[0].metrics.minimum_face_width_fraction == 0.096


@pytest.mark.parametrize(
    ("box", "reason"),
    [
        ((100, 100, 90, 200), "face_too_small_pixels"),
        ((100, 100, 150, 200), "face_too_small_relative"),
        ((-50, 100, 200, 200), "face_clipped"),
        ((950, 100, 200, 200), "face_clipped"),
        ((100, -50, 200, 200), "face_clipped"),
        ((100, 550, 200, 200), "face_clipped"),
    ],
)
def test_size_resolution_and_boundary_checks(
    box: tuple[float, float, float, float], reason: str
) -> None:
    result = assess(
        [observation(i / 25, box=box) for i in range(200)],
        policy=replace(STRICT_POLICY, minimum_face_width_fraction=0.16),
    )
    assert reason in result.decisions[0].reason_codes
    assert result.decisions[0].status == "rejected"


def test_short_bounded_turn_is_retained_with_its_actual_failure() -> None:
    result = assess([observation(i / 25, yaw=40 if 75 <= i < 80 else 0) for i in range(200)])
    assert len(result.decisions) == 1
    run = result.decisions[0]
    assert run.status == "candidate"
    assert len(run.tolerated_gap_intervals) == 1
    assert run.tolerated_gap_intervals[0] == pytest.approx((3.0, 3.2))
    assert "brief_visual_gap_tolerated" in run.reason_codes
    assert run.metrics is not None
    assert run.metrics.passing_fraction == pytest.approx(0.975)
    assert run.metrics.maximum_absolute_yaw_pitch_roll_degrees == pytest.approx((40.0, 0.0, 0.0))
    assert result.windows[1].status == "rejected"


def test_leading_trailing_failures_and_new_short_fragments_do_not_pass() -> None:
    result = assess(
        [
            observation(i / 25, yaw=40 if i < 5 or 175 <= i < 180 or i >= 195 else 0)
            for i in range(200)
        ],
        policy=replace(STRICT_POLICY, maximum_bad_gap_seconds=0),
    )
    assert [(r.source_start_seconds, r.source_end_seconds, r.status) for r in result.decisions] == [
        (0, 0.2, "rejected"),
        (0.2, 7, "candidate"),
        (7, 7.2, "rejected"),
        (7.2, 7.8, "rejected"),
        (7.8, 8, "rejected"),
    ]
    assert result.decisions[3].reason_codes == ("visual_segment_too_short",)


def test_many_short_failures_cannot_bypass_passing_fraction() -> None:
    result = assess([observation(i / 25, yaw=40 if i % 2 else 0) for i in range(200)])
    assert all(run.status != "candidate" for run in result.decisions)
    assert result.decisions[0].reason_codes == ("visual_passing_fraction_insufficient",)


def test_missing_face_and_unavailable_pose_are_uncertain() -> None:
    result = assess([])
    assert result.decisions[0].status == "uncertain"
    assert result.decisions[0].metrics is not None
    assert result.decisions[0].metrics.observed_fraction == 0
    result = assess(
        [replace(observation(i / 25), head_pose_yaw_pitch_roll_degrees=None) for i in range(200)]
    )
    assert result.decisions[0].status == "uncertain"
    assert result.decisions[0].reason_codes == ("visual_pose_unavailable",)
    assert (
        assess([observation(i / 25, yaw=999) for i in range(200)]).decisions[0].status
        == "uncertain"
    )


def test_missing_time_is_not_interpolated_or_counted_as_passing() -> None:
    result = assess([observation(i / 25) for i in range(200) if not 75 <= i < 80])
    run = result.decisions[0]
    assert run.status == "candidate"
    assert len(run.tolerated_gap_intervals) == 1
    assert run.tolerated_gap_intervals[0] == pytest.approx((3.0, 3.2))
    assert run.metrics is not None
    assert run.metrics.observed_fraction == pytest.approx(0.975)
    assert run.metrics.passing_fraction == pytest.approx(0.975)
    result = assess([observation(i / 25) for i in range(200) if i % 2 == 0])
    assert result.decisions[0].status == "uncertain"
    assert result.decisions[0].reason_codes == ("visual_evidence_insufficient",)


def test_never_bridges_shot_boundaries_or_restores_sync_rejections() -> None:
    regions = (
        VisualRegion(10, 11, 1, "candidate", ("structural_policy_passed",)),
        VisualRegion(11, 12, 1, "candidate", ("structural_policy_passed",)),
        VisualRegion(12, 15, 2, "rejected", ("audio_video_offset",)),
        VisualRegion(15, 18, 2, "uncertain", ("synchronization_not_verified",)),
    )
    result = assess([observation(10 + i / 25) for i in range(200)], intervals=regions)
    assert result.decisions[0].reason_codes == ("visual_segment_too_short",)
    assert result.decisions[1].reason_codes == ("visual_segment_too_short",)
    assert result.decisions[2].reason_codes == ("audio_video_offset",)
    assert result.decisions[3].status == "uncertain"
    assert result.decisions[2].metrics is None


def test_partial_grid_boundary_and_face_identity_use_original_timeline() -> None:
    result = assess(
        [observation(10 + i / 25) for i in range(100)]
        + [observation(10 + i / 25, yaw=50, face_id=2) for i in range(100)],
        intervals=(VisualRegion(10.01, 13.91, 1, "candidate", ("input",)),),
    )
    assert len(result.decisions) == 1
    assert result.decisions[0].status == "candidate"
    assert result.decisions[0].source_start_seconds == 10.01
    assert result.decisions[0].source_end_seconds == 13.91


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_absolute_yaw_degrees", -1),
        ("maximum_absolute_pitch_degrees", 181),
        ("maximum_absolute_roll_degrees", float("nan")),
        ("minimum_face_size_pixels", True),
        ("minimum_face_width_fraction", 1.1),
        ("maximum_clipped_fraction", -0.1),
        ("maximum_bad_gap_seconds", float("inf")),
        ("minimum_duration_seconds", 0),
        ("minimum_observed_fraction", 0),
        ("minimum_passing_fraction", 1.1),
    ],
)
def test_invalid_policy_fails(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(VisualPolicy(), **{field: value})


def test_duplicate_observations_and_overlapping_inputs_fail() -> None:
    with pytest.raises(ValueError, match="unique"):
        assess([observation(0), observation(0)])
    with pytest.raises(ValueError, match="ordered"):
        assess(
            [],
            intervals=(
                VisualRegion(1, 4, 1, "candidate", ("input",)),
                VisualRegion(3, 5, 1, "candidate", ("input",)),
            ),
        )


def test_new_measurement_ends_previous_sample_support() -> None:
    result = assess(
        [
            observation(0),
            observation(0.01, yaw=40),
            observation(0.04),
            observation(0.05, yaw=40),
            observation(0.07),
        ],
        end=0.08,
        policy=replace(STRICT_POLICY, maximum_bad_gap_seconds=0, minimum_duration_seconds=0.001),
    )
    assert [item.status for item in result.decisions] == [
        "candidate",
        "rejected",
        "candidate",
        "rejected",
        "candidate",
    ]
    assert [item.source_end_seconds for item in result.decisions] == pytest.approx(
        [0.01, 0.04, 0.05, 0.07, 0.08]
    )


def test_opposite_head_turns_cannot_cancel_to_a_frontal_average() -> None:
    result = assess([observation(i / 25, yaw=30 if i % 2 else -30) for i in range(200)])
    assert all(item.status == "rejected" for item in result.decisions)
    assert result.decisions[0].reason_codes == ("head_yaw_exceeded",)


def test_default_policy_accepts_useful_three_quarter_views_and_lower_resolution() -> None:
    observations = [
        observation(i / 25, yaw=65 if i % 2 else -40, pitch=30, box=(100, 100, 80, 80))
        for i in range(200)
    ]
    relaxed = assess(observations, policy=VisualPolicy())
    assert len(relaxed.decisions) == 1
    assert relaxed.decisions[0].status == "candidate"
    assert relaxed.decisions[0].source_start_seconds == 0
    assert relaxed.decisions[0].source_end_seconds == 8
    strict = assess(observations, policy=STRICT_POLICY)
    assert all(item.status == "rejected" for item in strict.decisions)


@pytest.mark.parametrize(
    ("yaw", "pitch", "size", "reason"),
    [
        (85, 0, 100, "head_yaw_exceeded"),
        (0, 50, 100, "head_pitch_exceeded"),
        (0, 0, 60, "face_too_small_pixels"),
    ],
)
def test_default_policy_still_rejects_extreme_pose_and_insufficient_detail(
    yaw: float, pitch: float, size: float, reason: str
) -> None:
    result = assess(
        [observation(i / 25, yaw=yaw, pitch=pitch, box=(100, 100, size, size)) for i in range(200)],
        policy=VisualPolicy(),
    )
    assert result.decisions[0].status == "rejected"
    assert reason in result.decisions[0].reason_codes
