from typing import Literal

import pytest

from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkCandidatePolicy,
    DeepTalkFaceObservation,
    DeepTalkPreliminarySegment,
    DeepTalkScoreWindow,
    DeepTalkSpeechInterval,
    build_candidate_runs,
    build_preliminary_segments,
)


def make_face(timestamp_seconds: float, face_id: int) -> DeepTalkFaceObservation:
    return DeepTalkFaceObservation(
        source_timestamp_seconds=timestamp_seconds,
        face_id=face_id,
        bounding_box_xywh=(10.0, 20.0, 100.0, 120.0),
        head_pose_yaw_pitch_roll_degrees=(2.0, -1.0, 0.5),
        raw_face_quality_score=0.8,
    )


def make_result(
    *,
    faces: tuple[DeepTalkFaceObservation, ...],
    raw_scores: dict[int, float],
    speech_status: Literal["confirmed", "rejected", "incomplete"] = "confirmed",
) -> DeepTalkAnalysisResult:
    return DeepTalkAnalysisResult(
        face_observations=faces,
        speech_intervals=(
            DeepTalkSpeechInterval(
                source_start_seconds=0.2,
                source_end_seconds=0.8,
                status=speech_status,
            ),
        ),
        score_windows=(
            DeepTalkScoreWindow(
                source_start_seconds=0.0,
                source_end_seconds=1.0,
                raw_scores=raw_scores,
            ),
        ),
        provenance=DeepTalkAnalysisProvenance(backend_version="0.3.1"),
    )


def make_candidate_segment(
    start_seconds: float,
    end_seconds: float,
    face_id: int = 7,
    raw_score: float = 1.0,
) -> DeepTalkPreliminarySegment:
    return DeepTalkPreliminarySegment(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        status="candidate",
        face_id=face_id,
        reason_codes=("single_visible_face", "positive_raw_active_speaker_score"),
        visible_face_ids=(face_id,),
        raw_score=raw_score,
    )


def make_run_result(
    *,
    observations: tuple[DeepTalkFaceObservation, ...],
    segments: tuple[DeepTalkPreliminarySegment, ...],
) -> DeepTalkAnalysisResult:
    return DeepTalkAnalysisResult(
        face_observations=observations,
        speech_intervals=(),
        score_windows=(),
        provenance=DeepTalkAnalysisProvenance(backend_version="0.3.1"),
        preliminary_segments=segments,
    )


def test_splits_score_windows_at_speech_bounds_and_marks_a_candidate() -> None:
    result = make_result(
        faces=(make_face(0.3, 7), make_face(0.7, 7)),
        raw_scores={7: 1.25},
    )

    segments = build_preliminary_segments(result)

    assert [
        (
            segment.source_start_seconds,
            segment.source_end_seconds,
            segment.status,
            segment.face_id,
            segment.reason_codes,
        )
        for segment in segments
    ] == [
        (0.0, 0.2, "rejected", None, ("no_speech",)),
        (
            0.2,
            0.8,
            "candidate",
            7,
            ("single_visible_face", "positive_raw_active_speaker_score"),
        ),
        (0.8, 1.0, "rejected", None, ("no_speech",)),
    ]
    assert segments[1].visible_face_ids == (7,)
    assert segments[1].raw_score == 1.25


def test_rejects_speech_without_a_visible_face() -> None:
    segments = build_preliminary_segments(make_result(faces=(), raw_scores={}))

    assert segments[1].status == "rejected"
    assert segments[1].reason_codes == ("no_visible_face",)


def test_rejects_multiple_simultaneous_faces() -> None:
    result = make_result(
        faces=(make_face(0.4, 1), make_face(0.4, 2)),
        raw_scores={1: 1.5, 2: 0.75},
    )

    segment = build_preliminary_segments(result)[1]

    assert segment.status == "rejected"
    assert segment.reason_codes == ("multiple_visible_faces",)
    assert segment.visible_face_ids == (1, 2)
    assert segment.face_id is None


def test_rejects_a_face_identity_change_within_the_window() -> None:
    result = make_result(
        faces=(make_face(0.3, 1), make_face(0.7, 2)),
        raw_scores={1: 1.5, 2: 1.25},
    )

    segment = build_preliminary_segments(result)[1]

    assert segment.status == "rejected"
    assert segment.reason_codes == ("face_identity_changed",)
    assert segment.visible_face_ids == (1, 2)


def test_rejects_a_non_positive_score_and_preserves_it() -> None:
    result = make_result(faces=(make_face(0.4, 7),), raw_scores={7: -0.25})

    segment = build_preliminary_segments(result)[1]

    assert segment.status == "rejected"
    assert segment.face_id == 7
    assert segment.reason_codes == ("non_positive_active_speaker_score",)
    assert segment.raw_score == -0.25


def test_marks_a_visible_face_without_a_score_as_uncertain() -> None:
    result = make_result(faces=(make_face(0.4, 7),), raw_scores={})

    segment = build_preliminary_segments(result)[1]

    assert segment.status == "uncertain"
    assert segment.face_id == 7
    assert segment.reason_codes == ("no_active_speaker_score",)
    assert segment.raw_score is None


def test_rejected_vad_interval_does_not_count_as_speech() -> None:
    result = make_result(
        faces=(make_face(0.4, 7),),
        raw_scores={7: 1.25},
        speech_status="rejected",
    )

    segments = build_preliminary_segments(result)

    assert len(segments) == 1
    assert segments[0].source_start_seconds == 0.0
    assert segments[0].source_end_seconds == 1.0
    assert segments[0].status == "rejected"
    assert segments[0].reason_codes == ("no_speech",)


def test_coalesces_sub_nanosecond_gaps_between_speech_intervals() -> None:
    result = DeepTalkAnalysisResult(
        face_observations=(make_face(0.25, 7), make_face(0.75, 7)),
        speech_intervals=(
            DeepTalkSpeechInterval(0.0, 0.5, "confirmed"),
            DeepTalkSpeechInterval(0.5 + 5e-10, 1.0, "incomplete"),
        ),
        score_windows=(DeepTalkScoreWindow(0.0, 1.0, {7: 1.25}),),
        provenance=DeepTalkAnalysisProvenance(backend_version="0.3.1"),
    )

    segments = build_preliminary_segments(result)

    assert [(segment.source_start_seconds, segment.source_end_seconds) for segment in segments] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]
    assert all(segment.status == "candidate" for segment in segments)


def test_candidate_contract_requires_a_positive_score() -> None:
    with pytest.raises(ValueError, match="require one face and a positive raw score"):
        DeepTalkPreliminarySegment(
            source_start_seconds=0.0,
            source_end_seconds=1.0,
            status="candidate",
            face_id=7,
            reason_codes=("single_visible_face", "positive_raw_active_speaker_score"),
            visible_face_ids=(7,),
            raw_score=0.0,
        )


def test_merges_adjacent_same_face_candidates_and_summarizes_evidence() -> None:
    result = make_run_result(
        observations=tuple(make_face(index / 25, 7) for index in range(50)),
        segments=(
            make_candidate_segment(0.0, 1.0, raw_score=1.0),
            make_candidate_segment(1.0, 2.0, raw_score=2.0),
        ),
    )

    runs = build_candidate_runs(result)

    assert len(runs) == 1
    run = runs[0]
    assert run.source_start_seconds == 0.0
    assert run.source_end_seconds == 2.0
    assert run.duration_seconds == 2.0
    assert run.status == "candidate"
    assert run.face_id == 7
    assert run.reason_codes == ("structural_policy_passed",)
    assert run.preliminary_segment_count == 2
    assert run.face_observation_count == 50
    assert run.expected_face_observation_count == 50
    assert run.face_visibility_fraction == 1.0
    assert run.maximum_face_gap_seconds == pytest.approx(0.0, abs=1e-12)
    assert (run.raw_score_min, run.raw_score_mean, run.raw_score_max) == (1.0, 1.5, 2.0)


def test_candidate_run_score_mean_is_weighted_by_segment_duration() -> None:
    result = make_run_result(
        observations=tuple(make_face(index / 25, 7) for index in range(50)),
        segments=(
            make_candidate_segment(0.0, 0.5, raw_score=1.0),
            make_candidate_segment(0.5, 2.0, raw_score=3.0),
        ),
    )

    run = build_candidate_runs(result)[0]

    assert run.raw_score_mean == 2.5


def test_rejected_or_different_face_regions_form_hard_run_boundaries() -> None:
    rejected = DeepTalkPreliminarySegment(
        source_start_seconds=1.0,
        source_end_seconds=2.0,
        status="rejected",
        face_id=None,
        reason_codes=("no_visible_face",),
        visible_face_ids=(),
        raw_score=None,
    )
    result = make_run_result(
        observations=tuple(
            [*(make_face(index / 25, 7) for index in range(25))]
            + [*(make_face(2.0 + index / 25, 8) for index in range(25))]
        ),
        segments=(
            make_candidate_segment(0.0, 1.0, face_id=7),
            rejected,
            make_candidate_segment(2.0, 3.0, face_id=8),
        ),
    )
    permissive = DeepTalkCandidatePolicy(minimum_duration_seconds=0.5)

    runs = build_candidate_runs(result, permissive)

    assert [(run.source_start_seconds, run.source_end_seconds, run.face_id) for run in runs] == [
        (0.0, 1.0, 7),
        (2.0, 3.0, 8),
    ]
    assert all(run.status == "candidate" for run in runs)


def test_rejects_a_candidate_run_that_is_too_short() -> None:
    result = make_run_result(
        observations=tuple(make_face(index / 25, 7) for index in range(25)),
        segments=(make_candidate_segment(0.0, 1.0),),
    )

    run = build_candidate_runs(result)[0]

    assert run.status == "rejected"
    assert run.reason_codes == ("candidate_too_short",)


def test_rejects_insufficient_face_coverage() -> None:
    result = make_run_result(
        observations=tuple(make_face(index / 25, 7) for index in range(30)),
        segments=(make_candidate_segment(0.0, 2.0),),
    )
    policy = DeepTalkCandidatePolicy(maximum_face_gap_seconds=2.0)

    run = build_candidate_runs(result, policy)[0]

    assert run.face_visibility_fraction == 0.6
    assert run.status == "rejected"
    assert run.reason_codes == ("insufficient_face_coverage",)


def test_rejects_an_unstable_visibility_gap_even_with_sufficient_coverage() -> None:
    observed_indices = (*range(21), *range(29, 50))
    result = make_run_result(
        observations=tuple(make_face(index / 25, 7) for index in observed_indices),
        segments=(make_candidate_segment(0.0, 2.0),),
    )
    policy = DeepTalkCandidatePolicy(minimum_face_coverage=0.8)

    run = build_candidate_runs(result, policy)[0]

    assert run.face_visibility_fraction == 0.84
    assert run.maximum_face_gap_seconds == pytest.approx(0.32)
    assert run.status == "rejected"
    assert run.reason_codes == ("unstable_face_visibility",)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("minimum_duration_seconds", 0.0, "minimum_duration_seconds"),
        ("minimum_face_coverage", 1.1, "minimum_face_coverage"),
        ("maximum_face_gap_seconds", -0.1, "maximum_face_gap_seconds"),
    ],
)
def test_candidate_policy_rejects_invalid_thresholds(
    keyword: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DeepTalkCandidatePolicy(**{keyword: value})
