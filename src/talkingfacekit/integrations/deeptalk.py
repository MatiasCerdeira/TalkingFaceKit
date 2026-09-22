"""Offline adapter for the experimental DeepTalk-ASD integration."""

from __future__ import annotations

import math
import os
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module, metadata
from itertools import chain, pairwise
from numbers import Real
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Literal, cast

import av
import numpy as np
from numpy.typing import NDArray

from talkingfacekit.audio import DecodedAudioChunk
from talkingfacekit.io.audio import stream_audio_chunks
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.video import DecodedVideoFrame

_DEEPTALK_DISTRIBUTION = "deeptalk-asd"
_DEEPTALK_SUPPORTED_VERSION = "0.3.1"
_VIDEO_FPS = 25
_FRAME_PERIOD_SECONDS = 1 / _VIDEO_FPS
_VIDEO_TIMESTAMP_TOLERANCE_SECONDS = 1e-9
_AUDIO_SAMPLE_RATE_HZ = 16_000
_AUDIO_FRAME_SAMPLE_COUNT = 480
_AUDIO_REPAIR_BLOCK_SAMPLE_COUNT = 8_192
_AUDIO_TIMESTAMP_TOLERANCE_SECONDS = 1e-3
_AUDIO_SAMPLE_COUNT_TOLERANCE = 1e-7
_SCORE_WINDOW_SECONDS = 1.0
_INSPIREFACE_RESOURCE_ENV = "INSPIREFACE_RESOURCE_PATH"
_INSPIREFACE_RESOURCE_SHA256 = "5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431"

_EventKind = Literal["video", "audio"]
_DeepTalkEvent = tuple[_EventKind, float, object]
_SpeechIntervalStatus = Literal["confirmed", "rejected", "incomplete"]
_AudioTimelineRepairKind = Literal["audio_gap_filled", "audio_overlap_trimmed"]
_PreliminarySegmentStatus = Literal["candidate", "rejected", "uncertain"]
_CandidateRunStatus = Literal["candidate", "rejected"]
_PreliminarySegmentReason = Literal[
    "no_speech",
    "no_visible_face",
    "multiple_visible_faces",
    "face_identity_changed",
    "no_active_speaker_score",
    "non_positive_active_speaker_score",
    "single_visible_face",
    "positive_raw_active_speaker_score",
]
_CandidateRunReason = Literal[
    "structural_policy_passed",
    "candidate_too_short",
    "insufficient_face_coverage",
    "unstable_face_visibility",
]
_PRELIMINARY_SEGMENT_REASON_VALUES = frozenset(
    {
        "no_speech",
        "no_visible_face",
        "multiple_visible_faces",
        "face_identity_changed",
        "no_active_speaker_score",
        "non_positive_active_speaker_score",
        "single_visible_face",
        "positive_raw_active_speaker_score",
    }
)
_CANDIDATE_RUN_REASON_VALUES = frozenset(
    {
        "structural_policy_passed",
        "candidate_too_short",
        "insufficient_face_coverage",
        "unstable_face_visibility",
    }
)


@dataclass(frozen=True, slots=True)
class DeepTalkFaceObservation:
    """Record one DeepTalk face observation on the source-media timeline.

    Parameters
    ----------
    source_timestamp_seconds
        Timestamp of the 25 Hz source-timeline slot submitted to DeepTalk, in seconds.
    face_id
        DeepTalk's stable face identity. The same identity keys score-window results.
    bounding_box_xywh
        Face rectangle ``(x, y, width, height)`` in source-frame pixels.
    head_pose_yaw_pitch_roll_degrees
        Raw InspireFace head-pose angles ``(yaw, pitch, roll)`` in degrees. They are preserved as
        backend evidence and are not interpreted as a camera-facing decision.
    raw_face_quality_score
        Raw InspireFace quality value copied from DeepTalk. Its range is backend-defined; it is
        neither calibrated nor thresholded by this adapter.
    """

    source_timestamp_seconds: float
    face_id: int
    bounding_box_xywh: tuple[float, float, float, float]
    head_pose_yaw_pitch_roll_degrees: tuple[float, float, float]
    raw_face_quality_score: float

    def __post_init__(self) -> None:
        """Validate the copied timestamp, identity, rectangle, pose, and quality evidence."""
        if not math.isfinite(self.source_timestamp_seconds) or self.source_timestamp_seconds < 0:
            raise ValueError(
                "source_timestamp_seconds must be finite and non-negative, "
                f"got {self.source_timestamp_seconds}"
            )
        if isinstance(self.face_id, bool) or not isinstance(self.face_id, int):
            raise TypeError(f"face_id must be an integer, got {type(self.face_id).__name__}")
        if len(self.bounding_box_xywh) != 4 or not all(
            math.isfinite(value) for value in self.bounding_box_xywh
        ):
            raise ValueError("bounding_box_xywh must contain four finite pixel values")
        if self.bounding_box_xywh[2] <= 0 or self.bounding_box_xywh[3] <= 0:
            raise ValueError("bounding_box_xywh width and height must be positive")
        if len(self.head_pose_yaw_pitch_roll_degrees) != 3 or not all(
            math.isfinite(value) for value in self.head_pose_yaw_pitch_roll_degrees
        ):
            raise ValueError(
                "head_pose_yaw_pitch_roll_degrees must contain three finite degree values"
            )
        if not math.isfinite(self.raw_face_quality_score):
            raise ValueError("raw_face_quality_score must be finite")


@dataclass(frozen=True, slots=True)
class DeepTalkScoreWindow:
    """Preserve unchanged DeepTalk scores for one source-timeline window.

    ``raw_scores`` contains DeepTalk's final score for each face identity. Values are not
    probabilities and are not thresholded or smoothed. An empty mapping is retained as evidence.
    The bounds are consecutive diagnostic bounds; DeepTalk 0.3.1 treats audio as half-open but
    includes video at both bounds, so a face at an exact boundary can affect adjacent windows.

    Parameters
    ----------
    source_start_seconds
        Inclusive source-media start of the diagnostic window, in seconds.
    source_end_seconds
        Nominal source-media end of the diagnostic window, in seconds.
    raw_scores
        Immutable snapshot mapping DeepTalk face identities to unchanged scores.
    """

    source_start_seconds: float
    source_end_seconds: float
    raw_scores: Mapping[int, float]

    def __post_init__(self) -> None:
        """Validate the window and snapshot mutable DeepTalk score dictionaries."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("source_end_seconds must be finite and after source_start_seconds")

        copied_scores: dict[int, float] = {}
        for face_id, score in self.raw_scores.items():
            if isinstance(face_id, bool) or not isinstance(face_id, int):
                raise TypeError("raw score face IDs must be integers")
            if not math.isfinite(score):
                raise ValueError(f"raw score for face {face_id} must be finite, got {score}")
            copied_scores[face_id] = score
        object.__setattr__(self, "raw_scores", MappingProxyType(copied_scores))


@dataclass(frozen=True, slots=True)
class DeepTalkSpeechInterval:
    """Record one DeepTalk VAD interval on the source-media timeline.

    Parameters
    ----------
    source_start_seconds
        Inclusive source-media start in seconds, including DeepTalk's retained prefix audio.
    source_end_seconds
        Exclusive source-media end in seconds.
    status
        Whether DeepTalk confirmed, rejected, or reached EOF during the interval.
    """

    source_start_seconds: float
    source_end_seconds: float
    status: _SpeechIntervalStatus

    def __post_init__(self) -> None:
        """Validate source-timeline bounds and the copied VAD status."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("source_end_seconds must be finite and after source_start_seconds")
        if self.status not in {"confirmed", "rejected", "incomplete"}:
            raise ValueError(f"unsupported DeepTalk speech interval status: {self.status}")


@dataclass(frozen=True, slots=True)
class DeepTalkAnalysisProvenance:
    """Describe the fixed DeepTalk media transformation and optional capabilities.

    Parameters
    ----------
    backend_version
        Installed ``deeptalk-asd`` distribution version.
    video_sample_rate_hz
        Video grid submitted to the backend.
    audio_sample_rate_hz
        Mono signed-16-bit PCM rate submitted to the backend.
    speaker_embeddings_available
        Always false for this adapter because voiceprints are deliberately disabled.
    """

    backend_version: str
    video_sample_rate_hz: int = _VIDEO_FPS
    audio_sample_rate_hz: int = _AUDIO_SAMPLE_RATE_HZ
    speaker_embeddings_available: bool = False


@dataclass(frozen=True, slots=True)
class DeepTalkAudioTimelineRepair:
    """Describe one explicit repair applied to the source-audio timeline.

    Parameters
    ----------
    kind
        Whether missing source time was filled with silence or overlapping samples were trimmed.
    source_start_seconds
        Inclusive start of the affected region on the source-media timeline.
    source_end_seconds
        Exclusive end of the affected region on the source-media timeline.
    adjusted_sample_count
        Number of source-rate samples inserted or removed per channel.
    sample_rate_hz
        Source sample rate used to quantize the repair.
    """

    kind: _AudioTimelineRepairKind
    source_start_seconds: float
    source_end_seconds: float
    adjusted_sample_count: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        """Validate the repair kind, source bounds, and sample count."""
        if self.kind not in {"audio_gap_filled", "audio_overlap_trimmed"}:
            raise ValueError(f"unsupported audio timeline repair kind: {self.kind}")
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("repair source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("repair source_end_seconds must be finite and after its start")
        if (
            isinstance(self.adjusted_sample_count, bool)
            or not isinstance(self.adjusted_sample_count, int)
            or self.adjusted_sample_count <= 0
        ):
            raise ValueError("repair adjusted_sample_count must be a positive integer")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("repair sample_rate_hz must be a positive integer")

    @property
    def duration_seconds(self) -> float:
        """Duration inserted or removed after source-rate sample quantization."""
        return self.adjusted_sample_count / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class DeepTalkPreliminarySegment:
    """Describe one explainable pre-calibration segment decision.

    A ``candidate`` has confirmed or EOF-truncated speech, exactly one observed face identity, and
    a positive raw DeepTalk score. It is only eligible for later visual-quality, camera-facing,
    synchronization, stability, and calibrated-score checks; it is not an accepted training clip.
    Multiple visible faces or identity changes are rejected because the current client requirement
    allows the policy to prefer precision over coverage.

    Parameters
    ----------
    source_start_seconds, source_end_seconds
        Half-open bounds on the original source timeline.
    status
        Preliminary ``candidate``, ``rejected``, or ``uncertain`` classification.
    face_id
        Sole observed face identity when one can be assigned, otherwise ``None``.
    reason_codes
        Stable machine-readable explanations for the status.
    visible_face_ids
        Sorted distinct identities observed within the segment.
    raw_score
        Unchanged DeepTalk score for ``face_id``, or ``None`` when it was unavailable.
    """

    source_start_seconds: float
    source_end_seconds: float
    status: _PreliminarySegmentStatus
    face_id: int | None
    reason_codes: tuple[_PreliminarySegmentReason, ...]
    visible_face_ids: tuple[int, ...]
    raw_score: float | None

    def __post_init__(self) -> None:
        """Validate source bounds, identities, reasons, and unchanged score evidence."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("segment source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("segment source_end_seconds must be finite and after its start")
        if self.status not in {"candidate", "rejected", "uncertain"}:
            raise ValueError(f"unsupported preliminary segment status: {self.status}")
        if not self.reason_codes:
            raise ValueError("preliminary segment reason_codes must not be empty")
        if len(set(self.reason_codes)) != len(self.reason_codes) or any(
            reason not in _PRELIMINARY_SEGMENT_REASON_VALUES for reason in self.reason_codes
        ):
            raise ValueError("preliminary segment reason_codes must be supported and unique")
        if tuple(sorted(set(self.visible_face_ids))) != self.visible_face_ids or any(
            isinstance(face_id, bool) or not isinstance(face_id, int)
            for face_id in self.visible_face_ids
        ):
            raise ValueError("visible_face_ids must contain sorted unique integers")
        if self.face_id is not None and (
            isinstance(self.face_id, bool) or not isinstance(self.face_id, int)
        ):
            raise TypeError("segment face_id must be an integer or None")
        if self.face_id is not None and self.face_id not in self.visible_face_ids:
            raise ValueError("segment face_id must be present in visible_face_ids")
        if self.raw_score is not None and not math.isfinite(self.raw_score):
            raise ValueError("segment raw_score must be finite when present")
        if self.raw_score is not None and self.face_id is None:
            raise ValueError("segment raw_score requires a face_id")
        if self.status == "candidate" and (
            self.face_id is None or self.raw_score is None or self.raw_score <= 0
        ):
            raise ValueError("candidate segments require one face and a positive raw score")


@dataclass(frozen=True, slots=True)
class DeepTalkCandidatePolicy:
    """Configure conservative structural checks for continuous candidate runs.

    These defaults are intentionally strict starting points for dataset curation, not calibrated
    guarantees. They are kept explicit so a future evaluation set can replace them without
    changing the evidence or run-building contracts.
    """

    minimum_duration_seconds: float = 2.0
    minimum_face_coverage: float = 0.90
    maximum_face_gap_seconds: float = 0.20

    def __post_init__(self) -> None:
        """Validate duration, coverage, and visibility-gap thresholds."""
        if not math.isfinite(self.minimum_duration_seconds) or self.minimum_duration_seconds <= 0:
            raise ValueError("minimum_duration_seconds must be finite and positive")
        if not math.isfinite(self.minimum_face_coverage) or not 0 < self.minimum_face_coverage <= 1:
            raise ValueError("minimum_face_coverage must be finite and in (0, 1]")
        if not math.isfinite(self.maximum_face_gap_seconds) or self.maximum_face_gap_seconds < 0:
            raise ValueError("maximum_face_gap_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DeepTalkCandidateRun:
    """Summarize one continuous same-face run after structural policy checks.

    A run is formed only from adjacent preliminary candidates for the same DeepTalk face identity.
    Passing this stage means the run is long enough and its face detections are sufficiently dense
    and stable. It still requires camera-facing, visual-quality, synchronization, and calibrated
    active-speaker checks before it can become an accepted training clip.
    """

    source_start_seconds: float
    source_end_seconds: float
    status: _CandidateRunStatus
    face_id: int
    reason_codes: tuple[_CandidateRunReason, ...]
    preliminary_segment_count: int
    face_observation_count: int
    expected_face_observation_count: int
    face_visibility_fraction: float
    maximum_face_gap_seconds: float
    raw_score_min: float
    raw_score_mean: float
    raw_score_max: float

    def __post_init__(self) -> None:
        """Validate source bounds, policy decision, observation metrics, and score summary."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("run source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("run source_end_seconds must be finite and after its start")
        if self.status not in {"candidate", "rejected"}:
            raise ValueError(f"unsupported candidate run status: {self.status}")
        if isinstance(self.face_id, bool) or not isinstance(self.face_id, int):
            raise TypeError("run face_id must be an integer")
        if not self.reason_codes:
            raise ValueError("candidate run reason_codes must not be empty")
        if len(set(self.reason_codes)) != len(self.reason_codes) or any(
            reason not in _CANDIDATE_RUN_REASON_VALUES for reason in self.reason_codes
        ):
            raise ValueError("candidate run reason_codes must be supported and unique")
        if self.status == "candidate" and self.reason_codes != ("structural_policy_passed",):
            raise ValueError("candidate runs require only structural_policy_passed")
        if self.status == "rejected" and "structural_policy_passed" in self.reason_codes:
            raise ValueError("rejected runs cannot include structural_policy_passed")
        for name, count in (
            ("preliminary_segment_count", self.preliminary_segment_count),
            ("expected_face_observation_count", self.expected_face_observation_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.face_observation_count, bool)
            or not isinstance(self.face_observation_count, int)
            or self.face_observation_count < 0
        ):
            raise ValueError("face_observation_count must be a non-negative integer")
        if (
            not math.isfinite(self.face_visibility_fraction)
            or not 0 <= self.face_visibility_fraction <= 1
        ):
            raise ValueError("face_visibility_fraction must be finite and in [0, 1]")
        if not math.isfinite(self.maximum_face_gap_seconds) or self.maximum_face_gap_seconds < 0:
            raise ValueError("maximum_face_gap_seconds must be finite and non-negative")
        scores = (self.raw_score_min, self.raw_score_mean, self.raw_score_max)
        if not all(math.isfinite(score) and score > 0 for score in scores):
            raise ValueError("candidate run raw score summary must be finite and positive")
        if not self.raw_score_min <= self.raw_score_mean <= self.raw_score_max:
            raise ValueError("candidate run raw score summary must satisfy min <= mean <= max")

    @property
    def duration_seconds(self) -> float:
        """Continuous run duration on the source-media timeline."""
        return self.source_end_seconds - self.source_start_seconds


@dataclass(frozen=True, slots=True)
class DeepTalkAnalysisResult:
    """Contain copied face observations and diagnostic score windows for one sequence.

    Parameters
    ----------
    face_observations
        Observed identities, boxes, raw head pose, and raw quality at submitted 25 Hz video slots.
    speech_intervals
        Completed, rejected, and EOF-truncated VAD intervals on the source timeline.
    score_windows
        Consecutive source-timeline windows containing unchanged DeepTalk scores.
    provenance
        Backend version, fixed media rates, and optional capability status.
    audio_timeline_repairs
        Explicit gap fills and overlap trims applied before resampling for DeepTalk.
    issues
        Immutable diagnostic codes for non-fatal source coverage limitations.
    preliminary_segments
        Explainable pre-calibration decisions derived from speech, face, and score evidence.
    candidate_policy
        Explicit initial thresholds used to assess continuous same-face candidate runs.
    candidate_runs
        Continuous preliminary candidates with duration and face-visibility metrics.
    """

    face_observations: tuple[DeepTalkFaceObservation, ...]
    speech_intervals: tuple[DeepTalkSpeechInterval, ...]
    score_windows: tuple[DeepTalkScoreWindow, ...]
    provenance: DeepTalkAnalysisProvenance
    audio_timeline_repairs: tuple[DeepTalkAudioTimelineRepair, ...] = ()
    issues: tuple[str, ...] = ()
    preliminary_segments: tuple[DeepTalkPreliminarySegment, ...] = ()
    candidate_policy: DeepTalkCandidatePolicy = DeepTalkCandidatePolicy()
    candidate_runs: tuple[DeepTalkCandidateRun, ...] = ()


def build_preliminary_segments(
    result: DeepTalkAnalysisResult,
) -> tuple[DeepTalkPreliminarySegment, ...]:
    """Build conservative pre-calibration decisions from one DeepTalk result.

    Each raw score window is split at non-rejected VAD boundaries. Non-speech, no-face,
    multi-face, changing-identity, and non-positive-score regions are rejected. A sole visible face
    with speech but no corresponding score is uncertain. Only speech regions with one identity and
    a positive raw score become candidates, and candidates still require all later policy stages.

    Parameters
    ----------
    result
        Immutable DeepTalk evidence on the source timeline. Existing preliminary segments are
        ignored so callers can reproduce the classification from the raw observations.

    Returns
    -------
    tuple[DeepTalkPreliminarySegment, ...]
        Ordered, non-empty subintervals covering every available score window.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`DeepTalkAnalysisResult`.
    """
    if not isinstance(result, DeepTalkAnalysisResult):
        raise TypeError(f"result must be a DeepTalkAnalysisResult, got {type(result).__name__}")

    active_speech_intervals = tuple(
        interval for interval in result.speech_intervals if interval.status != "rejected"
    )
    ordered_windows = sorted(result.score_windows, key=lambda window: window.source_start_seconds)
    ordered_observations = sorted(
        result.face_observations,
        key=lambda observation: observation.source_timestamp_seconds,
    )
    observation_timestamps = [
        observation.source_timestamp_seconds for observation in ordered_observations
    ]
    segments: list[DeepTalkPreliminarySegment] = []

    for window in ordered_windows:
        first_observation = bisect_left(observation_timestamps, window.source_start_seconds)
        after_last_observation = bisect_left(observation_timestamps, window.source_end_seconds)
        window_observations = ordered_observations[first_observation:after_last_observation]
        boundaries = {window.source_start_seconds, window.source_end_seconds}
        for interval in active_speech_intervals:
            if (
                interval.source_start_seconds < window.source_end_seconds
                and interval.source_end_seconds > window.source_start_seconds
            ):
                boundaries.add(max(window.source_start_seconds, interval.source_start_seconds))
                boundaries.add(min(window.source_end_seconds, interval.source_end_seconds))

        ordered_boundaries: list[float] = []
        for boundary in sorted(boundaries):
            if (
                not ordered_boundaries
                or boundary - ordered_boundaries[-1] > _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            ):
                ordered_boundaries.append(boundary)
        for start_seconds, end_seconds in pairwise(ordered_boundaries):
            segments.append(
                _classify_preliminary_interval(
                    window,
                    active_speech_intervals,
                    window_observations,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )

    return tuple(segments)


def build_candidate_runs(
    result: DeepTalkAnalysisResult,
    policy: DeepTalkCandidatePolicy | None = None,
) -> tuple[DeepTalkCandidateRun, ...]:
    """Merge adjacent preliminary candidates and apply structural stability checks.

    Only adjacent candidate segments belonging to the same face are merged. Rejected or uncertain
    preliminary regions therefore form hard boundaries. The resulting run records duration, face
    detection coverage, the longest missing-face gap, and a summary of the unchanged raw scores.

    Parameters
    ----------
    result
        DeepTalk evidence containing preliminary segment decisions.
    policy
        Thresholds to apply. When omitted, ``result.candidate_policy`` is used.

    Returns
    -------
    tuple[DeepTalkCandidateRun, ...]
        Ordered same-face runs. Runs that fail a threshold remain present as rejected evidence.
    """
    if not isinstance(result, DeepTalkAnalysisResult):
        raise TypeError(f"result must be a DeepTalkAnalysisResult, got {type(result).__name__}")
    resolved_policy = result.candidate_policy if policy is None else policy
    if not isinstance(resolved_policy, DeepTalkCandidatePolicy):
        raise TypeError(
            f"policy must be a DeepTalkCandidatePolicy, got {type(resolved_policy).__name__}"
        )

    merged_segments: list[list[DeepTalkPreliminarySegment]] = []
    for segment in sorted(
        result.preliminary_segments,
        key=lambda item: (item.source_start_seconds, item.source_end_seconds),
    ):
        if segment.status != "candidate":
            continue
        if segment.face_id is None or segment.raw_score is None:
            raise ValueError("candidate preliminary segments require a face ID and raw score")
        if merged_segments:
            previous = merged_segments[-1][-1]
            gap_seconds = segment.source_start_seconds - previous.source_end_seconds
            if (
                previous.face_id == segment.face_id
                and abs(gap_seconds) <= _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            ):
                merged_segments[-1].append(segment)
                continue
        merged_segments.append([segment])

    observations_by_face: dict[int, list[float]] = {}
    for observation in result.face_observations:
        observations_by_face.setdefault(observation.face_id, []).append(
            observation.source_timestamp_seconds
        )
    for timestamps in observations_by_face.values():
        timestamps.sort()

    return tuple(
        _assess_candidate_run(
            segments,
            observations_by_face.get(cast(int, segments[0].face_id), []),
            video_sample_rate_hz=result.provenance.video_sample_rate_hz,
            policy=resolved_policy,
        )
        for segments in merged_segments
    )


def _assess_candidate_run(
    segments: list[DeepTalkPreliminarySegment],
    face_observation_timestamps: list[float],
    *,
    video_sample_rate_hz: int,
    policy: DeepTalkCandidatePolicy,
) -> DeepTalkCandidateRun:
    start_seconds = segments[0].source_start_seconds
    end_seconds = segments[-1].source_end_seconds
    face_id = cast(int, segments[0].face_id)
    duration_seconds = end_seconds - start_seconds
    first_observation = bisect_left(face_observation_timestamps, start_seconds)
    after_last_observation = bisect_left(face_observation_timestamps, end_seconds)
    observed_timestamps = sorted(
        set(face_observation_timestamps[first_observation:after_last_observation])
    )
    expected_observation_count = max(
        1,
        math.ceil(duration_seconds * video_sample_rate_hz - _VIDEO_TIMESTAMP_TOLERANCE_SECONDS),
    )
    visibility_fraction = min(len(observed_timestamps) / expected_observation_count, 1.0)
    frame_period_seconds = 1 / video_sample_rate_hz
    if observed_timestamps:
        visibility_gaps = [
            max(0.0, observed_timestamps[0] - start_seconds),
            max(0.0, end_seconds - (observed_timestamps[-1] + frame_period_seconds)),
        ]
        visibility_gaps.extend(
            max(0.0, current - previous - frame_period_seconds)
            for previous, current in pairwise(observed_timestamps)
        )
        maximum_gap_seconds = max(visibility_gaps)
    else:
        maximum_gap_seconds = duration_seconds

    reason_codes: list[_CandidateRunReason] = []
    if duration_seconds < policy.minimum_duration_seconds:
        reason_codes.append("candidate_too_short")
    if visibility_fraction < policy.minimum_face_coverage:
        reason_codes.append("insufficient_face_coverage")
    if maximum_gap_seconds > policy.maximum_face_gap_seconds:
        reason_codes.append("unstable_face_visibility")
    status: _CandidateRunStatus = "rejected" if reason_codes else "candidate"
    if not reason_codes:
        reason_codes.append("structural_policy_passed")

    raw_scores = [cast(float, segment.raw_score) for segment in segments]
    duration_weighted_score = (
        sum(
            cast(float, segment.raw_score)
            * (segment.source_end_seconds - segment.source_start_seconds)
            for segment in segments
        )
        / duration_seconds
    )
    return DeepTalkCandidateRun(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        status=status,
        face_id=face_id,
        reason_codes=tuple(reason_codes),
        preliminary_segment_count=len(segments),
        face_observation_count=len(observed_timestamps),
        expected_face_observation_count=expected_observation_count,
        face_visibility_fraction=visibility_fraction,
        maximum_face_gap_seconds=maximum_gap_seconds,
        raw_score_min=min(raw_scores),
        raw_score_mean=duration_weighted_score,
        raw_score_max=max(raw_scores),
    )


def _classify_preliminary_interval(
    score_window: DeepTalkScoreWindow,
    active_speech_intervals: tuple[DeepTalkSpeechInterval, ...],
    face_observations: list[DeepTalkFaceObservation],
    *,
    start_seconds: float,
    end_seconds: float,
) -> DeepTalkPreliminarySegment:
    faces_by_timestamp: dict[float, set[int]] = {}
    for observation in face_observations:
        if start_seconds <= observation.source_timestamp_seconds < end_seconds:
            faces_by_timestamp.setdefault(observation.source_timestamp_seconds, set()).add(
                observation.face_id
            )
    visible_face_ids = tuple(
        sorted({face_id for face_ids in faces_by_timestamp.values() for face_id in face_ids})
    )
    speech_active = any(
        interval.source_start_seconds < end_seconds and interval.source_end_seconds > start_seconds
        for interval in active_speech_intervals
    )
    if not speech_active:
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("no_speech",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )

    if not visible_face_ids:
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("no_visible_face",),
            visible_face_ids=(),
            raw_score=None,
        )
    if any(len(face_ids) > 1 for face_ids in faces_by_timestamp.values()):
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("multiple_visible_faces",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )
    if len(visible_face_ids) > 1:
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("face_identity_changed",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )

    face_id = visible_face_ids[0]
    raw_score = score_window.raw_scores.get(face_id)
    if raw_score is None:
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="uncertain",
            face_id=face_id,
            reason_codes=("no_active_speaker_score",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )
    if raw_score <= 0:
        return DeepTalkPreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=face_id,
            reason_codes=("non_positive_active_speaker_score",),
            visible_face_ids=visible_face_ids,
            raw_score=raw_score,
        )
    return DeepTalkPreliminarySegment(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        status="candidate",
        face_id=face_id,
        reason_codes=("single_visible_face", "positive_raw_active_speaker_score"),
        visible_face_ids=visible_face_ids,
        raw_score=raw_score,
    )


def analyze_sequence(
    sequence: TalkingFaceSequence,
    *,
    candidate_policy: DeepTalkCandidatePolicy | None = None,
) -> DeepTalkAnalysisResult:
    """Run the experimental DeepTalk-ASD 0.3.1 adapter on one sequence offline.

    Video is sampled lazily onto a 25 Hz source-time grid. Audio is streamed through PyAV's
    libswresample boundary as mono 16 kHz signed 16-bit PCM in 30 ms frames. Both streams are fed
    in chronological media-time order, with video preceding audio at equal timestamps. DeepTalk is
    evaluated incrementally in approximately one-second windows so its ten-second audio buffer
    cannot discard early windows before evaluation.

    Parameters
    ----------
    sequence
        Source and half-open source-media interval to analyze. The source must contain audio.
    candidate_policy
        Optional structural thresholds for continuous candidate runs. Defaults are conservative
        starting values and have not yet been calibrated on a labelled evaluation set.

    Returns
    -------
    DeepTalkAnalysisResult
        Backend-owned objects copied into immutable face, speech, score, and provenance values.

    Raises
    ------
    TypeError
        If ``sequence`` is not a :class:`TalkingFaceSequence`.
    ValueError
        If the source has no audio, its decoded format changes within the selected interval, or no
        audio samples can be decoded.
    RuntimeError
        If the optional dependency is absent, detector construction fails, DeepTalk is not version
        0.3.1, or its expected private shim boundary changed.

    Notes
    -----
    Detector construction resolves only the required face, VAD, and LR-ASD assets through
    DeepTalk's hash-verifying public model manager. Normal unit tests replace that boundary and do
    not download or run models.
    """
    if not isinstance(sequence, TalkingFaceSequence):
        raise TypeError(f"sequence must be a TalkingFaceSequence, got {type(sequence).__name__}")
    resolved_candidate_policy = (
        DeepTalkCandidatePolicy() if candidate_policy is None else candidate_policy
    )
    if not isinstance(resolved_candidate_policy, DeepTalkCandidatePolicy):
        raise TypeError(
            "candidate_policy must be a DeepTalkCandidatePolicy, "
            f"got {type(resolved_candidate_policy).__name__}"
        )
    if not sequence.source.metadata.has_audio:
        raise ValueError(f"DeepTalk analysis requires an audio stream: {sequence.source.path}")

    deeptalk_module = _load_deeptalk_module()
    detector = _create_deeptalk_detector(deeptalk_module)
    video_events = _iter_deeptalk_video_frames(sequence, deeptalk_module)
    audio_timeline_repairs: list[DeepTalkAudioTimelineRepair] = []
    audio_events = _iter_deeptalk_audio_frames(
        sequence,
        deeptalk_module,
        audio_timeline_repairs,
    )

    face_observations: list[DeepTalkFaceObservation] = []
    speech_intervals: list[DeepTalkSpeechInterval] = []
    score_windows: list[DeepTalkScoreWindow] = []
    issues: list[str] = []
    next_window_start_seconds = 0.0
    next_window_end_seconds = _SCORE_WINDOW_SECONDS
    final_audio_end_seconds: float | None = None
    final_video_timestamp_seconds: float | None = None
    latest_utterance: object | None = None
    backend = cast(Any, detector)  # DeepTalk 0.3.1 does not publish typing metadata.

    for event_kind, event_time_seconds, payload in _merge_av_events(video_events, audio_events):
        if event_kind == "video":
            final_video_timestamp_seconds = event_time_seconds
            profiles = backend.append_video(payload, create_time=event_time_seconds)
            face_observations.extend(
                _copy_face_observations(
                    sequence.start_seconds + event_time_seconds,
                    profiles,
                )
            )
            continue

        latest_utterance = backend.append_audio(payload, create_time=event_time_seconds)
        completed_interval = _copy_speech_interval(
            latest_utterance,
            source_offset_seconds=sequence.start_seconds,
            relative_end_seconds=event_time_seconds,
            include_incomplete=False,
        )
        if completed_interval is not None:
            speech_intervals.append(completed_interval)
        final_audio_end_seconds = event_time_seconds
        while next_window_end_seconds <= event_time_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
            score_window = _evaluate_score_window(
                backend,
                source_offset_seconds=sequence.start_seconds,
                start_seconds=next_window_start_seconds,
                end_seconds=next_window_end_seconds,
            )
            score_windows.append(score_window)
            _prune_deeptalk_video_buffer(backend, before_seconds=next_window_end_seconds)
            next_window_start_seconds = next_window_end_seconds
            next_window_end_seconds += _SCORE_WINDOW_SECONDS

    if final_audio_end_seconds is None:
        raise ValueError("DeepTalk analysis received no decoded audio samples")

    analysis_end_seconds = (
        sequence.duration_seconds
        if sequence.duration_seconds is not None
        else final_audio_end_seconds
    )
    if final_audio_end_seconds + _AUDIO_TIMESTAMP_TOLERANCE_SECONDS < analysis_end_seconds:
        raise ValueError(
            "DeepTalk audio ended before the sequence interval: "
            f"audio_end={sequence.start_seconds + final_audio_end_seconds}, "
            f"sequence_end={sequence.start_seconds + analysis_end_seconds}"
        )

    if final_video_timestamp_seconds is None:
        issues.append("no_video_frames")
    elif sequence.duration_seconds is None:
        video_coverage_end_seconds = final_video_timestamp_seconds + _FRAME_PERIOD_SECONDS
        if (
            final_audio_end_seconds + _AUDIO_TIMESTAMP_TOLERANCE_SECONDS
            < video_coverage_end_seconds
        ):
            issues.append("audio_ended_before_video")
            maximum_source_timestamp = sequence.start_seconds + final_audio_end_seconds
            face_observations = [
                observation
                for observation in face_observations
                if observation.source_timestamp_seconds
                <= maximum_source_timestamp + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            ]
        elif video_coverage_end_seconds + _FRAME_PERIOD_SECONDS < final_audio_end_seconds:
            issues.append("video_ended_before_audio")

    incomplete_interval = _copy_speech_interval(
        latest_utterance,
        source_offset_seconds=sequence.start_seconds,
        relative_end_seconds=final_audio_end_seconds,
        include_incomplete=True,
    )
    if incomplete_interval is not None:
        speech_intervals.append(incomplete_interval)

    for repair in audio_timeline_repairs:
        if repair.kind not in issues:
            issues.append(repair.kind)

    if next_window_start_seconds < analysis_end_seconds - _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
        score_window = _evaluate_score_window(
            backend,
            source_offset_seconds=sequence.start_seconds,
            start_seconds=next_window_start_seconds,
            end_seconds=analysis_end_seconds,
        )
        score_windows.append(score_window)
        _prune_deeptalk_video_buffer(backend, before_seconds=analysis_end_seconds)

    raw_result = DeepTalkAnalysisResult(
        face_observations=tuple(face_observations),
        speech_intervals=tuple(speech_intervals),
        score_windows=tuple(score_windows),
        provenance=DeepTalkAnalysisProvenance(
            backend_version=_DEEPTALK_SUPPORTED_VERSION,
        ),
        audio_timeline_repairs=tuple(audio_timeline_repairs),
        issues=tuple(issues),
    )
    preliminary_result = replace(
        raw_result,
        preliminary_segments=build_preliminary_segments(raw_result),
        candidate_policy=resolved_candidate_policy,
    )
    return replace(
        preliminary_result,
        candidate_runs=build_candidate_runs(preliminary_result),
    )


def _sample_video_frames_25_fps(
    sequence: TalkingFaceSequence,
) -> Iterator[tuple[float, DecodedVideoFrame]]:
    """Sample source PTS onto a lazy 25 FPS grid for a sequence interval."""
    frames = stream_video_frames(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        selected_frame = next(frames)
    except StopIteration:
        return

    next_frame = next(frames, None)
    first_frame_timestamp_seconds = selected_frame.timestamp_seconds
    slot_index = max(
        0,
        int((first_frame_timestamp_seconds - sequence.start_seconds) * _VIDEO_FPS),
    )

    while True:
        slot_timestamp_seconds = sequence.start_seconds + slot_index / _VIDEO_FPS
        if (
            slot_timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            < first_frame_timestamp_seconds
        ):
            slot_index += 1
            continue
        if sequence.end_seconds is not None and slot_timestamp_seconds >= sequence.end_seconds:
            return

        while next_frame is not None:
            selected_distance_seconds = abs(
                selected_frame.timestamp_seconds - slot_timestamp_seconds
            )
            next_distance_seconds = abs(next_frame.timestamp_seconds - slot_timestamp_seconds)
            if (
                next_distance_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
                >= selected_distance_seconds
            ):
                break
            selected_frame = next_frame
            next_frame = next(frames, None)

        if (
            next_frame is None
            and slot_timestamp_seconds
            > selected_frame.timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
        ):
            return

        yield slot_timestamp_seconds, selected_frame
        slot_index += 1


def _iter_deeptalk_video_frames(
    sequence: TalkingFaceSequence,
    deeptalk_module: ModuleType,
) -> Iterator[tuple[float, object]]:
    for source_timestamp_seconds, source_frame in _sample_video_frames_25_fps(sequence):
        relative_timestamp_seconds = source_timestamp_seconds - sequence.start_seconds
        yield relative_timestamp_seconds, _to_deeptalk_video_frame(source_frame, deeptalk_module)


def _to_deeptalk_video_frame(
    source_frame: DecodedVideoFrame,
    deeptalk_module: ModuleType,
) -> object:
    rgb_bytes = source_frame.rgb.tobytes(order="C")
    backend = cast(Any, deeptalk_module)
    return backend.VideoFrame(
        width=source_frame.rgb.shape[1],
        height=source_frame.rgb.shape[0],
        type=backend.VideoBufferType.RGB24,
        data=rgb_bytes,
    )


def _iter_deeptalk_audio_frames(
    sequence: TalkingFaceSequence,
    deeptalk_module: ModuleType,
    repairs: list[DeepTalkAudioTimelineRepair] | None = None,
) -> Iterator[tuple[float, object]]:
    """Normalize source timestamps and lazily produce continuous DeepTalk PCM frames."""
    chunks = stream_audio_chunks(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        first_chunk = next(chunks)
    except StopIteration as error:
        raise ValueError("DeepTalk analysis received no decoded audio samples") from error

    repair_records = [] if repairs is None else repairs
    anchor_seconds = sequence.start_seconds
    input_sample_rate_hz = first_chunk.sample_rate_hz
    input_format = (
        first_chunk.sample_rate_hz,
        first_chunk.channel_count,
        first_chunk.channel_layout,
    )
    maximum_output_sample_count = (
        max(
            0,
            math.floor(
                (sequence.end_seconds - sequence.start_seconds) * _AUDIO_SAMPLE_RATE_HZ
                + _AUDIO_SAMPLE_COUNT_TOLERANCE
            ),
        )
        if sequence.end_seconds is not None
        else None
    )
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=_AUDIO_SAMPLE_RATE_HZ,
        frame_size=_AUDIO_FRAME_SAMPLE_COUNT,
    )
    normalized_input_sample_count = 0
    emitted_sample_count = 0
    backend = cast(Any, deeptalk_module)

    def convert_outputs(
        output_frames: list[av.AudioFrame],
    ) -> Iterator[tuple[float, object]]:
        nonlocal emitted_sample_count
        for output_frame in output_frames:
            pcm = np.ascontiguousarray(
                output_frame.to_ndarray().reshape(-1),
                dtype=np.int16,
            )
            if maximum_output_sample_count is not None:
                remaining_sample_count = maximum_output_sample_count - emitted_sample_count
                if remaining_sample_count <= 0:
                    continue
                pcm = pcm[:remaining_sample_count]
            if pcm.size == 0:
                continue

            emitted_sample_count += int(pcm.size)
            relative_end_seconds = emitted_sample_count / _AUDIO_SAMPLE_RATE_HZ
            audio_frame = backend.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=_AUDIO_SAMPLE_RATE_HZ,
                num_channels=1,
                samples_per_channel=int(pcm.size),
            )
            yield relative_end_seconds, audio_frame

    def resample_source_samples(
        samples: NDArray[np.float32],
    ) -> Iterator[tuple[float, object]]:
        if samples.shape[0] == 0:
            return
        input_frame = _samples_to_av_frame(
            samples,
            sample_rate_hz=input_sample_rate_hz,
            channel_layout=first_chunk.channel_layout,
        )
        yield from convert_outputs(resampler.resample(input_frame))

    def fill_silence(sample_count: int) -> Iterator[tuple[float, object]]:
        remaining_sample_count = sample_count
        while remaining_sample_count > 0:
            block_sample_count = min(
                remaining_sample_count,
                _AUDIO_REPAIR_BLOCK_SAMPLE_COUNT,
            )
            silence = np.zeros(
                (block_sample_count, first_chunk.channel_count),
                dtype=np.float32,
            )
            yield from resample_source_samples(silence)
            remaining_sample_count -= block_sample_count

    for chunk_index, chunk in enumerate(chain((first_chunk,), chunks)):
        if chunk_index > 0:
            _validate_audio_chunk_format(chunk, expected=input_format)

        expected_start_seconds = (
            anchor_seconds + normalized_input_sample_count / input_sample_rate_hz
        )
        difference_seconds = chunk.start_timestamp_seconds - expected_start_seconds
        tolerance_seconds = max(
            _AUDIO_TIMESTAMP_TOLERANCE_SECONDS,
            1 / input_sample_rate_hz,
        )
        if difference_seconds > tolerance_seconds:
            gap_sample_count = _quantize_audio_repair_sample_count(
                difference_seconds,
                input_sample_rate_hz,
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_gap_filled",
                    source_start_seconds=expected_start_seconds,
                    source_end_seconds=chunk.start_timestamp_seconds,
                    adjusted_sample_count=gap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            yield from fill_silence(gap_sample_count)
            normalized_input_sample_count += gap_sample_count

        samples = chunk.samples
        if difference_seconds < -tolerance_seconds:
            overlap_sample_count = min(
                _quantize_audio_repair_sample_count(
                    -difference_seconds,
                    input_sample_rate_hz,
                ),
                chunk.sample_count,
            )
            overlap_end_seconds = (
                chunk.start_timestamp_seconds + overlap_sample_count / input_sample_rate_hz
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_overlap_trimmed",
                    source_start_seconds=chunk.start_timestamp_seconds,
                    source_end_seconds=overlap_end_seconds,
                    adjusted_sample_count=overlap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            samples = np.ascontiguousarray(samples[overlap_sample_count:], dtype=np.float32)

        normalized_input_sample_count += int(samples.shape[0])
        yield from resample_source_samples(samples)

    if sequence.end_seconds is not None:
        expected_end_seconds = anchor_seconds + normalized_input_sample_count / input_sample_rate_hz
        trailing_gap_seconds = sequence.end_seconds - expected_end_seconds
        tolerance_seconds = max(
            _AUDIO_TIMESTAMP_TOLERANCE_SECONDS,
            1 / input_sample_rate_hz,
        )
        if trailing_gap_seconds > tolerance_seconds:
            trailing_gap_sample_count = _quantize_audio_repair_sample_count(
                trailing_gap_seconds,
                input_sample_rate_hz,
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_gap_filled",
                    source_start_seconds=expected_end_seconds,
                    source_end_seconds=sequence.end_seconds,
                    adjusted_sample_count=trailing_gap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            yield from fill_silence(trailing_gap_sample_count)

    yield from convert_outputs(resampler.resample(None))


def _samples_to_av_frame(
    samples: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    channel_layout: str,
) -> av.AudioFrame:
    planar_samples = np.ascontiguousarray(samples.transpose(), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(
        planar_samples,
        format="fltp",
        layout=channel_layout,
    )
    frame.sample_rate = sample_rate_hz
    return frame


def _quantize_audio_repair_sample_count(duration_seconds: float, sample_rate_hz: int) -> int:
    """Round one positive timestamp discrepancy to the nearest source sample."""
    sample_count = math.floor(duration_seconds * sample_rate_hz + 0.5)
    if sample_count <= 0:
        raise RuntimeError(
            "Audio timestamp discrepancy exceeded tolerance but quantized to no samples"
        )
    return sample_count


def _append_audio_timeline_repair(
    repairs: list[DeepTalkAudioTimelineRepair],
    repair: DeepTalkAudioTimelineRepair,
) -> None:
    """Append a repair, coalescing adjacent adjustments of the same kind and rate."""
    if not repairs:
        repairs.append(repair)
        return

    previous = repairs[-1]
    if (
        previous.kind == repair.kind
        and previous.sample_rate_hz == repair.sample_rate_hz
        and abs(repair.source_start_seconds - previous.source_end_seconds)
        <= _AUDIO_TIMESTAMP_TOLERANCE_SECONDS
    ):
        repairs[-1] = DeepTalkAudioTimelineRepair(
            kind=previous.kind,
            source_start_seconds=previous.source_start_seconds,
            source_end_seconds=max(previous.source_end_seconds, repair.source_end_seconds),
            adjusted_sample_count=(previous.adjusted_sample_count + repair.adjusted_sample_count),
            sample_rate_hz=previous.sample_rate_hz,
        )
        return

    repairs.append(repair)


def _validate_audio_chunk_format(
    chunk: DecodedAudioChunk,
    *,
    expected: tuple[int, int, str],
) -> None:
    observed = (chunk.sample_rate_hz, chunk.channel_count, chunk.channel_layout)
    if observed != expected:
        raise ValueError(
            "Decoded audio format changed within the sequence: "
            f"expected={expected}, actual={observed}"
        )


def _merge_av_events(
    video_events: Iterator[tuple[float, object]],
    audio_events: Iterator[tuple[float, object]],
) -> Iterator[_DeepTalkEvent]:
    """Merge sorted offline streams, preferring video at equal media timestamps."""
    video_event = next(video_events, None)
    audio_event = next(audio_events, None)
    while video_event is not None or audio_event is not None:
        if video_event is not None and (audio_event is None or video_event[0] <= audio_event[0]):
            yield "video", video_event[0], video_event[1]
            video_event = next(video_events, None)
        else:
            if audio_event is None:
                raise RuntimeError("A/V event merge reached an invalid state")
            yield "audio", audio_event[0], audio_event[1]
            audio_event = next(audio_events, None)


def _load_deeptalk_module() -> ModuleType:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        deeptalk_asd = import_module("deeptalk_asd")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepTalk analysis requires the 'active-speaker-deeptalk' extra"
        ) from error
    return deeptalk_asd


def _create_deeptalk_detector(deeptalk_module: ModuleType) -> object:
    _require_supported_deeptalk_version()
    backend = cast(Any, deeptalk_module)
    explicit_face_resource = _validate_explicit_inspireface_resource()
    model_cache_dir = Path(backend.get_model_cache_dir())
    face_resource = explicit_face_resource or Path(backend.ensure_model("Pikachu", model_cache_dir))
    for model_name in (
        "silero_vad.onnx",
        "audio_frontend.onnx",
        "visual_frontend.onnx",
        "av_backend.onnx",
    ):
        backend.ensure_model(model_name, model_cache_dir)

    detector = backend.ASDDetectorFactory(
        face_detector={"type": "inspireface", "model_dir": str(face_resource)},
        turn_detector={"type": "silero-vad", "model_dir": str(model_cache_dir)},
        speaker_detector={
            "type": "LR-ASD-ONNX",
            "model_dir": str(model_cache_dir),
            "voiceprint_model_name": "disabled-by-talkingfacekit",
        },
    ).create()
    if detector is None:
        raise RuntimeError("DeepTalk-ASD failed to create its detector")
    _configure_deeptalk_0_3_1_for_offline(detector)
    return cast(object, detector)


def _validate_explicit_inspireface_resource() -> Path | None:
    """Validate DeepTalk's default InspireFace asset before native code receives it."""
    configured_path = os.environ.get(_INSPIREFACE_RESOURCE_ENV)
    if configured_path is None:
        return None

    resource_path = Path(configured_path)
    if not resource_path.is_file():
        raise RuntimeError(
            f"{_INSPIREFACE_RESOURCE_ENV} must name the DeepTalk Pikachu model file: "
            f"{resource_path}"
        )

    digest = sha256()
    with resource_path.open("rb") as resource_file:
        for chunk in iter(lambda: resource_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != _INSPIREFACE_RESOURCE_SHA256:
        raise RuntimeError(
            f"{_INSPIREFACE_RESOURCE_ENV} does not match the supported DeepTalk Pikachu model hash"
        )
    return resource_path


def _require_supported_deeptalk_version() -> None:
    try:
        installed_version = metadata.version(_DEEPTALK_DISTRIBUTION)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("DeepTalk-ASD is not installed") from error
    if installed_version != _DEEPTALK_SUPPORTED_VERSION:
        raise RuntimeError(
            f"The offline shim supports only deeptalk-asd==0.3.1, got {installed_version}"
        )


def _configure_deeptalk_0_3_1_for_offline(detector: object) -> None:
    """Disable DeepTalk 0.3.1 realtime assumptions for deterministic offline input.

    DeepTalk otherwise drops some exact 25 Hz frames through a second float-based throttle and
    expires tracks and VAD state according to wall-clock time. Voiceprints are also disabled so the
    optional native Sherpa linkage cannot silently change score semantics. Every dependency on the
    private layout is validated and contained in this function.
    """
    _require_supported_deeptalk_version()

    detector_type = type(detector)
    if detector_type.__module__ != "deeptalk_asd.asd" or detector_type.__name__ != "ASD":
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected deeptalk_asd.asd.ASD"
        )
    speaker_detector = getattr(detector, "_speaker_detector", None)
    speaker_type = type(speaker_detector)
    if (
        speaker_type.__module__ != "deeptalk_asd.speaker_detector.lrasd_onnx"
        or speaker_type.__name__ != "LRASDOnnxSpeakerDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected LRASDOnnxSpeakerDetector"
        )
    face_detector = getattr(detector, "_face_detector", None)
    face_type = type(face_detector)
    if (
        face_type.__module__ != "deeptalk_asd.face_detector.inspireface_detector"
        or face_type.__name__ != "InspireFaceDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected InspireFaceDetector"
        )
    turn_detector = getattr(detector, "_turn_detector", None)
    turn_type = type(turn_detector)
    if (
        turn_type.__module__ != "deeptalk_asd.turn_detector.silero_vad_turn_detector"
        or turn_type.__name__ != "SileroVadTurnDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected SileroVadTurnDetector"
        )
    vad = getattr(turn_detector, "_vad", None)
    vad_type = type(vad)
    if (
        vad_type.__module__ != "deeptalk_asd.turn_detector.vad.silero_vad"
        or vad_type.__name__ != "SileroVAD"
    ):
        raise RuntimeError("DeepTalk 0.3.1 detector structure changed: expected SileroVAD")

    backend = cast(Any, speaker_detector)
    face_backend = cast(Any, face_detector)
    vad_backend = cast(Any, vad)
    try:
        video_frame_rate = backend.video_frame_rate
        audio_sample_rate = backend.audio_sample_rate
        minimum_frame_interval = backend._min_frame_interval
        _max_track_age = backend.max_track_age
        voice_extractor = backend.voice_extractor
        voice_profiles = backend.voice_profiles
        _last_reset_time = vad_backend._last_reset_time
        remove_stale_faces = face_backend._remove_stale_faces
    except AttributeError as error:
        raise RuntimeError(
            f"DeepTalk 0.3.1 detector structure changed: missing {error.name}"
        ) from error

    if video_frame_rate != _VIDEO_FPS or audio_sample_rate != _AUDIO_SAMPLE_RATE_HZ:
        raise RuntimeError(
            "DeepTalk 0.3.1 detector uses unexpected media rates: "
            f"video={video_frame_rate}, audio={audio_sample_rate}"
        )
    if isinstance(minimum_frame_interval, bool) or not isinstance(minimum_frame_interval, Real):
        raise TypeError("DeepTalk 0.3.1 _min_frame_interval has an unexpected type")
    if not math.isclose(
        float(minimum_frame_interval),
        _FRAME_PERIOD_SECONDS,
        rel_tol=0.0,
        abs_tol=_VIDEO_TIMESTAMP_TOLERANCE_SECONDS,
    ):
        raise RuntimeError("DeepTalk 0.3.1 _min_frame_interval has an unexpected value")
    if voice_extractor is not None and not hasattr(voice_extractor, "extract_from_samples"):
        raise RuntimeError("DeepTalk 0.3.1 voice_extractor has an unexpected structure")
    if not isinstance(voice_profiles, dict):
        raise TypeError("DeepTalk 0.3.1 voice_profiles has an unexpected structure")
    if not callable(remove_stale_faces):
        raise TypeError("DeepTalk 0.3.1 face expiry hook has an unexpected structure")

    backend._min_frame_interval = 0.0
    backend.max_track_age = math.inf
    backend.voice_extractor = None
    backend.voice_profiles.clear()
    vad_backend._last_reset_time = math.inf
    face_backend._remove_stale_faces = lambda: None

    face_module = import_module(face_type.__module__)
    face_module_backend = cast(Any, face_module)
    disappearance_interval = getattr(face_module, "FACE_DISAPPEARANCE_INTERVAL_SECONDS", None)
    if isinstance(disappearance_interval, bool) or not isinstance(disappearance_interval, Real):
        raise TypeError("DeepTalk 0.3.1 face disappearance interval has an unexpected value")
    face_module_backend.FACE_DISAPPEARANCE_INTERVAL_SECONDS = 0


def _copy_speech_interval(
    utterance: object | None,
    *,
    source_offset_seconds: float,
    relative_end_seconds: float,
    include_incomplete: bool,
) -> DeepTalkSpeechInterval | None:
    """Copy a terminal or EOF-truncated DeepTalk utterance without retaining audio frames."""
    if utterance is None:
        return None

    state_name = getattr(getattr(utterance, "turn_state", None), "name", None)
    if not isinstance(state_name, str):
        raise TypeError("DeepTalk utterance does not expose a named turn_state")
    terminal_statuses: dict[str, _SpeechIntervalStatus] = {
        "TURN_END": "confirmed",
        "TURN_REJECTED": "rejected",
    }
    active_states = {"TURN_START", "TURN_CONTINUE", "TURN_CONFIRMED", "TURN_SILENCE"}
    if include_incomplete:
        if state_name not in active_states:
            return None
        interval_status: _SpeechIntervalStatus = "incomplete"
    else:
        terminal_status = terminal_statuses.get(state_name)
        if terminal_status is None:
            return None
        interval_status = terminal_status

    duration_method = getattr(utterance, "duration_seconds", None)
    if not callable(duration_method):
        raise TypeError("DeepTalk utterance does not expose duration_seconds()")
    duration_seconds = float(duration_method())
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise RuntimeError(f"DeepTalk returned an invalid utterance duration: {duration_seconds}")

    source_end_seconds = source_offset_seconds + relative_end_seconds
    source_start_seconds = max(source_offset_seconds, source_end_seconds - duration_seconds)
    if source_end_seconds <= source_start_seconds:
        return None
    return DeepTalkSpeechInterval(
        source_start_seconds=source_start_seconds,
        source_end_seconds=source_end_seconds,
        status=interval_status,
    )


def _prune_deeptalk_video_buffer(backend: Any, *, before_seconds: float) -> None:
    """Discard mouth images older than completed score windows."""
    speaker_detector = getattr(backend, "_speaker_detector", None)
    if speaker_detector is None:
        return
    video_buffer = getattr(speaker_detector, "video_buffer", None)
    if not isinstance(video_buffer, dict):
        raise TypeError("DeepTalk 0.3.1 video_buffer has an unexpected structure")

    for face_id, frames in list(video_buffer.items()):
        retained_frames: list[Any] = []
        for frame_record in cast(list[Any], frames):
            if not isinstance(frame_record, tuple) or len(frame_record) != 2:
                raise RuntimeError("DeepTalk 0.3.1 video buffer record has an unexpected structure")
            timestamp_seconds = float(frame_record[1])
            if timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS >= before_seconds:
                retained_frames.append(frame_record)
        video_buffer[face_id] = retained_frames


def _copy_face_observations(
    source_timestamp_seconds: float,
    profiles: object,
) -> list[DeepTalkFaceObservation]:
    observations: list[DeepTalkFaceObservation] = []
    for profile in cast(list[Any], profiles):
        rectangle = profile.face_rectangle
        head_pose = profile.head_pose
        observations.append(
            DeepTalkFaceObservation(
                source_timestamp_seconds=source_timestamp_seconds,
                face_id=int(profile.id),
                bounding_box_xywh=(
                    float(rectangle.x),
                    float(rectangle.y),
                    float(rectangle.width),
                    float(rectangle.height),
                ),
                head_pose_yaw_pitch_roll_degrees=(
                    float(head_pose.yaw),
                    float(head_pose.pitch),
                    float(head_pose.roll),
                ),
                raw_face_quality_score=float(profile.face_image_score),
            )
        )
    return observations


def _evaluate_score_window(
    backend: Any,
    *,
    source_offset_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> DeepTalkScoreWindow:
    scores = cast(Mapping[Any, Any], backend.evaluate(start_seconds, end_seconds))
    copied_scores = {int(face_id): float(score) for face_id, score in scores.items()}

    return DeepTalkScoreWindow(
        source_start_seconds=source_offset_seconds + start_seconds,
        source_end_seconds=source_offset_seconds + end_seconds,
        raw_scores=copied_scores,
    )
