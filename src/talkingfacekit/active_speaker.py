"""Framework-independent active-speaker evidence and structural selection policies.

The initial experimental backend is DeepTalk-ASD. Scores remain uncalibrated backend evidence;
these immutable values and pure selection functions perform no model loading or media I/O.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Literal, cast

_VIDEO_FPS = 25
_AUDIO_SAMPLE_RATE_HZ = 16_000
_VIDEO_TIMESTAMP_TOLERANCE_SECONDS = 1e-9

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
class FaceObservation:
    """Record one DeepTalk face observation on the source-media timeline.

    Parameters
    ----------
    source_timestamp_seconds
        Timestamp of the 25 Hz source-timeline slot submitted to DeepTalk, in seconds.
    face_id
        DeepTalk's local face-track ID. The same ID keys score-window results, but it is not a
        physical-person identity and can be reused after a track disappears.
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
class ActiveSpeakerScoreWindow:
    """Preserve unchanged DeepTalk scores for one source-timeline window.

    ``raw_scores`` contains DeepTalk's final score for each local face-track ID. Values are not
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
        Immutable snapshot mapping DeepTalk local face-track IDs to unchanged scores.
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
class SpeechInterval:
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
class ActiveSpeakerProvenance:
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
class ShotBoundary:
    """Record one PySceneDetect scene cut on the source timeline.

    Parameters
    ----------
    source_timestamp_seconds
        Timestamp of the sampled 25 Hz video slot where ``ContentDetector`` found the cut, in
        seconds on the original source-media timeline.

    Raises
    ------
    ValueError
        If the timestamp is invalid.
    """

    source_timestamp_seconds: float

    def __post_init__(self) -> None:
        """Validate the source timestamp."""
        if not math.isfinite(self.source_timestamp_seconds) or self.source_timestamp_seconds < 0:
            raise ValueError(
                "shot boundary source_timestamp_seconds must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class AudioTimelineRepair:
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
class PreliminarySegment:
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
class CandidatePolicy:
    """Configure conservative structural checks for continuous candidate runs.

    These defaults are intentionally strict starting points for dataset curation, not calibrated
    guarantees. They are kept explicit so a future evaluation set can replace them without
    changing the evidence or run-building contracts.

    Parameters
    ----------
    minimum_duration_seconds
        Minimum final run duration, including any explicitly bridged temporal gaps.
    minimum_face_coverage
        Minimum fraction of expected 25 Hz observations for both final runs and bridgeable gaps.
    maximum_face_gap_seconds
        Maximum time without the same face being observed in a final run or bridgeable gap.
    maximum_bridge_gap_seconds
        Maximum rejected interval that may be bridged when positive ASD evidence for the same face
        exists immediately before and after it. The same limit acts as offline look-ahead grace;
        trailing rejected time is never added without later positive evidence.
    """

    minimum_duration_seconds: float = 2.0
    minimum_face_coverage: float = 0.90
    maximum_face_gap_seconds: float = 0.20
    maximum_bridge_gap_seconds: float = 1.25

    def __post_init__(self) -> None:
        """Validate duration, coverage, and visibility-gap thresholds."""
        if not math.isfinite(self.minimum_duration_seconds) or self.minimum_duration_seconds <= 0:
            raise ValueError("minimum_duration_seconds must be finite and positive")
        if not math.isfinite(self.minimum_face_coverage) or not 0 < self.minimum_face_coverage <= 1:
            raise ValueError("minimum_face_coverage must be finite and in (0, 1]")
        if not math.isfinite(self.maximum_face_gap_seconds) or self.maximum_face_gap_seconds < 0:
            raise ValueError("maximum_face_gap_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_bridge_gap_seconds)
            or self.maximum_bridge_gap_seconds < 0
        ):
            raise ValueError("maximum_bridge_gap_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """Summarize one continuous same-face run after structural policy checks.

    A run is formed from preliminary candidates for one DeepTalk face-track ID. Brief rejected gaps
    may be retained only by the explicit temporal-continuity policy and remain listed in
    ``bridged_gap_intervals``. Raw-score summary fields cover positive candidate evidence only;
    unchanged negative scores remain in the preliminary segments and score windows. Passing this
    stage means the run is long enough and its face detections are sufficiently dense and stable.
    It still requires camera-facing, visual-quality, synchronization, and calibrated active-speaker
    checks before it can become an accepted training clip.
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
    bridged_gap_intervals: tuple[tuple[float, float], ...] = ()

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
        previous_end_seconds = self.source_start_seconds
        for interval in self.bridged_gap_intervals:
            if len(interval) != 2:
                raise ValueError("bridged gap intervals must contain start and end seconds")
            start_seconds, end_seconds = interval
            if (
                not math.isfinite(start_seconds)
                or not math.isfinite(end_seconds)
                or start_seconds < self.source_start_seconds
                or end_seconds > self.source_end_seconds
                or end_seconds <= start_seconds
                or start_seconds < previous_end_seconds
            ):
                raise ValueError("bridged gap intervals must be ordered within the candidate run")
            previous_end_seconds = end_seconds

    @property
    def duration_seconds(self) -> float:
        """Continuous run duration on the source-media timeline."""
        return self.source_end_seconds - self.source_start_seconds


@dataclass(frozen=True, slots=True)
class ActiveSpeakerResult:
    """Contain copied face observations and diagnostic score windows for one sequence.

    Parameters
    ----------
    face_observations
        Observed local track IDs, boxes, raw head pose, and raw quality at submitted 25 Hz video
        slots.
    speech_intervals
        Completed, rejected, and EOF-truncated VAD intervals on the source timeline.
    score_windows
        Consecutive source-timeline windows containing unchanged DeepTalk scores.
    provenance
        Backend version, fixed media rates, and optional capability status.
    shot_boundaries
        Scene cuts detected by PySceneDetect from the frames already sampled for DeepTalk. These
        boundaries are independent of DeepTalk face-track IDs and remain on the source timeline.
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

    face_observations: tuple[FaceObservation, ...]
    speech_intervals: tuple[SpeechInterval, ...]
    score_windows: tuple[ActiveSpeakerScoreWindow, ...]
    provenance: ActiveSpeakerProvenance
    shot_boundaries: tuple[ShotBoundary, ...] = ()
    audio_timeline_repairs: tuple[AudioTimelineRepair, ...] = ()
    issues: tuple[str, ...] = ()
    preliminary_segments: tuple[PreliminarySegment, ...] = ()
    candidate_policy: CandidatePolicy = CandidatePolicy()
    candidate_runs: tuple[CandidateRun, ...] = ()


def build_preliminary_segments(
    result: ActiveSpeakerResult,
) -> tuple[PreliminarySegment, ...]:
    """Build conservative pre-calibration decisions from one DeepTalk result.

    Each raw score window is split at non-rejected VAD boundaries and scene-cut boundaries.
    Non-speech, no-face, multi-face, changing-identity, and non-positive-score regions are rejected.
    A sole visible face with speech but no corresponding score is uncertain. Only speech regions
    with one identity and a positive raw score become candidates, and candidates still require all
    later policy stages.

    Parameters
    ----------
    result
        Immutable DeepTalk evidence on the source timeline. Existing preliminary segments are
        ignored so callers can reproduce the classification from the raw observations.

    Returns
    -------
    tuple[PreliminarySegment, ...]
        Ordered, non-empty subintervals covering every available score window.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`ActiveSpeakerResult`.
    """
    if not isinstance(result, ActiveSpeakerResult):
        raise TypeError(f"result must be a ActiveSpeakerResult, got {type(result).__name__}")

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
    segments: list[PreliminarySegment] = []

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
        for shot_boundary in result.shot_boundaries:
            if (
                window.source_start_seconds
                < shot_boundary.source_timestamp_seconds
                < window.source_end_seconds
            ):
                boundaries.add(shot_boundary.source_timestamp_seconds)

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
    result: ActiveSpeakerResult,
    policy: CandidatePolicy | None = None,
) -> tuple[CandidateRun, ...]:
    """Apply temporal continuity and structural stability checks to candidate evidence.

    Adjacent candidates for the same face are merged within one shot. A short rejected interval can
    also be bridged when it contains only an ASD-negative or brief no-speech decision, the same sole
    face remains sufficiently visible, and positive ASD evidence resumes immediately afterward.
    Raw preliminary decisions remain unchanged. Scene cuts, ambiguity, identity changes,
    missing evidence, excessive duration, or unstable visibility always prevent bridging.

    Parameters
    ----------
    result
        DeepTalk evidence containing preliminary segment decisions.
    policy
        Thresholds to apply. When omitted, ``result.candidate_policy`` is used.

    Returns
    -------
    tuple[CandidateRun, ...]
        Ordered same-face runs. Runs that fail a threshold remain present as rejected evidence.
    """
    if not isinstance(result, ActiveSpeakerResult):
        raise TypeError(f"result must be a ActiveSpeakerResult, got {type(result).__name__}")
    resolved_policy = result.candidate_policy if policy is None else policy
    if not isinstance(resolved_policy, CandidatePolicy):
        raise TypeError(f"policy must be a CandidatePolicy, got {type(resolved_policy).__name__}")

    observations_by_face: dict[int, list[float]] = {}
    for observation in result.face_observations:
        observations_by_face.setdefault(observation.face_id, []).append(
            observation.source_timestamp_seconds
        )
    for timestamps in observations_by_face.values():
        timestamps.sort()

    merged_segments: list[list[PreliminarySegment]] = []
    pending_gap: list[PreliminarySegment] = []
    for segment in sorted(
        result.preliminary_segments,
        key=lambda item: (item.source_start_seconds, item.source_end_seconds),
    ):
        if segment.status != "candidate":
            if merged_segments:
                pending_gap.append(segment)
            continue
        if segment.face_id is None or segment.raw_score is None:
            raise ValueError("candidate preliminary segments require a face ID and raw score")
        if merged_segments:
            previous_candidate = merged_segments[-1][-1]
            face_observation_timestamps = observations_by_face.get(segment.face_id, [])
            if _can_join_candidate_segments(
                result,
                previous_candidate,
                segment,
                pending_gap,
                face_observation_timestamps,
                policy=resolved_policy,
            ):
                merged_segments[-1].extend(pending_gap)
                merged_segments[-1].append(segment)
                pending_gap = []
                continue
        merged_segments.append([segment])
        pending_gap = []

    return tuple(
        _assess_candidate_run(
            segments,
            observations_by_face.get(cast(int, segments[0].face_id), []),
            video_sample_rate_hz=result.provenance.video_sample_rate_hz,
            policy=resolved_policy,
        )
        for segments in merged_segments
    )


def _can_join_candidate_segments(
    result: ActiveSpeakerResult,
    previous: PreliminarySegment,
    current: PreliminarySegment,
    gap_segments: list[PreliminarySegment],
    face_observation_timestamps: list[float],
    *,
    policy: CandidatePolicy,
) -> bool:
    """Return whether two positive ASD segments have safe temporal continuity."""
    if previous.face_id != current.face_id:
        return False
    gap_start_seconds = previous.source_end_seconds
    gap_end_seconds = current.source_start_seconds
    gap_duration_seconds = gap_end_seconds - gap_start_seconds
    if any(
        gap_start_seconds - _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
        <= boundary.source_timestamp_seconds
        <= gap_end_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
        for boundary in result.shot_boundaries
    ):
        return False
    if gap_duration_seconds <= _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
        return not gap_segments
    if (
        not gap_segments
        or gap_duration_seconds
        > policy.maximum_bridge_gap_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
    ):
        return False

    face_id = cast(int, previous.face_id)
    bridgeable_reasons = {
        ("non_positive_active_speaker_score",),
        ("no_speech",),
    }
    if any(
        segment.visible_face_ids != (face_id,) or segment.reason_codes not in bridgeable_reasons
        for segment in gap_segments
    ):
        return False
    _, _, visibility_fraction, maximum_gap_seconds = _face_visibility_metrics(
        face_observation_timestamps,
        start_seconds=gap_start_seconds,
        end_seconds=gap_end_seconds,
        video_sample_rate_hz=result.provenance.video_sample_rate_hz,
    )
    return (
        visibility_fraction >= policy.minimum_face_coverage
        and maximum_gap_seconds <= policy.maximum_face_gap_seconds
    )


def _assess_candidate_run(
    segments: list[PreliminarySegment],
    face_observation_timestamps: list[float],
    *,
    video_sample_rate_hz: int,
    policy: CandidatePolicy,
) -> CandidateRun:
    start_seconds = segments[0].source_start_seconds
    end_seconds = segments[-1].source_end_seconds
    face_id = cast(int, segments[0].face_id)
    duration_seconds = end_seconds - start_seconds
    (
        face_observation_count,
        expected_observation_count,
        visibility_fraction,
        maximum_gap_seconds,
    ) = _face_visibility_metrics(
        face_observation_timestamps,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        video_sample_rate_hz=video_sample_rate_hz,
    )
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

    candidate_segments = [segment for segment in segments if segment.status == "candidate"]
    raw_scores = [cast(float, segment.raw_score) for segment in candidate_segments]
    scored_duration_seconds = math.fsum(
        segment.source_end_seconds - segment.source_start_seconds for segment in candidate_segments
    )
    duration_weighted_score = (
        math.fsum(
            cast(float, segment.raw_score)
            * (segment.source_end_seconds - segment.source_start_seconds)
            for segment in candidate_segments
        )
        / scored_duration_seconds
    )
    minimum_raw_score = min(raw_scores)
    maximum_raw_score = max(raw_scores)
    # Positive duration weights guarantee these bounds; floating-point rounding may exceed them.
    duration_weighted_score = min(
        maximum_raw_score, max(minimum_raw_score, duration_weighted_score)
    )
    bridged_gap_intervals = tuple(
        (previous.source_end_seconds, current.source_start_seconds)
        for previous, current in pairwise(candidate_segments)
        if current.source_start_seconds - previous.source_end_seconds
        > _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
    )
    return CandidateRun(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        status=status,
        face_id=face_id,
        reason_codes=tuple(reason_codes),
        preliminary_segment_count=len(segments),
        face_observation_count=face_observation_count,
        expected_face_observation_count=expected_observation_count,
        face_visibility_fraction=visibility_fraction,
        maximum_face_gap_seconds=maximum_gap_seconds,
        raw_score_min=minimum_raw_score,
        raw_score_mean=duration_weighted_score,
        raw_score_max=maximum_raw_score,
        bridged_gap_intervals=bridged_gap_intervals,
    )


def _face_visibility_metrics(
    face_observation_timestamps: list[float],
    *,
    start_seconds: float,
    end_seconds: float,
    video_sample_rate_hz: int,
) -> tuple[int, int, float, float]:
    """Measure expected and observed same-face coverage and its largest gap."""
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
    return (
        len(observed_timestamps),
        expected_observation_count,
        visibility_fraction,
        maximum_gap_seconds,
    )


def _classify_preliminary_interval(
    score_window: ActiveSpeakerScoreWindow,
    active_speech_intervals: tuple[SpeechInterval, ...],
    face_observations: list[FaceObservation],
    *,
    start_seconds: float,
    end_seconds: float,
) -> PreliminarySegment:
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
        return PreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("no_speech",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )

    if not visible_face_ids:
        return PreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("no_visible_face",),
            visible_face_ids=(),
            raw_score=None,
        )
    if any(len(face_ids) > 1 for face_ids in faces_by_timestamp.values()):
        return PreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=None,
            reason_codes=("multiple_visible_faces",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )
    if len(visible_face_ids) > 1:
        return PreliminarySegment(
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
        return PreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="uncertain",
            face_id=face_id,
            reason_codes=("no_active_speaker_score",),
            visible_face_ids=visible_face_ids,
            raw_score=None,
        )
    if raw_score <= 0:
        return PreliminarySegment(
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="rejected",
            face_id=face_id,
            reason_codes=("non_positive_active_speaker_score",),
            visible_face_ids=visible_face_ids,
            raw_score=raw_score,
        )
    return PreliminarySegment(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        status="candidate",
        face_id=face_id,
        reason_codes=("single_visible_face", "positive_raw_active_speaker_score"),
        visible_face_ids=visible_face_ids,
        raw_score=raw_score,
    )
