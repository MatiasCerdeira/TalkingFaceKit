"""Experimental useful-face selection from timestamped pose and bounding-box evidence."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

VisualStatus = Literal["candidate", "rejected", "uncertain"]
_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class VisualPolicy:
    """Configure experimental geometry thresholds, independently of eye gaze or image quality.

    Attributes
    ----------
    maximum_absolute_yaw_degrees, maximum_absolute_pitch_degrees, maximum_absolute_roll_degrees
        Inclusive absolute head-angle limits in degrees. Defaults admit three-quarter views and
        moderate head movement; they do not require a frontal face or ocular eye contact.
    minimum_face_size_pixels
        Minimum shorter side of the original face box in source pixels, before resizing.
    minimum_face_width_fraction
        Minimum visible face width divided by source image width, in ``[0, 1]``.
    maximum_clipped_fraction
        Maximum box area outside the source image divided by original box area, in ``[0, 1]``.
    maximum_bad_gap_seconds
        Maximum brief failure or missing-evidence gap between passing observations to retain.
        Leading/trailing failures and gaps across upstream intervals are never bridged.
    minimum_duration_seconds
        Minimum duration after visual trimming, in source seconds.
    minimum_observed_fraction
        Minimum time fraction with actual measurements in a retained run. Missing measurements
        are never counted as passing observations, including when a brief gap is tolerated.
    minimum_passing_fraction
        Minimum duration fraction satisfying all frame-level checks, excluding tolerated gaps.
    """

    maximum_absolute_yaw_degrees: float = 70.0
    maximum_absolute_pitch_degrees: float = 40.0
    maximum_absolute_roll_degrees: float = 45.0
    minimum_face_size_pixels: float = 64.0
    minimum_face_width_fraction: float = 0.04
    maximum_clipped_fraction: float = 0.05
    maximum_bad_gap_seconds: float = 0.20
    minimum_duration_seconds: float = 2.0
    minimum_observed_fraction: float = 0.90
    minimum_passing_fraction: float = 0.90

    def __post_init__(self) -> None:
        """Reject non-finite, boolean, negative, and out-of-range thresholds."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "minimum_face_width_fraction",
            "maximum_clipped_fraction",
            "minimum_observed_fraction",
            "minimum_passing_fraction",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "maximum_absolute_yaw_degrees",
            "maximum_absolute_pitch_degrees",
            "maximum_absolute_roll_degrees",
        ):
            if getattr(self, name) > 180:
                raise ValueError(f"{name} must not exceed 180 degrees")
        if self.minimum_duration_seconds == 0 or self.minimum_observed_fraction == 0:
            raise ValueError("minimum duration and observed fraction must be positive")


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """Copy a face measurement without depending on a particular tracker.

    Attributes
    ----------
    source_timestamp_seconds, face_id
        Non-negative source seconds and local track identity.
    bounding_box_xywh
        Finite ``(x, y, width, height)`` in source pixels, with positive dimensions.
    head_pose_yaw_pitch_roll_degrees
        Finite raw angles in degrees, or ``None`` when pose is unavailable. No ocular gaze is
        inferred. Unavailable or physically implausible pose cannot pass the visual stage.
    """

    source_timestamp_seconds: float
    face_id: int
    bounding_box_xywh: tuple[float, float, float, float]
    head_pose_yaw_pitch_roll_degrees: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        """Validate source time, local identity, geometry, and optional angles."""
        _validate_bounds(
            self.source_timestamp_seconds, self.source_timestamp_seconds + 1, self.face_id
        )
        box = self.bounding_box_xywh
        if len(box) != 4 or not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
            raise ValueError("face box requires four finite pixel values and positive dimensions")
        pose = self.head_pose_yaw_pitch_roll_degrees
        if pose is not None and (len(pose) != 3 or not all(math.isfinite(v) for v in pose)):
            raise ValueError("pose requires three finite angles or None")


@dataclass(frozen=True, slots=True)
class VisualRegion:
    """Represent a half-open source interval, its local face, status, and explanations.

    Attributes
    ----------
    source_start_seconds, source_end_seconds, face_id
        Ordered non-negative source seconds and non-negative local track identity.
    status, reason_codes
        Stage-local selection and nonempty diagnostic codes. Upstream rejections and uncertainty
        may be supplied as inputs; this stage preserves them and never promotes them.
    """

    source_start_seconds: float
    source_end_seconds: float
    face_id: int
    status: VisualStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate interval bounds, selection status and reasons."""
        _validate_bounds(self.source_start_seconds, self.source_end_seconds, self.face_id)
        if self.status not in {"candidate", "rejected", "uncertain"}:
            raise ValueError("unsupported visual status")
        if not self.reason_codes or any(not isinstance(r, str) or not r for r in self.reason_codes):
            raise ValueError("visual regions require nonempty reason codes")


@dataclass(frozen=True, slots=True)
class VisualMetrics:
    """Summarize measured geometry without treating tolerated gaps as good observations.

    Attributes
    ----------
    observed_fraction, passing_fraction
        Fractions of interval duration with measurements and with passing measurements.
    maximum_absolute_yaw_pitch_roll_degrees
        Largest measured absolute angles per axis, or ``None`` without pose evidence.
    minimum_face_size_pixels, minimum_face_width_fraction, maximum_clipped_fraction
        Measured extrema in source pixels and dimensionless fractions, or ``None`` without boxes.
    """

    observed_fraction: float
    passing_fraction: float
    maximum_absolute_yaw_pitch_roll_degrees: tuple[float, float, float] | None
    minimum_face_size_pixels: float | None
    minimum_face_width_fraction: float | None
    maximum_clipped_fraction: float | None

    def __post_init__(self) -> None:
        """Validate dimensionless fractions and finite, non-negative geometry extrema."""
        for value in (self.observed_fraction, self.passing_fraction):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("visual time fractions must be in [0, 1]")
        if self.passing_fraction > self.observed_fraction + _EPSILON:
            raise ValueError("passing time cannot exceed observed time")
        for extreme in (
            self.minimum_face_size_pixels,
            self.minimum_face_width_fraction,
            self.maximum_clipped_fraction,
        ):
            if extreme is not None and (not math.isfinite(extreme) or extreme < 0):
                raise ValueError("visual geometry extrema must be finite and non-negative")
        if any(
            value is not None and value > 1
            for value in (self.minimum_face_width_fraction, self.maximum_clipped_fraction)
        ):
            raise ValueError("box fractions must be in [0, 1]")
        pose = self.maximum_absolute_yaw_pitch_roll_degrees
        if pose is not None and (
            len(pose) != 3 or any(not math.isfinite(v) or v < 0 for v in pose)
        ):
            raise ValueError("absolute angle extrema require three finite non-negative values")


@dataclass(frozen=True, slots=True)
class VisualDecision(VisualRegion):
    """Keep a visual region with metrics and explicitly tolerated source-time gaps.

    Attributes
    ----------
    metrics
        Geometry measured within this decision; ``None`` for unchanged upstream non-candidates.
    tolerated_gap_intervals
        Ordered ``(start, end)`` source-second pairs retained between passing evidence. Their
        original failures remain in ``VisualResult.windows``.
    """

    metrics: VisualMetrics | None = None
    tolerated_gap_intervals: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        """Keep tolerated gaps ordered and inside this decision's source interval."""
        VisualRegion.__post_init__(self)
        previous_end = self.source_start_seconds
        for start, end in self.tolerated_gap_intervals:
            _validate_bounds(start, end, self.face_id)
            if start < previous_end or end > self.source_end_seconds:
                raise ValueError("tolerated gaps must be ordered within their visual decision")
            previous_end = end


@dataclass(frozen=True, slots=True)
class VisualResult:
    """Preserve raw visual classifications alongside temporally filtered decisions.

    Attributes
    ----------
    policy, windows, decisions
        Applied thresholds, raw classified regions and final decisions on the source timeline.
    image_width, image_height, video_sample_rate_hz
        Source pixel dimensions and observation grid frequency; no source media is rewritten.
    """

    policy: VisualPolicy
    windows: tuple[VisualRegion, ...]
    decisions: tuple[VisualDecision, ...]
    image_width: int
    image_height: int
    video_sample_rate_hz: float


@dataclass(frozen=True, slots=True)
class _Cell:
    start: float
    end: float
    observation: VisualObservation | None
    reasons: tuple[str, ...]


def assess_visual_suitability(
    intervals: Iterable[VisualRegion],
    observations: Iterable[VisualObservation],
    *,
    image_width: int,
    image_height: int,
    video_sample_rate_hz: float = 25.0,
    policy: VisualPolicy | None = None,
) -> VisualResult:
    """Trim candidates by usable orientation, source resolution and in-frame box coverage.

    Parameters
    ----------
    intervals
        Ordered non-overlapping upstream decisions. Only candidates are evaluated; shot and
        identity boundaries remain hard stops, including between adjacent same-face inputs.
    observations
        Measurements on the source timeline, unique per local face and timestamp. Each supports
        at most one grid period after its timestamp. Long gaps are explicitly unmeasurable.
    image_width, image_height
        Positive integer source dimensions, in pixels.
    video_sample_rate_hz
        Positive finite observation rate; repeated source frames do not increase source detail.
    policy
        Experimental thresholds, or defaults when omitted. Q and ocular gaze are not used.

    Returns
    -------
    VisualResult
        Complete partition of input intervals, unchanged upstream non-candidates, raw visual
        failures, retained brief gaps and metrics. No I/O, model inference or mutation occurs.

    Raises
    ------
    ValueError
        If dimensions, rate, ordering, or observation uniqueness are invalid.
    """
    for dimension in (image_width, image_height):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("image dimensions must be positive integers")
    if (
        isinstance(video_sample_rate_hz, bool)
        or not math.isfinite(video_sample_rate_hz)
        or video_sample_rate_hz <= 0
    ):
        raise ValueError("video_sample_rate_hz must be finite and positive")
    resolved = VisualPolicy() if policy is None else policy
    by_face: dict[int, list[VisualObservation]] = defaultdict(list)
    for observation in observations:
        by_face[observation.face_id].append(observation)
    timestamps: dict[int, list[float]] = {}
    for face_id, values in by_face.items():
        values.sort(key=lambda item: item.source_timestamp_seconds)
        times = [item.source_timestamp_seconds for item in values]
        if len(times) != len(set(times)):
            raise ValueError("face observation timestamps must be unique per face")
        timestamps[face_id] = times
    windows: list[VisualRegion] = []
    decisions: list[VisualDecision] = []
    previous_end = 0.0
    period = 1 / video_sample_rate_hz
    for interval in intervals:
        if interval.source_start_seconds < previous_end - _EPSILON:
            raise ValueError("visual input intervals must be ordered and non-overlapping")
        previous_end = interval.source_end_seconds
        if interval.status != "candidate":
            decisions.append(
                VisualDecision(
                    interval.source_start_seconds,
                    interval.source_end_seconds,
                    interval.face_id,
                    interval.status,
                    interval.reason_codes,
                )
            )
            continue
        cells = _interval_cells(
            interval,
            by_face.get(interval.face_id, []),
            timestamps.get(interval.face_id, []),
            period,
            image_width,
            image_height,
            resolved,
        )
        windows.extend(_raw_regions(cells, interval.face_id))
        decisions.extend(
            _temporal_decisions(cells, interval.face_id, image_width, image_height, resolved)
        )
    return VisualResult(
        resolved, tuple(windows), tuple(decisions), image_width, image_height, video_sample_rate_hz
    )


def _box_metrics(
    observation: VisualObservation, width: int, height: int
) -> tuple[float, float, float]:
    x, y, w, h = observation.bounding_box_xywh
    visible_w = max(0.0, min(width, x + w) - max(0.0, x))
    visible_h = max(0.0, min(height, y + h) - max(0.0, y))
    return min(w, h), visible_w / width, max(0.0, min(1.0, 1 - visible_w * visible_h / (w * h)))


def _classify(
    observation: VisualObservation, width: int, height: int, policy: VisualPolicy
) -> tuple[str, ...]:
    reasons = []
    pose = observation.head_pose_yaw_pitch_roll_degrees
    if pose is None or any(abs(value) > 180 for value in pose):
        reasons.append("visual_pose_unavailable")
    else:
        for angle, limit, reason in zip(
            pose,
            (
                policy.maximum_absolute_yaw_degrees,
                policy.maximum_absolute_pitch_degrees,
                policy.maximum_absolute_roll_degrees,
            ),
            ("head_yaw_exceeded", "head_pitch_exceeded", "head_roll_exceeded"),
            strict=True,
        ):
            if abs(angle) > limit + _EPSILON:
                reasons.append(reason)
    size, fraction, clipped = _box_metrics(observation, width, height)
    if size + _EPSILON < policy.minimum_face_size_pixels:
        reasons.append("face_too_small_pixels")
    if fraction + _EPSILON < policy.minimum_face_width_fraction:
        reasons.append("face_too_small_relative")
    if clipped > policy.maximum_clipped_fraction + _EPSILON:
        reasons.append("face_clipped")
    return tuple(reasons)


def _interval_cells(
    interval: VisualRegion,
    values: list[VisualObservation],
    times: list[float],
    period: float,
    width: int,
    height: int,
    policy: VisualPolicy,
) -> list[_Cell]:
    cells = []
    cursor = interval.source_start_seconds
    first = max(0, bisect_left(times, cursor) - 1)
    for index in range(first, bisect_left(times, interval.source_end_seconds)):
        observation = values[index]
        start = max(cursor, observation.source_timestamp_seconds)
        end = min(interval.source_end_seconds, observation.source_timestamp_seconds + period)
        if index + 1 < len(times):
            end = min(end, times[index + 1])
        if end <= cursor + _EPSILON:
            continue
        if start > cursor + _EPSILON:
            cells.append(_Cell(cursor, start, None, ("visual_face_evidence_missing",)))
        cells.append(_Cell(start, end, observation, _classify(observation, width, height, policy)))
        cursor = end
    if cursor < interval.source_end_seconds - _EPSILON:
        cells.append(
            _Cell(cursor, interval.source_end_seconds, None, ("visual_face_evidence_missing",))
        )
    return cells


def _failure_status(reasons: tuple[str, ...]) -> VisualStatus:
    if not reasons:
        return "candidate"
    return "uncertain" if all(reason.startswith("visual_") for reason in reasons) else "rejected"


def _raw_regions(cells: list[_Cell], face_id: int) -> list[VisualRegion]:
    regions: list[VisualRegion] = []
    for cell in cells:
        reasons = cell.reasons or ("visual_policy_passed",)
        if regions and regions[-1].reason_codes == reasons:
            regions[-1] = replace(regions[-1], source_end_seconds=cell.end)
        else:
            regions.append(
                VisualRegion(cell.start, cell.end, face_id, _failure_status(cell.reasons), reasons)
            )
    return regions


def _metrics(cells: list[_Cell], width: int, height: int) -> VisualMetrics:
    duration = cells[-1].end - cells[0].start
    measured = [
        cell
        for cell in cells
        if cell.observation is not None
        and cell.observation.head_pose_yaw_pitch_roll_degrees is not None
        and all(abs(value) <= 180 for value in cell.observation.head_pose_yaw_pitch_roll_degrees)
    ]
    poses = [
        cell.observation.head_pose_yaw_pitch_roll_degrees
        for cell in measured
        if cell.observation is not None
        and cell.observation.head_pose_yaw_pitch_roll_degrees is not None
    ]
    boxes = [
        _box_metrics(cell.observation, width, height)
        for cell in cells
        if cell.observation is not None
    ]
    return VisualMetrics(
        min(1.0, math.fsum(c.end - c.start for c in measured) / duration),
        min(1.0, math.fsum(c.end - c.start for c in cells if not c.reasons) / duration),
        (
            max(abs(pose[0]) for pose in poses),
            max(abs(pose[1]) for pose in poses),
            max(abs(pose[2]) for pose in poses),
        )
        if poses
        else None,
        min(box[0] for box in boxes) if boxes else None,
        min(box[1] for box in boxes) if boxes else None,
        max(box[2] for box in boxes) if boxes else None,
    )


def _temporal_decisions(
    cells: list[_Cell], face_id: int, width: int, height: int, policy: VisualPolicy
) -> list[VisualDecision]:
    retained = [not cell.reasons for cell in cells]
    tolerated: list[tuple[float, float]] = []
    index = 0
    while index < len(cells):
        if retained[index]:
            index += 1
            continue
        first = index
        while index < len(cells) and not retained[index]:
            index += 1
        if (
            first > 0
            and index < len(cells)
            and cells[index - 1].end - cells[first].start
            <= policy.maximum_bad_gap_seconds + _EPSILON
        ):
            retained[first:index] = [True] * (index - first)
            tolerated.append((cells[first].start, cells[index - 1].end))
    decisions = []
    first = 0
    while first < len(cells):
        end = first + 1
        while (
            end < len(cells)
            and retained[end] == retained[first]
            and (retained[first] or cells[end].reasons == cells[first].reasons)
        ):
            end += 1
        group = cells[first:end]
        metrics = _metrics(group, width, height)
        reasons: tuple[str, ...]
        if retained[first]:
            status: VisualStatus = "candidate"
            reasons = ("visual_policy_passed",)
            if metrics.observed_fraction + _EPSILON < policy.minimum_observed_fraction:
                status, reasons = "uncertain", ("visual_evidence_insufficient",)
            elif metrics.passing_fraction + _EPSILON < policy.minimum_passing_fraction:
                status, reasons = "rejected", ("visual_passing_fraction_insufficient",)
            elif group[-1].end - group[0].start + _EPSILON < policy.minimum_duration_seconds:
                status, reasons = "rejected", ("visual_segment_too_short",)
        else:
            reasons = tuple(dict.fromkeys(reason for cell in group for reason in cell.reasons))
            status = _failure_status(reasons)
        gaps = tuple((a, b) for a, b in tolerated if group[0].start <= a and b <= group[-1].end)
        if status == "candidate" and gaps:
            reasons += ("brief_visual_gap_tolerated",)
        decisions.append(
            VisualDecision(group[0].start, group[-1].end, face_id, status, reasons, metrics, gaps)
        )
        first = end
    return decisions


def _validate_bounds(start: float, end: float, face_id: int) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("visual bounds must be finite, non-negative and ordered")
    if isinstance(face_id, bool) or not isinstance(face_id, int) or face_id < 0:
        raise ValueError("face_id must be a non-negative integer")
