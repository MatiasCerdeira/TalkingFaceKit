"""Backend-independent synchronization evidence and conservative candidate decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Literal

SyncStatus = Literal["synchronized", "offset_detected", "uncertain"]


@dataclass(frozen=True, slots=True)
class SynchronizationPolicy:
    """Configure experimental sync verification, in seconds and raw SyncNet distance units.

    Positive offsets mean audible speech is late relative to the visible mouth. Thresholds are
    explicit experimental defaults, not calibrated probabilities or universal perceptual limits.
    Windows partition each candidate, never crossing its identity or shot boundaries.

    Attributes
    ----------
    tolerance_seconds
        Maximum absolute offset passing verification, inclusive.
    minimum_confidence
        Minimum median-minus-best feature distance; not a probability.
    window_seconds, minimum_window_seconds
        Maximum partition duration and minimum measurable duration, in source seconds.
    maximum_offset_seconds
        Maximum searched lag magnitude. Short windows can search a smaller range.
    """

    tolerance_seconds: float = 0.08
    minimum_confidence: float = 3.0
    window_seconds: float = 5.0
    minimum_window_seconds: float = 2.0
    maximum_offset_seconds: float = 1.0

    def __post_init__(self) -> None:
        """Reject non-finite, inconsistent, or unresolvable thresholds."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_window_seconds > self.window_seconds:
            raise ValueError("minimum_window_seconds must not exceed window_seconds")
        if self.maximum_offset_seconds <= self.tolerance_seconds:
            raise ValueError("maximum_offset_seconds must exceed tolerance_seconds")
        if self.maximum_offset_seconds < 0.04:
            raise ValueError("maximum_offset_seconds must span at least one 25 Hz frame")


@dataclass(frozen=True, slots=True)
class SynchronizationWindow:
    """Record one source-time measurement, retaining failed measurements explicitly.

    ``estimated_offset_seconds`` is positive for late audio. Confidence is a raw difference
    between median and minimum feature distance, never a probability. A boundary optimum or
    weak confidence is uncertain; its numerical estimate remains diagnostic evidence only.

    Attributes
    ----------
    source_start_seconds, source_end_seconds
        Half-open measured bounds on the original source timeline.
    face_id
        Existing local face-track ID, with no cross-video identity meaning.
    status, reason_codes
        Measurement conclusion and stable diagnostic explanations.
    estimated_offset_seconds
        Best relative audio lag in seconds, or ``None`` without a measurement.
    confidence, minimum_distance
        Raw SyncNet distance evidence, or ``None`` when unmeasurable.
    search_limit_seconds
        Actual symmetric search range for this window, or ``None`` without inference.
    """

    source_start_seconds: float
    source_end_seconds: float
    face_id: int
    status: SyncStatus
    reason_codes: tuple[str, ...]
    estimated_offset_seconds: float | None = None
    confidence: float | None = None
    minimum_distance: float | None = None
    search_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate bounds, status, and finite model evidence."""
        _validate_bounds(self.source_start_seconds, self.source_end_seconds, self.face_id)
        if self.status not in {"synchronized", "offset_detected", "uncertain"}:
            raise ValueError("unsupported synchronization status")
        if not self.reason_codes or any(not reason for reason in self.reason_codes):
            raise ValueError("synchronization evidence requires reasons")
        evidence = (
            self.estimated_offset_seconds,
            self.confidence,
            self.minimum_distance,
            self.search_limit_seconds,
        )
        if any(value is not None and not math.isfinite(value) for value in evidence):
            raise ValueError("synchronization evidence must be finite")
        if self.status != "uncertain" and any(value is None for value in evidence):
            raise ValueError("measurable synchronization requires complete evidence")
        if self.search_limit_seconds is not None and self.search_limit_seconds <= 0:
            raise ValueError("search limit must be positive")
        if (
            self.estimated_offset_seconds is not None
            and self.search_limit_seconds is not None
            and abs(self.estimated_offset_seconds) > self.search_limit_seconds + 1e-9
        ):
            raise ValueError("offset must be inside the searched range")


@dataclass(frozen=True, slots=True)
class SynchronizationDecision:
    """Summarize a complete candidate; passing sync still requires later visual checks.

    Attributes
    ----------
    source_start_seconds, source_end_seconds, face_id
        Original candidate's half-open source bounds in seconds and local track ID.
    status, reason_codes
        Retained candidate, confirmed rejection, or uncertainty with explanations.
    estimated_offset_seconds
        Median reliable audio lag only when all windows are measurable and consistent.
    window_count
        Number of measured or explicitly uncertain windows covering this candidate.
    """

    source_start_seconds: float
    source_end_seconds: float
    face_id: int
    status: Literal["candidate", "rejected", "uncertain"]
    reason_codes: tuple[str, ...]
    estimated_offset_seconds: float | None
    window_count: int

    def __post_init__(self) -> None:
        """Validate a source-bounded decision without inventing a missing offset."""
        _validate_bounds(self.source_start_seconds, self.source_end_seconds, self.face_id)
        if self.status not in {"candidate", "rejected", "uncertain"} or not self.reason_codes:
            raise ValueError("synchronization decision requires a supported status and reasons")
        if (
            isinstance(self.window_count, bool)
            or not isinstance(self.window_count, int)
            or self.window_count < 0
        ):
            raise ValueError("window_count must be a non-negative integer")
        if self.estimated_offset_seconds is not None and not math.isfinite(
            self.estimated_offset_seconds
        ):
            raise ValueError("estimated offset must be finite")


@dataclass(frozen=True, slots=True)
class SynchronizationResult:
    """Keep synchronization evidence and provenance without modifying ASD observations.

    Attributes
    ----------
    policy, windows, decisions
        Applied policy, complete per-window evidence, and per-candidate conclusions.
    model_sha256, backend
        Exact checkpoint hash and model identity used for reproducibility.
    elapsed_seconds
        Wall time for candidate decoding and verification, excluding model construction.
    video_sample_rate_hz, audio_sample_rate_hz
        Model-input sampling rates; they do not redefine source timestamps.
    """

    policy: SynchronizationPolicy
    windows: tuple[SynchronizationWindow, ...]
    decisions: tuple[SynchronizationDecision, ...]
    model_sha256: str
    elapsed_seconds: float
    backend: str = "SyncNet-v2"
    video_sample_rate_hz: int = 25
    audio_sample_rate_hz: int = 16_000

    def __post_init__(self) -> None:
        """Require finite duration and explicit checkpoint identity."""
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("synchronization elapsed_seconds must be finite and non-negative")
        if len(self.model_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.model_sha256
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256 hex digest")


def candidate_windows(
    start_seconds: float, end_seconds: float, policy: SynchronizationPolicy
) -> tuple[tuple[float, float], ...]:
    """Partition a candidate completely into equally sized source-time windows.

    Parameters
    ----------
    start_seconds, end_seconds
        Finite half-open source bounds, with non-negative start and positive duration.
    policy
        Supplies the maximum window duration. Small windows remain explicit evidence.

    Returns
    -------
    tuple of tuple of float
        Contiguous ``(start, end)`` pairs covering the whole candidate.
    """
    _validate_bounds(start_seconds, end_seconds, 0)
    count = max(1, math.ceil((end_seconds - start_seconds) / policy.window_seconds))
    width = (end_seconds - start_seconds) / count
    return tuple(
        (
            start_seconds + index * width,
            end_seconds if index == count - 1 else start_seconds + (index + 1) * width,
        )
        for index in range(count)
    )


def decide_synchronization(
    start_seconds: float,
    end_seconds: float,
    face_id: int,
    windows: tuple[SynchronizationWindow, ...],
    policy: SynchronizationPolicy,
) -> SynchronizationDecision:
    """Reject confirmed offsets; require complete confident coverage to pass this stage.

    An offset in any reliable window rejects the full candidate. Inconsistent offsets remain
    in individual windows and suppress the global estimate, so a later repair cannot mistake
    a changing offset for one constant shift. Missing or uncertain coverage cannot pass.

    Parameters
    ----------
    start_seconds, end_seconds, face_id
        Candidate's half-open source bounds and local track identity.
    windows
        Ordered contiguous measurements for this candidate. Missing tail coverage is uncertain.
    policy
        Supplies the permitted spread for one aggregate offset, in seconds.

    Returns
    -------
    SynchronizationDecision
        Rejection, retained candidate, or uncertainty with optional constant-offset estimate.

    Raises
    ------
    ValueError
        If window bounds or identities conflict with this candidate.
    """
    _validate_bounds(start_seconds, end_seconds, face_id)
    previous_end = start_seconds
    for window in windows:
        if window.face_id != face_id or abs(window.source_start_seconds - previous_end) > 1e-6:
            raise ValueError("sync windows must continuously cover the same candidate face")
        if window.source_end_seconds > end_seconds + 1e-6:
            raise ValueError("sync window exceeds candidate")
        previous_end = window.source_end_seconds
    complete = bool(windows) and abs(previous_end - end_seconds) <= 1e-6
    offsets = [
        window.estimated_offset_seconds
        for window in windows
        if window.status != "uncertain" and window.estimated_offset_seconds is not None
    ]
    consistent = bool(offsets) and max(offsets) - min(offsets) <= policy.tolerance_seconds + 1e-9
    estimate = float(median(offsets)) if consistent else None
    reasons: tuple[str, ...]
    if any(window.status == "offset_detected" for window in windows):
        status: Literal["candidate", "rejected", "uncertain"] = "rejected"
        reasons = ("audio_video_offset",)
    elif complete and all(window.status == "synchronized" for window in windows):
        status = "candidate"
        reasons = ("synchronization_passed",)
    else:
        status = "uncertain"
        reasons = ("synchronization_not_verified",)
    if offsets and not consistent:
        reasons += ("inconsistent_audio_video_offsets",)
    if not complete or any(window.status == "uncertain" for window in windows):
        estimate = None
    return SynchronizationDecision(
        start_seconds, end_seconds, face_id, status, reasons, estimate, len(windows)
    )


def _validate_bounds(start: float, end: float, face_id: int) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("synchronization bounds must be finite, non-negative and ordered")
    if isinstance(face_id, bool) or not isinstance(face_id, int) or face_id < 0:
        raise ValueError("face_id must be a non-negative integer")
