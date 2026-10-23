"""Typed configuration and immutable results for complete video analysis.

Construction performs no I/O. Convenience save/render methods delegate to integration boundaries;
no model, tracker, decoder or backend-owned object is retained in these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from talkingfacekit.active_speaker import ActiveSpeakerResult, CandidatePolicy
from talkingfacekit.synchronization import SynchronizationPolicy, SynchronizationResult
from talkingfacekit.video import VideoSource
from talkingfacekit.visual import VisualPolicy, VisualRegion, VisualResult, VisualStatus

if TYPE_CHECKING:
    from talkingfacekit.sequence import TalkingFaceSequence


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configure explicit stages without loading models.

    Parameters
    ----------
    candidate_policy
        Structural duration, same-face coverage and continuity limits.
    visual_policy
        Optional pose, size and framing filter. ``None`` disables this stage.
    syncnet_model
        Explicit local official SyncNet-v2 checkpoint, or ``None`` to disable verification.
        Construction normalizes the path; loading and hash verification occur during analysis.
    synchronization_policy
        A/V offset and confidence thresholds. Custom limits require ``syncnet_model``.

    Notes
    -----
    DeepTalk-ASD remains the experimental active-speaker backend. Optional extras are required
    only when executing their stages. Defaults preserve the existing structural-only selection.
    """

    candidate_policy: CandidatePolicy = field(default_factory=CandidatePolicy)
    visual_policy: VisualPolicy | None = None
    syncnet_model: Path | None = None
    synchronization_policy: SynchronizationPolicy = field(default_factory=SynchronizationPolicy)

    def __post_init__(self) -> None:
        """Validate stage configuration without accessing files or optional frameworks."""
        if not isinstance(self.candidate_policy, CandidatePolicy):
            raise TypeError("candidate_policy must be a CandidatePolicy")
        if self.visual_policy is not None and not isinstance(self.visual_policy, VisualPolicy):
            raise TypeError("visual_policy must be a VisualPolicy or None")
        if not isinstance(self.synchronization_policy, SynchronizationPolicy):
            raise TypeError("synchronization_policy must be a SynchronizationPolicy")
        if self.syncnet_model is not None:
            object.__setattr__(self, "syncnet_model", Path(self.syncnet_model))
        elif self.synchronization_policy != SynchronizationPolicy():
            raise ValueError("custom synchronization_policy requires syncnet_model")


@dataclass(frozen=True, slots=True)
class AnalysisTimings:
    """Measure wall time in seconds, independently of source-media duration.

    Attributes
    ----------
    active_speaker_seconds
        Active-speaker model construction, media preparation and analysis.
    synchronization_initialization_seconds
        SyncNet construction time, zero when disabled; shared across a collection run.
    visual_filter_seconds
        Pure geometry filtering time, zero when disabled.
    total_seconds
        This video's complete analysis, excluding initial source inspection, saving and rendering.
        Shared SyncNet construction is recorded separately and is not added once per video.
        Synchronization execution time is also available in its result's ``elapsed_seconds``.
    """

    active_speaker_seconds: float
    synchronization_initialization_seconds: float = 0.0
    visual_filter_seconds: float = 0.0
    total_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Require finite, non-negative elapsed durations."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TalkingFaceSegment:
    """Describe an assessed interval and retain its original media identity.

    Attributes
    ----------
    source
        Shared complete-stream identity and metadata; no media pixels are retained.
    start_seconds, end_seconds
        Half-open bounds in seconds on the original source timeline.
    face_id
        Local face-track identity, meaningful only within this analysis.
    status, reason_codes
        Final enabled-stage classification and machine-readable explanations. ``candidate`` means
        selected under this configuration, not a guarantee of training suitability.
    """

    source: VideoSource
    start_seconds: float
    end_seconds: float
    face_id: int
    status: VisualStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate source identity and source-timeline classification."""
        if not isinstance(self.source, VideoSource):
            raise TypeError("segment source must be a VideoSource")
        VisualRegion(
            self.start_seconds, self.end_seconds, self.face_id, self.status, self.reason_codes
        )

    @property
    def duration_seconds(self) -> float:
        """Duration of this interval in source seconds."""
        return self.end_seconds - self.start_seconds

    def to_sequence(self) -> TalkingFaceSequence:
        """Create a lightweight sequence sharing the source, without decoding or copying media.

        Returns
        -------
        TalkingFaceSequence
            Source-timeline interval with no attached tracks or inference results.
        """
        from talkingfacekit.sequence import TalkingFaceSequence

        return TalkingFaceSequence(self.source, self.start_seconds, self.end_seconds)


@dataclass(frozen=True, slots=True)
class VideoAnalysisResult:
    """Collect every enabled analysis stage for one source interval.

    Attributes
    ----------
    source, start_seconds, end_seconds
        Original source and requested half-open interval; ``None`` ends at EOF.
    active_speaker
        Immutable observations, raw scores, structural candidates, reasons and provenance.
    synchronization, visual_filter
        Optional later-stage results. Raw upstream evidence is never modified.
    timings
        Recorded wall times. No decoding or inference is performed when querying results.

    Notes
    -----
    ``segments`` describes assessed structural runs after enabled stages; it is not a complete
    partition of the video. Preliminary non-speech/no-face/ambiguous regions remain available in
    ``active_speaker.preliminary_segments``. ``selected_segments`` contains only retained runs.
    """

    source: VideoSource
    start_seconds: float
    end_seconds: float | None
    active_speaker: ActiveSpeakerResult
    synchronization: SynchronizationResult | None = None
    visual_filter: VisualResult | None = None
    timings: AnalysisTimings = field(default_factory=lambda: AnalysisTimings(0.0))

    def __post_init__(self) -> None:
        """Validate the requested interval and synchronization coverage."""
        if not isinstance(self.source, VideoSource):
            raise TypeError("result source must be a VideoSource")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be finite and non-negative")
        if self.end_seconds is not None and (
            not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must be finite and after start_seconds")
        if not isinstance(self.active_speaker, ActiveSpeakerResult):
            raise TypeError("active_speaker must be an ActiveSpeakerResult")
        if self.synchronization is not None and not isinstance(
            self.synchronization, SynchronizationResult
        ):
            raise TypeError("synchronization must be a SynchronizationResult or None")
        if self.visual_filter is not None and not isinstance(self.visual_filter, VisualResult):
            raise TypeError("visual_filter must be a VisualResult or None")
        if not isinstance(self.timings, AnalysisTimings):
            raise TypeError("timings must be AnalysisTimings")
        regions = decision_regions(self.active_speaker, self.synchronization)
        for region in regions:
            if region.source_start_seconds < self.start_seconds - 1e-8 or (
                self.end_seconds is not None and region.source_end_seconds > self.end_seconds + 1e-8
            ):
                raise ValueError(
                    "analysis decisions must stay inside the requested source interval"
                )

    @property
    def segments(self) -> tuple[TalkingFaceSegment, ...]:
        """Assessed runs with final enabled-stage status, source bounds and reasons."""
        regions = (
            self.visual_filter.decisions
            if self.visual_filter is not None
            else decision_regions(self.active_speaker, self.synchronization)
        )
        return tuple(
            TalkingFaceSegment(
                self.source,
                item.source_start_seconds,
                item.source_end_seconds,
                item.face_id,
                item.status,
                item.reason_codes,
            )
            for item in regions
        )

    @property
    def selected_segments(self) -> tuple[TalkingFaceSegment, ...]:
        """Segments retained by every enabled stage, on the original media timeline."""
        return tuple(item for item in self.segments if item.status == "candidate")

    def save(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        """Persist evidence, selection, source hash and provenance as versioned JSON.

        Parameters
        ----------
        output_path
            Destination ending in ``.json``. Parent directories are created as needed.
        overwrite
            Whether an existing result may be replaced after successful serialization.

        Returns
        -------
        pathlib.Path
            Saved JSON path. The source video is read for hashing and never modified.
        """
        from talkingfacekit.io.analysis import save_analysis_result

        return save_analysis_result(self, output_path, overwrite=overwrite)

    def render_report(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        """Render in-memory evidence as diagnostic HTML without rerunning models.

        Parameters
        ----------
        output_path
            HTML destination. Its parent directory must exist. The original video must remain
            accessible at its source path for playback.
        overwrite
            Whether the explicitly requested destination may be replaced.

        Returns
        -------
        pathlib.Path
            HTML report referencing the original media without copying or re-encoding it.
        """
        from talkingfacekit.demo import render_active_speaker_report
        from talkingfacekit.sequence import TalkingFaceSequence

        return render_active_speaker_report(
            TalkingFaceSequence(self.source, self.start_seconds, self.end_seconds),
            self.active_speaker,
            output_path,
            overwrite=overwrite,
            synchronization=self.synchronization,
            visual_filter=self.visual_filter,
        )


def decision_regions(
    result: ActiveSpeakerResult, synchronization: SynchronizationResult | None = None
) -> tuple[VisualRegion, ...]:
    """Combine structural decisions with complete optional synchronization evidence.

    Parameters
    ----------
    result
        Structural runs, including rejected runs.
    synchronization
        Optional verification covering every structural candidate exactly once.

    Returns
    -------
    tuple of VisualRegion
        Ordered decisions with upstream rejections preserved.

    Raises
    ------
    ValueError
        If synchronization decisions omit, duplicate or add a structural candidate.
    """
    regions = [
        VisualRegion(
            run.source_start_seconds,
            run.source_end_seconds,
            run.face_id,
            run.status,
            run.reason_codes,
        )
        for run in result.candidate_runs
        if synchronization is None or run.status != "candidate"
    ]
    if synchronization is not None:
        expected = {
            (run.source_start_seconds, run.source_end_seconds, run.face_id)
            for run in result.candidate_runs
            if run.status == "candidate"
        }
        actual = {
            (item.source_start_seconds, item.source_end_seconds, item.face_id)
            for item in synchronization.decisions
        }
        if actual != expected or len(actual) != len(synchronization.decisions):
            raise ValueError("sync decisions must cover every structural candidate exactly once")
        regions.extend(
            VisualRegion(
                item.source_start_seconds,
                item.source_end_seconds,
                item.face_id,
                item.status,
                item.reason_codes,
            )
            for item in synchronization.decisions
        )
    return tuple(sorted(regions, key=lambda item: item.source_start_seconds))
