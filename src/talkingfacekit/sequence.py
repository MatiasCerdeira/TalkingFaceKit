"""Core types and the user-facing talking-face sequence aggregate."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from talkingfacekit.io.video import inspect_video_metadata
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack, LandmarkTracker
from talkingfacekit.video import VideoSource


@dataclass(frozen=True, slots=True, eq=False)
class TalkingFaceSequence:
    """Collect the data and operations associated with a talking-face sequence.

    Direct construction performs no filesystem access. Use :meth:`from_video` to create a sequence
    from a local media file through the default video integration.

    Attributes
    ----------
    source
        Immutable source identity and complete-stream metadata. Multiple sequences may share it.
    start_seconds
        Start of the represented interval in seconds.
    end_seconds
        End of the represented interval in seconds, or ``None`` when it is unknown or unbounded.

    Raises
    ------
    TypeError
        If ``source`` is not a :class:`VideoSource` instance.
    ValueError
        If the represented interval is invalid.
    """

    source: VideoSource
    start_seconds: float = 0.0
    end_seconds: float | None = None
    _landmark_tracks: dict[str, FaceLandmarkTrack] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the represented source-media interval."""
        if not isinstance(self.source, VideoSource):
            raise TypeError(
                f"source must be a VideoSource instance, got {type(self.source).__name__}"
            )
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0.0:
            raise ValueError(
                f"start_seconds must be finite and non-negative, got {self.start_seconds}"
            )

        if self.end_seconds is not None:
            if not math.isfinite(self.end_seconds):
                raise ValueError(
                    f"end_seconds must be finite when provided, got {self.end_seconds}"
                )
            if self.end_seconds <= self.start_seconds:
                raise ValueError(
                    "end_seconds must be greater than start_seconds, "
                    f"got start={self.start_seconds}, end={self.end_seconds}"
                )

    @classmethod
    def from_video(cls, video_path: str | Path) -> "TalkingFaceSequence":
        """Create a sequence by inspecting a local video file.

        This convenience constructor delegates media access to the PyAV integration. It inspects
        stream metadata but does not decode video frames or audio samples.

        Parameters
        ----------
        video_path
            Path to a local media file.

        Returns
        -------
        TalkingFaceSequence
            Open-ended sequence initialized with a source for the first video stream.

        Raises
        ------
        FileNotFoundError
            If ``video_path`` does not exist.
        IsADirectoryError
            If ``video_path`` refers to a directory.
        ValueError
            If the path is not a regular file, the media cannot be inspected, or it has no video
            stream.
        """
        path = Path(video_path)
        metadata = inspect_video_metadata(path)
        return cls(
            source=VideoSource(path=path, metadata=metadata),
            start_seconds=0.0,
            end_seconds=None,
        )

    def clip(
        self,
        start_seconds: float,
        end_seconds: float | None = None,
    ) -> "TalkingFaceSequence":
        """Create a lightweight view of a smaller source-timeline interval.

        The new sequence reuses the same :class:`VideoSource` without decoding media. Timestamps
        remain relative to that source, and attached landmark tracks are not copied.

        Parameters
        ----------
        start_seconds
            Inclusive start in seconds on the source-media timeline.
        end_seconds
            Exclusive end in seconds, or ``None`` to use this sequence's current end.

        Returns
        -------
        TalkingFaceSequence
            New sequence representing the requested interval.

        Raises
        ------
        ValueError
            If the interval is invalid or extends outside this sequence.
        """
        resolved_end_seconds = self.end_seconds if end_seconds is None else end_seconds
        if start_seconds < self.start_seconds:
            raise ValueError(
                "clip start_seconds must not precede the current sequence start, "
                f"got {start_seconds} before {self.start_seconds}"
            )
        if self.end_seconds is not None and (
            resolved_end_seconds is None or resolved_end_seconds > self.end_seconds
        ):
            raise ValueError(
                "clip end_seconds must not exceed the current sequence end, "
                f"got {resolved_end_seconds} after {self.end_seconds}"
            )

        return TalkingFaceSequence(
            source=self.source,
            start_seconds=start_seconds,
            end_seconds=resolved_end_seconds,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Duration of this sequence's declared interval in seconds.

        This value is distinct from ``source.metadata.stream_duration_seconds``, which describes
        the complete source stream. An open-ended sequence has no declared duration and returns
        ``None`` even when its source reports an estimated duration.
        """
        if self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds

    @property
    def landmark_tracks(self) -> Mapping[str, FaceLandmarkTrack]:
        """Named landmark results currently attached to this sequence.

        The returned mapping is a read-only live view. Result records and their arrays must be
        treated as immutable.
        """
        return MappingProxyType(self._landmark_tracks)

    def track_landmarks(
        self,
        tracker: LandmarkTracker,
        *,
        name: str,
        overwrite: bool = False,
    ) -> FaceLandmarkTrack:
        """Run a landmark backend and attach its complete result under a name.

        Tracking is transactional at the sequence level: the backend computes the complete result
        before this object changes. If tracking raises, existing results remain untouched.

        Parameters
        ----------
        tracker
            Local backend implementing the landmark-tracker contract.
        name
            Non-empty result name. Different names allow backend or configuration comparisons.
        overwrite
            Whether an existing result with the same name may be replaced after successful
            tracking.

        Returns
        -------
        FaceLandmarkTrack
            The result that was attached.

        Raises
        ------
        ValueError
            If ``name`` is empty or already exists while ``overwrite`` is false.
        """
        if not name.strip():
            raise ValueError("landmark track name must not be empty")
        if name in self._landmark_tracks and not overwrite:
            raise ValueError(f"landmark track already exists: {name}")

        result = tracker.track(
            self.source.path,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        self._landmark_tracks[name] = result
        return result
