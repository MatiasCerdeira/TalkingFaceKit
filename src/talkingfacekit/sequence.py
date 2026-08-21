"""Core types and the user-facing talking-face sequence aggregate."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from talkingfacekit.io.video import inspect_video_metadata
from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack, LandmarkTracker


@dataclass(slots=True, eq=False)
class TalkingFaceSequence:
    """Collect the data and operations associated with a talking-face sequence.

    Direct construction performs no filesystem access. Use :meth:`from_video` to create a sequence
    from a local media file through the default video integration.

    Attributes
    ----------
    path
        Path identifying the sequence's source media. The path does not need to exist when the
        model is constructed.
    metadata
        Metadata for the primary video stream.
    start_seconds
        Start of the represented interval in seconds.
    end_seconds
        End of the represented interval in seconds, or ``None`` when it is unknown or unbounded.
    """

    path: Path
    metadata: VideoMetadata
    start_seconds: float = 0.0
    end_seconds: float | None = None
    _landmark_tracks: dict[str, FaceLandmarkTrack] = field(
        default_factory=dict, init=False, repr=False
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
            Sequence initialized with metadata for the first video stream.

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
            path=path,
            metadata=metadata,
            start_seconds=0.0,
            end_seconds=metadata.stream_duration_seconds,
        )

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
            self.path,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        self._landmark_tracks[name] = result
        return result
