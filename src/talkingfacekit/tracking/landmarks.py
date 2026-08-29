"""Backend-independent facial-landmark data contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class FaceLandmarkTrack:
    """Facial landmarks aligned with decoded source-video frames.

    The arrays contain one row per decoded frame in temporal order. A failed detection remains in
    the timeline: its ``detected`` value is false and its complete landmark row is ``NaN``. Arrays
    are retained without copying; callers must not mutate them after construction.

    Attributes
    ----------
    tracker_name
        Stable human-readable identifier for the tracking implementation.
    tracker_version
        Installed tracker version, or ``None`` when the backend does not expose one.
    topology
        Identifier describing the landmark count and point ordering.
    coordinate_system
        Description of the meaning, units, and orientation of the three coordinates.
    frame_indices
        ``int64`` array shaped ``(frame_count,)``. Values are zero-based decoded frame indices in
        the source video and must be strictly increasing.
    timestamps_seconds
        ``float64`` array shaped ``(frame_count,)`` containing source presentation timestamps in
        seconds. Values must be finite and strictly increasing.
    landmarks
        ``float32`` array shaped ``(frame_count, landmark_count, 3)``. Coordinate ranges are
        defined by ``coordinate_system``. Missing rows contain only ``NaN``.
    detected
        Boolean array shaped ``(frame_count,)`` indicating whether a face was detected.
    """

    tracker_name: str
    tracker_version: str | None
    topology: str
    coordinate_system: str
    frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
    landmarks: NDArray[np.float32]
    detected: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate the landmark and timeline contracts."""
        for field_name, value in (
            ("tracker_name", self.tracker_name),
            ("topology", self.topology),
            ("coordinate_system", self.coordinate_system),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        if self.frame_indices.dtype != np.dtype(np.int64):
            raise TypeError(f"frame_indices must have dtype int64, got {self.frame_indices.dtype}")
        if self.timestamps_seconds.dtype != np.dtype(np.float64):
            raise TypeError(
                f"timestamps_seconds must have dtype float64, got {self.timestamps_seconds.dtype}"
            )
        if self.landmarks.dtype != np.dtype(np.float32):
            raise TypeError(f"landmarks must have dtype float32, got {self.landmarks.dtype}")
        if self.detected.dtype != np.dtype(np.bool_):
            raise TypeError(f"detected must have dtype bool, got {self.detected.dtype}")

        if self.frame_indices.ndim != 1:
            raise ValueError(
                f"frame_indices must have shape (frame_count,), got {self.frame_indices.shape}"
            )
        frame_count = self.frame_indices.shape[0]
        if frame_count == 0:
            raise ValueError("a landmark track must contain at least one frame")
        if self.timestamps_seconds.shape != (frame_count,):
            raise ValueError(
                "timestamps_seconds must have shape "
                f"({frame_count},), got {self.timestamps_seconds.shape}"
            )
        if self.detected.shape != (frame_count,):
            raise ValueError(
                f"detected must have shape ({frame_count},), got {self.detected.shape}"
            )
        if self.landmarks.ndim != 3 or self.landmarks.shape[0] != frame_count:
            raise ValueError(
                "landmarks must have shape (frame_count, landmark_count, 3), "
                f"got {self.landmarks.shape}"
            )
        if self.landmarks.shape[1] == 0 or self.landmarks.shape[2] != 3:
            raise ValueError(
                "landmarks must have shape (frame_count, landmark_count, 3) with at least one "
                f"landmark, got {self.landmarks.shape}"
            )

        if np.any(self.frame_indices < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if not np.all(np.isfinite(self.timestamps_seconds)):
            raise ValueError("timestamps_seconds must contain only finite values")
        if np.any(np.diff(self.timestamps_seconds) <= 0):
            raise ValueError("timestamps_seconds must be strictly increasing")

        if not np.all(np.isfinite(self.landmarks[self.detected])):
            raise ValueError("detected landmark rows must contain only finite values")
        if not np.all(np.isnan(self.landmarks[~self.detected])):
            raise ValueError("undetected landmark rows must contain only NaN values")

    @property
    def frame_count(self) -> int:
        """Number of decoded frames represented by this track."""
        return int(self.frame_indices.shape[0])

    @property
    def landmark_count(self) -> int:
        """Number of ordered landmark points per frame."""
        return int(self.landmarks.shape[1])


class LandmarkTracker(Protocol):
    """Minimal contract implemented by a local video-landmark backend."""

    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        """Track landmarks over a half-open interval of a local video."""
        ...
