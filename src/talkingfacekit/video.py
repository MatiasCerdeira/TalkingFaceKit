"""Backend-independent decoded video data."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

VideoFrameArray = NDArray[np.uint8]
TimestampArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class VideoData:
    """Decoded RGB video frames and their source timestamps.

    Parameters
    ----------
    frames
        RGB frames with shape ``(frame_count, height, width, 3)``, dtype ``uint8``, and values in
        the inclusive range ``[0, 255]``.
    timestamps_seconds
        One timestamp per frame with shape ``(frame_count,)`` and dtype ``float64``. Values are
        finite, strictly increasing seconds on the source media timeline.

    Notes
    -----
    Arrays are stored without copying to avoid duplicating large video buffers. Although the
    dataclass fields cannot be rebound, NumPy buffers remain mutable and callers should treat them
    as read-only after construction.

    Raises
    ------
    TypeError
        If either value is not a NumPy array.
    ValueError
        If either array violates its documented shape, dtype, size, or timestamp invariants.
    """

    frames: VideoFrameArray
    timestamps_seconds: TimestampArray

    def __post_init__(self) -> None:
        """Validate the decoded video data contract."""
        if not isinstance(self.frames, np.ndarray):
            raise TypeError(
                f"frames must be a NumPy array; observed {type(self.frames).__name__}"
            )
        if self.frames.dtype != np.dtype(np.uint8):
            raise ValueError(
                f"frames must have dtype uint8; observed {self.frames.dtype}"
            )
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(
                "frames must have shape (frame_count, height, width, 3); "
                f"observed {self.frames.shape}"
            )
        if self.frames.shape[0] == 0:
            raise ValueError(
                "frames must contain at least one frame; observed frame_count 0"
            )
        if self.frames.shape[1] == 0 or self.frames.shape[2] == 0:
            raise ValueError(
                f"frames must have positive height and width; observed shape {self.frames.shape}"
            )

        if not isinstance(self.timestamps_seconds, np.ndarray):
            raise TypeError(
                "timestamps_seconds must be a NumPy array; "
                f"observed {type(self.timestamps_seconds).__name__}"
            )
        if self.timestamps_seconds.dtype != np.dtype(np.float64):
            raise ValueError(
                "timestamps_seconds must have dtype float64; "
                f"observed {self.timestamps_seconds.dtype}"
            )
        if self.timestamps_seconds.ndim != 1:
            raise ValueError(
                "timestamps_seconds must have shape (frame_count,); "
                f"observed {self.timestamps_seconds.shape}"
            )
        if self.timestamps_seconds.shape[0] != self.frames.shape[0]:
            raise ValueError(
                "timestamps_seconds must contain one value per frame; "
                f"observed {self.timestamps_seconds.shape[0]} timestamps for "
                f"{self.frames.shape[0]} frames"
            )
        if not np.all(np.isfinite(self.timestamps_seconds)):
            invalid_count = int(np.count_nonzero(~np.isfinite(self.timestamps_seconds)))
            raise ValueError(
                f"timestamps_seconds must contain only finite values; observed {invalid_count} "
                "non-finite values"
            )

        timestamp_differences = np.diff(self.timestamps_seconds)
        invalid_order = np.flatnonzero(timestamp_differences <= 0)
        if invalid_order.size:
            first_index = int(invalid_order[0])
            raise ValueError(
                "timestamps_seconds must be strictly increasing; "
                f"observed {self.timestamps_seconds[first_index]} then "
                f"{self.timestamps_seconds[first_index + 1]} at indices "
                f"{first_index} and {first_index + 1}"
            )
