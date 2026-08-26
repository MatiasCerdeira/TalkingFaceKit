"""Backend-independent decoded-video data contracts."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class DecodedVideoFrame:
    """One decoded RGB frame on the source-media timeline.

    The shared video integration yields these records one at a time. Streaming code does not
    retain previous frames, although consumers may explicitly keep arrays when their workflow
    requires it. The pixel buffer is stored without copying and must be treated as read-only.

    Parameters
    ----------
    frame_index
        Zero-based decode index in the source video stream. The index remains tied to the source,
        so the first frame in a requested interval may have a value greater than zero.
    timestamp_seconds
        Finite presentation timestamp in seconds on the source-media timeline.
    rgb
        RGB pixels with shape ``(height, width, 3)``, dtype ``uint8``, and values in the inclusive
        range ``[0, 255]``.

    Raises
    ------
    TypeError
        If the frame index is not an integer or the pixel value is not a NumPy array.
    ValueError
        If the frame index, timestamp, dtype, or pixel shape violates the contract.
    """

    frame_index: int
    timestamp_seconds: float
    rgb: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Validate the frame identity, timestamp, and RGB pixel contract."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError(
                f"frame_index must be an integer, got {type(self.frame_index).__name__}"
            )
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if not math.isfinite(self.timestamp_seconds):
            raise ValueError(f"timestamp_seconds must be finite, got {self.timestamp_seconds}")

        if not isinstance(self.rgb, np.ndarray):
            raise TypeError(f"rgb must be a NumPy array, got {type(self.rgb).__name__}")
        if self.rgb.dtype != np.dtype(np.uint8):
            raise ValueError(f"rgb must have dtype uint8, got {self.rgb.dtype}")
        if self.rgb.ndim != 3 or self.rgb.shape[2:] != (3,):
            raise ValueError(f"rgb must have shape (height, width, 3), got {self.rgb.shape}")
        if self.rgb.shape[0] == 0 or self.rgb.shape[1] == 0:
            raise ValueError(f"rgb must have positive height and width, got shape {self.rgb.shape}")
