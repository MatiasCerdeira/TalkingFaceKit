"""Backend-independent metadata types."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata for a video's primary stream.

    Attributes
    ----------
    width
        Encoded frame width in pixels.
    height
        Encoded frame height in pixels.
    average_fps
        Average frame rate in frames per second, or ``None`` when it is unknown.
    stream_duration_seconds
        Duration in seconds, or ``None`` when it is unknown.
    has_audio
        Whether the source includes at least one audio stream.

    Raises
    ------
    TypeError
        If either dimension is not an integer or ``has_audio`` is not boolean.
    ValueError
        If either dimension is not positive, or if a provided frame rate or duration is not
        finite and positive.
    """

    width: int
    height: int
    average_fps: float | None
    stream_duration_seconds: float | None
    has_audio: bool

    def __post_init__(self) -> None:
        """Validate dimensions, frame rate, and duration."""
        for field_name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if not isinstance(self.has_audio, bool):
            raise TypeError(f"has_audio must be boolean, got {type(self.has_audio).__name__}")

        if self.average_fps is not None and (
            not math.isfinite(self.average_fps) or self.average_fps <= 0.0
        ):
            raise ValueError(
                f"average_fps must be finite and positive when provided, got {self.average_fps}"
            )

        if self.stream_duration_seconds is not None and (
            not math.isfinite(self.stream_duration_seconds) or self.stream_duration_seconds <= 0.0
        ):
            raise ValueError(
                "stream_duration_seconds must be finite and positive when provided, "
                f"got {self.stream_duration_seconds}"
            )
