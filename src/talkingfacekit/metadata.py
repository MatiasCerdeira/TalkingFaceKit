"""Backend-independent metadata types."""

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
    """

    width: int
    height: int
    average_fps: float | None
    stream_duration_seconds: float | None
    has_audio: bool
