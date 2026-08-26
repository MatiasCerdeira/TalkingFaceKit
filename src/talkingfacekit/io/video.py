"""Video inspection and streaming through the PyAV integration boundary."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.video import DecodedVideoFrame


def inspect_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Inspect a local media file and return metadata for its primary video stream.

    The first video stream supplies the dimensions and average frame rate. Its duration is used
    when available; otherwise, the container duration is used as a fallback. No frames or audio
    samples are decoded.

    Parameters
    ----------
    video_path
        Path to a local media file.

    Returns
    -------
    VideoMetadata
        Backend-independent metadata for the first video stream.

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
    path = _validate_video_path(video_path)

    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"Media file has no video stream: {path}")

            video_stream = container.streams.video[0]
            duration_seconds = (
                float(video_stream.duration * video_stream.time_base)
                if video_stream.duration is not None and video_stream.time_base is not None
                else (
                    float(container.duration / av.time_base)
                    if container.duration is not None
                    else None
                )
            )
            return VideoMetadata(
                width=video_stream.width,
                height=video_stream.height,
                average_fps=(
                    float(video_stream.average_rate)
                    if video_stream.average_rate is not None
                    else None
                ),
                stream_duration_seconds=duration_seconds,
                has_audio=bool(container.streams.audio),
            )
    except av.FFmpegError as error:
        raise ValueError(f"Could not inspect media file {path}: {error}") from error


def stream_video_frames(
    video_path: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Stream decoded RGB frames from the first video stream.

    PyAV owns media decoding at this integration boundary. Frames are yielded one at a time over
    the half-open interval ``[start_seconds, end_seconds)`` without retaining earlier pixel
    buffers. Source decode indices and presentation timestamps are preserved; FPS is never used to
    synthesize time. The caller controls whether yielded arrays are retained.

    Parameters
    ----------
    video_path
        Path to a local media file.
    start_seconds
        Inclusive interval start in seconds. Must be finite and non-negative.
    end_seconds
        Exclusive interval end in seconds, or ``None`` to continue to end-of-stream.

    Returns
    -------
    collections.abc.Iterator[DecodedVideoFrame]
        Lazily decoded frames in strictly increasing presentation-time order. Pixels use RGB
        channel order, shape ``(height, width, 3)``, and dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    IsADirectoryError
        If ``video_path`` refers to a directory.
    ValueError
        If the path or interval is invalid, the media cannot be decoded, no video stream exists,
        timestamps are absent or unordered, or no frames fall inside the interval.

    Notes
    -----
    Path and interval validation happen when this function is called. Media opening and frame
    decoding begin when the returned iterator is consumed.
    """
    path = _validate_video_path(video_path)
    _validate_interval(start_seconds, end_seconds)
    return _stream_video_frames(path, start_seconds=start_seconds, end_seconds=end_seconds)


def _stream_video_frames(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float | None,
) -> Iterator[DecodedVideoFrame]:
    yielded_frame = False
    previous_timestamp_seconds: float | None = None

    try:
        with av.open(video_path) as container:
            if not container.streams.video:
                raise ValueError(f"Media file has no video stream: {video_path}")
            video_stream = container.streams.video[0]

            for frame_index, frame in enumerate(container.decode(video_stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(
                        f"Decoded frame {frame_index} has no presentation timestamp or time base"
                    )
                timestamp_seconds = float(frame.pts * frame.time_base)
                if not math.isfinite(timestamp_seconds):
                    raise ValueError(
                        f"Decoded frame {frame_index} has non-finite timestamp {timestamp_seconds}"
                    )
                if timestamp_seconds < start_seconds:
                    continue
                if end_seconds is not None and timestamp_seconds >= end_seconds:
                    break
                if (
                    previous_timestamp_seconds is not None
                    and timestamp_seconds <= previous_timestamp_seconds
                ):
                    raise ValueError(
                        "Decoded frame timestamps must be strictly increasing; "
                        f"frame {frame_index} produced {timestamp_seconds} after "
                        f"{previous_timestamp_seconds}"
                    )

                rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                decoded_frame = DecodedVideoFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    rgb=rgb,
                )
                yielded_frame = True
                previous_timestamp_seconds = timestamp_seconds
                yield decoded_frame
    except av.FFmpegError as error:
        raise ValueError(f"Could not decode media file {video_path}: {error}") from error

    if not yielded_frame:
        raise ValueError(
            f"No decoded video frames fall within [{start_seconds}, {end_seconds}) seconds"
        )


def _validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Video path is a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a regular file: {path}")
    return path


def _validate_interval(start_seconds: float, end_seconds: float | None) -> None:
    if not math.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError(f"start_seconds must be finite and non-negative, got {start_seconds}")
    if end_seconds is not None:
        if not math.isfinite(end_seconds):
            raise ValueError(f"end_seconds must be finite when provided, got {end_seconds}")
        if end_seconds <= start_seconds:
            raise ValueError(
                "end_seconds must be greater than start_seconds, "
                f"got start={start_seconds}, end={end_seconds}"
            )
