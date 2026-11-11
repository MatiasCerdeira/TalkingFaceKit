"""Physical MP4 clip export through the PyAV integration boundary."""

from __future__ import annotations

import os
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import numpy as np
from av.container import OutputContainer

from talkingfacekit.audio import DecodedAudioChunk
from talkingfacekit.io.audio import _EmptyAudioIntervalError, stream_audio_chunks
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.video import DecodedVideoFrame, VideoSource

_VIDEO_TIME_BASE = Fraction(1, 90_000)


def export_clip(
    source: VideoSource,
    output_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    overwrite: bool = False,
) -> Path:
    """Decode and re-encode one bounded source interval as an MP4 file.

    Video and optional audio use the sequence start as their shared output-time origin. Video frame
    timestamps and audio samples are selected from the half-open source interval
    ``[start_seconds, end_seconds)``. The first source video and audio streams are encoded as H.264
    and AAC while preserving dimensions, sample aspect ratio, display rotation, average frame rate,
    sample rate, and channel layout when those values are available. Odd widths or heights gain one
    edge-replicated column or row on the right or bottom because ``yuv420p`` requires even
    dimensions; source pixel coordinates are unchanged. An interval without audio samples produces
    a video-only MP4.

    Parameters
    ----------
    source
        Source media identity and inspected primary-stream metadata.
    output_path
        Destination ending in ``.mp4``. Parent directories are created as needed.
    start_seconds, end_seconds
        Bounded half-open interval on the original source timeline, in seconds.
    overwrite
        Whether an existing destination may be atomically replaced after encoding succeeds.

    Returns
    -------
    pathlib.Path
        Completed MP4 path.

    Raises
    ------
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    IsADirectoryError
        If the destination refers to a directory.
    ValueError
        If the destination is not an MP4 path, aliases the source, or media cannot be decoded or
        encoded with valid timestamps and consistent stream characteristics.
    """
    path = _validate_output_path(source, output_path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        _encode_clip(source, temporary_path, start_seconds=start_seconds, end_seconds=end_seconds)
        if overwrite:
            temporary_path.replace(path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def _validate_output_path(source: VideoSource, output_path: str | Path, *, overwrite: bool) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("clip output must end in .mp4")
    if path.is_dir():
        raise IsADirectoryError(f"clip output is a directory: {path}")
    if path.resolve() == source.path.resolve():
        raise ValueError("clip output must differ from source media")
    if path.exists() and not overwrite:
        raise FileExistsError(f"clip output already exists: {path}")
    return path


def _encode_clip(
    source: VideoSource,
    output_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> None:
    video_frames = stream_video_frames(
        source.path, start_seconds=start_seconds, end_seconds=end_seconds
    )
    try:
        with av.open(source.path) as source_container:
            sample_aspect_ratio = source_container.streams.video[0].sample_aspect_ratio
            # Display rotation is stream-level side data that PyAV exposes on decoded frames.
            first_source_frame = next(source_container.decode(video=0), None)
            display_rotation = 0 if first_source_frame is None else first_source_frame.rotation
    except av.FFmpegError as error:
        raise ValueError(f"Could not inspect media file {source.path}: {error}") from error
    first_video_frame = next(video_frames)
    audio_chunks: Iterator[DecodedAudioChunk] | None = None
    first_audio_chunk: DecodedAudioChunk | None = None
    if source.metadata.has_audio:
        audio_chunks = stream_audio_chunks(
            source.path, start_seconds=start_seconds, end_seconds=end_seconds
        )
        try:
            first_audio_chunk = next(audio_chunks)
        except _EmptyAudioIntervalError:
            audio_chunks = None

    try:
        with av.open(
            output_path,
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        ) as container:
            video_stream = _add_video_stream(
                container,
                source,
                first_video_frame,
                sample_aspect_ratio=sample_aspect_ratio,
                display_rotation=display_rotation,
            )
            audio_stream = (
                _add_audio_stream(container, first_audio_chunk)
                if first_audio_chunk is not None
                else None
            )
            _encode_streams(
                container,
                video_stream,
                audio_stream,
                first_video_frame,
                video_frames,
                first_audio_chunk,
                audio_chunks,
                start_seconds=start_seconds,
            )
            for packet in video_stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                for packet in audio_stream.encode():
                    container.mux(packet)
    except av.FFmpegError as error:
        raise ValueError(f"Could not encode MP4 clip from {source.path}: {error}") from error


def _add_video_stream(
    container: OutputContainer,
    source: VideoSource,
    first_frame: DecodedVideoFrame,
    *,
    sample_aspect_ratio: Fraction | None,
    display_rotation: int,
) -> av.VideoStream:
    rate = (
        Fraction(source.metadata.average_fps).limit_denominator(100_000)
        if source.metadata.average_fps is not None
        else None
    )
    stream = container.add_stream("libx264", rate=rate)
    height, width = first_frame.rgb.shape[:2]
    # yuv420p requires even dimensions; odd sources gain one replicated row/column.
    stream.width = width + width % 2
    stream.height = height + height % 2
    stream.pix_fmt = "yuv420p"
    if sample_aspect_ratio is not None:
        stream.sample_aspect_ratio = sample_aspect_ratio
    if display_rotation:
        stream.set_display_rotation(display_rotation)
    stream.time_base = _VIDEO_TIME_BASE
    stream.codec_context.time_base = _VIDEO_TIME_BASE
    stream.codec_context.max_b_frames = 0
    return stream


def _add_audio_stream(container: OutputContainer, first_chunk: DecodedAudioChunk) -> av.AudioStream:
    stream = container.add_stream("aac", rate=first_chunk.sample_rate_hz)
    channel_layout = first_chunk.channel_layout
    if channel_layout == f"{first_chunk.channel_count} channels":
        channel_layout = {1: "mono", 2: "stereo"}.get(first_chunk.channel_count, channel_layout)
    stream.codec_context.layout = channel_layout
    time_base = Fraction(1, first_chunk.sample_rate_hz)
    stream.time_base = time_base
    stream.codec_context.time_base = time_base
    return stream


def _encode_streams(
    container: OutputContainer,
    video_stream: av.VideoStream,
    audio_stream: av.AudioStream | None,
    first_video_frame: DecodedVideoFrame,
    video_frames: Iterator[DecodedVideoFrame],
    first_audio_chunk: DecodedAudioChunk | None,
    audio_chunks: Iterator[DecodedAudioChunk] | None,
    *,
    start_seconds: float,
) -> None:
    next_video: DecodedVideoFrame | None = first_video_frame
    next_audio = first_audio_chunk
    previous_video_pts: int | None = None

    while next_video is not None or next_audio is not None:
        if next_audio is None or (
            next_video is not None
            and next_video.timestamp_seconds <= next_audio.start_timestamp_seconds
        ):
            assert next_video is not None
            previous_video_pts = _encode_video_frame(
                container,
                video_stream,
                next_video,
                start_seconds=start_seconds,
                previous_pts=previous_video_pts,
                source_dimensions=(
                    first_video_frame.rgb.shape[0],
                    first_video_frame.rgb.shape[1],
                ),
            )
            next_video = next(video_frames, None)
        else:
            assert audio_stream is not None
            _encode_audio_chunk(
                container,
                audio_stream,
                next_audio,
                first_audio_chunk,
                start_seconds=start_seconds,
            )
            assert audio_chunks is not None
            next_audio = next(audio_chunks, None)


def _encode_video_frame(
    container: OutputContainer,
    stream: av.VideoStream,
    frame: DecodedVideoFrame,
    *,
    start_seconds: float,
    previous_pts: int | None,
    source_dimensions: tuple[int, int],
) -> int:
    height, width = frame.rgb.shape[:2]
    if (height, width) != source_dimensions:
        raise ValueError(
            "decoded video dimensions changed during clip export: "
            f"expected {source_dimensions}, got {(height, width)}"
        )
    pixels = np.pad(frame.rgb, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    output_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    output_frame.time_base = _VIDEO_TIME_BASE
    output_frame.pts = round((frame.timestamp_seconds - start_seconds) / float(_VIDEO_TIME_BASE))
    if output_frame.pts < 0 or (previous_pts is not None and output_frame.pts <= previous_pts):
        raise ValueError("video timestamps cannot be represented as strictly increasing MP4 PTS")
    for packet in stream.encode(output_frame):
        container.mux(packet)
    return output_frame.pts


def _encode_audio_chunk(
    container: OutputContainer,
    stream: av.AudioStream,
    chunk: DecodedAudioChunk,
    first_chunk: DecodedAudioChunk | None,
    *,
    start_seconds: float,
) -> None:
    assert first_chunk is not None
    if (chunk.sample_rate_hz, chunk.channel_layout, chunk.channel_count) != (
        first_chunk.sample_rate_hz,
        first_chunk.channel_layout,
        first_chunk.channel_count,
    ):
        raise ValueError("decoded audio characteristics changed during clip export")
    output_frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(chunk.samples.T),
        format="fltp",
        layout=stream.codec_context.layout.name,
    )
    output_frame.sample_rate = chunk.sample_rate_hz
    output_frame.time_base = Fraction(1, chunk.sample_rate_hz)
    output_frame.pts = round((chunk.start_timestamp_seconds - start_seconds) * chunk.sample_rate_hz)
    if output_frame.pts < 0:
        raise ValueError("audio timestamp cannot be represented as a non-negative MP4 PTS")
    for packet in stream.encode(output_frame):
        container.mux(packet)
