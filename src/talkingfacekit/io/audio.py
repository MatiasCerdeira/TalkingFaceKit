"""Audio streaming through the PyAV integration boundary."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from talkingfacekit.audio import DecodedAudioChunk

_SAMPLE_OFFSET_TOLERANCE = 1e-7


def stream_audio_chunks(
    media_path: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Iterator[DecodedAudioChunk]:
    """Stream decoded audio chunks from the first audio stream.

    PyAV owns decoding at this integration boundary. Chunks are yielded over the half-open source
    interval ``[start_seconds, end_seconds)`` without retaining earlier sample buffers. Chunks that
    cross an interval boundary are trimmed at sample precision. The function preserves sample rate,
    channel count, channel layout, decode-order sample indices, and source presentation timestamps.
    It converts decoded PCM to contiguous sample-major ``float32`` without resampling, remixing, or
    clipping.

    Parameters
    ----------
    media_path
        Path to a local media file.
    start_seconds
        Inclusive interval start in seconds. Must be finite and non-negative.
    end_seconds
        Exclusive interval end in seconds, or ``None`` to continue to end-of-stream.

    Returns
    -------
    collections.abc.Iterator[DecodedAudioChunk]
        Lazily decoded chunks with shape ``(sample_count, channel_count)`` and dtype ``float32``.

    Raises
    ------
    FileNotFoundError
        If ``media_path`` does not exist.
    IsADirectoryError
        If ``media_path`` refers to a directory.
    ValueError
        If the path or interval is invalid, the media cannot be decoded, no audio stream exists,
        audio metadata or timestamps are invalid, decoded samples violate the contract, or no
        samples fall inside the interval.

    Notes
    -----
    Path and interval validation happen when this function is called. Media opening and decoding
    begin when the returned iterator is consumed.
    """
    path = _validate_media_path(media_path)
    _validate_interval(start_seconds, end_seconds)
    return _stream_audio_chunks(path, start_seconds=start_seconds, end_seconds=end_seconds)


def _stream_audio_chunks(
    media_path: Path,
    *,
    start_seconds: float,
    end_seconds: float | None,
) -> Iterator[DecodedAudioChunk]:
    yielded_chunk = False
    decoded_sample_index = 0
    previous_timestamp_seconds: float | None = None

    try:
        with av.open(media_path) as container:
            if not container.streams.audio:
                raise ValueError(f"Media file has no audio stream: {media_path}")
            audio_stream = container.streams.audio[0]

            for frame_index, frame in enumerate(container.decode(audio_stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(
                        f"Decoded audio frame {frame_index} has no presentation timestamp or time base"
                    )
                timestamp_seconds = float(frame.pts * frame.time_base)
                if not math.isfinite(timestamp_seconds):
                    raise ValueError(
                        "Decoded audio frame "
                        f"{frame_index} has non-finite timestamp {timestamp_seconds}"
                    )

                sample_rate_hz = frame.sample_rate
                if sample_rate_hz is None or sample_rate_hz <= 0:
                    raise ValueError(
                        f"Decoded audio frame {frame_index} has invalid sample rate {sample_rate_hz}"
                    )
                sample_count = frame.samples
                if sample_count <= 0:
                    raise ValueError(
                        f"Decoded audio frame {frame_index} has invalid sample count {sample_count}"
                    )

                frame_start_sample_index = decoded_sample_index
                decoded_sample_index += sample_count
                frame_end_seconds = timestamp_seconds + sample_count / sample_rate_hz

                if end_seconds is not None and timestamp_seconds >= end_seconds:
                    break
                if frame_end_seconds <= start_seconds:
                    continue

                first_sample_offset = max(
                    0,
                    _first_sample_at_or_after(
                        start_seconds,
                        frame_timestamp_seconds=timestamp_seconds,
                        sample_rate_hz=sample_rate_hz,
                    ),
                )
                final_sample_offset = sample_count
                if end_seconds is not None:
                    final_sample_offset = min(
                        sample_count,
                        _first_sample_at_or_after(
                            end_seconds,
                            frame_timestamp_seconds=timestamp_seconds,
                            sample_rate_hz=sample_rate_hz,
                        ),
                    )

                if first_sample_offset >= final_sample_offset:
                    continue

                samples = _audio_frame_to_float32(frame, frame_index=frame_index)
                selected_samples = np.ascontiguousarray(
                    samples[first_sample_offset:final_sample_offset],
                    dtype=np.float32,
                )
                selected_timestamp_seconds = (
                    timestamp_seconds + first_sample_offset / sample_rate_hz
                )
                if (
                    previous_timestamp_seconds is not None
                    and selected_timestamp_seconds <= previous_timestamp_seconds
                ):
                    raise ValueError(
                        "Decoded audio chunk timestamps must be strictly increasing; "
                        f"frame {frame_index} produced {selected_timestamp_seconds} after "
                        f"{previous_timestamp_seconds}"
                    )

                chunk = DecodedAudioChunk(
                    start_sample_index=frame_start_sample_index + first_sample_offset,
                    start_timestamp_seconds=selected_timestamp_seconds,
                    sample_rate_hz=sample_rate_hz,
                    channel_layout=frame.layout.name,
                    samples=selected_samples,
                )
                yielded_chunk = True
                previous_timestamp_seconds = selected_timestamp_seconds
                yield chunk

                if end_seconds is not None and frame_end_seconds >= end_seconds:
                    break
    except av.FFmpegError as error:
        raise ValueError(f"Could not decode media file {media_path}: {error}") from error

    if not yielded_chunk:
        raise ValueError(
            f"No decoded audio samples fall within [{start_seconds}, {end_seconds}) seconds"
        )


def _first_sample_at_or_after(
    target_seconds: float,
    *,
    frame_timestamp_seconds: float,
    sample_rate_hz: int,
) -> int:
    relative_sample_position = (target_seconds - frame_timestamp_seconds) * sample_rate_hz
    return math.ceil(relative_sample_position - _SAMPLE_OFFSET_TOLERANCE)


def _audio_frame_to_float32(
    frame: av.AudioFrame,
    *,
    frame_index: int,
) -> NDArray[np.float32]:
    raw_samples = frame.to_ndarray()
    channel_count = len(frame.layout.channels)
    expected_value_count = frame.samples * channel_count
    if channel_count <= 0 or raw_samples.size != expected_value_count:
        raise ValueError(
            f"Decoded audio frame {frame_index} has incompatible sample layout: "
            f"samples={frame.samples}, channels={channel_count}, array_shape={raw_samples.shape}"
        )

    if frame.format.is_planar:
        expected_shape = (channel_count, frame.samples)
        if raw_samples.shape != expected_shape:
            raise ValueError(
                f"Decoded planar audio frame {frame_index} must have shape {expected_shape}, "
                f"got {raw_samples.shape}"
            )
        sample_major = raw_samples.transpose()
    else:
        sample_major = raw_samples.reshape(frame.samples, channel_count)

    if np.issubdtype(sample_major.dtype, np.floating):
        normalized = sample_major.astype(np.float32, copy=False)
    elif np.issubdtype(sample_major.dtype, np.signedinteger):
        full_scale = float(1 << (sample_major.dtype.itemsize * 8 - 1))
        normalized = sample_major.astype(np.float32) / full_scale
    elif np.issubdtype(sample_major.dtype, np.unsignedinteger):
        midpoint = float(1 << (sample_major.dtype.itemsize * 8 - 1))
        normalized = (sample_major.astype(np.float32) - midpoint) / midpoint
    else:
        raise ValueError(
            f"Decoded audio frame {frame_index} has unsupported dtype {sample_major.dtype}"
        )

    return np.ascontiguousarray(normalized, dtype=np.float32)


def _validate_media_path(media_path: str | Path) -> Path:
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Media path is a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Media path is not a regular file: {path}")
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
