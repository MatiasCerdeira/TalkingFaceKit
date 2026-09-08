import wave
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest

from talkingfacekit import DecodedAudioChunk, stream_audio_chunks

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def test_streams_sample_major_audio_over_a_half_open_interval() -> None:
    chunks = list(stream_audio_chunks(FIXTURE, start_seconds=0.015, end_seconds=0.035))

    assert [chunk.start_sample_index for chunk in chunks] == [696, 1608]
    assert [chunk.start_timestamp_seconds for chunk in chunks] == pytest.approx([0.015, 0.034])
    assert [chunk.end_timestamp_seconds for chunk in chunks] == pytest.approx([0.034, 0.035])
    assert [chunk.samples.shape for chunk in chunks] == [(912, 2), (48, 2)]
    assert all(chunk.sample_rate_hz == 48_000 for chunk in chunks)
    assert all(chunk.channel_layout == "stereo" for chunk in chunks)
    assert all(chunk.samples.dtype == np.dtype(np.float32) for chunk in chunks)
    assert all(chunk.samples.flags.c_contiguous for chunk in chunks)


def test_normalizes_integer_pcm_without_resampling_or_remixing(tmp_path: Path) -> None:
    audio_path = tmp_path / "integer-pcm.wav"
    input_samples = np.array([-32768, 0, 32767], dtype="<i2")
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(input_samples.tobytes())

    chunks = list(stream_audio_chunks(audio_path))

    assert len(chunks) == 1
    assert chunks[0].sample_rate_hz == 8_000
    assert chunks[0].channel_count == 1
    assert chunks[0].channel_layout == "1 channels"
    assert chunks[0].samples[:, 0] == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_validates_path_and_interval_before_decoding(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Media not found"):
        stream_audio_chunks(tmp_path / "missing.webm")

    with pytest.raises(ValueError, match="end_seconds must be greater"):
        stream_audio_chunks(FIXTURE, start_seconds=0.1, end_seconds=0.1)


def test_rejects_an_interval_without_samples() -> None:
    chunks = stream_audio_chunks(FIXTURE, start_seconds=10.0, end_seconds=11.0)

    with pytest.raises(ValueError, match="No decoded audio samples fall within"):
        list(chunks)


def test_rejects_media_without_an_audio_stream(tmp_path: Path) -> None:
    video_path = tmp_path / "video-only.mkv"
    with av.open(video_path, mode="w") as container:
        stream = cast(av.VideoStream, container.add_stream("ffv1", rate=25))
        stream.width = 2
        stream.height = 2
        frame = av.VideoFrame.from_ndarray(np.zeros((2, 2, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    chunks = stream_audio_chunks(video_path)

    with pytest.raises(ValueError, match="Media file has no audio stream"):
        list(chunks)


def test_decoded_audio_chunk_validates_shape_dtype_and_finite_values() -> None:
    with pytest.raises(ValueError, match="dtype float32"):
        DecodedAudioChunk(
            start_sample_index=0,
            start_timestamp_seconds=0.0,
            sample_rate_hz=16_000,
            channel_layout="mono",
            samples=np.zeros((4, 1), dtype=np.float64),
        )

    with pytest.raises(ValueError, match="only finite"):
        DecodedAudioChunk(
            start_sample_index=0,
            start_timestamp_seconds=0.0,
            sample_rate_hz=16_000,
            channel_layout="mono",
            samples=np.array([[np.nan]], dtype=np.float32),
        )


def test_decoded_audio_chunk_exposes_sample_and_channel_counts() -> None:
    chunk = DecodedAudioChunk(
        start_sample_index=4,
        start_timestamp_seconds=0.25,
        sample_rate_hz=8,
        channel_layout="stereo",
        samples=np.zeros((4, 2), dtype=np.float32),
    )

    assert chunk.sample_count == 4
    assert chunk.channel_count == 2
    assert chunk.end_timestamp_seconds == pytest.approx(0.75)
