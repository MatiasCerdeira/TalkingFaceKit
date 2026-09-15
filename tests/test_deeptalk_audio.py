from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from talkingfacekit import DecodedAudioChunk, TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.integrations import deeptalk

deeptalk_asd = pytest.importorskip("deeptalk_asd")


def make_sequence(
    *, start_seconds: float = 10.0, end_seconds: float | None = None
) -> TalkingFaceSequence:
    return TalkingFaceSequence(
        source=VideoSource(
            path=Path("source.mp4"),
            metadata=VideoMetadata(3, 2, 25.0, 30.0, True),
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def make_chunk(
    start_timestamp_seconds: float,
    samples: NDArray[np.float32],
    *,
    sample_rate_hz: int = 48_000,
) -> DecodedAudioChunk:
    return DecodedAudioChunk(
        start_sample_index=0,
        start_timestamp_seconds=start_timestamp_seconds,
        sample_rate_hz=sample_rate_hz,
        channel_layout="stereo",
        samples=np.ascontiguousarray(samples, dtype=np.float32),
    )


def install_audio_chunks(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[DecodedAudioChunk],
    *,
    calls: list[tuple[Path, float, float | None]] | None = None,
) -> None:
    def fake_stream_audio_chunks(
        media_path: str | Path,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> Iterator[DecodedAudioChunk]:
        if calls is not None:
            calls.append((Path(media_path), start_seconds, end_seconds))
        yield from chunks

    monkeypatch.setattr(deeptalk, "stream_audio_chunks", fake_stream_audio_chunks)


def adapt_audio(sequence: TalkingFaceSequence) -> list[tuple[float, object]]:
    return list(deeptalk._iter_deeptalk_audio_frames(sequence, deeptalk_asd))


def test_resamples_stereo_48khz_to_flushed_deeptalk_pcm_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.column_stack(
        (
            np.full(4_800, 0.25, dtype=np.float32),
            np.full(4_800, 0.75, dtype=np.float32),
        )
    )
    calls: list[tuple[Path, float, float | None]] = []
    install_audio_chunks(monkeypatch, [make_chunk(10.0, samples)], calls=calls)

    events = adapt_audio(make_sequence(end_seconds=10.1))

    assert calls == [(Path("source.mp4"), 10.0, 10.1)]
    assert [timestamp for timestamp, _ in events] == pytest.approx([0.03, 0.06, 0.09, 0.1])
    frames = [cast(Any, frame) for _, frame in events]
    assert [frame.samples_per_channel for frame in frames] == [480, 480, 480, 160]
    assert all(frame.sample_rate == 16_000 for frame in frames)
    assert all(frame.num_channels == 1 for frame in frames)
    pcm = np.concatenate([np.frombuffer(frame.data, dtype=np.int16) for frame in frames])
    assert pcm.dtype == np.dtype(np.int16)
    assert pcm == pytest.approx(np.full(1_600, 16_384, dtype=np.int16), abs=1)


def test_accepts_sub_millisecond_pts_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_chunk(10.0, np.zeros((648, 2), dtype=np.float32))
    second = make_chunk(
        10.014,
        np.zeros((672, 2), dtype=np.float32),
    )
    install_audio_chunks(monkeypatch, [first, second])

    events = adapt_audio(make_sequence())

    assert events


def test_rejects_a_significant_audio_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    first = make_chunk(10.0, np.zeros((1_440, 2), dtype=np.float32))
    second = make_chunk(
        10.04,
        np.zeros((1_440, 2), dtype=np.float32),
    )
    install_audio_chunks(monkeypatch, [first, second])

    with pytest.raises(ValueError, match="gap that DeepTalk cannot represent"):
        adapt_audio(make_sequence())


def test_trims_resampler_rounding_at_the_sequence_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_count = 100
    duration_seconds = sample_count / 48_000
    install_audio_chunks(
        monkeypatch,
        [make_chunk(10.0, np.zeros((sample_count, 2), dtype=np.float32))],
    )

    events = adapt_audio(make_sequence(end_seconds=10.0 + duration_seconds))

    assert len(events) == 1
    relative_end_seconds, frame = events[0]
    assert cast(Any, frame).samples_per_channel == 33
    assert relative_end_seconds <= duration_seconds


def test_rejects_missing_trailing_audio_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    install_audio_chunks(
        monkeypatch,
        [make_chunk(10.0, np.zeros((1_440, 2), dtype=np.float32))],
    )

    with pytest.raises(ValueError, match="trailing gap"):
        adapt_audio(make_sequence(end_seconds=10.1))
