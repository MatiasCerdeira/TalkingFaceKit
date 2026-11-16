from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from numpy.typing import NDArray

import talkingfacekit.io.clip as clip_io
from talkingfacekit import TalkingFaceSequence, stream_audio_chunks, stream_video_frames

_FRAME_RATE = 10
_SAMPLE_RATE = 48_000


def _write_synthetic_source(
    path: Path,
    *,
    with_audio: bool,
    width: int = 32,
    height: int = 24,
    video_codec: str = "ffv1",
    audio_codec: str = "flac",
    audio_layout: str = "mono",
    audio_duration_seconds: float = 0.6,
    sample_aspect_ratio: Fraction | None = None,
) -> None:
    """Write a tiny source with visible frame IDs and a timed tone onset."""
    with av.open(path, mode="w", format="matroska") as container:
        video_stream = cast(av.VideoStream, container.add_stream(video_codec, rate=_FRAME_RATE))
        video_stream.width = width
        video_stream.height = height
        if sample_aspect_ratio is not None:
            video_stream.sample_aspect_ratio = sample_aspect_ratio

        audio_stream: av.AudioStream | None = None
        if with_audio:
            audio_stream = cast(
                av.AudioStream, container.add_stream(audio_codec, rate=_SAMPLE_RATE)
            )
            audio_stream.codec_context.layout = audio_layout

        for index in range(6):
            pixels = np.full((height, width, 3), 20 + index * 35, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, _FRAME_RATE)
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        if audio_stream is not None:
            sample_count = round(audio_duration_seconds * _SAMPLE_RATE)
            times = np.arange(sample_count, dtype=np.float64) / _SAMPLE_RATE
            signal = np.zeros(sample_count, dtype=np.float64)
            tone = times >= 0.25
            signal[tone] = 0.6 * np.sin(2 * np.pi * 1_000 * times[tone])
            samples = np.round(signal * np.iinfo(np.int16).max).astype(np.int16)
            channel_count = 1 if audio_layout == "mono" else 2
            interleaved_samples = np.repeat(samples[:, np.newaxis], channel_count, axis=1)
            audio_frame = av.AudioFrame.from_ndarray(
                interleaved_samples.reshape(1, -1), format="s16", layout=audio_layout
            )
            audio_frame.sample_rate = _SAMPLE_RATE
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, _SAMPLE_RATE)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def _decoded_audio(path: Path) -> tuple[NDArray[np.float32], float, float]:
    chunks = list(stream_audio_chunks(path))
    samples = np.concatenate([chunk.samples[:, 0] for chunk in chunks])
    return samples, chunks[0].start_timestamp_seconds, chunks[-1].end_timestamp_seconds


def test_exports_the_half_open_interval_with_one_audio_video_origin(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "new-directory" / "clip.mp4"
    _write_synthetic_source(source, with_audio=True)

    result = TalkingFaceSequence.from_video(source).clip(0.15, 0.45).export_clip(output)

    assert result == output
    frames = list(stream_video_frames(output))
    assert [round(float(frame.rgb.mean())) for frame in frames] == pytest.approx(
        [90, 125, 160], abs=4
    )
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx(
        [0.05, 0.15, 0.25], abs=1 / 90_000
    )

    samples, audio_start, audio_end = _decoded_audio(output)
    assert audio_start == pytest.approx(0.0, abs=1 / _SAMPLE_RATE)
    assert audio_end == pytest.approx(0.3, abs=0.025)

    window_size = _SAMPLE_RATE // 100
    window_rms = np.asarray(
        [
            np.sqrt(np.mean(samples[index : index + window_size] ** 2))
            for index in range(0, len(samples) - window_size + 1, window_size)
        ]
    )
    onset_seconds = int(np.flatnonzero(window_rms > 0.1)[0]) * window_size / _SAMPLE_RATE
    assert onset_seconds == pytest.approx(0.1, abs=0.025)


def test_exports_valid_monotonic_mp4_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "clip.mp4"
    _write_synthetic_source(source, with_audio=True)

    TalkingFaceSequence.from_video(source).clip(0.15, 0.45).export_clip(output)

    with av.open(output) as container:
        assert container.format.name == "mov,mp4,m4a,3gp,3g2,mj2"
        assert container.duration is not None
        # The last selected 10 Hz frame is displayed for its nominal frame duration. Because the
        # first selected source frame begins 50 ms after the requested start, video ends at 350 ms
        # while sample-precise audio ends at the requested 300 ms boundary.
        assert container.duration / av.time_base == pytest.approx(0.35, abs=1 / _FRAME_RATE)
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1

    for stream_kind in ("video", "audio"):
        with av.open(output) as container:
            stream = (
                container.streams.video[0] if stream_kind == "video" else container.streams.audio[0]
            )
            timestamps = []
            for frame in container.decode(stream):
                if (
                    isinstance(frame, av.VideoFrame | av.AudioFrame)
                    and frame.pts is not None
                    and frame.time_base is not None
                ):
                    timestamps.append(float(frame.pts * frame.time_base))
            assert timestamps
            assert timestamps == sorted(timestamps)
            assert timestamps[0] >= 0.0


def test_exports_video_without_audio(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mkv"
    output = tmp_path / "video-only.mp4"
    _write_synthetic_source(source, with_audio=False)

    TalkingFaceSequence.from_video(source).clip(0.1, 0.4).export_clip(output)

    with av.open(output) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 0
    assert len(list(stream_video_frames(output))) == 3


@pytest.mark.parametrize(
    ("source_layout", "expected_layout", "channel_count"),
    (("mono", "mono", 1), ("stereo", "stereo", 2)),
)
def test_normalizes_unpositioned_pcm_layouts_for_aac(
    tmp_path: Path, source_layout: str, expected_layout: str, channel_count: int
) -> None:
    source = tmp_path / f"pcm-{source_layout}.mkv"
    output = tmp_path / f"pcm-{source_layout}.mp4"
    _write_synthetic_source(
        source,
        with_audio=True,
        audio_codec="pcm_s16le",
        audio_layout=source_layout,
    )
    source_chunks = list(stream_audio_chunks(source, end_seconds=0.3))
    assert source_chunks[0].channel_layout == f"{channel_count} channels"

    TalkingFaceSequence.from_video(source).clip(0.0, 0.3).export_clip(output)

    with av.open(output) as container:
        assert len(container.streams.audio) == 1
        audio_stream = container.streams.audio[0]
        assert audio_stream.codec_context.name == "aac"
        assert audio_stream.codec_context.layout.name == expected_layout
        assert len(audio_stream.codec_context.layout.channels) == channel_count


def test_exports_video_only_when_audio_has_no_samples_in_the_interval(tmp_path: Path) -> None:
    source = tmp_path / "short-audio.mkv"
    output = tmp_path / "video-only-interval.mp4"
    _write_synthetic_source(source, with_audio=True, audio_duration_seconds=0.1)

    TalkingFaceSequence.from_video(source).clip(0.3, 0.5).export_clip(output)

    with av.open(output) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 0
    assert [frame.timestamp_seconds for frame in stream_video_frames(output)] == pytest.approx(
        [0.0, 0.1], abs=1 / 90_000
    )


def test_preserves_source_sample_aspect_ratio(tmp_path: Path) -> None:
    source = tmp_path / "anamorphic.mkv"
    output = tmp_path / "anamorphic.mp4"
    sample_aspect_ratio = Fraction(2, 1)
    _write_synthetic_source(
        source,
        with_audio=False,
        video_codec="mpeg4",
        sample_aspect_ratio=sample_aspect_ratio,
    )

    with av.open(source) as container:
        assert container.streams.video[0].sample_aspect_ratio == sample_aspect_ratio

    TalkingFaceSequence.from_video(source).clip(0.0, 0.3).export_clip(output)

    with av.open(output) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (32, 24)
        assert stream.sample_aspect_ratio == sample_aspect_ratio


def test_pads_odd_dimensions_without_moving_source_pixels(tmp_path: Path) -> None:
    source = tmp_path / "odd.mkv"
    output = tmp_path / "odd.mp4"
    _write_synthetic_source(source, with_audio=False, width=33, height=25)

    TalkingFaceSequence.from_video(source).clip(0.0, 0.3).export_clip(output)

    exported = list(stream_video_frames(output))
    assert len(exported) == 3
    assert exported[0].rgb.shape == (26, 34, 3)
    original = next(iter(stream_video_frames(source)))
    assert np.abs(exported[0].rgb[:25, :33].astype(int) - original.rgb.astype(int)).max() <= 3


def test_preserves_source_display_rotation(tmp_path: Path) -> None:
    unrotated = tmp_path / "unrotated.mkv"
    source = tmp_path / "rotated.mp4"
    output = tmp_path / "clip.mp4"
    _write_synthetic_source(unrotated, with_audio=False, video_codec="mpeg4")
    with av.open(unrotated) as input_container, av.open(source, mode="w") as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream_from_template(input_stream)
        output_stream.set_display_rotation(90)
        for packet in input_container.demux(input_stream):
            if packet.dts is not None:
                packet.stream = output_stream
                output_container.mux(packet)

    with av.open(source) as container:
        assert next(container.decode(video=0)).rotation == 90

    TalkingFaceSequence.from_video(source).clip(0.0, 0.3).export_clip(output)

    with av.open(output) as container:
        frame = next(container.decode(video=0))
        assert frame.rotation == 90
        assert (frame.width, frame.height) == (32, 24)


def test_exports_multiple_sequences_with_a_plain_loop(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _write_synthetic_source(source, with_audio=False)
    sequence = TalkingFaceSequence.from_video(source)
    clips = (sequence.clip(0.0, 0.2), sequence.clip(0.3, 0.5))

    outputs = [clip.export_clip(tmp_path / f"clip-{index}.mp4") for index, clip in enumerate(clips)]

    assert all(path.is_file() for path in outputs)
    assert [len(list(stream_video_frames(path))) for path in outputs] == [2, 2]


def test_protects_existing_output_unless_overwrite_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "clip.mp4"
    _write_synthetic_source(source, with_audio=False)
    output.write_bytes(b"keep me")
    clip = TalkingFaceSequence.from_video(source).clip(0.0, 0.2)

    with pytest.raises(FileExistsError, match="already exists"):
        clip.export_clip(output)
    assert output.read_bytes() == b"keep me"

    assert clip.export_clip(output, overwrite=True) == output
    assert output.read_bytes() != b"keep me"


def test_rejects_unbounded_non_mp4_and_source_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_synthetic_source(source, with_audio=False)
    sequence = TalkingFaceSequence.from_video(source)

    with pytest.raises(ValueError, match="bounded"):
        sequence.export_clip(tmp_path / "unbounded.mp4")
    with pytest.raises(ValueError, match=r"\.mp4"):
        sequence.clip(0.0, 0.2).export_clip(tmp_path / "clip.mkv")
    with pytest.raises(ValueError, match="source media"):
        sequence.clip(0.0, 0.2).export_clip(source)


def test_removes_partial_output_when_encoding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "clip.mp4"
    _write_synthetic_source(source, with_audio=False)
    encode_frame = clip_io._encode_video_frame
    calls = 0

    def fail_after_first_frame(*args: Any, **kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("simulated encoder failure")
        return encode_frame(*args, **kwargs)

    monkeypatch.setattr(clip_io, "_encode_video_frame", fail_after_first_frame)

    with pytest.raises(RuntimeError, match="simulated encoder failure"):
        TalkingFaceSequence.from_video(source).clip(0.0, 0.3).export_clip(output)

    assert not output.exists()
    assert not list(tmp_path.glob(".clip.mp4.*.tmp.mp4"))
