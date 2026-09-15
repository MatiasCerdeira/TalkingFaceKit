from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from talkingfacekit import DecodedVideoFrame, TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.integrations import deeptalk


def make_sequence(
    *, start_seconds: float = 0.0, end_seconds: float | None = None
) -> TalkingFaceSequence:
    return TalkingFaceSequence(
        source=VideoSource(
            path=Path("source.mp4"),
            metadata=VideoMetadata(3, 2, 25.0, 2.0, True),
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def make_frame(frame_index: int, timestamp_seconds: float) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
    )


def install_frames(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[DecodedVideoFrame],
    *,
    calls: list[tuple[Path, float, float | None]] | None = None,
    consumed_indices: list[int] | None = None,
) -> None:
    def fake_stream_video_frames(
        video_path: str | Path,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> Iterator[DecodedVideoFrame]:
        if calls is not None:
            calls.append((Path(video_path), start_seconds, end_seconds))
        for frame in frames:
            if consumed_indices is not None:
                consumed_indices.append(frame.frame_index)
            yield frame

    monkeypatch.setattr(deeptalk, "stream_video_frames", fake_stream_video_frames)


@pytest.mark.parametrize(
    ("source_fps", "expected_slot_count"),
    [
        (24, 24),
        (25, 25),
        (30, 25),
        (50, 25),
    ],
)
def test_samples_common_source_rates_at_25_fps(
    monkeypatch: pytest.MonkeyPatch,
    source_fps: int,
    expected_slot_count: int,
) -> None:
    frames = [make_frame(index, index / source_fps) for index in range(source_fps)]
    install_frames(monkeypatch, frames)

    sampled = list(deeptalk._sample_video_frames_25_fps(make_sequence(end_seconds=1.0)))

    assert [slot for slot, _ in sampled] == pytest.approx(
        [index / 25 for index in range(expected_slot_count)]
    )
    for slot, selected_frame in sampled:
        expected_frame = min(frames, key=lambda frame: abs(frame.timestamp_seconds - slot))
        assert selected_frame is expected_frame


def test_samples_irregular_pts_by_nearest_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = [0.0, 0.03, 0.09, 0.13]
    install_frames(
        monkeypatch,
        [make_frame(index, timestamp) for index, timestamp in enumerate(timestamps)],
    )

    sampled = list(deeptalk._sample_video_frames_25_fps(make_sequence(end_seconds=0.14)))

    assert [slot for slot, _ in sampled] == pytest.approx([0.0, 0.04, 0.08, 0.12])
    assert [frame.frame_index for _, frame in sampled] == [0, 1, 2, 3]


def test_preserves_source_frame_and_streams_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [make_frame(7, 0.54), make_frame(8, 0.58), make_frame(9, 0.62)]
    calls: list[tuple[Path, float, float | None]] = []
    consumed_indices: list[int] = []
    install_frames(
        monkeypatch,
        frames,
        calls=calls,
        consumed_indices=consumed_indices,
    )
    sampled = deeptalk._sample_video_frames_25_fps(
        make_sequence(start_seconds=0.5, end_seconds=0.7)
    )

    assert calls == []
    assert consumed_indices == []

    slot_timestamp_seconds, source_frame = next(sampled)

    assert calls == [(Path("source.mp4"), 0.5, 0.7)]
    assert consumed_indices == [7, 8]
    assert slot_timestamp_seconds == pytest.approx(0.54)
    assert source_frame is frames[0]
    assert source_frame.frame_index == 7
    assert source_frame.timestamp_seconds == pytest.approx(0.54)


def test_does_not_create_slots_before_the_first_source_pts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_frames(monkeypatch, [make_frame(0, 5.0), make_frame(1, 5.04)])

    sampled = list(deeptalk._sample_video_frames_25_fps(make_sequence(end_seconds=5.1)))

    assert [slot for slot, _ in sampled] == pytest.approx([5.0, 5.04])
    assert all(slot >= 5.0 for slot, _ in sampled)


def test_does_not_create_slots_after_the_last_source_pts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_frames(monkeypatch, [make_frame(0, 0.0), make_frame(1, 0.06)])

    sampled = list(deeptalk._sample_video_frames_25_fps(make_sequence(end_seconds=0.2)))

    assert [slot for slot, _ in sampled] == pytest.approx([0.0, 0.04])
    assert all(slot <= 0.06 for slot, _ in sampled)


@pytest.mark.parametrize("start_seconds", [0.0, 10.0])
def test_nearest_neighbor_ties_prefer_the_earlier_frame_independently_of_offset(
    monkeypatch: pytest.MonkeyPatch,
    start_seconds: float,
) -> None:
    frames = [
        make_frame(3, start_seconds + 0.02),
        make_frame(4, start_seconds + 0.06),
    ]
    install_frames(monkeypatch, frames)

    sampled = list(
        deeptalk._sample_video_frames_25_fps(
            make_sequence(start_seconds=start_seconds, end_seconds=start_seconds + 0.1)
        )
    )

    assert [slot for slot, _ in sampled] == pytest.approx([start_seconds + 0.04])
    assert [frame.frame_index for _, frame in sampled] == [3]


def test_converts_non_contiguous_rgb_pixels_to_packed_deeptalk_rgb24() -> None:
    deeptalk_asd = pytest.importorskip("deeptalk_asd")
    backing = np.arange(2 * 6 * 3, dtype=np.uint8).reshape(2, 6, 3)
    rgb = backing[:, ::2, :]
    assert not rgb.flags.c_contiguous
    source_frame = DecodedVideoFrame(frame_index=4, timestamp_seconds=1.0, rgb=rgb)

    converted = cast(Any, deeptalk._to_deeptalk_video_frame(source_frame, deeptalk_asd))

    assert converted.width == 3
    assert converted.height == 2
    assert converted.type == deeptalk_asd.VideoBufferType.RGB24
    assert bytes(converted.data) == rgb.tobytes(order="C")
