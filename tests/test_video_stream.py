from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import DecodedVideoFrame, stream_video_frames

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def test_streams_rgb_frames_over_a_half_open_interval() -> None:
    frames = list(stream_video_frames(FIXTURE, start_seconds=0.04, end_seconds=0.1))

    assert [frame.frame_index for frame in frames] == [1, 2]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.042, 0.083])
    assert [frame.rgb.shape for frame in frames] == [(1080, 1920, 3)] * 2
    assert all(frame.rgb.dtype == np.dtype(np.uint8) for frame in frames)


def test_validates_path_and_interval_before_decoding(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video not found"):
        stream_video_frames(tmp_path / "missing.webm")

    with pytest.raises(ValueError, match="end_seconds must be greater"):
        stream_video_frames(FIXTURE, start_seconds=0.1, end_seconds=0.1)


def test_rejects_an_interval_without_frames() -> None:
    frames = stream_video_frames(FIXTURE, start_seconds=0.001, end_seconds=0.04)

    with pytest.raises(ValueError, match="No decoded video frames fall within"):
        list(frames)


def test_decoded_frame_validates_rgb_layout() -> None:
    with pytest.raises(ValueError, match=r"shape \(height, width, 3\)"):
        DecodedVideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            rgb=np.zeros((2, 2, 4), dtype=np.uint8),
        )
