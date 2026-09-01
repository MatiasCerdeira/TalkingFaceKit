from pathlib import Path

import pytest

from talkingfacekit import VideoMetadata, VideoSource


def test_constructs_video_source_without_accessing_its_path() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)

    source = VideoSource(Path("video-that-does-not-need-to-exist.webm"), metadata)

    assert source.path == Path("video-that-does-not-need-to-exist.webm")
    assert source.metadata is metadata


def test_rejects_invalid_video_source_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be a VideoMetadata instance"):
        VideoSource(Path("portrait.webm"), object())  # type: ignore[arg-type]
