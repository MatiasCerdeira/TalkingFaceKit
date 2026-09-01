import pytest

from talkingfacekit import VideoMetadata


@pytest.mark.parametrize("width", [True, 1920.5])
def test_rejects_non_integer_width(width: object) -> None:
    with pytest.raises(TypeError, match="width must be an integer"):
        VideoMetadata(width, 1080, 24.0, 10.0, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("height", [False, 1080.5])
def test_rejects_non_integer_height(height: object) -> None:
    with pytest.raises(TypeError, match="height must be an integer"):
        VideoMetadata(1920, height, 24.0, 10.0, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [0, -1])
def test_rejects_non_positive_width(width: int) -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        VideoMetadata(width, 1080, 24.0, 10.0, True)


@pytest.mark.parametrize("height", [0, -1])
def test_rejects_non_positive_height(height: int) -> None:
    with pytest.raises(ValueError, match="height must be positive"):
        VideoMetadata(1920, height, 24.0, 10.0, True)


@pytest.mark.parametrize("average_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_average_fps(average_fps: float) -> None:
    with pytest.raises(ValueError, match="average_fps must be finite and positive"):
        VideoMetadata(1920, 1080, average_fps, 10.0, True)


@pytest.mark.parametrize(
    "stream_duration_seconds",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_rejects_invalid_stream_duration(stream_duration_seconds: float) -> None:
    with pytest.raises(ValueError, match="stream_duration_seconds must be finite and positive"):
        VideoMetadata(1920, 1080, 24.0, stream_duration_seconds, True)


def test_accepts_unknown_average_fps_and_duration() -> None:
    metadata = VideoMetadata(1920, 1080, None, None, False)

    assert metadata.average_fps is None
    assert metadata.stream_duration_seconds is None


def test_rejects_non_boolean_audio_presence() -> None:
    with pytest.raises(TypeError, match="has_audio must be boolean"):
        VideoMetadata(1920, 1080, 24.0, 10.0, 1)  # type: ignore[arg-type]
