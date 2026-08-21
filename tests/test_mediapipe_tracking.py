import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


class FakeImage:
    observed_shapes: ClassVar[list[tuple[int, ...]]] = []

    def __init__(
        self, *, image_format: object, data: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
    ):
        del image_format
        self.observed_shapes.append(data.shape)
        assert data.dtype == np.dtype(np.uint8)


class FakeLandmark:
    def __init__(self, point_index: int) -> None:
        self.x = point_index / 477
        self.y = 0.25
        self.z = -0.1


class FakeLandmarker:
    observed_timestamps_ms: ClassVar[list[int]] = []
    closed: ClassVar[bool] = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> SimpleNamespace:
        del image
        self.observed_timestamps_ms.append(timestamp_ms)
        faces = [] if timestamp_ms == 42 else [[FakeLandmark(index) for index in range(478)]]
        return SimpleNamespace(face_landmarks=faces)

    def close(self) -> None:
        type(self).closed = True


class FakeFaceLandmarker:
    @staticmethod
    def create_from_options(options: object) -> FakeLandmarker:
        del options
        return FakeLandmarker()


class FakeOptions:
    def __init__(self, **values: object) -> None:
        self.values = values


class FakeBaseOptions:
    def __init__(self, *, model_asset_path: str) -> None:
        self.model_asset_path = model_asset_path


def install_fake_mediapipe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("mediapipe")
    module.__dict__.update(
        {
            "__version__": "test-version",
            "Image": FakeImage,
            "ImageFormat": SimpleNamespace(SRGB=object()),
            "tasks": SimpleNamespace(
                BaseOptions=FakeBaseOptions,
                vision=SimpleNamespace(
                    FaceLandmarkerOptions=FakeOptions,
                    FaceLandmarker=FakeFaceLandmarker,
                    RunningMode=SimpleNamespace(VIDEO=object()),
                ),
            ),
        }
    )
    monkeypatch.setitem(sys.modules, "mediapipe", module)


def test_streams_frames_through_mediapipe_and_preserves_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeImage.observed_shapes = []
    FakeLandmarker.observed_timestamps_ms = []
    FakeLandmarker.closed = False
    install_fake_mediapipe(monkeypatch)
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"fake model handled by fake MediaPipe")

    tracker = MediaPipeFaceTracker(model_path)
    track = tracker.track(FIXTURE, start_seconds=0.0, end_seconds=0.1)

    assert track.tracker_name == "mediapipe-face-landmarker"
    assert track.tracker_version == "test-version"
    assert track.topology == "mediapipe-face-landmarker-478"
    assert track.frame_indices.tolist() == [0, 1, 2]
    assert track.timestamps_seconds.tolist() == pytest.approx([0.0, 0.042, 0.083])
    assert track.detected.tolist() == [True, False, True]
    assert track.landmarks.shape == (3, 478, 3)
    assert track.landmarks.dtype == np.dtype(np.float32)
    assert track.landmarks[0, 477, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(track.landmarks[1]))
    assert FakeLandmarker.observed_timestamps_ms == [0, 42, 83]
    assert FakeImage.observed_shapes == [(1080, 1920, 3)] * 3
    assert FakeLandmarker.closed is True


def test_rejects_missing_model_before_loading_mediapipe(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MediaPipe model not found"):
        MediaPipeFaceTracker(tmp_path / "missing.task")


def test_rejects_invalid_confidence(tmp_path: Path) -> None:
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"model")

    with pytest.raises(ValueError, match="min_tracking_confidence"):
        MediaPipeFaceTracker(model_path, min_tracking_confidence=1.1)
