from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from talkingfacekit import DecodedVideoFrame, TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.integrations import smirk


def landmarks() -> NDArray[np.float64]:
    points = np.empty((478, 2), dtype=np.float64)
    points[::2] = [30.0, 20.0]
    points[1::2] = [130.0, 100.0]
    return points


def source_frame(index: int, timestamp: float) -> DecodedVideoFrame:
    rgb = np.full((120, 160, 3), [240, 127, 3], dtype=np.uint8)
    return DecodedVideoFrame(index, timestamp, rgb)


def install_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: list[DecodedVideoFrame],
    *,
    miss_at: set[int] | None = None,
) -> tuple[TalkingFaceSequence, Path, dict[str, Any]]:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"Only the fake MediaPipe boundary reads this test asset.")
    sequence = TalkingFaceSequence(
        VideoSource(Path("not-opened.mp4"), VideoMetadata(160, 120, 30.0, 10.0, True)),
        0.5,
        0.7,
    )
    observed: dict[str, Any] = {"consumed": [], "images": [], "closed": False, "warp_calls": 0}

    def stream(
        path: str | Path, *, start_seconds: float, end_seconds: float | None
    ) -> Iterator[DecodedVideoFrame]:
        observed["request"] = (path, start_seconds, end_seconds)
        try:
            for frame in frames:
                observed["consumed"].append(frame.frame_index)
                yield frame
        finally:
            observed["stream_closed"] = True

    def image(*, image_format: object, data: NDArray[np.uint8]) -> object:
        assert image_format == "srgb"
        observed["images"].append(data.copy())
        return object()

    def detect(image: object) -> SimpleNamespace:
        del image
        if miss_at is not None and len(observed["images"]) in miss_at:
            return SimpleNamespace(face_landmarks=[])
        return SimpleNamespace(
            face_landmarks=[[SimpleNamespace(x=x / 160, y=y / 120) for x, y in landmarks()]]
        )

    def close() -> None:
        observed["closed"] = True

    def create_from_options(options: object) -> SimpleNamespace:
        observed["options"] = options
        return SimpleNamespace(detect=detect, close=close)

    mediapipe = SimpleNamespace(
        Image=image,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        tasks=SimpleNamespace(
            BaseOptions=SimpleNamespace,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE="image"),
                FaceLandmarkerOptions=SimpleNamespace,
                FaceLandmarker=SimpleNamespace(create_from_options=create_from_options),
            ),
        ),
    )
    transform = SimpleNamespace(params=np.eye(3), inverse=object())

    def estimate_transform(
        kind: str, source: NDArray[np.float64], target: NDArray[np.float64]
    ) -> object:
        observed["crop"] = (kind, source, target)
        return transform

    def warp(
        rgb: NDArray[np.uint8],
        inverse: object,
        *,
        output_shape: tuple[int, int],
        preserve_range: bool,
    ) -> NDArray[np.float64]:
        assert inverse is transform.inverse
        assert output_shape == (224, 224)
        assert preserve_range is True
        observed["warp_calls"] += 1
        observed["warp_rgb"] = rgb.copy()
        return np.full((224, 224, 3), [240.9, 127.1, 3.99], dtype=np.float64)

    transforms = SimpleNamespace(estimate_transform=estimate_transform, warp=warp)

    def load(name: str) -> Any:
        return {"mediapipe": mediapipe, "skimage.transform": transforms}[name]

    observed["loader"] = load
    monkeypatch.setattr(smirk, "stream_video_frames", stream)
    monkeypatch.setattr(smirk, "import_module", load)
    return sequence, model, observed


def test_streams_native_frames_with_official_detection_options_and_rgb_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [source_frame(17, 0.54), source_frame(18, 0.579)]
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, frames)
    inputs = smirk.iter_smirk_inputs(sequence, landmarker_model=model)
    assert observed["consumed"] == []
    first = next(inputs)
    assert isinstance(first, smirk.SmirkInputFrame)
    assert observed["consumed"] == [17]
    assert observed["request"] == (sequence.source.path, 0.5, 0.7)
    assert first.frame_index == 17
    assert first.timestamp_seconds == 0.54
    assert first.image.shape == (1, 3, 224, 224)
    assert first.image.dtype == np.float32
    assert first.image.flags.c_contiguous
    np.testing.assert_array_equal(
        first.image[0, :, 0, 0], np.array([240, 127, 3], dtype=np.float32) / np.float32(255)
    )
    np.testing.assert_array_equal(observed["images"][0], frames[0].rgb)
    np.testing.assert_array_equal(observed["warp_rgb"], frames[0].rgb)
    np.testing.assert_array_equal(observed["crop"][1], [[17.5, -2.5], [17.5, 122.5], [142.5, -2.5]])
    options = observed["options"]
    assert options.base_options.model_asset_path == str(model)
    assert options.running_mode == "image"
    assert options.num_faces == 1
    assert options.min_face_detection_confidence == 0.1
    assert options.min_face_presence_confidence == 0.1
    assert options.output_face_blendshapes is True
    assert options.output_facial_transformation_matrixes is True
    second = next(inputs)
    assert isinstance(second, smirk.SmirkInputFrame)
    assert second.timestamp_seconds == 0.579
    assert second.frame_index == 18
    first.image.fill(0)
    assert np.any(second.image)
    assert list(inputs) == []
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


def test_consecutive_missing_faces_preserve_source_rows_and_resume_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence, model, observed = install_boundary(
        tmp_path,
        monkeypatch,
        [
            source_frame(17, 0.54),
            source_frame(18, 0.579),
            source_frame(19, 0.62),
            source_frame(20, 0.66),
        ],
        miss_at={2, 3},
    )
    inputs = smirk.iter_smirk_inputs(sequence, landmarker_model=model)
    assert isinstance(next(inputs), smirk.SmirkInputFrame)
    missing = next(inputs)
    assert missing == smirk.SmirkMissingFrame(18, 0.579)
    assert observed["consumed"] == [17, 18]
    assert observed["warp_calls"] == 1
    assert bool(observed["closed"]) is False
    assert next(inputs) == smirk.SmirkMissingFrame(19, 0.62)
    last = next(inputs)
    assert isinstance(last, smirk.SmirkInputFrame)
    assert (last.frame_index, last.timestamp_seconds) == (20, 0.66)
    assert list(inputs) == []
    assert observed["warp_calls"] == 2
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


@pytest.mark.parametrize("miss_at", [{1}, {2}, {1, 2}])
def test_leading_trailing_and_all_missing_frames_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, miss_at: set[int]
) -> None:
    frames = [source_frame(17, 0.54), source_frame(18, 0.579)]
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, frames, miss_at=miss_at)
    results = list(smirk.iter_smirk_inputs(sequence, landmarker_model=model))
    assert [(row.frame_index, row.timestamp_seconds) for row in results] == [
        (frame.frame_index, frame.timestamp_seconds) for frame in frames
    ]
    assert [isinstance(row, smirk.SmirkMissingFrame) for row in results] == [
        index in miss_at for index in (1, 2)
    ]
    assert observed["warp_calls"] == 2 - len(miss_at)
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


def test_closing_after_a_missing_frame_releases_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence, model, observed = install_boundary(
        tmp_path, monkeypatch, [source_frame(17, 0.54), source_frame(18, 0.579)], miss_at={1}
    )
    inputs = smirk.iter_smirk_inputs(sequence, landmarker_model=model)
    assert isinstance(next(inputs), smirk.SmirkMissingFrame)
    inputs.close()
    assert observed["consumed"] == [17]
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


@pytest.mark.parametrize("stage", ["decoder", "detector", "crop"])
def test_unexpected_errors_propagate_instead_of_becoming_missing_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, [source_frame(17, 0.54)])
    if stage == "decoder":

        def broken_stream(*args: object, **kwargs: object) -> Iterator[DecodedVideoFrame]:
            try:
                yield source_frame(17, 0.54)
                raise RuntimeError("broken decoder")
            finally:
                observed["stream_closed"] = True

        monkeypatch.setattr(smirk, "stream_video_frames", broken_stream)
    else:

        def fail(*args: object) -> Any:
            raise RuntimeError(f"broken {stage}")

        if stage == "detector":
            monkeypatch.setattr(smirk._SmirkLandmarker, "detect", fail)
        else:
            monkeypatch.setattr(smirk, "_crop_source_points", fail)
    with pytest.raises(RuntimeError, match=f"broken {stage}"):
        list(smirk.iter_smirk_inputs(sequence, landmarker_model=model))
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


def test_closing_an_interrupted_iterator_releases_detector_and_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence, model, observed = install_boundary(
        tmp_path, monkeypatch, [source_frame(17, 0.54), source_frame(18, 0.579)]
    )
    inputs = smirk.iter_smirk_inputs(sequence, landmarker_model=model)
    try:
        next(inputs)
    finally:
        inputs.close()
    assert observed["consumed"] == [17]
    assert observed["closed"] is True
    assert observed["stream_closed"] is True


def test_missing_optional_dependency_closes_the_started_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, [source_frame(17, 0.54)])

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(smirk, "import_module", missing)
    with pytest.raises(ImportError, match="requires scikit-image"):
        next(smirk.iter_smirk_inputs(sequence, landmarker_model=model))
    assert observed["stream_closed"] is True


def test_empty_source_fails_without_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="at least one video frame"):
        next(smirk.iter_smirk_inputs(sequence, landmarker_model=model))
    assert observed["closed"] is False
    assert observed["stream_closed"] is True


@pytest.mark.parametrize("problem", ["empty", "nonfinite", "wrong_shape"])
def test_rejects_unusable_face_geometry(problem: str) -> None:
    points = landmarks()
    if problem == "empty":
        points[:] = 1
    elif problem == "nonfinite":
        points[0] = np.nan
    else:
        points = points[:477]
    with pytest.raises(ValueError):
        smirk._crop_source_points(points)


def test_validates_prepared_input_range_and_transform() -> None:
    image = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with pytest.raises(ValueError, match="invertible"):
        smirk.SmirkInputFrame(0, 0, image, np.diag([0.0, 1.0, 1.0]))
    image[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        smirk.SmirkInputFrame(0, 0, image, np.eye(3))


def test_real_warp_matches_official_bgr_crop_and_rgb_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transforms = pytest.importorskip("skimage.transform")
    cv2 = pytest.importorskip("cv2")
    frame = source_frame(17, 0.54)
    frame.rgb[:] = np.random.default_rng(42).integers(0, 256, frame.rgb.shape, dtype=np.uint8)
    sequence, model, observed = install_boundary(tmp_path, monkeypatch, [frame])
    loader = observed["loader"]
    monkeypatch.setattr(
        smirk,
        "import_module",
        lambda name: transforms if name == "skimage.transform" else loader(name),
    )
    results = list(smirk.iter_smirk_inputs(sequence, landmarker_model=model))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, smirk.SmirkInputFrame)
    # Reference control points are derived independently from the official crop's extent formula.
    transform = transforms.estimate_transform(
        "similarity",
        np.array([[17.5, -2.5], [17.5, 122.5], [142.5, -2.5]]),
        np.array([[0, 0], [0, 223], [223, 0]]),
    )
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    reference = transforms.warp(
        bgr, transform.inverse, output_shape=(224, 224), preserve_range=True
    ).astype(np.uint8)
    reference_rgb = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), (224, 224))
    expected = np.ascontiguousarray(reference_rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_allclose(result.source_to_crop, transform.params)
