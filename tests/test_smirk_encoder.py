from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from talkingfacekit import TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations import smirk
from talkingfacekit.integrations._smirk_model import build_smirk_model


class Tensor:
    def __init__(self, values: NDArray[np.float32]) -> None:
        self.values = values
        self.shape = values.shape

    def detach(self) -> "Tensor":
        return self

    def to(self, *args: object, **kwargs: object) -> "Tensor":
        return self

    def numpy(self) -> NDArray[np.float32]:
        return self.values


def input_frame(index: int = 600, timestamp: float = 20.0) -> smirk.SmirkInputFrame:
    return smirk.SmirkInputFrame(
        index, timestamp, np.zeros((1, 3, 224, 224), dtype=np.float32), np.eye(3)
    )


def install_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, weights: object | None = None
) -> tuple[smirk.SmirkEncoder, dict[str, Any]]:
    checkpoint = tmp_path / "smirk.pt"
    checkpoint.write_bytes(b"Fake checkpoint consumed only by the fake torch boundary")
    observed: dict[str, Any] = {"inference": False, "load_calls": 0, "model_calls": 0}
    sizes = {
        "shape_params": 300,
        "expression_params": 50,
        "pose_params": 3,
        "jaw_params": 3,
        "eyelid_params": 2,
        "cam": 3,
    }
    outputs = {
        name: Tensor(np.full((1, size), 0.1, dtype=np.float32)) for name, size in sizes.items()
    }
    observed["outputs"] = outputs

    class Model:
        def state_dict(self) -> dict[str, Tensor]:
            return {"weight": Tensor(np.zeros(1, dtype=np.float32))}

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            observed["state"] = state
            observed["strict"] = strict

        def float(self) -> "Model":
            return self

        def to(self, device: str) -> "Model":
            observed["device"] = device
            return self

        def eval(self) -> "Model":
            observed["eval"] = True
            return self

        def __call__(self, tensor: Tensor) -> dict[str, Tensor]:
            assert observed["inference"] is True
            observed["model_calls"] += 1
            observed["input"] = tensor.values
            return outputs

    def load(path: Path, *, map_location: str, weights_only: bool) -> object:
        assert path == checkpoint
        assert map_location == "cpu"
        assert weights_only is True
        observed["load_calls"] += 1
        return (
            weights
            if weights is not None
            else {
                "smirk_encoder.weight": Tensor(np.ones(1, dtype=np.float32)),
                "smirk_generator.unused": Tensor(np.zeros(1, dtype=np.float32)),
            }
        )

    @contextmanager
    def inference_mode() -> Generator[None, None, None]:
        observed["inference"] = True
        try:
            yield
        finally:
            observed["inference"] = False

    torch = SimpleNamespace(
        load=load,
        is_tensor=lambda value: isinstance(value, Tensor),
        from_numpy=Tensor,
        inference_mode=inference_mode,
        float32="float32",
    )
    monkeypatch.setattr(
        smirk, "import_module", lambda name: {"torch": torch, "timm": object()}[name]
    )
    monkeypatch.setattr(smirk, "build_smirk_model", lambda *_: Model())
    return smirk.SmirkEncoder(checkpoint), observed


def test_loads_encoder_once_and_returns_owned_facial_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)
    first = encoder.predict(input_frame())
    second = encoder.predict(input_frame(601, 20.033333))
    assert observed["load_calls"] == 1
    assert observed["strict"] is True
    assert observed["eval"] is True
    assert set(observed["state"]) == {"weight"}
    assert observed["device"] == "cpu"
    assert first.frame_index == 600
    assert first.timestamp_seconds == 20
    assert second.timestamp_seconds == 20.033333
    assert first.parameters.model_variant == "flame-2020-generic"
    assert first.parameters.shape.shape == (300,)
    assert first.parameters.expression.shape == (50,)
    assert first.parameters.global_pose.shape == (3,)
    assert first.eyelids.shape == (2,)
    assert first.camera.shape == (3,)
    assert len(first.checkpoint_sha256) == 64
    observed["outputs"]["expression_params"].values[:] = 99
    np.testing.assert_array_equal(first.parameters.expression, second.parameters.expression)
    assert np.all(first.parameters.expression < 1)


@pytest.mark.parametrize("weights", [{"wrong": 1}, {"smirk_encoder.weight": "not a tensor"}, []])
def test_rejects_incomplete_or_non_tensor_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: object
) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        install_encoder(tmp_path, monkeypatch, weights=weights)


@pytest.mark.parametrize("problem", ["missing", "shape", "nonfinite", "eyelids"])
def test_invalid_outputs_never_escape_as_valid_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)
    if problem == "missing":
        del observed["outputs"]["expression_params"]
    elif problem == "shape":
        observed["outputs"]["expression_params"] = Tensor(np.zeros((2, 50), dtype=np.float32))
    elif problem == "nonfinite":
        observed["outputs"]["expression_params"].values[0, 0] = np.nan
    else:
        observed["outputs"]["eyelid_params"].values[:] = 2
    with pytest.raises(ValueError):
        encoder.predict(input_frame())


def test_track_closes_preparation_if_encoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)

    def inputs(*args: object, **kwargs: object) -> Generator[smirk.SmirkInputFrame, None, None]:
        try:
            yield input_frame()
            observed["read_second"] = True
            yield input_frame(601, 20.033333)
        finally:
            observed["closed"] = True

    monkeypatch.setattr(smirk, "iter_smirk_inputs", inputs)
    del observed["outputs"]["cam"]
    sequence = TalkingFaceSequence(
        VideoSource(Path("unused.mp4"), VideoMetadata(1, 1, 30, 1, True))
    )
    with pytest.raises(ValueError, match="cam"):
        next(encoder.track(sequence, landmarker_model="unused.task"))
    assert observed["closed"] is True
    assert "read_second" not in observed


@pytest.mark.parametrize("all_missing", [False, True])
def test_track_preserves_missing_rows_without_running_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_missing: bool
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)
    rows: list[smirk.SmirkInputFrame | smirk.SmirkMissingFrame] = [
        smirk.SmirkMissingFrame(703, 23.433333),
        smirk.SmirkMissingFrame(704, 23.466667),
    ]
    if not all_missing:
        rows.insert(0, input_frame(702, 23.4))
        rows.append(input_frame(705, 23.5))

    def inputs(
        *args: object, **kwargs: object
    ) -> Generator[smirk.SmirkInputFrame | smirk.SmirkMissingFrame, None, None]:
        try:
            yield from rows
        finally:
            observed["closed"] = True

    monkeypatch.setattr(smirk, "iter_smirk_inputs", inputs)
    sequence = TalkingFaceSequence(
        VideoSource(Path("unused.mp4"), VideoMetadata(1, 1, 30, 30, True))
    )
    results = list(encoder.track(sequence, landmarker_model="unused.task"))
    assert [(row.frame_index, row.timestamp_seconds) for row in results] == [
        (row.frame_index, row.timestamp_seconds) for row in rows
    ]
    missing = [row for row in results if isinstance(row, smirk.SmirkMissingFrame)]
    assert missing == [row for row in rows if isinstance(row, smirk.SmirkMissingFrame)]
    assert all(row.reason == "face_landmarks_unavailable" for row in missing)
    assert observed["model_calls"] == (0 if all_missing else 2)
    assert observed["load_calls"] == 1
    assert observed["closed"] is True


def test_closing_track_after_a_missing_frame_closes_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)

    def inputs(*args: object, **kwargs: object) -> Generator[smirk.SmirkMissingFrame, None, None]:
        try:
            yield smirk.SmirkMissingFrame(703, 23.433333)
            observed["read_second"] = True
            yield smirk.SmirkMissingFrame(704, 23.466667)
        finally:
            observed["closed"] = True

    monkeypatch.setattr(smirk, "iter_smirk_inputs", inputs)
    sequence = TalkingFaceSequence(
        VideoSource(Path("unused.mp4"), VideoMetadata(1, 1, 30, 30, True))
    )
    results = encoder.track(sequence, landmarker_model="unused.task")
    assert isinstance(next(results), smirk.SmirkMissingFrame)
    results.close()
    assert observed["closed"] is True
    assert observed["model_calls"] == 0
    assert "read_second" not in observed


@pytest.mark.parametrize("all_missing", [False, True])
def test_extract_collects_parameters_absences_extras_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_missing: bool
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Fake source accessed for its hash, never decoded by this boundary")
    landmarker = tmp_path / "face.task"
    landmarker.write_bytes(b"Fake landmarker model")
    rows: list[smirk.SmirkParameterFrame | smirk.SmirkMissingFrame] = [
        smirk.SmirkMissingFrame(703, 23.433333),
        smirk.SmirkMissingFrame(704, 23.466667),
    ]
    if not all_missing:
        rows.insert(0, encoder.predict(input_frame(702, 23.4)))
        rows.append(encoder.predict(input_frame(705, 23.5)))

    def track(
        sequence: TalkingFaceSequence, *, landmarker_model: str | Path
    ) -> Generator[smirk.SmirkParameterFrame | smirk.SmirkMissingFrame, None, None]:
        assert Path(landmarker_model) == landmarker
        try:
            yield from rows
        finally:
            observed["closed"] = True

    monkeypatch.setattr(encoder, "track", track)
    monkeypatch.setattr(smirk, "version", lambda name: f"fake-{name}-version")
    monkeypatch.setattr(smirk, "code_revision", lambda: {"commit": "abc123", "dirty": True})
    sequence = TalkingFaceSequence(
        VideoSource(source, VideoMetadata(160, 120, 30, 30, True)), 20, 25
    )
    result = encoder.extract(sequence, landmarker_model=landmarker)
    assert result.frame_count == len(rows)
    assert result.estimated_count == (0 if all_missing else 2)
    assert result.source.path == source.resolve()
    assert (result.start_seconds, result.end_seconds) == (20, 25)
    assert result.source_sha256 == sha256(source.read_bytes()).hexdigest()
    assert result.provenance["checkpoint_sha256"] == encoder.checkpoint_sha256
    assert result.provenance["landmarker_sha256"] == sha256(landmarker.read_bytes()).hexdigest()
    assert result.provenance["upstream_revision"] == smirk.SMIRK_UPSTREAM_REVISION
    assert result.provenance["version_torch"] == "fake-torch-version"
    assert result.provenance["code_dirty"] == "true"
    assert observed["closed"] is True
    for index, row in enumerate(rows):
        assert result.frame_indices[index] == row.frame_index
        assert result.timestamps_seconds[index] == row.timestamp_seconds
        if isinstance(row, smirk.SmirkMissingFrame):
            assert not result.estimated[index]
            assert result.missing_reasons[index] == row.reason
            assert np.all(np.isnan(result.shape[index]))
            assert np.all(np.isnan(result.extras["smirk_camera"][index]))
        else:
            for name in ("shape", "expression", "global_pose", "jaw_pose"):
                np.testing.assert_array_equal(
                    getattr(result, name)[index], getattr(row.parameters, name)
                )
            np.testing.assert_array_equal(result.extras["smirk_eyelids"][index], row.eyelids)
            np.testing.assert_array_equal(result.extras["smirk_camera"][index], row.camera)
            np.testing.assert_array_equal(
                result.extras["smirk_source_to_crop"][index], row.source_to_crop
            )
    path = result.save(tmp_path / "result.npz")
    loaded = FlameResult.load(path)
    assert loaded.shape.tobytes() == result.shape.tobytes()
    assert loaded.provenance == result.provenance
    assert observed["model_calls"] == (0 if all_missing else 2)


def test_extract_does_not_return_a_partial_result_after_tracker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder, observed = install_encoder(tmp_path, monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Fake source/model for hashing only")

    def track(*args: object, **kwargs: object) -> Generator[smirk.SmirkMissingFrame, None, None]:
        try:
            yield smirk.SmirkMissingFrame(703, 23.433333)
            raise RuntimeError("tracker failed")
        finally:
            observed["closed"] = True

    monkeypatch.setattr(encoder, "track", track)
    monkeypatch.setattr(smirk, "version", lambda name: "test-version")
    sequence = TalkingFaceSequence(
        VideoSource(source, VideoMetadata(160, 120, 30, 30, True)), 20, 25
    )
    with pytest.raises(RuntimeError, match="tracker failed"):
        encoder.extract(sequence, landmarker_model=source)
    assert observed["closed"] is True


def test_architecture_uses_offline_backbones_and_preserves_upstream_output_semantics() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def create_model(name: str, *, pretrained: bool, features_only: bool) -> Any:
        calls.append((name, pretrained, features_only))
        backbone = torch.nn.Module()
        backbone.feature_info = [{"num_chs": 4}]
        backbone.forward = lambda image: [torch.zeros(image.shape[0], 4, 1, 1)]
        return backbone

    model = cast(Any, build_smirk_model(torch, SimpleNamespace(create_model=create_model)))
    with torch.no_grad():
        for head in [
            model.pose_encoder.pose_cam_layers[0],
            model.shape_encoder.shape_layers[0],
            model.expression_encoder.expression_layers[0],
        ]:
            head.weight.zero_()
            head.bias.zero_()
        model.shape_encoder.shape_layers[0].bias.fill_(0.75)
        model.expression_encoder.expression_layers[0].bias[50:] = torch.tensor(
            [-0.5, 1.5, -1, -0.5, 0.5]
        )
        outputs = model(torch.zeros(1, 3, 224, 224))
    assert len(calls) == 3
    assert all(not pretrained and features_only for _, pretrained, features_only in calls)
    torch.testing.assert_close(outputs["shape_params"], torch.full((1, 300), 0.75))
    torch.testing.assert_close(outputs["eyelid_params"], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(outputs["jaw_params"], torch.tensor([[0.0, -0.2, 0.2]]))
    assert "shape_encoder.shape_layers.0.weight" in model.state_dict()
    assert "expression_encoder.expression_layers.0.weight" in model.state_dict()
