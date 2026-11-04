import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.flame import FlameDecoder
from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.video import VideoSource


def model_document() -> dict[str, Any]:
    template = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [-1, 2, 0]], dtype=np.float64)
    directions = np.zeros((5, 3, 400))
    directions[:, 0, 0] = 0.2
    directions[:, 1, 300] = 0.3
    weights = np.zeros((5, 5))
    weights[:, 0] = 1
    weights[3] = [0, 0, 1, 0, 0]
    return {
        "v_template": template,
        "shapedirs": directions,
        "posedirs": np.zeros((5, 3, 36)),
        "J_regressor": np.eye(5),
        "weights": weights,
        "kintree_table": np.array([[-1, 0, 1, 1, 1], [0, 1, 2, 3, 4]], dtype=np.int64),
        "f": np.array([[0, 1, 3], [1, 2, 4]], dtype=np.int64),
    }


def write_model(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "model.pkl"
    with path.open("wb") as stream:
        pickle.dump(model_document() if document is None else document, stream, protocol=2)
    return path


def parameters(*, all_missing: bool = False) -> FlameResult:
    mask = np.array([True, False, True], dtype=np.bool_)
    if all_missing:
        mask[:] = False

    def values(size: int) -> Any:
        array = np.zeros((3, size), dtype=np.float32)
        array[~mask] = np.nan
        return array

    return FlameResult(
        source=VideoSource(
            Path("/not-required/source.mp4"), VideoMetadata(640, 480, 30, None, True)
        ),
        source_sha256="a" * 64,
        start_seconds=0,
        end_seconds=1,
        model_variant="flame-2020-generic",
        frame_indices=np.array([10, 11, 12], dtype=np.int64),
        timestamps_seconds=np.array([0.1, 0.2, 0.4], dtype=np.float64),
        estimated=mask,
        missing_reasons=tuple("" if valid else "face_landmarks_unavailable" for valid in mask),
        shape=values(2),
        expression=values(1),
        global_pose=values(3),
        jaw_pose=values(3),
        extras={},
        extra_conventions={},
        provenance={"backend": "synthetic", "checkpoint_sha256": "b" * 64},
    )


def test_neutral_parameters_preserve_template_source_timeline_and_absences(tmp_path: Path) -> None:
    decoder = FlameDecoder(write_model(tmp_path))
    result = parameters()
    mesh = decoder.decode(result)
    expected = model_document()["v_template"].astype(np.float32)
    np.testing.assert_array_equal(mesh.vertices[0], expected)
    np.testing.assert_array_equal(mesh.vertices[2], expected)
    assert np.all(np.isnan(mesh.vertices[1]))
    np.testing.assert_array_equal(mesh.frame_indices, result.frame_indices)
    np.testing.assert_array_equal(mesh.timestamps_seconds, result.timestamps_seconds)
    np.testing.assert_array_equal(mesh.detected, result.estimated)
    assert mesh.triangles.dtype == np.int32
    assert not np.shares_memory(mesh.frame_indices, result.frame_indices)
    assert decoder.provenance["model_sha256"] in mesh.topology


def test_shape_and_expression_use_distinct_columns_of_the_model_basis(tmp_path: Path) -> None:
    decoder = FlameDecoder(write_model(tmp_path))
    result = parameters()
    result.shape[result.estimated, 0] = 2
    result.expression[result.estimated, 0] = 3
    mesh = decoder.decode(result)
    expected = model_document()["v_template"] + [0.4, 0.9, 0]
    np.testing.assert_allclose(mesh.vertices[0], expected, atol=1e-6)


def test_jaw_rotates_about_its_joint_and_pose_correctives_are_applied(tmp_path: Path) -> None:
    document = model_document()
    # Jaw is joint 2: its R[0,0]-1 is corrective feature 9. Vertex 4 remains root-skinned.
    document["posedirs"][4, 2, 9] = 0.25
    decoder = FlameDecoder(write_model(tmp_path, document))
    result = parameters()
    result.jaw_pose[result.estimated, 2] = np.pi / 2
    mesh = decoder.decode(result)
    np.testing.assert_allclose(mesh.vertices[0, 3], [0, 3, 0], atol=1e-6)
    np.testing.assert_allclose(mesh.vertices[0, 4], [-1, 2, -0.25], atol=1e-6)
    np.testing.assert_allclose(mesh.vertices[0, 2], [0, 2, 0], atol=1e-6)


def test_global_axis_angle_pose_rotates_the_complete_head(tmp_path: Path) -> None:
    decoder = FlameDecoder(write_model(tmp_path))
    result = parameters()
    result.global_pose[result.estimated, 2] = np.pi / 2
    mesh = decoder.decode(result)
    np.testing.assert_allclose(mesh.vertices[0, 1], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(mesh.vertices[0, 3], [-2, 1, 0], atol=1e-6)


def test_smirk_eyelids_are_added_after_pose_and_require_explicit_assets(tmp_path: Path) -> None:
    model = write_model(tmp_path)
    result = parameters()
    result.global_pose[result.estimated, 2] = np.pi / 2
    baseline = FlameDecoder(model).decode(result)
    eyelids = np.full((3, 2), np.nan, dtype=np.float32)
    eyelids[result.estimated] = [0.4, 0.7]
    result = replace(
        result,
        extras={"smirk_eyelids": eyelids},
        extra_conventions={"smirk_eyelids": "Left/right SMIRK coefficients"},
    )
    with pytest.raises(ValueError, match="explicit smirk_eyelids assets"):
        FlameDecoder(model).decode(result)
    left, right = np.zeros((5, 3)), np.zeros((5, 3))
    left[:, 0], right[:, 2] = 0.1, 0.2
    np.save(tmp_path / "l_eyelid.npy", left)
    np.save(tmp_path / "r_eyelid.npy", right)
    mesh = FlameDecoder(model, smirk_eyelids=tmp_path).decode(result)
    expected = baseline.vertices[0] + [0.04, 0, 0.14]
    np.testing.assert_allclose(mesh.vertices[0], expected, atol=1e-6)
    assert np.all(np.isnan(mesh.vertices[1]))


def test_all_missing_and_different_batch_sizes_remain_source_aligned(tmp_path: Path) -> None:
    decoder = FlameDecoder(write_model(tmp_path))
    result = parameters()
    np.testing.assert_allclose(
        decoder.decode(result, batch_size=1).vertices,
        decoder.decode(result, batch_size=32).vertices,
        equal_nan=True,
    )
    mesh = decoder.decode(parameters(all_missing=True))
    assert mesh.frame_count == 3
    assert not np.any(mesh.detected)
    assert np.all(np.isnan(mesh.vertices))


@pytest.mark.parametrize(
    "problem", ["basis", "shape", "expression", "batch_zero", "batch_bool", "smirk_extension"]
)
def test_incompatible_inputs_fail_explicitly(tmp_path: Path, problem: str) -> None:
    decoder = FlameDecoder(write_model(tmp_path))
    result = parameters()
    kwargs: dict[str, Any] = {}
    if problem == "basis":
        result = replace(result, model_variant="flame-2023-open")
    elif problem in {"shape", "expression"}:
        value = np.full((3, 301 if problem == "shape" else 101), np.nan, dtype=np.float32)
        value[result.estimated] = 0
        result = (
            replace(result, shape=value)
            if problem == "shape"
            else replace(result, expression=value)
        )
    elif problem.startswith("batch"):
        kwargs["batch_size"] = 0 if problem == "batch_zero" else True
    else:
        result = replace(result, provenance={"backend": "smirk", "checkpoint_sha256": "b" * 64})
    with pytest.raises((TypeError, ValueError)):
        decoder.decode(result, **kwargs)


@pytest.mark.parametrize("problem", ["nonfinite", "tree", "weights", "triangles", "directions"])
def test_model_geometry_is_validated_before_evaluation(tmp_path: Path, problem: str) -> None:
    document = model_document()
    if problem == "nonfinite":
        document["v_template"][0, 0] = np.nan
    elif problem == "tree":
        document["kintree_table"][0, 2] = 3
    elif problem == "weights":
        document["weights"][0] = 0
    elif problem == "triangles":
        document["f"][0, 0] = 999
    else:
        document["shapedirs"] = document["shapedirs"][:, :, :300]
    with pytest.raises(ValueError):
        FlameDecoder(write_model(tmp_path, document))


def test_reads_legacy_chumpy_leaf_and_csc_arrays_without_importing_their_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = model_document()
    with monkeypatch.context() as patch:
        chumpy, ch = ModuleType("chumpy"), ModuleType("chumpy.ch")
        sparse, csc = ModuleType("scipy.sparse"), ModuleType("scipy.sparse.csc")
        ch_class = type("Ch", (), {"__module__": "chumpy.ch"})
        csc_class = type("csc_matrix", (), {"__module__": "scipy.sparse.csc"})
        ch.Ch = ch_class  # type: ignore[attr-defined]  # Synthetic legacy pickle module.
        csc.csc_matrix = csc_class  # type: ignore[attr-defined]  # Synthetic legacy pickle module.
        for module in (chumpy, ch, ModuleType("scipy"), sparse, csc):
            patch.setitem(sys.modules, module.__name__, module)
        leaf = ch_class()
        leaf.x = document["shapedirs"]
        matrix = csc_class()
        matrix._shape = (5, 5)
        matrix.indices = np.arange(5, dtype=np.int32)
        matrix.indptr = np.arange(6, dtype=np.int32)
        matrix.data = np.ones(5)
        document["shapedirs"], document["J_regressor"] = leaf, matrix
        model = write_model(tmp_path, document)
    decoder = FlameDecoder(model)
    mesh = decoder.decode(parameters())
    np.testing.assert_allclose(mesh.vertices[0], model_document()["v_template"])


def test_model_loader_refuses_unrecognized_pickle_globals(tmp_path: Path) -> None:
    document = model_document()
    document["unexpected"] = Path("unsupported-model-object")
    with pytest.raises(ValueError, match="Unsupported FLAME model pickle global"):
        FlameDecoder(write_model(tmp_path, document))


def test_geometry_runs_in_fresh_process_without_inference_or_legacy_packages(
    tmp_path: Path,
) -> None:
    model = write_model(tmp_path)
    result = parameters().save(tmp_path / "result.npz")
    script = """
import sys
from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.flame import FlameDecoder
mesh = FlameDecoder(sys.argv[1]).decode(FlameResult.load(sys.argv[2]))
assert mesh.vertex_count == 5 and mesh.frame_count == 3
assert not any(name in sys.modules for name in ('torch', 'mediapipe', 'chumpy', 'scipy', 'skimage'))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(model), str(result)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
