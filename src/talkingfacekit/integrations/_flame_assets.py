"""Read FLAME2020's legacy numerical storage without Chumpy or SciPy execution.

Only the known NumPy constructors and data-only Ch/CSC containers are recognized. Model files
must still come from a trusted source; this reader is not a sandbox for arbitrary pickle data.
"""

from __future__ import annotations

import codecs
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class _LegacyCh:
    """Storage for the raw ``x`` array of a serialized Chumpy leaf, without graph execution."""

    x: object


class _LegacyCSC:
    """Storage for the arrays of a serialized SciPy compressed sparse column matrix."""

    _shape: tuple[int, int]
    indices: NDArray[np.int32]
    indptr: NDArray[np.int32]
    data: NDArray[np.float64]


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        # NumPy's legacy pickle constructors are not described by its public typing stubs.
        multiarray = cast(Any, np.core.multiarray)
        constructors: dict[tuple[str, str], object] = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): multiarray.scalar,
            ("__builtin__", "set"): set,
            ("builtins", "set"): set,
            ("_codecs", "encode"): codecs.encode,
            ("chumpy.ch", "Ch"): _LegacyCh,
            ("scipy.sparse.csc", "csc_matrix"): _LegacyCSC,
            ("scipy.sparse._csc", "csc_matrix"): _LegacyCSC,
        }
        try:
            return constructors[(module, name)]
        except KeyError:
            raise ValueError(f"Unsupported FLAME model pickle global: {module}.{name}") from None


@dataclass(frozen=True, slots=True)
class FlameModelData:
    """Validated float32 geometry arrays, int32 triangles and int64 joint parents."""

    template: NDArray[np.float32]
    directions: NDArray[np.float32]
    pose_directions: NDArray[np.float32]
    joint_regressor: NDArray[np.float32]
    weights: NDArray[np.float32]
    parents: NDArray[np.int64]
    triangles: NDArray[np.int32]


def load_flame_model(path: Path) -> FlameModelData:
    """Read and validate the five-joint FLAME2020 numerical model, without optional imports."""
    with path.open("rb") as stream:
        document = _ModelUnpickler(stream, encoding="latin1").load()
    if not isinstance(document, dict):
        raise TypeError("FLAME model must contain a dictionary")
    required = {
        "v_template",
        "shapedirs",
        "posedirs",
        "J_regressor",
        "weights",
        "kintree_table",
        "f",
    }
    if not required.issubset(document):
        raise ValueError("FLAME model is missing required geometry fields")
    template = _float_array(document["v_template"], "v_template")
    if template.ndim != 2 or template.shape[1] != 3 or template.shape[0] < 3:
        raise ValueError("FLAME template must have shape (vertex_count, 3)")
    count = template.shape[0]
    directions = _float_array(document["shapedirs"], "shapedirs")
    pose_directions = _float_array(document["posedirs"], "posedirs")
    regressor = _float_array(document["J_regressor"], "J_regressor")
    weights = _float_array(document["weights"], "weights")
    if directions.ndim != 3 or directions.shape[:2] != (count, 3) or directions.shape[2] <= 300:
        raise ValueError(
            "FLAME shapedirs must contain 300 shape components followed by expressions"
        )
    if pose_directions.shape != (count, 3, 36) or regressor.shape != (5, count):
        raise ValueError("FLAME requires five joints and 36 pose-corrective components")
    if (
        weights.shape != (count, 5)
        or np.any(weights < 0)
        or not np.allclose(weights.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError(
            "FLAME skinning weights must be nonnegative (vertex_count, 5) rows summing to 1"
        )
    tree = _integer_array(document["kintree_table"], "kintree_table")
    if (
        tree.shape != (2, 5)
        or not np.array_equal(tree[1], np.arange(5))
        or not np.array_equal(tree[0, 1:], [0, 1, 1, 1])
    ):
        raise ValueError("FLAME requires the root/neck/jaw/left-eye/right-eye joint tree")
    parents = tree[0].copy()
    parents[0] = -1
    faces = _integer_array(document["f"], "f")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("FLAME faces must have shape (triangle_count, 3)")
    if np.any(faces < 0) or np.any(faces >= count):
        raise ValueError("FLAME faces contain out-of-range vertex indices")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("FLAME faces must reference three distinct vertices")
    return FlameModelData(
        template, directions, pose_directions, regressor, weights, parents, faces.astype(np.int32)
    )


def _float_array(value: object, name: str) -> NDArray[np.float32]:
    if isinstance(value, _LegacyCh):
        value = getattr(value, "x", None)
    if isinstance(value, _LegacyCSC):
        value = _dense_csc(value)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError(f"FLAME {name} must be a numeric array or a raw Chumpy leaf")
    array = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"FLAME {name} must contain only finite values")
    return array


def _integer_array(value: object, name: str) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu":
        raise ValueError(f"FLAME {name} must be an integer array")
    return np.array(value, dtype=np.int64, copy=True)


def _dense_csc(value: _LegacyCSC) -> NDArray[np.float64]:
    shape = getattr(value, "_shape", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("Invalid FLAME sparse regressor shape")
    indices = _integer_array(getattr(value, "indices", None), "sparse indices")
    pointers = _integer_array(getattr(value, "indptr", None), "sparse indptr")
    data = getattr(value, "data", None)
    if (
        indices.ndim != 1
        or not isinstance(data, np.ndarray)
        or data.ndim != 1
        or data.shape != indices.shape
        or data.dtype.kind not in "fiu"
    ):
        raise ValueError("Invalid FLAME sparse regressor data")
    if (
        pointers.shape != (shape[1] + 1,)
        or pointers[0] != 0
        or pointers[-1] != data.size
        or np.any(np.diff(pointers) < 0)
    ):
        raise ValueError("Invalid FLAME sparse regressor column pointers")
    if np.any(indices < 0) or np.any(indices >= shape[0]):
        raise ValueError("Invalid FLAME sparse regressor row indices")
    dense = np.zeros(shape, dtype=np.float64)
    columns = np.repeat(np.arange(shape[1]), np.diff(pointers))
    np.add.at(dense, (indices, columns), data)
    return dense
