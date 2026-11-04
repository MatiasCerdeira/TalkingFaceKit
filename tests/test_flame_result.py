from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from talkingfacekit.flame import FlameResult
from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.video import VideoSource


def make_result(*, all_missing: bool = False, all_estimated: bool = False) -> FlameResult:
    mask = np.array([True, False, False, True], dtype=np.bool_)
    if all_missing:
        mask[:] = False
    if all_estimated:
        mask[:] = True

    def parameters(size: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        array = np.arange(4 * size, dtype=np.float32).reshape(4, size)
        array[~mask] = np.nan
        return array

    transforms = np.repeat(np.eye(3)[None], 4, axis=0)
    transforms[~mask] = np.nan
    return FlameResult(
        source=VideoSource(
            Path("/unavailable/original-video.mp4"), VideoMetadata(640, 480, 30, None, True)
        ),
        source_sha256="a" * 64,
        start_seconds=20.0,
        end_seconds=25.0,
        model_variant="synthetic-basis",
        frame_indices=np.array([702, 703, 704, 705], dtype=np.int64),
        timestamps_seconds=np.array([23.4, 23.433333, 23.466667, 23.5], dtype=np.float64),
        estimated=mask,
        missing_reasons=tuple("" if found else "face_landmarks_unavailable" for found in mask),
        shape=parameters(4),
        expression=parameters(7),
        global_pose=parameters(3),
        jaw_pose=parameters(3),
        extras={"test_camera": parameters(3), "test_transform": transforms},
        extra_conventions={
            "test_camera": "Synthetic float32 camera",
            "test_transform": "Source-to-crop float64 matrix",
        },
        provenance={"backend": "synthetic", "checkpoint_sha256": "b" * 64, "version": "1.2.3"},
    )


def test_complete_result_preserves_absences_and_freezes_metadata() -> None:
    result = make_result()
    assert result.frame_count == 4
    assert result.estimated_count == 2
    assert np.all(np.isnan(result.expression[1:3]))
    with pytest.raises(TypeError):
        cast(Any, result.provenance)["backend"] = "changed"
    copied = dict(result.extra_conventions)
    replaced = replace(result, extra_conventions=copied)
    copied.clear()
    assert replaced.extra_conventions == result.extra_conventions


@pytest.mark.parametrize(
    "problem",
    [
        "indices_dtype",
        "indices_order",
        "indices_negative",
        "timestamps_dtype",
        "timestamps_order",
        "timestamps_nonfinite",
        "mask_dtype",
        "mask_length",
        "empty",
        "outside_start",
        "outside_end",
        "interval",
        "reasons_length",
        "reason_on_estimate",
        "reason_missing",
        "basis",
        "source_hash",
        "provenance",
        "checkpoint_hash",
        "extra_keys",
        "extra_dtype",
        "extra_rows",
        "extra_missing",
    ],
)
def test_rejects_inconsistent_result_contracts(problem: str) -> None:
    result = make_result()
    changes: dict[str, Any] = {}
    if problem == "indices_dtype":
        changes["frame_indices"] = result.frame_indices.astype(np.int32)
    elif problem == "indices_order":
        changes["frame_indices"] = result.frame_indices[::-1]
    elif problem == "indices_negative":
        result.frame_indices[0] = -1
    elif problem == "timestamps_dtype":
        changes["timestamps_seconds"] = result.timestamps_seconds.astype(np.float32)
    elif problem == "timestamps_order":
        result.timestamps_seconds[1] = result.timestamps_seconds[0]
    elif problem == "timestamps_nonfinite":
        result.timestamps_seconds[1] = np.inf
    elif problem == "mask_dtype":
        changes["estimated"] = result.estimated.astype(np.int64)
    elif problem == "mask_length":
        changes["estimated"] = result.estimated[:1]
    elif problem == "empty":
        changes["frame_indices"] = result.frame_indices[:0]
    elif problem == "outside_start":
        changes["start_seconds"] = 23.41
    elif problem == "outside_end":
        changes["end_seconds"] = 23.5
    elif problem == "interval":
        changes["end_seconds"] = 19
    elif problem == "reasons_length":
        changes["missing_reasons"] = ("",)
    elif problem == "reason_on_estimate":
        changes["missing_reasons"] = ("unexpected", *result.missing_reasons[1:])
    elif problem == "reason_missing":
        changes["missing_reasons"] = ("", "", "", "")
    elif problem == "basis":
        changes["model_variant"] = ""
    elif problem == "source_hash":
        changes["source_sha256"] = "not a hash"
    elif problem == "provenance":
        changes["provenance"] = {"backend": "synthetic"}
    elif problem == "checkpoint_hash":
        changes["provenance"] = {"backend": "synthetic", "checkpoint_sha256": "BAD"}
    elif problem == "extra_keys":
        changes["extra_conventions"] = {}
    elif problem == "extra_dtype":
        changes["extras"] = {**result.extras, "test_camera": np.zeros((4, 3), dtype=np.int64)}
    elif problem == "extra_rows":
        changes["extras"] = {**result.extras, "test_camera": np.zeros((3, 3), dtype=np.float32)}
    else:
        result.extras["test_camera"][1] = 0
    with pytest.raises((TypeError, ValueError)):
        replace(result, **changes)


@pytest.mark.parametrize("field", ["shape", "expression", "global_pose", "jaw_pose"])
@pytest.mark.parametrize(
    "problem", ["dtype", "components", "nonfinite_estimate", "fabricated_absence"]
)
def test_parameter_arrays_require_finite_estimates_and_nan_absences(
    field: str, problem: str
) -> None:
    result = make_result()
    values = getattr(result, field).copy()
    if problem == "dtype":
        values = values.astype(np.float64)
    elif problem == "components":
        values = values[:, :0]
    elif problem == "nonfinite_estimate":
        values[0, 0] = np.nan
    else:
        values[1, 0] = 0
    with pytest.raises((TypeError, ValueError)):
        replace(result, **{field: values})


def test_all_missing_and_open_ended_results_remain_representable() -> None:
    result = replace(make_result(all_missing=True), end_seconds=None)
    assert result.frame_count == 4
    assert result.estimated_count == 0
    assert np.all(np.isnan(result.shape))
