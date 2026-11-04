from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from talkingfacekit.flame import FlameParameters


def parameters() -> FlameParameters:
    return FlameParameters(
        "test-basis",
        np.zeros(4, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )


def test_parameter_dimensions_follow_the_basis_without_loading_a_framework() -> None:
    result = parameters()
    assert result.shape.shape == (4,)
    assert result.expression.shape == (7,)
    assert result.model_variant == "test-basis"


@pytest.mark.parametrize("field", ["shape", "expression", "global_pose", "jaw_pose"])
@pytest.mark.parametrize("invalid", ["dtype", "empty", "nonfinite", "dimensions"])
def test_rejects_invalid_parameter_arrays(field: str, invalid: str) -> None:
    value = np.zeros(3, dtype=np.float32)
    if invalid == "dtype":
        value = value.astype(np.float64)
    elif invalid == "empty":
        value = value[:0]
    elif invalid == "nonfinite":
        value[0] = np.nan
    else:
        value = value.reshape(1, 3)
    with pytest.raises((TypeError, ValueError)):
        replace(parameters(), **{field: cast(Any, value)})


@pytest.mark.parametrize("field", ["global_pose", "jaw_pose"])
def test_pose_requires_three_axis_angle_components(field: str) -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        replace(parameters(), **{field: cast(Any, np.zeros(6, dtype=np.float32))})


def test_basis_identity_is_required() -> None:
    with pytest.raises(ValueError, match="model_variant"):
        replace(parameters(), model_variant=" ")
