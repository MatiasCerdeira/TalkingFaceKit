"""Opt-in FLAME geometry evaluation using local official assets and saved parameters."""

import os

import numpy as np
import pytest

from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.flame import FlameDecoder


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "TALKINGFACEKIT_FLAME_MODEL",
            "TALKINGFACEKIT_FLAME_RESULT",
            "TALKINGFACEKIT_FLAME_SMIRK_EYELIDS",
        )
    ),
    reason="set TALKINGFACEKIT_FLAME_MODEL, FLAME_RESULT and FLAME_SMIRK_EYELIDS for real geometry",
)
def test_real_flame_preserves_saved_timeline_and_produces_finite_geometry() -> None:
    result = FlameResult.load(os.environ["TALKINGFACEKIT_FLAME_RESULT"])
    decoder = FlameDecoder(
        os.environ["TALKINGFACEKIT_FLAME_MODEL"],
        smirk_eyelids=os.environ["TALKINGFACEKIT_FLAME_SMIRK_EYELIDS"],
    )
    mesh = decoder.decode(result)
    assert mesh.vertex_count == 5023
    assert mesh.triangle_count == 9976
    np.testing.assert_array_equal(mesh.frame_indices, result.frame_indices)
    np.testing.assert_array_equal(mesh.timestamps_seconds, result.timestamps_seconds)
    np.testing.assert_array_equal(mesh.detected, result.estimated)
    assert np.all(np.isfinite(mesh.vertices[mesh.detected]))
    assert np.all(np.isnan(mesh.vertices[~mesh.detected]))
