"""Explicit local FLAME2020 geometry evaluation from saved facial parameters.

This NumPy integration needs neither SMIRK inference, Chumpy, SciPy, PyTorch nor a renderer.
Model assets remain external and subject to their own terms; none are downloaded or bundled.
"""

from collections.abc import Mapping
from hashlib import file_digest
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations._flame_assets import load_flame_model
from talkingfacekit.integrations._flame_model import evaluate_flame
from talkingfacekit.mesh import FaceMeshTrack


class FlameDecoder:
    """Load a local generic FLAME2020 model once and evaluate complete parameter intervals.

    Parameters
    ----------
    model
        Explicit local ``generic_model.pkl`` from the official FLAME2020 download. Only
        numeric arrays, raw Chumpy leaf storage and CSC regressors are supported; no legacy
        computation graph is evaluated. Use trusted model files, not arbitrary pickles.
    smirk_eyelids
        Optional directory containing upstream ``l_eyelid.npy`` and ``r_eyelid.npy``.
        Both are required to decode SMIRK results faithfully. They are never silently omitted.

    Attributes
    ----------
    provenance
        Read-only text mapping of model basis, numerical implementation, loaded asset hashes
        and explicit neutral neck/eye assumptions. No geometry or model files are distributed.

    Notes
    -----
    Construction loads assets but never accesses the network. This first decoder supports the
    declared ``flame-2020-generic`` basis only; structural validation cannot identify a model's
    year or gender from a filename. Use the matching official asset. Parameters remain raw per
    frame. Geometry retains native asset units and axes, including global and jaw rotations;
    crop camera/projection is a separate operation. This is CPU inference without gradients.
    """

    def __init__(self, model: str | Path, *, smirk_eyelids: str | Path | None = None) -> None:
        model_path = Path(model).expanduser()
        self._model = load_flame_model(model_path)
        provenance = {
            "model_variant": "flame-2020-generic",
            "model_sha256": _sha256(model_path),
            "implementation": "talkingfacekit-flame-numpy-v1",
            "neck_pose": "neutral, not estimated",
            "eye_pose": "neutral, not estimated",
        }
        self._eyelids: NDArray[np.float32] | None = None
        if smirk_eyelids is not None:
            directory = Path(smirk_eyelids).expanduser()
            values: list[NDArray[np.float32]] = []
            for name in ("l_eyelid", "r_eyelid"):
                path = directory / f"{name}.npy"
                array = np.load(path, allow_pickle=False)
                if (
                    not isinstance(array, np.ndarray)
                    or array.dtype.kind != "f"
                    or array.shape != self._model.template.shape
                    or not np.all(np.isfinite(array))
                ):
                    raise ValueError(
                        f"{name} must be a finite floating-point (vertex_count, 3) array"
                    )
                values.append(np.array(array, dtype=np.float32, copy=True))
                provenance[f"{name}_sha256"] = _sha256(path)
            self._eyelids = np.stack(values)
        self.provenance: Mapping[str, str] = MappingProxyType(provenance)

    def decode(self, result: FlameResult, *, batch_size: int = 32) -> FaceMeshTrack:
        """Generate an aligned triangular mesh track without inference or source-video access.

        Parameters
        ----------
        result
            Complete interval in ``flame-2020-generic`` basis, with up to 300 shape components
            and the model's available expressions. SMIRK requires its ``smirk_eyelids`` extra
            and matching decoder assets. Camera and source-to-crop extras do not deform 3D.
        batch_size
            Positive number of estimated frames per numerical batch, bounding intermediate
            memory. The returned mesh materializes all vertices of the interval.

        Returns
        -------
        FaceMeshTrack
            Float32 vertices in native FLAME coordinates/units, shared int32 triangles and
            copied source timeline/mask. Absent rows remain entirely NaN. Per-frame shape,
            expression and head/jaw pose are preserved, with neutral neck and eye rotation.
            SMIRK eyelid offsets are applied after skinning, as in its reference decoder.

        Raises
        ------
        ValueError
            If basis, coefficient dimensions, batch size or eyelid evidence are incompatible.
            No version conversion, eye/neck inference, smoothing or interpolation is implicit.
        """
        if not isinstance(result, FlameResult):
            raise TypeError("result must be a FlameResult")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if result.model_variant != "flame-2020-generic":
            raise ValueError("FLAME decoder requires model_variant='flame-2020-generic'")
        if (
            result.shape.shape[1] > 300
            or result.expression.shape[1] > self._model.directions.shape[2] - 300
        ):
            raise ValueError("FLAME coefficient dimensions exceed the loaded model basis")
        eyelids = result.extras.get("smirk_eyelids")
        if result.provenance["backend"] == "smirk" and eyelids is None:
            raise ValueError("SMIRK results require the smirk_eyelids extra")
        if eyelids is not None:
            if self._eyelids is None:
                raise ValueError("SMIRK eyelid coefficients require explicit smirk_eyelids assets")
            if eyelids.shape != (result.frame_count, 2) or eyelids.dtype != np.dtype(np.float32):
                raise ValueError("smirk_eyelids must be float32 with shape (frame_count, 2)")
            if np.any((eyelids[result.estimated] < 0) | (eyelids[result.estimated] > 1)):
                raise ValueError("smirk_eyelids estimates must be in [0, 1]")
        vertices = np.full(
            (result.frame_count, self._model.template.shape[0], 3), np.nan, dtype=np.float32
        )
        available = np.flatnonzero(result.estimated)
        for offset in range(0, available.size, batch_size):
            rows = available[offset : offset + batch_size]
            batch = evaluate_flame(
                self._model,
                result.shape[rows],
                result.expression[rows],
                result.global_pose[rows],
                result.jaw_pose[rows],
            )
            if eyelids is not None and self._eyelids is not None:
                # Preserve upstream's right-then-left addition after posed skinning.
                batch += self._eyelids[1][None] * eyelids[rows, 1, None, None]
                batch += self._eyelids[0][None] * eyelids[rows, 0, None, None]
            vertices[rows] = batch
        extension = "SMIRK eyelid offsets applied" if eyelids is not None else "no eyelid extension"
        return FaceMeshTrack(
            topology=f"flame-2020-generic-sha256-{self.provenance['model_sha256']}",
            coordinate_system=(
                "FLAME2020 native asset coordinates and units; global head/jaw pose applied; "
                f"neck/eye rotations neutral; {extension}; no camera projection"
            ),
            frame_indices=result.frame_indices.copy(),
            timestamps_seconds=result.timestamps_seconds.copy(),
            vertices=vertices,
            triangles=self._model.triangles.copy(),
            detected=result.estimated.copy(),
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
