"""Framework-independent values for estimated FLAME facial parameters.

Construction is pure. Save/load convenience methods delegate to the persistence boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.video import VideoSource

FloatArray = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class FlameParameters:
    """Represent one facial estimate in an explicitly identified FLAME parameter basis.

    Attributes
    ----------
    model_variant
        Model/basis identifier, such as ``flame-2020-generic``. Coefficients from different
        bases must not be treated as interchangeable solely because their sizes match.
    shape, expression
        Nonempty finite float32 coefficient vectors. Their lengths are backend/model specific.
        Shape is the original per-image estimate, without temporal aggregation or smoothing.
    global_pose, jaw_pose
        Finite float32 vectors of shape ``(3,)`` encoding axis-angle rotations in radians
        under the identified model's coordinate convention.

    Notes
    -----
    Camera and estimator-specific eyelid deformations belong to the integration's result.
    No eye/neck estimate or metric reconstruction is implied. Arrays are retained without
    copying and must be treated as read-only. Construction performs no I/O or model execution.
    """

    model_variant: str
    shape: NDArray[np.float32]
    expression: NDArray[np.float32]
    global_pose: NDArray[np.float32]
    jaw_pose: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate parameter dimensions, dtype, basis identity and finite values."""
        if not isinstance(self.model_variant, str) or not self.model_variant.strip():
            raise ValueError("model_variant must be a nonempty string")
        for name in ("shape", "expression", "global_pose", "jaw_pose"):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float32):
                raise TypeError(f"{name} must be a float32 NumPy array")
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a nonempty finite vector")
            if name in {"global_pose", "jaw_pose"} and values.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")


@dataclass(frozen=True, slots=True, eq=False)
class FlameResult:
    """Retain a complete source interval's facial estimates and explicit absences.

    Attributes
    ----------
    source, source_sha256
        Original video identity/metadata and SHA-256 of its bytes at extraction time.
        Construction and loading do not open or verify the original video.
    start_seconds, end_seconds
        Requested half-open interval on the source timeline; ``None`` ends at EOF.
    model_variant
        Explicit FLAME parameter basis, with the semantics of :class:`FlameParameters`.
    frame_indices, timestamps_seconds
        Nonempty, strictly increasing source indices (int64) and PTS in seconds (float64),
        both shaped ``(N,)``. Every decoded frame is retained without FPS conversion.
    estimated, missing_reasons
        Boolean mask ``(N,)`` and one reason per row: empty for an estimate, a nonempty code
        for an absence. An estimate is not a guarantee of geometric or training quality.
    shape, expression
        Float32 arrays ``(N, S)`` and ``(N, E)`` with positive model-specific dimensions.
        Shape remains the raw per-image estimate, without identity aggregation.
    global_pose, jaw_pose
        Float32 arrays ``(N, 3)`` containing axis-angle rotations in radians.
    extras, extra_conventions
        Equally keyed mappings of named float32/float64 arrays and descriptions of their
        units, axes and coordinate conventions. Every array's first axis is the same N.
        Extensions retain backend-specific meaning; they are not interchangeable by shape.
    provenance
        Text metadata including ``backend`` and ``checkpoint_sha256``. Integrations also
        record preprocessing/model identities, runtime versions and execution settings.

    Notes
    -----
    All parameter and extra arrays contain finite values in estimated rows and only NaN in
    absent rows. NaN is a storage sentinel, never an estimated parameter. No images, framework
    objects, temporal interpolation, segment selection or geometry are retained. Arrays are
    stored without copying and must be treated as read-only; mappings are copied and frozen.
    The complete result materializes compact numerical outputs, not video pixels.
    """

    source: VideoSource
    source_sha256: str
    start_seconds: float
    end_seconds: float | None
    model_variant: str
    frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
    estimated: NDArray[np.bool_]
    missing_reasons: tuple[str, ...]
    shape: NDArray[np.float32]
    expression: NDArray[np.float32]
    global_pose: NDArray[np.float32]
    jaw_pose: NDArray[np.float32]
    extras: Mapping[str, FloatArray]
    extra_conventions: Mapping[str, str]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate timeline, parameter alignment, absence sentinels and provenance."""
        if not isinstance(self.source, VideoSource):
            raise TypeError("source must be a VideoSource")
        _validate_sha256(self.source_sha256, "source_sha256")
        if (
            isinstance(self.start_seconds, bool)
            or not math.isfinite(self.start_seconds)
            or self.start_seconds < 0
        ):
            raise ValueError("start_seconds must be finite and non-negative")
        if self.end_seconds is not None and (
            isinstance(self.end_seconds, bool)
            or not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must be finite and greater than start_seconds")
        if not isinstance(self.model_variant, str) or not self.model_variant.strip():
            raise ValueError("model_variant must be a nonempty string")
        for name, dtype in (
            ("frame_indices", np.int64),
            ("timestamps_seconds", np.float64),
            ("estimated", np.bool_),
        ):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.dtype(dtype):
                raise TypeError(f"{name} has an invalid dtype")
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional array")
        count = self.frame_count
        if self.timestamps_seconds.shape != (count,) or self.estimated.shape != (count,):
            raise ValueError("timeline and estimated mask must have matching lengths")
        if np.any(self.frame_indices < 0) or np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be non-negative and strictly increasing")
        times = self.timestamps_seconds
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps_seconds must be finite and strictly increasing")
        if np.any(times < self.start_seconds) or (
            self.end_seconds is not None and np.any(times >= self.end_seconds)
        ):
            raise ValueError("timestamps_seconds must stay inside the requested interval")
        if not isinstance(self.missing_reasons, tuple) or len(self.missing_reasons) != count:
            raise ValueError("missing_reasons must be a tuple with one reason per frame")
        for available, reason in zip(self.estimated, self.missing_reasons, strict=True):
            if not isinstance(reason, str) or (reason != "" if available else not reason.strip()):
                raise ValueError(
                    "estimated rows require empty reasons; absent rows require reason codes"
                )
        for name in ("shape", "expression", "global_pose", "jaw_pose"):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float32):
                raise TypeError(f"{name} must have dtype float32")
            if values.ndim != 2 or values.shape[0] != count or values.shape[1] == 0:
                raise ValueError(f"{name} must have shape (frame_count, positive_component_count)")
            if name in {"global_pose", "jaw_pose"} and values.shape[1] != 3:
                raise ValueError(f"{name} must have shape (frame_count, 3)")
            _validate_estimated_rows(values, self.estimated, name)
        conventions = _text_mapping(self.extra_conventions, "extra_conventions")
        if set(self.extras) != set(conventions):
            raise ValueError("extras and extra_conventions must have exactly matching keys")
        for name, values in self.extras.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("extra array names must be identifiers")
            if not isinstance(values, np.ndarray) or values.dtype not in (
                np.dtype(np.float32),
                np.dtype(np.float64),
            ):
                raise TypeError(f"extra {name} must be a float32 or float64 NumPy array")
            if (
                values.ndim < 2
                or values.shape[0] != count
                or any(size == 0 for size in values.shape[1:])
            ):
                raise ValueError(
                    f"extra {name} must have shape (frame_count, ...), with nonempty components"
                )
            _validate_estimated_rows(values, self.estimated, name)
        provenance = _text_mapping(self.provenance, "provenance")
        if "backend" not in provenance or "checkpoint_sha256" not in provenance:
            raise ValueError("provenance must identify backend and checkpoint_sha256")
        for name, value in provenance.items():
            if name.endswith("_sha256"):
                _validate_sha256(value, name)
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))
        object.__setattr__(self, "extra_conventions", MappingProxyType(conventions))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def frame_count(self) -> int:
        """Number of decoded frames, including absences."""
        return int(self.frame_indices.size)

    @property
    def estimated_count(self) -> int:
        """Number of available estimates, independently of quality eligibility."""
        return int(np.count_nonzero(self.estimated))

    def save(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        """Save compact outputs and metadata as a versioned compressed NPZ archive.

        Parameters
        ----------
        output_path
            Destination ending in ``.npz``. Parent directories are created as needed.
        overwrite
            Whether an existing archive may be atomically replaced; false by default.

        Returns
        -------
        pathlib.Path
            Completed archive path. Source media and model assets are never opened.
        """
        from talkingfacekit.io.flame import save_flame_result

        return save_flame_result(self, output_path, overwrite=overwrite)

    @classmethod
    def load(cls, input_path: str | Path) -> FlameResult:
        """Load and validate a saved result without source media or optional frameworks.

        Parameters
        ----------
        input_path
            Existing TalkingFaceKit FLAME ``.npz`` archive.

        Returns
        -------
        FlameResult
            Owned numerical arrays, source metadata and stored provenance. The archive's
            schema and invariants are validated, without executing any estimator.
        """
        from talkingfacekit.io.flame import load_flame_result

        return load_flame_result(input_path)


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _text_mapping(values: Mapping[str, str], name: str) -> dict[str, str]:
    copied = dict(values)
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in copied.items()
    ):
        raise ValueError(f"{name} must map nonempty strings to nonempty strings")
    return copied


def _validate_estimated_rows(values: FloatArray, estimated: NDArray[np.bool_], name: str) -> None:
    if not np.all(np.isfinite(values[estimated])):
        raise ValueError(f"estimated {name} rows must be finite")
    if not np.all(np.isnan(values[~estimated])):
        raise ValueError(f"absent {name} rows must contain only NaN")
