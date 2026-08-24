"""Persistence boundary for backend-independent facial-landmark tracks."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.tracking.landmarks import FaceLandmarkTrack

_SCHEMA_VERSION = 1
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "tracker_name",
        "tracker_version",
        "tracker_version_present",
        "topology",
        "coordinate_system",
        "frame_indices",
        "timestamps_seconds",
        "landmarks",
        "detected",
    }
)


def save_landmark_track(
    track: FaceLandmarkTrack,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a complete landmark track as a compressed, versioned NPZ file.

    The archive preserves the exact NumPy dtypes, source frame indices, source timestamps,
    detection mask, tracker provenance, topology, and coordinate-system description. Data is
    written to a temporary file in the destination directory and moved into place only after the
    archive has been completed.

    Parameters
    ----------
    track
        Validated landmark track to persist.
    output_path
        Destination path ending in ``.npz``. Its parent directory must already exist.
    overwrite
        Whether an existing regular file may be replaced after the new archive is complete.

    Returns
    -------
    pathlib.Path
        Destination path supplied by the caller, normalized as a ``Path``.

    Raises
    ------
    FileNotFoundError
        If the destination directory does not exist.
    IsADirectoryError
        If ``output_path`` refers to a directory.
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    ValueError
        If the destination does not use the ``.npz`` extension.
    """
    path = _validate_output_path(output_path, overwrite=overwrite)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int64),
                tracker_name=np.asarray(track.tracker_name),
                tracker_version=np.asarray(track.tracker_version or ""),
                tracker_version_present=np.asarray(
                    track.tracker_version is not None, dtype=np.bool_
                ),
                topology=np.asarray(track.topology),
                coordinate_system=np.asarray(track.coordinate_system),
                frame_indices=track.frame_indices,
                timestamps_seconds=track.timestamps_seconds,
                landmarks=track.landmarks,
                detected=track.detected,
            )

        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return path


def load_landmark_track(input_path: str | Path) -> FaceLandmarkTrack:
    """Load and validate a landmark track from a TalkingFaceKit NPZ archive.

    Parameters
    ----------
    input_path
        Path to an archive previously written by :func:`save_landmark_track`.

    Returns
    -------
    FaceLandmarkTrack
        Validated in-memory landmark track. Arrays do not retain a reference to the archive.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    IsADirectoryError
        If ``input_path`` refers to a directory.
    ValueError
        If the path is not a regular ``.npz`` file, required fields are missing, or the schema
        version or stored data violates the landmark contract.
    """
    path = _validate_input_path(input_path)

    with np.load(path, allow_pickle=False) as archive:
        missing_keys = _REQUIRED_KEYS.difference(archive.files)
        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"Landmark archive is missing required fields: {formatted_keys}")

        schema_version = _read_integer_scalar(archive, "schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported landmark archive schema version {schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )

        tracker_version_present = _read_boolean_scalar(archive, "tracker_version_present")
        tracker_version_value = _read_text_scalar(archive, "tracker_version")
        tracker_version = tracker_version_value if tracker_version_present else None

        frame_indices = cast(NDArray[np.int64], archive["frame_indices"].copy())
        timestamps_seconds = cast(NDArray[np.float64], archive["timestamps_seconds"].copy())
        landmarks = cast(NDArray[np.float32], archive["landmarks"].copy())
        detected = cast(NDArray[np.bool_], archive["detected"].copy())

        return FaceLandmarkTrack(
            tracker_name=_read_text_scalar(archive, "tracker_name"),
            tracker_version=tracker_version,
            topology=_read_text_scalar(archive, "topology"),
            coordinate_system=_read_text_scalar(archive, "coordinate_system"),
            frame_indices=frame_indices,
            timestamps_seconds=timestamps_seconds,
            landmarks=landmarks,
            detected=detected,
        )


def _validate_output_path(output_path: str | Path, *, overwrite: bool) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Landmark output path must end in .npz: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Landmark output directory not found: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"Landmark output parent is not a directory: {path.parent}")
    if path.is_dir():
        raise IsADirectoryError(f"Landmark output path is a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Landmark output already exists: {path}")
    return path


def _validate_input_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark archive not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Landmark archive path is a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Landmark archive path is not a regular file: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Landmark archive path must end in .npz: {path}")
    return path


def _read_text_scalar(archive: np.lib.npyio.NpzFile, field_name: str) -> str:
    value = archive[field_name]
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Landmark archive field {field_name} must be a text scalar")
    return str(value.item())


def _read_integer_scalar(archive: np.lib.npyio.NpzFile, field_name: str) -> int:
    value = archive[field_name]
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Landmark archive field {field_name} must be an integer scalar")
    return int(value.item())


def _read_boolean_scalar(archive: np.lib.npyio.NpzFile, field_name: str) -> bool:
    value = archive[field_name]
    if value.shape != () or value.dtype != np.dtype(np.bool_):
        raise ValueError(f"Landmark archive field {field_name} must be a boolean scalar")
    return bool(value.item())
