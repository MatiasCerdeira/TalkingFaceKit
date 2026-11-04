"""Versioned, pickle-free NPZ persistence for completed facial-parameter results."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.flame import FlameResult, FloatArray
from talkingfacekit.metadata import VideoMetadata
from talkingfacekit.video import VideoSource

_SCHEMA_VERSION = 1
_ARRAY_FIELDS = (
    "frame_indices",
    "timestamps_seconds",
    "estimated",
    "shape",
    "expression",
    "global_pose",
    "jaw_pose",
    "missing_reasons",
)
_METADATA_FIELDS = {
    "schema_version",
    "kind",
    "source",
    "source_metadata",
    "source_sha256",
    "start_seconds",
    "end_seconds",
    "model_variant",
    "extra_conventions",
    "provenance",
}


def save_flame_result(
    result: FlameResult, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically save numerical arrays and JSON metadata without reading source assets.

    Parameters
    ----------
    result
        Completed, validated result. Arrays must not be mutated during saving.
    output_path
        Destination ending in ``.npz``. Parent directories are created. Source media cannot
        be overwritten. No original video, checkpoint or framework is needed for this call.
    overwrite
        Whether an existing file may be atomically replaced after serialization succeeds.

    Returns
    -------
    pathlib.Path
        Completed archive path. Existing files are refused by default, including races.
    """
    if not isinstance(result, FlameResult):
        raise TypeError("result must be a FlameResult")
    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("FLAME output must end in .npz")
    if path.is_dir():
        raise IsADirectoryError(f"FLAME output is a directory: {path}")
    if path.resolve() == result.source.path.resolve():
        raise ValueError("FLAME output must differ from source video")
    if path.exists() and not overwrite:
        raise FileExistsError(f"FLAME output already exists: {path}")
    document = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "talkingfacekit.flame",
        "source": str(result.source.path),
        "source_metadata": asdict(result.source.metadata),
        "source_sha256": result.source_sha256,
        "start_seconds": result.start_seconds,
        "end_seconds": result.end_seconds,
        "model_variant": result.model_variant,
        "extra_conventions": dict(result.extra_conventions),
        "provenance": dict(result.provenance),
    }
    arrays = {name: np.asarray(getattr(result, name)) for name in _ARRAY_FIELDS}
    arrays["metadata"] = np.asarray(json.dumps(document, allow_nan=False, ensure_ascii=False))
    arrays.update({f"extra__{name}": values for name, values in result.extras.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        if overwrite:
            temporary_path.replace(path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_flame_result(input_path: str | Path) -> FlameResult:
    """Read schema-1 output, rejecting malformed arrays and inconsistent metadata.

    Parameters
    ----------
    input_path
        Existing ``.npz`` archive written by :func:`save_flame_result`.

    Returns
    -------
    FlameResult
        Validated in-memory data, independent from the closed archive. Loading uses
        ``allow_pickle=False`` and never imports inference frameworks or opens source assets.

    Raises
    ------
    ValueError
        If the extension, schema, metadata, field set or numerical contracts are invalid.
    """
    path = Path(input_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("FLAME archive must end in .npz")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("FLAME archive is missing metadata")
        raw_metadata = archive["metadata"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
            raise ValueError("FLAME archive metadata must be a Unicode JSON scalar")
        document = _mapping(json.loads(str(raw_metadata.item())), "metadata")
        schema = document.get("schema_version")
        if type(schema) is not int or schema != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported FLAME archive schema version: {schema}")
        if set(document) != _METADATA_FIELDS or document["kind"] != "talkingfacekit.flame":
            raise ValueError("Invalid FLAME archive metadata fields or kind")
        conventions = _strings(document["extra_conventions"], "extra_conventions")
        required = {*_ARRAY_FIELDS, "metadata", *(f"extra__{name}" for name in conventions)}
        if set(archive.files) != required or len(archive.files) != len(required):
            raise ValueError("FLAME archive has missing, unexpected or duplicate array fields")
        metadata = _mapping(document["source_metadata"], "source_metadata")
        if set(metadata) != {
            "width",
            "height",
            "average_fps",
            "stream_duration_seconds",
            "has_audio",
        }:
            raise ValueError("Invalid source_metadata fields")
        width, height, has_audio = metadata["width"], metadata["height"], metadata["has_audio"]
        if type(width) is not int or type(height) is not int or not isinstance(has_audio, bool):
            raise ValueError("Invalid source_metadata dimensions or has_audio")
        source = VideoSource(
            Path(_text(document["source"], "source")),
            VideoMetadata(
                width,
                height,
                _optional_number(metadata["average_fps"], "average_fps"),
                _optional_number(metadata["stream_duration_seconds"], "stream_duration_seconds"),
                has_audio,
            ),
        )
        reasons = archive["missing_reasons"]
        if reasons.ndim != 1 or reasons.dtype.kind != "U":
            raise ValueError("missing_reasons must be a Unicode vector")
        return FlameResult(
            source=source,
            source_sha256=_text(document["source_sha256"], "source_sha256"),
            start_seconds=_number(document["start_seconds"], "start_seconds"),
            end_seconds=_optional_number(document["end_seconds"], "end_seconds"),
            model_variant=_text(document["model_variant"], "model_variant"),
            frame_indices=cast(NDArray[np.int64], archive["frame_indices"].copy()),
            timestamps_seconds=cast(NDArray[np.float64], archive["timestamps_seconds"].copy()),
            estimated=cast(NDArray[np.bool_], archive["estimated"].copy()),
            missing_reasons=tuple(str(reason) for reason in reasons),
            shape=cast(NDArray[np.float32], archive["shape"].copy()),
            expression=cast(NDArray[np.float32], archive["expression"].copy()),
            global_pose=cast(NDArray[np.float32], archive["global_pose"].copy()),
            jaw_pose=cast(NDArray[np.float32], archive["jaw_pose"].copy()),
            extras={
                name: cast(FloatArray, archive[f"extra__{name}"].copy()) for name in conventions
            },
            extra_conventions=conventions,
            provenance=_strings(document["provenance"], "provenance"),
        )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _strings(value: object, name: str) -> dict[str, str]:
    return {key: _text(item, name) for key, item in _mapping(value, name).items()}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)
