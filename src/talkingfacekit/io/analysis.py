"""Atomic JSON persistence for complete analysis results and collection summaries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from talkingfacekit.analysis import VideoAnalysisResult
from talkingfacekit.io.provenance import code_revision

if TYPE_CHECKING:
    from talkingfacekit.collection import CollectionAnalysisResult


def save_analysis_result(
    result: VideoAnalysisResult, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Save full evidence and selection using the additive version-1 analysis JSON schema.

    Parameters
    ----------
    result
        Completed result. The original video is read for SHA-256 verification metadata.
    output_path
        JSON destination. Parent directories are created; source media cannot be overwritten.
    overwrite
        Whether a prior output may be atomically replaced after serialization succeeds.

    Returns
    -------
    pathlib.Path
        Written JSON path, compatible with the existing single-video CLI output.
    """
    path = _validate_output_path(output_path, overwrite, (result.source.path,))
    with result.source.path.open("rb") as source_stream:
        source_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()
    evidence = result.active_speaker
    document: dict[str, object] = {
        "schema_version": 1,
        "source": str(result.source.path.resolve()),
        "source_metadata": asdict(result.source.metadata),
        "source_sha256": source_sha256,
        "package_version": version("talkingfacekit"),
        "code_revision": code_revision(),
        "interval": [result.start_seconds, result.end_seconds],
        "face_observations": [asdict(item) for item in evidence.face_observations],
        "speech_intervals": [asdict(item) for item in evidence.speech_intervals],
        "score_windows": [
            {
                "source_start_seconds": item.source_start_seconds,
                "source_end_seconds": item.source_end_seconds,
                "raw_scores": dict(item.raw_scores),
            }
            for item in evidence.score_windows
        ],
        "preliminary_segments": [asdict(item) for item in evidence.preliminary_segments],
        "candidate_runs": [asdict(item) for item in evidence.candidate_runs],
        "candidate_policy": asdict(evidence.candidate_policy),
        "shot_boundaries": [asdict(item) for item in evidence.shot_boundaries],
        "audio_timeline_repairs": [asdict(item) for item in evidence.audio_timeline_repairs],
        "issues": list(evidence.issues),
        "provenance": asdict(evidence.provenance),
        "synchronization": None
        if result.synchronization is None
        else asdict(result.synchronization),
        "visual_filter": None if result.visual_filter is None else asdict(result.visual_filter),
        "selected_intervals": [
            {
                "source_start_seconds": item.start_seconds,
                "source_end_seconds": item.end_seconds,
                "face_id": item.face_id,
                "status": item.status,
                "reason_codes": item.reason_codes,
            }
            for item in result.selected_segments
        ],
        "timings": {
            "deeptalk_seconds": result.timings.active_speaker_seconds,
            "syncnet_initialization_seconds": result.timings.synchronization_initialization_seconds,
            "visual_filter_seconds": result.timings.visual_filter_seconds,
            "total_seconds": result.timings.total_seconds,
        },
    }
    return _write_json(document, path, overwrite=overwrite)


def save_collection_result(
    result: CollectionAnalysisResult, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Save ordered source outcomes, failure diagnostics and selected source-time intervals.

    Parameters
    ----------
    result
        Complete or partial collection result, with the planned source count.
    output_path
        JSON summary destination. Original media cannot be overwritten.
    overwrite
        Whether an existing summary may be atomically replaced.

    Returns
    -------
    pathlib.Path
        Summary referencing per-video JSON when present. No video is copied or decoded.
    """
    path = _validate_output_path(output_path, overwrite, tuple(item.path for item in result.items))
    items: list[dict[str, object]] = []
    for item in result.items:
        items.append(
            {
                "source": str(item.path.resolve()),
                "output": None if item.output_path is None else str(item.output_path.resolve()),
                "status": "failed" if item.error is not None else "completed",
                "error": None
                if item.error is None
                else {"type": item.error.error_type, "message": item.error.message},
                "selected_intervals": []
                if item.result is None
                else [
                    {
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                        "face_id": segment.face_id,
                        "reason_codes": segment.reason_codes,
                    }
                    for segment in item.result.selected_segments
                ],
            }
        )
    return _write_json(
        {
            "schema_version": 1,
            "kind": "collection",
            "total_videos": result.total_videos,
            "completed_videos": len(result.items),
            "successful_videos": len(result.results),
            "failed_videos": len(result.failures),
            "videos": items,
        },
        path,
        overwrite=overwrite,
    )


def _validate_output_path(
    output_path: str | Path, overwrite: bool, sources: tuple[Path, ...]
) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ValueError("analysis output must end in .json")
    if path.is_dir():
        raise IsADirectoryError(f"analysis output is a directory: {path}")
    if path.resolve() in {source.resolve() for source in sources}:
        raise ValueError("analysis output must differ from source video")
    if path.exists() and not overwrite:
        raise FileExistsError(f"analysis output already exists: {path}")
    return path


def _write_json(document: dict[str, object], path: Path, *, overwrite: bool) -> Path:
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized)
        if overwrite:
            temporary_path.replace(path)
        else:
            # Linking within the destination filesystem atomically refuses an existing path.
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
