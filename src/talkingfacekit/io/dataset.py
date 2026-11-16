"""Atomic JSON metadata linking one exported clip to its facial-parameter archive."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from talkingfacekit.analysis import TalkingFaceSegment
from talkingfacekit.flame import FlameResult
from talkingfacekit.io.analysis import _write_json
from talkingfacekit.io.provenance import code_revision

if TYPE_CHECKING:
    from talkingfacekit.dataset import DatasetVideo

_SCHEMA_VERSION = 1


def save_dataset_metadata(
    videos: tuple[DatasetVideo, ...], root: Path, total_videos: int, output_path: Path
) -> Path:
    """Atomically write the version-1 ``talkingfacekit.dataset`` index of a folder run.

    Parameters
    ----------
    videos
        Completed video reports so far, in collection order.
    root
        Dataset root; every listed output path is stored relative to it, in POSIX form.
    total_videos
        Number of videos planned for the run, so a partial index is recognizable.
    output_path
        Destination ``dataset.json``. It is always replaced: callers check conflicts first.

    Returns
    -------
    pathlib.Path
        Written index path. Sources are absolute; no media is read.
    """

    def relative(path: Path | None) -> str | None:
        return None if path is None else path.relative_to(root).as_posix()

    entries: list[dict[str, object]] = []
    for video in videos:
        entries.append(
            {
                "source": str(video.source.resolve()),
                "directory": relative(video.directory),
                "analysis": relative(video.analysis_path),
                "status": "failed" if video.error is not None else "completed",
                "error": None
                if video.error is None
                else {"type": video.error.error_type, "message": video.error.message},
                "segments": [
                    {
                        "directory": relative(item.directory),
                        "start_seconds": item.segment.start_seconds,
                        "end_seconds": item.segment.end_seconds,
                        "face_id": item.segment.face_id,
                        "metadata": relative(item.metadata_path),
                        "error": None
                        if item.error is None
                        else {
                            "stage": item.error.stage,
                            "type": item.error.error_type,
                            "message": item.error.message,
                        },
                    }
                    for item in video.segments
                ],
            }
        )
    segments = [item for video in videos for item in video.segments]
    failed_segments = sum(item.error is not None for item in segments)
    document: dict[str, object] = {
        "kind": "talkingfacekit.dataset",
        "schema_version": _SCHEMA_VERSION,
        "total_videos": total_videos,
        "completed_videos": len(videos),
        "failed_videos": sum(video.error is not None for video in videos),
        "exported_segments": len(segments) - failed_segments,
        "failed_segments": failed_segments,
        "videos": entries,
        "code_revision": code_revision(),
    }
    return _write_json(document, output_path, overwrite=True)


def save_segment_metadata(
    segment: TalkingFaceSegment,
    flame: FlameResult,
    output_path: Path,
    *,
    clip_name: str,
    flame_name: str,
    overwrite: bool = False,
) -> Path:
    """Write the version-1 ``talkingfacekit.segment`` JSON document beside its outputs.

    Parameters
    ----------
    segment
        Selected source interval, local face-track identity and final decision reasons.
    flame
        Saved parameters for the same source interval; only summary values are copied.
    output_path
        Destination ``segment.json``. Its parent must already contain the named outputs.
    clip_name, flame_name
        File names of the clip and parameter archive, relative to ``output_path.parent``.
    overwrite
        Whether an existing document may be atomically replaced.

    Returns
    -------
    pathlib.Path
        Completed metadata path. The clip's local time ``t`` maps to source PTS
        ``t + clip.source_time_offset_seconds``; FLAME timestamps are already source PTS.
    """
    document: dict[str, object] = {
        "kind": "talkingfacekit.segment",
        "schema_version": _SCHEMA_VERSION,
        "source": str(segment.source.path.resolve()),
        "source_sha256": flame.source_sha256,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "face_id": segment.face_id,
        "status": segment.status,
        "reason_codes": list(segment.reason_codes),
        "clip": {"path": clip_name, "source_time_offset_seconds": segment.start_seconds},
        "flame": {
            "path": flame_name,
            "backend": flame.provenance["backend"],
            "frame_count": flame.frame_count,
            "estimated_count": flame.estimated_count,
        },
        "code_revision": code_revision(),
    }
    return _write_json(document, output_path, overwrite=overwrite)
