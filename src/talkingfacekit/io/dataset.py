"""Atomic JSON metadata linking one exported clip to its facial-parameter archive."""

from __future__ import annotations

from pathlib import Path

from talkingfacekit.analysis import TalkingFaceSegment
from talkingfacekit.flame import FlameResult
from talkingfacekit.io.analysis import _write_json
from talkingfacekit.io.provenance import code_revision

_SCHEMA_VERSION = 1


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
