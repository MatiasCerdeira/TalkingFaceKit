"""Persist typed evaluations using the existing benchmark report schema."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from talkingfacekit.evaluation import EvaluationCaseResult, EvaluationResult
from talkingfacekit.io.analysis import _validate_output_path, _write_json
from talkingfacekit.io.provenance import code_revision


def save_evaluation_result(
    result: EvaluationResult, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write evaluation data without inference, preserving compatibility with the comparator.

    Parameters
    ----------
    result
        Completed analyses and human references. No source media is copied or exported.
    output_path
        JSON destination. Parent directories are created; reference inputs cannot be replaced.
    overwrite
        Allow atomic replacement of an existing result after serialization succeeds.

    Returns
    -------
    pathlib.Path
        Saved version-1 report with additive candidate references and collection scope metadata.
    """
    path = _validate_output_path(
        output_path,
        overwrite,
        (result.manifest, *(case.analysis.source.path for case in result.cases)),
    )
    if result.manifest.is_dir() and path.resolve().is_relative_to(result.manifest.resolve()):
        raise ValueError("evaluation output must be outside the annotations directory")
    return _write_json(
        {
            "schema_version": 1,
            "package_version": version("talkingfacekit"),
            "code_revision": code_revision(),
            "manifest": str(result.manifest),
            "manifest_sha256": result.manifest_sha256,
            "cases": [_case_document(case) for case in result.cases],
            "overall": asdict(result.metrics),
            "overall_before_visual_filter": asdict(result.metrics_before_visual_filter),
            "syncnet_initialization_seconds": result.syncnet_initialization_seconds,
            "selected_segments": [
                {
                    "source": str(segment.source.path),
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "face_id": segment.face_id,
                    "reason_codes": segment.reason_codes,
                }
                for segment in result.selected_segments
            ],
            "unannotated_videos": [str(path) for path in result.unannotated_videos],
            "unmatched_annotation_files": [
                file.as_posix() for file in result.unmatched_annotation_files
            ],
        },
        path,
        overwrite=overwrite,
    )


def _case_document(case: EvaluationCaseResult) -> dict[str, object]:
    reference, analysis = case.reference, case.analysis
    evidence = analysis.active_speaker
    return {
        "id": reference.case_id,
        "file": reference.file.as_posix(),
        "sha256": reference.sha256,
        "window": [reference.window.start_seconds, reference.window.end_seconds],
        "selected": [[item.start_seconds, item.end_seconds] for item in reference.selected],
        "ignore": [[item.start_seconds, item.end_seconds] for item in reference.ignored],
        "predicted": [[item.start_seconds, item.end_seconds] for item in case.predicted],
        "predicted_before_visual_filter": [
            [item.start_seconds, item.end_seconds] for item in case.predicted_before_visual_filter
        ],
        "metrics_before_visual_filter": asdict(case.metrics_before_visual_filter),
        "metrics": asdict(case.metrics),
        "backend_version": evidence.provenance.backend_version,
        "candidate_policy": asdict(evidence.candidate_policy),
        "issues": list(evidence.issues),
        "synchronization": None
        if analysis.synchronization is None
        else asdict(analysis.synchronization),
        "visual_filter": None if analysis.visual_filter is None else asdict(analysis.visual_filter),
        "timings": {"deeptalk_seconds": analysis.timings.active_speaker_seconds},
        "diagnostics": {
            "preliminary_segments": [asdict(item) for item in evidence.preliminary_segments],
            "candidate_runs": [asdict(item) for item in evidence.candidate_runs],
            "shot_boundaries": [item.source_timestamp_seconds for item in evidence.shot_boundaries],
        },
    }
