"""Command-line workflows for TalkingFaceKit."""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np

from talkingfacekit.demo import render_active_speaker_report, render_benchmark_report
from talkingfacekit.evaluation import (
    EvaluationInterval,
    combine_evaluation_metrics,
    load_evaluation_cases,
    parse_evaluation_cases,
    prepare_evaluation_manifest,
    score_evaluation_case,
    verify_evaluation_videos,
)
from talkingfacekit.integrations.deeptalk import DeepTalkAnalysisResult, analyze_sequence
from talkingfacekit.integrations.syncnet import SyncNetVerifier
from talkingfacekit.io.landmarks import load_landmark_track, save_landmark_track
from talkingfacekit.io.video import inspect_video_metadata
from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.rendering import render_face_mesh_html
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.synchronization import SynchronizationPolicy, SynchronizationResult
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker
from talkingfacekit.tracking.mediapipe_mesh import build_mediapipe_face_mesh
from talkingfacekit.visual import (
    VisualObservation,
    VisualPolicy,
    VisualRegion,
    VisualResult,
    assess_visual_suitability,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TalkingFaceKit command-line interface.

    Parameters
    ----------
    argv
        Arguments excluding the executable name, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code. Successful commands return zero; invalid input exits through argparse.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    command = cast(str, arguments.command)

    try:
        if command == "extract-landmarks":
            return _extract_landmarks(arguments)
        if command == "inspect-landmarks":
            return _inspect_landmarks(arguments)
        if command == "render-mesh":
            return _render_mesh(arguments)
        if command == "analyze-video":
            return _analyze_video(arguments)
        if command == "evaluate-benchmark":
            return _evaluate_benchmark(arguments)
        if command == "prepare-benchmark":
            return _prepare_benchmark(arguments)
        if command == "render-benchmark":
            saved_path = render_benchmark_report(
                arguments.result,
                arguments.videos_dir,
                arguments.output,
                overwrite=arguments.overwrite,
            )
            print(f"report: {saved_path}")
            return 0
    except (
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")

    raise RuntimeError(f"Unsupported command: {command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m talkingfacekit",
        description="Process talking-face videos with explicit, timestamped data contracts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract-landmarks",
        help="Track one face through a video interval and save every selected frame to NPZ.",
    )
    extract_parser.add_argument("video", type=Path, help="Local video file to process.")
    extract_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="MediaPipe Face Landmarker .task model.",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npz archive.",
    )
    extract_parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help="Inclusive interval start on the source timeline; default: sequence start.",
    )
    extract_parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Exclusive interval end on the source timeline; default: end of stream.",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after extraction completes.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-landmarks",
        help="Validate an existing landmark NPZ archive and print its summary.",
    )
    inspect_parser.add_argument("archive", type=Path, help="Landmark .npz archive to inspect.")

    render_parser = subparsers.add_parser(
        "render-mesh",
        help="Build and render an animated MediaPipe face mesh to offline HTML.",
    )
    render_parser.add_argument("archive", type=Path, help="MediaPipe landmark .npz archive.")
    render_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Source video used only to obtain width and height for aspect correction.",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination self-contained .html file.",
    )
    render_parser.add_argument(
        "--title",
        default="TalkingFaceKit animated face mesh",
        help="Heading displayed above the interactive mesh.",
    )
    render_parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Positive display-only multiplier for relative depth; default: 1.0.",
    )
    render_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after rendering completes.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-video",
        help="Run experimental active-speaker analysis and print a diagnostic summary.",
    )
    analyze_parser.add_argument("video", type=Path, help="Local video file to analyze.")
    analyze_parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help="Inclusive interval start on the source timeline; default: sequence start.",
    )
    analyze_parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Exclusive interval end on the source timeline; default: end of stream.",
    )
    analyze_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a synchronized diagnostic report to this .html file.",
    )
    analyze_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing HTML report; otherwise an existing destination gets a numbered "
            "sibling."
        ),
    )
    analyze_parser.add_argument(
        "--output", type=Path, help="Save analysis and sync evidence as JSON."
    )
    _add_syncnet_arguments(analyze_parser)
    _add_visual_arguments(analyze_parser)

    preparation_parser = subparsers.add_parser(
        "prepare-benchmark",
        help="Generate a hashed benchmark manifest from simple human annotations.",
    )
    preparation_parser.add_argument(
        "annotations", type=Path, help="Annotations JSON list or directory of JSON lists."
    )
    preparation_parser.add_argument(
        "--videos-dir", type=Path, required=True, help="Directory containing annotated videos."
    )
    preparation_parser.add_argument(
        "--output", type=Path, required=True, help="Destination generated manifest JSON."
    )
    preparation_mode = preparation_parser.add_mutually_exclusive_group()
    preparation_mode.add_argument(
        "--overwrite", action="store_true", help="Replace an existing manifest."
    )
    preparation_mode.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing manifest matches annotations and videos without writing.",
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate-benchmark",
        help="Verify labeled videos and compare selected intervals with DeepTalk candidate runs.",
    )
    evaluation_parser.add_argument("manifest", type=Path, help="Versioned benchmark JSON manifest.")
    evaluation_parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Local directory containing benchmark videos.",
    )
    evaluation_parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=None,
        help="File containing the approved SHA-256 digest of the manifest.",
    )
    evaluation_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate labels and SHA-256 hashes without loading inference models.",
    )
    evaluation_parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON path for metrics and predictions."
    )
    evaluation_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing JSON output file."
    )
    _add_syncnet_arguments(evaluation_parser)
    _add_visual_arguments(evaluation_parser)
    comparison_parser = subparsers.add_parser(
        "render-benchmark",
        help="Compare saved human/library intervals in an interactive HTML without inference.",
    )
    comparison_parser.add_argument(
        "result", type=Path, help="Saved evaluate-benchmark JSON output."
    )
    comparison_parser.add_argument("--videos-dir", type=Path, required=True)
    comparison_parser.add_argument("--output", type=Path, required=True, help="Destination .html.")
    comparison_parser.add_argument("--overwrite", action="store_true")
    return parser


def _add_syncnet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--syncnet-model",
        type=Path,
        default=None,
        help="Verify every candidate using this official local syncnet_v2.model checkpoint.",
    )
    parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=80.0,
        help="Maximum absolute offset passing sync; experimental default: 80 ms.",
    )
    parser.add_argument(
        "--sync-minimum-confidence",
        type=float,
        default=3.0,
        help="Minimum raw SyncNet confidence; experimental default: 3.0.",
    )
    parser.add_argument(
        "--sync-maximum-offset-seconds",
        type=float,
        default=1.0,
        help="Maximum offset searched in either direction; default: 1 second.",
    )


def _add_visual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visual-filter",
        action="store_true",
        help="Filter candidates by experimental usable head pose, face size and framing.",
    )
    defaults = VisualPolicy()
    for flag, field, description in (
        ("maximum-yaw", "maximum_absolute_yaw_degrees", "Maximum absolute yaw in degrees"),
        ("maximum-pitch", "maximum_absolute_pitch_degrees", "Maximum absolute pitch in degrees"),
        ("maximum-roll", "maximum_absolute_roll_degrees", "Maximum absolute roll in degrees"),
        (
            "minimum-face-pixels",
            "minimum_face_size_pixels",
            "Minimum shorter face-box side in source pixels",
        ),
        (
            "minimum-face-width-fraction",
            "minimum_face_width_fraction",
            "Minimum visible face width / image width",
        ),
        (
            "maximum-clipped-fraction",
            "maximum_clipped_fraction",
            "Maximum fraction of face box outside image",
        ),
        (
            "maximum-bad-gap-seconds",
            "maximum_bad_gap_seconds",
            "Maximum brief visual gap to bridge",
        ),
        (
            "minimum-duration-seconds",
            "minimum_duration_seconds",
            "Minimum duration after visual trimming",
        ),
        (
            "minimum-observed-fraction",
            "minimum_observed_fraction",
            "Minimum measured time fraction",
        ),
        (
            "minimum-passing-fraction",
            "minimum_passing_fraction",
            "Minimum time fraction passing visual checks",
        ),
    ):
        value = getattr(defaults, field)
        parser.add_argument(
            f"--visual-{flag}",
            dest=f"visual_{field}",
            type=float,
            default=value,
            help=f"{description}; experimental default: {value}.",
        )


def _make_visual_policy(arguments: argparse.Namespace) -> VisualPolicy | None:
    policy = VisualPolicy(
        **{name: getattr(arguments, f"visual_{name}") for name in VisualPolicy.__dataclass_fields__}
    )
    if not arguments.visual_filter:
        if policy != VisualPolicy():
            raise ValueError("visual thresholds require --visual-filter")
        return None
    return policy


def _apply_visual_filter(
    sequence: TalkingFaceSequence,
    result: DeepTalkAnalysisResult,
    policy: VisualPolicy | None,
    synchronization: SynchronizationResult | None,
) -> VisualResult | None:
    if policy is None:
        return None
    intervals = [
        VisualRegion(
            run.source_start_seconds,
            run.source_end_seconds,
            run.face_id,
            run.status,
            run.reason_codes,
        )
        for run in result.candidate_runs
        if synchronization is None or run.status != "candidate"
    ]
    if synchronization is not None:
        expected = {
            (run.source_start_seconds, run.source_end_seconds, run.face_id)
            for run in result.candidate_runs
            if run.status == "candidate"
        }
        actual = {
            (item.source_start_seconds, item.source_end_seconds, item.face_id)
            for item in synchronization.decisions
        }
        if actual != expected or len(actual) != len(synchronization.decisions):
            raise ValueError("sync decisions must cover every structural candidate exactly once")
        intervals.extend(
            VisualRegion(
                item.source_start_seconds,
                item.source_end_seconds,
                item.face_id,
                item.status,
                item.reason_codes,
            )
            for item in synchronization.decisions
        )
    intervals.sort(key=lambda item: item.source_start_seconds)
    return assess_visual_suitability(
        intervals,
        (
            VisualObservation(
                item.source_timestamp_seconds,
                item.face_id,
                item.bounding_box_xywh,
                item.head_pose_yaw_pitch_roll_degrees,
            )
            for item in result.face_observations
        ),
        image_width=sequence.source.metadata.width,
        image_height=sequence.source.metadata.height,
        video_sample_rate_hz=result.provenance.video_sample_rate_hz,
        policy=policy,
    )


def _selected_regions(
    result: DeepTalkAnalysisResult,
    synchronization: SynchronizationResult | None,
    visual: VisualResult | None = None,
) -> tuple[VisualRegion, ...]:
    if visual is not None:
        return tuple(item for item in visual.decisions if item.status == "candidate")
    if synchronization is not None:
        return tuple(
            VisualRegion(
                item.source_start_seconds,
                item.source_end_seconds,
                item.face_id,
                item.status,
                item.reason_codes,
            )
            for item in synchronization.decisions
            if item.status == "candidate"
        )
    return tuple(
        VisualRegion(
            run.source_start_seconds,
            run.source_end_seconds,
            run.face_id,
            run.status,
            run.reason_codes,
        )
        for run in result.candidate_runs
        if run.status == "candidate"
    )


def _print_visual_summary(visual: VisualResult | None) -> None:
    if visual is None:
        return
    print(
        "visual filter: experimental frontal pose / source size / framing; eye gaze not evaluated"
    )
    for item in visual.decisions:
        print(
            f"  {item.source_start_seconds:.2f}–{item.source_end_seconds:.2f}s "
            f"face={item.face_id} {item.status}: {', '.join(item.reason_codes)}"
        )


def _make_syncnet_verifier(arguments: argparse.Namespace) -> SyncNetVerifier | None:
    policy = SynchronizationPolicy(
        tolerance_seconds=arguments.sync_tolerance_ms / 1000,
        minimum_confidence=arguments.sync_minimum_confidence,
        maximum_offset_seconds=arguments.sync_maximum_offset_seconds,
    )
    if arguments.syncnet_model is None:
        if policy != SynchronizationPolicy():
            raise ValueError("sync thresholds require --syncnet-model")
        return None
    return SyncNetVerifier(arguments.syncnet_model, policy=policy)


def _extract_landmarks(arguments: argparse.Namespace) -> int:
    video_path = cast(Path, arguments.video)
    model_path = cast(Path, arguments.model)
    output_path = cast(Path, arguments.output)
    start_seconds = cast(float | None, arguments.start_seconds)
    end_seconds = cast(float | None, arguments.end_seconds)
    overwrite = cast(bool, arguments.overwrite)

    sequence = TalkingFaceSequence.from_video(video_path)
    if start_seconds is not None or end_seconds is not None:
        resolved_start_seconds = sequence.start_seconds if start_seconds is None else start_seconds
        sequence = sequence.clip(resolved_start_seconds, end_seconds)
    tracker = MediaPipeFaceTracker(model_path)
    track = sequence.track_landmarks(tracker, name="mediapipe")
    saved_path = save_landmark_track(track, output_path, overwrite=overwrite)

    _print_summary(track)
    print(f"output: {saved_path}")
    return 0


def _inspect_landmarks(arguments: argparse.Namespace) -> int:
    archive_path = cast(Path, arguments.archive)
    track = load_landmark_track(archive_path)

    _print_summary(track)
    print(f"archive: {archive_path}")
    return 0


def _render_mesh(arguments: argparse.Namespace) -> int:
    archive_path = cast(Path, arguments.archive)
    video_path = cast(Path, arguments.video)
    output_path = cast(Path, arguments.output)
    title = cast(str, arguments.title)
    depth_scale = cast(float, arguments.depth_scale)
    overwrite = cast(bool, arguments.overwrite)

    track = load_landmark_track(archive_path)
    metadata = inspect_video_metadata(video_path)
    mesh = build_mediapipe_face_mesh(
        track,
        image_width=metadata.width,
        image_height=metadata.height,
    )
    saved_path = render_face_mesh_html(
        mesh,
        output_path,
        title=title,
        depth_scale=depth_scale,
        overwrite=overwrite,
    )

    _print_mesh_summary(mesh)
    print(f"output: {saved_path}")
    return 0


def _analyze_video(arguments: argparse.Namespace) -> int:
    video_path = cast(Path, arguments.video)
    start_seconds = cast(float | None, arguments.start_seconds)
    end_seconds = cast(float | None, arguments.end_seconds)
    report_path = cast(Path | None, arguments.report)
    overwrite = cast(bool, arguments.overwrite)
    output_path = cast(Path | None, arguments.output)
    if report_path is None and output_path is None and overwrite:
        raise ValueError("--overwrite requires --report or --output")
    if output_path is not None:
        if output_path.suffix.lower() != ".json":
            raise ValueError("analysis output must end in .json")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"analysis output already exists: {output_path}")
        if output_path.resolve() == video_path.resolve():
            raise ValueError("analysis output must differ from source video")
    visual_policy = _make_visual_policy(arguments)
    verifier = _make_syncnet_verifier(arguments)

    sequence = TalkingFaceSequence.from_video(video_path)
    if start_seconds is not None or end_seconds is not None:
        resolved_start_seconds = sequence.start_seconds if start_seconds is None else start_seconds
        sequence = sequence.clip(resolved_start_seconds, end_seconds)

    previous_logging_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        started = time.perf_counter()
        result = analyze_sequence(sequence)
        deeptalk_seconds = time.perf_counter() - started
    finally:
        logging.disable(previous_logging_disable_level)
    synchronization = None if verifier is None else verifier.verify(sequence, result)
    visual = _apply_visual_filter(sequence, result, visual_policy, synchronization)
    _print_active_speaker_summary(sequence, result)
    if synchronization is not None:
        _print_synchronization_summary(synchronization)
    _print_visual_summary(visual)
    if output_path is not None:
        with sequence.source.path.open("rb") as source_stream:
            source_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()
        document = {
            "schema_version": 1,
            "source": str(sequence.source.path.resolve()),
            "source_sha256": source_sha256,
            "package_version": version("talkingfacekit"),
            "code_revision": _code_revision(),
            "interval": [sequence.start_seconds, sequence.end_seconds],
            "face_observations": [asdict(item) for item in result.face_observations],
            "speech_intervals": [asdict(item) for item in result.speech_intervals],
            "score_windows": [
                {
                    "source_start_seconds": item.source_start_seconds,
                    "source_end_seconds": item.source_end_seconds,
                    "raw_scores": dict(item.raw_scores),
                }
                for item in result.score_windows
            ],
            "preliminary_segments": [asdict(item) for item in result.preliminary_segments],
            "candidate_runs": [asdict(item) for item in result.candidate_runs],
            "candidate_policy": asdict(result.candidate_policy),
            "shot_boundaries": [asdict(item) for item in result.shot_boundaries],
            "audio_timeline_repairs": [asdict(item) for item in result.audio_timeline_repairs],
            "issues": list(result.issues),
            "provenance": asdict(result.provenance),
            "synchronization": None if synchronization is None else asdict(synchronization),
            "visual_filter": None if visual is None else asdict(visual),
            "selected_intervals": [
                asdict(item) for item in _selected_regions(result, synchronization, visual)
            ],
            "timings": {
                "deeptalk_seconds": deeptalk_seconds,
                "syncnet_initialization_seconds": 0
                if verifier is None
                else verifier.initialization_seconds,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"output: {output_path}")
    if report_path is not None:
        resolved_report_path = _available_report_path(report_path, overwrite=overwrite)
        if synchronization is None and visual is None:
            saved_path = render_active_speaker_report(
                sequence, result, resolved_report_path, overwrite=overwrite
            )
        else:
            saved_path = render_active_speaker_report(
                sequence,
                result,
                resolved_report_path,
                overwrite=overwrite,
                synchronization=synchronization,
                visual_filter=visual,
            )
        print(f"report: {saved_path}")
    return 0


def _prepare_benchmark(arguments: argparse.Namespace) -> int:
    annotations_path = cast(Path, arguments.annotations).expanduser()
    videos_dir = cast(Path, arguments.videos_dir)
    output_path = cast(Path, arguments.output).expanduser()
    overwrite = cast(bool, arguments.overwrite)
    check = cast(bool, arguments.check)
    if output_path.resolve() == annotations_path.resolve():
        raise ValueError("manifest output must differ from the human annotations file")
    if annotations_path.is_dir() and output_path.resolve().is_relative_to(
        annotations_path.resolve()
    ):
        raise ValueError("manifest output must be outside the annotations directory")
    if output_path.exists() and not overwrite and not check:
        raise FileExistsError(f"benchmark manifest already exists: {output_path}")
    manifest = prepare_evaluation_manifest(annotations_path, videos_dir)
    if check:
        if load_evaluation_cases(output_path) != parse_evaluation_cases(manifest):
            raise ValueError(
                f"benchmark manifest is out of date: {output_path}; regenerate with --overwrite"
            )
        print(f"manifest is up to date: {output_path}")
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if overwrite else "x", encoding="utf-8") as destination:
        destination.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest: {output_path}")
    return 0


def _evaluate_benchmark(arguments: argparse.Namespace) -> int:
    manifest_path = cast(Path, arguments.manifest)
    videos_dir = cast(Path, arguments.videos_dir)
    manifest_lock = cast(Path | None, arguments.manifest_lock)
    verify_only = cast(bool, arguments.verify_only)
    output_path = cast(Path | None, arguments.output)
    overwrite = cast(bool, arguments.overwrite)
    if verify_only and output_path is not None:
        raise ValueError("--output cannot be combined with --verify-only")
    if overwrite and output_path is None:
        raise ValueError("--overwrite requires --output")
    if output_path is not None and output_path.exists() and not overwrite:
        raise FileExistsError(f"benchmark output already exists: {output_path}")

    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_lock is not None:
        expected_digest = manifest_lock.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError(f"invalid benchmark manifest lock digest: {manifest_lock}")
        if manifest_digest != expected_digest:
            raise ValueError(
                f"benchmark manifest SHA-256 mismatch: expected {expected_digest}, "
                f"got {manifest_digest}"
            )

    cases = load_evaluation_cases(manifest_path)
    video_paths = verify_evaluation_videos(cases, videos_dir)
    print(f"verified {len(cases)} benchmark cases and their source video hashes")
    if verify_only:
        return 0

    visual_policy = _make_visual_policy(arguments)
    verifier = _make_syncnet_verifier(arguments)
    case_reports: list[dict[str, object]] = []
    case_metrics = []
    before_visual_metrics = []
    for case in cases:
        sequence = TalkingFaceSequence.from_video(video_paths[case.case_id]).clip(
            case.window.start_seconds, case.window.end_seconds
        )
        started = time.perf_counter()
        result = analyze_sequence(sequence)
        deeptalk_seconds = time.perf_counter() - started
        if not result.score_windows:
            raise RuntimeError(f"{case.case_id}: analysis returned no score windows")
        synchronization = None if verifier is None else verifier.verify(sequence, result)
        if synchronization is not None:
            _print_synchronization_summary(synchronization)
        before_visual = tuple(
            EvaluationInterval(item.source_start_seconds, item.source_end_seconds)
            for item in _selected_regions(result, synchronization)
        )
        visual = _apply_visual_filter(sequence, result, visual_policy, synchronization)
        _print_visual_summary(visual)
        predicted = tuple(
            EvaluationInterval(item.source_start_seconds, item.source_end_seconds)
            for item in _selected_regions(result, synchronization, visual)
        )
        before_metrics = score_evaluation_case(case, before_visual)
        before_visual_metrics.append(before_metrics)
        metrics = score_evaluation_case(case, predicted)
        case_metrics.append(metrics)
        print(
            f"{case.case_id}: precision={_format_ratio(metrics.precision)} "
            f"recall={_format_ratio(metrics.recall)} "
            f"false_accept={metrics.false_positive_seconds:.2f}s "
            f"missed={metrics.false_negative_seconds:.2f}s"
        )
        case_reports.append(
            {
                "id": case.case_id,
                "file": case.file.as_posix(),
                "sha256": case.sha256,
                "window": [case.window.start_seconds, case.window.end_seconds],
                "selected": [[item.start_seconds, item.end_seconds] for item in case.selected],
                "ignore": [[item.start_seconds, item.end_seconds] for item in case.ignored],
                "predicted": [[item.start_seconds, item.end_seconds] for item in predicted],
                "predicted_before_visual_filter": [
                    [item.start_seconds, item.end_seconds] for item in before_visual
                ],
                "metrics_before_visual_filter": asdict(before_metrics),
                "metrics": asdict(metrics),
                "backend_version": result.provenance.backend_version,
                "candidate_policy": asdict(result.candidate_policy),
                "issues": list(result.issues),
                "synchronization": None if synchronization is None else asdict(synchronization),
                "visual_filter": None if visual is None else asdict(visual),
                "timings": {"deeptalk_seconds": deeptalk_seconds},
                "diagnostics": {
                    "preliminary_segments": [asdict(item) for item in result.preliminary_segments],
                    "candidate_runs": [asdict(item) for item in result.candidate_runs],
                    "shot_boundaries": [
                        item.source_timestamp_seconds for item in result.shot_boundaries
                    ],
                },
            }
        )
    overall = combine_evaluation_metrics(tuple(case_metrics))
    before_overall = combine_evaluation_metrics(tuple(before_visual_metrics))
    if visual_policy is not None:
        print(
            f"before visual: precision={_format_ratio(before_overall.precision)} "
            f"recall={_format_ratio(before_overall.recall)} "
            f"false_accept={before_overall.false_positive_seconds:.2f}s "
            f"missed={before_overall.false_negative_seconds:.2f}s"
        )
    print(
        f"overall: precision={_format_ratio(overall.precision)} "
        f"recall={_format_ratio(overall.recall)} "
        f"false_accept={overall.false_positive_seconds:.2f}s "
        f"missed={overall.false_negative_seconds:.2f}s"
    )
    if output_path is not None:
        report = {
            "schema_version": 1,
            "package_version": version("talkingfacekit"),
            "code_revision": _code_revision(),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "cases": case_reports,
            "overall": asdict(overall),
            "overall_before_visual_filter": asdict(before_overall),
            "syncnet_initialization_seconds": None
            if verifier is None
            else verifier.initialization_seconds,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"output: {output_path}")
    return 0


def _print_synchronization_summary(result: SynchronizationResult) -> None:
    print(
        f"synchronization: {result.backend}, {result.elapsed_seconds:.2f}s; positive offset = late audio"
    )
    for window in result.windows:
        offset = (
            "not measurable"
            if window.estimated_offset_seconds is None
            else f"{window.estimated_offset_seconds * 1000:+.0f}ms"
        )
        confidence = "n/a" if window.confidence is None else f"{window.confidence:.3f}"
        print(
            f"  [{window.source_start_seconds:.3f}, {window.source_end_seconds:.3f}) "
            f"face_{window.face_id} {window.status} offset={offset} confidence={confidence} "
            f"({','.join(window.reason_codes)})"
        )
    for decision in result.decisions:
        print(
            f"  sync decision [{decision.source_start_seconds:.3f}, {decision.source_end_seconds:.3f}): "
            f"{decision.status} ({','.join(decision.reason_codes)})"
        )


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _code_revision() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    if commit.returncode != 0 or status.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _available_report_path(path: Path, *, overwrite: bool) -> Path:
    """Return the requested report path or the first available numbered sibling."""
    if overwrite or not path.is_file():
        return path

    version = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{version}{path.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def _print_active_speaker_summary(
    sequence: TalkingFaceSequence,
    result: DeepTalkAnalysisResult,
) -> None:
    face_ids = sorted({observation.face_id for observation in result.face_observations})
    interval_end = "EOF" if sequence.end_seconds is None else f"{sequence.end_seconds:.3f}"

    print(f"video: {sequence.source.path}")
    print(f"interval: [{sequence.start_seconds:.3f}, {interval_end}) seconds")
    print(f"backend: DeepTalk-ASD {result.provenance.backend_version}")
    print(f"face IDs: {face_ids if face_ids else 'none'}")
    print(f"face observations: {len(result.face_observations)}")

    print("scene cut boundaries:")
    if not result.shot_boundaries:
        print("  none")
    for boundary in result.shot_boundaries:
        print(f"  {boundary.source_timestamp_seconds:.3f}")

    print("speech intervals:")
    if not result.speech_intervals:
        print("  none")
    for interval in result.speech_intervals:
        print(
            f"  [{interval.source_start_seconds:.3f}, "
            f"{interval.source_end_seconds:.3f}) {interval.status}"
        )

    print("raw active-speaker scores:")
    if not result.score_windows:
        print("  none")
    for window in result.score_windows:
        scores = ", ".join(
            f"face_{face_id}={score:+.3f}" for face_id, score in sorted(window.raw_scores.items())
        )
        print(
            f"  [{window.source_start_seconds:.3f}, "
            f"{window.source_end_seconds:.3f}) {scores or 'no face scores'}"
        )

    print("preliminary segments:")
    if not result.preliminary_segments:
        print("  none")
    for segment in result.preliminary_segments:
        face = "none" if segment.face_id is None else f"face_{segment.face_id}"
        reasons = ",".join(segment.reason_codes)
        print(
            f"  [{segment.source_start_seconds:.3f}, "
            f"{segment.source_end_seconds:.3f}) {segment.status} "
            f"{face} ({reasons})"
        )

    policy = result.candidate_policy
    print("continuous candidate runs:")
    print(
        "  policy: "
        f"minimum_duration={policy.minimum_duration_seconds:.3f}s, "
        f"minimum_face_coverage={policy.minimum_face_coverage:.1%}, "
        f"maximum_face_gap={policy.maximum_face_gap_seconds:.3f}s, "
        f"maximum_bridge_gap={policy.maximum_bridge_gap_seconds:.3f}s"
    )
    if not result.candidate_runs:
        print("  none")
    for run in result.candidate_runs:
        reasons = ",".join(run.reason_codes)
        bridged_gaps = ",".join(
            f"[{start_seconds:.3f},{end_seconds:.3f})"
            for start_seconds, end_seconds in run.bridged_gap_intervals
        )
        print(
            f"  [{run.source_start_seconds:.3f}, {run.source_end_seconds:.3f}) "
            f"{run.status} face_{run.face_id} duration={run.duration_seconds:.3f}s "
            f"coverage={run.face_visibility_fraction:.1%} "
            f"max_gap={run.maximum_face_gap_seconds:.3f}s "
            f"score[min/mean/max]={run.raw_score_min:+.3f}/"
            f"{run.raw_score_mean:+.3f}/{run.raw_score_max:+.3f} ({reasons}) "
            f"bridged_gaps={bridged_gaps or 'none'}"
        )

    print("audio timeline repairs:")
    if not result.audio_timeline_repairs:
        print("  none")
    for repair in result.audio_timeline_repairs:
        action = "filled silence" if repair.kind == "audio_gap_filled" else "trimmed overlap"
        print(
            f"  [{repair.source_start_seconds:.3f}, "
            f"{repair.source_end_seconds:.3f}) {action} "
            f"({repair.duration_seconds:.3f} seconds)"
        )

    print(f"issues: {', '.join(result.issues) if result.issues else 'none'}")
    print("note: structurally passing candidates are not yet accepted training segments")


def _print_summary(track: FaceLandmarkTrack) -> None:
    detected_count = int(np.count_nonzero(track.detected))
    missing_count = track.frame_count - detected_count
    first_timestamp = float(track.timestamps_seconds[0])
    last_timestamp = float(track.timestamps_seconds[-1])

    print(f"tracker: {track.tracker_name}")
    print(f"topology: {track.topology}")
    print(f"frames: {track.frame_count}")
    print(f"landmarks per frame: {track.landmark_count}")
    print(f"detected frames: {detected_count}")
    print(f"missing frames: {missing_count}")
    print(f"timestamp range: [{first_timestamp:.6f}, {last_timestamp:.6f}] seconds")
    print(f"landmark array shape: {track.landmarks.shape}")


def _print_mesh_summary(mesh: FaceMeshTrack) -> None:
    detected_count = int(np.count_nonzero(mesh.detected))
    missing_count = mesh.frame_count - detected_count

    print(f"topology: {mesh.topology}")
    print(f"frames: {mesh.frame_count}")
    print(f"vertices per frame: {mesh.vertex_count}")
    print(f"triangles per frame: {mesh.triangle_count}")
    print(f"detected frames: {detected_count}")
    print(f"missing frames: {missing_count}")
    print(f"vertex array shape: {mesh.vertices.shape}")
