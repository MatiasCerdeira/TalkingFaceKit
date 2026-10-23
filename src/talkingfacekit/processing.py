"""Application orchestration shared by sequence, collection and command-line facades."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from talkingfacekit.analysis import (
    AnalysisConfig,
    AnalysisTimings,
    VideoAnalysisResult,
    decision_regions,
)
from talkingfacekit.collection import (
    AnalysisFailure,
    CollectionAnalysisResult,
    CollectionProgress,
    ErrorPolicy,
    TalkingFaceCollection,
    VideoAnalysisOutcome,
)
from talkingfacekit.evaluation import (
    EvaluationCaseResult,
    EvaluationResult,
    load_evaluation_cases,
    parse_evaluation_cases,
    prepare_evaluation_manifest,
    verify_evaluation_videos,
)
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.visual import VisualObservation, assess_visual_suitability

if TYPE_CHECKING:
    from talkingfacekit.integrations.syncnet import SyncNetVerifier


class VideoAnalyzer:
    """Execute analysis stages with one configuration and an optional reusable SyncNet model.

    Parameters
    ----------
    config
        Optional stage configuration. Construction loads SyncNet when explicitly configured;
        active-speaker state is recreated per video. This runner is sequential and must not be
        shared across concurrent calls.

    Notes
    -----
    Prefer ``sequence.analyze`` or ``collection.analyze`` for ordinary use. A runner is useful
    when analyzing multiple manually chosen source intervals with the same SyncNet checkpoint.
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self._config = AnalysisConfig() if config is None else config
        if not isinstance(self.config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig or None")
        self._verifier: SyncNetVerifier | None = None
        if self.config.syncnet_model is not None:
            from talkingfacekit.integrations.syncnet import SyncNetVerifier

            self._verifier = SyncNetVerifier(
                self.config.syncnet_model, policy=self.config.synchronization_policy
            )

    @property
    def config(self) -> AnalysisConfig:
        """Validated immutable stage configuration for this runner."""
        return self._config

    @property
    def synchronization_initialization_seconds(self) -> float:
        """One-time SyncNet construction cost, zero when verification is disabled."""
        return 0.0 if self._verifier is None else self._verifier.initialization_seconds

    def analyze(self, sequence: TalkingFaceSequence) -> VideoAnalysisResult:
        """Analyze one sequence without mutating its source, bounds or attached tracks.

        Parameters
        ----------
        sequence
            Original media and optional source-time bounds.

        Returns
        -------
        VideoAnalysisResult
            Complete evidence and selection for the enabled stages. Exceptions propagate.
        """
        from talkingfacekit.integrations import deeptalk

        if not isinstance(sequence, TalkingFaceSequence):
            raise TypeError("sequence must be a TalkingFaceSequence")
        started = time.perf_counter()
        active_speaker = deeptalk.analyze_sequence(
            sequence, candidate_policy=self.config.candidate_policy
        )
        active_speaker_seconds = time.perf_counter() - started
        synchronization = (
            None if self._verifier is None else self._verifier.verify(sequence, active_speaker)
        )
        regions = decision_regions(active_speaker, synchronization)
        visual = None
        visual_seconds = 0.0
        if self.config.visual_policy is not None:
            visual_started = time.perf_counter()
            visual = assess_visual_suitability(
                regions,
                (
                    VisualObservation(
                        item.source_timestamp_seconds,
                        item.face_id,
                        item.bounding_box_xywh,
                        item.head_pose_yaw_pitch_roll_degrees,
                    )
                    for item in active_speaker.face_observations
                ),
                image_width=sequence.source.metadata.width,
                image_height=sequence.source.metadata.height,
                video_sample_rate_hz=active_speaker.provenance.video_sample_rate_hz,
                policy=self.config.visual_policy,
            )
            visual_seconds = time.perf_counter() - visual_started
        return VideoAnalysisResult(
            source=sequence.source,
            start_seconds=sequence.start_seconds,
            end_seconds=sequence.end_seconds,
            active_speaker=active_speaker,
            synchronization=synchronization,
            visual_filter=visual,
            timings=AnalysisTimings(
                active_speaker_seconds=active_speaker_seconds,
                synchronization_initialization_seconds=(
                    0.0 if self._verifier is None else self._verifier.initialization_seconds
                ),
                visual_filter_seconds=visual_seconds,
                total_seconds=time.perf_counter() - started,
            ),
        )


def analyze_sequence(
    sequence: TalkingFaceSequence, config: AnalysisConfig | None = None
) -> VideoAnalysisResult:
    """Run the complete configured analysis of one source-timeline sequence.

    Parameters
    ----------
    sequence
        Source and optional interval, without materialized pixel or audio buffers.
    config
        Optional stage configuration. Defaults preserve structural-only behavior.

    Returns
    -------
    VideoAnalysisResult
        Immutable evidence and final selection. Model/media errors propagate without modifying
        the input sequence or its landmark tracks.
    """
    return VideoAnalyzer(config).analyze(sequence)


def analyze_collection(
    collection: TalkingFaceCollection,
    config: AnalysisConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    on_error: ErrorPolicy = "continue",
    on_progress: Callable[[CollectionProgress], None] | None = None,
    overwrite: bool = False,
) -> CollectionAnalysisResult:
    """Run sequential collection analysis with per-source outcomes and optional checkpoints.

    Parameters
    ----------
    collection, config
        Ordered sources and shared configuration; models are never shared with another run.
    output_dir, overwrite
        Optional per-video evidence and incremental summary persistence; existing outputs require
        explicit overwrite. Output conflicts fail before model construction.
    on_error
        Record source failures and continue, or record and reraise the original exception.
    on_progress
        Optional callback after each saved outcome. Its exceptions are never swallowed.

    Returns
    -------
    CollectionAnalysisResult
        Typed outcomes in collection order, including successful empty selections.
    """
    if on_error not in {"continue", "raise"}:
        raise ValueError("on_error must be 'continue' or 'raise'")
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable or None")
    directory = None if output_dir is None else Path(output_dir)
    outputs = (
        ()
        if directory is None
        else tuple(
            directory / f"{index:04d}-{path.stem}.json" for index, path in enumerate(collection, 1)
        )
    )
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        sources = {path.resolve() for path in collection}
        for output in (*outputs, directory / "collection.json"):
            if output.exists() and not overwrite:
                raise FileExistsError(f"analysis output already exists: {output}")
            if output.is_dir():
                raise IsADirectoryError(f"analysis output is a directory: {output}")
            if output.resolve() in sources:
                raise ValueError("analysis output must differ from source videos")
    if not collection:
        empty = CollectionAnalysisResult((), 0)
        if directory is not None:
            empty.save(directory / "collection.json", overwrite=overwrite)
        return empty
    analyzer = VideoAnalyzer(config)
    items: list[VideoAnalysisOutcome] = []

    def record(outcome: VideoAnalysisOutcome) -> None:
        items.append(outcome)
        if directory is not None:
            CollectionAnalysisResult(tuple(items), len(collection)).save(
                directory / "collection.json", overwrite=True
            )
        if on_progress is not None:
            on_progress(CollectionProgress(len(items), len(collection), outcome))

    for index, path in enumerate(collection):
        try:
            result = analyzer.analyze(TalkingFaceSequence.from_video(path))
        except Exception as error:
            record(
                VideoAnalysisOutcome(
                    path, error=AnalysisFailure(path, type(error).__name__, str(error))
                )
            )
            if on_error == "raise":
                raise
            continue
        output_path = (
            None if directory is None else result.save(outputs[index], overwrite=overwrite)
        )
        record(VideoAnalysisOutcome(path, result=result, output_path=output_path))
    return CollectionAnalysisResult(tuple(items), len(collection))


def evaluate_benchmark(
    reference_path: str | Path,
    videos_dir: str | Path,
    config: AnalysisConfig | None = None,
    *,
    collection: TalkingFaceCollection | None = None,
    video_aliases: Mapping[str, str] | None = None,
    on_progress: Callable[[EvaluationCaseResult], None] | None = None,
) -> EvaluationResult:
    """Analyze labeled windows and return data for scoring, candidate reuse and visualization.

    Parameters
    ----------
    reference_path
        Version-1 manifest, simple annotation JSON list, or directory of annotation lists.
    videos_dir
        Local source directory. Complete source hashes are verified before loading models.
    config
        Shared analysis configuration, with the same defaults as ``collection.analyze``.
    collection
        Optional source scope. Only matching labeled windows run; excluded source and annotation
        names are reported explicitly. Without a scope, every referenced video must exist.
    video_aliases
        Explicit annotation-to-local relative filename mapping for simple annotations only.
        Hashed manifests already identify their exact source files and cannot be remapped.
    on_progress
        Optional callback after each scored window. Errors and interruption propagate.

    Returns
    -------
    EvaluationResult
        Typed analyses, candidate references and duration-weighted metrics. No files are written,
        human labels are not modified, and models run only on the annotated source intervals.
    """
    path = Path(reference_path).expanduser()
    root = Path(videos_dir).expanduser().resolve()
    include_files = (
        None
        if collection is None
        else frozenset(PurePosixPath(item.resolve().relative_to(root)) for item in collection)
    )
    # A list is the simple human annotation format; an object is the approved hashed manifest.
    is_annotations = path.is_dir() or path.read_text(encoding="utf-8").lstrip().startswith("[")
    unmatched: tuple[PurePosixPath, ...] = ()
    if is_annotations:
        document = prepare_evaluation_manifest(
            path, root, include_files=include_files, video_aliases=video_aliases
        )
        cases = parse_evaluation_cases(document)
        digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        # The annotation loader owns this field and emits only relative POSIX filename strings.
        unmatched = tuple(
            PurePosixPath(item)
            for item in cast(list[str], document.get("unmatched_annotation_files", []))
        )
    else:
        if video_aliases:
            raise ValueError("video aliases apply to simple annotations, not hashed manifests")
        cases = load_evaluation_cases(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if include_files is not None:
            unmatched = tuple(
                sorted({case.file for case in cases if case.file not in include_files})
            )
            cases = tuple(case for case in cases if case.file in include_files)
    if not cases:
        raise ValueError("no annotated windows match the video collection")
    video_paths = verify_evaluation_videos(cases, root)
    analyzer = VideoAnalyzer(config)
    results: list[EvaluationCaseResult] = []
    for case in cases:
        sequence = TalkingFaceSequence.from_video(video_paths[case.case_id]).clip(
            case.window.start_seconds, case.window.end_seconds
        )
        analysis = analyzer.analyze(sequence)
        if not analysis.active_speaker.score_windows:
            raise RuntimeError(f"{case.case_id}: analysis returned no score windows")
        result = EvaluationCaseResult(case, analysis)
        # Validate source-time predictions before exposing the completed case.
        _ = result.metrics, result.metrics_before_visual_filter
        results.append(result)
        if on_progress is not None:
            on_progress(result)
    return EvaluationResult(
        cases=tuple(results),
        manifest=path,
        manifest_sha256=digest,
        syncnet_initialization_seconds=None
        if analyzer.config.syncnet_model is None
        else analyzer.synchronization_initialization_seconds,
        unannotated_videos=()
        if collection is None
        else tuple(item for item in collection if item.resolve() not in set(video_paths.values())),
        unmatched_annotation_files=unmatched,
    )
