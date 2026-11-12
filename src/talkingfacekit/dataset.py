"""Turn selected source segments into linked clip, facial-parameter and metadata files."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from talkingfacekit.analysis import AnalysisConfig, TalkingFaceSegment
from talkingfacekit.collection import AnalysisFailure, CollectionProgress, TalkingFaceCollection
from talkingfacekit.flame import FlameResult
from talkingfacekit.sequence import TalkingFaceSequence

FlameExtractor = Callable[[TalkingFaceSequence], FlameResult]
"""Estimate facial parameters for a source-timeline sequence, e.g. ``SmirkEncoder.extract``."""

SegmentStage = Literal["export", "flame", "metadata"]

CLIP_NAME = "clip.mp4"
FLAME_NAME = "flame.npz"
METADATA_NAME = "segment.json"
DATASET_NAME = "dataset.json"
ANALYSIS_DIR = "analysis"


@dataclass(frozen=True, slots=True)
class SegmentFailure:
    """Describe a failed segment without retaining an exception or its traceback.

    Attributes
    ----------
    stage
        Step that raised: clip ``export``, ``flame`` estimation/validation/saving, or
        ``metadata`` writing.
    error_type, message
        Python exception class name and diagnostic text.
    """

    stage: SegmentStage
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class SegmentExport:
    """Report the outputs of one segment, or why none were kept.

    Attributes
    ----------
    segment
        Input segment, unchanged.
    directory
        Segment output directory. It does not exist after a failure.
    clip_path, flame_path, metadata_path
        Completed files, or all ``None`` when ``error`` is set.
    error
        Failure description, or ``None`` when every output was written.
    """

    segment: TalkingFaceSegment
    directory: Path
    clip_path: Path | None = None
    flame_path: Path | None = None
    metadata_path: Path | None = None
    error: SegmentFailure | None = None


@dataclass(frozen=True, slots=True)
class DatasetVideo:
    """Report one collection source: its analysis outcome and exported segments.

    Attributes
    ----------
    source
        Input video path, in collection order.
    directory
        ``output_dir/NNNN-stem`` holding this video's ``segment_NNN`` directories. It exists
        only when at least one segment was exported.
    analysis_path
        Saved per-video analysis JSON, or ``None`` when inspection or analysis failed.
    error
        Analysis failure, or ``None`` for a successful analysis, including one that selected
        no segments.
    segments
        One report per selected segment; empty after an analysis failure.
    """

    source: Path
    directory: Path
    analysis_path: Path | None
    error: AnalysisFailure | None
    segments: tuple[SegmentExport, ...]


def build_dataset(
    collection: TalkingFaceCollection,
    output_dir: str | Path,
    *,
    extract_flame: FlameExtractor,
    config: AnalysisConfig | None = None,
    overwrite: bool = False,
) -> tuple[DatasetVideo, ...]:
    """Analyze every video and export its selected segments, one video at a time.

    This composes :meth:`TalkingFaceCollection.analyze` and :func:`export_segments`: each
    video's segments are exported as soon as its analysis has been saved, so an interruption
    keeps every completed video. The output layout is::

        output_dir/
            analysis/NNNN-stem.json, analysis/collection.json
            NNNN-stem/segment_NNN/{clip.mp4, flame.npz, segment.json}
            dataset.json

    ``NNNN`` is the 1-based collection position, as in the collection's analysis files, so
    videos with the same stem stay separate. ``dataset.json`` is rewritten atomically after
    every video and lists exactly this run's videos, segments and failures.

    Parameters
    ----------
    collection
        Ordered sources. Analysis failures are recorded and processing continues.
    output_dir
        Dataset root directory, created as needed. It should not be inside a recursively
        discovered input folder.
    extract_flame
        Facial-parameter extractor passed to :func:`export_segments`.
    config
        Shared analysis configuration, or the structural-only defaults.
    overwrite
        Whether results of a previous run may be replaced. Otherwise an existing
        ``dataset.json`` or video directory raises before any model runs.

    Returns
    -------
    tuple of DatasetVideo
        One report per source, in collection order.

    Raises
    ------
    FileExistsError
        If outputs exist and ``overwrite`` is false.

    Notes
    -----
    With ``overwrite``, files of an earlier run that this run does not rewrite (for example a
    ``segment_004`` when only three segments are now selected) are left in place but are not
    listed in ``dataset.json``, which is the authoritative index.
    """
    from talkingfacekit.io.dataset import save_dataset_metadata

    root = Path(output_dir)
    directories = tuple(
        root / f"{index:04d}-{path.stem}" for index, path in enumerate(collection, start=1)
    )
    metadata_path = root / DATASET_NAME
    if not overwrite:
        for path in (metadata_path, *directories):
            if path.exists():
                raise FileExistsError(f"dataset output already exists: {path}")
    videos: list[DatasetVideo] = []

    def export_video(update: CollectionProgress) -> None:
        outcome = update.outcome
        directory = directories[len(videos)]
        segments = (
            ()
            if outcome.result is None
            else export_segments(
                outcome.result.selected_segments,
                directory,
                extract_flame=extract_flame,
                overwrite=overwrite,
            )
        )
        videos.append(
            DatasetVideo(outcome.path, directory, outcome.output_path, outcome.error, segments)
        )
        save_dataset_metadata(tuple(videos), root, len(collection), metadata_path)

    collection.analyze(
        config,
        output_dir=root / ANALYSIS_DIR,
        on_error="continue",
        on_progress=export_video,
        overwrite=overwrite,
    )
    if not collection:
        save_dataset_metadata((), root, 0, metadata_path)
    return tuple(videos)


def export_segments(
    segments: Iterable[TalkingFaceSegment],
    output_dir: str | Path,
    *,
    extract_flame: FlameExtractor,
    overwrite: bool = False,
) -> tuple[SegmentExport, ...]:
    """Export each segment's MP4, facial parameters and linking metadata, in order.

    Each segment is written to ``output_dir/segment_NNN/`` (1-based, input order) as
    ``clip.mp4``, ``flame.npz`` and, last, ``segment.json``; a complete directory therefore
    always contains the metadata file. Parameters are estimated on ``segment.to_sequence()``,
    i.e. on the original source and timeline, never on the re-encoded clip. Both files cover
    the same decoded frames: clip time ``t`` corresponds to source PTS ``t + start_seconds``.

    Parameters
    ----------
    segments
        Segments to export, usually ``VideoAnalysisResult.selected_segments``. Any status is
        accepted and recorded in the metadata.
    output_dir
        Directory for the segment subdirectories. It is created only when there is a segment.
    extract_flame
        Called once per segment with its source-timeline sequence, for example
        ``lambda sequence: encoder.extract(sequence, landmarker_model=path)``. Its result
        must describe exactly that source and interval.
    overwrite
        Whether existing segment files may be replaced. Otherwise any conflict raises before
        the first segment is processed.

    Returns
    -------
    tuple of SegmentExport
        One report per input segment. A failed segment keeps no files or directory and does
        not stop later segments.

    Raises
    ------
    FileExistsError
        If an output already exists and ``overwrite`` is false.
    IsADirectoryError
        If an output file path is a directory.

    Notes
    -----
    Processing is sequential. Only :class:`Exception` subclasses become segment failures;
    ``KeyboardInterrupt`` removes the current segment's partial files and propagates.
    """
    items = tuple(segments)
    if not items:
        return ()
    root = Path(output_dir)
    directories = tuple(root / f"segment_{index:03d}" for index in range(1, len(items) + 1))
    for directory in directories:
        for name in (CLIP_NAME, FLAME_NAME, METADATA_NAME):
            path = directory / name
            if path.is_dir():
                raise IsADirectoryError(f"segment output is a directory: {path}")
            if path.exists() and not overwrite:
                raise FileExistsError(f"segment output already exists: {path}")
    return tuple(
        _export_segment(segment, directory, extract_flame, overwrite=overwrite)
        for segment, directory in zip(items, directories, strict=True)
    )


def _export_segment(
    segment: TalkingFaceSegment,
    directory: Path,
    extract_flame: FlameExtractor,
    *,
    overwrite: bool,
) -> SegmentExport:
    """Write one segment completely, or leave none of its named outputs behind.

    Without ``overwrite`` the caller guarantees that no named output existed, so cleanup only
    removes files from this call. With ``overwrite``, stale outputs of a previous run are removed
    too, so a leftover ``segment.json`` can never describe a missing or replaced clip.
    """
    from talkingfacekit.io.dataset import save_segment_metadata

    stage: SegmentStage = "export"
    try:
        sequence = segment.to_sequence()
        clip_path = sequence.export_clip(directory / CLIP_NAME, overwrite=overwrite)
        stage = "flame"
        flame = extract_flame(sequence)
        _validate_flame(flame, segment)
        flame_path = flame.save(directory / FLAME_NAME, overwrite=overwrite)
        stage = "metadata"
        metadata_path = save_segment_metadata(
            segment,
            flame,
            directory / METADATA_NAME,
            clip_name=CLIP_NAME,
            flame_name=FLAME_NAME,
            overwrite=overwrite,
        )
    except BaseException as error:
        for name in (METADATA_NAME, FLAME_NAME, CLIP_NAME):
            (directory / name).unlink(missing_ok=True)
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
        if not isinstance(error, Exception):
            raise
        return SegmentExport(
            segment, directory, error=SegmentFailure(stage, type(error).__name__, str(error))
        )
    return SegmentExport(segment, directory, clip_path, flame_path, metadata_path)


def _validate_flame(flame: object, segment: TalkingFaceSegment) -> None:
    """Require parameters for exactly the segment's source and half-open interval."""
    if not isinstance(flame, FlameResult):
        raise TypeError("extract_flame must return a FlameResult")
    if flame.source.path.resolve() != segment.source.path.resolve():
        raise ValueError("FLAME result describes a different source video")
    if (flame.start_seconds, flame.end_seconds) != (segment.start_seconds, segment.end_seconds):
        raise ValueError(
            "FLAME result interval does not match the segment: "
            f"[{flame.start_seconds}, {flame.end_seconds}) != "
            f"[{segment.start_seconds}, {segment.end_seconds})"
        )
