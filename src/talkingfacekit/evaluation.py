"""Small, reproducible benchmark for selected talking-face intervals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from talkingfacekit.analysis import TalkingFaceSegment, VideoAnalysisResult, decision_regions


@dataclass(frozen=True, slots=True)
class EvaluationInterval:
    """Represent a half-open interval in seconds on the original media timeline.

    Parameters
    ----------
    start_seconds, end_seconds
        Finite source timestamps with ``0 <= start_seconds < end_seconds``.
    """

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Reject invalid interval bounds."""
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("interval start_seconds must be finite and non-negative")
        if not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds:
            raise ValueError("interval end_seconds must be finite and after start_seconds")

    @property
    def duration_seconds(self) -> float:
        """Return the interval duration in seconds."""
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Describe one labeled source-video window and its expected usable intervals.

    Parameters
    ----------
    case_id
        Stable human-readable case identifier.
    file
        Relative POSIX path below the shared video directory.
    sha256
        Lowercase SHA-256 digest of the complete, unchanged source file.
    window
        Window to analyze on the original source timeline.
    selected
        Human-selected usable intervals within ``window``.
    ignored
        Intervals excluded from scoring because a human could not judge them.
    """

    case_id: str
    file: PurePosixPath
    sha256: str
    window: EvaluationInterval
    selected: tuple[EvaluationInterval, ...]
    ignored: tuple[EvaluationInterval, ...]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Summarize duration-weighted agreement over scored source seconds.

    ``precision`` is ``None`` when nothing was predicted, and ``recall`` is
    ``None`` when the human selected no seconds. Ignored intervals are excluded.
    """

    true_positive_seconds: float
    false_positive_seconds: float
    false_negative_seconds: float
    ignored_seconds: float
    precision: float | None
    recall: float | None


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """Keep a human reference and its analysis on the original source timeline.

    Parameters
    ----------
    reference
        Human labels and the exact window analyzed, in source seconds.
    analysis
        Complete model evidence and typed segment decisions for that window.
    """

    reference: EvaluationCase
    analysis: VideoAnalysisResult

    @property
    def predicted(self) -> tuple[EvaluationInterval, ...]:
        """Selected source-time intervals after all configured stages."""
        return tuple(
            EvaluationInterval(item.start_seconds, item.end_seconds)
            for item in self.analysis.selected_segments
        )

    @property
    def predicted_before_visual_filter(self) -> tuple[EvaluationInterval, ...]:
        """Selected source-time intervals before optional visual filtering."""
        return tuple(
            EvaluationInterval(item.source_start_seconds, item.source_end_seconds)
            for item in decision_regions(
                self.analysis.active_speaker, self.analysis.synchronization
            )
            if item.status == "candidate"
        )

    @property
    def metrics(self) -> EvaluationMetrics:
        """Duration-weighted agreement with the human reference."""
        return score_evaluation_case(self.reference, self.predicted)

    @property
    def metrics_before_visual_filter(self) -> EvaluationMetrics:
        """Agreement before optional visual filtering, using the same human labels."""
        return score_evaluation_case(self.reference, self.predicted_before_visual_filter)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Retain evaluation data independently of the optional HTML view.

    Parameters
    ----------
    cases
        Analyzed human windows with their complete evidence, in input order.
    manifest, manifest_sha256
        Reference input path and digest. For annotation lists/directories the digest covers the
        generated, canonical JSON manifest, including source-video hashes.
    syncnet_initialization_seconds
        One-time model initialization duration, or None when disabled.
    unannotated_videos, unmatched_annotation_files
        Collection sources without labels and annotation video names outside the collection.
        Neither is treated as a negative example or included in the metrics.
    """

    cases: tuple[EvaluationCaseResult, ...]
    manifest: Path
    manifest_sha256: str
    syncnet_initialization_seconds: float | None = None
    unannotated_videos: tuple[Path, ...] = ()
    unmatched_annotation_files: tuple[PurePosixPath, ...] = ()

    @property
    def selected_segments(self) -> tuple[TalkingFaceSegment, ...]:
        """Candidate references with source path, source-second bounds, face ID and reasons."""
        return tuple(segment for case in self.cases for segment in case.analysis.selected_segments)

    @property
    def metrics(self) -> EvaluationMetrics:
        """Micro-aggregate duration-weighted metrics over the evaluated windows."""
        return combine_evaluation_metrics(tuple(case.metrics for case in self.cases))

    @property
    def metrics_before_visual_filter(self) -> EvaluationMetrics:
        """Micro-aggregate agreement before optional visual filtering."""
        return combine_evaluation_metrics(
            tuple(case.metrics_before_visual_filter for case in self.cases)
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save evaluation data as JSON compatible with the existing benchmark comparator.

        Parameters
        ----------
        path
            JSON destination. Parent directories are created; reference inputs are protected.
        overwrite
            Allow atomic replacement of an existing result. Defaults to refusing replacement.

        Returns
        -------
        pathlib.Path
            Saved report, including typed candidate references serialized as source intervals.
            Saving runs no models and exports no physical video clips.
        """
        from talkingfacekit.io.evaluation import save_evaluation_result

        return save_evaluation_result(self, path, overwrite=overwrite)


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Load and validate a version-1 benchmark manifest.

    Parameters
    ----------
    path
        UTF-8 JSON manifest containing a non-empty ``cases`` list.

    Returns
    -------
    tuple[EvaluationCase, ...]
        Validated cases in manifest order.

    Raises
    ------
    FileNotFoundError
        If the manifest is missing.
    TypeError, ValueError
        If its schema, timing, labels, or paths are invalid.
    """
    try:
        document = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid benchmark JSON: {error}") from error
    return parse_evaluation_cases(document)


def parse_evaluation_cases(document: object) -> tuple[EvaluationCase, ...]:
    """Validate a decoded version-1 manifest without reading files or loading models.

    Parameters
    ----------
    document
        Decoded JSON manifest with schema version 1 and a non-empty ``cases`` list.

    Returns
    -------
    tuple[EvaluationCase, ...]
        Validated cases in input order, on the original source timeline.

    Raises
    ------
    TypeError, ValueError
        If the schema, timing, labels, identifiers, or paths are invalid.
    """
    root = _mapping(document, "manifest")
    if root.get("schema_version") != 1 or isinstance(root.get("schema_version"), bool):
        raise ValueError("benchmark manifest requires schema_version 1")
    raw_cases = _array(root.get("cases"), "cases")
    if not raw_cases:
        raise ValueError("benchmark manifest requires at least one case")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        item = _mapping(raw_case, f"cases[{index}]")
        case_id = _nonempty_string(item.get("id"), f"cases[{index}].id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)
        file = _relative_video_path(item.get("file"), f"{case_id}.file")
        digest = _nonempty_string(item.get("sha256"), f"{case_id}.sha256")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{case_id}.sha256 must be a lowercase SHA-256 digest")
        window = EvaluationInterval(
            _number(item.get("start_seconds"), f"{case_id}.start_seconds"),
            _number(item.get("end_seconds"), f"{case_id}.end_seconds"),
        )
        selected = _intervals(item.get("selected"), f"{case_id}.selected", window)
        ignored = _intervals(item.get("ignore", []), f"{case_id}.ignore", window)
        if _overlap_seconds(selected, ignored) > 0:
            raise ValueError(f"{case_id}.selected and ignore must not overlap")
        cases.append(EvaluationCase(case_id, file, digest, window, selected, ignored))
    return tuple(cases)


def prepare_evaluation_manifest(
    annotations_path: Path,
    videos_dir: Path,
    *,
    include_files: frozenset[PurePosixPath] | None = None,
    video_aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a hashed manifest from simple human annotations without running models.

    Parameters
    ----------
    annotations_path
        UTF-8 JSON list, or a directory recursively containing such lists. Directory files are
        read in relative POSIX path order; rows retain their order within each file. Objects have
        ``file``, ``start_seconds``, ``end_seconds``, and ``selected`` (source-second pairs).
        Each file/window pair must be unique across all annotation files.
    videos_dir
        Local directory containing the referenced videos.
    include_files
        Optional collection scope. Other annotation video names are recorded in
        ``unmatched_annotation_files`` without requiring those media locally. Default is strict.
    video_aliases
        Explicit annotation filename to local relative filename mapping. Labels are unchanged.

    Returns
    -------
    dict[str, object]
        Validated version-1 manifest with automatically generated case IDs and file hashes.

    Raises
    ------
    FileNotFoundError
        If annotations or a required video are missing.
    TypeError, ValueError
        If annotations are empty or have invalid paths, intervals, or duplicate cases.
    """
    root = videos_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark video directory not found: {videos_dir}")
    cases: list[dict[str, object]] = []
    aliases = {
        _relative_video_path(source, "video alias source"): _relative_video_path(
            target, "video alias target"
        )
        for source, target in ({} if video_aliases is None else video_aliases).items()
    }
    unmatched: set[PurePosixPath] = set()
    seen_ids: set[str] = set()
    known_hashes: dict[Path, str] = {}
    for location, row in _annotation_rows(annotations_path.expanduser()):
        item = _mapping(row, location)
        unknown = item.keys() - {"file", "start_seconds", "end_seconds", "selected"}
        if unknown:
            raise ValueError(f"{location} has unknown fields: {', '.join(sorted(unknown))}")
        file = _relative_video_path(item.get("file"), f"{location}.file")
        window = EvaluationInterval(
            _number(item.get("start_seconds"), f"{location}.start_seconds"),
            _number(item.get("end_seconds"), f"{location}.end_seconds"),
        )
        selected = _intervals(item.get("selected"), f"{location}.selected", window)
        file = aliases.get(file, file)
        if include_files is not None and file not in include_files:
            unmatched.add(file)
            continue
        case_id = f"{file.as_posix()}:{window.start_seconds}-{window.end_seconds}"
        if case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case: {case_id} in {location}")
        seen_ids.add(case_id)
        video_path = root.joinpath(*file.parts).resolve()
        if not video_path.is_relative_to(root):
            raise ValueError(f"video escapes benchmark directory: {file}")
        if not video_path.is_file():
            raise FileNotFoundError(f"benchmark video not found: {video_path}")
        if video_path not in known_hashes:
            known_hashes[video_path] = _sha256_file(video_path)
        cases.append(
            {
                "id": case_id,
                "file": file.as_posix(),
                "sha256": known_hashes[video_path],
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "selected": [
                    [interval.start_seconds, interval.end_seconds] for interval in selected
                ],
            }
        )
    manifest: dict[str, object] = {"schema_version": 1, "cases": cases}
    if include_files is not None:
        manifest["unmatched_annotation_files"] = [file.as_posix() for file in sorted(unmatched)]
    return manifest


def _annotation_rows(path: Path) -> Iterator[tuple[str, object]]:
    """Read legacy lists or sorted directory lists, preserving source locations for errors."""
    sources = (
        sorted(path.rglob("*.json"), key=lambda item: item.relative_to(path).as_posix())
        if path.is_dir()
        else [path]
    )
    if not sources:
        raise ValueError(f"annotations directory contains no JSON files: {path}")
    for source in sources:
        try:
            document = cast(object, json.loads(source.read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid annotations JSON in {source}: {error}") from error
        rows = _array(document, f"annotations in {source}")
        if not rows:
            raise ValueError(f"annotations require at least one case: {source}")
        for index, row in enumerate(rows):
            yield f"{source}[{index}]", row


def verify_evaluation_videos(
    cases: tuple[EvaluationCase, ...], videos_dir: Path
) -> dict[str, Path]:
    """Resolve all source files and verify their complete-file SHA-256 digests.

    Parameters
    ----------
    cases
        Cases loaded from a benchmark manifest.
    videos_dir
        Local copy of the shared video folder; files may be nested below it.

    Returns
    -------
    dict[str, Path]
        Verified absolute file path keyed by case ID.

    Raises
    ------
    FileNotFoundError
        If the video directory or a required file does not exist.
    ValueError
        If a file escapes the directory or differs from the manifest digest.
    """
    root = videos_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark video directory not found: {videos_dir}")
    verified: dict[str, Path] = {}
    known_hashes: dict[Path, str] = {}
    for case in cases:
        path = root.joinpath(*case.file.parts).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"video for {case.case_id} escapes benchmark directory: {case.file}")
        if not path.is_file():
            raise FileNotFoundError(f"video for {case.case_id} not found: {path}")
        if path not in known_hashes:
            known_hashes[path] = _sha256_file(path)
        digest = known_hashes[path]
        if digest != case.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {case.case_id}: expected {case.sha256}, got {digest}"
            )
        verified[case.case_id] = path
    return verified


def score_evaluation_case(
    case: EvaluationCase, predicted: tuple[EvaluationInterval, ...]
) -> EvaluationMetrics:
    """Compare predicted candidate runs with human-selected intervals by duration.

    Parameters
    ----------
    case
        Human reference and scoring window.
    predicted
        Non-overlapping, ordered candidate intervals on the source timeline.

    Returns
    -------
    EvaluationMetrics
        Overlap, erroneous selection, missed usable time, and ratios.

    Raises
    ------
    ValueError
        If prediction intervals overlap or leave the case window.
    """
    _validate_ordered_intervals(predicted, "predicted", case.window)
    predicted_ignored = _overlap_seconds(predicted, case.ignored)
    predicted_seconds = sum(interval.duration_seconds for interval in predicted) - predicted_ignored
    selected_seconds = sum(interval.duration_seconds for interval in case.selected)
    true_positive = _overlap_seconds(predicted, case.selected)
    false_positive = max(0.0, predicted_seconds - true_positive)
    false_negative = max(0.0, selected_seconds - true_positive)
    return EvaluationMetrics(
        true_positive_seconds=true_positive,
        false_positive_seconds=false_positive,
        false_negative_seconds=false_negative,
        ignored_seconds=sum(interval.duration_seconds for interval in case.ignored),
        precision=true_positive / predicted_seconds if predicted_seconds > 0 else None,
        recall=true_positive / selected_seconds if selected_seconds > 0 else None,
    )


def combine_evaluation_metrics(metrics: tuple[EvaluationMetrics, ...]) -> EvaluationMetrics:
    """Micro-average case metrics by summing evaluated seconds.

    Parameters
    ----------
    metrics
        Metrics for one or more benchmark cases.

    Returns
    -------
    EvaluationMetrics
        Aggregate duration-weighted metrics.
    """
    true_positive = sum(item.true_positive_seconds for item in metrics)
    false_positive = sum(item.false_positive_seconds for item in metrics)
    false_negative = sum(item.false_negative_seconds for item in metrics)
    predicted = true_positive + false_positive
    selected = true_positive + false_negative
    return EvaluationMetrics(
        true_positive,
        false_positive,
        false_negative,
        sum(item.ignored_seconds for item in metrics),
        true_positive / predicted if predicted > 0 else None,
        true_positive / selected if selected > 0 else None,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as video:
        for chunk in iter(lambda: video.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _relative_video_path(value: object, name: str) -> PurePosixPath:
    file_text = _nonempty_string(value, name)
    file = PurePosixPath(file_text)
    if "\\" in file_text or ":" in file_text or file.is_absolute():
        raise ValueError(f"{name} must be a relative POSIX path")
    if any(part in {".", ".."} for part in file_text.split("/")):
        raise ValueError(f"{name} must stay inside the video directory")
    return file


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _intervals(
    value: object, name: str, window: EvaluationInterval
) -> tuple[EvaluationInterval, ...]:
    rows = _array(value, name)
    intervals: list[EvaluationInterval] = []
    for index, row in enumerate(rows):
        pair = _array(row, f"{name}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain [start_seconds, end_seconds]")
        intervals.append(
            EvaluationInterval(
                _number(pair[0], f"{name}[{index}][0]"),
                _number(pair[1], f"{name}[{index}][1]"),
            )
        )
    result = tuple(intervals)
    _validate_ordered_intervals(result, name, window)
    return result


def _validate_ordered_intervals(
    intervals: tuple[EvaluationInterval, ...], name: str, window: EvaluationInterval
) -> None:
    previous_end = window.start_seconds
    for interval in intervals:
        if interval.start_seconds < previous_end or interval.end_seconds > window.end_seconds:
            raise ValueError(f"{name} must be ordered, non-overlapping, and inside the case window")
        previous_end = interval.end_seconds


def _overlap_seconds(
    left: tuple[EvaluationInterval, ...], right: tuple[EvaluationInterval, ...]
) -> float:
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        first = left[left_index]
        second = right[right_index]
        total += max(
            0.0,
            min(first.end_seconds, second.end_seconds)
            - max(first.start_seconds, second.start_seconds),
        )
        if first.end_seconds <= second.end_seconds:
            left_index += 1
        else:
            right_index += 1
    return total
