"""Lightweight video collections and typed outcomes for sequential analysis."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from talkingfacekit.evaluation import EvaluationCaseResult, EvaluationResult

from talkingfacekit.analysis import AnalysisConfig, TalkingFaceSegment, VideoAnalysisResult

ErrorPolicy = Literal["continue", "raise"]


@dataclass(frozen=True, slots=True)
class AnalysisFailure:
    """Describe a failed source without retaining an exception or its traceback.

    Attributes
    ----------
    path
        Source whose inspection or analysis failed.
    error_type, message
        Python exception class name and diagnostic text. A failure is distinct from successful
        analysis that selected no segments.
    """

    path: Path
    error_type: str
    message: str

    def __post_init__(self) -> None:
        """Normalize the source path and require textual diagnostics."""
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("failure error_type must be nonempty text")
        if not isinstance(self.message, str):
            raise TypeError("failure message must be text")


@dataclass(frozen=True, slots=True)
class VideoAnalysisOutcome:
    """Record exactly one successful result or failure in collection input order.

    Attributes
    ----------
    path
        Input source path.
    result, error
        Exactly one is present. Successful zero-selection results remain successes.
    output_path
        Saved per-video JSON, when persistence was requested and analysis succeeded.
    """

    path: Path
    result: VideoAnalysisResult | None = None
    error: AnalysisFailure | None = None
    output_path: Path | None = None

    def __post_init__(self) -> None:
        """Require one outcome and a matching source identity."""
        object.__setattr__(self, "path", Path(self.path))
        if (self.result is None) == (self.error is None):
            raise ValueError("an outcome requires exactly one result or error")
        if self.result is not None and self.result.source.path != self.path:
            raise ValueError("outcome path must match its result source")
        if self.error is not None and self.error.path != self.path:
            raise ValueError("outcome path must match its error source")
        if self.output_path is not None and self.result is None:
            raise ValueError("saved output requires a successful result")


@dataclass(frozen=True, slots=True)
class CollectionProgress:
    """Report a completed source after its optional output has been saved.

    Attributes
    ----------
    completed, total
        Completed and planned source counts, including per-source failures.
    outcome
        Most recent source's typed success or failure.
    """

    completed: int
    total: int
    outcome: VideoAnalysisOutcome

    def __post_init__(self) -> None:
        """Require valid completed/planned counts and a recorded outcome."""
        for value in (self.completed, self.total):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("progress counts must be positive integers")
        if self.completed > self.total:
            raise ValueError("completed count must not exceed total")


@dataclass(frozen=True, slots=True)
class CollectionAnalysisResult:
    """Retain ordered successes and failures without media pixel buffers.

    Attributes
    ----------
    items
        One outcome per completed source, in input order.
    total_videos
        Planned count. A saved summary during a run may contain fewer completed items.
    """

    items: tuple[VideoAnalysisOutcome, ...]
    total_videos: int

    def __post_init__(self) -> None:
        """Validate progress and freeze the outcome sequence."""
        object.__setattr__(self, "items", tuple(self.items))
        if (
            isinstance(self.total_videos, bool)
            or not isinstance(self.total_videos, int)
            or self.total_videos < len(self.items)
        ):
            raise ValueError("total_videos must be an integer at least as large as completed items")

    @property
    def results(self) -> tuple[VideoAnalysisResult, ...]:
        """Successful analyses, including videos with no selected segments."""
        return tuple(item.result for item in self.items if item.result is not None)

    @property
    def failures(self) -> tuple[AnalysisFailure, ...]:
        """Per-source failures with no successful analysis result."""
        return tuple(item.error for item in self.items if item.error is not None)

    @property
    def selected_segments(self) -> tuple[TalkingFaceSegment, ...]:
        """Retained segments across sources; every segment identifies its own source."""
        return tuple(segment for result in self.results for segment in result.selected_segments)

    def save(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        """Save a JSON summary of source outcomes and selected intervals.

        Parameters
        ----------
        output_path
            Destination ending in ``.json``. Parents are created as needed.
        overwrite
            Whether an existing summary may be replaced atomically.

        Returns
        -------
        pathlib.Path
            Summary path. Full evidence lives in the per-video JSON outputs when requested.
        """
        from talkingfacekit.io.analysis import save_collection_result

        return save_collection_result(self, output_path, overwrite=overwrite)


@dataclass(frozen=True, slots=True)
class TalkingFaceCollection:
    """Identify videos to inspect and analyze, without eagerly opening all media.

    Parameters
    ----------
    videos
        Ordered source paths. Construction normalizes paths but performs no media I/O. Each
        source becomes a ``TalkingFaceSequence`` only when its analysis starts, allowing corrupt
        files to be represented as individual failures rather than preventing folder loading.
    """

    videos: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Snapshot paths and reject duplicate input identities without opening files."""
        paths = tuple(Path(path).absolute() for path in self.videos)
        if len(set(paths)) != len(paths):
            raise ValueError("collection videos must not contain duplicate paths")
        object.__setattr__(self, "videos", paths)

    @classmethod
    def from_folder(cls, folder: str | Path, *, recursive: bool = False) -> TalkingFaceCollection:
        """Discover video paths deterministically without decoding or running models.

        Parameters
        ----------
        folder
            Existing local directory. An empty directory produces an empty collection.
        recursive
            Whether to include descendants. Defaults to files directly inside the directory.

        Returns
        -------
        TalkingFaceCollection
            Absolute paths sorted by relative POSIX path, with case-insensitive video extensions.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        NotADirectoryError
            If ``folder`` is not a directory.
        """
        from talkingfacekit.io.collection import discover_videos

        return cls(discover_videos(folder, recursive=recursive))

    @classmethod
    def from_videos(cls, videos: Iterable[str | Path]) -> TalkingFaceCollection:
        """Snapshot explicitly chosen source paths in caller order, without opening files.

        Parameters
        ----------
        videos
            Finite iterable of local video paths. Unsupported or corrupt media fail at analysis.

        Returns
        -------
        TalkingFaceCollection
            Lightweight collection accepting arbitrary container suffixes when explicitly chosen.
        """
        return cls(tuple(Path(path) for path in videos))

    def __len__(self) -> int:
        """Number of discovered or explicitly supplied video paths."""
        return len(self.videos)

    def __iter__(self) -> Iterator[Path]:
        """Iterate source paths in processing order without opening media."""
        return iter(self.videos)

    def analyze(
        self,
        config: AnalysisConfig | None = None,
        *,
        output_dir: str | Path | None = None,
        on_error: ErrorPolicy = "continue",
        on_progress: Callable[[CollectionProgress], None] | None = None,
        overwrite: bool = False,
    ) -> CollectionAnalysisResult:
        """Analyze each source sequentially with explicit progress and failure handling.

        Parameters
        ----------
        config
            Shared stage configuration, or structural-only defaults.
        output_dir
            Optional directory for one full JSON per success and ``collection.json``. The summary
            is updated atomically after each outcome; completed files survive a later failure.
            Output names include input order to distinguish identical file stems.
        on_error
            ``continue`` records inspection/model exceptions; ``raise`` records then reraises the
            first failure. Configuration, persistence and callback errors always propagate.
        on_progress
            Optional callback after each recorded/saved outcome. Collection orchestration does
            not print or configure logging; the experimental backend may emit its own messages.
            KeyboardInterrupt always propagates.
        overwrite
            Explicit permission to replace this run's output names. Conflicts are checked before
            inference. No automatic cache lookup or resume is performed.

        Returns
        -------
        CollectionAnalysisResult
            Ordered results and failures. An empty collection runs no models.
        """
        from talkingfacekit.processing import analyze_collection

        return analyze_collection(
            self,
            config,
            output_dir=output_dir,
            on_error=on_error,
            on_progress=on_progress,
            overwrite=overwrite,
        )

    def evaluate(
        self,
        reference_path: str | Path,
        *,
        videos_dir: str | Path,
        config: AnalysisConfig | None = None,
        video_aliases: Mapping[str, str] | None = None,
        on_progress: Callable[[EvaluationCaseResult], None] | None = None,
    ) -> EvaluationResult:
        """Analyze this collection's annotated windows and compare them with human labels.

        Parameters
        ----------
        reference_path
            Hashed benchmark manifest, simple annotation JSON list, or annotation directory.
        videos_dir
            Root used by annotation-relative filenames. Collection sources must be below it.
        config
            Same stage configuration as ``analyze``; default is structural-only analysis.
        video_aliases
            Explicit annotation filename to local relative filename mapping for simple labels.
        on_progress
            Optional callback with each completed, scored window. Errors propagate.

        Returns
        -------
        EvaluationResult
            Typed candidates, full analyses and metrics. Unannotated sources and unmatched
            annotation filenames are listed separately and excluded from scoring. No matching
            windows raises ValueError before models load. Human annotations are never changed.
        """
        from talkingfacekit.processing import evaluate_benchmark

        return evaluate_benchmark(
            reference_path,
            videos_dir,
            config,
            collection=self,
            video_aliases=video_aliases,
            on_progress=on_progress,
        )
