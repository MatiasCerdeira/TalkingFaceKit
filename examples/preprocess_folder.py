"""Preprocess a folder through the public Python API and render its successful analyses.

Import ``preprocess_folder`` from an interactive session, or adjust the paths at the bottom.
Requires the active-speaker-deeptalk extra; SyncNet additionally requires its extra and an explicit
official local checkpoint. This example processes complete videos sequentially.
"""

from pathlib import Path

from talkingfacekit import (
    AnalysisConfig,
    CollectionAnalysisResult,
    CollectionProgress,
    TalkingFaceCollection,
    VisualPolicy,
)


def show_progress(progress: CollectionProgress) -> None:
    """Print one update after each source outcome has been saved."""
    outcome = progress.outcome
    if outcome.error is not None:
        detail = f"FAILED: {outcome.error.message}"
    else:
        assert outcome.result is not None
        detail = f"{len(outcome.result.selected_segments)} selected segments"
    print(f"[{progress.completed}/{progress.total}] {outcome.path.name}: {detail}")


def preprocess_folder(
    folder: Path, output_dir: Path, *, syncnet_model: Path | None = None
) -> CollectionAnalysisResult:
    """Analyze a folder, saving per-video JSON, progress and diagnostic HTML.

    Parameters
    ----------
    folder, output_dir
        Source directory and output directory. Existing output names are protected.
    syncnet_model
        Optional explicit official checkpoint; omitted configuration skips synchronization.

    Returns
    -------
    CollectionAnalysisResult
        Successful results, per-source errors and selected segments on original source timelines.
    """
    collection = TalkingFaceCollection.from_folder(folder)
    print(f"Found {len(collection)} videos")
    report = collection.analyze(
        AnalysisConfig(visual_policy=VisualPolicy(), syncnet_model=syncnet_model),
        output_dir=output_dir,
        on_progress=show_progress,
    )
    for outcome in report.items:
        if outcome.result is not None and outcome.output_path is not None:
            outcome.result.render_report(outcome.output_path.with_suffix(".html"))
    for segment in report.selected_segments:
        print(
            segment.source.path.name,
            f"{segment.start_seconds:.2f}–{segment.end_seconds:.2f}s",
            ", ".join(segment.reason_codes),
        )
    return report


if __name__ == "__main__":
    preprocess_folder(Path("data/eval/videos"), Path("reports/folder-demo"))
