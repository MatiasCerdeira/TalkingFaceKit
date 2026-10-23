import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from talkingfacekit import (
    ActiveSpeakerProvenance,
    ActiveSpeakerResult,
    ActiveSpeakerScoreWindow,
    AnalysisConfig,
    CandidateRun,
    CollectionProgress,
    TalkingFaceCollection,
    TalkingFaceSequence,
    VideoAnalysisOutcome,
    VideoAnalysisResult,
    VideoMetadata,
    VideoSource,
)
from talkingfacekit.collection import ErrorPolicy
from talkingfacekit.integrations import deeptalk

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def raw_result() -> ActiveSpeakerResult:
    return ActiveSpeakerResult(
        (),
        (),
        (ActiveSpeakerScoreWindow(0, 2, {1: 2.0}),),
        ActiveSpeakerProvenance("0.3.1"),
        candidate_runs=(
            CandidateRun(
                0, 2, "candidate", 1, ("structural_policy_passed",), 1, 50, 50, 1, 0, 2, 2, 2
            ),
        ),
    )


def test_folder_discovery_is_lazy_sorted_case_insensitive_and_explicitly_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("z.webm", "A.MP4", "b.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"not inspected during discovery")
    (nested / "clip.mov").write_bytes(b"also not inspected")
    (tmp_path / "directory.mp4").mkdir()

    def forbid(*args: object, **kwargs: object) -> ActiveSpeakerResult:
        pytest.fail("Discovery must not run inference")

    monkeypatch.setattr(deeptalk, "analyze_sequence", forbid)
    collection = TalkingFaceCollection.from_folder(tmp_path)
    assert [path.name for path in collection] == ["A.MP4", "b.mp4", "z.webm"]
    assert len(collection) == 3
    assert all(path.is_absolute() for path in collection)
    recursive = TalkingFaceCollection.from_folder(tmp_path, recursive=True)
    assert [path.relative_to(tmp_path).as_posix() for path in recursive] == [
        "A.MP4",
        "b.mp4",
        "nested/clip.mov",
        "z.webm",
    ]
    with pytest.raises(FileNotFoundError):
        TalkingFaceCollection.from_folder(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        TalkingFaceCollection.from_folder(tmp_path / "z.webm")


def test_collection_continues_records_progress_and_saves_completed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("a.webm", "c.webm"):
        shutil.copyfile(FIXTURE, videos / name)
    (videos / "b.mp4").write_bytes(b"corrupt")
    calls: list[Path] = []

    def analyze(sequence: TalkingFaceSequence, **kwargs: object) -> ActiveSpeakerResult:
        calls.append(sequence.source.path)
        return raw_result()

    monkeypatch.setattr(deeptalk, "analyze_sequence", analyze)
    output = tmp_path / "outputs"
    updates: list[CollectionProgress] = []

    def progress(update: CollectionProgress) -> None:
        updates.append(update)
        checkpoint = json.loads((output / "collection.json").read_text())
        assert checkpoint["completed_videos"] == update.completed
        assert checkpoint["total_videos"] == 3
        if update.outcome.output_path is not None:
            assert update.outcome.output_path.is_file()

    collection = TalkingFaceCollection.from_folder(videos)
    result = collection.analyze(output_dir=output, on_progress=progress)
    assert [path.name for path in calls] == ["a.webm", "c.webm"]
    assert [(item.completed, item.total) for item in updates] == [(1, 3), (2, 3), (3, 3)]
    assert len(result.results) == len(result.selected_segments) == 2
    assert len(result.failures) == 1
    assert result.failures[0].path.name == "b.mp4"
    assert result.failures[0].error_type == "ValueError"
    assert [segment.source.path.name for segment in result.selected_segments] == [
        "a.webm",
        "c.webm",
    ]
    assert (output / "0001-a.json").is_file()
    assert (output / "0003-c.json").is_file()
    saved = json.loads((output / "collection.json").read_text())
    assert [item["status"] for item in saved["videos"]] == ["completed", "failed", "completed"]
    assert saved["successful_videos"] == 2
    # Overwrite conflicts are detected before any model call.
    with pytest.raises(FileExistsError):
        collection.analyze(output_dir=output)
    assert len(calls) == 2


def test_raise_policy_saves_partial_summary_and_reraises_original_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, bad, later = (tmp_path / name for name in ("a.webm", "b.webm", "c.webm"))
    for path in (good, bad, later):
        shutil.copyfile(FIXTURE, path)
    failure = RuntimeError("backend failed")
    calls: list[Path] = []

    def analyze(sequence: TalkingFaceSequence, **kwargs: object) -> ActiveSpeakerResult:
        calls.append(sequence.source.path)
        if sequence.source.path == bad:
            raise failure
        return raw_result()

    monkeypatch.setattr(deeptalk, "analyze_sequence", analyze)
    output = tmp_path / "outputs"
    collection = TalkingFaceCollection.from_videos([good, bad, later])
    with pytest.raises(RuntimeError) as error:
        collection.analyze(output_dir=output, on_error="raise")
    assert error.value is failure
    assert calls == [good, bad]
    saved = json.loads((output / "collection.json").read_text())
    assert saved["completed_videos"] == 2
    assert saved["total_videos"] == 3
    assert saved["videos"][1]["error"]["message"] == "backend failed"
    assert (output / "0001-a.json").is_file()


def test_empty_collection_runs_no_models_even_with_explicit_sync_model(tmp_path: Path) -> None:
    collection = TalkingFaceCollection.from_folder(tmp_path)
    result = collection.analyze(
        AnalysisConfig(syncnet_model=Path("missing.model")), output_dir=tmp_path / "outputs"
    )
    assert not result.items
    assert not result.results
    assert not result.failures
    assert not result.selected_segments
    assert result.total_videos == 0
    assert (tmp_path / "outputs" / "collection.json").is_file()


def test_zero_selection_is_a_success_and_callbacks_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = ActiveSpeakerResult((), (), (), ActiveSpeakerProvenance("0.3.1"))
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda *args, **kwargs: raw)
    collection = TalkingFaceCollection.from_videos([FIXTURE])
    result = collection.analyze()
    assert len(result.results) == 1
    assert not result.selected_segments
    assert not result.failures

    def fail(update: CollectionProgress) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        collection.analyze(on_progress=fail)


def test_duplicate_stems_have_distinct_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = (tmp_path / "one" / "same.webm", tmp_path / "two" / "same.webm")
    for path in paths:
        path.parent.mkdir()
        shutil.copyfile(FIXTURE, path)
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda *args, **kwargs: raw_result())
    report = TalkingFaceCollection.from_videos(paths).analyze(output_dir=tmp_path / "outputs")
    assert [item.output_path.name for item in report.items if item.output_path is not None] == [
        "0001-same.json",
        "0002-same.json",
    ]


def test_invalid_options_and_outcomes_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        TalkingFaceCollection.from_videos([FIXTURE, FIXTURE])
    with pytest.raises(ValueError, match="on_error"):
        TalkingFaceCollection.from_videos([]).analyze(on_error=cast(ErrorPolicy, "ignore"))
    with pytest.raises(ValueError, match="exactly one"):
        VideoAnalysisOutcome(FIXTURE)
    result = VideoAnalysisResult(
        VideoSource(FIXTURE, VideoMetadata(640, 480, 25, 2, True)), 0, None, raw_result()
    )
    with pytest.raises(ValueError, match="match its result"):
        VideoAnalysisOutcome(Path("another.mp4"), result=result)


def test_keyboard_interrupt_is_not_converted_to_source_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: object, **kwargs: object) -> ActiveSpeakerResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(deeptalk, "analyze_sequence", interrupt)
    with pytest.raises(KeyboardInterrupt):
        TalkingFaceCollection.from_videos([FIXTURE]).analyze()
