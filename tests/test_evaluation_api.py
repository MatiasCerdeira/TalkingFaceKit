import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import pytest

from talkingfacekit import (
    ActiveSpeakerProvenance,
    ActiveSpeakerResult,
    ActiveSpeakerScoreWindow,
    CandidatePolicy,
    CandidateRun,
    EvaluationCaseResult,
    TalkingFaceCollection,
    TalkingFaceSequence,
    evaluate_benchmark,
)
from talkingfacekit.demo import render_benchmark_report
from talkingfacekit.evaluation import prepare_evaluation_manifest
from talkingfacekit.integrations import deeptalk

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


@pytest.fixture
def analyzed_windows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    calls: list[tuple[float, float]] = []

    def analyze(
        sequence: TalkingFaceSequence, *, candidate_policy: CandidatePolicy | None = None
    ) -> ActiveSpeakerResult:
        start, end = sequence.start_seconds, sequence.end_seconds
        assert end is not None
        calls.append((start, end))
        return ActiveSpeakerResult(
            (),
            (),
            (ActiveSpeakerScoreWindow(start, end, {1: 2.0}),),
            ActiveSpeakerProvenance("test"),
            candidate_runs=(
                CandidateRun(
                    start + 0.1,
                    end - 0.1,
                    "candidate",
                    1,
                    ("structural_policy_passed",),
                    1,
                    8,
                    8,
                    1,
                    0,
                    2,
                    2,
                    2,
                ),
            ),
        )

    monkeypatch.setattr(deeptalk, "analyze_sequence", analyze)
    return calls


def annotation(file: str, start: float = 0, end: float = 1) -> dict[str, object]:
    return {
        "file": file,
        "start_seconds": start,
        "end_seconds": end,
        "selected": [[start + 0.1, end - 0.1]],
    }


def test_collection_evaluates_only_labeled_windows_and_reuses_data_in_existing_comparator(
    tmp_path: Path, analyzed_windows: list[tuple[float, float]]
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    source = videos / "iphone.webm"
    shutil.copyfile(FIXTURE, source)
    unannotated = videos / "unannotated.webm"
    shutil.copyfile(FIXTURE, unannotated)
    labels = tmp_path / "annotations"
    labels.mkdir()
    human = labels / "human.json"
    human.write_text(
        json.dumps(
            [
                annotation("old-phone.webm", 0, 0.5),
                annotation("old-phone.webm", 0.5, 1),
                annotation("missing.webm"),
            ]
        )
    )
    original = human.read_bytes()
    completed: list[EvaluationCaseResult] = []
    result = TalkingFaceCollection.from_folder(videos).evaluate(
        labels,
        videos_dir=videos,
        video_aliases={"old-phone.webm": "iphone.webm"},
        on_progress=completed.append,
    )
    assert analyzed_windows == [(0, 0.5), (0.5, 1)]
    assert tuple(completed) == result.cases
    assert result.unannotated_videos == (unannotated,)
    assert result.unmatched_annotation_files == (PurePosixPath("missing.webm"),)
    assert result.metrics.precision == result.metrics.recall == 1
    assert result.metrics_before_visual_filter == result.metrics
    clips = result.selected_segments
    assert len(clips) == 2
    assert [(clip.start_seconds, clip.end_seconds) for clip in clips] == [(0.1, 0.4), (0.6, 0.9)]
    assert all(clip.source.path == source for clip in clips)
    assert clips[0].to_sequence().source is clips[0].source

    output = result.save(tmp_path / "demo" / "demo.json")
    document = json.loads(output.read_text())
    assert document["selected_segments"][0]["source"] == str(source)
    assert document["unannotated_videos"] == [str(unannotated)]
    assert document["unmatched_annotation_files"] == ["missing.webm"]
    html = render_benchmark_report(output, videos, output.with_suffix(".html"))
    payload = json.loads(html.read_text().split("const REPORT = ", 1)[1].split(";\n", 1)[0])
    assert len(payload["cases"]) == 2
    assert payload["cases"][1]["window"] == [0.5, 1]
    assert payload["overall"]["precision"] == payload["overall"]["recall"] == 1
    assert document["cases"][0]["diagnostics"]["candidate_runs"][0]["reason_codes"] == [
        "structural_policy_passed"
    ]
    assert analyzed_windows == [(0, 0.5), (0.5, 1)]  # Saving/rendering never repeats inference.
    assert human.read_bytes() == original
    with pytest.raises(FileExistsError):
        result.save(output)
    with pytest.raises(ValueError, match="outside the annotations"):
        result.save(labels / "result.json", overwrite=True)
    with pytest.raises(ValueError):
        result.save(human, overwrite=True)
    assert human.read_bytes() == original


def test_public_manifest_evaluation_preserves_hashes_and_detects_media_drift(
    tmp_path: Path, analyzed_windows: list[tuple[float, float]]
) -> None:
    source = tmp_path / "sample.webm"
    shutil.copyfile(FIXTURE, source)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([annotation(source.name)]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(prepare_evaluation_manifest(labels, tmp_path)))
    result = evaluate_benchmark(manifest, tmp_path)
    assert result.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result.metrics.precision == result.metrics.recall == 1
    assert analyzed_windows == [(0, 1)]
    source.write_bytes(b"changed media")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_benchmark(manifest, tmp_path)
    assert analyzed_windows == [(0, 1)]
    with pytest.raises(ValueError, match="not hashed manifests"):
        evaluate_benchmark(manifest, tmp_path, video_aliases={source.name: "other.webm"})


@pytest.mark.parametrize("scope", [False, True])
def test_missing_annotations_fail_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: bool
) -> None:
    from talkingfacekit import processing

    def forbid_model(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid references must fail before model construction")

    monkeypatch.setattr(processing, "VideoAnalyzer", forbid_model)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([annotation("missing.webm")]))
    # Keep strict standalone behavior: missing media raises; a collection with no matches is empty.
    if scope:
        with pytest.raises(ValueError, match="at least one case"):
            TalkingFaceCollection.from_videos(()).evaluate(labels, videos_dir=tmp_path)
    else:
        with pytest.raises(FileNotFoundError, match="video not found"):
            evaluate_benchmark(labels, tmp_path)


def test_aliases_cannot_escape_video_directory(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([annotation("phone.webm")]))
    with pytest.raises(ValueError, match="inside the video directory"):
        evaluate_benchmark(labels, tmp_path, video_aliases={"phone.webm": "../outside.webm"})
