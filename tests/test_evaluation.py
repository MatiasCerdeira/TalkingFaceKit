import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from talkingfacekit import cli
from talkingfacekit.evaluation import (
    EvaluationCase,
    EvaluationInterval,
    combine_evaluation_metrics,
    load_evaluation_cases,
    prepare_evaluation_manifest,
    score_evaluation_case,
    verify_evaluation_videos,
)
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkCandidateRun,
    DeepTalkScoreWindow,
)
from talkingfacekit.sequence import TalkingFaceSequence


def write_manifest(path: Path, *, file: str, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "sample",
                        "file": file,
                        "sha256": digest,
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "selected": [[0.2, 0.8]],
                        "ignore": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_prepare_cli_generates_a_manifest_accepted_by_verification(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "interview.mp4"
    video.write_bytes(b"shared video fixture")
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": video.name,
                    "start_seconds": 30,
                    "end_seconds": 90,
                    "selected": [[34.2, 46.8], [58.1, 71.5]],
                },
                {
                    "file": video.name,
                    "start_seconds": 100,
                    "end_seconds": 160,
                    "selected": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    command = [
        "prepare-benchmark",
        str(annotations_path),
        "--videos-dir",
        str(video_dir),
        "--output",
        str(manifest_path),
    ]
    assert cli.main(command) == 0
    cases = load_evaluation_cases(manifest_path)
    assert len({case.case_id for case in cases}) == 2
    assert all(case.sha256 == hashlib.sha256(video.read_bytes()).hexdigest() for case in cases)
    assert cases[0].window == EvaluationInterval(30, 90)
    assert cases[0].selected == (EvaluationInterval(34.2, 46.8), EvaluationInterval(58.1, 71.5))
    assert cases[1].selected == ()
    assert (
        cli.main(
            [
                "evaluate-benchmark",
                str(manifest_path),
                "--videos-dir",
                str(video_dir),
                "--verify-only",
            ]
        )
        == 0
    )

    saved_manifest = manifest_path.read_bytes()
    with pytest.raises(SystemExit):
        cli.main(command)
    assert manifest_path.read_bytes() == saved_manifest
    assert cli.main([*command, "--overwrite"]) == 0
    with pytest.raises(SystemExit):
        cli.main([*command[:-1], str(annotations_path), "--overwrite"])
    assert isinstance(json.loads(annotations_path.read_text(encoding="utf-8")), list)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("file", "../outside.mp4", "stay inside"),
        ("selected", [[0.0, 2.0]], "inside the case window"),
        ("selected", None, "JSON array"),
        ("selected", [[0.3, 0.8], [0.4, 0.9]], "non-overlapping"),
    ],
)
def test_preparation_rejects_invalid_annotations(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    annotation: dict[str, object] = {
        "file": "sample.mp4",
        "start_seconds": 0,
        "end_seconds": 1,
        "selected": [],
    }
    annotation[field] = value
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([annotation]), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_evaluation_manifest(path, tmp_path)


def test_scores_seconds_and_excludes_ignored_regions() -> None:
    case = EvaluationCase(
        case_id="mixed",
        file=PurePosixPath("mixed.mp4"),
        sha256="0" * 64,
        window=EvaluationInterval(0.0, 10.0),
        selected=(EvaluationInterval(2.0, 6.0),),
        ignored=(EvaluationInterval(8.0, 9.0),),
    )
    metrics = score_evaluation_case(
        case,
        (EvaluationInterval(1.0, 4.0), EvaluationInterval(5.0, 10.0)),
    )

    assert metrics.true_positive_seconds == pytest.approx(3.0)
    assert metrics.false_positive_seconds == pytest.approx(4.0)
    assert metrics.false_negative_seconds == pytest.approx(1.0)
    assert metrics.ignored_seconds == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(3 / 7)
    assert metrics.recall == pytest.approx(3 / 4)


def test_negative_only_case_reports_undefined_recall() -> None:
    case = EvaluationCase(
        case_id="negative",
        file=PurePosixPath("negative.mp4"),
        sha256="0" * 64,
        window=EvaluationInterval(0.0, 5.0),
        selected=(),
        ignored=(),
    )
    metrics = score_evaluation_case(case, (EvaluationInterval(1.0, 3.0),))

    assert metrics.false_positive_seconds == 2.0
    assert metrics.precision == 0.0
    assert metrics.recall is None
    aggregate = combine_evaluation_metrics((metrics,))
    assert aggregate.precision == 0.0
    assert aggregate.recall is None


def test_manifest_and_video_hash_must_match(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "sample.mp4"
    video.write_bytes(b"shared fixture")
    manifest_path = tmp_path / "manifest.json"
    expected_digest = hashlib.sha256(video.read_bytes()).hexdigest()
    write_manifest(manifest_path, file=video.name, digest=expected_digest)

    cases = load_evaluation_cases(manifest_path)
    assert verify_evaluation_videos(cases, video_dir) == {"sample": video}

    video.write_bytes(b"different fixture")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_evaluation_videos(cases, video_dir)


def test_manifest_rejects_path_traversal_and_label_overlap(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, file="../outside.mp4", digest="0" * 64)
    with pytest.raises(ValueError, match="stay inside"):
        load_evaluation_cases(manifest_path)

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["cases"][0]["file"] = "sample.mp4"
    document["cases"][0]["ignore"] = [[0.5, 0.9]]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap"):
        load_evaluation_cases(manifest_path)


def test_cli_verifies_then_scores_without_real_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "example1.webm"
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        file=fixture.name,
        digest=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    )
    lock_path = tmp_path / "manifest.sha256"
    lock_path.write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n")
    report_path = tmp_path / "report.json"
    analyzed: list[tuple[float, float | None]] = []

    def fake_analyze(sequence: TalkingFaceSequence) -> DeepTalkAnalysisResult:
        analyzed.append((sequence.start_seconds, sequence.end_seconds))
        return DeepTalkAnalysisResult(
            face_observations=(),
            speech_intervals=(),
            score_windows=(DeepTalkScoreWindow(0.0, 1.0, {}),),
            provenance=DeepTalkAnalysisProvenance(backend_version="0.3.1"),
            candidate_runs=(
                DeepTalkCandidateRun(
                    source_start_seconds=0.2,
                    source_end_seconds=0.8,
                    status="candidate",
                    face_id=1,
                    reason_codes=("structural_policy_passed",),
                    preliminary_segment_count=1,
                    face_observation_count=15,
                    expected_face_observation_count=15,
                    face_visibility_fraction=1.0,
                    maximum_face_gap_seconds=0.04,
                    raw_score_min=1.0,
                    raw_score_mean=1.0,
                    raw_score_max=1.0,
                ),
            ),
        )

    monkeypatch.setattr(cli, "analyze_sequence", fake_analyze)
    command = [
        "evaluate-benchmark",
        str(manifest_path),
        "--videos-dir",
        str(fixture.parent),
        "--manifest-lock",
        str(lock_path),
    ]
    assert cli.main([*command, "--verify-only"]) == 0
    assert not analyzed

    assert cli.main([*command, "--output", str(report_path)]) == 0
    assert analyzed == [(0.0, 1.0)]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall"]["precision"] == 1.0
    assert report["overall"]["recall"] == 1.0
    assert report["manifest_sha256"] == lock_path.read_text().strip()
    assert report["cases"][0]["predicted"] == [[0.2, 0.8]]
    assert "overall: precision=100.0% recall=100.0%" in capsys.readouterr().out

    lock_path.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*command, "--verify-only"])
    assert exit_info.value.code == 2
    assert "manifest SHA-256 mismatch" in capsys.readouterr().err
