import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict, cast

import pytest

from talkingfacekit import cli
from talkingfacekit.demo.benchmark_report import render_benchmark_report


class SourceDocument(TypedDict):
    schema_version: int
    code_revision: str
    cases: list[dict[str, object]]
    overall: dict[str, object]


class Diagnostics(TypedDict):
    preliminary_segments: list[dict[str, object]]
    candidate_runs: list[dict[str, object]]
    shot_boundaries: list[float]


class RenderedCase(TypedDict):
    id: str
    selected: list[list[float]]
    window: list[float]
    video_url: str
    regions: list[dict[str, object]]
    metrics: dict[str, float | None]
    diagnostics: Diagnostics | None


class Payload(TypedDict):
    cases: list[RenderedCase]
    overall: dict[str, float | None]


def write_result(tmp_path: Path) -> tuple[Path, Path, SourceDocument]:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "entrevista ágil.webm"
    video.write_bytes(b"synthetic media fixture: no decoding required")
    document: SourceDocument = {
        "schema_version": 1,
        "code_revision": "test-revision",
        "cases": [
            {
                "id": "interview:30-40",
                "file": video.name,
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "window": [30, 40],
                "selected": [[32, 36]],
                "predicted": [[31, 34], [35, 40]],
                "ignore": [[38, 39]],
                "metrics": {"precision": 123},
            }
        ],
        "overall": {"precision": 456},
    }
    result = tmp_path / "result.json"
    result.write_text(json.dumps(document), encoding="utf-8")
    return result, video_dir, document


def read_payload(path: Path) -> Payload:
    source = path.read_text(encoding="utf-8")
    payload = source.split("const REPORT = ", 1)[1].split(";\n", 1)[0]
    return cast(Payload, json.loads(payload))


def test_comparison_matches_scoring_and_ignored_time_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, videos, _ = write_result(tmp_path)

    def forbid_analysis(*args: object, **kwargs: object) -> None:
        pytest.fail("Rendering a saved run must not analyze videos")

    monkeypatch.setattr(cli, "analyze_sequence", forbid_analysis)
    output = tmp_path / "comparison.html"
    assert (
        cli.main(
            ["render-benchmark", str(result), "--videos-dir", str(videos), "--output", str(output)]
        )
        == 0
    )
    payload = read_payload(output)
    case = payload["cases"][0]
    assert case["regions"] == [
        {"start": 30, "end": 31, "status": "both_rejected"},
        {"start": 31, "end": 32, "status": "false_positive"},
        {"start": 32, "end": 34, "status": "both_selected"},
        {"start": 34, "end": 35, "status": "false_negative"},
        {"start": 35, "end": 36, "status": "both_selected"},
        {"start": 36, "end": 38, "status": "false_positive"},
        {"start": 38, "end": 39, "status": "ignored"},
        {"start": 39, "end": 40, "status": "false_positive"},
    ]
    assert case["metrics"]["false_positive_seconds"] == 4
    assert case["metrics"]["false_negative_seconds"] == 1
    assert case["metrics"]["precision"] == pytest.approx(3 / 7)
    assert payload["overall"] == case["metrics"]
    assert case["diagnostics"] is None
    assert case["video_url"] == (videos / "entrevista ágil.webm").as_uri()
    assert case["selected"] == [[32, 36]]


@pytest.mark.parametrize(
    ("selected", "predicted", "ignored", "status", "precision", "recall"),
    [
        ([], [], [], "both_rejected", None, None),
        ([[30, 40]], [], [], "false_negative", None, 0),
        ([], [[30, 40]], [], "false_positive", 0, None),
        ([], [[30, 40]], [[30, 40]], "ignored", None, None),
        ([[30, 32], [32, 40]], [[30, 40]], [], "both_selected", 1, 1),
    ],
)
def test_empty_and_touching_selections_keep_defined_metric_semantics(
    tmp_path: Path,
    selected: list[list[int]],
    predicted: list[list[int]],
    ignored: list[list[int]],
    status: str,
    precision: float | None,
    recall: float | None,
) -> None:
    result, videos, document = write_result(tmp_path)
    case = document["cases"][0]
    case.update(selected=selected, predicted=predicted, ignore=ignored)
    result.write_text(json.dumps(document))
    output = render_benchmark_report(result, videos, tmp_path / "report.html")
    actual = read_payload(output)["cases"][0]
    assert actual["regions"] == [{"start": 30, "end": 40, "status": status}]
    assert actual["metrics"]["precision"] == precision
    assert actual["metrics"]["recall"] == recall
    assert actual["selected"] == selected


def test_report_keeps_diagnostics_and_escapes_script_content(tmp_path: Path) -> None:
    result, videos, document = write_result(tmp_path)
    case = document["cases"][0]
    case["id"] = '</script><script>alert("unsafe")</script>'
    case["diagnostics"] = {
        "preliminary_segments": [
            {
                "source_start_seconds": 30,
                "source_end_seconds": 40,
                "status": "rejected",
                "face_id": None,
                "reason_codes": ["multiple_visible_faces"],
                "visible_face_ids": [1, 2],
                "raw_score": None,
            }
        ],
        "candidate_runs": [
            {
                "source_start_seconds": 31,
                "source_end_seconds": 34,
                "status": "candidate",
                "face_id": 1,
                "reason_codes": ["structural_policy_passed"],
                "face_visibility_fraction": 0.96,
                "maximum_face_gap_seconds": 0.04,
                "raw_score_mean": 1.2,
                "bridged_gap_intervals": [[32, 33]],
            }
        ],
        "shot_boundaries": [34],
    }
    result.write_text(json.dumps(document))
    output = render_benchmark_report(result, videos, tmp_path / "report.html")
    assert str(case["id"]) not in output.read_text()
    actual = read_payload(output)["cases"][0]
    assert actual["id"] == case["id"]
    diagnostic = actual["diagnostics"]
    assert diagnostic is not None
    assert diagnostic["preliminary_segments"][0]["reasons"] == ["multiple_visible_faces"]
    assert diagnostic["candidate_runs"][0]["bridged_gaps"] == [[32, 33]]
    assert diagnostic["shot_boundaries"] == [34]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predicted", [[29, 32]]),
        ("predicted", [[32, 41]]),
        ("predicted", [[31, 34], [33, 37]]),
        ("predicted", [[31, float("nan")]]),
        ("predicted", [[True, 35]]),
        ("ignore", [[33, 34]]),
        ("file", "../outside.mp4"),
        ("diagnostics", {"shot_boundaries": [41]}),
    ],
)
def test_rejects_invalid_run_without_replacing_an_existing_report(
    tmp_path: Path, field: str, value: object
) -> None:
    result, videos, document = write_result(tmp_path)
    document["cases"][0][field] = value
    result.write_text(json.dumps(document))
    output = tmp_path / "report.html"
    output.write_text("previous report")
    with pytest.raises((ValueError, TypeError)):
        render_benchmark_report(result, videos, output, overwrite=True)
    assert output.read_text() == "previous report"


def test_verifies_hashes_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    result, videos, _ = write_result(tmp_path)
    output = render_benchmark_report(result, videos, tmp_path / "report.html")
    previous = output.read_text()
    with pytest.raises(FileExistsError):
        render_benchmark_report(result, videos, output)
    assert render_benchmark_report(result, videos, output, overwrite=True) == output
    (videos / "entrevista ágil.webm").write_bytes(b"changed video")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        render_benchmark_report(result, videos, output, overwrite=True)
    assert output.read_text() == previous


def test_multiple_windows_of_one_video_keep_distinct_cases_and_global_metrics(
    tmp_path: Path,
) -> None:
    result, videos, document = write_result(tmp_path)
    second = dict(document["cases"][0])
    second.update(id="second", window=[50, 60], selected=[], predicted=[], ignore=[])
    document["cases"].append(second)
    result.write_text(json.dumps(document))
    output = render_benchmark_report(result, videos, tmp_path / "report.html")
    payload = read_payload(output)
    assert len(payload["cases"]) == 2
    assert payload["cases"][1]["window"] == [50, 60]
    assert payload["overall"]["precision"] == pytest.approx(3 / 7)
    assert payload["overall"]["recall"] == pytest.approx(3 / 4)


def test_generated_javascript_navigation_with_offline_media_stub(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required only for the offline JavaScript interaction test")
    result, videos, document = write_result(tmp_path)
    document["cases"][0]["diagnostics"] = {
        "preliminary_segments": [
            {
                "source_start_seconds": 34,
                "source_end_seconds": 35,
                "status": "rejected",
                "face_id": None,
                "reason_codes": ["multiple_visible_faces"],
                "visible_face_ids": [1, 2],
                "raw_score": None,
            }
        ]
    }
    second = dict(document["cases"][0])
    second.update(id="second", window=[50, 60], selected=[], predicted=[], ignore=[])
    second.pop("diagnostics")
    document["cases"].append(second)
    result.write_text(json.dumps(document))
    output = render_benchmark_report(result, videos, tmp_path / "report.html")
    completed = subprocess.run(
        [node, str(Path(__file__).with_name("benchmark_report_ui.cjs")), str(output)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "Comparison UI interactions passed" in completed.stdout
