import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from talkingfacekit import TalkingFaceSequence, cli
from talkingfacekit.integrations import deeptalk, syncnet
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkCandidateRun,
    DeepTalkFaceObservation,
    DeepTalkPreliminarySegment,
    DeepTalkScoreWindow,
)
from talkingfacekit.synchronization import (
    SynchronizationPolicy,
    SynchronizationResult,
    SynchronizationWindow,
    decide_synchronization,
)


def raw_result() -> DeepTalkAnalysisResult:
    return DeepTalkAnalysisResult(
        tuple(
            DeepTalkFaceObservation(
                i / 25, 1, (100, 100, 200, 200), (85 if 75 <= i < 100 else 0, 0, 0), 0.9
            )
            for i in range(200)
        ),
        (),
        (DeepTalkScoreWindow(0, 8, {1: 2.0}),),
        DeepTalkAnalysisProvenance(backend_version="0.3.1"),
        preliminary_segments=(
            DeepTalkPreliminarySegment(
                0,
                8,
                "candidate",
                1,
                ("single_visible_face", "positive_raw_active_speaker_score"),
                (1,),
                2.0,
            ),
        ),
        candidate_runs=(
            DeepTalkCandidateRun(
                0, 8, "candidate", 1, ("structural_policy_passed",), 1, 200, 200, 1, 0, 2, 2, 2
            ),
        ),
    )


def test_visual_filter_reaches_json_html_benchmark_and_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "example1.webm"
    raw = raw_result()
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda sequence, **kwargs: raw)
    output, html = tmp_path / "analysis.json", tmp_path / "analysis.html"
    assert (
        cli.main(
            [
                "analyze-video",
                str(fixture),
                "--visual-filter",
                "--output",
                str(output),
                "--report",
                str(html),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text())
    assert document["candidate_runs"][0]["status"] == "candidate"
    assert [
        (i["source_start_seconds"], i["source_end_seconds"]) for i in document["selected_intervals"]
    ] == [(0, 3), (4, 8)]
    assert document["visual_filter"]["decisions"][1]["reason_codes"] == ["head_yaw_exceeded"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "test",
                        "file": fixture.name,
                        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        "start_seconds": 0,
                        "end_seconds": 8,
                        "selected": [[0, 8]],
                    }
                ],
            }
        )
    )
    benchmark = tmp_path / "benchmark.json"
    assert (
        cli.main(
            [
                "evaluate-benchmark",
                str(manifest),
                "--videos-dir",
                str(fixture.parent),
                "--visual-filter",
                "--output",
                str(benchmark),
            ]
        )
        == 0
    )
    saved = json.loads(benchmark.read_text())
    assert saved["cases"][0]["predicted"] == [[0, 3], [4, 8]]
    assert saved["cases"][0]["predicted_before_visual_filter"] == [[0, 8]]
    assert saved["overall_before_visual_filter"]["recall"] == 1
    assert saved["overall"]["recall"] == 0.875
    comparison = tmp_path / "comparison.html"
    assert (
        cli.main(
            [
                "render-benchmark",
                str(benchmark),
                "--videos-dir",
                str(fixture.parent),
                "--output",
                str(comparison),
            ]
        )
        == 0
    )
    node = shutil.which("node")
    if node is not None:
        for path, mode in ((html, "active"), (comparison, "benchmark")):
            completed = subprocess.run(
                [node, str(Path(__file__).with_name("visual_filter_ui.cjs")), str(path), mode],
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr


def test_visual_stage_cannot_resurrect_sync_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "example1.webm"
    raw = raw_result()
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda sequence, **kwargs: raw)
    policy = SynchronizationPolicy()
    window = SynchronizationWindow(
        0, 8, 1, "offset_detected", ("audio_video_offset",), 0.32, 8, 6, 1
    )
    sync = SynchronizationResult(
        policy, (window,), (decide_synchronization(0, 8, 1, (window,), policy),), "a" * 64, 0.5
    )

    class FakeVerifier:
        initialization_seconds = 0.1

        def __init__(self, path: Path, *, policy: SynchronizationPolicy) -> None:
            pass

        def verify(
            self, sequence: TalkingFaceSequence, result: DeepTalkAnalysisResult
        ) -> SynchronizationResult:
            return sync

    monkeypatch.setattr(syncnet, "SyncNetVerifier", FakeVerifier)
    output = tmp_path / "analysis.json"
    assert (
        cli.main(
            [
                "analyze-video",
                str(fixture),
                "--visual-filter",
                "--syncnet-model",
                "test.model",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    saved = json.loads(output.read_text())
    assert saved["selected_intervals"] == []
    assert saved["visual_filter"]["windows"] == []
    assert saved["visual_filter"]["decisions"][0]["reason_codes"] == ["audio_video_offset"]
    assert raw.candidate_runs[0].status == "candidate"


@pytest.mark.parametrize(
    "args",
    [
        ["--visual-maximum-yaw", "15"],
        ["--visual-filter", "--visual-maximum-yaw", "nan"],
        ["--visual-filter", "--visual-minimum-face-width-fraction", "1.1"],
    ],
)
def test_visual_configuration_fails_before_models(
    args: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(
        sequence: TalkingFaceSequence, *, candidate_policy: object = None
    ) -> DeepTalkAnalysisResult:
        pytest.fail("Invalid configuration must fail before inference")

    monkeypatch.setattr(deeptalk, "analyze_sequence", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["analyze-video", "missing.webm", *args])
    assert error.value.code == 2
