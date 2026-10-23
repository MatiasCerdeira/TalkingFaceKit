import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from talkingfacekit import TalkingFaceSequence, cli
from talkingfacekit.integrations import deeptalk, syncnet
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkCandidateRun,
    DeepTalkScoreWindow,
)
from talkingfacekit.synchronization import (
    SynchronizationPolicy,
    SynchronizationResult,
    SynchronizationWindow,
    decide_synchronization,
)


def test_sync_rejection_reaches_json_html_and_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "example1.webm"
    raw = DeepTalkAnalysisResult(
        (),
        (),
        (DeepTalkScoreWindow(0, 10, {1: 2.0}),),
        DeepTalkAnalysisProvenance(backend_version="0.3.1"),
        candidate_runs=(
            DeepTalkCandidateRun(
                2, 7, "candidate", 1, ("structural_policy_passed",), 1, 125, 125, 1, 0.04, 1, 1, 1
            ),
        ),
    )
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda sequence, **kwargs: raw)
    policy = SynchronizationPolicy()
    window = SynchronizationWindow(
        2, 7, 1, "offset_detected", ("audio_video_offset",), 0.32, 8, 6, 1
    )
    verification = SynchronizationResult(
        policy, (window,), (decide_synchronization(2, 7, 1, (window,), policy),), "a" * 64, 0.5
    )
    loaded = []

    class FakeVerifier:
        initialization_seconds = 0.1

        def __init__(self, path: Path, *, policy: SynchronizationPolicy) -> None:
            loaded.append(path)

        def verify(
            self, sequence: TalkingFaceSequence, result: DeepTalkAnalysisResult
        ) -> SynchronizationResult:
            assert result is raw
            return verification

    monkeypatch.setattr(syncnet, "SyncNetVerifier", FakeVerifier)
    output = tmp_path / "analysis.json"
    html = tmp_path / "analysis.html"
    assert (
        cli.main(
            [
                "analyze-video",
                str(fixture),
                "--output",
                str(output),
                "--report",
                str(html),
                "--syncnet-model",
                "official.model",
            ]
        )
        == 0
    )
    saved = json.loads(output.read_text())
    assert saved["source_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert saved["candidate_runs"][0]["status"] == "candidate"
    assert saved["synchronization"]["decisions"][0]["status"] == "rejected"
    assert '"estimated_offset_seconds":0.32' in html.read_text()
    assert "DESCARTAR POR DESFASE" in html.read_text()
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
                        "end_seconds": 10,
                        "selected": [[2, 7]],
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
                "--output",
                str(benchmark),
                "--syncnet-model",
                "official.model",
            ]
        )
        == 0
    )
    case = json.loads(benchmark.read_text())["cases"][0]
    assert case["predicted"] == []
    assert case["synchronization"] == asdict(verification) | {
        "windows": [asdict(window) | {"reason_codes": ["audio_video_offset"]}],
        "decisions": [asdict(verification.decisions[0]) | {"reason_codes": ["audio_video_offset"]}],
    }
    assert (
        cli.main(
            [
                "render-benchmark",
                str(benchmark),
                "--videos-dir",
                str(fixture.parent),
                "--output",
                str(tmp_path / "comparison.html"),
            ]
        )
        == 0
    )
    before = len(loaded)
    assert (
        cli.main(
            [
                "evaluate-benchmark",
                str(manifest),
                "--videos-dir",
                str(fixture.parent),
                "--verify-only",
                "--syncnet-model",
                "missing.model",
            ]
        )
        == 0
    )
    assert len(loaded) == before


def test_custom_sync_thresholds_require_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(
        sequence: TalkingFaceSequence, *, candidate_policy: object = None
    ) -> DeepTalkAnalysisResult:
        pytest.fail("Invalid sync configuration must fail before analysis")

    monkeypatch.setattr(deeptalk, "analyze_sequence", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["analyze-video", "missing.webm", "--sync-tolerance-ms", "120"])
    assert error.value.code == 2
