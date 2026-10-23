import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from talkingfacekit import (
    ActiveSpeakerProvenance,
    ActiveSpeakerResult,
    ActiveSpeakerScoreWindow,
    AnalysisConfig,
    AnalysisTimings,
    CandidatePolicy,
    CandidateRun,
    FaceObservation,
    SynchronizationPolicy,
    SynchronizationResult,
    SynchronizationWindow,
    TalkingFaceSequence,
    VideoAnalysisResult,
    VideoAnalyzer,
    VideoMetadata,
    VideoSource,
    VisualPolicy,
)
from talkingfacekit.integrations import deeptalk, syncnet
from talkingfacekit.synchronization import decide_synchronization

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def evidence() -> ActiveSpeakerResult:
    return ActiveSpeakerResult(
        face_observations=tuple(
            FaceObservation(
                10 + index / 25,
                1,
                (100, 100, 200, 200),
                (85 if 75 <= index < 100 else 0, 0, 0),
                0.9,
            )
            for index in range(200)
        ),
        speech_intervals=(),
        score_windows=(ActiveSpeakerScoreWindow(10, 18, {1: 2.0}),),
        provenance=ActiveSpeakerProvenance("0.3.1"),
        candidate_runs=(
            CandidateRun(
                10, 18, "candidate", 1, ("structural_policy_passed",), 1, 200, 200, 1, 0, 2, 2, 2
            ),
        ),
    )


def sequence() -> TalkingFaceSequence:
    return TalkingFaceSequence(VideoSource(FIXTURE, VideoMetadata(640, 480, 25, 30, True)), 10, 18)


def test_sequence_analysis_has_typed_segments_and_preserves_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = evidence()
    original = sequence()
    policies: list[CandidatePolicy | None] = []

    def analyze(
        item: TalkingFaceSequence, *, candidate_policy: CandidatePolicy | None = None
    ) -> ActiveSpeakerResult:
        assert item is original
        policies.append(candidate_policy)
        return raw

    monkeypatch.setattr(deeptalk, "analyze_sequence", analyze)
    config = AnalysisConfig(candidate_policy=CandidatePolicy(minimum_duration_seconds=3))
    result = original.analyze(config)
    assert isinstance(result, VideoAnalysisResult)
    assert result.active_speaker is raw
    assert result.source is original.source
    assert policies == [config.candidate_policy]
    segment = result.selected_segments[0]
    assert (segment.start_seconds, segment.end_seconds, segment.duration_seconds) == (10, 18, 8)
    assert segment.to_sequence().source is original.source
    assert segment.to_sequence().start_seconds == 10
    assert not original.landmark_tracks
    assert result.synchronization is None
    assert result.visual_filter is None
    assert result.timings.total_seconds >= result.timings.active_speaker_seconds >= 0

    output = result.save(tmp_path / "nested" / "analysis.json")
    saved = json.loads(output.read_text())
    assert saved["interval"] == [10, 18]
    assert saved["selected_intervals"][0]["source_start_seconds"] == 10
    assert saved["score_windows"][0]["raw_scores"] == {"1": 2.0}
    assert saved["source_metadata"]["width"] == 640
    assert len(saved["source_sha256"]) == 64
    # Rendering the same result must never call a model again.
    monkeypatch.setattr(
        deeptalk, "analyze_sequence", lambda *args, **kwargs: pytest.fail("inference")
    )
    html = result.render_report(tmp_path / "analysis.html")
    assert "structural_policy_passed" in html.read_text()
    with pytest.raises(FileExistsError):
        result.save(output)
    assert result.save(output, overwrite=True) == output


def test_python_visual_selection_preserves_source_time(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = evidence()
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda *args, **kwargs: raw)
    result = sequence().analyze(AnalysisConfig(visual_policy=VisualPolicy()))
    assert [(item.start_seconds, item.end_seconds) for item in result.selected_segments] == [
        (10, 13),
        (14, 18),
    ]
    assert raw.candidate_runs[0].source_end_seconds == 18
    assert result.visual_filter is not None
    assert any("head_yaw_exceeded" in item.reason_codes for item in result.segments)


def test_sync_model_is_reused_and_visual_cannot_promote_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = evidence()
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda *args, **kwargs: raw)
    policy = SynchronizationPolicy()
    window = SynchronizationWindow(
        10, 18, 1, "offset_detected", ("audio_video_offset",), 0.32, 8, 6, 1
    )
    sync = SynchronizationResult(
        policy, (window,), (decide_synchronization(10, 18, 1, (window,), policy),), "a" * 64, 0.5
    )
    loads: list[Path] = []

    class Verifier:
        initialization_seconds = 0.1

        def __init__(self, path: Path, *, policy: SynchronizationPolicy) -> None:
            loads.append(path)

        def verify(
            self, item: TalkingFaceSequence, analysis: ActiveSpeakerResult
        ) -> SynchronizationResult:
            assert analysis is raw
            return sync

    monkeypatch.setattr(syncnet, "SyncNetVerifier", Verifier)
    runner = VideoAnalyzer(
        AnalysisConfig(syncnet_model=Path("explicit.model"), visual_policy=VisualPolicy())
    )
    first, second = runner.analyze(sequence()), runner.analyze(sequence())
    assert loads == [Path("explicit.model")]
    assert first.selected_segments == second.selected_segments == ()
    assert first.segments[0].reason_codes == ("audio_video_offset",)
    assert raw.candidate_runs[0].status == "candidate"
    with pytest.raises(ValueError, match="cover every structural candidate"):
        VideoAnalysisResult(sequence().source, 10, 18, raw, replace(sync, decisions=()))


def test_model_errors_leave_sequence_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    original = sequence()

    def fail(*args: object, **kwargs: object) -> ActiveSpeakerResult:
        raise RuntimeError("model failed")

    monkeypatch.setattr(deeptalk, "analyze_sequence", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        original.analyze()
    assert original.start_seconds == 10
    assert original.end_seconds == 18
    assert not original.landmark_tracks


def test_configuration_and_values_fail_before_loading_models() -> None:
    with pytest.raises(ValueError, match="requires syncnet_model"):
        AnalysisConfig(synchronization_policy=SynchronizationPolicy(tolerance_seconds=0.12))
    with pytest.raises(ValueError, match="finite and non-negative"):
        AnalysisTimings(float("nan"))
    with pytest.raises(ValueError, match="finite and after"):
        VideoAnalysisResult(sequence().source, 18, 10, evidence())
    with pytest.raises(ValueError, match="end in .json"):
        VideoAnalysisResult(sequence().source, 10, 18, evidence()).save("bad.txt")


def test_public_import_does_not_load_optional_models_or_backend_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, talkingfacekit; "
                "assert not {'torch', 'cv2', 'deeptalk_asd', "
                "'talkingfacekit.integrations.deeptalk', 'talkingfacekit.integrations.syncnet'} "
                ".intersection(sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_deeptalk_value_imports_are_identical_to_core_types() -> None:
    assert deeptalk.DeepTalkAnalysisResult is ActiveSpeakerResult
    assert deeptalk.DeepTalkCandidatePolicy is CandidatePolicy
