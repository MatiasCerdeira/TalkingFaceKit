from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from talkingfacekit import (
    FaceLandmarkTrack,
    FaceMeshTrack,
    TalkingFaceSequence,
    cli,
    load_landmark_track,
    save_landmark_track,
)
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkAudioTimelineRepair,
    DeepTalkFaceObservation,
    DeepTalkScoreWindow,
    DeepTalkSpeechInterval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def make_landmark_track() -> FaceLandmarkTrack:
    return FaceLandmarkTrack(
        tracker_name="fake-mediapipe",
        tracker_version="test",
        topology="test-2-points",
        coordinate_system="synthetic test coordinates",
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.042], dtype=np.float64),
        landmarks=np.zeros((2, 2, 3), dtype=np.float32),
        detected=np.asarray([True, True], dtype=np.bool_),
    )


def make_mesh_track() -> FaceMeshTrack:
    return FaceMeshTrack(
        topology="test-triangle",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.042], dtype=np.float64),
        vertices=np.zeros((2, 3, 3), dtype=np.float32),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        detected=np.asarray([True, True], dtype=np.bool_),
    )


class FakeMediaPipeFaceTracker:
    result: ClassVar[FaceLandmarkTrack]
    calls: ClassVar[list[tuple[Path, float, float | None]]]

    def __init__(self, model_path: Path) -> None:
        assert model_path == Path("model.task")

    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        self.calls.append((video_path, start_seconds, end_seconds))
        return self.result


def test_extracts_the_complete_track_to_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeMediaPipeFaceTracker.result = make_landmark_track()
    FakeMediaPipeFaceTracker.calls = []
    monkeypatch.setattr(cli, "MediaPipeFaceTracker", FakeMediaPipeFaceTracker)
    output_path = tmp_path / "video_landmarks.npz"

    exit_code = cli.main(
        [
            "extract-landmarks",
            str(FIXTURE),
            "--model",
            "model.task",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert len(FakeMediaPipeFaceTracker.calls) == 1
    video_path, start_seconds, end_seconds = FakeMediaPipeFaceTracker.calls[0]
    assert video_path == FIXTURE
    assert start_seconds == 0.0
    assert end_seconds is None
    assert output_path.is_file()
    assert load_landmark_track(output_path).landmarks.shape == (2, 2, 3)
    assert "detected frames: 2" in capsys.readouterr().out


def test_extracts_only_the_requested_video_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeMediaPipeFaceTracker.result = make_landmark_track()
    FakeMediaPipeFaceTracker.calls = []
    monkeypatch.setattr(cli, "MediaPipeFaceTracker", FakeMediaPipeFaceTracker)
    output_path = tmp_path / "clip_landmarks.npz"

    exit_code = cli.main(
        [
            "extract-landmarks",
            str(FIXTURE),
            "--model",
            "model.task",
            "--output",
            str(output_path),
            "--start-seconds",
            "0.5",
            "--end-seconds",
            "1.5",
        ]
    )

    assert exit_code == 0
    assert FakeMediaPipeFaceTracker.calls == [(FIXTURE, 0.5, 1.5)]
    assert output_path.is_file()


def test_uses_the_sequence_start_when_only_a_cli_end_is_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeMediaPipeFaceTracker.result = make_landmark_track()
    FakeMediaPipeFaceTracker.calls = []
    monkeypatch.setattr(cli, "MediaPipeFaceTracker", FakeMediaPipeFaceTracker)
    output_path = tmp_path / "clip_from_start_landmarks.npz"

    exit_code = cli.main(
        [
            "extract-landmarks",
            str(FIXTURE),
            "--model",
            "model.task",
            "--output",
            str(output_path),
            "--end-seconds",
            "1.5",
        ]
    )

    assert exit_code == 0
    assert FakeMediaPipeFaceTracker.calls == [(FIXTURE, 0.0, 1.5)]
    assert output_path.is_file()


def test_rejects_an_invalid_cli_interval_before_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeMediaPipeFaceTracker.result = make_landmark_track()
    FakeMediaPipeFaceTracker.calls = []
    monkeypatch.setattr(cli, "MediaPipeFaceTracker", FakeMediaPipeFaceTracker)
    output_path = tmp_path / "invalid_interval.npz"

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "extract-landmarks",
                str(FIXTURE),
                "--model",
                "model.task",
                "--output",
                str(output_path),
                "--start-seconds",
                "1.0",
                "--end-seconds",
                "1.0",
            ]
        )

    assert exit_info.value.code == 2
    assert "end_seconds must be greater than start_seconds" in capsys.readouterr().err
    assert FakeMediaPipeFaceTracker.calls == []
    assert not output_path.exists()


def test_inspects_an_existing_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_path = tmp_path / "video_landmarks.npz"
    save_landmark_track(make_landmark_track(), archive_path)

    exit_code = cli.main(["inspect-landmarks", str(archive_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "tracker: fake-mediapipe" in output
    assert "landmark array shape: (2, 2, 3)" in output


def test_builds_and_renders_mesh_from_landmark_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = tmp_path / "video_landmarks.npz"
    output_path = tmp_path / "face_mesh.html"
    save_landmark_track(make_landmark_track(), archive_path)
    expected_mesh = make_mesh_track()

    def fake_build_mesh(
        track: FaceLandmarkTrack,
        *,
        image_width: int,
        image_height: int,
    ) -> FaceMeshTrack:
        assert track.landmark_count == 2
        assert image_width == 1920
        assert image_height == 1080
        return expected_mesh

    def fake_render_mesh(
        mesh: FaceMeshTrack,
        destination: str | Path,
        *,
        title: str,
        depth_scale: float,
        overwrite: bool,
    ) -> Path:
        assert mesh is expected_mesh
        assert title == "Course demo"
        assert depth_scale == 1.5
        assert overwrite is False
        path = Path(destination)
        path.write_text("rendered", encoding="utf-8")
        return path

    monkeypatch.setattr(cli, "build_mediapipe_face_mesh", fake_build_mesh)
    monkeypatch.setattr(cli, "render_face_mesh_html", fake_render_mesh)

    exit_code = cli.main(
        [
            "render-mesh",
            str(archive_path),
            "--video",
            str(FIXTURE),
            "--output",
            str(output_path),
            "--title",
            "Course demo",
            "--depth-scale",
            "1.5",
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "rendered"
    output = capsys.readouterr().out
    assert "vertices per frame: 3" in output
    assert "triangles per frame: 1" in output


def test_analyzes_active_speakers_and_prints_a_readable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analyzed_intervals: list[tuple[Path, float, float | None]] = []
    rendered_reports: list[tuple[Path, DeepTalkAnalysisResult, Path, bool]] = []
    result = DeepTalkAnalysisResult(
        face_observations=(
            DeepTalkFaceObservation(
                source_timestamp_seconds=0.5,
                face_id=7,
                bounding_box_xywh=(10.0, 20.0, 30.0, 40.0),
            ),
        ),
        speech_intervals=(
            DeepTalkSpeechInterval(
                source_start_seconds=0.6,
                source_end_seconds=1.4,
                status="confirmed",
            ),
        ),
        score_windows=(
            DeepTalkScoreWindow(
                source_start_seconds=0.5,
                source_end_seconds=1.5,
                raw_scores={7: 1.25, 9: -0.5},
            ),
        ),
        provenance=DeepTalkAnalysisProvenance(backend_version="0.3.1"),
        audio_timeline_repairs=(
            DeepTalkAudioTimelineRepair(
                kind="audio_gap_filled",
                source_start_seconds=0.7,
                source_end_seconds=0.71,
                adjusted_sample_count=480,
                sample_rate_hz=48_000,
            ),
        ),
        issues=("audio_gap_filled",),
    )

    def fake_analyze_sequence(sequence: TalkingFaceSequence) -> DeepTalkAnalysisResult:
        analyzed_intervals.append(
            (sequence.source.path, sequence.start_seconds, sequence.end_seconds)
        )
        return result

    report_path = tmp_path / "active-speaker.html"

    def fake_render_active_speaker_report(
        sequence: TalkingFaceSequence,
        analysis_result: DeepTalkAnalysisResult,
        destination: str | Path,
        *,
        overwrite: bool,
    ) -> Path:
        rendered_reports.append(
            (sequence.source.path, analysis_result, Path(destination), overwrite)
        )
        report_path.write_text("rendered", encoding="utf-8")
        return report_path

    monkeypatch.setattr(cli, "analyze_sequence", fake_analyze_sequence)
    monkeypatch.setattr(cli, "render_active_speaker_report", fake_render_active_speaker_report)

    exit_code = cli.main(
        [
            "analyze-video",
            str(FIXTURE),
            "--start-seconds",
            "0.5",
            "--end-seconds",
            "1.5",
            "--report",
            str(report_path),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert analyzed_intervals == [(FIXTURE, 0.5, 1.5)]
    assert rendered_reports == [(FIXTURE, result, report_path, True)]
    output = capsys.readouterr().out
    assert "backend: DeepTalk-ASD 0.3.1" in output
    assert "face IDs: [7]" in output
    assert "[0.600, 1.400) confirmed" in output
    assert "face_7=+1.250, face_9=-0.500" in output
    assert "[0.700, 0.710) filled silence (0.010 seconds)" in output
    assert "issues: audio_gap_filled" in output
    assert "raw scores are not probabilities" in output
    assert f"report: {report_path}" in output


def test_rejects_report_overwrite_without_a_report_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["analyze-video", str(FIXTURE), "--overwrite"])

    assert exit_info.value.code == 2
    assert "--overwrite requires --report" in capsys.readouterr().err
