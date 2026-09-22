from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from talkingfacekit import TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.integrations import deeptalk


def make_sequence(
    *,
    start_seconds: float = 10.0,
    end_seconds: float | None = 20.0,
    has_audio: bool = True,
) -> TalkingFaceSequence:
    return TalkingFaceSequence(
        source=VideoSource(
            path=Path("source.mp4"),
            metadata=VideoMetadata(640, 360, 30.0, 30.0, has_audio),
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


class FakeDetector:
    def __init__(self) -> None:
        self.evaluate_calls: list[tuple[float, float]] = []
        self.operation_log: list[tuple[str, float]] = []
        self.rectangle = SimpleNamespace(x=1.0, y=2.0, width=30.0, height=40.0)
        self.profile = SimpleNamespace(
            id=7,
            face_rectangle=self.rectangle,
            head_pose=SimpleNamespace(yaw=8.0, pitch=-3.5, roll=1.25),
            face_image_score=0.875,
        )
        self.mutable_scores = {7: 1.75}

    def append_video(self, frame: object, *, create_time: float) -> list[object]:
        self.operation_log.append(("video", create_time))
        return [self.profile] if frame in {"face-start", "face-end"} else []

    def append_audio(self, frame: object, *, create_time: float) -> object | None:
        self.operation_log.append(("audio", create_time))
        if frame == "speech-end":
            return SimpleNamespace(
                turn_state=SimpleNamespace(name="TURN_END"),
                duration_seconds=lambda: 0.5,
            )
        if frame == "speech-open":
            return SimpleNamespace(
                turn_state=SimpleNamespace(name="TURN_CONFIRMED"),
                duration_seconds=lambda: 0.2,
            )
        return None

    def evaluate(self, start_time: float, end_time: float) -> dict[int, float]:
        self.evaluate_calls.append((start_time, end_time))
        self.operation_log.append(("evaluate", end_time))
        return {} if start_time == 0.0 else self.mutable_scores


def install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    detector: FakeDetector,
    *,
    video_events: list[tuple[float, object]],
    audio_events: list[tuple[float, object]],
) -> None:
    module = ModuleType("fake_deeptalk")
    monkeypatch.setattr(deeptalk, "_load_deeptalk_module", lambda: module)
    monkeypatch.setattr(deeptalk, "_create_deeptalk_detector", lambda _: detector)
    monkeypatch.setattr(
        deeptalk,
        "_iter_deeptalk_video_frames",
        lambda *_: iter(video_events),
    )
    monkeypatch.setattr(
        deeptalk,
        "_iter_deeptalk_audio_frames",
        lambda *_: iter(audio_events),
    )


def test_analyzes_one_sequence_on_a_single_source_relative_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = FakeDetector()
    install_fake_pipeline(
        monkeypatch,
        detector,
        video_events=[
            (0.0, "face-start"),
            (0.04, "tie-video"),
            (1.0, "window-boundary-video"),
            (9.96, "face-end"),
        ],
        audio_events=[
            (0.03, "audio-start"),
            (0.04, "tie-audio"),
            *((float(second), f"audio-{second}") for second in range(1, 11)),
        ],
    )

    result = deeptalk.analyze_sequence(make_sequence())

    append_order = [operation for operation in detector.operation_log if operation[0] != "evaluate"]
    assert [timestamp for _, timestamp in append_order] == sorted(
        timestamp for _, timestamp in append_order
    )
    assert append_order.index(("video", 0.04)) < append_order.index(("audio", 0.04))
    assert detector.operation_log.index(("video", 1.0)) < detector.operation_log.index(
        ("evaluate", 1.0)
    )
    assert all(0.0 <= timestamp <= 10.0 for _, timestamp in append_order)
    assert detector.evaluate_calls == [(float(index), float(index + 1)) for index in range(10)]

    assert [observation.source_timestamp_seconds for observation in result.face_observations] == [
        pytest.approx(10.0),
        pytest.approx(19.96),
    ]
    assert [observation.face_id for observation in result.face_observations] == [7, 7]
    assert result.face_observations[0].bounding_box_xywh == (1.0, 2.0, 30.0, 40.0)
    assert result.face_observations[0].head_pose_yaw_pitch_roll_degrees == (8.0, -3.5, 1.25)
    assert result.face_observations[0].raw_face_quality_score == 0.875
    assert len(result.score_windows) == 10
    assert result.score_windows[0].source_start_seconds == pytest.approx(10.0)
    assert result.score_windows[-1].source_end_seconds == pytest.approx(20.0)
    assert result.score_windows[0].raw_scores == {}
    assert result.score_windows[1].raw_scores == {7: 1.75}
    assert result.speech_intervals == ()
    assert result.provenance.backend_version == "0.3.1"
    assert result.provenance.speaker_embeddings_available is False
    assert result.issues == ()

    detector.rectangle.x = 999.0
    detector.profile.head_pose.yaw = 999.0
    detector.profile.face_image_score = 999.0
    detector.mutable_scores[7] = 999.0
    assert result.face_observations[0].bounding_box_xywh[0] == 1.0
    assert result.face_observations[0].head_pose_yaw_pitch_roll_degrees[0] == 8.0
    assert result.face_observations[0].raw_face_quality_score == 0.875
    assert result.score_windows[1].raw_scores[7] == 1.75


def test_keeps_a_final_partial_score_window(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = FakeDetector()
    install_fake_pipeline(
        monkeypatch,
        detector,
        video_events=[(0.0, "face-start")],
        audio_events=[(0.2, "final-audio")],
    )

    result = deeptalk.analyze_sequence(make_sequence(start_seconds=10.0, end_seconds=10.2))

    assert len(detector.evaluate_calls) == 1
    assert detector.evaluate_calls[0] == pytest.approx((0.0, 0.2))
    assert len(result.score_windows) == 1
    assert result.score_windows[0].source_start_seconds == pytest.approx(10.0)
    assert result.score_windows[0].source_end_seconds == pytest.approx(10.2)
    assert result.score_windows[0].raw_scores == {}


@pytest.mark.parametrize(
    ("audio_payload", "expected_status"),
    [("speech-end", "confirmed"), ("speech-open", "incomplete")],
)
def test_copies_vad_intervals_without_retaining_backend_frames(
    monkeypatch: pytest.MonkeyPatch,
    audio_payload: str,
    expected_status: str,
) -> None:
    detector = FakeDetector()
    install_fake_pipeline(
        monkeypatch,
        detector,
        video_events=[(0.0, "face-start")],
        audio_events=[(0.5, audio_payload)],
    )

    result = deeptalk.analyze_sequence(make_sequence(start_seconds=10.0, end_seconds=10.5))

    assert len(result.speech_intervals) == 1
    interval = result.speech_intervals[0]
    assert interval.source_end_seconds == pytest.approx(10.5)
    assert interval.status == expected_status
    expected_start = 10.0 if expected_status == "confirmed" else 10.3
    assert interval.source_start_seconds == pytest.approx(expected_start)


def test_reports_and_trims_video_after_open_ended_audio_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = FakeDetector()
    install_fake_pipeline(
        monkeypatch,
        detector,
        video_events=[(1.0, "face-start")],
        audio_events=[(0.5, "audio-end")],
    )

    result = deeptalk.analyze_sequence(make_sequence(start_seconds=10.0, end_seconds=None))

    assert result.issues == ("audio_ended_before_video",)
    assert result.face_observations == ()


def test_propagates_audio_timeline_repairs_as_structured_results_and_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = FakeDetector()
    install_fake_pipeline(
        monkeypatch,
        detector,
        video_events=[(0.0, "face-start")],
        audio_events=[(0.5, "audio-end")],
    )
    expected_repair = deeptalk.DeepTalkAudioTimelineRepair(
        kind="audio_gap_filled",
        source_start_seconds=10.1,
        source_end_seconds=10.2,
        adjusted_sample_count=4_800,
        sample_rate_hz=48_000,
    )

    def fake_audio_events(
        sequence: TalkingFaceSequence,
        module: ModuleType,
        repairs: list[deeptalk.DeepTalkAudioTimelineRepair],
    ) -> Iterator[tuple[float, object]]:
        del sequence, module
        repairs.append(expected_repair)
        return iter([(0.5, "audio-end")])

    monkeypatch.setattr(deeptalk, "_iter_deeptalk_audio_frames", fake_audio_events)

    result = deeptalk.analyze_sequence(make_sequence(start_seconds=10.0, end_seconds=10.5))

    assert result.audio_timeline_repairs == (expected_repair,)
    assert result.issues == ("audio_gap_filled",)


def test_prunes_completed_visual_windows_but_keeps_the_shared_boundary() -> None:
    speaker = SimpleNamespace(
        video_buffer={
            3: [("old", 0.96), ("boundary", 1.0), ("future", 1.04)],
        }
    )
    detector = SimpleNamespace(_speaker_detector=speaker)

    deeptalk._prune_deeptalk_video_buffer(detector, before_seconds=1.0)

    assert speaker.video_buffer == {3: [("boundary", 1.0), ("future", 1.04)]}


def test_rejects_non_finite_copied_backend_values() -> None:
    with pytest.raises(ValueError, match="raw score for face 3 must be finite"):
        deeptalk.DeepTalkScoreWindow(
            source_start_seconds=0.0,
            source_end_seconds=1.0,
            raw_scores={3: float("nan")},
        )

    with pytest.raises(ValueError, match="width and height must be positive"):
        deeptalk.DeepTalkFaceObservation(
            source_timestamp_seconds=0.0,
            face_id=3,
            bounding_box_xywh=(0.0, 0.0, 0.0, 10.0),
            head_pose_yaw_pitch_roll_degrees=(0.0, 0.0, 0.0),
            raw_face_quality_score=0.5,
        )

    with pytest.raises(ValueError, match="three finite degree values"):
        deeptalk.DeepTalkFaceObservation(
            source_timestamp_seconds=0.0,
            face_id=3,
            bounding_box_xywh=(0.0, 0.0, 10.0, 10.0),
            head_pose_yaw_pitch_roll_degrees=(float("nan"), 0.0, 0.0),
            raw_face_quality_score=0.5,
        )

    with pytest.raises(ValueError, match="raw_face_quality_score must be finite"):
        deeptalk.DeepTalkFaceObservation(
            source_timestamp_seconds=0.0,
            face_id=3,
            bounding_box_xywh=(0.0, 0.0, 10.0, 10.0),
            head_pose_yaw_pitch_roll_degrees=(0.0, 0.0, 0.0),
            raw_face_quality_score=float("inf"),
        )


def test_rejects_invalid_explicit_inspireface_resource(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_resource = tmp_path / "Pikachu"
    invalid_resource.write_bytes(b"not an InspireFace resource")
    monkeypatch.setenv("INSPIREFACE_RESOURCE_PATH", str(invalid_resource))

    with pytest.raises(RuntimeError, match="does not match the supported DeepTalk Pikachu"):
        deeptalk._validate_explicit_inspireface_resource()


def test_detector_creation_does_not_acquire_the_disabled_voiceprint_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_models: list[str] = []
    factory_arguments: dict[str, object] = {}
    expected_detector = object()

    def ensure_model(model_name: str, cache_dir: Path) -> Path:
        assert cache_dir == tmp_path
        requested_models.append(model_name)
        return cache_dir / model_name

    class FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            factory_arguments.update(kwargs)

        def create(self) -> object:
            return expected_detector

    module = ModuleType("fake_deeptalk")
    module.get_model_cache_dir = lambda: tmp_path  # type: ignore[attr-defined]
    module.ensure_model = ensure_model  # type: ignore[attr-defined]
    module.ASDDetectorFactory = FakeFactory  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "talkingfacekit.integrations.deeptalk.metadata.version",
        lambda _: "0.3.1",
    )
    monkeypatch.setattr(deeptalk, "_configure_deeptalk_0_3_1_for_offline", lambda _: None)

    detector = deeptalk._create_deeptalk_detector(module)

    assert detector is expected_detector
    assert requested_models == [
        "Pikachu",
        "silero_vad.onnx",
        "audio_frontend.onnx",
        "visual_frontend.onnx",
        "av_backend.onnx",
    ]
    assert factory_arguments["speaker_detector"] == {
        "type": "LR-ASD-ONNX",
        "model_dir": str(tmp_path),
        "voiceprint_model_name": "disabled-by-talkingfacekit",
    }


def test_disables_onnx_telemetry_before_import_unless_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_module = ModuleType("deeptalk_asd")
    observed_values: list[str | None] = []

    def fake_import_module(name: str) -> ModuleType:
        assert name == "deeptalk_asd"
        observed_values.append(os.environ.get("ORT_DISABLE_TELEMETRY"))
        return imported_module

    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(deeptalk, "import_module", fake_import_module)

    assert deeptalk._load_deeptalk_module() is imported_module
    assert observed_values == ["1"]

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    assert deeptalk._load_deeptalk_module() is imported_module
    assert observed_values == ["1", "0"]


def test_rejects_a_sequence_without_audio_before_constructing_deeptalk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> ModuleType:
        raise AssertionError("DeepTalk must not be loaded for a source without audio")

    monkeypatch.setattr(deeptalk, "_load_deeptalk_module", fail_if_loaded)

    with pytest.raises(ValueError, match="requires an audio stream"):
        deeptalk.analyze_sequence(make_sequence(has_audio=False))


def test_offline_shim_keeps_every_exact_25hz_slot_without_wall_clock_expiry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("deeptalk_asd")
    ASD = cast(Any, import_module("deeptalk_asd.asd").ASD)
    FaceData = cast(Any, import_module("deeptalk_asd.speaker_detector.interface").FaceData)
    LRASDOnnxSpeakerDetector = cast(
        Any,
        import_module("deeptalk_asd.speaker_detector.lrasd_onnx").LRASDOnnxSpeakerDetector,
    )
    InspireFaceDetector = cast(
        Any,
        import_module("deeptalk_asd.face_detector.inspireface_detector").InspireFaceDetector,
    )
    SileroVadTurnDetector = cast(
        Any,
        import_module("deeptalk_asd.turn_detector.silero_vad_turn_detector").SileroVadTurnDetector,
    )
    SileroVAD = cast(
        Any,
        import_module("deeptalk_asd.turn_detector.vad.silero_vad").SileroVAD,
    )

    detector = ASD.__new__(ASD)
    speaker = LRASDOnnxSpeakerDetector.__new__(LRASDOnnxSpeakerDetector)
    face_detector = InspireFaceDetector.__new__(InspireFaceDetector)
    turn_detector = SileroVadTurnDetector.__new__(SileroVadTurnDetector)
    vad = SileroVAD.__new__(SileroVAD)
    speaker.video_frame_rate = 25
    speaker.audio_sample_rate = 16_000
    speaker._min_frame_interval = 1 / 25
    speaker._last_appended_video_time = {}
    speaker.last_face_timestamps = {}
    speaker.video_buffer = defaultdict(list)
    speaker.voice_profiles = {}
    speaker.voice_extractor = None
    speaker.max_track_age = 5.0
    speaker._extract_mouth_image = lambda _: np.zeros((112, 112), dtype=np.uint8)
    face_detector._remove_stale_faces = lambda: None
    vad._last_reset_time = 0.0
    turn_detector._vad = vad
    detector._speaker_detector = speaker
    detector._face_detector = face_detector
    detector._turn_detector = turn_detector
    face = FaceData(
        id=3,
        face_image=np.zeros((4, 4, 3), dtype=np.uint8),
        face_rect_x=0.0,
        face_rect_y=0.0,
        face_rect_width=4.0,
        face_rect_height=4.0,
    )

    deeptalk._configure_deeptalk_0_3_1_for_offline(detector)
    for timestamp in [0.0, 0.04, 0.08, 0.12, 0.16]:
        speaker.append_video([face], create_time=timestamp)

    assert speaker._min_frame_interval == 0.0
    assert speaker.max_track_age == float("inf")
    assert speaker.voice_extractor is None
    assert vad._last_reset_time == float("inf")
    assert [timestamp for _, timestamp in speaker.video_buffer[3]] == [
        0.0,
        0.04,
        0.08,
        0.12,
        0.16,
    ]


def test_detector_creation_rejects_an_incompatible_deeptalk_version_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "talkingfacekit.integrations.deeptalk.metadata.version",
        lambda _: "0.3.2",
    )

    with pytest.raises(RuntimeError, match="supports only deeptalk-asd==0.3.1"):
        deeptalk._create_deeptalk_detector(ModuleType("unsupported_deeptalk"))
