from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        )
        self.mutable_scores = {7: 1.75}

    def append_video(self, frame: object, *, create_time: float) -> list[object]:
        self.operation_log.append(("video", create_time))
        return [self.profile] if frame in {"face-start", "face-end"} else []

    def append_audio(self, frame: object, *, create_time: float) -> None:
        del frame
        self.operation_log.append(("audio", create_time))

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
    assert len(result.score_windows) == 10
    assert result.score_windows[0].source_start_seconds == pytest.approx(10.0)
    assert result.score_windows[-1].source_end_seconds == pytest.approx(20.0)
    assert result.score_windows[0].raw_scores == {}
    assert result.score_windows[1].raw_scores == {7: 1.75}

    detector.rectangle.x = 999.0
    detector.mutable_scores[7] = 999.0
    assert result.face_observations[0].bounding_box_xywh[0] == 1.0
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
    from deeptalk_asd.asd import ASD  # type: ignore[import-untyped]
    from deeptalk_asd.speaker_detector.interface import (  # type: ignore[import-untyped]
        FaceData,
    )
    from deeptalk_asd.speaker_detector.lrasd_onnx import (  # type: ignore[import-untyped]
        LRASDOnnxSpeakerDetector,
    )

    detector = ASD.__new__(ASD)
    speaker = LRASDOnnxSpeakerDetector.__new__(LRASDOnnxSpeakerDetector)
    speaker.video_frame_rate = 25
    speaker.audio_sample_rate = 16_000
    speaker._min_frame_interval = 1 / 25
    speaker._last_appended_video_time = {}
    speaker.last_face_timestamps = {}
    speaker.video_buffer = defaultdict(list)
    speaker.voice_profiles = {}
    speaker.max_track_age = 5.0
    speaker._extract_mouth_image = lambda _: np.zeros((112, 112), dtype=np.uint8)
    detector._speaker_detector = speaker
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
