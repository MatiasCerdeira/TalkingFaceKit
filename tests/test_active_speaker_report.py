import json
from pathlib import Path

import pytest

from talkingfacekit import TalkingFaceSequence, VideoMetadata, VideoSource
from talkingfacekit.demo import render_active_speaker_report
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkAudioTimelineRepair,
    DeepTalkFaceObservation,
    DeepTalkScoreWindow,
    DeepTalkSpeechInterval,
)


def make_sequence(source_path: Path) -> TalkingFaceSequence:
    return TalkingFaceSequence(
        source=VideoSource(
            path=source_path,
            metadata=VideoMetadata(640, 360, 25.0, 2.0, True),
        ),
        start_seconds=0.5,
        end_seconds=1.5,
    )


def make_result() -> DeepTalkAnalysisResult:
    return DeepTalkAnalysisResult(
        face_observations=(
            DeepTalkFaceObservation(
                source_timestamp_seconds=0.52,
                face_id=2,
                bounding_box_xywh=(320.25, 40.0, 120.0, 140.0),
            ),
            DeepTalkFaceObservation(
                source_timestamp_seconds=0.52,
                face_id=1,
                bounding_box_xywh=(20.0, 45.0, 110.0, 135.0),
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
                raw_scores={2: 2.125, 1: -0.25},
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


def extract_report_payload(report_html: str) -> dict[str, object]:
    prefix = "    const REPORT = "
    start = report_html.index(prefix) + len(prefix)
    end = report_html.index(";\n    const COLORS", start)
    payload = json.loads(report_html[start:end])
    assert isinstance(payload, dict)
    return payload


def test_renders_a_self_contained_synchronized_report_payload(tmp_path: Path) -> None:
    source_path = tmp_path / "conversation.mp4"
    source_path.write_bytes(b"synthetic-video-placeholder")
    output_path = tmp_path / "active-speaker.html"

    saved_path = render_active_speaker_report(
        make_sequence(source_path),
        make_result(),
        output_path,
    )

    assert saved_path == output_path
    report_html = output_path.read_text(encoding="utf-8")
    assert "TalkingFaceKit · Active speaker report" in report_html
    assert "Top candidate" in report_html
    assert "video.currentTime" in report_html
    assert "fetch(" not in report_html
    payload = extract_report_payload(report_html)
    assert payload["video"] == {
        "name": "conversation.mp4",
        "url": source_path.resolve().as_uri(),
        "width": 640,
        "height": 360,
    }
    assert payload["interval"] == {"start": 0.5, "end": 1.5}
    assert payload["faceIds"] == [1, 2]
    assert payload["faceFrames"] == [
        [0.52, [[1, 20.0, 45.0, 110.0, 135.0], [2, 320.25, 40.0, 120.0, 140.0]]]
    ]
    assert payload["scoreWindows"] == [[0.5, 1.5, [[1, -0.25], [2, 2.125]]]]
    assert payload["speechIntervals"] == [[0.6, 1.4, "confirmed"]]
    assert payload["repairs"] == [["audio_gap_filled", 0.7, 0.71, 480, 48_000]]
    assert payload["issues"] == ["audio_gap_filled"]


def test_escapes_source_names_in_html_and_embedded_json(tmp_path: Path) -> None:
    source_path = tmp_path / "<demo>&.mp4"
    source_path.write_bytes(b"synthetic-video-placeholder")
    output_path = tmp_path / "report.html"

    render_active_speaker_report(make_sequence(source_path), make_result(), output_path)

    report_html = output_path.read_text(encoding="utf-8")
    assert "&lt;demo&gt;&amp;.mp4" in report_html
    assert "\\u003cdemo\\u003e\\u0026.mp4" in report_html
    assert extract_report_payload(report_html)["video"] == {
        "name": "<demo>&.mp4",
        "url": source_path.resolve().as_uri(),
        "width": 640,
        "height": 360,
    }


def test_requires_explicit_overwrite_and_replaces_only_after_rendering(tmp_path: Path) -> None:
    source_path = tmp_path / "conversation.mp4"
    source_path.write_bytes(b"synthetic-video-placeholder")
    output_path = tmp_path / "report.html"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        render_active_speaker_report(make_sequence(source_path), make_result(), output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"
    render_active_speaker_report(
        make_sequence(source_path),
        make_result(),
        output_path,
        overwrite=True,
    )
    assert "const REPORT" in output_path.read_text(encoding="utf-8")


def test_validates_source_and_destination_paths(tmp_path: Path) -> None:
    missing_source = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="source video not found"):
        render_active_speaker_report(
            make_sequence(missing_source),
            make_result(),
            tmp_path / "report.html",
        )

    source_path = tmp_path / "conversation.mp4"
    source_path.write_bytes(b"synthetic-video-placeholder")
    with pytest.raises(ValueError, match="must end in .html"):
        render_active_speaker_report(
            make_sequence(source_path),
            make_result(),
            tmp_path / "report.txt",
        )
