import wave
from pathlib import Path

import pytest

from talkingfacekit import TalkingFaceSequence

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def test_reads_video_metadata_without_writing_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sequence = TalkingFaceSequence.from_video(FIXTURE)
    metadata = sequence.metadata

    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.average_fps == pytest.approx(23.976, abs=0.001)
    assert metadata.stream_duration_seconds == pytest.approx(2.719, abs=0.001)
    assert metadata.has_audio is True
    assert sequence.start_seconds == 0.0
    assert sequence.end_seconds == metadata.stream_duration_seconds
    assert capsys.readouterr().out == ""


def test_rejects_missing_video_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.webm"

    with pytest.raises(FileNotFoundError, match="Video not found"):
        TalkingFaceSequence.from_video(missing_path)


def test_rejects_directory_as_video_file(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="Video path is a directory"):
        TalkingFaceSequence.from_video(tmp_path)


def test_rejects_media_without_video_stream(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio-only.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x00\x00")

    with pytest.raises(ValueError, match="Media file has no video stream"):
        TalkingFaceSequence.from_video(audio_path)


def test_rejects_malformed_media(tmp_path: Path) -> None:
    malformed_path = tmp_path / "malformed.webm"
    malformed_path.write_bytes(b"not a media file")

    with pytest.raises(ValueError, match="Could not inspect media file"):
        TalkingFaceSequence.from_video(malformed_path)
