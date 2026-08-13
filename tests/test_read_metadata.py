from pathlib import Path

import pytest

from talkingfacekit.sequence import TalkingFaceSequence

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def test_reads_video_metadata() -> None:
    sequence = TalkingFaceSequence(FIXTURE)
    metadata = sequence.metadata

    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.average_fps == pytest.approx(23.976, abs=0.001)
    if metadata.stream_duration_seconds is not None:
        assert metadata.stream_duration_seconds == pytest.approx(2.719, abs=0.1)
    assert metadata.has_audio is True
