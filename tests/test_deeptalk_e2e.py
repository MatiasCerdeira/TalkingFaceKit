from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.integrations.deeptalk import analyze_sequence

_VIDEO_PATH_ENV = "TALKINGFACEKIT_DEEPTALK_TEST_VIDEO"


@pytest.mark.skipif(
    not os.environ.get(_VIDEO_PATH_ENV),
    reason=f"set {_VIDEO_PATH_ENV} to run the real DeepTalk end-to-end test",
)
def test_deeptalk_real_video_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = os.environ.get(_VIDEO_PATH_ENV)
    assert video_path is not None
    monkeypatch.chdir(tmp_path)

    sequence = TalkingFaceSequence.from_video(Path(video_path))
    result = analyze_sequence(sequence)

    assert result.face_observations
    scores = [score for window in result.score_windows for score in window.raw_scores.values()]
    assert scores
    assert all(math.isfinite(score) for score in scores)
