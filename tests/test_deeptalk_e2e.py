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
    configured_video_path = os.environ.get(_VIDEO_PATH_ENV)
    assert configured_video_path is not None
    video_path = Path(configured_video_path).expanduser().resolve()
    monkeypatch.chdir(tmp_path)

    sequence = TalkingFaceSequence.from_video(video_path)
    result = analyze_sequence(sequence)

    assert result.face_observations
    assert [
        observation.source_timestamp_seconds for observation in result.face_observations
    ] == sorted(observation.source_timestamp_seconds for observation in result.face_observations)
    assert result.score_windows
    assert [window.source_start_seconds for window in result.score_windows] == sorted(
        window.source_start_seconds for window in result.score_windows
    )
    scores = [score for window in result.score_windows for score in window.raw_scores.values()]
    assert scores
    assert all(math.isfinite(score) for score in scores)
    assert result.provenance.backend_version == "0.3.1"
    assert result.provenance.speaker_embeddings_available is False
