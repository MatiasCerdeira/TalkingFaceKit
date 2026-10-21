"""Opt-in differential offset recovery with official models and a supplied clean speech clip."""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.integrations.deeptalk import analyze_sequence
from talkingfacekit.integrations.syncnet import (
    SyncNetVerifier,
    _read_audio,
    measure_feature_offset,
)


@pytest.mark.skipif(
    not (
        os.environ.get("TALKINGFACEKIT_SYNCNET_TEST_MODEL")
        and os.environ.get("TALKINGFACEKIT_SYNCNET_TEST_VIDEO")
    ),
    reason="set TALKINGFACEKIT_SYNCNET_TEST_MODEL and TALKINGFACEKIT_SYNCNET_TEST_VIDEO for real models",
)
def test_real_model_recovers_injected_pcm_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPTALK_ASD_OFFLINE", "1")
    start = float(os.environ.get("TALKINGFACEKIT_SYNCNET_TEST_START_SECONDS", "0"))
    sequence = TalkingFaceSequence.from_video(
        Path(os.environ["TALKINGFACEKIT_SYNCNET_TEST_VIDEO"])
    ).clip(start, start + 5)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        analysis = analyze_sequence(sequence)
    finally:
        logging.disable(previous)
    assert analysis.face_observations
    verifier = SyncNetVerifier(os.environ["TALKINGFACEKIT_SYNCNET_TEST_MODEL"])
    face_id = analysis.face_observations[0].face_id
    video, reason = verifier._read_video(
        sequence, analysis.face_observations, face_id, start, start + 5
    )
    assert reason is None
    audio, reason = _read_audio(sequence, start, start + 5)
    assert reason is None
    estimates = {}
    for lag in (0, -0.32, -0.16, 0.16, 0.32):
        shifted = np.zeros_like(audio)
        count = round(abs(lag) * 16_000)
        if lag > 0:
            shifted[count:] = audio[:-count]
        elif lag < 0:
            shifted[:-count] = audio[count:]
        else:
            shifted = audio
        visual_features, audio_features = verifier.extract_features(video, shifted)
        window = measure_feature_offset(
            visual_features,
            audio_features,
            start_seconds=start,
            end_seconds=start + 5,
            face_id=face_id,
            policy=verifier.policy,
        )
        assert window.status != "uncertain", window
        assert window.estimated_offset_seconds is not None
        estimates[lag] = window.estimated_offset_seconds
    for lag, estimate in estimates.items():
        assert estimate - estimates[0] == pytest.approx(lag, abs=0.04)
