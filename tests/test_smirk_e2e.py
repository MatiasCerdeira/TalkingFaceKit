"""Opt-in real SMIRK inference with explicit local models and source media."""

import os
from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.smirk import SmirkEncoder, SmirkMissingFrame
from talkingfacekit.io.video import stream_video_frames

pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "TALKINGFACEKIT_SMIRK_CHECKPOINT",
            "TALKINGFACEKIT_SMIRK_LANDMARKER",
            "TALKINGFACEKIT_SMIRK_VIDEO",
        )
    ),
    reason="set TALKINGFACEKIT_SMIRK_CHECKPOINT, SMIRK_LANDMARKER and SMIRK_VIDEO for real inference",
)


def test_real_smirk_preserves_source_time_and_returns_finite_parameters() -> None:
    start = float(os.environ.get("TALKINGFACEKIT_SMIRK_START_SECONDS", "20"))
    video = Path(os.environ["TALKINGFACEKIT_SMIRK_VIDEO"])
    sequence = TalkingFaceSequence.from_video(video).clip(start, start + 0.1)
    encoder = SmirkEncoder(os.environ["TALKINGFACEKIT_SMIRK_CHECKPOINT"])
    predictions = list(
        encoder.track(sequence, landmarker_model=os.environ["TALKINGFACEKIT_SMIRK_LANDMARKER"])
    )
    assert predictions
    assert [p.timestamp_seconds for p in predictions] == sorted(
        p.timestamp_seconds for p in predictions
    )
    assert all(start <= p.timestamp_seconds < start + 0.1 for p in predictions)
    expected = [
        (f.frame_index, f.timestamp_seconds)
        for f in stream_video_frames(video, start_seconds=start, end_seconds=start + 0.1)
    ]
    assert [(p.frame_index, p.timestamp_seconds) for p in predictions] == expected
    estimated_count = 0
    for result in predictions:
        if isinstance(result, SmirkMissingFrame):
            assert result.reason == "face_landmarks_unavailable"
            continue
        estimated_count += 1
        assert result.parameters.shape.shape == (300,)
        assert result.parameters.expression.shape == (50,)
        assert result.checkpoint_sha256 == encoder.checkpoint_sha256
        assert np.all(np.isfinite(result.parameters.expression))
        assert np.all(np.isfinite(result.camera))
    assert estimated_count > 0


def test_real_smirk_result_round_trips(tmp_path: Path) -> None:
    start = float(os.environ.get("TALKINGFACEKIT_SMIRK_START_SECONDS", "20"))
    duration = float(os.environ.get("TALKINGFACEKIT_SMIRK_DURATION_SECONDS", "0.1"))
    video = Path(os.environ["TALKINGFACEKIT_SMIRK_VIDEO"])
    sequence = TalkingFaceSequence.from_video(video).clip(start, start + duration)
    encoder = SmirkEncoder(os.environ["TALKINGFACEKIT_SMIRK_CHECKPOINT"])
    result = encoder.extract(
        sequence, landmarker_model=os.environ["TALKINGFACEKIT_SMIRK_LANDMARKER"]
    )
    expected_timeline = [
        (frame.frame_index, frame.timestamp_seconds)
        for frame in stream_video_frames(video, start_seconds=start, end_seconds=start + duration)
    ]
    assert (
        list(zip(result.frame_indices, result.timestamps_seconds, strict=True)) == expected_timeline
    )
    assert result.estimated_count > 0
    assert result.provenance["checkpoint_sha256"] == encoder.checkpoint_sha256
    path = result.save(tmp_path / "flame.npz")
    loaded = FlameResult.load(path)
    assert loaded.source == result.source
    assert loaded.provenance == result.provenance
    assert loaded.source_sha256 == result.source_sha256
    assert loaded.missing_reasons == result.missing_reasons
    assert loaded.extra_conventions == result.extra_conventions
    for name in (
        "frame_indices",
        "timestamps_seconds",
        "estimated",
        "shape",
        "expression",
        "global_pose",
        "jaw_pose",
    ):
        assert getattr(loaded, name).tobytes() == getattr(result, name).tobytes()
    for name, values in result.extras.items():
        assert loaded.extras[name].tobytes() == values.tobytes()
