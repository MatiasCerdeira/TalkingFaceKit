from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from talkingfacekit import (
    DecodedAudioChunk,
    DecodedVideoFrame,
    TalkingFaceSequence,
    VideoMetadata,
    VideoSource,
)
from talkingfacekit.integrations import syncnet
from talkingfacekit.integrations.deeptalk import (
    DeepTalkAnalysisProvenance,
    DeepTalkAnalysisResult,
    DeepTalkCandidateRun,
)
from talkingfacekit.synchronization import SynchronizationPolicy, SynchronizationWindow


def sequence() -> TalkingFaceSequence:
    return TalkingFaceSequence(
        VideoSource(Path("source.mp4"), VideoMetadata(640, 480, 25, 10, True)), 2, 7
    )


def test_invalid_checkpoint_is_rejected_before_loading_a_model(tmp_path: Path) -> None:
    path = tmp_path / "untrusted.model"
    path.write_bytes(b"not the official weights")
    with pytest.raises(ValueError, match="SHA-256"):
        syncnet.SyncNetVerifier(path)


def test_verify_only_structural_candidates_and_preserve_uncertainty() -> None:
    run = DeepTalkCandidateRun(
        2, 7, "candidate", 1, ("structural_policy_passed",), 1, 125, 125, 1, 0.04, 1, 1, 1
    )
    result = DeepTalkAnalysisResult(
        (),
        (),
        (),
        DeepTalkAnalysisProvenance(backend_version="0.3.1"),
        candidate_runs=(
            run,
            replace(run, status="rejected", reason_codes=("candidate_too_short",)),
        ),
    )
    verifier = syncnet.SyncNetVerifier.__new__(syncnet.SyncNetVerifier)
    verifier.policy = SynchronizationPolicy()
    verifier.model_sha256 = syncnet.SYNCNET_MODEL_SHA256
    calls = []

    def measure(*args: object) -> SynchronizationWindow:
        calls.append(args)
        return SynchronizationWindow(2, 7, 1, "uncertain", ("sync_low_confidence",))

    cast(Any, verifier)._measure_window = measure
    verified = verifier.verify(sequence(), result)
    assert len(calls) == 1
    assert verified.decisions[0].status == "uncertain"
    assert result.candidate_runs[0].status == "candidate"


def test_read_audio_uses_source_timestamps_and_rejects_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def chunk(start: float, length: int) -> DecodedAudioChunk:
        return DecodedAudioChunk(
            0, start, 16_000, "mono", np.full((length, 1), 0.25, dtype=np.float32)
        )

    monkeypatch.setattr(
        syncnet,
        "stream_audio_chunks",
        lambda *args, **kwargs: iter([chunk(2, 8000), chunk(2.5, 8000)]),
    )
    samples, reason = syncnet._read_audio(sequence(), 2, 3)
    assert reason is None
    assert samples.shape == (16_000,)
    np.testing.assert_allclose(samples, 8192, atol=1)
    monkeypatch.setattr(
        syncnet,
        "stream_audio_chunks",
        lambda *args, **kwargs: iter([chunk(2, 8000), chunk(2.6, 6400)]),
    )
    _, reason = syncnet._read_audio(sequence(), 2, 3)
    assert reason == "sync_audio_timeline_discontinuity"


def test_short_candidate_never_invokes_neural_inference() -> None:
    verifier = syncnet.SyncNetVerifier.__new__(syncnet.SyncNetVerifier)
    verifier.policy = SynchronizationPolicy()
    window = verifier.measure_window(sequence(), (), 1, 2, 3)
    assert window.reason_codes == ("sync_window_too_short",)


def test_audio_clips_after_reading_ordered_source_pts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Vorbis/WebM can have 128 samples at 48 kHz with starts quantized to milliseconds.
    # Clipping the first chunk to 2.0 would give the same start as the next raw chunk.
    first = DecodedAudioChunk(0, 1.998, 48_000, "mono", np.full((128, 1), 0.25, dtype=np.float32))
    second = DecodedAudioChunk(
        128, 2.0, 48_000, "mono", np.full((48_000, 1), 0.25, dtype=np.float32)
    )

    def stream(*args: object, **kwargs: object) -> object:
        assert "start_seconds" not in kwargs
        return iter((first, second))

    monkeypatch.setattr(syncnet, "stream_audio_chunks", stream)
    samples, reason = syncnet._read_audio(sequence(), 2, 3)
    assert reason is None
    assert len(samples) == 16_000
    np.testing.assert_allclose(samples, 8192, atol=1)


def test_source_readers_advance_once_across_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_calls = []
    video_calls = []

    def audio_stream(*args: object, **kwargs: object) -> object:
        audio_calls.append(args)
        return iter(
            [
                DecodedAudioChunk(
                    0, 1.9, 16_000, "mono", np.full((20_800, 1), 0.25, dtype=np.float32)
                ),
                DecodedAudioChunk(
                    20_800, 3.2, 16_000, "mono", np.full((12_800, 1), 0.5, dtype=np.float32)
                ),
            ]
        )

    def video_stream(*args: object, **kwargs: object) -> object:
        video_calls.append(args)
        return iter(
            [DecodedVideoFrame(i, i / 25, np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(101)]
        )

    monkeypatch.setattr(syncnet, "stream_audio_chunks", audio_stream)
    monkeypatch.setattr(syncnet, "stream_video_frames", video_stream)
    reader = syncnet._SourceWindowReader(sequence())
    first, reason = syncnet._read_audio(sequence(), 2, 3, reader=reader)
    assert reason is None
    second, reason = syncnet._read_audio(sequence(), 3, 4, reader=reader)
    assert reason is None
    np.testing.assert_allclose(first, 8192, atol=1)
    np.testing.assert_allclose(second[:3200], 8192, atol=1)
    np.testing.assert_allclose(second[3200:], 16384, atol=1)
    a, b = reader.video_at(2), reader.video_at(3.5)
    assert a is not None and a.timestamp_seconds == 2
    assert b is not None and b.timestamp_seconds == pytest.approx(3.48)
    assert len(audio_calls) == len(video_calls) == 1
