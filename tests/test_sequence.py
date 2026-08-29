from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import FaceLandmarkTrack, TalkingFaceSequence, VideoMetadata


def make_track(value: float = 0.0) -> FaceLandmarkTrack:
    return FaceLandmarkTrack(
        tracker_name="fake",
        tracker_version=None,
        topology="one-point",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([0], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0], dtype=np.float64),
        landmarks=np.full((1, 1, 3), value, dtype=np.float32),
        detected=np.asarray([True], dtype=np.bool_),
    )


class FakeTracker:
    def __init__(self, result: FaceLandmarkTrack) -> None:
        self.result = result
        self.calls: list[tuple[Path, float, float | None]] = []

    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        self.calls.append((video_path, start_seconds, end_seconds))
        return self.result


class FailingTracker:
    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        del video_path, start_seconds, end_seconds
        raise RuntimeError("tracking failed")


def test_constructs_sequence_without_accessing_its_media_path() -> None:
    metadata = VideoMetadata(
        width=640,
        height=480,
        average_fps=24.0,
        stream_duration_seconds=2.0,
        has_audio=True,
    )
    sequence = TalkingFaceSequence(
        path=Path("video-that-does-not-need-to-exist.webm"),
        metadata=metadata,
        start_seconds=0.5,
        end_seconds=1.5,
    )

    assert sequence.path == Path("video-that-does-not-need-to-exist.webm")
    assert sequence.metadata == metadata
    assert sequence.start_seconds == 0.5
    assert sequence.end_seconds == 1.5


@pytest.mark.parametrize("start_seconds", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_rejects_invalid_sequence_start(start_seconds: float) -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)

    with pytest.raises(ValueError, match="start_seconds must be finite and non-negative"):
        TalkingFaceSequence(Path("portrait.webm"), metadata, start_seconds=start_seconds)


@pytest.mark.parametrize("end_seconds", [float("inf"), float("-inf"), float("nan")])
def test_rejects_non_finite_sequence_end(end_seconds: float) -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)

    with pytest.raises(ValueError, match="end_seconds must be finite when provided"):
        TalkingFaceSequence(Path("portrait.webm"), metadata, end_seconds=end_seconds)


@pytest.mark.parametrize("end_seconds", [0.5, 0.4])
def test_rejects_sequence_end_not_after_start(end_seconds: float) -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)

    with pytest.raises(ValueError, match="end_seconds must be greater than start_seconds"):
        TalkingFaceSequence(
            Path("portrait.webm"),
            metadata,
            start_seconds=0.5,
            end_seconds=end_seconds,
        )


def test_clips_sequence_without_decoding_or_mutating_the_parent() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata, 0.5, 1.5)
    sequence.track_landmarks(FakeTracker(make_track()), name="parent")

    clip = sequence.clip(0.75, 1.25)

    assert clip is not sequence
    assert clip.path == sequence.path
    assert clip.metadata is metadata
    assert clip.start_seconds == 0.75
    assert clip.end_seconds == 1.25
    assert dict(clip.landmark_tracks) == {}
    assert sequence.start_seconds == 0.5
    assert sequence.end_seconds == 1.5
    assert "parent" in sequence.landmark_tracks


def test_clip_uses_the_parent_end_when_end_is_omitted() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata, 0.5, 1.5)

    clip = sequence.clip(1.0)

    assert clip.start_seconds == 1.0
    assert clip.end_seconds == 1.5


def test_rejects_clip_start_before_parent() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata, 0.5, 1.5)

    with pytest.raises(ValueError, match="clip start_seconds must not precede"):
        sequence.clip(0.4, 1.0)


def test_rejects_clip_end_after_parent() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata, 0.5, 1.5)

    with pytest.raises(ValueError, match="clip end_seconds must not exceed"):
        sequence.clip(1.0, 1.6)


def test_tracks_landmarks_and_attaches_the_complete_result() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata, 0.5, 1.5)
    expected = make_track()
    tracker = FakeTracker(expected)

    result = sequence.track_landmarks(tracker, name="fake")

    assert result is expected
    assert sequence.landmark_tracks["fake"] is expected
    assert tracker.calls == [(Path("portrait.webm"), 0.5, 1.5)]


def test_does_not_mutate_sequence_when_tracking_fails() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata)

    with pytest.raises(RuntimeError, match="tracking failed"):
        sequence.track_landmarks(FailingTracker(), name="failed")

    assert dict(sequence.landmark_tracks) == {}


def test_requires_explicit_overwrite_for_an_existing_landmark_track() -> None:
    metadata = VideoMetadata(640, 480, 24.0, 2.0, True)
    sequence = TalkingFaceSequence(Path("portrait.webm"), metadata)
    first = FakeTracker(make_track())
    replacement = FakeTracker(make_track(1.0))
    sequence.track_landmarks(first, name="comparison")

    with pytest.raises(ValueError, match="landmark track already exists"):
        sequence.track_landmarks(replacement, name="comparison")

    assert replacement.calls == []
    replaced = sequence.track_landmarks(replacement, name="comparison", overwrite=True)
    assert sequence.landmark_tracks["comparison"] is replaced
