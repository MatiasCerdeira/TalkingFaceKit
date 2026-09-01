from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import FaceLandmarkTrack, TalkingFaceSequence, VideoMetadata, VideoSource


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


def make_source() -> VideoSource:
    return VideoSource(
        path=Path("portrait.webm"),
        metadata=VideoMetadata(640, 480, 24.0, 2.0, True),
    )


def test_constructs_sequence_without_accessing_its_media_path() -> None:
    source = VideoSource(
        path=Path("video-that-does-not-need-to-exist.webm"),
        metadata=VideoMetadata(
            width=640,
            height=480,
            average_fps=24.0,
            stream_duration_seconds=2.0,
            has_audio=True,
        ),
    )
    sequence = TalkingFaceSequence(
        source=source,
        start_seconds=0.5,
        end_seconds=1.5,
    )

    assert sequence.source is source
    assert sequence.source.path == Path("video-that-does-not-need-to-exist.webm")
    assert sequence.source.metadata.stream_duration_seconds == 2.0
    assert sequence.start_seconds == 0.5
    assert sequence.end_seconds == 1.5
    assert sequence.duration_seconds == 1.0


def test_open_ended_sequence_has_no_declared_duration() -> None:
    sequence = TalkingFaceSequence(make_source())

    assert sequence.source.metadata.stream_duration_seconds == 2.0
    assert sequence.duration_seconds is None


def test_requires_a_video_source() -> None:
    with pytest.raises(TypeError, match="source must be a VideoSource instance"):
        TalkingFaceSequence(object())  # type: ignore[arg-type]


def test_prevents_mutating_the_source_or_interval() -> None:
    sequence = TalkingFaceSequence(make_source(), start_seconds=0.5, end_seconds=1.5)
    start_field = "start_seconds"
    path_field = "path"

    with pytest.raises(AttributeError):
        setattr(sequence, start_field, -1.0)
    with pytest.raises(AttributeError):
        setattr(sequence.source, path_field, Path("other.webm"))


@pytest.mark.parametrize("start_seconds", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_rejects_invalid_sequence_start(start_seconds: float) -> None:
    with pytest.raises(ValueError, match="start_seconds must be finite and non-negative"):
        TalkingFaceSequence(make_source(), start_seconds=start_seconds)


@pytest.mark.parametrize("end_seconds", [float("inf"), float("-inf"), float("nan")])
def test_rejects_non_finite_sequence_end(end_seconds: float) -> None:
    with pytest.raises(ValueError, match="end_seconds must be finite when provided"):
        TalkingFaceSequence(make_source(), end_seconds=end_seconds)


@pytest.mark.parametrize("end_seconds", [0.5, 0.4])
def test_rejects_sequence_end_not_after_start(end_seconds: float) -> None:
    with pytest.raises(ValueError, match="end_seconds must be greater than start_seconds"):
        TalkingFaceSequence(
            make_source(),
            start_seconds=0.5,
            end_seconds=end_seconds,
        )


def test_clips_sequence_without_decoding_or_mutating_the_parent() -> None:
    source = make_source()
    sequence = TalkingFaceSequence(source, 0.5, 1.5)
    sequence.track_landmarks(FakeTracker(make_track()), name="parent")

    clip = sequence.clip(0.75, 1.25)

    assert clip is not sequence
    assert clip.source is source
    assert clip.start_seconds == 0.75
    assert clip.end_seconds == 1.25
    assert clip.duration_seconds == 0.5
    assert dict(clip.landmark_tracks) == {}
    assert sequence.start_seconds == 0.5
    assert sequence.end_seconds == 1.5
    assert "parent" in sequence.landmark_tracks


def test_clip_uses_the_parent_end_when_end_is_omitted() -> None:
    sequence = TalkingFaceSequence(make_source(), 0.5, 1.5)

    clip = sequence.clip(1.0)

    assert clip.start_seconds == 1.0
    assert clip.end_seconds == 1.5


def test_clip_remains_open_when_its_parent_is_open() -> None:
    sequence = TalkingFaceSequence(make_source())

    clip = sequence.clip(1.0)

    assert clip.start_seconds == 1.0
    assert clip.end_seconds is None
    assert clip.duration_seconds is None


def test_rejects_clip_start_before_parent() -> None:
    sequence = TalkingFaceSequence(make_source(), 0.5, 1.5)

    with pytest.raises(ValueError, match="clip start_seconds must not precede"):
        sequence.clip(0.4, 1.0)


def test_rejects_clip_end_after_parent() -> None:
    sequence = TalkingFaceSequence(make_source(), 0.5, 1.5)

    with pytest.raises(ValueError, match="clip end_seconds must not exceed"):
        sequence.clip(1.0, 1.6)


def test_tracks_landmarks_and_attaches_the_complete_result() -> None:
    sequence = TalkingFaceSequence(make_source(), 0.5, 1.5)
    expected = make_track()
    tracker = FakeTracker(expected)

    result = sequence.track_landmarks(tracker, name="fake")

    assert result is expected
    assert sequence.landmark_tracks["fake"] is expected
    assert tracker.calls == [(Path("portrait.webm"), 0.5, 1.5)]


def test_does_not_mutate_sequence_when_tracking_fails() -> None:
    sequence = TalkingFaceSequence(make_source())

    with pytest.raises(RuntimeError, match="tracking failed"):
        sequence.track_landmarks(FailingTracker(), name="failed")

    assert dict(sequence.landmark_tracks) == {}


def test_requires_explicit_overwrite_for_an_existing_landmark_track() -> None:
    sequence = TalkingFaceSequence(make_source())
    first = FakeTracker(make_track())
    replacement = FakeTracker(make_track(1.0))
    sequence.track_landmarks(first, name="comparison")

    with pytest.raises(ValueError, match="landmark track already exists"):
        sequence.track_landmarks(replacement, name="comparison")

    assert replacement.calls == []
    replaced = sequence.track_landmarks(replacement, name="comparison", overwrite=True)
    assert sequence.landmark_tracks["comparison"] is replaced
