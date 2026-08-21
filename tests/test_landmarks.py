import numpy as np
import pytest

from talkingfacekit import FaceLandmarkTrack


def make_landmark_track() -> FaceLandmarkTrack:
    landmarks = np.full((2, 3, 3), np.nan, dtype=np.float32)
    landmarks[0] = np.arange(9, dtype=np.float32).reshape(3, 3)
    return FaceLandmarkTrack(
        tracker_name="test-tracker",
        tracker_version="1.0",
        topology="test-3-points",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([4, 5], dtype=np.int64),
        timestamps_seconds=np.asarray([0.2, 0.3], dtype=np.float64),
        landmarks=landmarks,
        detected=np.asarray([True, False], dtype=np.bool_),
    )


def test_represents_detected_and_missing_frames_on_one_timeline() -> None:
    track = make_landmark_track()

    assert track.frame_count == 2
    assert track.landmark_count == 3
    assert track.frame_indices.tolist() == [4, 5]
    assert np.all(np.isfinite(track.landmarks[0]))
    assert np.all(np.isnan(track.landmarks[1]))


def test_rejects_non_nan_landmarks_for_an_undetected_frame() -> None:
    track = make_landmark_track()
    invalid_landmarks = track.landmarks.copy()
    invalid_landmarks[1, 0, 0] = 0.0

    with pytest.raises(ValueError, match="undetected landmark rows"):
        FaceLandmarkTrack(
            tracker_name=track.tracker_name,
            tracker_version=track.tracker_version,
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=track.timestamps_seconds,
            landmarks=invalid_landmarks,
            detected=track.detected,
        )


def test_rejects_non_increasing_timestamps() -> None:
    track = make_landmark_track()

    with pytest.raises(ValueError, match="timestamps_seconds must be strictly increasing"):
        FaceLandmarkTrack(
            tracker_name=track.tracker_name,
            tracker_version=track.tracker_version,
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=np.asarray([0.2, 0.2], dtype=np.float64),
            landmarks=track.landmarks,
            detected=track.detected,
        )


def test_rejects_wrong_landmark_dtype() -> None:
    track = make_landmark_track()

    with pytest.raises(TypeError, match="landmarks must have dtype float32"):
        FaceLandmarkTrack(
            tracker_name=track.tracker_name,
            tracker_version=track.tracker_version,
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=track.timestamps_seconds,
            landmarks=track.landmarks.astype(np.float64),
            detected=track.detected,
        )
