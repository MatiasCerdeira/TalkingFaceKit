from pathlib import Path

import numpy as np
import pytest

from talkingfacekit import FaceLandmarkTrack, load_landmark_track, save_landmark_track


def make_landmark_track(*, tracker_name: str = "test-tracker") -> FaceLandmarkTrack:
    landmarks = np.full((3, 4, 3), np.nan, dtype=np.float32)
    landmarks[0] = np.arange(12, dtype=np.float32).reshape(4, 3)
    landmarks[2] = np.arange(12, 24, dtype=np.float32).reshape(4, 3)
    return FaceLandmarkTrack(
        tracker_name=tracker_name,
        tracker_version="1.2.3",
        topology="test-4-points",
        coordinate_system="synthetic test coordinates",
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        timestamps_seconds=np.asarray([0.4, 0.45, 0.5], dtype=np.float64),
        landmarks=landmarks,
        detected=np.asarray([True, False, True], dtype=np.bool_),
    )


def test_round_trips_a_complete_landmark_track(tmp_path: Path) -> None:
    expected = make_landmark_track()
    archive_path = tmp_path / "face_landmarks.npz"

    saved_path = save_landmark_track(expected, archive_path)
    loaded = load_landmark_track(saved_path)

    assert saved_path == archive_path
    assert loaded.tracker_name == expected.tracker_name
    assert loaded.tracker_version == expected.tracker_version
    assert loaded.topology == expected.topology
    assert loaded.coordinate_system == expected.coordinate_system
    np.testing.assert_array_equal(loaded.frame_indices, expected.frame_indices)
    np.testing.assert_array_equal(loaded.timestamps_seconds, expected.timestamps_seconds)
    np.testing.assert_allclose(loaded.landmarks, expected.landmarks, equal_nan=True)
    np.testing.assert_array_equal(loaded.detected, expected.detected)


def test_requires_explicit_overwrite(tmp_path: Path) -> None:
    archive_path = tmp_path / "face_landmarks.npz"
    save_landmark_track(make_landmark_track(tracker_name="first"), archive_path)

    with pytest.raises(FileExistsError, match="already exists"):
        save_landmark_track(make_landmark_track(tracker_name="replacement"), archive_path)

    loaded_first = load_landmark_track(archive_path)
    assert loaded_first.tracker_name == "first"

    save_landmark_track(
        make_landmark_track(tracker_name="replacement"),
        archive_path,
        overwrite=True,
    )
    loaded_replacement = load_landmark_track(archive_path)
    assert loaded_replacement.tracker_name == "replacement"


def test_rejects_an_archive_with_missing_fields(tmp_path: Path) -> None:
    archive_path = tmp_path / "incomplete.npz"
    np.savez_compressed(archive_path, frame_indices=np.asarray([0], dtype=np.int64))

    with pytest.raises(ValueError, match="missing required fields"):
        load_landmark_track(archive_path)


def test_rejects_non_npz_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in .npz"):
        save_landmark_track(make_landmark_track(), tmp_path / "landmarks.json")
