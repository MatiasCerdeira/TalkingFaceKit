from dataclasses import dataclass

import numpy as np
import pytest

from talkingfacekit import FaceLandmarkTrack
from talkingfacekit.tracking import mediapipe_mesh


@dataclass(frozen=True)
class Connection:
    start: int
    end: int


def make_landmark_track() -> FaceLandmarkTrack:
    landmarks = np.full((2, 478, 3), np.nan, dtype=np.float32)
    landmarks[0, :, 0] = 0.75
    landmarks[0, :, 1] = 0.25
    landmarks[0, :, 2] = -0.1
    return FaceLandmarkTrack(
        tracker_name="mediapipe-face-landmarker",
        tracker_version="test",
        topology="mediapipe-face-landmarker-478",
        coordinate_system="MediaPipe test coordinates",
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.04], dtype=np.float64),
        landmarks=landmarks,
        detected=np.asarray([True, False], dtype=np.bool_),
    )


def make_test_triangles() -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
    return np.asarray(
        [[index % 466, (index % 466) + 1, (index % 466) + 2] for index in range(852)],
        dtype=np.int32,
    )


def test_builds_aspect_corrected_surface_and_preserves_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mediapipe_mesh, "_load_mediapipe_triangles", make_test_triangles)
    landmarks = make_landmark_track()

    mesh = mediapipe_mesh.build_mediapipe_face_mesh(
        landmarks,
        image_width=1920,
        image_height=1080,
    )

    assert mesh.frame_count == 2
    assert mesh.vertex_count == 468
    assert mesh.triangle_count == 852
    assert mesh.frame_indices.tolist() == [0, 1]
    assert mesh.timestamps_seconds.tolist() == [0.0, 0.04]
    assert mesh.detected.tolist() == [True, False]
    assert mesh.vertices[0, 0].tolist() == pytest.approx([4.0 / 9.0, 0.25, 8.0 / 45.0])
    assert np.all(np.isnan(mesh.vertices[1]))


def test_decodes_directed_connection_cycles_into_triangles() -> None:
    connections = [
        Connection(2, 4),
        Connection(4, 7),
        Connection(7, 2),
        Connection(1, 3),
        Connection(3, 8),
        Connection(8, 1),
    ]

    triangles = mediapipe_mesh._connections_to_triangles(connections)

    assert triangles.dtype == np.dtype(np.int32)
    assert triangles.tolist() == [[2, 4, 7], [1, 3, 8]]


def test_rejects_non_mediapipe_landmark_topology() -> None:
    track = make_landmark_track()
    incompatible = FaceLandmarkTrack(
        tracker_name=track.tracker_name,
        tracker_version=track.tracker_version,
        topology="some-other-478-topology",
        coordinate_system=track.coordinate_system,
        frame_indices=track.frame_indices,
        timestamps_seconds=track.timestamps_seconds,
        landmarks=track.landmarks,
        detected=track.detected,
    )

    with pytest.raises(ValueError, match="requires topology"):
        mediapipe_mesh.build_mediapipe_face_mesh(
            incompatible,
            image_width=1920,
            image_height=1080,
        )
