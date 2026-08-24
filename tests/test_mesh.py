import numpy as np
import pytest

from talkingfacekit import FaceMeshTrack


def make_mesh_track() -> FaceMeshTrack:
    vertices = np.full((2, 4, 3), np.nan, dtype=np.float32)
    vertices[0] = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return FaceMeshTrack(
        topology="test-quad",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([4, 5], dtype=np.int64),
        timestamps_seconds=np.asarray([0.2, 0.3], dtype=np.float64),
        vertices=vertices,
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        detected=np.asarray([True, False], dtype=np.bool_),
    )


def test_represents_one_topology_across_a_detected_and_missing_timeline() -> None:
    track = make_mesh_track()

    assert track.frame_count == 2
    assert track.vertex_count == 4
    assert track.triangle_count == 2
    assert np.all(np.isfinite(track.vertices[0]))
    assert np.all(np.isnan(track.vertices[1]))


def test_rejects_triangle_index_outside_vertex_axis() -> None:
    track = make_mesh_track()

    with pytest.raises(ValueError, match="triangle indices must be in"):
        FaceMeshTrack(
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=track.timestamps_seconds,
            vertices=track.vertices,
            triangles=np.asarray([[0, 1, 4]], dtype=np.int32),
            detected=track.detected,
        )


def test_rejects_degenerate_triangle() -> None:
    track = make_mesh_track()

    with pytest.raises(ValueError, match="three distinct vertices"):
        FaceMeshTrack(
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=track.timestamps_seconds,
            vertices=track.vertices,
            triangles=np.asarray([[0, 1, 1]], dtype=np.int32),
            detected=track.detected,
        )


def test_rejects_finite_vertices_for_an_undetected_frame() -> None:
    track = make_mesh_track()
    invalid_vertices = track.vertices.copy()
    invalid_vertices[1, 0] = 0.0

    with pytest.raises(ValueError, match="undetected vertex rows"):
        FaceMeshTrack(
            topology=track.topology,
            coordinate_system=track.coordinate_system,
            frame_indices=track.frame_indices,
            timestamps_seconds=track.timestamps_seconds,
            vertices=invalid_vertices,
            triangles=track.triangles,
            detected=track.detected,
        )
