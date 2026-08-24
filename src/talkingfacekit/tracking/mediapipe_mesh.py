"""Convert MediaPipe facial landmarks into backend-independent triangular meshes."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack

_LANDMARK_TOPOLOGY = "mediapipe-face-landmarker-478"
_SURFACE_VERTEX_COUNT = 468
_SURFACE_TRIANGLE_COUNT = 852
_MESH_TOPOLOGY = "mediapipe-face-mesh-468-852"
_MESH_COORDINATE_SYSTEM = (
    "Centered MediaPipe display coordinates measured in image-height units: x increases "
    "left-to-right, y increases bottom-to-top, and z is negated MediaPipe-relative depth; x and z "
    "are corrected by the source image aspect ratio"
)


class _Connection(Protocol):
    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


def build_mediapipe_face_mesh(
    track: FaceLandmarkTrack,
    *,
    image_width: int,
    image_height: int,
) -> FaceMeshTrack:
    """Build an animated triangular facial surface from a MediaPipe landmark track.

    The first 468 landmarks become surface vertices. The ten iris landmarks remain outside this
    surface because MediaPipe exposes them as separate contours rather than tessellated skin. The
    fixed 852-triangle topology is loaded from the installed MediaPipe integration.

    Normalized image coordinates are centered and converted to image-height units. Horizontal and
    depth coordinates are multiplied by ``image_width / image_height`` to preserve the source image
    aspect ratio, the vertical axis is flipped to point upward, and MediaPipe-relative depth is
    negated for a conventional display orientation. This conversion is suitable for visualization;
    it does not produce metric 3D geometry.

    Parameters
    ----------
    track
        Validated MediaPipe 478-landmark timeline.
    image_width
        Positive source-video width in pixels.
    image_height
        Positive source-video height in pixels.

    Returns
    -------
    FaceMeshTrack
        Frame-aligned mesh sequence with 468 vertices and 852 triangles per detected frame.

    Raises
    ------
    ImportError
        If the optional MediaPipe dependency is not installed.
    TypeError
        If an image dimension is not an integer.
    ValueError
        If the dimensions are non-positive, the landmark topology is incompatible, or MediaPipe's
        tessellation does not satisfy the expected 468-vertex, 852-triangle contract.
    """
    _validate_image_dimension("image_width", image_width)
    _validate_image_dimension("image_height", image_height)
    if track.topology != _LANDMARK_TOPOLOGY:
        raise ValueError(
            f"MediaPipe mesh conversion requires topology {_LANDMARK_TOPOLOGY!r}, "
            f"got {track.topology!r}"
        )
    if track.landmark_count != 478:
        raise ValueError(
            f"MediaPipe mesh conversion requires 478 landmarks, got {track.landmark_count}"
        )

    aspect_ratio = np.float32(image_width / image_height)
    vertices = track.landmarks[:, :_SURFACE_VERTEX_COUNT, :].copy()
    vertices[..., 0] = (vertices[..., 0] - np.float32(0.5)) * aspect_ratio
    vertices[..., 1] = np.float32(0.5) - vertices[..., 1]
    vertices[..., 2] = -vertices[..., 2] * aspect_ratio

    return FaceMeshTrack(
        topology=_MESH_TOPOLOGY,
        coordinate_system=_MESH_COORDINATE_SYSTEM,
        frame_indices=track.frame_indices.copy(),
        timestamps_seconds=track.timestamps_seconds.copy(),
        vertices=vertices,
        triangles=_load_mediapipe_triangles(),
        detected=track.detected.copy(),
    )


def _validate_image_dimension(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _load_mediapipe_triangles() -> NDArray[np.int32]:
    try:
        # MediaPipe 1.0.0 does not publish a py.typed marker or stub files.
        mediapipe = importlib.import_module("mediapipe")
    except ImportError as error:
        raise ImportError(
            "MediaPipe mesh conversion is optional. Install it with "
            "`uv sync --extra tracking-mediapipe`."
        ) from error

    connections = cast(
        Sequence[_Connection],
        mediapipe.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )
    triangles = _connections_to_triangles(connections)
    if triangles.shape != (_SURFACE_TRIANGLE_COUNT, 3):
        raise ValueError(
            "MediaPipe face tessellation must contain "
            f"{_SURFACE_TRIANGLE_COUNT} triangles, got {triangles.shape[0]}"
        )
    if int(triangles.max()) >= _SURFACE_VERTEX_COUNT:
        raise ValueError(
            "MediaPipe face tessellation references an iris or unknown vertex; "
            f"expected indices below {_SURFACE_VERTEX_COUNT}"
        )
    return triangles


def _connections_to_triangles(connections: Sequence[_Connection]) -> NDArray[np.int32]:
    if len(connections) % 3 != 0:
        raise ValueError(
            "MediaPipe face tessellation connections must form groups of three directed edges"
        )

    triangles: list[tuple[int, int, int]] = []
    for offset in range(0, len(connections), 3):
        first, second, third = connections[offset : offset + 3]
        if not (
            first.end == second.start and second.end == third.start and third.end == first.start
        ):
            raise ValueError(
                "MediaPipe face tessellation connection group "
                f"{offset // 3} does not form one directed triangle"
            )
        triangles.append((first.start, first.end, second.end))

    return np.asarray(triangles, dtype=np.int32)
