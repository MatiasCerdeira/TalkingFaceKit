"""Backend-independent animated triangular-mesh data contracts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FaceMeshTrack:
    """Triangular facial meshes aligned with decoded source-video frames.

    The topology is fixed across the complete timeline: every frame supplies new positions for the
    same ordered vertices, while ``triangles`` defines how those vertices form the surface. Failed
    detections remain aligned with the source timeline and contain only ``NaN`` vertex positions.
    Arrays are retained without copying; callers must not mutate them after construction.

    Attributes
    ----------
    topology
        Identifier describing the vertex ordering and triangle connectivity.
    coordinate_system
        Meaning, orientation, and units of the three vertex coordinates.
    frame_indices
        ``int64`` array shaped ``(frame_count,)`` containing strictly increasing, zero-based source
        decode indices.
    timestamps_seconds
        ``float64`` array shaped ``(frame_count,)`` containing strictly increasing source
        presentation timestamps in seconds.
    vertices
        ``float32`` array shaped ``(frame_count, vertex_count, 3)``. Detected rows contain finite
        coordinates; missing rows contain only ``NaN``.
    triangles
        ``int32`` array shaped ``(triangle_count, 3)``. Each row contains three distinct, zero-based
        indices into the vertex axis and is shared by every frame.
    detected
        Boolean array shaped ``(frame_count,)`` indicating whether a facial surface is available.
    """

    topology: str
    coordinate_system: str
    frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]
    detected: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate the mesh topology, coordinates, and source timeline."""
        for field_name, value in (
            ("topology", self.topology),
            ("coordinate_system", self.coordinate_system),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        if self.frame_indices.dtype != np.dtype(np.int64):
            raise TypeError(
                f"frame_indices must have dtype int64, got {self.frame_indices.dtype}"
            )
        if self.timestamps_seconds.dtype != np.dtype(np.float64):
            raise TypeError(
                f"timestamps_seconds must have dtype float64, got {self.timestamps_seconds.dtype}"
            )
        if self.vertices.dtype != np.dtype(np.float32):
            raise TypeError(
                f"vertices must have dtype float32, got {self.vertices.dtype}"
            )
        if self.triangles.dtype != np.dtype(np.int32):
            raise TypeError(
                f"triangles must have dtype int32, got {self.triangles.dtype}"
            )
        if self.detected.dtype != np.dtype(np.bool_):
            raise TypeError(f"detected must have dtype bool, got {self.detected.dtype}")

        if self.frame_indices.ndim != 1:
            raise ValueError(
                f"frame_indices must have shape (frame_count,), got {self.frame_indices.shape}"
            )
        frame_count = self.frame_indices.shape[0]
        if frame_count == 0:
            raise ValueError("a face mesh track must contain at least one frame")
        if self.timestamps_seconds.shape != (frame_count,):
            raise ValueError(
                "timestamps_seconds must have shape "
                f"({frame_count},), got {self.timestamps_seconds.shape}"
            )
        if self.detected.shape != (frame_count,):
            raise ValueError(
                f"detected must have shape ({frame_count},), got {self.detected.shape}"
            )
        if self.vertices.ndim != 3 or self.vertices.shape[0] != frame_count:
            raise ValueError(
                "vertices must have shape (frame_count, vertex_count, 3), "
                f"got {self.vertices.shape}"
            )
        if self.vertices.shape[1] == 0 or self.vertices.shape[2] != 3:
            raise ValueError(
                "vertices must have shape (frame_count, vertex_count, 3) with at least one vertex, "
                f"got {self.vertices.shape}"
            )
        vertex_count = self.vertices.shape[1]

        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,):
            raise ValueError(
                f"triangles must have shape (triangle_count, 3), got {self.triangles.shape}"
            )
        if self.triangles.shape[0] == 0:
            raise ValueError("a face mesh topology must contain at least one triangle")
        if np.any(self.triangles < 0) or np.any(self.triangles >= vertex_count):
            raise ValueError(
                f"triangle indices must be in [0, {vertex_count}), got an out-of-range index"
            )
        degenerate_triangles = (
            (self.triangles[:, 0] == self.triangles[:, 1])
            | (self.triangles[:, 1] == self.triangles[:, 2])
            | (self.triangles[:, 2] == self.triangles[:, 0])
        )
        if np.any(degenerate_triangles):
            raise ValueError("each triangle must reference three distinct vertices")

        if np.any(self.frame_indices < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if not np.all(np.isfinite(self.timestamps_seconds)):
            raise ValueError("timestamps_seconds must contain only finite values")
        if np.any(np.diff(self.timestamps_seconds) <= 0):
            raise ValueError("timestamps_seconds must be strictly increasing")

        if not np.all(np.isfinite(self.vertices[self.detected])):
            raise ValueError("detected vertex rows must contain only finite values")
        if not np.all(np.isnan(self.vertices[~self.detected])):
            raise ValueError("undetected vertex rows must contain only NaN values")

    @property
    def frame_count(self) -> int:
        """Number of decoded frames represented by this mesh track."""
        return int(self.frame_indices.shape[0])

    @property
    def vertex_count(self) -> int:
        """Number of ordered vertices in every facial mesh."""
        return int(self.vertices.shape[1])

    @property
    def triangle_count(self) -> int:
        """Number of triangular faces shared by every frame."""
        return int(self.triangles.shape[0])
