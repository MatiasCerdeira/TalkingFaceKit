"""Offline interactive HTML rendering for animated facial meshes."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.mesh import FaceMeshTrack

_MATERIAL_COLOR = "rgb(92, 152, 214)"
_BACKGROUND_COLOR = "rgb(17, 21, 28)"


class _PlotlyIO(Protocol):
    def write_html(
        self,
        figure: dict[str, object],
        file: str,
        *,
        config: dict[str, object],
        auto_play: bool,
        include_plotlyjs: bool,
        full_html: bool,
    ) -> None: ...


def render_face_mesh_html(
    mesh: FaceMeshTrack,
    output_path: str | Path,
    *,
    title: str = "TalkingFaceKit animated face mesh",
    depth_scale: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Render an animated face-mesh timeline to a self-contained interactive HTML file.

    The renderer uses a neutral material and virtual lighting; it does not sample colors or pixels
    from the source video. Users can orbit and zoom the camera, step through the source timeline,
    and play or pause the mesh animation. Plotly's JavaScript runtime is embedded in the output, so
    the resulting file works offline.

    The renderer multiplies displayed z coordinates by ``depth_scale`` without mutating ``mesh``.
    A value of ``1.0`` preserves the supplied relative depth. Playback uses the median interval
    between source timestamps because Plotly animation frames use one display duration; every
    frame's original timestamp remains visible in the title and slider.

    Parameters
    ----------
    mesh
        Validated animated triangular facial surface.
    output_path
        Destination path ending in ``.html``. Its parent directory must already exist.
    title
        Non-empty heading embedded in the interactive figure.
    depth_scale
        Positive finite display-only multiplier for the z axis. It does not alter the mesh object.
    overwrite
        Whether an existing regular file may be replaced after rendering completes.

    Returns
    -------
    pathlib.Path
        Destination path supplied by the caller, normalized as a ``Path``.

    Raises
    ------
    FileNotFoundError
        If the destination directory does not exist.
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    IsADirectoryError
        If the destination refers to a directory.
    ImportError
        If the optional Plotly dependency is not installed.
    ValueError
        If the path, title, depth scale, or detection timeline cannot be rendered.
    """
    path = _validate_output_path(output_path, overwrite=overwrite)
    if not title.strip():
        raise ValueError("render title must not be empty")
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be finite and positive, got {depth_scale}")
    if not np.any(mesh.detected):
        raise ValueError("face mesh renderer requires at least one detected frame")

    figure = _build_figure(mesh, title=title, depth_scale=depth_scale)
    plotly_io = _load_plotly_io()
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.html",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        plotly_io.write_html(
            figure,
            str(temporary_path),
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            auto_play=False,
            include_plotlyjs=True,
            full_html=True,
        )
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return path


def _build_figure(
    mesh: FaceMeshTrack,
    *,
    title: str,
    depth_scale: float,
) -> dict[str, object]:
    traces = [
        _mesh_trace(mesh.vertices[frame_position], mesh.triangles, depth_scale=depth_scale)
        for frame_position in range(mesh.frame_count)
    ]
    frame_names = [f"frame-{frame_position:06d}" for frame_position in range(mesh.frame_count)]
    frames: list[dict[str, object]] = []
    for frame_position, (trace, frame_name) in enumerate(zip(traces, frame_names, strict=True)):
        timestamp = float(mesh.timestamps_seconds[frame_position])
        detected_label = "detected" if bool(mesh.detected[frame_position]) else "missing"
        frames.append(
            {
                "name": frame_name,
                "data": [trace],
                "traces": [0],
                "layout": {
                    "title": {
                        "text": (
                            f"{title}<br><sup>frame {int(mesh.frame_indices[frame_position])} · "
                            f"{timestamp:.3f} s · {detected_label}</sup>"
                        )
                    }
                },
            }
        )

    frame_duration_ms = _median_frame_duration_ms(mesh.timestamps_seconds)
    slider_steps = _slider_steps(mesh, frame_names)
    axis_ranges = _axis_ranges(mesh, depth_scale=depth_scale)
    initial_timestamp = float(mesh.timestamps_seconds[0])
    initial_detection = "detected" if bool(mesh.detected[0]) else "missing"

    return {
        "data": [traces[0]],
        "frames": frames,
        "layout": {
            "title": {
                "text": (
                    f"{title}<br><sup>frame {int(mesh.frame_indices[0])} · "
                    f"{initial_timestamp:.3f} s · {initial_detection}</sup>"
                ),
                "x": 0.5,
                "xanchor": "center",
            },
            "paper_bgcolor": _BACKGROUND_COLOR,
            "plot_bgcolor": _BACKGROUND_COLOR,
            "font": {"color": "rgb(232, 237, 244)"},
            "margin": {"l": 0, "r": 0, "t": 76, "b": 100},
            "scene": {
                "bgcolor": _BACKGROUND_COLOR,
                "aspectmode": "data",
                "camera": {
                    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "eye": {"x": 0.45, "y": 0.05, "z": 1.8},
                },
                "xaxis": _hidden_axis(axis_ranges[0]),
                "yaxis": _hidden_axis(axis_ranges[1]),
                "zaxis": _hidden_axis(axis_ranges[2]),
            },
            "updatemenus": [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.03,
                    "yanchor": "top",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "frame": {
                                        "duration": frame_duration_ms,
                                        "redraw": True,
                                    },
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            "sliders": [
                {
                    "active": 0,
                    "x": 0.08,
                    "len": 0.84,
                    "y": 0.02,
                    "currentvalue": {"prefix": "Source time: ", "suffix": " s"},
                    "transition": {"duration": 0},
                    "steps": slider_steps,
                }
            ],
        },
    }


def _mesh_trace(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    *,
    depth_scale: float,
) -> dict[str, object]:
    if not np.all(np.isfinite(vertices)):
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        triangle_indices: list[int] = []
    else:
        x_values = vertices[:, 0].tolist()
        y_values = vertices[:, 1].tolist()
        z_values = (vertices[:, 2] * np.float32(depth_scale)).tolist()
        triangle_indices = triangles[:, 0].tolist()

    return {
        "type": "mesh3d",
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "i": triangle_indices,
        "j": [] if not triangle_indices else triangles[:, 1].tolist(),
        "k": [] if not triangle_indices else triangles[:, 2].tolist(),
        "color": _MATERIAL_COLOR,
        "flatshading": False,
        "hoverinfo": "skip",
        "lighting": {
            "ambient": 0.35,
            "diffuse": 0.8,
            "specular": 0.35,
            "roughness": 0.55,
            "fresnel": 0.1,
        },
        "lightposition": {"x": -1.5, "y": 2.0, "z": 3.0},
        "showscale": False,
    }


def _slider_steps(mesh: FaceMeshTrack, frame_names: list[str]) -> list[dict[str, object]]:
    label_stride = max(1, math.ceil(mesh.frame_count / 8))
    steps: list[dict[str, object]] = []
    for frame_position, frame_name in enumerate(frame_names):
        timestamp = float(mesh.timestamps_seconds[frame_position])
        show_label = frame_position % label_stride == 0 or frame_position == mesh.frame_count - 1
        steps.append(
            {
                "label": f"{timestamp:.2f}" if show_label else "",
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )
    return steps


def _median_frame_duration_ms(timestamps_seconds: NDArray[np.float64]) -> int:
    if timestamps_seconds.shape[0] == 1:
        return 1_000
    median_seconds = float(np.median(np.diff(timestamps_seconds)))
    return max(1, round(median_seconds * 1_000.0))


def _axis_ranges(
    mesh: FaceMeshTrack,
    *,
    depth_scale: float,
) -> tuple[list[float], list[float], list[float]]:
    detected_vertices = mesh.vertices[mesh.detected].copy()
    detected_vertices[..., 2] *= np.float32(depth_scale)
    flattened = detected_vertices.reshape(-1, 3)
    minimum = np.min(flattened, axis=0)
    maximum = np.max(flattened, axis=0)
    span = maximum - minimum
    fallback_span = max(float(np.max(span)), 1.0) * 0.05
    padding = np.maximum(span * np.float32(0.08), np.float32(fallback_span))
    ranges = [
        [float(minimum[axis] - padding[axis]), float(maximum[axis] + padding[axis])]
        for axis in range(3)
    ]
    return ranges[0], ranges[1], ranges[2]


def _hidden_axis(axis_range: list[float]) -> dict[str, object]:
    return {
        "range": axis_range,
        "showbackground": False,
        "showgrid": False,
        "showline": False,
        "showticklabels": False,
        "title": {"text": ""},
        "visible": False,
        "zeroline": False,
    }


def _validate_output_path(output_path: str | Path, *, overwrite: bool) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".html":
        raise ValueError(f"Face mesh render output path must end in .html: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Face mesh render output directory not found: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"Face mesh render output parent is not a directory: {path.parent}")
    if path.is_dir():
        raise IsADirectoryError(f"Face mesh render output path is a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Face mesh render output already exists: {path}")
    return path


def _load_plotly_io() -> _PlotlyIO:
    try:
        plotly_io = importlib.import_module("plotly.io")
    except ImportError as error:
        raise ImportError(
            "Plotly rendering is optional. Install it with `uv sync --extra rendering`."
        ) from error
    return cast(_PlotlyIO, plotly_io)
