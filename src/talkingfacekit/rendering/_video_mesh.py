"""Offline video/mesh comparison driven by original source presentation timestamps."""

import base64
import html
import importlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np

from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.rendering.plotly import _axis_ranges, _hidden_axis, _mesh_trace


def render_video_mesh_html(
    mesh: FaceMeshTrack,
    output_path: Path,
    *,
    source_video: Path,
    title: str,
    depth_scale: float,
    overwrite: bool,
    video_time_offset_seconds: float,
) -> Path:
    """Write a comparison document after the public renderer validates common options."""
    source = source_video.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mesh comparison source video not found: {source}")
    if source == output_path.resolve():
        raise ValueError("mesh comparison output must differ from source video")
    if mesh.timestamps_seconds[0] - video_time_offset_seconds < 0:
        raise ValueError("video offset places mesh frames before the browser media timeline")
    try:
        offline = cast(Any, importlib.import_module("plotly.offline"))
    except ImportError as error:
        raise ImportError("Video/mesh comparison requires the optional rendering extra") from error
    first_available = int(np.flatnonzero(mesh.detected)[0])
    trace = _mesh_trace(mesh.vertices[first_available], mesh.triangles, depth_scale=depth_scale)
    trace["visible"] = False
    ranges = _axis_ranges(mesh, depth_scale=depth_scale)
    payload = {
        "videoUrl": source.as_uri(),
        "videoName": source.name,
        "times": mesh.timestamps_seconds.tolist(),
        "indices": mesh.frame_indices.tolist(),
        "detected": mesh.detected.tolist(),
        "vertexCount": mesh.vertex_count,
        "vertices": base64.b64encode(np.asarray(mesh.vertices, dtype="<f4").tobytes()).decode(
            "ascii"
        ),
        "depthScale": depth_scale,
        "videoOffset": video_time_offset_seconds,
        "trace": trace,
        "layout": {
            "paper_bgcolor": "#11151c",
            "font": {"color": "#e8edf4"},
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "uirevision": "mesh-comparison",
            "scene": {
                "bgcolor": "#11151c",
                "aspectmode": "data",
                "camera": {"up": {"x": 0, "y": 1, "z": 0}, "eye": {"x": 0.45, "y": 0.05, "z": 1.8}},
                "xaxis": _hidden_axis(ranges[0]),
                "yaxis": _hidden_axis(ranges[1]),
                "zaxis": _hidden_axis(ranges[2]),
            },
        },
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    replacements = {
        "TITLE": html.escape(title),
        "DATA": serialized,
        "PLOTLY": offline.get_plotlyjs(),
    }
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    rendered = re.sub(
        r"__TFK_(TITLE|DATA|PLOTLY)__", lambda match: replacements[match[1]], template
    )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
        if overwrite:
            temporary.replace(output_path)
        else:
            os.link(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path
