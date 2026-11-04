import base64
import importlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.rendering import render_face_mesh_html


def comparison_mesh() -> FaceMeshTrack:
    vertices = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0.25]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0.25]],
            [[0.5, 0, 0], [1.5, 0, 0], [0.5, 1, 0.5]],
        ],
        dtype=np.float32,
    )
    vertices[1] = np.nan
    return FaceMeshTrack(
        topology="synthetic-triangle",
        coordinate_system="test coordinates",
        frame_indices=np.array([600, 601, 602], dtype=np.int64),
        timestamps_seconds=np.array([20, 20.03, 20.11], dtype=np.float64),
        vertices=vertices,
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        detected=np.array([True, False, True], dtype=np.bool_),
    )


def render_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, offset: float = 0
) -> Path:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"Referenced only; the renderer does not decode this fixture")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(get_plotlyjs=lambda: "/* offline Plotly fixture */"),
    )
    return render_face_mesh_html(
        comparison_mesh(),
        tmp_path / "comparison.html",
        source_video=video,
        video_time_offset_seconds=offset,
    )


def test_comparison_preserves_binary_vertices_timestamps_and_single_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = render_comparison(tmp_path, monkeypatch)
    document = path.read_text()
    match = re.search(
        r'<script id="mesh-data" type="application/json">(.*?)</script>', document, re.DOTALL
    )
    assert match is not None
    data = json.loads(match[1])
    expected = comparison_mesh()
    assert data["videoUrl"] == (tmp_path / "source.mp4").as_uri()
    assert data["times"] == expected.timestamps_seconds.tolist()
    assert data["indices"] == expected.frame_indices.tolist()
    assert data["detected"] == [True, False, True]
    assert base64.b64decode(data["vertices"]) == expected.vertices.astype("<f4").tobytes()
    assert data["trace"]["i"] == [0] and data["trace"]["j"] == [1] and data["trace"]["k"] == [2]
    assert data["trace"]["visible"] is False
    assert "<video" in document
    assert "requestVideoFrameCallback" in document
    assert "/* offline Plotly fixture */" in document
    assert "__TFK_" not in document


def test_comparison_escapes_title_and_source_name_without_replacing_payload_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-<script>-__TFK_PLOTLY__.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(get_plotlyjs=lambda: "/* fixture */"),
    )
    output = render_face_mesh_html(
        comparison_mesh(),
        tmp_path / "safe.html",
        source_video=source,
        title='</script><script>alert("unsafe")</script>__TFK_DATA__',
    ).read_text()
    assert '<script>alert("unsafe")</script>' not in output
    assert "&lt;/script&gt;" in output
    match = re.search(
        r'<script id="mesh-data" type="application/json">(.*?)</script>', output, re.DOTALL
    )
    assert match is not None
    assert json.loads(match[1])["videoName"] == source.name


def test_comparison_can_start_with_missing_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render_comparison(tmp_path, monkeypatch)
    mesh = comparison_mesh()
    vertices = mesh.vertices.copy()
    vertices[0] = np.nan
    mesh = replace(mesh, vertices=vertices, detected=np.array([False, False, True]))
    output = render_face_mesh_html(
        mesh, tmp_path / "leading-missing.html", source_video=tmp_path / "source.mp4"
    ).read_text()
    match = re.search(
        r'<script id="mesh-data" type="application/json">(.*?)</script>', output, re.DOTALL
    )
    assert match is not None
    data = json.loads(match[1])
    assert data["detected"] == [False, False, True]
    assert data["trace"]["visible"] is False
    assert data["trace"]["i"] == [0]


def test_comparison_requires_original_video_and_explicit_valid_time_mapping(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source video"):
        render_face_mesh_html(
            comparison_mesh(), tmp_path / "missing.html", source_video=tmp_path / "missing.mp4"
        )
    with pytest.raises(ValueError, match="requires source_video"):
        render_face_mesh_html(
            comparison_mesh(), tmp_path / "offset.html", video_time_offset_seconds=1
        )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    with pytest.raises(ValueError, match="before the browser media timeline"):
        render_face_mesh_html(
            comparison_mesh(),
            tmp_path / "offset.html",
            source_video=source,
            video_time_offset_seconds=21,
        )


def test_comparison_refuses_overwrite_and_preserves_previous_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = render_comparison(tmp_path, monkeypatch)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        render_face_mesh_html(comparison_mesh(), path, source_video=tmp_path / "source.mp4")

    def fail(*args: object) -> None:
        raise OSError("cannot publish")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="cannot publish"):
        render_face_mesh_html(
            comparison_mesh(), tmp_path / "failed.html", source_video=tmp_path / "source.mp4"
        )
    assert path.read_bytes() == original
    assert sorted(item.name for item in tmp_path.iterdir()) == ["comparison.html", "source.mp4"]


@pytest.mark.parametrize("frame_callbacks", [True, False])
@pytest.mark.parametrize("offset", [0.0, 5.0])
def test_video_clock_controller_handles_vfr_missing_seeking_and_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame_callbacks: bool, offset: float
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser-free controller regression test")
    path = render_comparison(tmp_path, monkeypatch, offset=offset)
    completed = subprocess.run(
        [
            node,
            str(Path(__file__).with_name("mesh_video_controller.cjs")),
            str(path),
            str(frame_callbacks),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
