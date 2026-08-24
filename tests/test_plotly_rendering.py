from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest

from talkingfacekit import FaceMeshTrack
from talkingfacekit.rendering import plotly as plotly_renderer


def make_mesh_track() -> FaceMeshTrack:
    vertices = np.asarray(
        [
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.2]],
            [[-0.4, -0.5, 0.0], [0.6, -0.5, 0.0], [0.1, 0.5, 0.3]],
        ],
        dtype=np.float32,
    )
    return FaceMeshTrack(
        topology="test-triangle",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([10, 11], dtype=np.int64),
        timestamps_seconds=np.asarray([0.5, 0.54], dtype=np.float64),
        vertices=vertices,
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        detected=np.asarray([True, True], dtype=np.bool_),
    )


class FakePlotlyIO:
    observed_figures: ClassVar[list[dict[str, object]]] = []

    @classmethod
    def write_html(
        cls,
        figure: dict[str, object],
        file: str,
        *,
        config: dict[str, object],
        auto_play: bool,
        include_plotlyjs: bool,
        full_html: bool,
    ) -> None:
        assert config["responsive"] is True
        assert auto_play is False
        assert include_plotlyjs is True
        assert full_html is True
        cls.observed_figures.append(figure)
        Path(file).write_text("<html>rendered mesh</html>", encoding="utf-8")


def test_renders_every_mesh_frame_to_self_contained_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakePlotlyIO.observed_figures.clear()
    monkeypatch.setattr(plotly_renderer, "_load_plotly_io", lambda: FakePlotlyIO)
    output_path = tmp_path / "face-mesh.html"

    saved_path = plotly_renderer.render_face_mesh_html(make_mesh_track(), output_path)

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8") == "<html>rendered mesh</html>"
    assert len(FakePlotlyIO.observed_figures) == 1
    observed_figure = FakePlotlyIO.observed_figures[0]
    frames = cast(list[dict[str, object]], observed_figure["frames"])
    initial_data = cast(list[dict[str, object]], observed_figure["data"])
    assert len(frames) == 2
    initial_trace = initial_data[0]
    assert initial_trace["i"] == [0]
    assert initial_trace["j"] == [1]
    assert initial_trace["k"] == [2]


def test_requires_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plotly_renderer, "_load_plotly_io", lambda: FakePlotlyIO)
    output_path = tmp_path / "face-mesh.html"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        plotly_renderer.render_face_mesh_html(make_mesh_track(), output_path)
