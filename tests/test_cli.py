from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from talkingfacekit import (
    FaceLandmarkTrack,
    FaceMeshTrack,
    cli,
    load_landmark_track,
    save_landmark_track,
)

FIXTURE = Path(__file__).parent / "fixtures" / "example1.webm"


def make_landmark_track() -> FaceLandmarkTrack:
    return FaceLandmarkTrack(
        tracker_name="fake-mediapipe",
        tracker_version="test",
        topology="test-2-points",
        coordinate_system="synthetic test coordinates",
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.042], dtype=np.float64),
        landmarks=np.zeros((2, 2, 3), dtype=np.float32),
        detected=np.asarray([True, True], dtype=np.bool_),
    )


def make_mesh_track() -> FaceMeshTrack:
    return FaceMeshTrack(
        topology="test-triangle",
        coordinate_system="test coordinates",
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.042], dtype=np.float64),
        vertices=np.zeros((2, 3, 3), dtype=np.float32),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        detected=np.asarray([True, True], dtype=np.bool_),
    )


class FakeMediaPipeFaceTracker:
    result: ClassVar[FaceLandmarkTrack]

    def __init__(self, model_path: Path) -> None:
        assert model_path == Path("model.task")

    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        assert video_path == FIXTURE
        assert start_seconds == 0.0
        assert end_seconds is not None
        return self.result


def test_extracts_the_complete_track_to_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeMediaPipeFaceTracker.result = make_landmark_track()
    monkeypatch.setattr(cli, "MediaPipeFaceTracker", FakeMediaPipeFaceTracker)
    output_path = tmp_path / "video_landmarks.npz"

    exit_code = cli.main(
        [
            "extract-landmarks",
            str(FIXTURE),
            "--model",
            "model.task",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert load_landmark_track(output_path).landmarks.shape == (2, 2, 3)
    assert "detected frames: 2" in capsys.readouterr().out


def test_inspects_an_existing_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_path = tmp_path / "video_landmarks.npz"
    save_landmark_track(make_landmark_track(), archive_path)

    exit_code = cli.main(["inspect-landmarks", str(archive_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "tracker: fake-mediapipe" in output
    assert "landmark array shape: (2, 2, 3)" in output


def test_builds_and_renders_mesh_from_landmark_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = tmp_path / "video_landmarks.npz"
    output_path = tmp_path / "face_mesh.html"
    save_landmark_track(make_landmark_track(), archive_path)
    expected_mesh = make_mesh_track()

    def fake_build_mesh(
        track: FaceLandmarkTrack,
        *,
        image_width: int,
        image_height: int,
    ) -> FaceMeshTrack:
        assert track.landmark_count == 2
        assert image_width == 1920
        assert image_height == 1080
        return expected_mesh

    def fake_render_mesh(
        mesh: FaceMeshTrack,
        destination: str | Path,
        *,
        title: str,
        depth_scale: float,
        overwrite: bool,
    ) -> Path:
        assert mesh is expected_mesh
        assert title == "Course demo"
        assert depth_scale == 1.5
        assert overwrite is False
        path = Path(destination)
        path.write_text("rendered", encoding="utf-8")
        return path

    monkeypatch.setattr(cli, "build_mediapipe_face_mesh", fake_build_mesh)
    monkeypatch.setattr(cli, "render_face_mesh_html", fake_render_mesh)

    exit_code = cli.main(
        [
            "render-mesh",
            str(archive_path),
            "--video",
            str(FIXTURE),
            "--output",
            str(output_path),
            "--title",
            "Course demo",
            "--depth-scale",
            "1.5",
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "rendered"
    output = capsys.readouterr().out
    assert "vertices per frame: 3" in output
    assert "triangles per frame: 1" in output
