"""Command-line workflows for TalkingFaceKit."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from talkingfacekit.io.landmarks import load_landmark_track, save_landmark_track
from talkingfacekit.io.video import inspect_video_metadata
from talkingfacekit.mesh import FaceMeshTrack
from talkingfacekit.rendering import render_face_mesh_html
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack
from talkingfacekit.tracking.mediapipe import MediaPipeFaceTracker
from talkingfacekit.tracking.mediapipe_mesh import build_mediapipe_face_mesh


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TalkingFaceKit command-line interface.

    Parameters
    ----------
    argv
        Arguments excluding the executable name, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code. Successful commands return zero; invalid input exits through argparse.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    command = cast(str, arguments.command)

    try:
        if command == "extract-landmarks":
            return _extract_landmarks(arguments)
        if command == "inspect-landmarks":
            return _inspect_landmarks(arguments)
        if command == "render-mesh":
            return _render_mesh(arguments)
    except (
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        ImportError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")

    raise RuntimeError(f"Unsupported command: {command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m talkingfacekit",
        description="Process talking-face videos with explicit, timestamped data contracts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract-landmarks",
        help="Track one face through a video interval and save every selected frame to NPZ.",
    )
    extract_parser.add_argument("video", type=Path, help="Local video file to process.")
    extract_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="MediaPipe Face Landmarker .task model.",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npz archive.",
    )
    extract_parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Inclusive interval start on the source timeline; default: 0.0.",
    )
    extract_parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Exclusive interval end on the source timeline; default: source end.",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after extraction completes.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-landmarks",
        help="Validate an existing landmark NPZ archive and print its summary.",
    )
    inspect_parser.add_argument("archive", type=Path, help="Landmark .npz archive to inspect.")

    render_parser = subparsers.add_parser(
        "render-mesh",
        help="Build and render an animated MediaPipe face mesh to offline HTML.",
    )
    render_parser.add_argument("archive", type=Path, help="MediaPipe landmark .npz archive.")
    render_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Source video used only to obtain width and height for aspect correction.",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination self-contained .html file.",
    )
    render_parser.add_argument(
        "--title",
        default="TalkingFaceKit animated face mesh",
        help="Heading displayed above the interactive mesh.",
    )
    render_parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Positive display-only multiplier for relative depth; default: 1.0.",
    )
    render_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after rendering completes.",
    )
    return parser


def _extract_landmarks(arguments: argparse.Namespace) -> int:
    video_path = cast(Path, arguments.video)
    model_path = cast(Path, arguments.model)
    output_path = cast(Path, arguments.output)
    start_seconds = cast(float, arguments.start_seconds)
    end_seconds = cast(float | None, arguments.end_seconds)
    overwrite = cast(bool, arguments.overwrite)

    sequence = TalkingFaceSequence.from_video(video_path)
    sequence = sequence.clip(start_seconds, end_seconds)
    tracker = MediaPipeFaceTracker(model_path)
    track = sequence.track_landmarks(tracker, name="mediapipe")
    saved_path = save_landmark_track(track, output_path, overwrite=overwrite)

    _print_summary(track)
    print(f"output: {saved_path}")
    return 0


def _inspect_landmarks(arguments: argparse.Namespace) -> int:
    archive_path = cast(Path, arguments.archive)
    track = load_landmark_track(archive_path)

    _print_summary(track)
    print(f"archive: {archive_path}")
    return 0


def _render_mesh(arguments: argparse.Namespace) -> int:
    archive_path = cast(Path, arguments.archive)
    video_path = cast(Path, arguments.video)
    output_path = cast(Path, arguments.output)
    title = cast(str, arguments.title)
    depth_scale = cast(float, arguments.depth_scale)
    overwrite = cast(bool, arguments.overwrite)

    track = load_landmark_track(archive_path)
    metadata = inspect_video_metadata(video_path)
    mesh = build_mediapipe_face_mesh(
        track,
        image_width=metadata.width,
        image_height=metadata.height,
    )
    saved_path = render_face_mesh_html(
        mesh,
        output_path,
        title=title,
        depth_scale=depth_scale,
        overwrite=overwrite,
    )

    _print_mesh_summary(mesh)
    print(f"output: {saved_path}")
    return 0


def _print_summary(track: FaceLandmarkTrack) -> None:
    detected_count = int(np.count_nonzero(track.detected))
    missing_count = track.frame_count - detected_count
    first_timestamp = float(track.timestamps_seconds[0])
    last_timestamp = float(track.timestamps_seconds[-1])

    print(f"tracker: {track.tracker_name}")
    print(f"topology: {track.topology}")
    print(f"frames: {track.frame_count}")
    print(f"landmarks per frame: {track.landmark_count}")
    print(f"detected frames: {detected_count}")
    print(f"missing frames: {missing_count}")
    print(f"timestamp range: [{first_timestamp:.6f}, {last_timestamp:.6f}] seconds")
    print(f"landmark array shape: {track.landmarks.shape}")


def _print_mesh_summary(mesh: FaceMeshTrack) -> None:
    detected_count = int(np.count_nonzero(mesh.detected))
    missing_count = mesh.frame_count - detected_count

    print(f"topology: {mesh.topology}")
    print(f"frames: {mesh.frame_count}")
    print(f"vertices per frame: {mesh.vertex_count}")
    print(f"triangles per frame: {mesh.triangle_count}")
    print(f"detected frames: {detected_count}")
    print(f"missing frames: {missing_count}")
    print(f"vertex array shape: {mesh.vertices.shape}")
