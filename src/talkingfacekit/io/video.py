"""Video file inspection through the PyAV integration boundary."""

from pathlib import Path

import av

from talkingfacekit.metadata import VideoMetadata


def inspect_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Inspect a local media file and return metadata for its primary video stream.

    The first video stream supplies the dimensions and average frame rate. Its duration is used
    when available; otherwise, the container duration is used as a fallback. No frames or audio
    samples are decoded.

    Parameters
    ----------
    video_path
        Path to a local media file.

    Returns
    -------
    VideoMetadata
        Backend-independent metadata for the first video stream.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    IsADirectoryError
        If ``video_path`` refers to a directory.
    ValueError
        If the path is not a regular file, the media cannot be inspected, or it has no video
        stream.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Video path is a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a regular file: {path}")

    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"Media file has no video stream: {path}")

            video_stream = container.streams.video[0]
            duration_seconds = (
                float(video_stream.duration * video_stream.time_base)
                if video_stream.duration is not None and video_stream.time_base is not None
                else (
                    float(container.duration / av.time_base)
                    if container.duration is not None
                    else None
                )
            )
            return VideoMetadata(
                width=video_stream.width,
                height=video_stream.height,
                average_fps=(
                    float(video_stream.average_rate)
                    if video_stream.average_rate is not None
                    else None
                ),
                stream_duration_seconds=duration_seconds,
                has_audio=bool(container.streams.audio),
            )
    except av.FFmpegError as error:
        raise ValueError(f"Could not inspect media file {path}: {error}") from error
