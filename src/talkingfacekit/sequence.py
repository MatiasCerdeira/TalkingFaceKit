from dataclasses import dataclass
from pathlib import Path

import av


@dataclass
class VideoMetadata:
    width: int
    height: int
    average_fps: float | None
    stream_duration_seconds: float | None
    has_audio: bool


class TalkingFaceSequence:
    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.metadata: VideoMetadata = self._inspect_video()
        self.start_seconds: float = 0.0
        self.end_seconds: float | None = self.metadata.stream_duration_seconds

    # PUBLIC

    # PRIVATE
    def _inspect_video(self) -> VideoMetadata:
        with av.open(self.path) as container:
            video_stream = container.streams.video[0]

            metadata = VideoMetadata(
                width=video_stream.width,
                height=video_stream.height,
                average_fps=(
                    float(video_stream.average_rate)
                    if video_stream.average_rate is not None
                    else None
                ),
                stream_duration_seconds=(
                    float(video_stream.duration * video_stream.time_base)
                    if video_stream.duration is not None
                    and video_stream.time_base is not None
                    else None
                ),
                has_audio=bool(container.streams.audio),
            )
            print("Duracion: ", metadata.stream_duration_seconds)
        return metadata
