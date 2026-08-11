from dataclasses import dataclass
from pathlib import Path
import av


@dataclass
class VideoMetadata:
    width: int
    height: int
    average_fps: float | None
    source_duration_seconds: float
    has_audio: bool


class TalkingFaceSequence:
    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.metadata = self._inspect_video()
        self.start_seconds = 0.0
        self.end_seconds = self.metadata.source_duration_seconds

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
                source_duration_seconds=10,
                has_audio=bool(container.streams.audio),
            )
        return metadata
        # Abrir el video, leer duracion, FPS, resolucion, audio
