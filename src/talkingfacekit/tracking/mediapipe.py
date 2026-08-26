"""MediaPipe Face Landmarker integration."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.tracking.landmarks import FaceLandmarkTrack
from talkingfacekit.video import DecodedVideoFrame

_LANDMARK_COUNT = 478
_TOPOLOGY = "mediapipe-face-landmarker-478"
_COORDINATE_SYSTEM = (
    "MediaPipe normalized image coordinates: x increases left-to-right, y increases top-to-bottom, "
    "and z is relative depth on approximately the same scale as x"
)


class _Landmark(Protocol):
    x: float
    y: float
    z: float


class _LandmarkerResult(Protocol):
    face_landmarks: list[list[_Landmark]]


class _Landmarker(Protocol):
    def detect_for_video(self, image: object, timestamp_ms: int) -> _LandmarkerResult: ...

    def close(self) -> None: ...


class _Detector(Protocol):
    tracker_version: str | None

    def detect(
        self, rgb_frame: NDArray[np.uint8], timestamp_ms: int
    ) -> NDArray[np.float32] | None: ...


class _MediaPipeDetector:
    """Keep the untyped optional dependency behind a small typed boundary."""

    def __init__(self, tracker: MediaPipeFaceTracker) -> None:
        try:
            # MediaPipe 1.0.0 does not publish a py.typed marker or stub files.
            mediapipe = importlib.import_module("mediapipe")
        except ImportError as error:
            raise ImportError(
                "MediaPipe tracking is optional. Install it with "
                "`uv sync --extra tracking-mediapipe`."
            ) from error

        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(tracker.model_path)),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=tracker.min_face_detection_confidence,
            min_face_presence_confidence=tracker.min_face_presence_confidence,
            min_tracking_confidence=tracker.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = cast(
            _Landmarker,
            mediapipe.tasks.vision.FaceLandmarker.create_from_options(options),
        )
        self._image_class = mediapipe.Image
        self._image_format = mediapipe.ImageFormat.SRGB
        self.tracker_version = cast(str | None, getattr(mediapipe, "__version__", None))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._landmarker.close()

    def detect(self, rgb_frame: NDArray[np.uint8], timestamp_ms: int) -> NDArray[np.float32] | None:
        """Convert one RGB frame and return its first detected face."""
        image = self._image_class(image_format=self._image_format, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        if len(face) != _LANDMARK_COUNT:
            raise ValueError(
                f"MediaPipe returned {len(face)} landmarks; expected {_LANDMARK_COUNT}"
            )
        return np.asarray([(point.x, point.y, point.z) for point in face], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MediaPipeFaceTracker:
    """Track one face per video frame with MediaPipe Face Landmarker.

    The tracker consumes the shared RGB frame stream, sends each frame to MediaPipe, and retains
    only the resulting landmarks. Video opening, PyAV decoding, interval selection, and RGB
    conversion belong to the shared video integration. The model asset is supplied by the caller
    and is neither downloaded nor bundled by TalkingFaceKit.

    Parameters
    ----------
    model_path
        Path to a MediaPipe Face Landmarker ``.task`` model asset.
    min_face_detection_confidence
        Minimum initial face-detection confidence in the inclusive range ``[0, 1]``.
    min_face_presence_confidence
        Minimum face-presence confidence in the inclusive range ``[0, 1]``.
    min_tracking_confidence
        Minimum temporal tracking confidence in the inclusive range ``[0, 1]``.
    """

    model_path: Path
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        """Normalize and validate model configuration without loading MediaPipe."""
        object.__setattr__(self, "model_path", Path(self.model_path))
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")
        if self.model_path.is_dir():
            raise IsADirectoryError(f"MediaPipe model path is a directory: {self.model_path}")
        if not self.model_path.is_file():
            raise ValueError(f"MediaPipe model path is not a regular file: {self.model_path}")

        for field_name, value in (
            ("min_face_detection_confidence", self.min_face_detection_confidence),
            ("min_face_presence_confidence", self.min_face_presence_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1], got {value}")

    def track(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        end_seconds: float | None,
    ) -> FaceLandmarkTrack:
        """Track facial landmarks over a half-open interval of a local video.

        Parameters
        ----------
        video_path
            Local media file whose first video stream will be decoded.
        start_seconds
            Inclusive interval start in seconds. Must be finite and non-negative.
        end_seconds
            Exclusive interval end in seconds, or ``None`` to continue to end-of-stream.

        Returns
        -------
        FaceLandmarkTrack
            Frame-aligned MediaPipe landmarks. Undetected frames contain ``NaN`` coordinates.

        Raises
        ------
        FileNotFoundError
            If the video or model file does not exist.
        IsADirectoryError
            If the video or model path refers to a directory.
        ImportError
            If the optional MediaPipe dependency is not installed.
        ValueError
            If the interval, media, timestamps, decoded RGB frames, or tracker output violates its
            declared contract.
        """
        frames = stream_video_frames(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        with _MediaPipeDetector(self) as detector:
            return _track_frames(frames, detector)


def _track_frames(
    frames: Iterable[DecodedVideoFrame],
    detector: _Detector,
) -> FaceLandmarkTrack:
    frame_indices: list[int] = []
    timestamps_seconds: list[float] = []
    landmark_rows: list[NDArray[np.float32]] = []
    detection_values: list[bool] = []
    previous_timestamp_ms: int | None = None

    for frame in frames:
        timestamp_ms = round(frame.timestamp_seconds * 1_000.0)
        if previous_timestamp_ms is not None and timestamp_ms <= previous_timestamp_ms:
            raise ValueError(
                "MediaPipe requires strictly increasing millisecond timestamps; "
                f"frame {frame.frame_index} produced {timestamp_ms} after {previous_timestamp_ms}"
            )
        previous_timestamp_ms = timestamp_ms

        detected_landmarks = detector.detect(frame.rgb, timestamp_ms)
        if detected_landmarks is None:
            landmark_row = np.full((_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
            detected = False
        else:
            if detected_landmarks.dtype != np.dtype(np.float32):
                raise TypeError(
                    f"MediaPipe landmarks must have dtype float32, got {detected_landmarks.dtype}"
                )
            if detected_landmarks.shape != (_LANDMARK_COUNT, 3):
                raise ValueError(
                    "MediaPipe landmarks must have shape "
                    f"({_LANDMARK_COUNT}, 3), got {detected_landmarks.shape}"
                )
            landmark_row = detected_landmarks
            detected = True

        frame_indices.append(frame.frame_index)
        timestamps_seconds.append(frame.timestamp_seconds)
        landmark_rows.append(landmark_row)
        detection_values.append(detected)

    if not frame_indices:
        raise ValueError("Landmark tracking requires at least one decoded video frame")

    return FaceLandmarkTrack(
        tracker_name="mediapipe-face-landmarker",
        tracker_version=detector.tracker_version,
        topology=_TOPOLOGY,
        coordinate_system=_COORDINATE_SYSTEM,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        timestamps_seconds=np.asarray(timestamps_seconds, dtype=np.float64),
        landmarks=np.stack(landmark_rows).astype(np.float32, copy=False),
        detected=np.asarray(detection_values, dtype=np.bool_),
    )
