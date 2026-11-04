"""Experimental streaming facial-parameter estimation following official SMIRK.

The MediaPipe options and crop follow georgeretsi/smirk revision
c7de404c4389f073906a6db1adabf62efcea3f35. See SMIRK_LICENSE.txt. This module
keeps preprocessing, model loading and framework objects inside the optional integration.
No model weights are downloaded, and FLAME geometry is not evaluated here.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from hashlib import file_digest
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.flame import FlameParameters, FlameResult
from talkingfacekit.integrations._smirk_model import build_smirk_model
from talkingfacekit.io.provenance import code_revision
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.video import DecodedVideoFrame, VideoSource

_IMAGE_SIZE = 224
_LANDMARK_COUNT = 478
_CROP_SCALE = 1.4
SMIRK_UPSTREAM_REVISION = "c7de404c4389f073906a6db1adabf62efcea3f35"


@dataclass(frozen=True, slots=True)
class SmirkMissingFrame:
    """Record a source frame for which no encoder input could be prepared.

    Attributes
    ----------
    frame_index, timestamp_seconds
        Zero-based source decode index and original presentation timestamp in seconds.
    reason
        Stable code ``face_landmarks_unavailable``: MediaPipe returned no face landmarks.
        It does not identify which internal detection/presence stage failed or establish
        that the source image is unsuitable for every facial estimator.

    Notes
    -----
    This outcome contains no pixels, crop, or invented facial parameters. It is emitted
    by both input preparation and tracking, preserving the source timeline through gaps.
    """

    frame_index: int
    timestamp_seconds: float
    reason: Literal["face_landmarks_unavailable"] = "face_landmarks_unavailable"

    def __post_init__(self) -> None:
        """Validate source identity and the supported absence reason."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if isinstance(self.timestamp_seconds, bool) or not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        if self.reason != "face_landmarks_unavailable":
            raise ValueError("unsupported SMIRK missing-frame reason")


@dataclass(frozen=True, slots=True, eq=False)
class SmirkInputFrame:
    """One prepared encoder input retaining its source-frame identity.

    Attributes
    ----------
    frame_index, timestamp_seconds
        Zero-based source decode index and original presentation timestamp in seconds.
    image
        C-contiguous RGB ``float32`` array shaped ``(1, 3, 224, 224)`` in ``[0, 1]``.
        It can be converted to a framework tensor at the encoder boundary.
    source_to_crop
        Finite ``float64`` homogeneous ``(3, 3)`` matrix mapping source pixel coordinates
        to crop pixel coordinates. It does not represent a FLAME camera.

    Notes
    -----
    Arrays are owned by this result and must be treated as read-only. Retaining outputs is
    the caller's choice; the iterator does not retain previous images.
    """

    frame_index: int
    timestamp_seconds: float
    image: NDArray[np.float32]
    source_to_crop: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate source identity, pixel layout, range and crop transformation."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if isinstance(self.timestamp_seconds, bool) or not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        if self.image.dtype != np.dtype(np.float32):
            raise TypeError("SMIRK image must have dtype float32")
        if self.image.shape != (1, 3, _IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError("SMIRK image must have shape (1, 3, 224, 224)")
        if not self.image.flags.c_contiguous:
            raise ValueError("SMIRK image must be C-contiguous")
        if not np.all(np.isfinite(self.image)) or np.any((self.image < 0) | (self.image > 1)):
            raise ValueError("SMIRK image must contain finite values in [0, 1]")
        if self.source_to_crop.dtype != np.dtype(np.float64):
            raise TypeError("source_to_crop must have dtype float64")
        if self.source_to_crop.shape != (3, 3) or not np.all(np.isfinite(self.source_to_crop)):
            raise ValueError("source_to_crop must be a finite (3, 3) matrix")
        if not np.allclose(self.source_to_crop[2], [0, 0, 1]):
            raise ValueError("source_to_crop must be a homogeneous affine transformation")
        if abs(float(np.linalg.det(self.source_to_crop[:2, :2]))) <= np.finfo(float).eps:
            raise ValueError("source_to_crop must be invertible")


@dataclass(frozen=True, slots=True, eq=False)
class SmirkParameterFrame:
    """One raw facial prediction with source identity and SMIRK-specific evidence.

    Attributes
    ----------
    frame_index, timestamp_seconds
        Original decode index and presentation timestamp in seconds.
    parameters
        Framework-independent FLAME2020 shape, expression and axis-angle poses.
    eyelids
        Finite float32 vector ``(2,)`` in ``[0, 1]`` for SMIRK's extra eyelid deformations,
        ordered left/right. These are not eyeball rotations or standard FLAME expressions.
    camera
        Finite float32 vector ``(3,)`` containing SMIRK's orthographic ``[scale, tx, ty]``
        camera for the crop. It is not a metric translation in the source video.
    source_to_crop
        Float64 ``(3, 3)`` mapping from source pixels to the prepared input crop.
    checkpoint_sha256
        Digest of the exact local encoder checkpoint used for this prediction.

    Notes
    -----
    Only compact NumPy outputs are retained, without input pixels or tensors. Arrays must
    be treated as read-only. No temporal smoothing, identity aggregation or quality guarantee
    is applied to the original per-image estimates.
    """

    frame_index: int
    timestamp_seconds: float
    parameters: FlameParameters
    eyelids: NDArray[np.float32]
    camera: NDArray[np.float32]
    source_to_crop: NDArray[np.float64]
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        """Validate prediction identity, available evidence and checkpoint provenance."""
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0 or isinstance(self.timestamp_seconds, bool):
            raise ValueError("invalid source-frame identity")
        if not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        if not isinstance(self.parameters, FlameParameters):
            raise TypeError("parameters must be FlameParameters")
        for name, size in (("eyelids", 2), ("camera", 3)):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float32):
                raise TypeError(f"{name} must be a float32 NumPy array")
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite vector of shape ({size},)")
        if np.any((self.eyelids < 0) | (self.eyelids > 1)):
            raise ValueError("eyelids must be in [0, 1]")
        if self.source_to_crop.dtype != np.dtype(np.float64):
            raise TypeError("source_to_crop must have dtype float64")
        if self.source_to_crop.shape != (3, 3) or not np.all(np.isfinite(self.source_to_crop)):
            raise ValueError("source_to_crop must be a finite (3, 3) matrix")
        if not np.allclose(self.source_to_crop[2], [0, 0, 1]):
            raise ValueError("source_to_crop must be a homogeneous affine transformation")
        if abs(float(np.linalg.det(self.source_to_crop[:2, :2]))) <= np.finfo(float).eps:
            raise ValueError("source_to_crop must be invertible")
        if len(self.checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")


class SmirkEncoder:
    """Load a local SMIRK checkpoint once and estimate facial parameters on demand.

    Parameters
    ----------
    checkpoint
        Local official SMIRK_em1 checkpoint or a complete encoder-only state dictionary.
        Loaded on CPU with ``weights_only=True`` and checked strictly before inference.
    device
        Explicit execution device. CPU is the initial reference; other devices must be
        supported by the installed runtime and are never selected as silent fallbacks.

    Notes
    -----
    Construction loads the model but never accesses the network. Backbones are initialized
    with ``pretrained=False`` because the SMIRK checkpoint must supply every encoder weight.
    Raw NumPy parameters are returned without requiring PyTorch3D or a FLAME model asset.
    The session is sequential and should not be shared across concurrent calls.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
    ) -> None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SMIRK checkpoint is not a regular file: {path}")
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be 'cpu', 'mps' or 'cuda'")
        with path.open("rb") as stream:
            self.checkpoint_sha256 = file_digest(stream, "sha256").hexdigest()
        self.device = device
        try:
            self._torch = cast(Any, import_module("torch"))
            timm = import_module("timm")
        except ImportError as error:
            raise ImportError("SMIRK requires the 'tracking-smirk' optional extra") from error
        model = cast(Any, build_smirk_model(self._torch, timm))
        checkpoint_data = self._torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_data, dict) or not all(
            isinstance(key, str) for key in checkpoint_data
        ):
            raise ValueError("SMIRK checkpoint must contain a string-keyed state dictionary")
        prefix = "smirk_encoder."
        weights = {
            key.removeprefix(prefix): value
            for key, value in checkpoint_data.items()
            if key.startswith(prefix)
        }
        if not weights:
            weights = checkpoint_data
        expected_keys = set(model.state_dict())
        if set(weights) != expected_keys or not all(
            self._torch.is_tensor(value) for value in weights.values()
        ):
            raise ValueError("SMIRK checkpoint does not contain the complete encoder state")
        model.load_state_dict(weights, strict=True)
        self._model = model.float().to(device).eval()

    def predict(self, frame: SmirkInputFrame) -> SmirkParameterFrame:
        """Estimate one prepared image and copy finite outputs to owned CPU NumPy arrays.

        Parameters
        ----------
        frame
            Validated RGB crop with original source identity and transformation.

        Returns
        -------
        SmirkParameterFrame
            300 shape and 50 expression coefficients, head/jaw pose, eyelids and crop camera.
            Source PTS are unchanged; model/runtime errors propagate without partial output.
        """
        if not isinstance(frame, SmirkInputFrame):
            raise TypeError("frame must be a SmirkInputFrame")
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(frame.image).to(self.device)
            outputs = self._model(tensor)
        if not isinstance(outputs, dict):
            raise TypeError("SMIRK encoder must return a parameter dictionary")

        def vector(name: str, size: int) -> NDArray[np.float32]:
            value = cast(Any, outputs.get(name))
            if not self._torch.is_tensor(value) or tuple(value.shape) != (1, size):
                raise ValueError(f"SMIRK {name} must have shape (1, {size})")
            return cast(
                NDArray[np.float32],
                value.detach().to(device="cpu", dtype=self._torch.float32).numpy()[0].copy(),
            )

        return SmirkParameterFrame(
            frame.frame_index,
            frame.timestamp_seconds,
            FlameParameters(
                "flame-2020-generic",
                vector("shape_params", 300),
                vector("expression_params", 50),
                vector("pose_params", 3),
                vector("jaw_params", 3),
            ),
            vector("eyelid_params", 2),
            vector("cam", 3),
            frame.source_to_crop.copy(),
            self.checkpoint_sha256,
        )

    def track(
        self, sequence: TalkingFaceSequence, *, landmarker_model: str | Path
    ) -> Generator[SmirkParameterFrame | SmirkMissingFrame, None, None]:
        """Stream facial predictions using this session's already-loaded encoder.

        Parameters
        ----------
        sequence
            Source and interval to process without FPS conversion or active-speaker analysis.
        landmarker_model
            Explicit local Face Landmarker task used by the upstream-compatible crop.

        Yields
        ------
        SmirkParameterFrame or SmirkMissingFrame
            One outcome per source frame, preserving index and PTS. Missing landmarks yield
            an explicit absence without executing the encoder; processing resumes at the next
            frame. No interpolation, segment splitting or quality selection is performed.

        Notes
        -----
        Configuration, decoding, crop validation and encoder errors propagate. Only an empty
        MediaPipe face-landmark result is a recoverable absence. Close the iterator when
        stopping early to release preparation resources.
        """
        with closing(iter_smirk_inputs(sequence, landmarker_model=landmarker_model)) as inputs:
            for frame in inputs:
                yield frame if isinstance(frame, SmirkMissingFrame) else self.predict(frame)

    def extract(
        self, sequence: TalkingFaceSequence, *, landmarker_model: str | Path
    ) -> FlameResult:
        """Collect a complete interval into a backend-independent, saveable result.

        Parameters
        ----------
        sequence
            Source interval to decode at native timestamps. Unlike ``track``, this method
            materializes compact output arrays for the complete interval, never video pixels.
        landmarker_model
            Local Face Landmarker asset. Its hash and runtime settings are recorded.

        Returns
        -------
        FlameResult
            Source metadata/hash, raw FLAME arrays, estimation mask, reasons and provenance.
            SMIRK camera, eyelids and source-to-crop transforms remain named extras with
            explicit conventions. Absent rows contain only NaN, with their reason preserved.

        Notes
        -----
        The source video is read once for SHA-256, in addition to streaming decoding. Hashing
        happens during extraction, never during save/load. Failures propagate without returning
        a partial result. This estimates one face per image; no target identity is enforced.
        """
        if not isinstance(sequence, TalkingFaceSequence):
            raise TypeError("sequence must be a TalkingFaceSequence")
        source_path = sequence.source.path.resolve()
        landmarker_path = Path(landmarker_model).resolve()
        with source_path.open("rb") as source_stream:
            source_sha256 = file_digest(source_stream, "sha256").hexdigest()
        with landmarker_path.open("rb") as model_stream:
            landmarker_sha256 = file_digest(model_stream, "sha256").hexdigest()
        revision = code_revision()
        provenance = {
            "backend": "smirk",
            "checkpoint_sha256": self.checkpoint_sha256,
            "landmarker_sha256": landmarker_sha256,
            "upstream_revision": SMIRK_UPSTREAM_REVISION,
            "device": self.device,
            "preprocessing": (
                "IMAGE;num_faces=1;detection=0.1;presence=0.1;blendshapes=true;"
                "transforms=true;crop_scale=1.4;size=224;RGB;float32;range=0..1"
            ),
            "code_commit": str(revision["commit"] or "unavailable"),
            "code_dirty": "unavailable"
            if revision["dirty"] is None
            else str(revision["dirty"]).lower(),
            **{
                f"version_{package}": version(package)
                for package in (
                    "talkingfacekit",
                    "numpy",
                    "torch",
                    "torchvision",
                    "timm",
                    "mediapipe",
                    "scikit-image",
                )
            },
        }
        with closing(self.track(sequence, landmarker_model=landmarker_path)) as predictions:
            rows = list(predictions)
        if not rows:
            raise ValueError("SMIRK extraction requires at least one video frame")
        count = len(rows)
        shape = np.full((count, 300), np.nan, dtype=np.float32)
        expression = np.full((count, 50), np.nan, dtype=np.float32)
        global_pose = np.full((count, 3), np.nan, dtype=np.float32)
        jaw_pose = np.full((count, 3), np.nan, dtype=np.float32)
        eyelids = np.full((count, 2), np.nan, dtype=np.float32)
        camera = np.full((count, 3), np.nan, dtype=np.float32)
        transforms = np.full((count, 3, 3), np.nan, dtype=np.float64)
        estimated = np.zeros(count, dtype=np.bool_)
        reasons: list[str] = []
        for index, row in enumerate(rows):
            if isinstance(row, SmirkMissingFrame):
                reasons.append(row.reason)
                continue
            if (
                row.checkpoint_sha256 != self.checkpoint_sha256
                or row.parameters.model_variant != "flame-2020-generic"
            ):
                raise ValueError("SMIRK prediction provenance does not match this encoder")
            estimated[index] = True
            reasons.append("")
            shape[index] = row.parameters.shape
            expression[index] = row.parameters.expression
            global_pose[index] = row.parameters.global_pose
            jaw_pose[index] = row.parameters.jaw_pose
            eyelids[index] = row.eyelids
            camera[index] = row.camera
            transforms[index] = row.source_to_crop
        return FlameResult(
            source=VideoSource(source_path, sequence.source.metadata),
            source_sha256=source_sha256,
            start_seconds=sequence.start_seconds,
            end_seconds=sequence.end_seconds,
            model_variant="flame-2020-generic",
            frame_indices=np.array([row.frame_index for row in rows], dtype=np.int64),
            timestamps_seconds=np.array([row.timestamp_seconds for row in rows], dtype=np.float64),
            estimated=estimated,
            missing_reasons=tuple(reasons),
            shape=shape,
            expression=expression,
            global_pose=global_pose,
            jaw_pose=jaw_pose,
            extras={
                "smirk_eyelids": eyelids,
                "smirk_camera": camera,
                "smirk_source_to_crop": transforms,
            },
            extra_conventions={
                "smirk_eyelids": (
                    "(N,2), float32, left/right eyelid deformation coefficients in [0,1]; "
                    "not eyeball rotations"
                ),
                "smirk_camera": (
                    "(N,3), float32, orthographic [scale,tx,ty] in input-crop coordinates; "
                    "not metric source translation"
                ),
                "smirk_source_to_crop": (
                    "(N,3,3), float64, homogeneous invertible affine matrix mapping "
                    "source pixels to 224x224 crop pixels"
                ),
            },
            provenance=provenance,
        )


def iter_smirk_inputs(
    sequence: TalkingFaceSequence,
    *,
    landmarker_model: str | Path,
) -> Generator[SmirkInputFrame | SmirkMissingFrame, None, None]:
    """Prepare source frames lazily using SMIRK's image-mode detection and crop.

    Parameters
    ----------
    sequence
        Source and half-open interval. Frames retain their native PTS and decode indices;
        there is no FPS resampling or dependency on active-speaker analysis.
    landmarker_model
        Explicit local MediaPipe Face Landmarker ``.task`` asset. It is not downloaded.

    Yields
    ------
    SmirkInputFrame or SmirkMissingFrame
        Normalized encoder input and crop transform, or an explicit absence when MediaPipe
        returns no face landmarks. Both outcomes retain source index and PTS; no frame is
        silently dropped, interpolated or replaced with the preceding crop.

    Raises
    ------
    ImportError
        If MediaPipe or scikit-image is unavailable when iteration starts.
    FileNotFoundError
        If a source or model asset is missing.
    ValueError
        If the interval has no video frames or a detected face defines an invalid crop.
        Other decoder, detector and dependency errors propagate; they are not missing faces.

    Notes
    -----
    The official demo starts from OpenCV BGR images. Here channel-independent warping is
    applied to the decoder's RGB images directly, preserving the encoder's RGB channel order.
    MediaPipe is used with IMAGE mode, one face, detection/presence thresholds of 0.1, and
    blendshape/transformation outputs enabled, matching the upstream helper configuration.
    Unlike the official cropped demo, a missing detection is recorded and processing continues;
    the detection, crop and encoder algorithms themselves are unchanged.
    """
    if not isinstance(sequence, TalkingFaceSequence):
        raise TypeError("sequence must be a TalkingFaceSequence")
    model_path = Path(landmarker_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model is not a regular file: {model_path}")
    frames = stream_video_frames(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        try:
            frame = next(frames)
        except StopIteration:
            raise ValueError("SMIRK input preparation requires at least one video frame") from None
        try:
            # scikit-image's transform objects do not expose typing metadata.
            transforms = cast(Any, import_module("skimage.transform"))
        except ImportError as error:
            raise ImportError("SMIRK input preparation requires scikit-image") from error

        with _SmirkLandmarker(model_path) as detector:
            while True:
                landmarks = detector.detect(frame)
                if landmarks is None:
                    yield SmirkMissingFrame(frame.frame_index, frame.timestamp_seconds)
                else:
                    source_points = _crop_source_points(landmarks)
                    destination_points = np.array(
                        [[0, 0], [0, _IMAGE_SIZE - 1], [_IMAGE_SIZE - 1, 0]], dtype=np.float64
                    )
                    transform = transforms.estimate_transform(
                        "similarity", source_points, destination_points
                    )
                    crop = np.asarray(
                        transforms.warp(
                            frame.rgb,
                            transform.inverse,
                            output_shape=(_IMAGE_SIZE, _IMAGE_SIZE),
                            preserve_range=True,
                        ),
                        dtype=np.uint8,
                    )
                    yield SmirkInputFrame(
                        frame.frame_index,
                        frame.timestamp_seconds,
                        _normalize_crop(crop),
                        np.array(transform.params, dtype=np.float64, copy=True),
                    )
                next_frame = next(frames, None)
                if next_frame is None:
                    break
                frame = next_frame
    finally:
        close = getattr(frames, "close", None)
        if callable(close):
            close()


def _crop_source_points(landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Construct the official square crop controls from source-pixel landmarks."""
    if landmarks.shape != (_LANDMARK_COUNT, 2) or not np.all(np.isfinite(landmarks)):
        raise ValueError("SMIRK requires 478 finite source-pixel x/y landmarks")
    minimum = landmarks.min(axis=0)
    maximum = landmarks.max(axis=0)
    center = maximum - (maximum - minimum) / 2
    size = int(float(np.mean(maximum - minimum)) * _CROP_SCALE)
    if size <= 0:
        raise ValueError("SMIRK landmarks define an empty face crop")
    half_size = size / 2
    return np.array(
        [
            [center[0] - half_size, center[1] - half_size],
            [center[0] - half_size, center[1] + half_size],
            [center[0] + half_size, center[1] - half_size],
        ],
        dtype=np.float64,
    )


def _normalize_crop(crop: NDArray[np.uint8]) -> NDArray[np.float32]:
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)


class _SmirkLandmarker:
    def __init__(self, model_path: Path) -> None:
        try:
            # Keep all untyped MediaPipe objects inside this integration boundary.
            self._module = cast(Any, import_module("mediapipe"))
        except ImportError as error:
            raise ImportError("SMIRK input preparation requires MediaPipe") from error
        vision = self._module.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=self._module.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._landmarker.close()

    def detect(self, frame: DecodedVideoFrame) -> NDArray[np.float64] | None:
        image = self._module.Image(
            image_format=self._module.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame.rgb),
        )
        faces = self._landmarker.detect(image).face_landmarks
        if not faces:
            return None
        height, width = frame.rgb.shape[:2]
        return np.array([(point.x * width, point.y * height) for point in faces[0]])
