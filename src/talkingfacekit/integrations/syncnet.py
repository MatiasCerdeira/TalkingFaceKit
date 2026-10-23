"""Optional CPU SyncNet-v2 verification using existing DeepTalk face tracks.

No face detector, scene detector, FFmpeg subprocess, or source rewrite is needed. Bounded
candidate windows are decoded with PyAV; original timestamps anchor both model inputs.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
from numpy.typing import NDArray

from talkingfacekit.active_speaker import ActiveSpeakerResult, FaceObservation
from talkingfacekit.audio import DecodedAudioChunk
from talkingfacekit.integrations._syncnet_model import build_syncnet_model
from talkingfacekit.io.audio import stream_audio_chunks
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.synchronization import (
    SynchronizationPolicy,
    SynchronizationResult,
    SynchronizationWindow,
    SyncStatus,
    candidate_windows,
    decide_synchronization,
)
from talkingfacekit.video import DecodedVideoFrame

SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
_FPS = 25
_SAMPLE_RATE = 16_000


class SyncNetVerifier:
    """Load one official SyncNet-v2 checkpoint and reuse it across candidate windows.

    Parameters
    ----------
    model_path
        Local official ``syncnet_v2.model``. Its SHA-256 is checked before safe tensor loading.
        The caller downloads the model explicitly; construction never uses the network.
    policy
        Offset, confidence, and bounded-window configuration. Positive output means late audio.
    batch_size
        Number of five-frame model inputs per CPU inference batch; bounds temporary memory.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        policy: SynchronizationPolicy | None = None,
        batch_size: int = 10,
    ) -> None:
        started = time.perf_counter()
        path = Path(model_path).expanduser()
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.model_sha256 != SYNCNET_MODEL_SHA256:
            raise ValueError("SyncNet checkpoint SHA-256 does not match official syncnet_v2.model")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.policy = policy or SynchronizationPolicy()
        self.batch_size = batch_size
        try:
            self._torch = import_module("torch")
            self._cv2 = import_module("cv2")
            self._speech = import_module("python_speech_features")
            self._filters = import_module("scipy.ndimage")
        except ImportError as error:
            raise RuntimeError("SyncNet requires the 'synchronization-syncnet' extra") from error
        backend = cast(Any, self._torch)
        self._model = build_syncnet_model(backend)
        model = cast(Any, self._model)
        model.load_state_dict(
            backend.load(path, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        self.initialization_seconds = time.perf_counter() - started

    def verify(
        self, sequence: TalkingFaceSequence, analysis: ActiveSpeakerResult
    ) -> SynchronizationResult:
        """Verify every structural candidate, preserving raw DeepTalk results unchanged.

        Windows fully partition candidates and never extend across a candidate's shot or identity
        boundary. Short/unsupported windows remain uncertain and cannot pass automatically.
        Model/runtime failures propagate; they are never represented as synchronized evidence.

        Parameters
        ----------
        sequence
            Source media and original half-open analysis interval.
        analysis
            DeepTalk result for that source. Existing candidate IDs and boxes are reused.

        Returns
        -------
        SynchronizationResult
            Per-window evidence and decisions with CPU wall time and exact checkpoint hash.
        """
        started = time.perf_counter()
        reader = _SourceWindowReader(sequence)
        windows: list[SynchronizationWindow] = []
        decisions = []
        for run in sorted(analysis.candidate_runs, key=lambda item: item.source_start_seconds):
            if run.status != "candidate":
                continue
            measured = tuple(
                self._measure_window(
                    sequence, analysis.face_observations, run.face_id, start, end, reader
                )
                for start, end in candidate_windows(
                    run.source_start_seconds, run.source_end_seconds, self.policy
                )
            )
            windows.extend(measured)
            decisions.append(
                decide_synchronization(
                    run.source_start_seconds,
                    run.source_end_seconds,
                    run.face_id,
                    measured,
                    self.policy,
                )
            )
        return SynchronizationResult(
            self.policy,
            tuple(windows),
            tuple(decisions),
            self.model_sha256,
            time.perf_counter() - started,
        )

    def measure_window(
        self,
        sequence: TalkingFaceSequence,
        observations: tuple[FaceObservation, ...],
        face_id: int,
        start_seconds: float,
        end_seconds: float,
    ) -> SynchronizationWindow:
        """Measure one bounded window on the source timeline without changing its media.

        Parameters
        ----------
        sequence, observations
            Original source interval and timestamped DeepTalk face observations.
        face_id
            Local identity to crop. No new face detector is run.
        start_seconds, end_seconds
            Half-open bounds inside the sequence, in source seconds.

        Returns
        -------
        SynchronizationWindow
            Measured or explicitly uncertain evidence for this source window.

        Raises
        ------
        ValueError
            If the requested interval exceeds the source sequence.
        """
        return self._measure_window(
            sequence,
            observations,
            face_id,
            start_seconds,
            end_seconds,
            _SourceWindowReader(sequence),
        )

    def _measure_window(
        self,
        sequence: TalkingFaceSequence,
        observations: tuple[FaceObservation, ...],
        face_id: int,
        start_seconds: float,
        end_seconds: float,
        reader: _SourceWindowReader,
    ) -> SynchronizationWindow:
        if start_seconds < sequence.start_seconds or (
            sequence.end_seconds is not None and end_seconds > sequence.end_seconds + 1e-6
        ):
            raise ValueError("synchronization window must be inside the sequence")
        if end_seconds - start_seconds > self.policy.window_seconds + 1e-6:
            raise ValueError("synchronization window exceeds the configured bounded duration")
        if end_seconds - start_seconds < self.policy.minimum_window_seconds - 1e-6:
            return _uncertain(start_seconds, end_seconds, face_id, "sync_window_too_short")
        video, reason = self._read_video(
            sequence, observations, face_id, start_seconds, end_seconds, reader=reader
        )
        if reason is not None:
            return _uncertain(start_seconds, end_seconds, face_id, reason)
        audio, reason = _read_audio(sequence, start_seconds, end_seconds, reader=reader)
        if reason is not None:
            return _uncertain(start_seconds, end_seconds, face_id, reason)
        visual_features, audio_features = self.extract_features(video, audio)
        return measure_feature_offset(
            visual_features,
            audio_features,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            face_id=face_id,
            policy=self.policy,
        )

    def extract_features(
        self, video: NDArray[np.uint8], audio: NDArray[np.int16]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Embed BGR uint8 ``(T,224,224,3)`` at 25 Hz and mono int16 PCM at 16 kHz.

        Inputs match original SyncNet preprocessing: unscaled BGR pixels and default MFCCs on
        full-scale int16 PCM. Returns paired ``(T-5,1024)`` float32 embeddings. No per-offset
        neural inference is required: each tower runs once, followed by cheap distance searches.

        Parameters
        ----------
        video
            Contiguous BGR uint8 frame crops, shape ``(T,224,224,3)``, sampled at 25 Hz.
        audio
            One-dimensional mono int16 PCM sampled at 16 kHz on the same time origin.

        Returns
        -------
        tuple of numpy.ndarray
            Visual and audio float32 feature matrices with matching ``(T-5,1024)`` shape,
            shortened if the audio has fewer usable MFCC windows.

        Raises
        ------
        ValueError
            If dtypes, dimensions, or available model-input duration are invalid.
        """
        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[1:] != (224, 224, 3):
            raise ValueError("SyncNet video requires BGR uint8 (T,224,224,3)")
        if audio.dtype != np.int16 or audio.ndim != 1:
            raise ValueError("SyncNet audio requires mono int16 samples")
        backend = cast(Any, self._torch)
        model = cast(Any, self._model)
        mfcc = np.asarray(cast(Any, self._speech).mfcc(audio, _SAMPLE_RATE).T, dtype=np.float32)
        count = min(video.shape[0] - 5, (mfcc.shape[1] - 20) // 4 + 1)
        if count <= 0:
            raise ValueError("insufficient samples for a SyncNet receptive field")
        visual_outputs = []
        audio_outputs = []
        with backend.inference_mode():
            for first in range(0, count, self.batch_size):
                indices = range(first, min(count, first + self.batch_size))
                visual_batch = np.stack([video[i : i + 5].transpose(3, 0, 1, 2) for i in indices])
                audio_batch = np.stack([mfcc[:, i * 4 : i * 4 + 20][None] for i in indices])
                visual_tensor = backend.from_numpy(visual_batch.astype(np.float32))
                audio_tensor = backend.from_numpy(audio_batch)
                visual_hidden = model.netcnnlip(visual_tensor)
                audio_hidden = model.netcnnaud(audio_tensor)
                visual_outputs.append(model.netfclip(visual_hidden.flatten(1)).numpy())
                audio_outputs.append(model.netfcaud(audio_hidden.flatten(1)).numpy())
        return (
            np.concatenate(visual_outputs).astype(np.float32),
            np.concatenate(audio_outputs).astype(np.float32),
        )

    def _read_video(
        self,
        sequence: TalkingFaceSequence,
        observations: tuple[FaceObservation, ...],
        face_id: int,
        start: float,
        end: float,
        *,
        reader: _SourceWindowReader | None = None,
    ) -> tuple[NDArray[np.uint8], str | None]:
        slots = start + np.arange(math.floor((end - start) * _FPS + 1e-6)) / _FPS
        faces = sorted(
            [
                item
                for item in observations
                if item.face_id == face_id
                and start - 0.04 <= item.source_timestamp_seconds < end + 0.04
            ],
            key=lambda item: item.source_timestamp_seconds,
        )
        empty = np.empty((0, 224, 224, 3), dtype=np.uint8)
        if not faces or slots.size < 6:
            return empty, "sync_face_track_missing"
        timestamps = np.asarray([item.source_timestamp_seconds for item in faces])
        distances = np.min(np.abs(slots[:, None] - timestamps[None, :]), axis=1)
        if np.any(distances > 0.10 + 1e-6):
            return empty, "sync_face_track_gap"
        boxes = np.asarray([item.bounding_box_xywh for item in faces])
        interpolated = np.stack(
            [np.interp(slots, timestamps, boxes[:, i]) for i in range(4)], axis=1
        )
        centers = np.column_stack(
            (
                interpolated[:, 0] + interpolated[:, 2] / 2,
                interpolated[:, 1] + interpolated[:, 3] / 2,
                np.maximum(interpolated[:, 2], interpolated[:, 3]) / 2,
            )
        )
        centers = cast(Any, self._filters).median_filter(centers, size=(13, 1), mode="nearest")
        source = reader or _SourceWindowReader(sequence)
        crops = []
        cv2 = cast(Any, self._cv2)
        for slot, (cx, cy, radius) in zip(slots, centers, strict=True):
            previous = source.video_at(float(slot))
            if previous is None:
                return empty, "sync_video_missing"
            if abs(previous.timestamp_seconds - slot) > 0.04 + 1e-6:
                return empty, "sync_video_timestamp_gap"
            # Original SyncNet crop_scale=0.4: include extra chin/lower-face context.
            x0, x1 = int(cx - radius * 1.4), int(cx + radius * 1.4)
            y0, y1 = int(cy - radius), int(cy + radius * 1.8)
            height, width = previous.rgb.shape[:2]
            crop = np.full((y1 - y0, x1 - x0, 3), 110, dtype=np.uint8)
            left, right = max(0, x0), min(width, x1)
            top, bottom = max(0, y0), min(height, y1)
            if right <= left or bottom <= top:
                return empty, "sync_face_outside_frame"
            crop[top - y0 : bottom - y0, left - x0 : right - x0] = previous.rgb[
                top:bottom, left:right, ::-1
            ]
            crops.append(cv2.resize(crop, (224, 224)))
        return np.stack(crops), None


def measure_feature_offset(
    visual: NDArray[np.float32],
    audio: NDArray[np.float32],
    *,
    start_seconds: float,
    end_seconds: float,
    face_id: int,
    policy: SynchronizationPolicy,
) -> SynchronizationWindow:
    """Search offsets on equal-support embeddings; positive lag means late audio.

    Unlike upstream zero padding at search edges, all lags use the same interior visual frames.
    Short windows use a reduced search range to retain at least one second of paired evidence.
    Boundary optima are uncertain because the true offset may lie outside the searched range.

    Parameters
    ----------
    visual, audio
        Paired finite ``(T,D)`` float32 features sampled at 25 Hz.
    start_seconds, end_seconds, face_id
        Original half-open source window and local face identity.
    policy
        Permitted offset, raw confidence threshold and maximum search radius.

    Returns
    -------
    SynchronizationWindow
        Signed best lag, raw confidence and actual search limit, or unmeasurable evidence.

    Raises
    ------
    ValueError
        If features have incompatible shapes or non-finite values.
    """
    if (
        visual.ndim != 2
        or audio.shape != visual.shape
        or not np.isfinite(visual).all()
        or not np.isfinite(audio).all()
    ):
        raise ValueError("paired SyncNet embeddings must have equal finite (T,D) shapes")
    limit = min(round(policy.maximum_offset_seconds * _FPS), (len(visual) - _FPS) // 2)
    if limit < 1:
        return _uncertain(start_seconds, end_seconds, face_id, "sync_insufficient_evidence")
    count = len(visual) - 2 * limit
    center = visual[limit : limit + count]
    curve = np.asarray(
        [
            np.linalg.norm(center - audio[limit + lag : limit + lag + count], axis=1).mean()
            for lag in range(-limit, limit + 1)
        ]
    )
    best = int(np.argmin(curve))
    offset = (best - limit) / _FPS
    confidence = float(np.median(curve) - curve[best])
    reasons = []
    if confidence < policy.minimum_confidence:
        reasons.append("sync_low_confidence")
    if best in {0, len(curve) - 1}:
        reasons.append("sync_search_boundary")
    if reasons:
        status: SyncStatus = "uncertain"
    elif abs(offset) > policy.tolerance_seconds + 1e-9:
        status = "offset_detected"
        reasons.append("audio_video_offset")
    else:
        status = "synchronized"
        reasons.append("synchronization_passed")
    return SynchronizationWindow(
        start_seconds,
        end_seconds,
        face_id,
        status,
        tuple(reasons),
        offset,
        confidence,
        float(curve[best]),
        limit / _FPS,
    )


def _read_audio(
    sequence: TalkingFaceSequence,
    start: float,
    end: float,
    *,
    reader: _SourceWindowReader | None = None,
) -> tuple[NDArray[np.int16], str | None]:
    """Place source samples on a bounded window, rejecting substantive PTS discontinuities."""
    # Decode source PTS before clipping. A millisecond-quantized stream can contain overlapping
    # frames whose starts are ordered, but trimming the first to `start` can make two starts equal.
    # Retain only this bounded window's chunks; do not materialize its decoded prefix.
    chunks = (reader or _SourceWindowReader(sequence)).audio_between(start, end)
    if not chunks:
        return np.empty(0, dtype=np.int16), "sync_audio_incomplete"
    first = chunks[0]
    count = round((end - start) * first.sample_rate_hz)
    samples = np.zeros((count, first.channel_count), dtype=np.float32)
    previous_end = 0
    for chunk in chunks:
        if (chunk.sample_rate_hz, chunk.channel_layout) != (
            first.sample_rate_hz,
            first.channel_layout,
        ):
            return np.empty(0, dtype=np.int16), "sync_audio_format_changed"
        source_position = round((chunk.start_timestamp_seconds - start) * first.sample_rate_hz)
        source_offset = max(0, -source_position)
        position = max(0, source_position)
        if abs(position - previous_end) > max(1, first.sample_rate_hz // 1000):
            return np.empty(0, dtype=np.int16), "sync_audio_timeline_discontinuity"
        length = min(len(chunk.samples) - source_offset, count - position)
        if length > 0:
            samples[position : position + length] = chunk.samples[
                source_offset : source_offset + length
            ]
        previous_end = position + length
    if count - previous_end > max(1, first.sample_rate_hz // 1000):
        return np.empty(0, dtype=np.int16), "sync_audio_incomplete"
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(samples.T), format="fltp", layout=first.channel_layout
    )
    frame.sample_rate = first.sample_rate_hz
    resampler = av.AudioResampler(format="s16", layout="mono", rate=_SAMPLE_RATE)
    converted = resampler.resample(frame) + resampler.resample(None)
    pcm = np.concatenate([item.to_ndarray().reshape(-1) for item in converted]).astype(np.int16)
    return pcm[: math.floor((end - start) * _SAMPLE_RATE)], None


def _uncertain(start: float, end: float, face_id: int, reason: str) -> SynchronizationWindow:
    return SynchronizationWindow(start, end, face_id, "uncertain", (reason,))


class _SourceWindowReader:
    """Advance source streams once across ordered windows, retaining their boundary frames."""

    def __init__(self, sequence: TalkingFaceSequence) -> None:
        self._sequence = sequence
        self._video: Iterator[DecodedVideoFrame] = iter(())
        self._audio: Iterator[DecodedAudioChunk] = iter(())
        self._previous_video: DecodedVideoFrame | None = None
        self._following_video: DecodedVideoFrame | None = None
        self._pending_audio: DecodedAudioChunk | None = None
        self._audio_tail: list[DecodedAudioChunk] = []
        self._video_started = False
        self._audio_started = False

    def video_at(self, timestamp: float) -> DecodedVideoFrame | None:
        if not self._video_started:
            sequence = self._sequence
            self._video = iter(
                stream_video_frames(
                    sequence.source.path,
                    start_seconds=max(0, sequence.start_seconds - 0.04),
                    end_seconds=None
                    if sequence.end_seconds is None
                    else sequence.end_seconds + 0.04,
                )
            )
            self._previous_video = next(self._video, None)
            self._following_video = next(self._video, None)
            self._video_started = True
        while (
            self._previous_video is not None
            and self._following_video is not None
            and abs(self._following_video.timestamp_seconds - timestamp)
            < abs(self._previous_video.timestamp_seconds - timestamp)
        ):
            self._previous_video = self._following_video
            self._following_video = next(self._video, None)
        return self._previous_video

    def audio_between(self, start: float, end: float) -> list[DecodedAudioChunk]:
        if not self._audio_started:
            self._audio = iter(
                stream_audio_chunks(
                    self._sequence.source.path, end_seconds=self._sequence.end_seconds
                )
            )
            self._pending_audio = next(self._audio, None)
            self._audio_started = True
        chunks = [chunk for chunk in self._audio_tail if chunk.end_timestamp_seconds > start]
        self._audio_tail = [chunk for chunk in chunks if chunk.end_timestamp_seconds > end]
        while self._pending_audio is not None and self._pending_audio.start_timestamp_seconds < end:
            chunk = self._pending_audio
            if chunk.end_timestamp_seconds > start:
                chunks.append(chunk)
                if chunk.end_timestamp_seconds > end:
                    self._audio_tail.append(chunk)
            self._pending_audio = next(self._audio, None)
        return chunks
