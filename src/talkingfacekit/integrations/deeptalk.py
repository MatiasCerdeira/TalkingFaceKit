"""Offline adapter for the experimental DeepTalk-ASD integration."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, metadata
from itertools import chain
from numbers import Real
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Literal, cast

import av
import numpy as np
from numpy.typing import NDArray

from talkingfacekit.audio import DecodedAudioChunk
from talkingfacekit.io.audio import stream_audio_chunks
from talkingfacekit.io.video import stream_video_frames
from talkingfacekit.sequence import TalkingFaceSequence
from talkingfacekit.video import DecodedVideoFrame

_DEEPTALK_DISTRIBUTION = "deeptalk-asd"
_DEEPTALK_SUPPORTED_VERSION = "0.3.1"
_VIDEO_FPS = 25
_FRAME_PERIOD_SECONDS = 1 / _VIDEO_FPS
_VIDEO_TIMESTAMP_TOLERANCE_SECONDS = 1e-9
_AUDIO_SAMPLE_RATE_HZ = 16_000
_AUDIO_FRAME_SAMPLE_COUNT = 480
_AUDIO_REPAIR_BLOCK_SAMPLE_COUNT = 8_192
_AUDIO_TIMESTAMP_TOLERANCE_SECONDS = 1e-3
_AUDIO_SAMPLE_COUNT_TOLERANCE = 1e-7
_SCORE_WINDOW_SECONDS = 1.0
_INSPIREFACE_RESOURCE_ENV = "INSPIREFACE_RESOURCE_PATH"
_INSPIREFACE_RESOURCE_SHA256 = "5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431"

_EventKind = Literal["video", "audio"]
_DeepTalkEvent = tuple[_EventKind, float, object]
_SpeechIntervalStatus = Literal["confirmed", "rejected", "incomplete"]
_AudioTimelineRepairKind = Literal["audio_gap_filled", "audio_overlap_trimmed"]


@dataclass(frozen=True, slots=True)
class DeepTalkFaceObservation:
    """Record one DeepTalk face observation on the source-media timeline.

    Parameters
    ----------
    source_timestamp_seconds
        Timestamp of the 25 Hz source-timeline slot submitted to DeepTalk, in seconds.
    face_id
        DeepTalk's stable face identity. The same identity keys score-window results.
    bounding_box_xywh
        Face rectangle ``(x, y, width, height)`` in source-frame pixels.
    """

    source_timestamp_seconds: float
    face_id: int
    bounding_box_xywh: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Validate the copied source timestamp, identity, and pixel rectangle."""
        if not math.isfinite(self.source_timestamp_seconds) or self.source_timestamp_seconds < 0:
            raise ValueError(
                "source_timestamp_seconds must be finite and non-negative, "
                f"got {self.source_timestamp_seconds}"
            )
        if isinstance(self.face_id, bool) or not isinstance(self.face_id, int):
            raise TypeError(f"face_id must be an integer, got {type(self.face_id).__name__}")
        if len(self.bounding_box_xywh) != 4 or not all(
            math.isfinite(value) for value in self.bounding_box_xywh
        ):
            raise ValueError("bounding_box_xywh must contain four finite pixel values")
        if self.bounding_box_xywh[2] <= 0 or self.bounding_box_xywh[3] <= 0:
            raise ValueError("bounding_box_xywh width and height must be positive")


@dataclass(frozen=True, slots=True)
class DeepTalkScoreWindow:
    """Preserve unchanged DeepTalk scores for one source-timeline window.

    ``raw_scores`` contains DeepTalk's final score for each face identity. Values are not
    probabilities and are not thresholded or smoothed. An empty mapping is retained as evidence.
    The bounds are consecutive diagnostic bounds; DeepTalk 0.3.1 treats audio as half-open but
    includes video at both bounds, so a face at an exact boundary can affect adjacent windows.

    Parameters
    ----------
    source_start_seconds
        Inclusive source-media start of the diagnostic window, in seconds.
    source_end_seconds
        Nominal source-media end of the diagnostic window, in seconds.
    raw_scores
        Immutable snapshot mapping DeepTalk face identities to unchanged scores.
    """

    source_start_seconds: float
    source_end_seconds: float
    raw_scores: Mapping[int, float]

    def __post_init__(self) -> None:
        """Validate the window and snapshot mutable DeepTalk score dictionaries."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("source_end_seconds must be finite and after source_start_seconds")

        copied_scores: dict[int, float] = {}
        for face_id, score in self.raw_scores.items():
            if isinstance(face_id, bool) or not isinstance(face_id, int):
                raise TypeError("raw score face IDs must be integers")
            if not math.isfinite(score):
                raise ValueError(f"raw score for face {face_id} must be finite, got {score}")
            copied_scores[face_id] = score
        object.__setattr__(self, "raw_scores", MappingProxyType(copied_scores))


@dataclass(frozen=True, slots=True)
class DeepTalkSpeechInterval:
    """Record one DeepTalk VAD interval on the source-media timeline.

    Parameters
    ----------
    source_start_seconds
        Inclusive source-media start in seconds, including DeepTalk's retained prefix audio.
    source_end_seconds
        Exclusive source-media end in seconds.
    status
        Whether DeepTalk confirmed, rejected, or reached EOF during the interval.
    """

    source_start_seconds: float
    source_end_seconds: float
    status: _SpeechIntervalStatus

    def __post_init__(self) -> None:
        """Validate source-timeline bounds and the copied VAD status."""
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("source_end_seconds must be finite and after source_start_seconds")
        if self.status not in {"confirmed", "rejected", "incomplete"}:
            raise ValueError(f"unsupported DeepTalk speech interval status: {self.status}")


@dataclass(frozen=True, slots=True)
class DeepTalkAnalysisProvenance:
    """Describe the fixed DeepTalk media transformation and optional capabilities.

    Parameters
    ----------
    backend_version
        Installed ``deeptalk-asd`` distribution version.
    video_sample_rate_hz
        Video grid submitted to the backend.
    audio_sample_rate_hz
        Mono signed-16-bit PCM rate submitted to the backend.
    speaker_embeddings_available
        Always false for this adapter because voiceprints are deliberately disabled.
    """

    backend_version: str
    video_sample_rate_hz: int = _VIDEO_FPS
    audio_sample_rate_hz: int = _AUDIO_SAMPLE_RATE_HZ
    speaker_embeddings_available: bool = False


@dataclass(frozen=True, slots=True)
class DeepTalkAudioTimelineRepair:
    """Describe one explicit repair applied to the source-audio timeline.

    Parameters
    ----------
    kind
        Whether missing source time was filled with silence or overlapping samples were trimmed.
    source_start_seconds
        Inclusive start of the affected region on the source-media timeline.
    source_end_seconds
        Exclusive end of the affected region on the source-media timeline.
    adjusted_sample_count
        Number of source-rate samples inserted or removed per channel.
    sample_rate_hz
        Source sample rate used to quantize the repair.
    """

    kind: _AudioTimelineRepairKind
    source_start_seconds: float
    source_end_seconds: float
    adjusted_sample_count: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        """Validate the repair kind, source bounds, and sample count."""
        if self.kind not in {"audio_gap_filled", "audio_overlap_trimmed"}:
            raise ValueError(f"unsupported audio timeline repair kind: {self.kind}")
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("repair source_start_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.source_end_seconds)
            or self.source_end_seconds <= self.source_start_seconds
        ):
            raise ValueError("repair source_end_seconds must be finite and after its start")
        if (
            isinstance(self.adjusted_sample_count, bool)
            or not isinstance(self.adjusted_sample_count, int)
            or self.adjusted_sample_count <= 0
        ):
            raise ValueError("repair adjusted_sample_count must be a positive integer")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("repair sample_rate_hz must be a positive integer")

    @property
    def duration_seconds(self) -> float:
        """Duration inserted or removed after source-rate sample quantization."""
        return self.adjusted_sample_count / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class DeepTalkAnalysisResult:
    """Contain copied face observations and diagnostic score windows for one sequence.

    Parameters
    ----------
    face_observations
        Observed identities and bounding boxes at submitted 25 Hz video slots.
    speech_intervals
        Completed, rejected, and EOF-truncated VAD intervals on the source timeline.
    score_windows
        Consecutive source-timeline windows containing unchanged DeepTalk scores.
    provenance
        Backend version, fixed media rates, and optional capability status.
    audio_timeline_repairs
        Explicit gap fills and overlap trims applied before resampling for DeepTalk.
    issues
        Immutable diagnostic codes for non-fatal source coverage limitations.
    """

    face_observations: tuple[DeepTalkFaceObservation, ...]
    speech_intervals: tuple[DeepTalkSpeechInterval, ...]
    score_windows: tuple[DeepTalkScoreWindow, ...]
    provenance: DeepTalkAnalysisProvenance
    audio_timeline_repairs: tuple[DeepTalkAudioTimelineRepair, ...] = ()
    issues: tuple[str, ...] = ()


def analyze_sequence(sequence: TalkingFaceSequence) -> DeepTalkAnalysisResult:
    """Run the experimental DeepTalk-ASD 0.3.1 adapter on one sequence offline.

    Video is sampled lazily onto a 25 Hz source-time grid. Audio is streamed through PyAV's
    libswresample boundary as mono 16 kHz signed 16-bit PCM in 30 ms frames. Both streams are fed
    in chronological media-time order, with video preceding audio at equal timestamps. DeepTalk is
    evaluated incrementally in approximately one-second windows so its ten-second audio buffer
    cannot discard early windows before evaluation.

    Parameters
    ----------
    sequence
        Source and half-open source-media interval to analyze. The source must contain audio.

    Returns
    -------
    DeepTalkAnalysisResult
        Backend-owned objects copied into immutable face, speech, score, and provenance values.

    Raises
    ------
    TypeError
        If ``sequence`` is not a :class:`TalkingFaceSequence`.
    ValueError
        If the source has no audio, its decoded format changes within the selected interval, or no
        audio samples can be decoded.
    RuntimeError
        If the optional dependency is absent, detector construction fails, DeepTalk is not version
        0.3.1, or its expected private shim boundary changed.

    Notes
    -----
    Detector construction resolves only the required face, VAD, and LR-ASD assets through
    DeepTalk's hash-verifying public model manager. Normal unit tests replace that boundary and do
    not download or run models.
    """
    if not isinstance(sequence, TalkingFaceSequence):
        raise TypeError(f"sequence must be a TalkingFaceSequence, got {type(sequence).__name__}")
    if not sequence.source.metadata.has_audio:
        raise ValueError(f"DeepTalk analysis requires an audio stream: {sequence.source.path}")

    deeptalk_module = _load_deeptalk_module()
    detector = _create_deeptalk_detector(deeptalk_module)
    video_events = _iter_deeptalk_video_frames(sequence, deeptalk_module)
    audio_timeline_repairs: list[DeepTalkAudioTimelineRepair] = []
    audio_events = _iter_deeptalk_audio_frames(
        sequence,
        deeptalk_module,
        audio_timeline_repairs,
    )

    face_observations: list[DeepTalkFaceObservation] = []
    speech_intervals: list[DeepTalkSpeechInterval] = []
    score_windows: list[DeepTalkScoreWindow] = []
    issues: list[str] = []
    next_window_start_seconds = 0.0
    next_window_end_seconds = _SCORE_WINDOW_SECONDS
    final_audio_end_seconds: float | None = None
    final_video_timestamp_seconds: float | None = None
    latest_utterance: object | None = None
    backend = cast(Any, detector)  # DeepTalk 0.3.1 does not publish typing metadata.

    for event_kind, event_time_seconds, payload in _merge_av_events(video_events, audio_events):
        if event_kind == "video":
            final_video_timestamp_seconds = event_time_seconds
            profiles = backend.append_video(payload, create_time=event_time_seconds)
            face_observations.extend(
                _copy_face_observations(
                    sequence.start_seconds + event_time_seconds,
                    profiles,
                )
            )
            continue

        latest_utterance = backend.append_audio(payload, create_time=event_time_seconds)
        completed_interval = _copy_speech_interval(
            latest_utterance,
            source_offset_seconds=sequence.start_seconds,
            relative_end_seconds=event_time_seconds,
            include_incomplete=False,
        )
        if completed_interval is not None:
            speech_intervals.append(completed_interval)
        final_audio_end_seconds = event_time_seconds
        while next_window_end_seconds <= event_time_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
            score_window = _evaluate_score_window(
                backend,
                source_offset_seconds=sequence.start_seconds,
                start_seconds=next_window_start_seconds,
                end_seconds=next_window_end_seconds,
            )
            score_windows.append(score_window)
            _prune_deeptalk_video_buffer(backend, before_seconds=next_window_end_seconds)
            next_window_start_seconds = next_window_end_seconds
            next_window_end_seconds += _SCORE_WINDOW_SECONDS

    if final_audio_end_seconds is None:
        raise ValueError("DeepTalk analysis received no decoded audio samples")

    analysis_end_seconds = (
        sequence.duration_seconds
        if sequence.duration_seconds is not None
        else final_audio_end_seconds
    )
    if final_audio_end_seconds + _AUDIO_TIMESTAMP_TOLERANCE_SECONDS < analysis_end_seconds:
        raise ValueError(
            "DeepTalk audio ended before the sequence interval: "
            f"audio_end={sequence.start_seconds + final_audio_end_seconds}, "
            f"sequence_end={sequence.start_seconds + analysis_end_seconds}"
        )

    if final_video_timestamp_seconds is None:
        issues.append("no_video_frames")
    elif sequence.duration_seconds is None:
        video_coverage_end_seconds = final_video_timestamp_seconds + _FRAME_PERIOD_SECONDS
        if (
            final_audio_end_seconds + _AUDIO_TIMESTAMP_TOLERANCE_SECONDS
            < video_coverage_end_seconds
        ):
            issues.append("audio_ended_before_video")
            maximum_source_timestamp = sequence.start_seconds + final_audio_end_seconds
            face_observations = [
                observation
                for observation in face_observations
                if observation.source_timestamp_seconds
                <= maximum_source_timestamp + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            ]
        elif video_coverage_end_seconds + _FRAME_PERIOD_SECONDS < final_audio_end_seconds:
            issues.append("video_ended_before_audio")

    incomplete_interval = _copy_speech_interval(
        latest_utterance,
        source_offset_seconds=sequence.start_seconds,
        relative_end_seconds=final_audio_end_seconds,
        include_incomplete=True,
    )
    if incomplete_interval is not None:
        speech_intervals.append(incomplete_interval)

    for repair in audio_timeline_repairs:
        if repair.kind not in issues:
            issues.append(repair.kind)

    if next_window_start_seconds < analysis_end_seconds - _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
        score_window = _evaluate_score_window(
            backend,
            source_offset_seconds=sequence.start_seconds,
            start_seconds=next_window_start_seconds,
            end_seconds=analysis_end_seconds,
        )
        score_windows.append(score_window)
        _prune_deeptalk_video_buffer(backend, before_seconds=analysis_end_seconds)

    return DeepTalkAnalysisResult(
        face_observations=tuple(face_observations),
        speech_intervals=tuple(speech_intervals),
        score_windows=tuple(score_windows),
        provenance=DeepTalkAnalysisProvenance(
            backend_version=_DEEPTALK_SUPPORTED_VERSION,
        ),
        audio_timeline_repairs=tuple(audio_timeline_repairs),
        issues=tuple(issues),
    )


def _sample_video_frames_25_fps(
    sequence: TalkingFaceSequence,
) -> Iterator[tuple[float, DecodedVideoFrame]]:
    """Sample source PTS onto a lazy 25 FPS grid for a sequence interval."""
    frames = stream_video_frames(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        selected_frame = next(frames)
    except StopIteration:
        return

    next_frame = next(frames, None)
    first_frame_timestamp_seconds = selected_frame.timestamp_seconds
    slot_index = max(
        0,
        int((first_frame_timestamp_seconds - sequence.start_seconds) * _VIDEO_FPS),
    )

    while True:
        slot_timestamp_seconds = sequence.start_seconds + slot_index / _VIDEO_FPS
        if (
            slot_timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
            < first_frame_timestamp_seconds
        ):
            slot_index += 1
            continue
        if sequence.end_seconds is not None and slot_timestamp_seconds >= sequence.end_seconds:
            return

        while next_frame is not None:
            selected_distance_seconds = abs(
                selected_frame.timestamp_seconds - slot_timestamp_seconds
            )
            next_distance_seconds = abs(next_frame.timestamp_seconds - slot_timestamp_seconds)
            if (
                next_distance_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
                >= selected_distance_seconds
            ):
                break
            selected_frame = next_frame
            next_frame = next(frames, None)

        if (
            next_frame is None
            and slot_timestamp_seconds
            > selected_frame.timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS
        ):
            return

        yield slot_timestamp_seconds, selected_frame
        slot_index += 1


def _iter_deeptalk_video_frames(
    sequence: TalkingFaceSequence,
    deeptalk_module: ModuleType,
) -> Iterator[tuple[float, object]]:
    for source_timestamp_seconds, source_frame in _sample_video_frames_25_fps(sequence):
        relative_timestamp_seconds = source_timestamp_seconds - sequence.start_seconds
        yield relative_timestamp_seconds, _to_deeptalk_video_frame(source_frame, deeptalk_module)


def _to_deeptalk_video_frame(
    source_frame: DecodedVideoFrame,
    deeptalk_module: ModuleType,
) -> object:
    rgb_bytes = source_frame.rgb.tobytes(order="C")
    backend = cast(Any, deeptalk_module)
    return backend.VideoFrame(
        width=source_frame.rgb.shape[1],
        height=source_frame.rgb.shape[0],
        type=backend.VideoBufferType.RGB24,
        data=rgb_bytes,
    )


def _iter_deeptalk_audio_frames(
    sequence: TalkingFaceSequence,
    deeptalk_module: ModuleType,
    repairs: list[DeepTalkAudioTimelineRepair] | None = None,
) -> Iterator[tuple[float, object]]:
    """Normalize source timestamps and lazily produce continuous DeepTalk PCM frames."""
    chunks = stream_audio_chunks(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        first_chunk = next(chunks)
    except StopIteration as error:
        raise ValueError("DeepTalk analysis received no decoded audio samples") from error

    repair_records = [] if repairs is None else repairs
    anchor_seconds = sequence.start_seconds
    input_sample_rate_hz = first_chunk.sample_rate_hz
    input_format = (
        first_chunk.sample_rate_hz,
        first_chunk.channel_count,
        first_chunk.channel_layout,
    )
    maximum_output_sample_count = (
        max(
            0,
            math.floor(
                (sequence.end_seconds - sequence.start_seconds) * _AUDIO_SAMPLE_RATE_HZ
                + _AUDIO_SAMPLE_COUNT_TOLERANCE
            ),
        )
        if sequence.end_seconds is not None
        else None
    )
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=_AUDIO_SAMPLE_RATE_HZ,
        frame_size=_AUDIO_FRAME_SAMPLE_COUNT,
    )
    normalized_input_sample_count = 0
    emitted_sample_count = 0
    backend = cast(Any, deeptalk_module)

    def convert_outputs(
        output_frames: list[av.AudioFrame],
    ) -> Iterator[tuple[float, object]]:
        nonlocal emitted_sample_count
        for output_frame in output_frames:
            pcm = np.ascontiguousarray(
                output_frame.to_ndarray().reshape(-1),
                dtype=np.int16,
            )
            if maximum_output_sample_count is not None:
                remaining_sample_count = maximum_output_sample_count - emitted_sample_count
                if remaining_sample_count <= 0:
                    continue
                pcm = pcm[:remaining_sample_count]
            if pcm.size == 0:
                continue

            emitted_sample_count += int(pcm.size)
            relative_end_seconds = emitted_sample_count / _AUDIO_SAMPLE_RATE_HZ
            audio_frame = backend.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=_AUDIO_SAMPLE_RATE_HZ,
                num_channels=1,
                samples_per_channel=int(pcm.size),
            )
            yield relative_end_seconds, audio_frame

    def resample_source_samples(
        samples: NDArray[np.float32],
    ) -> Iterator[tuple[float, object]]:
        if samples.shape[0] == 0:
            return
        input_frame = _samples_to_av_frame(
            samples,
            sample_rate_hz=input_sample_rate_hz,
            channel_layout=first_chunk.channel_layout,
        )
        yield from convert_outputs(resampler.resample(input_frame))

    def fill_silence(sample_count: int) -> Iterator[tuple[float, object]]:
        remaining_sample_count = sample_count
        while remaining_sample_count > 0:
            block_sample_count = min(
                remaining_sample_count,
                _AUDIO_REPAIR_BLOCK_SAMPLE_COUNT,
            )
            silence = np.zeros(
                (block_sample_count, first_chunk.channel_count),
                dtype=np.float32,
            )
            yield from resample_source_samples(silence)
            remaining_sample_count -= block_sample_count

    for chunk_index, chunk in enumerate(chain((first_chunk,), chunks)):
        if chunk_index > 0:
            _validate_audio_chunk_format(chunk, expected=input_format)

        expected_start_seconds = (
            anchor_seconds + normalized_input_sample_count / input_sample_rate_hz
        )
        difference_seconds = chunk.start_timestamp_seconds - expected_start_seconds
        tolerance_seconds = max(
            _AUDIO_TIMESTAMP_TOLERANCE_SECONDS,
            1 / input_sample_rate_hz,
        )
        if difference_seconds > tolerance_seconds:
            gap_sample_count = _quantize_audio_repair_sample_count(
                difference_seconds,
                input_sample_rate_hz,
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_gap_filled",
                    source_start_seconds=expected_start_seconds,
                    source_end_seconds=chunk.start_timestamp_seconds,
                    adjusted_sample_count=gap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            yield from fill_silence(gap_sample_count)
            normalized_input_sample_count += gap_sample_count

        samples = chunk.samples
        if difference_seconds < -tolerance_seconds:
            overlap_sample_count = min(
                _quantize_audio_repair_sample_count(
                    -difference_seconds,
                    input_sample_rate_hz,
                ),
                chunk.sample_count,
            )
            overlap_end_seconds = (
                chunk.start_timestamp_seconds + overlap_sample_count / input_sample_rate_hz
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_overlap_trimmed",
                    source_start_seconds=chunk.start_timestamp_seconds,
                    source_end_seconds=overlap_end_seconds,
                    adjusted_sample_count=overlap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            samples = np.ascontiguousarray(samples[overlap_sample_count:], dtype=np.float32)

        normalized_input_sample_count += int(samples.shape[0])
        yield from resample_source_samples(samples)

    if sequence.end_seconds is not None:
        expected_end_seconds = anchor_seconds + normalized_input_sample_count / input_sample_rate_hz
        trailing_gap_seconds = sequence.end_seconds - expected_end_seconds
        tolerance_seconds = max(
            _AUDIO_TIMESTAMP_TOLERANCE_SECONDS,
            1 / input_sample_rate_hz,
        )
        if trailing_gap_seconds > tolerance_seconds:
            trailing_gap_sample_count = _quantize_audio_repair_sample_count(
                trailing_gap_seconds,
                input_sample_rate_hz,
            )
            _append_audio_timeline_repair(
                repair_records,
                DeepTalkAudioTimelineRepair(
                    kind="audio_gap_filled",
                    source_start_seconds=expected_end_seconds,
                    source_end_seconds=sequence.end_seconds,
                    adjusted_sample_count=trailing_gap_sample_count,
                    sample_rate_hz=input_sample_rate_hz,
                ),
            )
            yield from fill_silence(trailing_gap_sample_count)

    yield from convert_outputs(resampler.resample(None))


def _samples_to_av_frame(
    samples: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    channel_layout: str,
) -> av.AudioFrame:
    planar_samples = np.ascontiguousarray(samples.transpose(), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(
        planar_samples,
        format="fltp",
        layout=channel_layout,
    )
    frame.sample_rate = sample_rate_hz
    return frame


def _quantize_audio_repair_sample_count(duration_seconds: float, sample_rate_hz: int) -> int:
    """Round one positive timestamp discrepancy to the nearest source sample."""
    sample_count = math.floor(duration_seconds * sample_rate_hz + 0.5)
    if sample_count <= 0:
        raise RuntimeError(
            "Audio timestamp discrepancy exceeded tolerance but quantized to no samples"
        )
    return sample_count


def _append_audio_timeline_repair(
    repairs: list[DeepTalkAudioTimelineRepair],
    repair: DeepTalkAudioTimelineRepair,
) -> None:
    """Append a repair, coalescing adjacent adjustments of the same kind and rate."""
    if not repairs:
        repairs.append(repair)
        return

    previous = repairs[-1]
    if (
        previous.kind == repair.kind
        and previous.sample_rate_hz == repair.sample_rate_hz
        and abs(repair.source_start_seconds - previous.source_end_seconds)
        <= _AUDIO_TIMESTAMP_TOLERANCE_SECONDS
    ):
        repairs[-1] = DeepTalkAudioTimelineRepair(
            kind=previous.kind,
            source_start_seconds=previous.source_start_seconds,
            source_end_seconds=max(previous.source_end_seconds, repair.source_end_seconds),
            adjusted_sample_count=(previous.adjusted_sample_count + repair.adjusted_sample_count),
            sample_rate_hz=previous.sample_rate_hz,
        )
        return

    repairs.append(repair)


def _validate_audio_chunk_format(
    chunk: DecodedAudioChunk,
    *,
    expected: tuple[int, int, str],
) -> None:
    observed = (chunk.sample_rate_hz, chunk.channel_count, chunk.channel_layout)
    if observed != expected:
        raise ValueError(
            "Decoded audio format changed within the sequence: "
            f"expected={expected}, actual={observed}"
        )


def _merge_av_events(
    video_events: Iterator[tuple[float, object]],
    audio_events: Iterator[tuple[float, object]],
) -> Iterator[_DeepTalkEvent]:
    """Merge sorted offline streams, preferring video at equal media timestamps."""
    video_event = next(video_events, None)
    audio_event = next(audio_events, None)
    while video_event is not None or audio_event is not None:
        if video_event is not None and (audio_event is None or video_event[0] <= audio_event[0]):
            yield "video", video_event[0], video_event[1]
            video_event = next(video_events, None)
        else:
            if audio_event is None:
                raise RuntimeError("A/V event merge reached an invalid state")
            yield "audio", audio_event[0], audio_event[1]
            audio_event = next(audio_events, None)


def _load_deeptalk_module() -> ModuleType:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        deeptalk_asd = import_module("deeptalk_asd")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepTalk analysis requires the 'active-speaker-deeptalk' extra"
        ) from error
    return deeptalk_asd


def _create_deeptalk_detector(deeptalk_module: ModuleType) -> object:
    _require_supported_deeptalk_version()
    backend = cast(Any, deeptalk_module)
    explicit_face_resource = _validate_explicit_inspireface_resource()
    model_cache_dir = Path(backend.get_model_cache_dir())
    face_resource = explicit_face_resource or Path(backend.ensure_model("Pikachu", model_cache_dir))
    for model_name in (
        "silero_vad.onnx",
        "audio_frontend.onnx",
        "visual_frontend.onnx",
        "av_backend.onnx",
    ):
        backend.ensure_model(model_name, model_cache_dir)

    detector = backend.ASDDetectorFactory(
        face_detector={"type": "inspireface", "model_dir": str(face_resource)},
        turn_detector={"type": "silero-vad", "model_dir": str(model_cache_dir)},
        speaker_detector={
            "type": "LR-ASD-ONNX",
            "model_dir": str(model_cache_dir),
            "voiceprint_model_name": "disabled-by-talkingfacekit",
        },
    ).create()
    if detector is None:
        raise RuntimeError("DeepTalk-ASD failed to create its detector")
    _configure_deeptalk_0_3_1_for_offline(detector)
    return cast(object, detector)


def _validate_explicit_inspireface_resource() -> Path | None:
    """Validate DeepTalk's default InspireFace asset before native code receives it."""
    configured_path = os.environ.get(_INSPIREFACE_RESOURCE_ENV)
    if configured_path is None:
        return None

    resource_path = Path(configured_path)
    if not resource_path.is_file():
        raise RuntimeError(
            f"{_INSPIREFACE_RESOURCE_ENV} must name the DeepTalk Pikachu model file: "
            f"{resource_path}"
        )

    digest = sha256()
    with resource_path.open("rb") as resource_file:
        for chunk in iter(lambda: resource_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != _INSPIREFACE_RESOURCE_SHA256:
        raise RuntimeError(
            f"{_INSPIREFACE_RESOURCE_ENV} does not match the supported DeepTalk Pikachu model hash"
        )
    return resource_path


def _require_supported_deeptalk_version() -> None:
    try:
        installed_version = metadata.version(_DEEPTALK_DISTRIBUTION)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("DeepTalk-ASD is not installed") from error
    if installed_version != _DEEPTALK_SUPPORTED_VERSION:
        raise RuntimeError(
            f"The offline shim supports only deeptalk-asd==0.3.1, got {installed_version}"
        )


def _configure_deeptalk_0_3_1_for_offline(detector: object) -> None:
    """Disable DeepTalk 0.3.1 realtime assumptions for deterministic offline input.

    DeepTalk otherwise drops some exact 25 Hz frames through a second float-based throttle and
    expires tracks and VAD state according to wall-clock time. Voiceprints are also disabled so the
    optional native Sherpa linkage cannot silently change score semantics. Every dependency on the
    private layout is validated and contained in this function.
    """
    _require_supported_deeptalk_version()

    detector_type = type(detector)
    if detector_type.__module__ != "deeptalk_asd.asd" or detector_type.__name__ != "ASD":
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected deeptalk_asd.asd.ASD"
        )
    speaker_detector = getattr(detector, "_speaker_detector", None)
    speaker_type = type(speaker_detector)
    if (
        speaker_type.__module__ != "deeptalk_asd.speaker_detector.lrasd_onnx"
        or speaker_type.__name__ != "LRASDOnnxSpeakerDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected LRASDOnnxSpeakerDetector"
        )
    face_detector = getattr(detector, "_face_detector", None)
    face_type = type(face_detector)
    if (
        face_type.__module__ != "deeptalk_asd.face_detector.inspireface_detector"
        or face_type.__name__ != "InspireFaceDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected InspireFaceDetector"
        )
    turn_detector = getattr(detector, "_turn_detector", None)
    turn_type = type(turn_detector)
    if (
        turn_type.__module__ != "deeptalk_asd.turn_detector.silero_vad_turn_detector"
        or turn_type.__name__ != "SileroVadTurnDetector"
    ):
        raise RuntimeError(
            "DeepTalk 0.3.1 detector structure changed: expected SileroVadTurnDetector"
        )
    vad = getattr(turn_detector, "_vad", None)
    vad_type = type(vad)
    if (
        vad_type.__module__ != "deeptalk_asd.turn_detector.vad.silero_vad"
        or vad_type.__name__ != "SileroVAD"
    ):
        raise RuntimeError("DeepTalk 0.3.1 detector structure changed: expected SileroVAD")

    backend = cast(Any, speaker_detector)
    face_backend = cast(Any, face_detector)
    vad_backend = cast(Any, vad)
    try:
        video_frame_rate = backend.video_frame_rate
        audio_sample_rate = backend.audio_sample_rate
        minimum_frame_interval = backend._min_frame_interval
        _max_track_age = backend.max_track_age
        voice_extractor = backend.voice_extractor
        voice_profiles = backend.voice_profiles
        _last_reset_time = vad_backend._last_reset_time
        remove_stale_faces = face_backend._remove_stale_faces
    except AttributeError as error:
        raise RuntimeError(
            f"DeepTalk 0.3.1 detector structure changed: missing {error.name}"
        ) from error

    if video_frame_rate != _VIDEO_FPS or audio_sample_rate != _AUDIO_SAMPLE_RATE_HZ:
        raise RuntimeError(
            "DeepTalk 0.3.1 detector uses unexpected media rates: "
            f"video={video_frame_rate}, audio={audio_sample_rate}"
        )
    if isinstance(minimum_frame_interval, bool) or not isinstance(minimum_frame_interval, Real):
        raise TypeError("DeepTalk 0.3.1 _min_frame_interval has an unexpected type")
    if not math.isclose(
        float(minimum_frame_interval),
        _FRAME_PERIOD_SECONDS,
        rel_tol=0.0,
        abs_tol=_VIDEO_TIMESTAMP_TOLERANCE_SECONDS,
    ):
        raise RuntimeError("DeepTalk 0.3.1 _min_frame_interval has an unexpected value")
    if voice_extractor is not None and not hasattr(voice_extractor, "extract_from_samples"):
        raise RuntimeError("DeepTalk 0.3.1 voice_extractor has an unexpected structure")
    if not isinstance(voice_profiles, dict):
        raise TypeError("DeepTalk 0.3.1 voice_profiles has an unexpected structure")
    if not callable(remove_stale_faces):
        raise TypeError("DeepTalk 0.3.1 face expiry hook has an unexpected structure")

    backend._min_frame_interval = 0.0
    backend.max_track_age = math.inf
    backend.voice_extractor = None
    backend.voice_profiles.clear()
    vad_backend._last_reset_time = math.inf
    face_backend._remove_stale_faces = lambda: None

    face_module = import_module(face_type.__module__)
    face_module_backend = cast(Any, face_module)
    disappearance_interval = getattr(face_module, "FACE_DISAPPEARANCE_INTERVAL_SECONDS", None)
    if isinstance(disappearance_interval, bool) or not isinstance(disappearance_interval, Real):
        raise TypeError("DeepTalk 0.3.1 face disappearance interval has an unexpected value")
    face_module_backend.FACE_DISAPPEARANCE_INTERVAL_SECONDS = 0


def _copy_speech_interval(
    utterance: object | None,
    *,
    source_offset_seconds: float,
    relative_end_seconds: float,
    include_incomplete: bool,
) -> DeepTalkSpeechInterval | None:
    """Copy a terminal or EOF-truncated DeepTalk utterance without retaining audio frames."""
    if utterance is None:
        return None

    state_name = getattr(getattr(utterance, "turn_state", None), "name", None)
    if not isinstance(state_name, str):
        raise TypeError("DeepTalk utterance does not expose a named turn_state")
    terminal_statuses: dict[str, _SpeechIntervalStatus] = {
        "TURN_END": "confirmed",
        "TURN_REJECTED": "rejected",
    }
    active_states = {"TURN_START", "TURN_CONTINUE", "TURN_CONFIRMED", "TURN_SILENCE"}
    if include_incomplete:
        if state_name not in active_states:
            return None
        interval_status: _SpeechIntervalStatus = "incomplete"
    else:
        terminal_status = terminal_statuses.get(state_name)
        if terminal_status is None:
            return None
        interval_status = terminal_status

    duration_method = getattr(utterance, "duration_seconds", None)
    if not callable(duration_method):
        raise TypeError("DeepTalk utterance does not expose duration_seconds()")
    duration_seconds = float(duration_method())
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise RuntimeError(f"DeepTalk returned an invalid utterance duration: {duration_seconds}")

    source_end_seconds = source_offset_seconds + relative_end_seconds
    source_start_seconds = max(source_offset_seconds, source_end_seconds - duration_seconds)
    if source_end_seconds <= source_start_seconds:
        return None
    return DeepTalkSpeechInterval(
        source_start_seconds=source_start_seconds,
        source_end_seconds=source_end_seconds,
        status=interval_status,
    )


def _prune_deeptalk_video_buffer(backend: Any, *, before_seconds: float) -> None:
    """Discard mouth images older than completed score windows."""
    speaker_detector = getattr(backend, "_speaker_detector", None)
    if speaker_detector is None:
        return
    video_buffer = getattr(speaker_detector, "video_buffer", None)
    if not isinstance(video_buffer, dict):
        raise TypeError("DeepTalk 0.3.1 video_buffer has an unexpected structure")

    for face_id, frames in list(video_buffer.items()):
        retained_frames: list[Any] = []
        for frame_record in cast(list[Any], frames):
            if not isinstance(frame_record, tuple) or len(frame_record) != 2:
                raise RuntimeError("DeepTalk 0.3.1 video buffer record has an unexpected structure")
            timestamp_seconds = float(frame_record[1])
            if timestamp_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS >= before_seconds:
                retained_frames.append(frame_record)
        video_buffer[face_id] = retained_frames


def _copy_face_observations(
    source_timestamp_seconds: float,
    profiles: object,
) -> list[DeepTalkFaceObservation]:
    observations: list[DeepTalkFaceObservation] = []
    for profile in cast(list[Any], profiles):
        rectangle = profile.face_rectangle
        observations.append(
            DeepTalkFaceObservation(
                source_timestamp_seconds=source_timestamp_seconds,
                face_id=int(profile.id),
                bounding_box_xywh=(
                    float(rectangle.x),
                    float(rectangle.y),
                    float(rectangle.width),
                    float(rectangle.height),
                ),
            )
        )
    return observations


def _evaluate_score_window(
    backend: Any,
    *,
    source_offset_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> DeepTalkScoreWindow:
    scores = cast(Mapping[Any, Any], backend.evaluate(start_seconds, end_seconds))
    copied_scores = {int(face_id): float(score) for face_id, score in scores.items()}

    return DeepTalkScoreWindow(
        source_start_seconds=source_offset_seconds + start_seconds,
        source_end_seconds=source_offset_seconds + end_seconds,
        raw_scores=copied_scores,
    )
