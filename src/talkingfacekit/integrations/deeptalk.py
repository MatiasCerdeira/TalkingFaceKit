"""Offline adapter for the experimental DeepTalk-ASD integration."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from itertools import chain
from numbers import Real
from types import MappingProxyType, ModuleType
from typing import Any, Literal, cast

import av
import numpy as np

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
_AUDIO_TIMESTAMP_TOLERANCE_SECONDS = 1e-3
_AUDIO_SAMPLE_COUNT_TOLERANCE = 1e-7
_SCORE_WINDOW_SECONDS = 1.0

_EventKind = Literal["video", "audio"]
_DeepTalkEvent = tuple[_EventKind, float, object]


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
        """Snapshot scores so mutable DeepTalk dictionaries cannot leak into the result."""
        object.__setattr__(self, "raw_scores", MappingProxyType(dict(self.raw_scores)))


@dataclass(frozen=True, slots=True)
class DeepTalkAnalysisResult:
    """Contain copied face observations and diagnostic score windows for one sequence.

    Parameters
    ----------
    face_observations
        Observed identities and bounding boxes at submitted 25 Hz video slots.
    score_windows
        Consecutive source-timeline windows containing unchanged DeepTalk scores.
    """

    face_observations: tuple[DeepTalkFaceObservation, ...]
    score_windows: tuple[DeepTalkScoreWindow, ...]


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
        Backend-owned objects copied into immutable face observations and raw score windows.

    Raises
    ------
    TypeError
        If ``sequence`` is not a :class:`TalkingFaceSequence`.
    ValueError
        If the source has no audio or decoded audio is discontinuous or does not cover a bounded
        sequence interval.
    RuntimeError
        If the optional dependency is absent, detector construction fails, DeepTalk is not version
        0.3.1, or its expected private shim boundary changed.

    Notes
    -----
    Detector construction may resolve DeepTalk's model assets through its public factory. Normal
    unit tests replace that boundary and do not download or run models.
    """
    if not isinstance(sequence, TalkingFaceSequence):
        raise TypeError(f"sequence must be a TalkingFaceSequence, got {type(sequence).__name__}")
    if not sequence.source.metadata.has_audio:
        raise ValueError(f"DeepTalk analysis requires an audio stream: {sequence.source.path}")

    deeptalk_module = _load_deeptalk_module()
    detector = _create_deeptalk_detector(deeptalk_module)
    video_events = _iter_deeptalk_video_frames(sequence, deeptalk_module)
    audio_events = _iter_deeptalk_audio_frames(sequence, deeptalk_module)

    face_observations: list[DeepTalkFaceObservation] = []
    score_windows: list[DeepTalkScoreWindow] = []
    next_window_start_seconds = 0.0
    next_window_end_seconds = _SCORE_WINDOW_SECONDS
    final_audio_end_seconds: float | None = None
    backend = cast(Any, detector)  # DeepTalk 0.3.1 does not publish typing metadata.

    for event_kind, event_time_seconds, payload in _merge_av_events(video_events, audio_events):
        if event_kind == "video":
            profiles = backend.append_video(payload, create_time=event_time_seconds)
            face_observations.extend(
                _copy_face_observations(
                    sequence.start_seconds + event_time_seconds,
                    profiles,
                )
            )
            continue

        backend.append_audio(payload, create_time=event_time_seconds)
        final_audio_end_seconds = event_time_seconds
        while next_window_end_seconds <= event_time_seconds + _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
            score_windows.append(
                _evaluate_score_window(
                    backend,
                    source_offset_seconds=sequence.start_seconds,
                    start_seconds=next_window_start_seconds,
                    end_seconds=next_window_end_seconds,
                )
            )
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

    if next_window_start_seconds < analysis_end_seconds - _VIDEO_TIMESTAMP_TOLERANCE_SECONDS:
        score_windows.append(
            _evaluate_score_window(
                backend,
                source_offset_seconds=sequence.start_seconds,
                start_seconds=next_window_start_seconds,
                end_seconds=analysis_end_seconds,
            )
        )

    return DeepTalkAnalysisResult(
        face_observations=tuple(face_observations),
        score_windows=tuple(score_windows),
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
) -> Iterator[tuple[float, object]]:
    """Convert source audio lazily to timestamped DeepTalk PCM frames."""
    chunks = stream_audio_chunks(
        sequence.source.path,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
    )
    try:
        first_chunk = next(chunks)
    except StopIteration as error:
        raise ValueError("DeepTalk analysis received no decoded audio samples") from error

    anchor_seconds = _validate_first_audio_chunk(sequence, first_chunk)
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
                (sequence.end_seconds - anchor_seconds) * _AUDIO_SAMPLE_RATE_HZ
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
    input_sample_count = 0
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
            relative_end_seconds = (
                anchor_seconds
                - sequence.start_seconds
                + emitted_sample_count / _AUDIO_SAMPLE_RATE_HZ
            )
            audio_frame = backend.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=_AUDIO_SAMPLE_RATE_HZ,
                num_channels=1,
                samples_per_channel=int(pcm.size),
            )
            yield relative_end_seconds, audio_frame

    for chunk_index, chunk in enumerate(chain((first_chunk,), chunks)):
        if chunk_index > 0:
            _validate_audio_chunk_format(chunk, expected=input_format)
            expected_start_seconds = anchor_seconds + input_sample_count / input_sample_rate_hz
            _validate_audio_continuity(
                chunk.start_timestamp_seconds,
                expected_start_seconds=expected_start_seconds,
            )

        input_frame = _decoded_chunk_to_av_frame(chunk)
        input_sample_count += chunk.sample_count
        yield from convert_outputs(resampler.resample(input_frame))

    decoded_end_seconds = anchor_seconds + input_sample_count / input_sample_rate_hz
    _validate_bounded_audio_end(sequence, decoded_end_seconds, input_sample_rate_hz)
    yield from convert_outputs(resampler.resample(None))


def _decoded_chunk_to_av_frame(chunk: DecodedAudioChunk) -> av.AudioFrame:
    planar_samples = np.ascontiguousarray(chunk.samples.transpose(), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(
        planar_samples,
        format="fltp",
        layout=chunk.channel_layout,
    )
    frame.sample_rate = chunk.sample_rate_hz
    return frame


def _validate_first_audio_chunk(
    sequence: TalkingFaceSequence,
    chunk: DecodedAudioChunk,
) -> float:
    tolerance_seconds = max(
        _AUDIO_TIMESTAMP_TOLERANCE_SECONDS,
        1 / chunk.sample_rate_hz,
    )
    difference_seconds = chunk.start_timestamp_seconds - sequence.start_seconds
    if difference_seconds > tolerance_seconds:
        raise ValueError(
            "Decoded audio has a leading gap: "
            f"expected={sequence.start_seconds}, actual={chunk.start_timestamp_seconds}, "
            f"gap={difference_seconds} seconds"
        )
    if difference_seconds < -tolerance_seconds:
        raise ValueError(
            "Decoded audio begins before the requested interval: "
            f"sequence_start={sequence.start_seconds}, actual={chunk.start_timestamp_seconds}"
        )
    return max(chunk.start_timestamp_seconds, sequence.start_seconds)


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


def _validate_audio_continuity(
    actual_start_seconds: float,
    *,
    expected_start_seconds: float,
) -> None:
    difference_seconds = actual_start_seconds - expected_start_seconds
    if difference_seconds > _AUDIO_TIMESTAMP_TOLERANCE_SECONDS:
        raise ValueError(
            "Decoded audio has a gap that DeepTalk cannot represent: "
            f"expected={expected_start_seconds}, actual={actual_start_seconds}, "
            f"gap={difference_seconds} seconds"
        )
    if difference_seconds < -_AUDIO_TIMESTAMP_TOLERANCE_SECONDS:
        raise ValueError(
            "Decoded audio overlaps earlier samples: "
            f"expected={expected_start_seconds}, actual={actual_start_seconds}, "
            f"overlap={-difference_seconds} seconds"
        )


def _validate_bounded_audio_end(
    sequence: TalkingFaceSequence,
    decoded_end_seconds: float,
    sample_rate_hz: int,
) -> None:
    if sequence.end_seconds is None:
        return
    tolerance_seconds = max(_AUDIO_TIMESTAMP_TOLERANCE_SECONDS, 1 / sample_rate_hz)
    difference_seconds = sequence.end_seconds - decoded_end_seconds
    if difference_seconds > tolerance_seconds:
        raise ValueError(
            "Decoded audio has a trailing gap: "
            f"decoded_end={decoded_end_seconds}, sequence_end={sequence.end_seconds}, "
            f"gap={difference_seconds} seconds"
        )
    if difference_seconds < -tolerance_seconds:
        raise ValueError(
            "Decoded audio extends beyond the requested interval: "
            f"decoded_end={decoded_end_seconds}, sequence_end={sequence.end_seconds}"
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
    try:
        import deeptalk_asd  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepTalk analysis requires the 'active-speaker-deeptalk' extra"
        ) from error
    return cast(ModuleType, deeptalk_asd)


def _create_deeptalk_detector(deeptalk_module: ModuleType) -> object:
    _require_supported_deeptalk_version()
    backend = cast(Any, deeptalk_module)
    detector = backend.ASDDetectorFactory().create()
    if detector is None:
        raise RuntimeError("DeepTalk-ASD failed to create its detector")
    _configure_deeptalk_0_3_1_for_offline(detector)
    return cast(object, detector)


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
    """Disable two DeepTalk 0.3.1 realtime assumptions for deterministic offline input.

    DeepTalk otherwise drops some exact 25 Hz frames through a second float-based throttle and
    immediately expires tracks by comparing relative media timestamps with ``perf_counter()``.
    Every dependency on its private layout is validated and contained in this function.
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

    backend = cast(Any, speaker_detector)
    try:
        video_frame_rate = backend.video_frame_rate
        audio_sample_rate = backend.audio_sample_rate
        minimum_frame_interval = backend._min_frame_interval
        _max_track_age = backend.max_track_age
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

    backend._min_frame_interval = 0.0
    backend.max_track_age = math.inf


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
