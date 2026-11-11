import json
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from talkingfacekit import TalkingFaceSegment, TalkingFaceSequence, stream_video_frames
from talkingfacekit.dataset import export_segments
from talkingfacekit.flame import FlameResult

_FRAME_RATE = 25
_SAMPLE_RATE = 16_000


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a tiny three-second source with audio and per-frame brightness changes."""
    path = tmp_path_factory.mktemp("source") / "source.mkv"
    with av.open(path, mode="w", format="matroska") as container:
        video = cast(av.VideoStream, container.add_stream("ffv1", rate=_FRAME_RATE))
        video.width, video.height = 32, 24
        audio = cast(av.AudioStream, container.add_stream("flac", rate=_SAMPLE_RATE))
        audio.codec_context.layout = "mono"
        for index in range(3 * _FRAME_RATE):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), index * 3, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, _FRAME_RATE)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        samples = np.zeros((1, 3 * _SAMPLE_RATE), dtype=np.int16)
        sound = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        sound.sample_rate, sound.pts, sound.time_base = _SAMPLE_RATE, 0, Fraction(1, _SAMPLE_RATE)
        container.mux(audio.encode(sound))
        container.mux(audio.encode())
    return path


def segments(video: Path, *bounds: tuple[float, float]) -> tuple[TalkingFaceSegment, ...]:
    source = TalkingFaceSequence.from_video(video).source
    return tuple(
        TalkingFaceSegment(source, start, end, 7, "candidate", ("visual_policy_passed",))
        for start, end in bounds
    )


def fake_flame(sequence: TalkingFaceSequence) -> FlameResult:
    """Return a synthetic result on the decoder's real source frames, as SMIRK would."""
    frames = list(
        stream_video_frames(
            sequence.source.path,
            start_seconds=sequence.start_seconds,
            end_seconds=sequence.end_seconds,
        )
    )
    count = len(frames)
    estimated = np.ones(count, dtype=np.bool_)
    estimated[0] = False

    def parameters(size: int) -> NDArray[np.float32]:
        values = np.zeros((count, size), dtype=np.float32)
        values[~estimated] = np.nan
        return values

    return FlameResult(
        source=sequence.source,
        source_sha256="a" * 64,
        start_seconds=sequence.start_seconds,
        end_seconds=sequence.end_seconds,
        model_variant="synthetic-basis",
        frame_indices=np.array([frame.frame_index for frame in frames], dtype=np.int64),
        timestamps_seconds=np.array([frame.timestamp_seconds for frame in frames]),
        estimated=estimated,
        missing_reasons=tuple("" if found else "face_landmarks_unavailable" for found in estimated),
        shape=parameters(4),
        expression=parameters(3),
        global_pose=parameters(3),
        jaw_pose=parameters(3),
        extras={},
        extra_conventions={},
        provenance={"backend": "synthetic", "checkpoint_sha256": "b" * 64},
    )


def clip_timestamps(path: Path) -> NDArray[np.float64]:
    with av.open(path) as container:
        return np.array([float(frame.time) for frame in container.decode(video=0)])


def test_exports_linked_clip_parameters_and_metadata_per_segment(
    tmp_path: Path, source_video: Path
) -> None:
    selected = segments(source_video, (0.5, 1.3), (1.6, 2.4))

    exports = export_segments(selected, tmp_path, extract_flame=fake_flame)

    assert [item.directory.name for item in exports] == ["segment_001", "segment_002"]
    for segment, item in zip(selected, exports, strict=True):
        assert item.error is None
        assert item.segment is segment
        assert item.clip_path == item.directory / "clip.mp4"
        assert item.flame_path == item.directory / "flame.npz"
        assert item.metadata_path == item.directory / "segment.json"

        metadata = json.loads(item.directory.joinpath("segment.json").read_text("utf-8"))
        assert metadata["kind"] == "talkingfacekit.segment"
        assert metadata["schema_version"] == 1
        assert metadata["source"] == str(source_video.resolve())
        assert metadata["source_sha256"] == "a" * 64
        assert (metadata["start_seconds"], metadata["end_seconds"]) == (
            segment.start_seconds,
            segment.end_seconds,
        )
        assert metadata["face_id"] == 7
        assert metadata["status"] == "candidate"
        assert metadata["reason_codes"] == ["visual_policy_passed"]
        assert metadata["clip"] == {
            "path": "clip.mp4",
            "source_time_offset_seconds": segment.start_seconds,
        }
        flame = FlameResult.load(item.directory / metadata["flame"]["path"])
        assert metadata["flame"]["frame_count"] == flame.frame_count
        assert (
            metadata["flame"]["estimated_count"] == flame.estimated_count == flame.frame_count - 1
        )
        assert metadata["flame"]["backend"] == "synthetic"

        # The parameters stay on the source timeline and describe exactly the clip's frames.
        assert flame.source.path.resolve() == source_video.resolve()
        assert (flame.start_seconds, flame.end_seconds) == (
            segment.start_seconds,
            segment.end_seconds,
        )
        clip_times = clip_timestamps(item.directory / metadata["clip"]["path"])
        assert clip_times.shape == flame.timestamps_seconds.shape
        np.testing.assert_allclose(
            clip_times + metadata["clip"]["source_time_offset_seconds"],
            flame.timestamps_seconds,
            atol=1e-3,
        )


def test_failed_segment_leaves_no_outputs_and_later_segments_continue(
    tmp_path: Path, source_video: Path
) -> None:
    calls: list[float] = []

    def fail_second(sequence: TalkingFaceSequence) -> FlameResult:
        calls.append(sequence.start_seconds)
        if len(calls) == 2:
            raise RuntimeError("simulated reconstruction failure")
        return fake_flame(sequence)

    exports = export_segments(
        segments(source_video, (0.0, 0.8), (0.8, 1.6), (1.6, 2.4)),
        tmp_path,
        extract_flame=fail_second,
    )

    assert calls == [0.0, 0.8, 1.6]
    assert [item.error is None for item in exports] == [True, False, True]
    failure = exports[1].error
    assert failure is not None
    assert (failure.stage, failure.error_type) == ("flame", "RuntimeError")
    assert "simulated reconstruction failure" in failure.message
    assert (exports[1].clip_path, exports[1].flame_path, exports[1].metadata_path) == (
        None,
        None,
        None,
    )
    assert not (tmp_path / "segment_002").exists()
    assert (tmp_path / "segment_003" / "segment.json").is_file()


def test_export_failure_is_recorded_before_reconstruction(
    tmp_path: Path, source_video: Path
) -> None:
    calls: list[TalkingFaceSequence] = []

    def record(sequence: TalkingFaceSequence) -> FlameResult:
        calls.append(sequence)
        return fake_flame(sequence)

    exports = export_segments(segments(source_video, (5.0, 6.0)), tmp_path, extract_flame=record)

    failure = exports[0].error
    assert failure is not None
    assert (failure.stage, failure.error_type) == ("export", "ValueError")
    assert calls == []
    assert not (tmp_path / "segment_001").exists()


def test_parameters_for_another_interval_are_rejected(tmp_path: Path, source_video: Path) -> None:
    def shifted(sequence: TalkingFaceSequence) -> FlameResult:
        return fake_flame(sequence.clip(sequence.start_seconds + 0.1))

    exports = export_segments(segments(source_video, (0.5, 1.3)), tmp_path, extract_flame=shifted)

    failure = exports[0].error
    assert failure is not None
    assert (failure.stage, failure.error_type) == ("flame", "ValueError")
    assert not (tmp_path / "segment_001").exists()


def test_existing_outputs_are_refused_before_any_work_unless_overwrite(
    tmp_path: Path, source_video: Path
) -> None:
    selected = segments(source_video, (0.5, 1.3))
    export_segments(selected, tmp_path, extract_flame=fake_flame)
    before = (tmp_path / "segment_001" / "segment.json").read_bytes()

    def forbid(sequence: TalkingFaceSequence) -> FlameResult:
        raise AssertionError("reconstruction must not run")

    with pytest.raises(FileExistsError):
        export_segments(selected, tmp_path, extract_flame=forbid)
    assert (tmp_path / "segment_001" / "segment.json").read_bytes() == before

    replaced = export_segments(selected, tmp_path, extract_flame=fake_flame, overwrite=True)
    assert replaced[0].error is None


def test_no_segments_creates_no_outputs(tmp_path: Path) -> None:
    output = tmp_path / "dataset"

    assert export_segments((), output, extract_flame=fake_flame) == ()
    assert not output.exists()


def test_keyboard_interrupt_is_not_recorded_as_a_segment_failure(
    tmp_path: Path, source_video: Path
) -> None:
    def interrupt(sequence: TalkingFaceSequence) -> FlameResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        export_segments(segments(source_video, (0.5, 1.3)), tmp_path, extract_flame=interrupt)
    assert not (tmp_path / "segment_001").exists()


def test_failed_overwrite_does_not_leave_a_stale_complete_segment(
    tmp_path: Path, source_video: Path
) -> None:
    selected = segments(source_video, (0.5, 1.3))
    export_segments(selected, tmp_path, extract_flame=fake_flame)

    def fail(sequence: TalkingFaceSequence) -> FlameResult:
        raise RuntimeError("simulated reconstruction failure")

    exports = export_segments(selected, tmp_path, extract_flame=fail, overwrite=True)

    assert exports[0].error is not None
    assert not (tmp_path / "segment_001").exists()
