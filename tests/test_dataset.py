import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from talkingfacekit import (
    ActiveSpeakerProvenance,
    ActiveSpeakerResult,
    ActiveSpeakerScoreWindow,
    CandidateRun,
    TalkingFaceCollection,
    TalkingFaceSegment,
    TalkingFaceSequence,
    stream_video_frames,
)
from talkingfacekit.dataset import build_dataset, export_segments
from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations import deeptalk

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


def selection(*bounds: tuple[float, float]) -> ActiveSpeakerResult:
    """Return DeepTalk evidence whose structural candidates are the given source intervals."""
    return ActiveSpeakerResult(
        (),
        (),
        (ActiveSpeakerScoreWindow(0, 3, {1: 2.0}),),
        ActiveSpeakerProvenance("0.3.1"),
        candidate_runs=tuple(
            CandidateRun(
                start, end, "candidate", 1, ("structural_policy_passed",), 1, 20, 20, 1, 0, 2, 2, 2
            )
            for start, end in bounds
        ),
    )


@pytest.fixture
def folder(tmp_path: Path, source_video: Path) -> Path:
    """Two decodable videos and one corrupt file, discovered as a, b (corrupt), c."""
    videos = tmp_path / "videos"
    videos.mkdir()
    shutil.copyfile(source_video, videos / "a.mkv")
    (videos / "b.mp4").write_bytes(b"corrupt")
    shutil.copyfile(source_video, videos / "c.mkv")
    return videos


def test_folder_dataset_links_videos_segments_and_failures(
    tmp_path: Path, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deeptalk, "analyze_sequence", lambda *args, **kwargs: selection((0.4, 1.2), (1.6, 2.4))
    )
    output = tmp_path / "dataset"

    videos = build_dataset(
        TalkingFaceCollection.from_folder(folder), output, extract_flame=fake_flame
    )

    assert [item.source.name for item in videos] == ["a.mkv", "b.mp4", "c.mkv"]
    assert [item.directory.name for item in videos] == ["0001-a", "0002-b", "0003-c"]
    assert videos[1].error is not None and videos[1].segments == ()
    assert not videos[1].directory.exists()
    document = json.loads((output / "dataset.json").read_text("utf-8"))
    assert document["kind"] == "talkingfacekit.dataset"
    assert document["schema_version"] == 1
    assert (document["total_videos"], document["completed_videos"]) == (3, 3)
    assert (document["failed_videos"], document["exported_segments"]) == (1, 4)
    assert document["failed_segments"] == 0
    entries = document["videos"]
    assert [entry["status"] for entry in entries] == ["completed", "failed", "completed"]
    assert entries[1]["error"]["type"] == "ValueError"
    assert entries[1]["analysis"] is None and entries[1]["segments"] == []
    for entry, video in zip(entries, ("a.mkv", "b.mp4", "c.mkv"), strict=True):
        assert entry["source"] == str((folder / video).resolve())
    for entry in (entries[0], entries[2]):
        assert (output / entry["analysis"]).is_file()
        bounds = [(item["start_seconds"], item["end_seconds"]) for item in entry["segments"]]
        assert bounds == [(0.4, 1.2), (1.6, 2.4)]
        for item in entry["segments"]:
            assert item["error"] is None and item["face_id"] == 1
            metadata = json.loads((output / item["metadata"]).read_text("utf-8"))
            assert metadata["source"] == entry["source"]
            assert (metadata["start_seconds"], metadata["end_seconds"]) == (
                item["start_seconds"],
                item["end_seconds"],
            )
            segment_directory = (output / item["metadata"]).parent
            assert segment_directory == output / item["directory"]
            flame = FlameResult.load(segment_directory / metadata["flame"]["path"])
            assert flame.source.path.resolve() == Path(entry["source"])
            clip_times = clip_timestamps(segment_directory / metadata["clip"]["path"])
            np.testing.assert_allclose(
                clip_times + metadata["clip"]["source_time_offset_seconds"],
                flame.timestamps_seconds,
                atol=1e-3,
            )


def test_folder_dataset_records_segment_failures_and_continues(
    tmp_path: Path, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deeptalk, "analyze_sequence", lambda *args, **kwargs: selection((0.4, 1.2), (1.6, 2.4))
    )
    calls = 0

    def fail_first(sequence: TalkingFaceSequence) -> FlameResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated reconstruction failure")
        return fake_flame(sequence)

    output = tmp_path / "dataset"
    build_dataset(TalkingFaceCollection.from_folder(folder), output, extract_flame=fail_first)

    document = json.loads((output / "dataset.json").read_text("utf-8"))
    assert (document["exported_segments"], document["failed_segments"]) == (3, 1)
    failed = document["videos"][0]["segments"][0]
    assert failed["metadata"] is None
    assert failed["error"] == {
        "stage": "flame",
        "type": "RuntimeError",
        "message": "simulated reconstruction failure",
    }
    assert not (output / failed["directory"]).exists()


def test_interrupted_folder_keeps_completed_videos_in_dataset_json(
    tmp_path: Path, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deeptalk, "analyze_sequence", lambda *args, **kwargs: selection((0.4, 1.2)))
    seen: list[Path] = []

    def interrupt_on_last_video(sequence: TalkingFaceSequence) -> FlameResult:
        seen.append(sequence.source.path)
        if sequence.source.path.name == "c.mkv":
            raise KeyboardInterrupt
        return fake_flame(sequence)

    output = tmp_path / "dataset"
    with pytest.raises(KeyboardInterrupt):
        build_dataset(
            TalkingFaceCollection.from_folder(folder),
            output,
            extract_flame=interrupt_on_last_video,
        )

    document = json.loads((output / "dataset.json").read_text("utf-8"))
    assert (document["total_videos"], document["completed_videos"]) == (3, 2)
    assert document["exported_segments"] == 1
    assert (output / document["videos"][0]["segments"][0]["metadata"]).is_file()
    assert not (output / "0003-c" / "segment_001").exists()


def test_existing_dataset_is_refused_before_analysis_unless_overwrite(
    tmp_path: Path, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def analyze(sequence: TalkingFaceSequence, **kwargs: object) -> ActiveSpeakerResult:
        calls.append(sequence.source.path)
        return selection((0.4, 1.2))

    monkeypatch.setattr(deeptalk, "analyze_sequence", analyze)
    collection = TalkingFaceCollection.from_folder(folder)
    output = tmp_path / "dataset"
    build_dataset(collection, output, extract_flame=fake_flame)
    before = (output / "dataset.json").read_bytes()

    with pytest.raises(FileExistsError):
        build_dataset(collection, output, extract_flame=fake_flame)
    assert len(calls) == 2
    assert (output / "dataset.json").read_bytes() == before

    build_dataset(collection, output, extract_flame=fake_flame, overwrite=True)
    assert len(calls) == 4


def test_empty_folder_writes_an_empty_dataset_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*args: object, **kwargs: object) -> ActiveSpeakerResult:
        raise AssertionError("no model may run")

    monkeypatch.setattr(deeptalk, "analyze_sequence", forbid)
    empty = tmp_path / "videos"
    empty.mkdir()

    assert (
        build_dataset(
            TalkingFaceCollection.from_folder(empty), tmp_path / "dataset", extract_flame=fake_flame
        )
        == ()
    )
    document = json.loads((tmp_path / "dataset" / "dataset.json").read_text("utf-8"))
    assert (document["total_videos"], document["completed_videos"]) == (0, 0)
    assert document["videos"] == []
