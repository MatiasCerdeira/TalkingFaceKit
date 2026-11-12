"""Opt-in real pipeline: DeepTalk selection, MP4 export and SMIRK parameters per segment."""

import json
import os
from pathlib import Path

import av
import numpy as np
import pytest

from talkingfacekit import AnalysisConfig, TalkingFaceSequence, VisualPolicy
from talkingfacekit.dataset import export_segments
from talkingfacekit.flame import FlameResult

pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "TALKINGFACEKIT_DATASET_VIDEO",
            "TALKINGFACEKIT_SMIRK_CHECKPOINT",
            "TALKINGFACEKIT_SMIRK_LANDMARKER",
        )
    ),
    reason="set TALKINGFACEKIT_DATASET_VIDEO, SMIRK_CHECKPOINT and SMIRK_LANDMARKER",
)


def test_real_segments_link_clip_frames_to_source_time_parameters(tmp_path: Path) -> None:
    from talkingfacekit.integrations.smirk import SmirkEncoder

    video = Path(os.environ["TALKINGFACEKIT_DATASET_VIDEO"])
    start = float(os.environ.get("TALKINGFACEKIT_DATASET_START_SECONDS", "0"))
    end_text = os.environ.get("TALKINGFACEKIT_DATASET_END_SECONDS")
    sequence = TalkingFaceSequence.from_video(video).clip(
        start, float(end_text) if end_text else None
    )
    result = sequence.analyze(AnalysisConfig(visual_policy=VisualPolicy()))
    assert result.selected_segments, "choose a video interval with at least one selected segment"
    encoder = SmirkEncoder(os.environ["TALKINGFACEKIT_SMIRK_CHECKPOINT"])
    landmarker = os.environ["TALKINGFACEKIT_SMIRK_LANDMARKER"]

    exports = export_segments(
        result.selected_segments,
        tmp_path,
        extract_flame=lambda clip: encoder.extract(clip, landmarker_model=landmarker),
    )

    for item in exports:
        assert item.error is None, item.error
        assert item.metadata_path is not None
        metadata = json.loads(item.metadata_path.read_text("utf-8"))
        flame = FlameResult.load(item.directory / metadata["flame"]["path"])
        assert flame.source.path.resolve() == video.resolve()
        assert flame.provenance["backend"] == "smirk"
        assert flame.estimated_count > 0
        assert metadata["face_id"] == item.segment.face_id
        with av.open(item.directory / metadata["clip"]["path"]) as container:
            clip_times = np.array([float(frame.time) for frame in container.decode(video=0)])
        assert clip_times.size == flame.frame_count
        np.testing.assert_allclose(
            clip_times + metadata["clip"]["source_time_offset_seconds"],
            flame.timestamps_seconds,
            atol=1e-3,
        )
