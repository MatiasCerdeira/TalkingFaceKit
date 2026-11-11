"""Analyze one video and export every selected segment as an MP4 clip.

Run from the repository root with the active-speaker extra and local models already prepared. The
printed source intervals make it easy to compare each exported clip against the original video.
"""

import argparse
from pathlib import Path

from talkingfacekit import AnalysisConfig, TalkingFaceSequence, VisualPolicy


def main() -> None:
    """Analyze one source and export its selected source-time intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--syncnet-model", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sequence = TalkingFaceSequence.from_video(args.video)
    result = sequence.analyze(
        AnalysisConfig(
            visual_policy=VisualPolicy(),
            syncnet_model=args.syncnet_model,
        )
    )
    if not result.selected_segments:
        print("No selected segments")
        return

    for index, segment in enumerate(result.selected_segments, start=1):
        output = args.output_dir / f"{args.video.stem}-segment-{index:03d}.mp4"
        segment.to_sequence().export_clip(output, overwrite=args.overwrite)
        print(
            f"{segment.start_seconds:.3f}-{segment.end_seconds:.3f}s "
            f"({segment.duration_seconds:.3f}s) -> {output}"
        )


if __name__ == "__main__":
    main()
