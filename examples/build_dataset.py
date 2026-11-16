"""Analyze one video and export each selected segment's MP4, FLAME parameters and metadata.

Run from the repository root with the active-speaker and tracking-smirk extras and local models
already prepared; --render also needs the rendering extra and the FLAME2020 assets. Nothing is
downloaded. Each segment is written to OUTPUT_DIR/segment_NNN/ as clip.mp4, flame.npz and
segment.json; --render adds comparison.html, the FLAME mesh beside its clip.
"""

import argparse
from pathlib import Path

from talkingfacekit import AnalysisConfig, TalkingFaceSequence, VisualPolicy
from talkingfacekit.dataset import SegmentExport, export_segments
from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.flame import FlameDecoder
from talkingfacekit.integrations.smirk import SmirkEncoder
from talkingfacekit.rendering import render_face_mesh_html


def main() -> None:
    """Run analysis, segment export and optional rendering, printing one line per segment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="Source start in seconds")
    parser.add_argument("--end", type=float, help="Source end in seconds; default is EOF")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smirk/SMIRK_em1.pt"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--syncnet-model", type=Path, help="Enable optional A/V verification")
    parser.add_argument("--render", action="store_true", help="Write comparison.html per segment")
    parser.add_argument(
        "--flame-model", type=Path, default=Path("models/flame2020/generic_model.pkl")
    )
    parser.add_argument("--smirk-eyelids", type=Path, default=Path("models/smirk"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    existing = sorted(args.output_dir.glob("segment_*"))
    if existing and not args.overwrite:
        parser.error("output_dir already contains segments; choose another path or --overwrite")

    # Load every model before the slow analysis so missing assets fail immediately.
    encoder = SmirkEncoder(args.checkpoint, device=args.device)
    decoder = (
        FlameDecoder(args.flame_model, smirk_eyelids=args.smirk_eyelids) if args.render else None
    )
    sequence = TalkingFaceSequence.from_video(args.video).clip(args.start, args.end)
    result = sequence.analyze(
        AnalysisConfig(visual_policy=VisualPolicy(), syncnet_model=args.syncnet_model)
    )
    print(f"Selected segments: {len(result.selected_segments)}")
    exports = export_segments(
        result.selected_segments,
        args.output_dir,
        extract_flame=lambda clip: encoder.extract(clip, landmarker_model=args.landmarker),
        overwrite=args.overwrite,
    )
    for item in exports:
        print(_describe(item))
        if decoder is not None and item.flame_path is not None and item.clip_path is not None:
            mesh = decoder.decode(FlameResult.load(item.flame_path))
            html = render_face_mesh_html(
                mesh,
                item.directory / "comparison.html",
                title=f"{args.video.name} {item.segment.start_seconds:.3f}s",
                overwrite=args.overwrite,
                source_video=item.clip_path,
                video_time_offset_seconds=item.segment.start_seconds,
            )
            print(f"  render -> {html}")
    stale = [path for path in existing if path.name > f"segment_{len(exports):03d}"]
    if stale:
        print(f"Warning: earlier runs left {len(stale)} extra segment directories: {stale}")
    failed = sum(item.error is not None for item in exports)
    print(f"Summary: exported={len(exports) - failed} failed={failed} -> {args.output_dir}")


def _describe(item: SegmentExport) -> str:
    """Format the source interval, face and outcome of one segment."""
    segment = item.segment
    prefix = (
        f"{item.directory.name} {segment.start_seconds:.3f}-{segment.end_seconds:.3f}s "
        f"face={segment.face_id}"
    )
    if item.error is not None:
        return (
            f"{prefix} FAILED at {item.error.stage}: {item.error.error_type}: {item.error.message}"
        )
    assert item.flame_path is not None
    flame = FlameResult.load(item.flame_path)
    return f"{prefix} frames={flame.frame_count} estimated={flame.estimated_count}"


if __name__ == "__main__":
    main()
