"""Decode a saved FLAME result and write an offline interactive mesh animation.

Run from the repository root with local FLAME2020/SMIRK eyelid assets and the rendering extra.
No encoder, MediaPipe session, GPU or network access is required. --with-video references the
original local video beside the mesh, using its playback clock.
"""

import argparse
from pathlib import Path

from talkingfacekit.flame import FlameResult
from talkingfacekit.integrations.flame import FlameDecoder
from talkingfacekit.rendering import render_face_mesh_html


def main() -> None:
    """Load compact saved parameters, generate geometry and render the original frame timeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/flame2020/generic_model.pkl"))
    parser.add_argument("--smirk-eyelids", type=Path, default=Path("models/smirk"))
    parser.add_argument("--output", type=Path, default=Path("reports/smirk/flame-mesh.html"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--with-video", action="store_true", help="Compare against the original video"
    )
    parser.add_argument("--video-time-offset", type=float, default=0.0)
    args = parser.parse_args()
    if args.video_time_offset != 0 and not args.with_video:
        parser.error("--video-time-offset requires --with-video")
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; choose another path or pass --overwrite")
    result = FlameResult.load(args.parameters)
    decoder = FlameDecoder(args.model, smirk_eyelids=args.smirk_eyelids)
    mesh = decoder.decode(result, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = render_face_mesh_html(
        mesh,
        args.output,
        title="TalkingFaceKit · FLAME2020 / SMIRK",
        overwrite=args.overwrite,
        source_video=result.source.path if args.with_video else None,
        video_time_offset_seconds=args.video_time_offset,
    )
    print(f"Model SHA-256: {decoder.provenance['model_sha256']}")
    print(f"Geometry: {mesh.vertex_count} vertices, {mesh.triangle_count} triangles")
    print(
        f"Frames: {mesh.frame_count}; estimated={result.estimated_count}; missing={mesh.frame_count - result.estimated_count}"
    )
    print(f"Animation: {output.resolve()}")


if __name__ == "__main__":
    main()
