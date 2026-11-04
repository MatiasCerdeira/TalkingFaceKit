"""Inspect raw SMIRK parameters from a short source-video interval.

Run from the repository root after installing the tracking-smirk extra and downloading the
official checkpoint. With --output it saves compact interval results, never media or meshes.
"""

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np

from talkingfacekit import TalkingFaceSequence
from talkingfacekit.integrations.smirk import SmirkEncoder, SmirkMissingFrame


def main() -> None:
    """Read explicit local assets and print source-aligned parameter diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smirk/SMIRK_em1.pt"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--start", type=float, default=20.0)
    parser.add_argument("--end", type=float, default=20.1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Save a complete interval as a versioned .npz")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.output is None:
        parser.error("--overwrite requires --output")
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error("output already exists; choose another path or pass --overwrite")

    clip = TalkingFaceSequence.from_video(args.video).clip(args.start, args.end)
    encoder = SmirkEncoder(args.checkpoint, device=args.device)
    print(f"Checkpoint SHA-256: {encoder.checkpoint_sha256}")
    if args.output is not None:
        result = encoder.extract(clip, landmarker_model=args.landmarker)
        saved = result.save(args.output, overwrite=args.overwrite)
        for index in np.flatnonzero(~result.estimated):
            print(
                f"frame={result.frame_indices[index]} time={result.timestamps_seconds[index]:.6f} "
                f"status=missing reason={result.missing_reasons[index]}"
            )
        print(
            f"Summary: frames={result.frame_count} estimated={result.estimated_count} "
            f"missing={result.frame_count - result.estimated_count}"
        )
        print(f"Saved: {saved}")
        return
    estimated = 0
    missing = 0
    with closing(encoder.track(clip, landmarker_model=args.landmarker)) as predictions:
        for result in predictions:
            if isinstance(result, SmirkMissingFrame):
                missing += 1
                print(
                    f"frame={result.frame_index} time={result.timestamp_seconds:.6f} "
                    f"status=missing reason={result.reason}"
                )
                continue
            estimated += 1
            parameters = result.parameters
            print(
                f"frame={result.frame_index} time={result.timestamp_seconds:.6f} "
                "status=estimated "
                f"shape={parameters.shape.shape} expression={parameters.expression.shape} "
                f"expression_norm={np.linalg.norm(parameters.expression):.6f} "
                f"jaw={parameters.jaw_pose.tolist()} eyelids={result.eyelids.tolist()}"
            )
    print(f"Summary: frames={estimated + missing} estimated={estimated} missing={missing}")


if __name__ == "__main__":
    main()
