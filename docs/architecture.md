# TalkingFaceKit Architecture

This is a living document for decisions that affect the structure or data contracts of the
library. Update it in the same change that introduces or modifies an architectural decision.

## Goal

TalkingFaceKit provides a reusable representation and processing pipeline for talking-face video
sequences. A sequence may combine video frames, audio samples, facial-animation parameters,
timestamps, and metadata without being tied to one dataset, tracker, or speech model.

## Design principles

1. Keep the domain model independent from external processing frameworks.
2. Make data contracts explicit: shapes, dtypes, units, ranges, channel order, and time bases.
3. Keep side effects and expensive operations at integration boundaries.
4. Support replaceable backends through small interfaces only when multiple implementations are
   needed.
5. Build and validate one small end-to-end workflow before generalizing the architecture.

## Dependency direction

Dependencies point toward the core:

```text
TalkingFaceSequence facade ---> video/audio/model integrations
             |                              |
             +--------------+---------------+
                            v
               core Python and NumPy values
```

Core data types must not contain OpenCV, PyTorch, PyAV, tracker, speech-model, or FFmpeg objects.
Direct construction must not access the filesystem. Integration modules translate external
frameworks into core Python and NumPy types; user-facing sequence methods may delegate to those
boundaries without containing backend-specific logic.

## Sequence aggregate pattern

`TalkingFaceSequence` is the mutable, user-facing aggregate for the data and operations associated
with one sequence. Alternate constructors such as `TalkingFaceSequence.from_video(path)` provide a
convenient API but delegate file and framework work to integration modules. Expensive operations
such as decoding and tracking remain explicit. Integrations compute typed results first, and
sequence methods attach them only after success so failures do not leave partial state. Metadata
and future result records remain immutable where practical; integrations must not mutate sequence
attributes directly.

The represented interval stays on the source-media timeline. Its start must be finite and
non-negative; its optional end must be finite and strictly greater than its start.
`TalkingFaceSequence.clip(...)` returns a new lightweight sequence contained within the current
interval. It reuses the source path and metadata, preserves source timestamps, performs no media
I/O, and does not copy attached result tracks.

Landmark tracking follows this pattern through `sequence.track_landmarks(tracker, name=...)`. The
sequence supplies its path and interval to a small backend contract, then owns the completed result.
Names make multiple backends or configurations comparable without coupling the aggregate to their
implementation details. Replacement is explicit, and a backend failure leaves the existing mapping
unchanged. Tracker implementations consume the shared video-frame stream instead of opening PyAV
containers themselves.

## Intended package boundaries

Create these modules only when real code needs them:

```text
src/talkingfacekit/
├── metadata.py       Backend-independent metadata value types
├── mesh.py           Backend-independent animated triangular-mesh contract
├── video.py          Backend-independent streamed-frame contract
├── rendering/
│   └── plotly.py      Optional offline interactive HTML renderer
├── sequence.py       User-facing sequence aggregate
├── io/
│   ├── landmarks.py  Versioned NPZ landmark persistence boundary
│   └── video.py      PyAV-based video inspection and RGB streaming boundary
└── tracking/
    ├── landmarks.py  Backend-independent landmark result and tracker contract
    ├── mediapipe.py  Optional MediaPipe landmark adapter
    └── mediapipe_mesh.py  MediaPipe landmark-to-surface conversion boundary
```

Avoid empty directories and placeholder abstractions. The first implementation should remain small.

## Data contracts

- Use NumPy arrays for framework-independent numerical data.
- Use `numpy.typing.NDArray` aliases to express array dtypes in public type signatures.
- Document and validate array shape, dtype, channel order, units, valid range, and time axis.
- Represent filesystem paths with `pathlib.Path` at public filesystem boundaries.
- Use seconds for public durations and timestamps unless an API explicitly declares another unit.
- Never infer or silently change FPS, sample rate, color order, or synchronization metadata.
- Keep backend-specific tensors and objects outside the core model.

`VideoMetadata` requires positive encoded width and height. Average FPS and stream duration may be
unknown, represented by `None`; when present, both values must be finite and positive.

`DecodedVideoFrame` establishes the shared streaming contract:

- `frame_index`: non-negative, zero-based source decode index;
- `timestamp_seconds`: finite source presentation timestamp in seconds;
- `rgb`: shape `(height, width, 3)` with positive spatial dimensions and dtype `uint8`;
- channel order is explicitly RGB and values use the inclusive range `[0, 255]`.

`stream_video_frames` owns PyAV container access, selects the first video stream, processes the
half-open interval `[start_seconds, end_seconds)`, converts frames explicitly to RGB, and requires
strictly increasing presentation timestamps. It yields one `DecodedVideoFrame` at a time and does
not retain previous arrays. Source FPS is metadata only and is never used to synthesize timestamps.
Callers decide whether to retain yielded pixels and therefore own any resulting memory growth.
Computer-vision backends should consume this boundary instead of duplicating PyAV access, interval
filtering, color conversion, or timestamp validation.

`FaceLandmarkTrack` currently establishes the landmark timeline contract:

- `frame_indices`: strictly increasing, zero-based source decode indices with dtype `int64`;
- `timestamps_seconds`: strictly increasing source presentation timestamps with dtype `float64`;
- `landmarks`: shape `(frame_count, landmark_count, 3)` with dtype `float32`;
- `detected`: one boolean per frame; missing detections remain aligned and contain only `NaN`;
- `topology` and `coordinate_system`: explicit strings identifying point ordering and coordinate
  meaning.

The MediaPipe adapter uses its 478-point topology. Its x and y values are normalized image
coordinates and z is MediaPipe-relative depth. It consumes `DecodedVideoFrame` records from the
shared stream and performs only MediaPipe-specific timestamp conversion and inference. In this
workflow, an RGB array is retained only while its frame is submitted to the tracker; pixels are not
part of the sequence data model. TalkingFaceKit does not download or bundle model assets.

`FaceMeshTrack` establishes the animated triangular-surface contract:

- `frame_indices`, `timestamps_seconds`, and `detected` retain the source landmark timeline;
- `vertices`: shape `(frame_count, vertex_count, 3)` with dtype `float32`;
- `triangles`: shape `(triangle_count, 3)` with dtype `int32`, fixed across all frames and containing
  valid, non-degenerate vertex indices;
- detected vertex rows contain finite coordinates and undetected rows contain only `NaN`;
- `topology` and `coordinate_system` explicitly identify connectivity and coordinate meaning.

The core mesh contract does not depend on MediaPipe or a renderer. The MediaPipe conversion
boundary maps landmarks 0 through 467 to the official 852-triangle facial tessellation; iris
landmarks 468 through 477 remain separate because they are contours rather than tessellated skin.
The conversion requires explicit source width and height, centers normalized coordinates, flips the
vertical axis, and aspect-corrects horizontal position and relative depth into image-height units.
It preserves MediaPipe-relative depth and therefore does not claim metric 3D reconstruction.

Rendering is a separate optional integration. The Plotly renderer consumes only `FaceMeshTrack`,
applies a display-only depth multiplier to a copy of z coordinates, and writes a self-contained
offline HTML file with a neutral material, virtual lighting, an orbital camera, and timeline
controls. It does not decode source pixels or add color data to the mesh contract. Original
timestamps remain visible; automatic playback uses their median interval because Plotly accepts one
display duration for the complete animation.

Completed landmark tracks may be persisted through `save_landmark_track` and restored through
`load_landmark_track`. The compressed NPZ schema is versioned independently from the Python package
and stores tracker provenance, topology, coordinate-system description, frame indices, original
timestamps, the complete landmark tensor, and its detection mask. Loading reconstructs a
`FaceLandmarkTrack`, so all current data-contract validation is applied again. Saving completes a
temporary archive before replacing the destination, and replacement must be requested explicitly.

Materialized multi-frame video arrays, audio layout, facial-parameter schemas, and cross-modal
timestamp semantics remain open decisions. They must be documented here before becoming public
contracts.

## Error handling

- Reject invalid inputs at boundaries with clear exceptions.
- Include the violated invariant and relevant observed value in error messages.
- Do not return `None` to hide processing or validation failures.
- Define domain-specific exception classes only when callers need to distinguish failure categories.

## Testing strategy

- Unit tests cover the core model and validation with small synthetic arrays.
- Integration tests cover external programs and optional backends separately.
- Tests must not download models or datasets automatically.
- A bug fix includes a regression test that fails without the fix.
- Cross-platform code must avoid assumptions about path separators, shell syntax, and hardware.

## Current decisions

| Decision                                   | Rationale                                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Python 3.11                                | Stable shared baseline for the team.                                                                   |
| `uv` with a committed lockfile             | Reproducible environments across macOS and Windows.                                                    |
| `src/` package layout                      | Prevents accidental imports from the repository root.                                                  |
| NumPy as the core numerical representation | Framework-independent arrays and NPZ support.                                                          |
| Ruff, mypy strict mode, and pytest         | Automated style, typing, and behavior checks.                                                          |
| Backend-independent core                   | Trackers and media frameworks can change without rewriting domain types.                               |
| Mutable sequence aggregate                 | One user-facing object coordinates explicit operations and owns their results.                         |
| PyAV isolated under `io`                   | Metadata inspection and frame decoding share one media boundary instead of leaking into trackers.      |
| Optional MediaPipe landmark backend        | Provides the first local, cross-platform tracking slice without making it a core dependency.           |
| Named transactional landmark results       | Supports comparisons and prevents failed work from leaving partial sequence state.                     |
| Shared RGB frame stream                    | Backends reuse source indices, timestamps, intervals, and RGB conversion without materializing video.   |
| Versioned NPZ landmark archives            | Makes expensive tracking results reusable while preserving NumPy dtypes and timeline alignment.        |
| Backend-independent animated mesh contract | Makes triangle geometry reusable by renderers, exporters, and future model-fitting backends.           |
| MediaPipe 468-vertex surface conversion    | Reuses the official 852-triangle topology while keeping relative-depth limitations explicit.           |
| Plotly as optional HTML renderer           | Provides a portable interactive demonstration without coupling core mesh data to a graphics framework. |

## Pending decisions

- Whether a materialized multi-frame video array should become a public contract.
- Alternative pixel formats or decoding backends beyond the canonical RGB `uint8` frame stream.
- Canonical audio layout, dtype, amplitude range, and channel convention.
- Cross-modal timestamp and synchronization representation beyond landmark source timestamps.
- Facial-animation parameter schema and FLAME conventions.
- Serialization formats and versioning policy for data other than landmark tracks.
- Optional dependency groups for future audio, FLAME, and speech backends.
