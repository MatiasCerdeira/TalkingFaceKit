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
convenient API but delegate file and framework work to integration modules. Future expensive
operations such as decoding or tracking will be explicit methods. Integrations compute typed
results first, and sequence methods attach them only after success so failures do not leave partial
state. Metadata and future result records remain immutable where practical; integrations must not
mutate sequence attributes directly.

Landmark tracking follows this pattern through `sequence.track_landmarks(tracker, name=...)`. The
sequence supplies its path and interval to a small backend contract, then owns the completed result.
Names make multiple backends or configurations comparable without coupling the aggregate to their
implementation details. Replacement is explicit, and a backend failure leaves the existing mapping
unchanged.

## Intended package boundaries

Create these modules only when real code needs them:

```text
src/talkingfacekit/
├── metadata.py       Backend-independent metadata value types
├── sequence.py       User-facing sequence aggregate
├── io/
│   └── video.py      PyAV-based video inspection boundary
└── tracking/
    ├── landmarks.py  Backend-independent landmark result and tracker contract
    └── mediapipe.py  Optional MediaPipe/PyAV streaming adapter
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

`FaceLandmarkTrack` currently establishes the landmark timeline contract:

- `frame_indices`: strictly increasing, zero-based source decode indices with dtype `int64`;
- `timestamps_seconds`: strictly increasing source presentation timestamps with dtype `float64`;
- `landmarks`: shape `(frame_count, landmark_count, 3)` with dtype `float32`;
- `detected`: one boolean per frame; missing detections remain aligned and contain only `NaN`;
- `topology` and `coordinate_system`: explicit strings identifying point ordering and coordinate
  meaning.

The MediaPipe adapter uses its 478-point topology. Its x and y values are normalized image
coordinates and z is MediaPipe-relative depth. It processes the half-open sequence interval
`[start_seconds, end_seconds)` using original presentation timestamps. RGB arrays exist only while a
single frame is being submitted to the tracker; they are not part of the sequence data model.
TalkingFaceKit does not download or bundle model assets.

Canonical video layout, color order, audio layout, facial-parameter schema, and timestamp semantics
remain open decisions. They must be documented here before becoming public contracts.

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

| Decision | Rationale |
| --- | --- |
| Python 3.11 | Stable shared baseline for the team. |
| `uv` with a committed lockfile | Reproducible environments across macOS and Windows. |
| `src/` package layout | Prevents accidental imports from the repository root. |
| NumPy as the core numerical representation | Framework-independent arrays and NPZ support. |
| Ruff, mypy strict mode, and pytest | Automated style, typing, and behavior checks. |
| Backend-independent core | Trackers and media frameworks can change without rewriting domain types. |
| Mutable sequence aggregate | One user-facing object coordinates explicit operations and owns their results. |
| PyAV isolated under `io` | `from_video` delegates media inspection without embedding PyAV logic in the aggregate. |
| Optional MediaPipe landmark backend | Provides the first local, cross-platform tracking slice without making it a core dependency. |
| Named transactional landmark results | Supports comparisons and prevents failed work from leaving partial sequence state. |
| Stream frames at tracking boundaries | Computer-vision backends receive RGB pixels without retaining an uncompressed video array. |

## Pending decisions

- Canonical video array shape and RGB/BGR color order.
- Canonical audio layout, dtype, amplitude range, and channel convention.
- Cross-modal timestamp and synchronization representation beyond landmark source timestamps.
- Facial-animation parameter schema and FLAME conventions.
- Serialization formats and versioning policy.
- Optional dependency groups for future audio, FLAME, and speech backends.
