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
video/audio/model integrations
              |
              v
       core sequence model
              |
              v
       NumPy and Python types
```

Core modules must not import OpenCV, PyTorch, tracker implementations, speech models, or FFmpeg
wrappers. Integration modules may translate those frameworks into core Python and NumPy types.

## Intended package boundaries

Create these modules only when real code needs them:

```text
src/talkingfacekit/
├── sequence.py       Core sequence and metadata types
├── validation.py     Backend-independent invariant checks
├── io/               Video, audio, and serialization boundaries
└── tracking/         Tracker interfaces and implementations
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

## Pending decisions

- Canonical video array shape and RGB/BGR color order.
- Canonical audio layout, dtype, amplitude range, and channel convention.
- Timestamp and synchronization representation.
- Facial-animation parameter schema and FLAME conventions.
- Serialization formats and versioning policy.
- Optional dependency groups for video, audio, tracking, and speech backends.
