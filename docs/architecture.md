# TalkingFaceKit Architecture

This is a living document for decisions that affect the structure or data contracts of the
library. Update it in the same change that introduces or modifies an architectural decision.

## Goal

TalkingFaceKit analyzes arbitrary video sources and produces explainable source-timestamp intervals
that are useful for talking-face datasets. The central question is which visible face, if any, is
producing audible speech. Camera orientation, visual quality, and audiovisual synchronization are
separate observations that later policy can use to accept, reject, or mark an interval uncertain.

The library owns media timelines, backend-independent results, temporal aggregation, policy, and
reports. It does not own the pretrained perception models. Existing landmark, mesh, and rendering
features remain supported as optional visual capabilities rather than the primary product workflow.

## Design principles

1. Keep the domain model independent from external processing frameworks.
2. Make data contracts explicit: shapes, dtypes, units, ranges, channel order, and time bases.
3. Keep side effects and expensive operations at integration boundaries.
4. Support replaceable backends through small interfaces only when multiple implementations are
   needed.
5. Build and validate one small end-to-end workflow before generalizing the architecture.
6. Preserve raw model observations and uncertainty; do not present an uncalibrated score as a
   probability or silently collapse ambiguous/off-screen speech into a boolean.

## Current delivery direction

The first client-visible workflow analyzes one video before any collection or batch abstraction is
built:

```text
VideoSource + source interval
        |
        +--> PyAV video frames with source PTS
        +--> PyAV audio chunks with source timestamps       [implemented]
                         |
                         v
             experimental DeepTalk-ASD adapter
       face tracks + speech intervals + raw ASD scores
                         |
                         v
              TalkingFaceKit segment policy
       candidate / rejected / uncertain + reasons
                         |
                         v
              JSON report + diagnostic overlay
```

DeepTalk-ASD 0.3.1 is the selected **experimental** first backend because the compatibility spike
proved that its InspireFace tracking, Silero VAD, and LR-ASD ONNX path work locally on the project's
Python 3.11 Apple Silicon environment. It remains behind a narrow integration boundary. Its objects,
synthetic time assumptions, buffering behavior, and score interpretation must not leak into core
contracts. The initial thesis slice disables speaker embeddings on macOS and records that capability
as unavailable rather than silently pretending it ran.

The experimental choice is not a permanent commitment. Evaluation on representative labeled clips
provides an explicit gate:

1. keep the DeepTalk adapter if the model behavior is useful and the wrapper is manageable;
2. adapt the same LR-ASD ONNX models more directly if the model is useful but the wrapper is the
   problem;
3. compare TalkNet only if LR-ASD quality is inadequate.

MediaPipe is not the active-speaker engine. It is the planned later source of head orientation and
visual-suitability observations. Exact A/V offset is also not inferred from ASD scores; a separate
SyncNet evaluation follows the active-speaker milestone. Folder discovery, collection results, and
batch execution come after the one-video result model has been validated.

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

`VideoSource` is the immutable identity of source media: its path and `VideoMetadata` always refer
to the complete primary video stream. `TalkingFaceSequence` is the user-facing aggregate for one
temporal selection and its results. Multiple sequences may share one source while owning distinct
intervals and result mappings. Source and interval fields remain immutable after validation; named
result mappings are the aggregate's intentionally mutable state.

Alternate constructors such as `TalkingFaceSequence.from_video(path)` provide a convenient API but
delegate file and framework work to integration modules. Expensive operations such as decoding and
tracking remain explicit. Integrations compute typed results first, and sequence methods attach
them only after success so failures do not leave partial state.

The represented interval stays on the source-media timeline. Its start must be finite and
non-negative; its optional end must be finite and strictly greater than its start.
`TalkingFaceSequence.clip(...)` returns a new lightweight sequence contained within the current
interval. It reuses the same `VideoSource`, preserves source timestamps, performs no media I/O, and
does not copy attached result tracks. Its `duration_seconds` property describes only the declared
sequence interval; `source.metadata.stream_duration_seconds` remains metadata for the complete
stream.

A sequence created by `from_video` starts at zero and has `end_seconds=None`, meaning decode to
end-of-stream. Reported stream/container duration is informational metadata, not an absolute source
timestamp, so it is not used as the sequence end. This avoids truncating media whose first PTS is
positive or whose reported duration is approximate.

Landmark tracking follows this pattern through `sequence.track_landmarks(tracker, name=...)`. The
sequence supplies its source path and interval to a small backend contract, then owns the completed
result. Names make multiple backends or configurations comparable without coupling the aggregate
to their implementation details. Replacement is explicit, and a backend failure leaves the
existing mapping unchanged. Tracker implementations consume the shared video-frame stream instead
of opening PyAV containers themselves.

## Intended package boundaries

Create these modules only when real code needs them:

```text
src/talkingfacekit/
├── audio.py          Backend-independent decoded-audio chunk contract
├── metadata.py       Backend-independent metadata value types
├── mesh.py           Backend-independent animated triangular-mesh contract
├── video.py          Backend-independent source and streamed-frame contracts
├── integrations/
│   └── deeptalk.py    Experimental offline DeepTalk A/V adapter and copied results
├── rendering/
│   └── plotly.py      Optional offline interactive HTML renderer
├── sequence.py       User-facing sequence aggregate
├── io/
│   ├── audio.py      PyAV-based audio streaming boundary
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

`VideoMetadata` requires integer positive encoded width and height plus boolean audio presence.
Average FPS and stream duration may be unknown, represented by `None`; when present, both values
must be finite and positive.

`VideoSource` contains a normalized `Path` and the complete source-stream metadata. Constructing it
does not access the filesystem. A sequence and every clip derived from it share the same source;
the source never contains selection-specific duration or result tracks.

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

The experimental DeepTalk module implements its initial offline adaptation. It samples source PTS
onto a 25 Hz grid anchored at the sequence start and chooses the nearest decoded frame
deterministically. It does not create slots before the first or after the last decoded PTS. RGB
pixels become packed DeepTalk `RGB24` only at this boundary. Audio is downmixed and resampled
through PyAV/libswresample to mono 16 kHz signed `int16`. It is emitted as 480-sample frames plus a
final partial frame. Source PTS
quantization up to one millisecond is tolerated; larger gaps or overlaps fail because DeepTalk would
otherwise concatenate them silently.

Both streams use relative media time (`source_time - sequence.start_seconds`) only inside the
adapter. Audio timestamps identify each chunk's exclusive end. A lazy two-way merge feeds video
before audio on exact ties, with no realtime playback, clock, thread, or wall-clock timestamp.
DeepTalk is evaluated incrementally in roughly one-second windows before its ten-second audio buffer
can evict earlier samples. After each evaluation, mouth images older than the completed window are
pruned while a frame exactly on the shared boundary is retained. Results are copied into immutable,
validated integration-specific face observations, VAD intervals, score windows, provenance, and
diagnostic issue codes on the original source timeline; no DeepTalk profiles, images, audio frames,
or embeddings escape. Open-ended sequences report unequal A/V coverage and do not retain face
observations after audio coverage ends.

The stored window bounds are consecutive, but DeepTalk 0.3.1 internally uses an inclusive end for
video while audio remains half-open. A video frame exactly on a boundary may therefore contribute
to both adjacent backend evaluations; the adapter does not distort timestamps to conceal this.

DeepTalk 0.3.1 requires one localized private shim for offline use. After verifying the installed
version and expected face, VAD, and speaker-detector layouts, the adapter disables the redundant
float-based video throttle, wall-clock face/track expiry, and wall-clock VAD reset. Voiceprints are
explicitly disabled, their model is not acquired, and provenance reports speaker embeddings as
unavailable. A changed version or required private field fails clearly instead of applying the shim
speculatively. An explicitly configured InspireFace resource is hash-checked before native code
receives it; models resolved through DeepTalk continue to use its hash-verifying model manager.

A CPU smoke test has exercised this complete boundary with the official DeepTalk-ASD 0.3.1 models
and a 10.6-second H.264/AAC file containing one frontal speaker. It produced one stable face identity
and finite scores while preserving the observed A/V timeline. This validates runtime execution and
wiring, not model accuracy, score calibration, or multi-person tracking. DeepTalk still declares
`sherpa-onnx` as a package dependency, but this adapter neither acquires the WeSpeaker model nor
initializes its extractor because voiceprints are outside this MVP and have broken native linkage on
the tested macOS host.

DeepTalk and MediaPipe may be installed together for sequential pipeline stages. Their upstream
metadata installs `opencv-python` and `opencv-contrib-python` respectively; the lock keeps both on
the same OpenCV version, and a combined optional-backend test imports `cv2`, DeepTalk, and MediaPipe
in one environment. This is an upstream packaging constraint to revalidate on backend upgrades, not
a reason to merge their responsibilities: DeepTalk supplies active-speaker evidence while MediaPipe
continues to supply the exportable 478-point landmark topology.

`DecodedAudioChunk` establishes the shared audio-streaming contract:

- `start_sample_index`: non-negative index of the first sample in complete-stream decode order;
- `start_timestamp_seconds`: finite source timestamp of the first retained sample;
- `sample_rate_hz`: positive decoded sample rate, preserved without resampling;
- `channel_layout`: non-empty FFmpeg layout name, preserved without remixing;
- `samples`: C-contiguous shape `(sample_count, channel_count)` with dtype `float32`.

`stream_audio_chunks` owns PyAV audio access, selects the first audio stream, and processes the same
half-open `[start_seconds, end_seconds)` semantics as video. A decoded frame that crosses a boundary
is trimmed at sample precision. Integer PCM is explicitly normalized by its full-scale range;
floating-point PCM preserves decoded amplitudes and is not clipped. The integration does not
resample or remix. It yields one chunk at a time, rejects missing/invalid timestamps and audio
metadata, requires strictly increasing yielded chunk timestamps, and fails clearly when a source
has no audio or an interval contains no samples.

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

Materialized multi-frame video arrays, materialized audio tracks, audio transformation provenance,
facial-parameter schemas, and cross-modal analysis-result semantics remain open decisions. They
must be documented here before becoming public contracts.

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
| Stable sequence scope, mutable results      | Source and interval cannot drift after validation; named completed results remain attachable.          |
| Shared immutable `VideoSource`              | Separates complete-stream identity/metadata from per-sequence intervals and results.                    |
| Open-ended sequence from inspected video    | Reported duration is not assumed to be an absolute final PTS; decoding continues safely to EOF.         |
| PyAV isolated under `io`                   | Metadata inspection and frame decoding share one media boundary instead of leaking into trackers.      |
| Optional MediaPipe landmark backend        | Provides the first local, cross-platform tracking slice without making it a core dependency.           |
| Named transactional landmark results       | Supports comparisons and prevents failed work from leaving partial sequence state.                     |
| Shared RGB frame stream                    | Backends reuse source indices, timestamps, intervals, and RGB conversion without materializing video.   |
| Shared timestamped audio chunk stream      | Backends receive bounded sample-major float32 chunks without implicit resampling or channel remixing.    |
| Versioned NPZ landmark archives            | Makes expensive tracking results reusable while preserving NumPy dtypes and timeline alignment.        |
| Backend-independent animated mesh contract | Makes triangle geometry reusable by renderers, exporters, and future model-fitting backends.           |
| MediaPipe 468-vertex surface conversion    | Reuses the official 852-triangle topology while keeping relative-depth limitations explicit.           |
| Plotly as optional HTML renderer           | Provides a portable interactive demonstration without coupling core mesh data to a graphics framework. |
| Single-video analysis before collection    | Validates the result model and client value before generalizing folder and batch orchestration.         |
| DeepTalk-ASD as experimental ASD backend   | Reuses a working face/VAD/LR-ASD pipeline while keeping its limitations outside the core.               |
| Offline source-time DeepTalk scheduling    | Keeps A/V deterministic without presenting media timestamps as realtime wall-clock values.             |
| DeepTalk and MediaPipe coexistence         | Allows active-speaker and dense-landmark stages in one environment while versions remain tested.        |
| TalkingFaceKit-owned segment policy        | Backend scores are evidence, not calibrated probabilities or final segment decisions.                  |
| Separate visual-quality and sync stages    | MediaPipe pose and SyncNet offset answer different questions from active-speaker attribution.           |

## Pending decisions

- Whether a materialized multi-frame video array should become a public contract.
- Alternative pixel formats or decoding backends beyond the canonical RGB `uint8` frame stream.
- Contracts for materialized audio, resampling, downmixing, normalization, and their provenance.
- Cross-modal timestamp and synchronization representation beyond landmark source timestamps.
- Exact core schemas for face tracks, speech intervals, raw ASD observations, segment decisions,
  issues, and analysis provenance; publish only those required by the one-video slice.
- Whether production DeepTalk execution needs process isolation to contain native-model crashes;
  the experimental adapter currently targets the optional in-process dependency.
- Thresholds, smoothing, score margins, and minimum-duration rules, which require labeled examples.
- Facial-animation parameter schema and FLAME conventions.
- Serialization formats and versioning policy for data other than landmark tracks.
- Optional dependency groups for future analysis, FLAME, and speech backends.
