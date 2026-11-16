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

The product direction updated on 2026-11-01 adds FLAME parameter extraction on selected segments,
public API simplification, measured performance improvements and code consolidation to the active
plan. Raw FLAME parameter estimation, saved results and FLAME2020 geometry evaluation are available;
training-quality validation and saved-analysis loading remain pending. See [product direction](../PROJECT_DIRECTION.md) and
[team tasks](team_tasks.md). The sections below describe the implemented architecture.

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

The shared single-video workflow is available from Python and the CLI. Sequential collections
reuse exactly this operation for each discovered source:

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
             continuous same-face run policy
       temporal continuity + duration + face coverage
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

MediaPipe is not the active-speaker engine. Its landmark and mesh workflows remain optional.
The first visual-selection stage reuses InspireFace pose and bounding boxes already copied by
DeepTalk. Exact A/V offset is not inferred from ASD scores; optional SyncNet verification measures
it separately. The folder workflow now adds lightweight source discovery, sequential execution,
typed per-source outcomes and incremental persistence around the same single-video result.

## Public analysis and sequential collections

### Preliminary SMIRK input preparation

`integrations.smirk.iter_smirk_inputs` is an experimental integration-level iterator, not a new
root export or sequence method. It consumes the existing PyAV RGB frame stream without resampling,
preserves source indices and PTS, and yields bounded-memory NumPy encoder inputs. MediaPipe runs in
IMAGE mode with the options from SMIRK's upstream helper: one face, detection and presence thresholds
0.1, blendshapes and transformation matrices enabled. This is separate from the existing VIDEO-mode
MediaPipe landmark tracker.

The crop follows the official extent formula, integer crop size and 1.4 scale, with scikit-image
similarity estimation and warping. Channel-independent warping accepts source RGB directly, then
casts to uint8 and normalizes to contiguous RGB float32 arrays shaped `(1, 3, 224, 224)`. Each result
also retains the source-to-crop matrix; this matrix is not a FLAME camera. No PyTorch tensor or
backend object escapes the boundary.

Optional imports and model loading begin during iteration, using an explicit local Landmarker
asset. Each decoded source frame yields either `SmirkInputFrame` or `SmirkMissingFrame`. An empty
MediaPipe landmark result yields the latter with original index/PTS and the stable reason
`face_landmarks_unavailable`; it does not distinguish detector versus presence-stage failure or
claim that the source is unusable by another estimator. Preparation continues at the next frame.
This orchestration policy differs from the official cropped demo's stop behavior, while preserving
the upstream detection options and crop algorithm. No frame is silently removed or replaced.
The decoder and detector are closed on completion, failure, or explicit iterator close. Consumers
that stop early must close the iterator.

The `tracking-smirk` extra now supplies the preparation and encoder dependencies. Numerical tests
compare the RGB warp against the official BGR preprocessing sequence. The original input iterator
remains usable independently of the encoder.

### Experimental SMIRK parameter estimation

`flame.FlameParameters` is a core NumPy value for one per-image estimate: explicit model basis,
shape/expression vectors, and head/jaw axis-angle rotations in radians. Core dimensions follow the
basis; no SMIRK framework object or import enters the value. Per-image shape is not silently
aggregated, and unsupported eye/neck parameters are not represented as measured zeros.

`integrations.smirk.SmirkEncoder` loads an explicit local checkpoint once on construction, hashes it,
uses tensor-only CPU loading, and requires every encoder state key before strict loading. Device
selection is explicit, CPU by default. `_smirk_model.build_smirk_model` adapts the upstream three
towers with matching state names and forward operations. Backbones use `pretrained=False`; the full
SMIRK checkpoint supplies the weights without acquiring ImageNet assets.

`predict` accepts one prepared crop, executes in inference mode, validates expected output shapes
and finite values, and copies compact results to CPU NumPy arrays. `SmirkParameterFrame` retains
source index/PTS, core facial parameters, extra left/right eyelid coefficients, crop camera,
source-to-crop transform and checkpoint hash. Its crop camera and eyelid deformation basis remain
integration-specific; a future PEAR adapter must not silently reinterpret them.

`track` composes the existing preparation iterator and `predict`, without active-speaker execution,
FPS conversion or retaining input pixels in predictions. It returns one `SmirkParameterFrame` or
`SmirkMissingFrame` per decoded source frame. Absences bypass the encoder and contain no crop or
fabricated parameters; only absent landmarks are recoverable, while configuration, decoder,
detector runtime, crop-validation and encoder errors propagate. Even an entirely undetected
interval retains one absence per frame. Source selection remains unchanged: this stage never
splits/rejects segments or interpolates gaps. Training-window eligibility is a subsequent policy,
which must account for absences and must not concatenate across gaps as if time were continuous.
Closing or failing the outer iterator also closes preparation resources. There is no public
sequence reconstruction facade or backend registry in this step.

### Complete FLAME results and persistence

`flame.FlameResult` is a framework-independent, complete-interval value. It stores source identity,
metadata and SHA-256, requested half-open interval, explicit model basis, int64 source frame indices,
float64 PTS, a boolean estimation mask, one absence reason per row, and float32 shape/expression and
head/jaw axis-angle arrays. Model-specific coefficient dimensions remain explicit in the arrays.
Every parameter row is finite when estimated and entirely NaN when absent; the mask describes
availability, not accuracy or training eligibility. Reasons are empty exactly on estimated rows.
Source frames and timestamps must increase strictly and remain within the requested interval.

Named NumPy `extras` retain actual backend-specific output, with equally keyed `extra_conventions`
describing units and axes. SMIRK stores `smirk_eyelids` (float32 left/right deformation coefficients),
`smirk_camera` (float32 crop orthographic camera) and `smirk_source_to_crop` (float64 homogeneous
source-pixel-to-crop matrices). Every extra uses the same frame axis and absence mask. These values
are not promoted to generic FLAME camera/eye semantics. Text provenance records backend, checkpoint
and preprocessing asset hashes, upstream revision, runtime versions, device, preprocessing settings
and local code commit/dirty state when available. Mappings are copied and frozen; arrays must be
treated as read-only.

`SmirkEncoder.extract` consumes `track` and materializes only compact outputs, including an entirely
undetected interval. It normalizes the source path to absolute and hashes source bytes once before
decoding, adding a full-file read to extraction. Asset and package provenance are collected here,
never during saving or loading. Runtime errors propagate without returning a partial complete
result. This entry point does not enforce a selected face ID, aggregate shape, interpolate gaps or
modify segment eligibility. The streaming `track` API remains available for bounded-memory callers.

`FlameResult.save` and `.load` are convenience facades that delegate lazily to `io.flame`, following
the existing analysis result pattern. Construction is pure. The compressed NPZ schema 1 stores
numeric arrays, a Unicode reason vector, namespaced extra arrays and a Unicode JSON metadata scalar
identified by `kind=talkingfacekit.flame`. Loading uses `allow_pickle=False`, rejects unsupported
versions and inconsistent fields, and reconstructs the core value to validate every array contract.
Loading needs neither inference extras nor source/model files and does not assert that any current
video matches the recorded source digest. No weights, tensors, images or mesh data enter the archive.

Saving creates parent directories, writes a temporary archive in the destination filesystem and
publishes it only after serialization succeeds. Existing outputs require explicit overwrite; an
atomic hard link refuses concurrent destination creation when overwrite is false, and replacement
is atomic when true. Failed writes clean up their temporary files and preserve previous outputs.
Saving cannot overwrite the recorded source path. This is complete-result persistence, not incremental
checkpointing, automatic inference caching or collection resumption.

### FLAME2020 geometry evaluation

`integrations.flame.FlameDecoder` loads an explicit local generic FLAME2020 asset and evaluates a
`FlameResult` into the existing `FaceMeshTrack`. `_flame_assets` isolates legacy asset reading and
validation; `_flame_model` evaluates the blend shapes, joint hierarchy, Rodrigues rotations,
pose correctives and linear blend skinning in NumPy. No upstream executable is imported in
production, and no FLAME/LBS source files, models or landmark embeddings are bundled.
Only vertices and triangles are needed; landmark interpolation and camera projection are separate.

The official pickle contains a raw Chumpy `Ch.x` coefficient array and a SciPy CSC joint regressor.
The reader maps those two serialized containers to data-only holders and expands their numerical
arrays with NumPy. It recognizes a limited set of NumPy pickle constructors and rejects unknown
globals before validating geometry; it is not a sandbox for untrusted files. It never imports
Chumpy/SciPy, patches NumPy globals or evaluates a legacy graph. Float32 conversion matches the
reference's numerical dtype. The model has five ordered root/neck/jaw/eye joints and 36 pose
correctives; expression columns start at offset 300. Structural checks cannot prove the asset's
declared year or gender, so callers must supply the matching trusted generic model.

`decode` rejects incompatible model bases/coefficient dimensions. It processes available frames
in bounded numerical batches (32 by default), leaving all absent vertex rows NaN and copying source
indices, PTS and mask. It materializes the complete mesh output, preserves raw per-frame identity
and expressions, applies global/jaw rotations, and explicitly fixes neck/eye rotations to neutral.
It does not reinterpret a crop camera as a world translation or convert native model units/axes.

For SMIRK, both the `smirk_eyelids` coefficients and local left/right deformation assets are required.
Offsets are added right-then-left after skinning, matching the upstream vertex path. They are
never silently ignored. Other geometry-compatible backends need not supply that extension; this
adapter depends on the shared result rather than importing the SMIRK encoder. Asset hashes and
neutral-pose assumptions are exposed in `decoder.provenance`; mesh topology includes the model
hash. No new dependency or generic backend protocol is introduced, and gradients/GPU are outside
this initial NumPy implementation.

The existing optional Plotly renderer can inspect the resulting topology. It still materializes
all animation frames, uses median display timing and does not overlay video. Local reference
comparison uses the pinned external SMIRK decoder and all available saved frames, independently
from analytic synthetic tests for shape, expression, pose, joints, eyelids and absent-frame handling.

### Source-video and mesh comparison

`render_face_mesh_html(..., source_video=...)` adds an offline side-by-side comparison without
changing the standalone 3D animation path. `_video_mesh` owns its serializer and HTML controller;
the renderer remains tracker-independent and accepts the same `FaceMeshTrack`. The original local
video is referenced by file URL, never copied or decoded in rendering. A local file picker can
rebind the same original when a browser refuses direct loading; no file is uploaded.

The video's presented-frame `mediaTime` drives frame selection through `requestVideoFrameCallback`,
with currentTime-based animation-frame/timeupdate fallbacks. A binary search selects the latest
mesh PTS at or before that source time, without nominal-FPS conversion or a second playback clock.
An explicit `video_time_offset_seconds` maps source PTS = browser media time + offset, zero by
default; the renderer does not infer that mapping or verify video identity. Missing rows hide the
mesh rather than displaying the last available geometry. Source index, mesh PTS and video source
time remain visible. Seeking and frame buttons use stored PTS, and playback stops at the last
sampled PTS. This is visual comparison, not a pixel-accurate overlay or an A/V sync measurement.

Plotly is initialized once with one topology, then only the active vertex coordinates are updated.
The controller serializes asynchronous updates and retains the latest pending frame; slow rendering
may skip visual updates during playback, so exact row inspection uses pause/seek. Camera interaction
is preserved. Geometry is encoded once as base64 little-endian float32, retaining NaN absences;
triangles, timestamps and mask are stored once. This reduces serialized size compared with the old
per-frame Plotly trace export, while still materializing the complete interval and embedding Plotly.

Generated metadata is escaped for inline JSON, user text is HTML-escaped, and output publication is
atomic with explicit overwrite. Controller regressions run in Node with a simulated DOM/media clock,
covering irregular PTS, source offsets, missing rows and seeking; these tests do not substitute for
interactive browser/rendering QA.

### Existing analysis contracts

`active_speaker.py` owns immutable framework-independent observations, raw score windows,
provenance, structural policies and pure candidate aggregation. These values were extracted from
the experimental adapter without changing policy or inference behavior. Its previous `DeepTalk*`
names remain aliases in `integrations.deeptalk` for compatibility. Neither the core nor downstream
SyncNet/report consumers import an active-speaker integration to obtain evidence types.

`analysis.py` defines `AnalysisConfig`, `AnalysisTimings`, `TalkingFaceSegment` and
`VideoAnalysisResult`. Configuration construction performs no I/O; defaults preserve structural
selection, while visual filtering and a specific SyncNet checkpoint are opt-in. A result retains
the original `VideoSource` and requested bounds, all enabled stage results, timings and final
selected segments. Its assessed segments are not a complete video partition: preliminary
non-speech/no-face/ambiguous regions remain in the raw structural evidence. Selected segments
identify their source and can become lightweight sequences without copying media or attached
results. Model execution never mutates the input sequence.

`processing.py` is the application boundary. `VideoAnalyzer` coordinates DeepTalk, synchronization,
then visual selection with one configuration. `TalkingFaceSequence.analyze` delegates here.
`analyze-video` and `evaluate-benchmark` use the same runner and do not own selection logic.
The runner retains an optional private SyncNet verifier across sequential calls, preserving the
existing benchmark behavior. DeepTalk tracking/VAD/model state is created per source. There is no
thread/process executor, model-sharing redesign or generic pipeline system.

`TalkingFaceCollection` contains immutable source paths rather than eagerly inspected sequences.
`from_folder` delegates to `io.collection`, which discovers common case-insensitive video suffixes
in deterministic relative-path order. Recursive discovery is explicit and does not follow directory
symlinks. `from_videos` preserves caller order. Each source becomes a sequence when analysis starts,
so a corrupt file is a typed per-source failure rather than a failure to load the collection.
Empty collections load no models. Direct construction of collection/result/segment values performs
no media I/O. Paths are made absolute at collection construction to preserve source identity if
the working directory later changes.

`CollectionAnalysisResult` retains ordered `VideoAnalysisOutcome` values and exposes successes,
failures and selected segments across sources. An outcome has exactly one result or
`AnalysisFailure`; a successful analysis with no selected time is never labelled a failure.
`on_error="continue"` records inspection/model exceptions; `"raise"` records and reraises the
original exception. Configuration, persistence, callback errors and KeyboardInterrupt propagate.
Progress callbacks receive a `CollectionProgress` after each outcome is recorded and optionally
saved. The application/collection layer does not print or configure process-wide logging;
CLI/example code owns progress presentation. The experimental upstream backend can still emit SDK
messages and configure logging during its own initialization.

`io.analysis` owns atomic JSON persistence. The single-video JSON keeps its additive version-1
schema, source SHA-256, code/model provenance, all stage evidence and selected source-time bounds.
Source metadata and extra timings are additive fields. Optional collection output saves one full
JSON per successful video and updates `collection.json` after every outcome, including failures.
Preflight checks reject conflicting output names before inference; overwrite is explicit.
Order-prefixed names distinguish identical stems. Writing failures propagate without destroying
previously completed files. This is persistence of progress, not automatic cache lookup or resume.
`VideoAnalysisResult.render_report` renders existing in-memory evidence without model execution,
referencing unchanged local media. A typed JSON reload API remains deferred.

### Evaluation data and the benchmark view

`EvaluationCaseResult` pairs a human `EvaluationCase` with the existing `VideoAnalysisResult`.
Its predictions and metrics derive from those values using the existing scoring rules.
`EvaluationResult` aggregates cases, metrics and `TalkingFaceSegment` references independently of
presentation. `processing.evaluate_benchmark` owns sequential execution, source-hash verification
and window clipping; the CLI delegates to it, retaining argument validation and console formatting.
`io.evaluation` persists the existing version-1 benchmark schema, with additive selected segment
references and scope metadata. `demo.render_benchmark_report` remains the same optional view over
that JSON; neither saving nor rendering repeats inference.

`collection.evaluate` restricts evaluation to human-labeled windows of collection sources.
Unannotated sources and unmatched annotation filenames are recorded, excluded from metrics, and
never interpreted as negative labels. Simple annotation inputs can explicitly map filenames with
`video_aliases`; timestamps and labels remain unchanged. Approved hashed manifests cannot be
remapped. Standalone evaluation without collection scope remains strict about missing media.
Result persistence protects input media and annotation files/directories. Evaluation does not
automatically export physical clips or introduce parallel execution, new model policy or a new
visualizer.

## Dependency direction

### Optional candidate synchronization

`SyncNetVerifier` follows DeepTalk structural selection and verifies every surviving candidate;
there is no calibrated ASD-based trigger for suspected desynchronization. It consumes existing
track IDs, source timestamps and boxes without running another detector. Bounded 25 Hz BGR face
crops and 16 kHz mono PCM come from the shared PyAV streams. Upstream SyncNet-v2 model architecture
is adapted under its MIT license, with a packaged notice. The official checkpoint is explicit,
SHA-256 checked and tensor-only loaded. The optional extra adds PyTorch; it does not install the
full `syncnet-python` pipeline or change the selected PySceneDetect version.
Verification advances one shared iterator for each source stream across ordered candidate
windows, retaining only the current window and boundary frames/chunks. Prefixes are not decoded
again for every candidate. Chunk clipping happens after reading original audio PTS so overlap
from millisecond quantization cannot create equal trimmed chunk starts at a window boundary.

Core synchronization dataclasses live in `synchronization.py`; optional loading, preprocessing,
and inference live in `integrations/syncnet.py`. DeepTalk observations and candidate runs stay
unchanged. Shared application orchestration retains two separate results and selects from sync decisions when
enabled. JSON stores policy, per-window evidence, per-candidate decisions, model hash and elapsed
time; both HTML reports reflect those decisions while showing the original ASD evidence.

Windows partition each candidate completely within its shot/identity bounds. The largest is five
seconds; less than two seconds is unmeasurable. Checkpoint embeddings use the original full-scale
BGR/int16 preprocessing. Each audio/visual tower runs once per window. Offset search compares
feature distances at all lags on the same interior frames, avoiding upstream's zero-padded search
edges. The search range is reduced in short windows to retain one second of common evidence.
An optimum on either search boundary is uncertain. Audio PTS discontinuities above 1 ms, video
sampling gaps and missing face evidence also prevent a confident measurement.

Positive public offset means late audio, verified with injected PCM delays. Resolution is one
25 Hz frame (40 ms), not sample-accurate lip alignment. Raw confidence is median search distance
minus best distance, not a probability. Initial ±80 ms / confidence 3.0 thresholds are experimental
and configurable; neither correct metadata nor a positive ASD score guarantees synchronization.
Any confident out-of-tolerance window rejects its candidate; all windows must pass to retain it.
Incomplete or uncertain measurements are excluded from automatic selection. Conflicting reliable
offsets suppress the global estimate, preserving each window for a future optional repair.
Failures to load or execute a model fail explicitly rather than silently passing candidates.
This stage never rewrites source media. Visual selection follows separately when enabled.

### Optional visual selection

`visual.py` contains framework-independent immutable observations, policy, raw classified regions,
metrics and temporally filtered decisions. `assess_visual_suitability` performs no I/O or inference.
The application runner copies existing pose and source-pixel boxes into those values and supplies source
image dimensions and the observation grid rate. No new runtime dependency or model is required.
`--visual-filter` enables the stage in both single-video analysis and benchmark evaluation; without
it the previous structural/sync selection is preserved. Custom visual thresholds require the flag.

The stage follows structural selection and, when enabled, SyncNet. Upstream rejected/uncertain
regions remain unchanged and cannot be promoted. Successful sync measurements already cover the
entire original candidate, including all retained visual subintervals; visual trimming does not
claim to have run a new independent SyncNet measurement on each shorter fragment. Every upstream
interval remains an independent shot/identity boundary, even if two adjacent intervals share an ID.

Current experimental inclusive limits are absolute yaw 70°, pitch 40°, roll 45°, original box
shorter side 64 pixels, visible width/image width 0.04, and box area outside the image at most
0.05. These defaults admit three-quarter views and moderate side orientation; strict frontal
alignment is not required for the clarified training use case. The initial 25°/20°/96px policy
remains reproducible through explicit configuration; its recall loss and the fixed-label replay
are recorded in `docs/research/visual_selection.md`. These are declared starting values, not calibrated limits. Pose signs remain backend evidence;
absolute comparisons do not depend on left/right display conventions. Missing or implausible pose
is uncertain. Resolution is measured in the original source, never after model-input upsampling.

One measurement supports at most one grid period after its source timestamp, shortened by the next
measurement and the input interval end. Missing time has explicit uncertain windows; no pose is
interpolated or extrapolated across long gaps. A failed gap up to 0.20 seconds can be bridged only
between passing evidence within one upstream interval. Leading/trailing failures are trimmed.
Retained runs require two seconds, at least 90% time with valid box/pose evidence, and at least
90% time passing all visual checks. This prevents repeated short failures from bypassing policy.
Unmeasured and failed time remain excluded from those fractions even when tolerated. Sustained
failures split candidates, and fragments below minimum duration are rejected.

The result retains a complete partition of input decisions, raw instantaneous windows, measured
extrema and time fractions, and exact tolerated-gap bounds. Raw DeepTalk and SyncNet results are
not modified. JSON `selected_intervals` reflects every enabled stage. Benchmarks save pre-visual
intervals and metrics from the same inference, alongside post-filter predictions. The comparator
recomputes both sets of metrics from the fixed human annotations and saved intervals. Older
version-1 reports without visual fields remain readable. Both HTML viewers expose applied limits,
before/after selection, reasons and measurements. The comparator validates that saved visual
decisions match predictions and cannot add time outside the pre-filter selection.

Frontal head orientation is not ocular gaze. Box clipping is not an occlusion detector. The raw
InspireFace quality score remains diagnostic: this stage certifies neither sharpness, lighting,
eye contact nor visibility of individual facial parts. Passing results remain candidates for
dataset review, not universally accepted training samples.

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

`TalkingFaceSequence.export_clip(...)` is the explicit physical-media boundary for bounded
sequences. It delegates to `io.clip`, which reuses the shared PyAV video and audio streams over the
same half-open source interval. Both output tracks subtract the sequence start rather than rebasing
independently, preserving their relative offset. Video is re-encoded as H.264 and optional audio as
AAC in MP4; dimensions, average frame rate, audio sample rate and layout are retained where the
encoders support them, including the source sample aspect ratio and display rotation. Because
H.264 `yuv420p` requires even dimensions, an odd source width or height gains one replicated
column or row on the right or bottom; source pixel coordinates are unchanged. Generic one- and
two-channel PCM layouts are mapped to AAC mono and stereo. A source with no audio samples inside
the selected interval produces a video-only MP4. The output is written beside its destination and
atomically published only after both encoders flush. Existing destinations are refused unless
replacement is explicit.

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
│   ├── clip.py       PyAV-based precise MP4 decode/re-encode boundary
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
final partial frame. Before resampling, source PTS are reconciled against a source-rate sample grid
anchored at the selected sequence start. Quantization up to one millisecond is treated as timestamp
jitter. Larger gaps are hard-compensated with streamed zero samples and overlaps are trimmed from
the later chunk, matching FFmpeg's hard timestamp-compensation semantics without materializing the
track. Bounded trailing gaps are filled to the requested end. Every adjustment is copied into an
immutable `DeepTalkAudioTimelineRepair` and surfaced through issue codes; DeepTalk therefore
receives continuous PCM without hiding changes to the source timeline.

The optional demo report lives under `talkingfacekit.demo`, outside the core and stable rendering
surface. It serializes copied DeepTalk observations into one offline HTML document and references
the unchanged local video by file URL. Browser playback time is the only synchronization clock for
boxes, pose, raw quality, VAD state, raw scores, preliminary decisions, continuous candidate-run
decisions, scene-cut and repair markers, and the diagnostic timeline. The report labels the largest
score during speech as a `top candidate`; it does not add a probability or accepted
training-segment decision.
It also derives a clickable review list from the same immutable evidence, labelling regions as
`CONSERVAR`, `DESCARTAR`, or `REVISAR` for the current policy stage. This isolation allows the demo
UI to be removed or replaced without changing core contracts or the DeepTalk adapter.

Both streams use relative media time (`source_time - sequence.start_seconds`) only inside the
adapter. Audio timestamps identify each chunk's exclusive end. A lazy two-way merge feeds video
before audio on exact ties, with no realtime playback, clock, thread, or wall-clock timestamp.
DeepTalk is evaluated incrementally in roughly one-second windows before its ten-second audio buffer
can evict earlier samples. After each evaluation, mouth images older than the completed window are
pruned while a frame exactly on the shared boundary is retained. Results are copied into immutable,
validated integration-specific face observations, VAD intervals, score windows, provenance, and
diagnostic issue codes on the original source timeline; no DeepTalk profiles, images, audio frames,
or embeddings escape. A face observation preserves the bounding box, raw InspireFace head pose in
degrees, and raw InspireFace quality score. Those values are diagnostic evidence rather than a
camera-facing or visual-suitability decision in the adapter; the optional visual stage interprets
pose and boxes separately. Open-ended sequences report unequal A/V coverage and
do not retain face observations after audio coverage ends.

The first TalkingFaceKit-owned policy is deliberately pre-calibration and remains represented by
integration-specific immutable values while DeepTalk is the only evaluated backend. The existing
25 Hz frames sampled from the shared PyAV stream are passed to PySceneDetect's `AdaptiveDetector`
with its defaults. Its minimum content change and rolling local ratio help distinguish hard cuts
between similarly framed speakers from sustained motion. Only RGB-to-BGR channel conversion is
added; the source is not opened or decoded a second time. Each detected scene cut is copied into an
immutable `DeepTalkShotBoundary` with its source timestamp.

The policy splits score windows at active VAD and scene-cut boundaries and rejects regions with no
speech, no visible face, simultaneous faces, changing face identity, or a non-positive raw score. A
missing score for one visible face is uncertain. One visible identity with a positive raw score is
only a `candidate`; it is not accepted until later stages pass. These preliminary decisions and raw
scores are immutable instantaneous evidence.

The temporal policy merges adjacent preliminary candidates for the same identity within one shot.
It may also bridge at most 1.25 seconds containing only non-positive ASD evidence or a brief VAD
interruption when positive ASD evidence resumes immediately for the same face. The gap itself must
retain at least 90% face coverage, no missing-face interval above 200 milliseconds, exactly one
visible face, continuous preliminary coverage, and no shot boundary. Ambiguity, identity change,
missing scores, missing faces, longer gaps, and trailing rejected time remain hard stops. This
offline look-ahead is the hangover: it delays the continuity decision but does not append grace
silence when positive evidence never returns. Every retained gap is recorded in
`bridged_gap_intervals`, while its red preliminary decision remains visible in the report.

This prevents an isolated ASD failure from fragmenting a clip without treating VAD plus one visible
face as proof of speaking, and it still prevents a reused DeepTalk local track ID from bridging a
scene cut. The structural policy then measures duration, detection coverage against the 25 Hz grid,
the largest interval without a face detection, and raw-score min/mean/max over positive candidate
evidence. Initial defaults require two seconds, 90% coverage, and no visibility gap above 200
milliseconds. They are explicit conservative starting values rather than calibrated guarantees;
failed runs stay in the result with stable reason codes. Even a passing run still requires
calibrated-score, pose, quality, and synchronization decisions. Every stage retains reason codes and
raw evidence so its thresholds can be replaced after evaluation without rewriting or hiding backend
output.

The stored window bounds are consecutive, but DeepTalk 0.3.1 internally uses an inclusive end for
video while audio remains half-open. A video frame exactly on a boundary may therefore contribute
to both adjacent backend evaluations; the adapter does not distort timestamps to conceal this.

DeepTalk 0.3.1 requires one localized private shim for offline use. After verifying the installed
version and expected face, VAD, and speaker-detector layouts, the adapter disables the redundant
float-based video throttle, wall-clock face/track expiry, and wall-clock VAD reset. Voiceprints are
explicitly disabled, their model is not acquired, and provenance reports speaker embeddings as
unavailable. Before importing ONNX Runtime through DeepTalk, the adapter disables its telemetry by
default unless the process explicitly configured a preference. A changed version or required
private field fails clearly instead of applying the shim speculatively. An explicitly configured
InspireFace resource is hash-checked before native code receives it; models resolved through
DeepTalk continue to use its hash-verifying model manager.

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

Human annotators fill JSON lists with only `file`, `start_seconds`, `end_seconds`, and `selected`.
The shared source is `benchmark/annotations/`, with one file per video and one or more completed
windows per file. Git versions those labels and the generated `benchmark/manifest.json`; raw videos
stay in shared storage and local ignored `data/`. Assignments and unfinished work are recorded
separately: an empty selection is an explicit reviewed rejection, never a pending annotation.

`prepare-benchmark` accepts a legacy single JSON list or recursively collects all `.json` files in
a directory, sorted by relative POSIX path with row order preserved. It validates each source,
reports its path on malformed JSON/rows, rejects empty inputs and duplicate file/window pairs
across files, and caches complete-video SHA-256 hashes for repeated windows. The CLI prevents an
output manifest inside the annotation directory. No identifiers or hashes are entered manually.
`--check` verifies the existing manifest against freshly prepared cases without writing; annotation
or media drift fails explicitly. It is mutually exclusive with `--overwrite`. Reviewed label changes
and the regenerated manifest travel in the same commit; existing runs retain their original labels
and manifest digest. The manifest schema and the single-list input remain backward compatible.

Explicit source-time windows avoid requiring media cuts or treating estimated stream duration as
an exact final timestamp. The annotations contain no tags or per-component labels. Verification of
labels and media hashes runs without loading models; the opt-in benchmark then compares DeepTalk
candidate runs with human selections by duration and writes a JSON report outside Git. This
evaluates the current selection stage without claiming that structural candidates are final
FLAME-ready clips. Public tests use only synthetic labels and tiny existing fixtures.

`evaluate-benchmark` also persists optional diagnostics per case: preliminary segments, candidate
runs (including rejections and bridged gaps), and detected shot boundaries. Version-1 results
without this additive field remain supported. The `render-benchmark` command reads a saved result,
reuses manifest validation, verifies local video hashes, and recomputes metrics from the stored
human/predicted intervals. It never loads models or reads a newer annotation file. The standalone
HTML lives under `talkingfacekit.demo`, references local videos without copying media, and provides
human/library timelines, difference navigation, context playback, and recorded policy evidence.
Ignored intervals have precedence in the temporal partition and are excluded from errors. Original
interval boundaries remain visible; temporal agreement does not imply equal segment boundaries.
Legacy results explicitly report missing diagnostics rather than inferring reasons. Rendering uses
the packaged HTML template, escapes embedded JSON, and replaces the destination only after input
validation and complete serialization. No runtime dependencies are added for this workflow.

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
| Shared typed analysis and sequential collection | Python facades and CLI reuse one operation; per-source outcomes and atomic progress outputs preserve completed work. |
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
