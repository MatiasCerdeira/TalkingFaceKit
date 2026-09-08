# DeepTalk-ASD Compatibility Spike

Date: 2026-09-07  
TalkingFaceKit revision: local working tree; no DeepTalk integration code or dependency changes  
DeepTalk-ASD version tested: `0.3.1`

## Thesis decision update — 2026-09-08

Proceed with DeepTalk-ASD as TalkingFaceKit's first **experimental active-speaker backend**. The
project is a noncommercial CS thesis, so the noncommercial terms of the default InspireFace weights
do not block this milestone. Their identity, hashes, and terms still belong in provenance, and a
future commercial use would require a new review.

This is a decision to reuse a promising pipeline behind a narrow adapter, not to make DeepTalk types
part of the core or to accept the wrapper's current timing and buffering behavior. The implementation
order is:

1. timestamped PyAV audio streaming — completed after this spike;
2. experimental DeepTalk adapter, initially with speaker embeddings disabled on macOS;
3. one-video JSON report and diagnostic overlay;
4. six-case labeled evaluation;
5. retain DeepTalk, move to direct LR-ASD ONNX, or compare TalkNet according to the evidence.

The original spike conclusion below distinguishes prototype suitability from production-library
hardening. Where it discusses commercial licensing as a blocker, this update is authoritative for
the thesis scope.

## Executive conclusion

DeepTalk-ASD receives a **pass as the thesis's experimental proof-of-concept backend** and remains
**unsuitable as an unadapted foundational runtime dependency in its current form**.

The good news is substantial:

- It installs and runs on Apple Silicon macOS with CPython 3.11.
- Its InspireFace detector, multi-face tracking, Silero VAD, and LR-ASD ONNX models all execute on
  CPU.
- It processed its 63.24-second, two-face demo in approximately 22.5 to 23.5 seconds after model
  loading, or roughly 69 to 72 decoded frames per second.
- It processed TalkingFaceKit's existing 2.71-second WebM fixture, detected its face, found speech
  turns, returned active-speaker scores, and produced a correctly annotated output video.
- All runtime tests succeeded with network access disabled after a one-time explicit model download.
- A forced NumPy 2.4.6 environment also completed the TalkingFaceKit fixture, suggesting the
  package's exact NumPy 1.26.4 pin may be unnecessarily restrictive.

The blockers and risks are also substantial:

- The default InspireFace `Pikachu` weights are restricted to noncommercial research use. They are
  usable for the current thesis, but any future commercial scope would need another model or rights.
- `deeptalk-asd==0.3.1` pins `numpy==1.26.4`, while TalkingFaceKit currently requires
  `numpy>=2.4.6`; a normal dependency resolver cannot satisfy both.
- The installed Sherpa-ONNX macOS wheel could not locate its ONNX Runtime dynamic library, so
  DeepTalk silently disabled speaker embeddings while still returning a successfully constructed
  ASD object.
- A temporary dynamic-loader workaround made speaker embeddings work, proving this is a native
  packaging/linkage problem, but such a workaround is not a suitable hidden production behavior.
- Supplying an invalid InspireFace resource path terminated the entire Python process with exit code
  134 (`SIGABRT`) rather than raising a recoverable Python exception.
- DeepTalk's exact-25-FPS throttling dropped approximately 24% of frames because a strict
  floating-point timestamp comparison treats some nominal 40 ms intervals as too short.
- Audio retention is bounded to 10 seconds, but per-face mouth-frame buffers are unbounded while a
  face remains visible.
- Silero VAD state is reset according to wall-clock processing time rather than media time. Repeated
  offline runs of the same input produced different turn boundaries and counts.
- The returned LR-ASD values are signed raw scores, not probabilities. Multiple visible faces often
  receive positive values in the same evaluation, so `score > 0` is not a sufficient segment policy.
- The upstream file demo synthesizes timestamps from frame index and average FPS instead of
  preserving source presentation timestamps, materializes the complete decoded audio track, and
  does not flush an unfinished speech turn at end-of-file.

The backend-independent timestamped audio streaming boundary was implemented after this spike.
DeepTalk is now the next experimental integration behind a narrow, clearly marked boundary, with
process isolation considered while the native crash risk exists. Adding it to the normal runtime
dependency set still requires a separate dependency decision by the team.

## Purpose and scope

This spike answers a narrow question: can DeepTalk-ASD provide a fast route to visible active-speaker
results without forcing TalkingFaceKit to build face detection, tracking, VAD, and audio-visual
speaker inference from scratch?

It evaluates:

- Python and platform installation;
- the actual dependency graph and footprint;
- deterministic model acquisition and offline execution;
- detector construction;
- face IDs and bounding boxes;
- VAD state transitions;
- LR-ASD score output;
- the optional speaker-embedding path;
- CPU throughput and process memory;
- behavior on the upstream sample and TalkingFaceKit's existing fixture;
- compatibility with TalkingFaceKit's NumPy version;
- timestamp and buffer behavior relevant to offline datasets;
- basic failure behavior and model licensing.

It does **not** establish detection accuracy on the client's domain. The two videos used here do not
have frame-level ground-truth active-speaker labels. Visual inspection can establish that the system
is functioning and producing plausible overlays, but not precision, recall, or acceptable false
positive rates.

## Isolation strategy

The TalkingFaceKit repository was not moved. Two disposable `uv` projects were created outside the
repository:

- `/private/tmp/talkingfacekit-deeptalk-spike.aeu1Dy`: exact upstream dependency environment using
  NumPy 1.26.4, plus model files, upstream demo media, generated media, and instrumentation;
- `/private/tmp/talkingfacekit-deeptalk-numpy2.HGhHLK`: forced NumPy 2.4.6 compatibility environment.

This was deliberate. It prevented DeepTalk's exact NumPy pin, native wheels, model assets, and
generated media from modifying:

- TalkingFaceKit's `pyproject.toml`;
- TalkingFaceKit's `uv.lock`;
- TalkingFaceKit's `.venv`;
- the repository's source or tests.

Only this research report was added to the repository. Temporary directories may disappear after a
restart or normal system cleanup and must not be treated as project artifacts.

## Test host

| Property | Observed value |
| --- | --- |
| Operating system | macOS 26.6.2, Darwin 25.6.0 |
| Architecture | Apple Silicon `arm64` |
| Python | CPython 3.11.15 managed by `uv` |
| `uv` | 0.11.8 |
| Inference provider | ONNX Runtime `CPUExecutionProvider` |
| FFmpeg / ffprobe | Homebrew binaries available under `/opt/homebrew/bin` |
| GPU/CUDA | Not used |
| Network during inference | Disabled; `DEEPTALK_ASD_OFFLINE=1` |

The first attempted installation selected the host's CPython 3.13 because the disposable project's
generated constraint was `>=3.11`. Resolution then failed because InspireFace 1.2.3.post5 only
published compatible CPython 3.11 and 3.12 wheels. Explicitly pinning the environment to Python 3.11
resolved the issue.

Although DeepTalk's package metadata says `Requires-Python: >=3.8`, the effective supported Python
range is narrower wherever its required InspireFace dependency has no wheel.

## Installed package and dependency footprint

The exact dependency environment installed successfully with:

```text
deeptalk-asd==0.3.1
numpy==1.26.4
inspireface==1.2.3.post5
onnxruntime==1.29.0
opencv-python==4.11.0.86
sherpa-onnx==1.13.7
scipy==1.17.1
pandas==3.0.5
```

DeepTalk declares the following direct dependencies:

```text
python-dotenv
termcolor
tornado
inspireface
onnxruntime
python_speech_features
numpy==1.26.4
opencv-python
scipy
pandas
tqdm
sherpa-onnx
```

The resolved environment installed 31 packages and occupied approximately 410 MiB. Model artifacts
occupied another approximately 48 MiB. This is much smaller and simpler than a PyTorch/CUDA stack,
but it is not a lightweight normal dependency for TalkingFaceKit's core.

The DeepTalk wheel itself is pure Python:

| Distribution | SHA-256 | Size |
| --- | --- | ---: |
| `deeptalk_asd-0.3.1-py3-none-any.whl` | `02692716e5237f280c2f5958c208d0cc7f54569c68f3ef5eb8da5dd77dba1a70` | 58,259 bytes |
| `deeptalk_asd-0.3.1.tar.gz` | `d20787ec02ea9eb5830916d0bbb248d64af48163326f550065405aa708b94cf8` | 46,169 bytes |

The platform complexity comes from native dependencies such as InspireFace, ONNX Runtime, and
Sherpa-ONNX rather than from the DeepTalk wheel.

## Model acquisition and provenance

DeepTalk 0.3.1 registers six model files hosted on its GitHub release `v0.2.1`. They were downloaded
once to an explicit temporary directory. DeepTalk verified the built-in SHA-256 values, and the
spike independently reproduced them.

| Artifact | Bytes | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| `audio_frontend.onnx` | 929,016 | `a1b55df7105fb730196b527f3e5c8d39f4ab38013a21d01c15404e870a8a5cb3` | LR-ASD audio frontend |
| `visual_frontend.onnx` | 1,603,451 | `af8e213b148d573008c14068ebb6255779b2cce1fef56416e276bd5f49ca6e29` | LR-ASD visual frontend |
| `av_backend.onnx` | 832,913 | `13ec6c03885e104b7bca0f30b0574f0e3ffc896ca6a7b9d71e75ca413c75eb55` | LR-ASD audio-visual backend |
| `silero_vad.onnx` | 2,327,524 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` | Silero voice activity detection |
| `Pikachu` | 16,783,360 | `5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431` | InspireFace resources |
| `wespeaker_zh_cnceleb_resnet34.onnx` | 26,534,363 | `f86cd6c509f331f0e20b07bd48d1b2eb7de54202643401c4e84695ac861a0e5a` | speaker embeddings |

Total exact size: 49,010,627 bytes.

DeepTalk's model manager supports an explicit cache directory and an offline flag. With
`DEEPTALK_ASD_OFFLINE=1`, all model lookups succeeded from the prepared directory while external
network access was unavailable. This part of its design is suitable for a reproducible integration,
provided TalkingFaceKit controls the directory and validates an asset manifest before native model
initialization.

### Licensing result

DeepTalk's Python code is MIT licensed, but that does not grant one license for every model it
orchestrates.

| Component | Current finding | Decision impact |
| --- | --- | --- |
| DeepTalk Python code | MIT | Suitable in principle |
| LR-ASD code/model path | DeepTalk identifies it as MIT | Record and verify the exact converted checkpoint provenance before distribution |
| Silero VAD | MIT model path according to DeepTalk | Record exact artifact attribution |
| Sherpa-ONNX runtime | Apache-2.0 according to DeepTalk | Runtime is suitable in principle; verify WeSpeaker model terms separately |
| InspireFace code | Open source | Code license alone does not cover default weights |
| InspireFace `Pikachu` weights | Noncommercial research restriction in the official InspireFace documentation | Usable for this thesis; future commercial use needs new rights or replacement |

Primary references:

- [DeepTalk-ASD on PyPI](https://pypi.org/project/deeptalk-asd/)
- [DeepTalk-ASD repository](https://github.com/huyyxy/DeepTalk-ASD)
- [InspireFace license statement](https://github.com/HyperInspire/InspireFace/blob/master/README.md#license)
- [LR-ASD repository](https://github.com/Junhua-Liao/LR-ASD)

## Experiments and results

### 1. Package import and detector construction

The following components loaded successfully with explicit local model paths and offline mode:

- InspireFace face detection and tracking;
- Silero VAD ONNX session;
- LR-ASD audio frontend ONNX session;
- LR-ASD visual frontend ONNX session;
- LR-ASD audio-visual backend ONNX session.

The factory returned an `ASD` instance. The first cold construction observed in the clean exact-pin
environment took approximately 25.7 seconds. Subsequent warm constructions generally completed in
less than one second. Startup time must be measured again on Windows and Linux and on a cold
production machine; the current results show that a long-lived worker would be preferable to
reinitializing models for every short video.

### 2. Upstream two-face sample

The upstream repository contains a 640×360, 25 FPS sample with:

- 1,581 decoded video frames;
- 63.24 seconds of video;
- 63.99 seconds of 44.1 kHz stereo AAC audio;
- two visible presenters.

The repository's `video_asd_demo.py` was run unchanged against the installed DeepTalk 0.3.1 package,
apart from command-line paths and the offline environment. The demo script is not included in the
wheel; the current upstream copy still labels its own project version as 0.3.0. Installed library
code came from the 0.3.1 wheel.

Observed results after warm loading:

| Measurement | Result |
| --- | ---: |
| Frames processed | 1,581 |
| Processing time reported by demo | 22.5–23.5 seconds |
| Reported throughput | approximately 69–72 FPS |
| Stable face IDs | 2 (`1`, `2`) |
| Face observations | 1,580 per ID; the detector's two-frame presentation window suppresses the first frame |
| Annotated output | created successfully with video and audio |
| Visual inspection | boxes stayed on both faces and the highlighted primary speaker visibly alternated |

This is a functionality result, not an accuracy score. The sample has no machine-readable
ground-truth labels, and the spike did not independently transcribe or label its Chinese-language
audio.

### 3. TalkingFaceKit fixture

TalkingFaceKit's existing `tests/fixtures/example1.webm` contains:

- 1920×1080 VP9 video at approximately 23.976 FPS;
- 65 frames and approximately 2.71 seconds;
- 48 kHz stereo Opus audio;
- one clearly visible face.

The upstream demo decoded the video, converted its complete audio track to 16 kHz mono signed
16-bit PCM through FFmpeg, detected one stable face ID, and generated an annotated MP4.

| Measurement | Result |
| --- | ---: |
| Frames processed | 65 |
| Processing time after warm initialization | approximately 1.1 seconds |
| Reported throughput | approximately 62 FPS |
| Stable face IDs | 1 (`1`) |
| First confirmed-turn score | `1.1` at demo time 1.08 seconds |
| First end-turn score | approximately `0.39` at demo time 1.33 seconds |
| Second confirmed-turn score | `0.8` at demo time 1.96 seconds |
| Second end-turn score | approximately `0.61` at demo time 2.29 seconds |

Frames extracted from the output during both confirmed turns visibly showed the face ID 1 box in
green with the `SPEAKING` label.

The demo observed a final `TURN_START` close to end-of-file but did not flush it into an end event.
An offline TalkingFaceKit adapter must define explicit end-of-stream behavior.

### 4. Offline execution

After the model download, detector construction and both video runs succeeded with:

```text
DEEPTALK_ASD_OFFLINE=1
DEEPTALK_ASD_CACHE_DIR=<explicit temporary model directory>
```

The sandbox denied general network access during inference, so successful completion did not depend
on a hidden runtime model fetch. ONNX Runtime emitted a telemetry device-ID persistence warning on
macOS, and Apple's network cache emitted local cache-database warnings, but neither blocked
inference or downloaded assets.

### 5. NumPy 2.4.6 override

Normal resolution is impossible because:

```text
TalkingFaceKit: numpy>=2.4.6
DeepTalk-ASD:   numpy==1.26.4
```

A second disposable project used `uv`'s `override-dependencies` mechanism to force NumPy 2.4.6.
That environment resolved OpenCV 5.0.0.93 instead of OpenCV 4.11.0.86. It successfully:

- imported DeepTalk;
- initialized InspireFace, Silero, and LR-ASD;
- decoded TalkingFaceKit's WebM fixture;
- detected the same face;
- produced the same two confirmed speech turns;
- returned very similar raw scores;
- generated an annotated output video.

This is encouraging but insufficient to delete or override the upstream pin globally. It proves
compatibility only for the exercised path on this host. A production decision requires upstream
clarification or a maintained fork plus tests across supported platforms and media cases.

### 6. Speaker-embedding path on macOS

DeepTalk directly declares and installed `sherpa-onnx==1.13.7`. Nevertheless, importing
`sherpa_onnx` failed because its native extension requested:

```text
@rpath/libonnxruntime.dylib
```

The installed ONNX Runtime 1.29.0 wheel instead provided:

```text
onnxruntime/capi/libonnxruntime.1.29.0.dylib
```

DeepTalk catches this import failure and logs the misleading message that Sherpa-ONNX is not
installed. It then constructs the LR-ASD detector with `voice_extractor=None`; the factory still
returns a successful overall `ASD` instance. Callers cannot rely on factory success to mean all
requested capabilities are available.

A temporary loader directory containing the expected unversioned and major-version names made the
native import succeed. With that directory on `DYLD_LIBRARY_PATH`:

- the 256-dimensional speaker-embedding extractor initialized;
- the full upstream demo completed;
- two voice profiles were created;
- throughput remained approximately 72 FPS;
- peak RSS increased from approximately 574 MB to approximately 675 MB.

This workaround demonstrates a fixable wheel/linkage problem. TalkingFaceKit should not create
unmanaged dynamic-library symlinks or silently depend on `DYLD_LIBRARY_PATH` in its public API.

### 7. Buffer and memory behavior

An instrumented upstream run retained the following state at the end of the 63.24-second sample:

| Buffer | Observed retention |
| --- | ---: |
| Audio | 160,000 samples, exactly the configured 10-second maximum |
| Track 1 mouth frames | 1,201 frames / 15,065,344 array bytes |
| Track 2 mouth frames | 1,201 frames / 15,065,344 array bytes |
| Total retained mouth-image array payload | 30,130,688 bytes |
| Peak RSS without working speaker embeddings | 573,882,368 bytes |
| Peak RSS with working speaker embeddings | 674,807,808 bytes |

The audio buffer uses `deque(maxlen=16000 * 10)`. The video buffer is a
`defaultdict(list)` and has no duration or frame limit. Tracks are only deleted after they have not
been updated for a configured wall-clock age. A continuously visible face therefore grows for the
entire video.

At the observed two-face rate, mouth-image payload alone grows by roughly 27 MiB per minute. Python
list/tuple/timestamp overhead and face-detector history add more. Long dataset videos require one of:

- time-bounded per-track deques;
- pruning frames older than the audio evaluation window;
- explicit chunk evaluation and reset with controlled overlap;
- a backend change that returns streaming observations without retaining the complete track.

### 8. Exact-25-FPS frame loss

The speaker detector attempts to normalize high-frame-rate input by retaining a frame only when:

```text
create_time - last_timestamp >= 1 / 25
```

The upstream demo generates `create_time = process_start + frame_index / 25`. Because binary
floating-point differences are sometimes fractionally below 0.04, an exact 25 FPS input retained
only 1,201 of 1,581 available frames per face: approximately 76% of the intended visual sequence.

This is especially serious because LR-ASD later interprets retained frame count using a fixed 25
FPS time base. TalkingFaceKit must implement a deterministic media-time resampler or correct this
logic upstream. It must not pass source frames through this comparison unchanged.

### 9. VAD repeatability

DeepTalk's Silero wrapper resets recurrent VAD state every five seconds using `time.time()`. That is
wall-clock execution time, not audio sample count or source media time.

Repeated runs of the same upstream sample produced different results:

- one instrumented run observed 28 `TURN_CONFIRMED` transitions and 27 `TURN_END` transitions;
- the speaker-embedding-enabled run observed 30 `TURN_CONFIRMED` transitions and 29 `TURN_END`
  transitions;
- several transition timestamps shifted by 40 to 160 ms.

The unfinished final turn explains one fewer end than confirmation in each run. The difference in
total turn count demonstrates that offline output can depend on machine speed and processing load.
VAD reset and flushing must be based on media samples/timestamps for deterministic dataset analysis.

### 10. Score semantics and ambiguity

DeepTalk documents `evaluate()` as returning `track_id -> average score`, where positive means
speaking and negative means not speaking. The experiment confirms these are not probabilities:

- values were negative for some tracks;
- values exceeded 1.0 frequently;
- the largest observed score without speaker embeddings was approximately 2.79;
- the demo simply treats every positive track as speaking and picks the highest positive value as
  primary.

In the 55 paired evaluations from one two-face run:

- face 1 had a positive value 40 times;
- face 2 had a positive value 48 times;
- therefore at least 33 of 55 evaluations, or 60%, had both faces positive simultaneously.

With speaker embeddings working, each face was positive in 51 of 59 evaluations, implying at least
43 of 59 evaluations, or approximately 73%, had both faces positive simultaneously.

The report layer must retain `raw_score` and backend provenance. A calibrated probability must not
be invented. TalkingFaceKit will need an explicit selection policy using the top score, top-versus-
second margin, temporal smoothing, VAD boundaries, minimum durations, and an ambiguous/none state.
Thresholds must be calibrated on labeled client examples.

### 11. Failure behavior

The Python factory catches many exceptions, logs a traceback, and returns `None`. This conflicts with
TalkingFaceKit's rule that boundary failures must raise clear exceptions rather than hide failure in
`None`.

Some native failures are worse. Constructing InspireFace with a nonexistent explicit resource path
terminated the interpreter with exit code 134 before Python could recover. This means validation in
the same process cannot guard against every malformed or incompatible native asset.

For an initial experimental integration, a child process provides valuable containment:

- the parent library can convert a crash into a structured backend failure;
- the DeepTalk environment can retain NumPy 1.26.4 independently;
- models can be loaded once in a long-lived worker;
- stdout/stderr noise can be captured rather than polluting library consumers.

Process isolation does not solve timestamp, buffer, licensing, or score-policy problems; those still
need explicit fixes.

## Upstream implementation quality observations

The following findings affect maintainability:

- No automated test files were found in the downloaded upstream repository snapshot.
- Public code is largely untyped and much of its documentation and logging is Chinese.
- Importing its logger configures global Python logging at `DEBUG` level.
- The video demo performs complete-audio extraction through an FFmpeg subprocess before video
  processing instead of streaming both source timelines.
- The video demo uses OpenCV and `frame_index / fps`, losing source PTS and variable-frame-rate
  behavior.
- `VideoFrame.get_plane()` and `VideoFrame.convert()` are declared but have empty implementations.
- Face detection catches broad exceptions and returns an empty list, which can make inference errors
  look like valid no-face frames.
- Factory annotations promise an `ASDInterface`, while runtime failure paths return `None`.
- Model-specific time bases, colors, shapes, and score semantics are not expressed through a typed
  stable result schema.
- The package was first released recently and should be treated as an emerging integration rather
  than a mature compatibility layer.

These do not invalidate the pretrained models. They mean TalkingFaceKit should use a narrow adapter,
tests, validation, and explicit provenance instead of exposing DeepTalk objects in its public core.

## Exit-criteria assessment

| Criterion | Result | Notes |
| --- | --- | --- |
| Installs with Python 3.11 and `uv` | Pass | Explicit Python 3.11 pin required |
| Constructs face, VAD, and LR-ASD components | Pass | Valid explicit model paths required |
| Produces face tracks and active-speaker scores | Pass | Two-face and one-face samples succeeded |
| Runs a second time without network | Pass | All inference runs used offline mode |
| Avoids silent partial initialization | Fail | Speaker embeddings silently disabled on macOS |
| Avoids process-level crashes | Fail | Invalid InspireFace path caused exit 134 |
| Has reasonable CPU throughput | Pass | Approximately 69–72 FPS on 640×360 two-face demo |
| Has bounded offline memory | Fail | Per-track video buffers are unbounded |
| Preserves correct media timing | Fail | Synthetic timestamps and exact-25-FPS frame drops |
| Produces deterministic offline VAD | Fail | Wall-clock state reset changed repeated results |
| Coexists with current project dependencies | Fail by default | Exact NumPy conflict; forced override worked in one test |
| Models and hashes are understood | Pass for identity | Six hashes and sizes recorded |
| Academic thesis use is compatible with known terms | Pass for current scope | Record the default InspireFace restriction in provenance |
| Accuracy is validated for client data | Not evaluated | Requires labeled representative clips |

## Recommended decision

Proceed with DeepTalk-ASD as an **experimental implementation source and proof-of-concept backend**,
not as an unquestioned package-level foundation.

The models and basic pipeline are promising enough to justify the next vertical slice. The package
wrapper needs containment and corrections before it can support large, arbitrary client datasets.
The fastest safe direction is:

1. Keep TalkingFaceKit's core backend-independent.
2. Implement timestamped audio decoding with the project's existing PyAV dependency.
3. Preserve original video and audio timestamps publicly.
4. Convert RGB video and 16 kHz mono `int16` audio only at the DeepTalk boundary.
5. Mark the DeepTalk integration experimental.
6. Initially run the backend in a controlled worker process or otherwise document the native-crash
   risk and validate every model path before launch.
7. Treat speaker embeddings as an explicit capability with status, not a silent optional success.
8. Use deterministic source-time scheduling, bounded windows, and explicit end-of-stream flushing.
9. Preserve DeepTalk values as raw scores and build the segment policy in TalkingFaceKit.
10. Retain InspireFace for the noncommercial thesis; revisit it only if the use case becomes
    commercial or redistribution requirements change.

## Required fixes or upstream questions before production adoption

### Packaging

- Ask upstream to relax or justify `numpy==1.26.4`.
- Add tested dependency bounds for OpenCV, ONNX Runtime, InspireFace, and Sherpa-ONNX.
- Fix or document the macOS Sherpa/ONNX Runtime library linkage.
- Publish an explicit platform/Python support matrix matching native wheels.
- Make speaker embeddings truly optional or fail construction when they were explicitly required.

### Timing and determinism

- Replace wall-clock VAD reset with audio-sample/media-time state management.
- Replace the floating-point frame-throttle comparison with deterministic resampling.
- Accept real source PTS rather than requiring synthetic `perf_counter` timestamps.
- Define behavior for video below 25 FPS, above 25 FPS, and variable frame rate.
- Flush or reject an unfinished VAD turn at EOF explicitly.

### Memory and errors

- Bound or prune per-track video buffers.
- Stop broad exception handling from turning detector errors into no-face observations.
- Raise typed initialization failures instead of returning `None`.
- Validate model manifests before entering native code.
- Determine whether InspireFace can still abort for corrupt-but-existing resources; use process
  isolation if native failure cannot be converted to a Python exception.

### Output semantics

- Document the mathematical range and calibration of LR-ASD scores.
- Expose per-window or per-frame scores rather than only an average per track and evaluation call.
- Expose VAD probability or evidence alongside state transitions.
- Provide explicit capability/provenance information for face detector, VAD, ASD, and voiceprint.

### Licensing and provenance for any future distribution

- If the project later becomes commercial, identify another face detector/tracker or obtain the
  appropriate InspireFace model rights.
- Record the original checkpoint and license for every converted ONNX file.
- Confirm distribution rights if TalkingFaceKit ever automates model download or redistributes an
  asset manifest.

## Proposed audio step — completed; adapter is next

The implementation did **not** begin with a folder-level sequence container. It added the missing
permanent media primitive required by every audio-visual backend:

```text
DecodedAudioChunk
    source start timestamp in seconds
    sample rate in Hz
    channel layout / channel count
    explicit sample dtype and shape

stream_audio_chunks(path, start_seconds, end_seconds)
    PyAV-based streaming
    original source timeline
    half-open interval semantics matching stream_video_frames
    no complete-track materialization
```

The next change is a narrow DeepTalk adapter that can:

- transform chunks to 16 kHz mono signed `int16` explicitly;
- feed video and audio in timestamp order;
- map source timestamps into a controlled monotonic coordinate only inside the adapter;
- evaluate bounded utterance windows;
- collect face ID, bounding box, VAD state, and raw active-speaker score;
- map all results back to source seconds;
- write a minimal one-video JSON report and annotated demonstration;
- label direct-to-camera and exact A/V sync as `not_evaluated` until their dedicated stages exist.

That vertical slice demonstrates client-visible progress while preserving the architecture already
built in TalkingFaceKit. Only after it works on labeled representative clips should it be generalized
into folder/collection processing.

## Reproduction outline

The spike used `uv` exclusively. The conceptual sequence was:

```bash
uv init --bare --python 3.11 --name talkingfacekit-deeptalk-spike
uv python pin 3.11
uv add deeptalk-asd==0.3.1

DEEPTALK_ASD_CACHE_DIR=<models> \
  uv run python -m deeptalk_asd download-models --cache-dir <models>

DEEPTALK_ASD_OFFLINE=1 \
DEEPTALK_ASD_CACHE_DIR=<models> \
INSPIREFACE_RESOURCE_PATH=<models>/Pikachu \
  uv run python <upstream-video-demo> \
  --input <video> \
  --output <temporary-output> \
  --model-dir <models>
```

The NumPy 2 experiment added this disposable-only override:

```toml
[tool.uv]
override-dependencies = ["numpy==2.4.6"]
```

That override is an experiment result, not a recommendation to add the same setting to
TalkingFaceKit without broader compatibility tests.
