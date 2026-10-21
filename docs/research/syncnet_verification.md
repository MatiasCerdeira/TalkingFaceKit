# SyncNet candidate verification — 2026-10-21

## Decision and scope

DeepTalk-ASD 0.3.1 continues to supply speech intervals, face tracks, raw LR-ASD scores and
structural candidates. The optional SyncNet-v2 stage verifies every surviving candidate because
there is no validated ASD-score trigger that identifies which candidates might be out of sync.
Confirmed offsets reject a candidate; insufficient evidence excludes it from automatic selection
as uncertain. Passing sync does not certify the pending orientation and visual-quality criteria.

Reuse DeepTalk track IDs and bounding boxes, PyAV source timestamps and existing scene boundaries.
Do not install the complete `syncnet-python` pipeline: it duplicates face detection and constrains
PySceneDetect below the version used by TalkingFaceKit. Instead adapt the original MIT-licensed
network and cropping convention and keep PyTorch behind the `synchronization-syncnet` extra.
No video, model weights or generated reports are committed. The source is never rewritten.

## Model and reproducibility

- Original model: <https://github.com/joonson/syncnet_python>.
- Explicit official checkpoint: <https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model>.
- SHA-256: `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`.
- Local evaluation: Python 3.11.15, macOS Apple Silicon, CPU, PyTorch 2.14.1, NumPy 1.26.4.
- Model inputs: original unscaled BGR crops of 224×224 at 25 Hz and mono int16 PCM at 16 kHz.
- State-dict loading is strict and tensor-only, after hash verification; inference uses eval mode.
- Adapted and original networks produced exactly identical visual and audio embeddings on the
  same synthetic inputs and checkpoint (maximum absolute difference 0.0 for both towers).

Each tower runs once per window. Search then compares embedding distances at relative lags;
it does not rerun neural inference for each possible offset. Unlike upstream's zero padding,
the search uses the same interior visual frames for all lags. Confidence is median distance
minus best distance and is not a probability; the preprocessing/search difference requires local
threshold validation rather than importing an upstream confidence threshold as a guarantee.

The final implementation advances each source stream once across ordered windows, keeping only
the current window and boundary frames/chunks. It reads original audio timestamps before clipping:
millisecond-rounded Vorbis/WebM PTS can otherwise produce equal trimmed starts at a candidate
boundary (observed around source time 42.08 in FactsMatter2016). Substantive gaps/overlaps above
1 ms remain uncertain. Video/face gaps also cannot silently pass.

## Controlled delay experiment

Use the source Piers_Sellers_JSC_2016_interview.mp4 from 20 to 25 seconds. Keep its face crops
fixed and introduce known delays directly into its PCM samples. Positive public offset means
late audio. These results check recovery relative to the original clip, whose absolute timing
has not been independently annotated as zero.

| Injected delay | Estimated total offset | Confidence | Difference from original estimate |
| --- | --- | --- | --- |
| 0 ms | +80 ms | 8.20 | 0 ms |
| −320 ms | −240 ms | 8.31 | −320 ms |
| −160 ms | −80 ms | 8.23 | −160 ms |
| +160 ms | +240 ms | 8.31 | +160 ms |
| +320 ms | +400 ms | 8.28 | +320 ms |

The model recovered each added shift exactly at its 40 ms frame resolution. Warm feature
inference took roughly 0.96–1.32 seconds per five-second window in this initial experiment,
excluding decode, cropping and model initialization. This is a small local experiment, not a
throughput guarantee. The reproducible opt-in regression is `tests/test_syncnet_e2e.py`; see
README for its environment variables. Normal tests neither download weights nor run this model.

## Selection policy and real-video outcome

Initial explicit experimental thresholds: absolute offset ≤80 ms, raw confidence ≥3.0,
windows of at most five seconds with a two-second minimum, search up to ±1 second. Short
windows search less to retain at least one second of common paired evidence. A best match on
the search boundary is uncertain; a weak or ambiguous match is never declared synchronized.
Every candidate is completely partitioned. One confidently out-of-tolerance window rejects
the candidate; all windows must be confidently in tolerance to keep it.

With the same five source hashes, human labels and 398 evaluated seconds as the first baseline:

| Metric | Structural baseline | With SyncNet verification |
| --- | --- | --- |
| Correctly selected seconds | 83.68 | 83.68 |
| Incorrectly selected seconds | 23.87 | 6.93 |
| Useful seconds missed | 4.32 | 4.32 |
| Precision | 77.81% | 92.35% |
| Recall | 95.09% | 95.09% |

Entrevista_01.webm previously contributed 16.94 incorrectly selected seconds. Verification
excluded all four candidates: 6–13 and 31.83–34 were rejected on confident offsets, while
16–18 and 26–31.77 were uncertain. Within 6–13, both windows estimated −560 ms, with
confidence 3.54 and 4.32. The last short candidate estimated −360 ms with confidence 3.08.
Other uncertain windows had diagnostic estimates with weak confidence; they are not confirmed
offsets. Do not describe the entire file as having one proven constant shift.

All candidates from the other four videos were retained. Their remaining discrepancies are
outside this sync stage. No human annotations were edited to obtain these results. This small
set is a regression baseline and cannot establish general model accuracy or calibrate every
language, camera angle, mouth occlusion, shift magnitude or source format.

## Local end-to-end cost

The final single-pass run (`reports/syncnet/benchmark-run02.json`) reproduced every prediction
and estimated window offset from the earlier repeated-decode run. Local elapsed times were:

| Video/window | DeepTalk | Added sync verification |
| --- | ---: | ---: |
| A_Flute_on_the_Space_Station, 0–90 s | 60.81 s | 0.00 s; no candidates |
| Entrevista_01, 0–38 s | 16.81 s | 5.18 s |
| FactsMatter2016, 30–120 s | 65.75 s | 9.90 s |
| How-to-start, 90–180 s | 30.36 s | 4.37 s |
| Piers_Sellers_JSC_2016_interview, 0–90 s | 41.98 s | 15.78 s |
| Total | 215.72 s | 35.23 s |

Model construction took another 0.59 seconds in this warm local run. Verification added about
16.3% of the time spent in the DeepTalk stages; total measured model stages were 251.54 seconds
for 398 seconds of source media. These are indicative same-host measurements, not a controlled
cross-machine benchmark; initialization, decoding, other host activity and candidate density vary.
The initial model import had taken several seconds on a colder process.

The repeated-decode prototype had spent 70.02 seconds in verification. Reusing the stream
iterators cut that to 35.23 seconds without changing offsets or decisions. In particular,
FactsMatter fell from 23.11 to 9.90 seconds and Piers from 33.13 to 15.78 seconds. This is why
the integration reuses existing track boxes and performs a single source scan per stream.

Final demo artifacts are `reports/syncnet/comparison.html`, `reports/syncnet/entrevista.html`
and `reports/syncnet/entrevista.json`. They reference local original media and stay outside Git.

## Deferred correction

The JSON stores original-media hash, checkpoint identity, policy and all per-window estimates.
Positive values would require advancing audio in a future correction step. An aggregate offset
is supplied only when all windows are measurable and their reliable offsets are consistent.
Changing offsets and unmeasurable regions preserve uncertainty; one global track shift must not
be inferred from them. Automatic correction is outside this implementation.
