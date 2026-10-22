# Experimental visual selection — 2026-10-22

## Scope and implementation

The optional `--visual-filter` stage interprets existing InspireFace head pose and source-pixel
boxes, without adding inference, dependencies or ocular gaze estimation. It follows structural
selection and optional SyncNet verification. Raw evidence and upstream decisions remain unchanged.
The backend-independent core is `talkingfacekit.visual`; CLI orchestration and both HTML reports
expose policy, per-time reasons, trimmed decisions and before/after selection.

Original inclusive defaults (superseded below): absolute yaw ≤25°, pitch ≤20°, roll ≤45°; shorter box side ≥96 pixels;
visible width / source width ≥0.08; outside-image box area fraction ≤0.05. Bounded failures up to
0.20 seconds can be retained between passing observations. Remaining runs require two seconds,
90% measurable time and 90% time passing all geometry checks. Missing measurements remain
uncertain. Image quality Q is not thresholded; clipping does not establish absence of occlusions.

## First fixed-label benchmark

Command, using already available models and unchanged versioned human annotations:

```bash
uv run --extra active-speaker-deeptalk --extra synchronization-syncnet \
  python -m talkingfacekit evaluate-benchmark benchmark/manifest.json \
  --videos-dir data/eval/videos --syncnet-model models/syncnet_v2.model \
  --visual-filter --output reports/visual/benchmark-run01.json
```

Both selections come from the same DeepTalk/SyncNet inference. Five videos, 398 evaluated seconds:

| Metric | Before visual filtering | After visual filtering |
| --- | ---: | ---: |
| Correctly selected seconds | 83.68 | 41.05 |
| Incorrectly selected seconds | 6.93 | 3.40 |
| Useful seconds missed | 4.32 | 46.95 |
| Precision | 92.3518% | 92.3510% |
| Recall | 95.0909% | 46.6477% |

The initial settings remove approximately equal proportions of useful and incorrectly retained
material, leaving precision essentially unchanged while substantially reducing recall. These
results do not validate the initial thresholds as a suitable default for human-defined usefulness.
The filter remains explicitly enabled; existing commands without the flag retain their selection.

`Piers_Sellers_JSC_2016_interview` loses 29.21 previously correctly selected seconds, principally
because yaw exceeds 25°. Its raw failing yaw windows total 22.96 seconds; additional losses occur
when trimming leaves fragments too short or with insufficient passing time. `How-to-start` loses
all 11.50 useful seconds, with both yaw and the 96-pixel requirement contributing. These causes
can overlap and their durations must not be added as disjoint errors. `FactsMatter2016` loses
1.92 previously useful seconds and removes 0.16 incorrectly selected seconds. The two videos
without surviving SyncNet candidates stay unchanged.

The comparison exposes **Antes visual**, final predictions, metrics recalculated from stored
intervals, raw visual reasons and exact tolerated gaps. Local generated artifacts remain ignored:
`reports/visual/comparison.html`, `reports/visual/benchmark-run01.json`, and the Piers diagnostic
HTML/JSON. Source hashes, manifest hash, code revision and complete applied policy remain recorded.

## Interpretation and next evaluation

This is a small regression set measuring temporal usefulness, not calibrated head-angle accuracy
or ocular eye contact. Before changing the recommended limits, inspect the lost useful intervals
against video and clarify how much lateral orientation is acceptable for the downstream training
task. Keep annotations fixed; record alternate tolerances as separate experiments and reserve
unseen videos for validation. A lower angular error or a more sophisticated gaze estimator alone
does not establish better dataset selection.

Core tests cover signed-angle limits, inclusive boundaries, size versus framing, clipping on all
edges, brief failures, repeated failures, missing pose/face evidence, source-grid boundaries,
minimum durations, hard shot/identity boundaries and preservation of upstream rejections. CLI
tests verify JSON, same-inference benchmark comparisons and actual generated report JavaScript.
They do not download models. The real benchmark above exercises the installed backends and assets.

## Criterion revision: useful three-quarter views

The user clarified that moderate profile views are useful for training; strict frontal orientation
is not required. Direct review of source frames supports this distinction: Piers at 21 seconds
has measured yaw about 25.8° and a clear useful face; at 45 seconds its three-quarter view is
about 52°. In How-to-start at 117 seconds, a valid face box has a 93-pixel shorter side and yaw
about 28.5°. Both the original angle and pixel-size assumptions excluded useful material.

The original loss of 42.63 useful seconds decomposes into 31.71 seconds failing geometric checks
and 10.92 seconds that passed those checks but were rejected after fragmentation by minimum
duration or passing-time rules. These are disjoint durations within the fixed human selections.
This is a policy mismatch, not evidence that the underlying pose estimator or SyncNet is broken.

The revised default is yaw ≤70°, pitch ≤40°, roll ≤45°, shorter box side ≥64px and visible face
width fraction ≥0.04. The bounds on clipping, missing evidence, brief gaps and minimum duration
are unchanged. A 70° threshold admits useful lateral views, while brief excursions remain
recorded and bounded. Extremely lateral views, insufficient source detail, missing pose and
upstream rejections still cannot silently pass. Strict 25°/20°/96px settings remain configurable.

Saved complete analyses for Piers, How-to-start and FactsMatter were reused to evaluate explicit
alternative policies. Their source hashes, local track IDs and pre-visual selected intervals
match the initial benchmark. The other two videos have no surviving sync candidates, so changing
visual thresholds does not affect them. Replaying the original thresholds reproduces the initial
metrics. No annotation or upstream selection was changed.

| Configuration | Precision | Recall | Useful seconds retained |
| --- | ---: | ---: | ---: |
| Before visual filtering | 92.35% | 95.09% | 83.68 |
| Original strict visual policy | 92.35% | 46.65% | 41.05 |
| Relax angles only, keep 96px and 0.08 width | 92.77% | 87.39% | 76.90 |
| Revised default, including 64px and 0.04 width | 92.43% | 95.09% | 83.68 |

The revised policy retains all previously selected useful seconds in this regression set and
removes 0.08 incorrectly selected seconds (6.85 rather than 6.93 seconds remain). This is an
exploratory criterion revision on already inspected data, not an independent accuracy claim.
Validation on unseen lateral and extreme-pose material remains necessary before calibration.

Updated ignored artifacts: `reports/visual/benchmark-run02.json`, `policy-comparison.json`,
`comparison-run02.html` and the `*-run02.html` diagnostics. The replay records the source analysis
paths and their hashes. The original strict outputs remain available for comparison.
