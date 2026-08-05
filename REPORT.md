# Group A implementation report

Implementation of `README_GroupA.md` (items A1–A6) for *"Verification-Driven
Post-Training Quantization of Deep Emotion Recognition Models for Raspberry Pi
Deployment"* (Sensua, Ahirwar, Kaur).

**Started:** 2026-07-31 · **Completed:** 2026-08-02 · **Machine:** RTX 4060
workstation, Windows 11, Python 3.10, TensorFlow 2.21.0, torch 2.6.0+cu124,
onnx2tf 1.28.8

This is the narrative record: what was found, what was decided and why, what
broke, and what the numbers came out as. `GROUP_A.md` is the runbook; this is the
account of getting there.

**All six items are run, and then re-run on a second execution path.** Late in the
study it emerged that `ai-edge-litert` 2.1.4 applies **no delegate unless asked**,
so every quantized measurement had been taken on TFLite's reference kernels rather
than the XNNPACK path a deployment runtime uses. On FER that is worth up to 10.6
accuracy points and ~80× speed. §7.6 reports both paths; the reference-kernel
numbers are kept, not replaced, because they are what the original measurements
were. **Where the two disagree, cite the delegated path** — it is the one that
corresponds to deployment.

The five findings the paper should carry:

1. **Dynamic range is the deployment answer** on both models — no statistically
   detectable accuracy difference from FP32 (p = 1.0 paired, not merely an
   overlapping interval) at ~4× compression. Accuracy, note, not behaviour: it
   still disagrees with FP32 on 4.9% of FER test images.
2. **Full INT8 collapses**, but by how much depends on both calibration and
   execution path. On FER the deployment-path penalty is **−10.7 points** (not
   the −21.25 measured on reference kernels), and it still varies 8.8 points
   across 18 calibration draws. Cite it as mean ± SD over seeds, on the delegated
   path (§7 A4, §7.6).
3. **The collapse is distributed, not localised.** A2 finds the largest error in
   the squeeze-and-excitation pools; A3 shows exempting those layers recovers
   almost nothing — **+0.98 points at best** on the deployment path, across
   k = 3 to 15 (§7 A3, §7.6).
4. **QAT fixes it on SER** — indistinguishable from FP32 at INT8 size — but
   dynamic range already achieves that in less space, so QAT matters only for
   targets that cannot run float activations at all. **And most of QAT's margin
   over post-training quantization is just the extra training**: fine-tuning the
   float model for the same 15 epochs and then quantizing recovers +5.83 of the
   +10.42 (p = 0.013), leaving QAT's own contribution at +4.58 and not
   significant (§7 A5-control).
5. **The format recommendation is architecture-dependent once latency is real —
   on the runtime version measured.** On SER, dynamic range is both the most
   accurate and the fastest build. On FER it is the most accurate and the
   *slowest* — 216 ms against full INT8's 7.7 ms — because **in ai-edge-litert
   2.1.4** XNNPACK's dynamic-range kernel coverage does not extend to the
   depthwise convolutions EfficientNet-B0 is built from. This is a version
   coverage gap, not a property of the format: a later XNNPACK that adds hybrid
   depthwise support would dissolve it (§7.6).

---

## 1. Objective and starting state

The task was to implement `README_GroupA.md`, which maps items A1–A6 onto five
scripts said to be "already delivered in `paper-fix.zip`".

The initial survey of `D:\Research FER and SER` found the two corrected
notebooks, `outputs/FER/`, `outputs/SER/`, and the README — **no `quant/`
directory and no `paper-fix.zip`**. A filesystem-wide search for the zip found
nothing. On that basis the task looked like writing the whole package from
scratch.

The zip was then supplied at `D:\Research FER and SER\paper-fix.zip` (created
16:14, a few minutes after the initial scan, which is why it was missed). It
contained 22 files: the five `quant_*` scripts, `common.py`, `config.py`, a
`pi/` benchmark, FER/SER training and audit scripts, and a report generator.

**This changed the task from "write" to "integrate, correct and run".**

### The gap that made integration real work

The delivered scripts were written against **their own** training pipeline. They
expected:

| Expected | Actually available |
|---|---|
| `artifacts/models/fer_baseline.pth` | `outputs/FER/jugantarFER_corrected.pth` |
| `artifacts/models/ser_baseline.keras` | `outputs/SER/jugantarSER_corrected.keras` |
| `artifacts/cache/fer_split.json` as `{"files": {...}}` with resolvable paths | `outputs/FER/fer_split.json`, flat, with Colab-relative paths |
| `artifacts/cache/ser_test.npz` | `outputs/SER/ser_test_set.npz` |
| `artifacts/cache/ser_train.npz` | **did not exist anywhere** |

Group A's rule 1 is that the test sets are frozen to what the *corrected
notebooks* produced. So the scripts could not be pointed at their own pipeline's
outputs; they had to be rewired to the corrected artifacts. That bridge is
`quant/quant_00_prepare.py`, which is not in the README's item table and had to
be written.

The SER training features were the sharpest instance: the notebook saved only
the test set, but calibration must come from **training** data (rule 2). So the
training spectrograms are rebuilt from RAVDESS actors 1–16 using the notebook's
exact `get_spectrogram`, and the rebuild is asserted to contain no val/test
actor.

---

## 2. Environment discovery

The default `python` on PATH (3.14) had nothing installed — not even numpy. Three
other interpreters existed; **Python 3.10** turned out to have the entire stack:
TensorFlow 2.21, torch, onnx2tf, onnxruntime, librosa, OpenCV, sklearn.

Both raw datasets were also on disk, which was not a given:

- FER: `D:\FER TFlite\archive (1)\dataset` — all **13,008** paths in
  `fer_split.json` resolve against it (folder totals 13,014; the 6 extra are the
  cross-class-cluster images the notebook dropped).
- RAVDESS: `D:\SER Tflite\ravdess-emotional-speech-audio`, actors 1–24 complete.

**Consequence: Group A could actually be run, not merely written.** That is why
this report contains measurements rather than only code. Machine-specific paths
live in a gitignored `paths.local.json`, so nothing machine-specific is committed.

---

## 3. Defects found in the prerequisites

### 3.1 The committed FER ONNX was unusable

`outputs/FER/jugantarFER_corrected.onnx` was **605 KB** — far too small for
EfficientNet-B0's ~4M parameters. It stored its weights in an external
`jugantarFER_corrected.onnx.data` sidecar that was never committed, so
`onnx.load` failed outright on `features.0.0.weight`.

The README lists this file as a Group A prerequisite, so the item was blocked as
committed. The `.pth` checkpoint is complete, so `quant_01` now re-exports a
self-contained ONNX (`export_params=True`) and verifies it against PyTorch before
anything downstream runs. **The repository artifact was replaced** with that
export: 16.0 MB, 3,991,664 parameters, loads standalone.

### 3.2 The SER checkpoint would not load

`jugantarSER_corrected.keras` was written by a newer Keras (Colab) than the
workstation's 3.12.2 and embeds `quantization_config` in its layer configs, which
older Keras rejects with a hard `TypeError`. A `custom_objects` shim did not
help — Keras resolves built-ins by registered module name and ignored it.

`common.load_keras_compat()` instead rewrites `config.json` **inside** the
`.keras` zip to drop the unknown key and loads the patched copy. Weights are
untouched; the key is `None` for every layer in this model. Results record
`keras_config_patched: true` whenever this path is taken.

### 3.3 A2's error-column mapping would have produced a meaningless ranking

The delivered `quant_03` looked for `rmse` and `scale` columns and, failing that,
fell back to "the last numeric column". In TensorFlow 2.21 the
`QuantizationDebugger` emits:

```
op_name, tensor_idx, num_elements, stddev, mean_error, max_abs_error,
mean_squared_error, scale, zero_point, tensor_name
```

There is **no `rmse` column**, so the fallback would have fired and ranked layers
on `zero_point` — a plausible-looking ranking that means nothing, which then
feeds A3's denylist. The README explicitly warns about this ("if it fell back to
a heuristic column, fix the column mapping rather than proceeding").

Columns are now resolved from explicit candidate lists, RMSE is derived as
`sqrt(mean_squared_error)`, the resolution is recorded in the output under
`column_resolution`, and the script **exits** rather than guessing. `quant_05`
refuses to consume a ranking whose source was the heuristic path.

### 3.4 FER preprocessing was subtly wrong

`torchvision.transforms.Resize` defaults to **bilinear**; PIL's `Image.resize`
defaults to **bicubic**. The delivered code used PIL's default, which would have
evaluated every quantized format on subtly different pixels than the notebook
reported its 82.49% on.

`config.FER_RESIZE_FILTER` pins bilinear, and `quant_00` verifies the PIL path
against the real torchvision transform: **max absolute difference 0.0**.

---

## 4. Defects in my own work, caught by testing

Recorded because both would have put false claims in the paper.

### 4.1 The SE regex matched TFLite's `Squeeze` operator

The first layer-wise run reported *"4 of the top 15 layers match the
squeeze-and-excitation pattern — Section 5.3 is supported"* for the **SER**
model, which is a plain Conv/BN/MaxPool CNN with **no SE blocks whatsoever**.

The pattern was matching `Squeeze1` in names like
`conv2d_1_2/convolution;sequential_1/conv2d_1_2/Squeeze1` — TFLite's ordinary
rank-reduction operator, unrelated to squeeze-and-excitation.

Fixed: bare `squeeze` never matches; `se` is only accepted as a whole path
segment (so `sequential` cannot match). Verified against a table of positive and
negative cases.

### 4.2 A5's weight transfer silently kept random weights

`load_baseline_weights()` walked `_layer_checkpoint_dependencies`, which is not
where a `.keras` archive stores weights. Every layer kept its random
initialisation and the rebuilt model scored **26.67%** 4-class against the
published 70.83%. It only warned.

The real layout is `layers/<layer_name>/vars/<i>`, with `i` already in Keras's
expected order. Fixed, plus per-layer shape validation and a hard raise instead
of a warning. The rebuild now reproduces the baseline **exactly**.

This was caught only because `quant_06` verifies the transfer against the
published number *before* training. Without that guard, QAT would have
fine-tuned from noise and produced a meaningless A5 result.

### 4.3 `quantize_model()` cannot wrap this architecture at all

Found when A5 was finally run. `tfmot.quantization.keras.quantize_model()`
raises on every `BatchNormalization` layer in the SER net:

```
RuntimeError: Layer batch_normalization:<BatchNormalization> is not supported.
```

tfmot only handles BN fused into a preceding Conv2D. The corrected architecture
puts the ReLU *inside* the convolution — `Conv2D(..., activation="relu")` — and
BN after it, so no BN in this model is fusable. The obvious fix, reordering to
Conv → BN → ReLU, is a different network and rule 3 forbids it.

Instead each layer is annotated individually and BN gets an explicit
`QuantizeConfig` that quantizes its output to 8 bits and leaves gamma/beta in
float, then `quantize_apply()` builds the model. Verified before use: the wrapped
model keeps the transferred weights (checked on a toy net with distinctive
weights, then on the real model), and the converted graph contains int8 tensors.

A second guard was added at the same time, because the first one nearly misfired.
After `quantize_apply()` but before fine-tuning, the model scores **26.67%**
4-class — the same number §4.2's bug produced, and easy to mistake for it. It is
not a bug: tfmot's `MovingAverageQuantizer` starts with default activation ranges
that clamp everything until training adapts them, and one epoch restores full
accuracy. The script now logs this checkpoint explicitly so the two failure modes
are distinguishable.

### 4.4 Two ways the re-runs would have silently mixed execution paths

Found while setting up §7.6's delegated re-runs, and both are the same species of
bug as the delegate problem itself — no error, just a plausible wrong number.

**The prediction cache did not know about the delegate.** `cached_predictions()`
keys on the model file's mtime and size. A delegated run of A3 would have rebuilt
the selective models (new mtime, cache miss, correctly recomputed) while pulling
the FP32 baseline and the reference formats it compares against from the
*reference-kernel* cache — one result file, two execution paths, no warning. The
key now carries an `_xnn` suffix.

**Staging then picked the wrong cache.** `quant_07` takes the newest cache per
format. Once `_xnn` entries existed they would be newest, so the next
reference-path McNemar staging would have silently ingested delegated
predictions. It now filters to caches matching the current path.

**And a crash worth recording rather than tidying away.** When A3's k=5 build
turned out to fail delegate preparation, the first fix recorded a null-accuracy
row but left `max(runs, key=...)` and the plot to consume it, so the sweep
completed all four builds and then died in the summary. Two runs were lost to
that. The lesson is mundane and general: adding a sentinel value means finding
everything that reads it.

A second lesson arrived later, and it is the more expensive one. The sentinel was
treated as a *finding* — the failure had been seen three times, so it was written
up as reproducible and a deployment-risk conclusion was built on it. All three
sightings were in one process, and the process was the variable that mattered.
See §7.6; the conclusion is withdrawn and the k=5 row is now measured.

---

## 5. Decisions

**A4 — the corrected variable.** The README's "one variable needs changing" is
the third calibration variable, originally "FER2013-only vs. mixed
FER2013+ExpW". Dropped: the corrected FER model is trained on the Kaggle
`sujaykapadnis/emotion-recognition-dataset` only, so calibrating from FER2013 or
ExpW would sample a distribution the model never saw — that measures domain
shift, not calibration sensitivity. The delivered script already swept seeds and
carried no dataset-source variable, so the correction was made **explicit**: the
reasoning is written into every results file under `dropped_variable_note`.

**A2 — structural, not name-based, hypothesis testing.** `onnx2tf` rewrites node
names to generic forms (`model/tf.math.reduce_mean_7/Mean`), so SE blocks cannot
be found by name in the FER graph at all. The debugger's `op_name` column gives
the TFLite **op type**, which can. `MEAN` is the marker: it is the
global-average-pool that opens each SE module, and the graph contains exactly
**17** of them — 16 MBConv SE modules plus one head pool, matching
EfficientNet-B0 exactly.

`LOGISTIC` is deliberately **excluded** despite the SE gate being a sigmoid:
EfficientNet's SiLU activation is `x*sigmoid(x)`, so `LOGISTIC` appears 65 times
across the network and is not SE-specific. Including it diluted the signal below
significance (40% vs 33% baseline share). Enrichment is tested with an exact
hypergeometric p-value rather than an arbitrary multiplier, primarily over the
layers above the sensitivity threshold — the paper's own definition of
"sensitive" — and secondarily over the top-k.

**A5 — SER first, in an isolated environment.** Option 3 from the README. It was
deliberately not run in the main environment: `tensorflow-model-optimization`
0.8.1 pins `absl-py~=1.2`, and installing it would **downgrade absl-py from 2.x
in the environment every other Group A item depends on**. It was run instead in a
dedicated `.venv-qat`, and the main environment was checked afterwards — still
absl-py 2.4.0 and TensorFlow 2.21.0, so the downgrade stayed contained. Results
in §7; the two defects that surfaced when the `tfmot` call was finally reached
are in §4.3.

**Rules enforced in code, not assumed.** `EXPECTED_N_TEST` (1948 FER / 240 SER)
is asserted by `check_frozen()` in every evaluating script; FER splits are proven
pairwise disjoint by path; the SER calibration pool is asserted to contain no
held-out actor.

---

## 6. Verification before any result was recorded

From `outputs/quant/quant_00_prepare.json`:

| Check | Result |
|---|---|
| All 13,008 FER split paths resolve | pass |
| FER splits pairwise disjoint | pass (0 shared) |
| FER test n = 1948 | matches notebook |
| PIL preprocessing vs torchvision | max diff **0.0** |
| FER baseline reproduced from checkpoint | 82.44% vs 82.49% published (−0.05 pt) |
| SER test n = 240 | matches notebook |
| SER calibration actors | 1–16 only |
| SER training features rebuilt | 2880 samples — matches notebook |
| SER baseline reproduced | 60.83% / 70.83% — **exact** |
| ONNX vs PyTorch | 1.57e−04 (tolerance 1e−03) |
| onnx2tf SavedModel vs ONNX | 1.79e−05 |

The FER baseline is 0.05 points below the published figure — GPU
non-determinism, well inside the 0.5-point tolerance. Everything else is exact.

---

## 7. Results

Committed under `outputs/quant/`. `quant_format_evaluation.json` is the source of
truth for A1/A6.

### A1 + A6 — full held-out test sets

**FER, n = 1948.** The FP32 TFLite control lands at 82.49%, matching the
published figure; the PyTorch re-run from the checkpoint gave 82.44% (§6), a
0.05-point gap consistent with GPU non-determinism and equal to one test image
in 1948. The conversion chain is therefore clean *to within that tolerance*, and
the large drops below are quantization rather than conversion — which is the role
the README assigns the fp32 build. It is not a digit-for-digit reproduction and
should not be described as one.

Δ pts is **format − fp32**, so a negative value means the format is worse than
the FP32 control. §7.5's paired table uses the opposite convention (A − B, the
tool's argument order), which is why FER dynrange appears there as +0.05.

| format | MB | acc % | bal acc | macro F1 | kappa | agree | Δ pts (fmt − fp32) | mean KL |
|---|---|---|---|---|---|---|---|---|
| fp32 | 16.04 | 82.49 | 81.18 | 0.819 | 0.756 | 1.000 | — | 0.0000 |
| fp16 | 8.08 | 82.55 | 81.22 | 0.820 | 0.757 | 0.999 | +0.05 | 0.0000 |
| dynrange | 4.53 | 82.44 | 80.87 | 0.818 | 0.755 | 0.951 | −0.05 | 0.0176 |
| int8_full_perchannel | 4.89 | 61.24 | 56.63 | 0.567 | 0.453 | 0.618 | **−21.25** | 0.6785 |
| int8_full_pertensor | 4.17 | 33.21 | 27.17 | 0.206 | 0.038 | 0.346 | **−49.28** | 1.0616 |

**SER, n = 240, 4-class summation** (deployed protocol — publish this):

| format | MB | 4-class % | macro F1 | CI95 | 8-class % |
|---|---|---|---|---|---|
| fp32 | 1.70 | 70.83 | 0.695 | [65.00, 76.25] | 60.83 |
| fp16 | 0.86 | 70.83 | 0.695 | [65.00, 76.25] | 60.83 |
| dynrange | 0.45 | 71.25 | 0.698 | [65.42, 77.08] | 60.42 |
| int8_full_perchannel | 0.45 | 59.58 | 0.568 | [53.33, 65.83] | 50.83 |
| int8_full_pertensor | 0.43 | 62.50 | 0.605 | [56.25, 68.75] | 52.08 |

The paper's central claim holds on both models, but it needs stating precisely.
Dynamic range shows **no statistically detectable accuracy difference from FP32
(p = 1.0, paired McNemar), with 95.1% top-1 agreement on FER** — while full INT8
collapses.

The two halves of that sentence are not the same claim. Accuracy is preserved;
*per-sample behaviour is not identical*. On FER the dynamic-range build disagrees
with FP32 on **4.9% of test images** (mean KL 0.0176), and those disagreements
happen to cancel almost exactly — 44 samples gained, 43 lost. "Preserves baseline
behaviour", unqualified, claims the stronger thing and is not what was measured.

This is also an argument for the methodology rather than an embarrassment to it.
A 5% behavioural divergence is invisible to an 8-sample spot check — the expected
number of disagreements in 8 images is 0.4 — so the original verification could
not have detected it in either direction. Measuring agreement and KL on the full
test set is what makes the distinction between "same accuracy" and "same model"
observable at all.

**A6.** For FER, per-channel beats per-tensor by 28 points (61.24 vs 33.21). For
SER the point estimates run the other way (62.50 vs 59.58) — but the paired test
in §7.5 shows **that difference is not significant**, so it must not be read as a
reversal. See §7.5 for the correction and why the marginal numbers misled.

### A2 — layer-wise error profiling

**FER:** 245 ranked layers, 9 above RMSE/scale 0.5. The highest-error tensor in
the entire network is a global-average-pool (`tf.math.reduce_mean_7/Mean`, 3.84).

Section 5.3 is **supported, structurally**: MEAN ops are **4.8× enriched** among
the 9 layers above threshold (3 of 9, hypergeometric p = **0.0181**). The wider
top-15 cut is weaker (2.9×, p = 0.0750) and both are reported, because the
conclusion depends on which cut you take and the paper should say so.

**SER:** 20 ranked layers, **none** above threshold (highest 0.464, `dense_1`).
Consistent with SER's much smaller INT8 drop. The SE question is reported as
`null`/undetermined — the model has no SE blocks, so there is nothing to confirm.

### A3 — selective (mixed-precision) quantization, SER

| k kept float | MB | acc % | Δ vs FP32 | agreement |
|---|---|---|---|---|
| 3 | 0.547 | 50.83 | −10.00 | 0.762 |
| 5 | 0.549 | 50.83 | −10.00 | 0.775 |
| 10 | 0.603 | 51.67 | −9.17 | 0.771 |
| 15 | 1.702 | 58.75 | −2.08 | 0.908 |

**Full INT8 is not recovered.** k=15 regains +7.92 points over full INT8, but the
artifact is then 1.702 MB — indistinguishable from FP32's 1.70 MB, so the
compression benefit is gone. This is the negative branch the README anticipates,
and it is publishable: it strengthens the case for dynamic range as the default
(0.45 MB at baseline accuracy) and bounds how much of the INT8 failure a handful
of layers can explain.

### A3 — selective (mixed-precision) quantization, FER

Layers ranked by A2's error profile, then the top *k* kept in float while
everything else goes INT8. Reference points: full INT8 per-channel is 61.24% at
4.891 MB, dynamic range 82.44% at 4.531 MB, FP32 82.49%.

| k kept float | MB | acc % | Δ vs FP32 | Δ vs full INT8 | agreement |
|---|---|---|---|---|---|
| 3 | 4.891 | 63.19 | −19.30 | +1.95 | 0.632 |
| 5 | 4.898 | 63.14 | −19.35 | +1.90 | 0.632 |
| 10 | 4.960 | 62.68 | −19.82 | +1.44 | 0.631 |
| 15 | 5.210 | 62.83 | −19.66 | +1.59 | 0.629 |

**The curve is flat.** Freeing the three worst layers buys 1.95 points; freeing
fifteen buys less. Every build stays about 19 points below FP32, and the 95% CIs
overlap almost entirely (k=3 spans 61.09–65.20, k=15 spans 60.68–64.89), so the
apparent decline with larger k is not a real ordering — these are paired builds
on one test set and nothing here is worth claiming without a McNemar test, which
is the same trap §7.5 documents for the format comparison.

Read alongside A2, this is the important negative result. A2 found quantization
error concentrated in the squeeze-and-excitation global-average-pools (4.8×
enriched among the worst layers), which invites the inference that protecting
them should fix the model. It does not. **Where the error is largest is not where
the failure comes from** — the collapse is distributed across the network, so no
small set of exemptions recovers it.

The deployment consequence is the same as on SER, and stronger: dynamic range is
smaller than every selective build (4.531 MB vs 4.891–5.210) *and* 19 points more
accurate. There is no k at which selective quantization is the right choice for
this model.

### A4 — calibration sensitivity, SER

18 configurations: {50, 200, 500} samples × {balanced, natural} × 3 seeds, each
evaluated on the full test set. FP32 baseline 60.83% (8-class).

| config | mean % | SD | min % | max % | seed spread |
|---|---|---|---|---|---|
| n50 balanced | 50.28 | 1.37 | 48.75 | 52.08 | 3.33 |
| n50 natural | 50.00 | 0.34 | 49.58 | 50.42 | 0.83 |
| n200 balanced | 51.81 | 1.99 | 50.00 | 54.58 | 4.58 |
| n200 natural | 51.81 | 2.89 | 48.33 | 55.42 | **7.08** |
| n500 balanced | 53.47 | 3.05 | 49.17 | 55.83 | 6.67 |
| n500 natural | 55.28 | 0.71 | 54.58 | 56.25 | 1.67 |

**Overall spread across all 18 runs: 7.92 points.**

Two things follow, and they point in different directions:

1. **More calibration data helps the mean** — 50.3% at n=50 rising to 53.5–55.3%
   at n=500. So calibration size is a real lever.
2. **But it does not buy stability.** The spread from the *random seed alone*,
   with every other variable fixed, reaches 7.08 points (n=200 natural) and
   6.67 points (n=500 balanced) — larger than at n=50. Standard deviation does
   not shrink monotonically with n.

This is the direct, quantitative answer to the paper's unexplained "4/8 to 7/8"
sentence. That anecdote describes a 3-of-8 swing on an 8-sample set; seed choice
alone moves full-test-set accuracy by up to 7 points here, which is more than
enough to produce 4/8 versus 7/8 on eight samples. **A substantial share of the
originally reported variation was calibration noise, not a property of the
quantization scheme.** The paper can now state mean ± SD across seeds with an
error-bar figure (`quant_calibration_ser.png`) instead of an anecdote — and
should report which seed policy it used, because the choice is worth several
points.

### A4 — calibration sensitivity, FER

Same 18 configurations, each evaluated on the frozen 1948-image test set. FP32
baseline 82.49% — identical to the corrected notebook's published figure.

| config | mean % | SD | min % | max % | seed spread |
|---|---|---|---|---|---|
| n50 balanced | 69.90 | 2.93 | 67.45 | 74.02 | 6.57 |
| n50 natural | 69.88 | 5.58 | 62.06 | 74.69 | 12.63 |
| n200 balanced | 64.85 | 6.00 | 57.29 | 71.97 | **14.68** |
| n200 natural | 67.93 | 4.52 | 61.60 | 71.87 | 10.27 |
| n500 balanced | 66.84 | 2.05 | 65.09 | 69.71 | 4.62 |
| n500 natural | 67.62 | 3.41 | 64.89 | 72.43 | 7.55 |

**Overall: mean 67.84%, SD 4.66, range 57.29–74.69 — a spread of 17.40 points.**

The FER picture is not the SER picture, and the difference matters:

1. **Calibration size is not a lever here.** Pooled means run 69.89 / 66.39 /
   67.23 for n = 50 / 200 / 500 — no trend, and every pairwise gap is inside one
   standard deviation. SER's clean "more data helps the mean" result does not
   generalise to this model.
2. **Class balance does nothing.** 67.20% balanced versus 68.48% natural pooled
   over nine runs each, against an SD of ~4.6. As close to a null as this study
   produces.
3. **Seed dominates everything else.** Spread from the seed alone, every other
   variable fixed, reaches 14.68 points at n=200 balanced. The n=500 balanced
   block is the tightest at 4.62, but the n=500 natural block is 7.55, so
   stability at n=500 is not established — with three seeds per block an SD is
   barely an estimate, and the apparent shrinkage is within what luck produces.

**Consequence for A1.** A1's committed full-INT8 per-channel figure of 61.24%
(−21.25 pts) used n=200 balanced, seed 42. Only 1 of these 18 draws falls below
it. It is a legitimate measurement, but it sits near the pessimistic tail of the
calibration distribution: the *expected* penalty over 18 draws is **−14.65 pts**
(range −7.80 to −25.20). The paper should not present −21.25 as "the" cost of
full INT8 on FER without saying it is one calibration draw. Reporting mean ± SD
over seeds, with `quant_calibration_fer.png` as the error-bar figure, is the
defensible form. This does not change A1's qualitative conclusion — full INT8
collapses on FER under every draw, the best of 18 still being 7.8 points below
FP32 — only the number attached to it.

Two notes on reading the table. The seed labels carry no meaning across blocks:
for a fixed seed the calibration subsets at different n are disjoint (measured:
0/48, 0/48, 3/48 overlap between n50-balanced and n200-balanced for seeds 0/1/2),
so a seed that looks consistently bad is coincidence, not a property of the seed.
And in the balanced condition the code takes `n // 4` per class, so the **"n=50"
rows use 48 images**, not 50; n=200 and n=500 divide evenly and are exact.

### A5 — quantization-aware training, SER

Fine-tuned from the corrected checkpoint for 15 epochs (lr 1e-4, batch 32,
validation carved from TRAIN actors only), in an isolated venv. The rebuilt
Keras-2 model reproduced the published baseline exactly before training —
8-class 0.6083, 4-class 0.7083 — so QAT started from the right weights.

| format | 8-class | 4-class | MB |
|---|---|---|---|
| fp32 | 0.6083 | 0.7083 | 1.701 |
| dynrange | 0.6042 | 0.7125 | 0.446 |
| int8_full_perchannel (PTQ) | 0.5083 | 0.5958 | 0.449 |
| **qat_int8** | **0.6042** | **0.7000** | **0.452** |

Paired McNemar on the deployed 4-class view, Holm-corrected:

| comparison | Δ acc | b / c | p_Holm | |
|---|---|---|---|---|
| qat_int8 vs int8_full_perchannel | **+10.42** | 32 / 7 | 0.00142 | significant |
| qat_int8 vs fp32 | −0.83 | 5 / 7 | 1.0 | tested non-difference |
| qat_int8 vs dynrange | −1.25 | 6 / 9 | 1.0 | tested non-difference |

The 8-class view agrees (+9.58 vs PTQ, p_Holm = 0.0095; +0.42 vs FP32, p_Holm =
1.0), so the conclusion does not depend on which view is taken.

**QAT recovers full INT8 on SER.** It is statistically indistinguishable from
FP32 at a quarter of FP32's size, and beats post-training full INT8 by a
significant margin. This is the first positive result in Group A.

**But read the deployment consequence carefully.** Dynamic range already matches
the baseline at 0.446 MB — smaller than the QAT build and 1.25 points better on
the 4-class view (untested difference). So QAT is *not* the recommendation for
ordinary CPU deployment; dynamic range remains that, and it costs nothing to
produce. QAT's value is specific: it is the only build here that is fully
integer *and* accurate, which matters when the target cannot execute float
activations at all — an NPU or microcontroller path where dynamic range is not
an option. State it that way, not as "QAT is better."

Three limitations, none of which touch the frozen test set. **The first has since
been measured and it changes the conclusion — see A5-control below.**

1. **The comparison carries a confound.** QAT received 15 epochs of fine-tuning
   that PTQ did not, so part of +10.42 could be extra training rather than
   quantization awareness. *(Originally argued away here on the grounds that QAT
   ends below the float baseline, so the extra epochs "did not make a better
   model". That inference was wrong: a model can become more quantizable without
   becoming more accurate, and that is exactly what happened. The control
   measured it.)*
2. **Early stopping used a leaky validation split.** The 10% validation set is
   drawn at random from the 2880 *augmented* training samples, so augmented
   copies of the same clip appear on both sides; validation accuracy saturates
   at 100% by epoch 2 and is not a meaningful number. It only chose when to
   stop — the reported accuracy comes from the frozen actor-21-24 test set — but
   it means the stopping point was steered by an optimistic signal.
3. **BatchNorm is quantized by an explicit config**, with gamma/beta left in
   float (see §4.3). The converted graph is INT8; the scale/offset parameters
   are not.

**On whether to rebuild FER in Keras**, which the script's automatic verdict
recommends: that verdict fires on a threshold and does not know the two models
differ. SER's PTQ failure is −11.25 points on a four-conv CNN; FER's is −10.68 on
the deployment path (−21.25 on reference kernels, §7.6) on EfficientNet-B0, and
A2/A3 showed that failure is distributed rather than concentrated in a few
layers. QAT rescuing the small model is encouraging but not evidence it rescues
the large one. **The two control sections below narrow the expected benefit
considerably** — read them before treating the rebuild as worthwhile at all.

### A5-control — how much of QAT's gain is just more training?

`quant_09_qat_control_ser.py` removes the confound the only way that settles it:
the same float model, fine-tuned for the same 15 epochs with the same
hyper-parameters, split and seed — then post-training quantized. The only
difference from A5 is that no quantization wrappers were applied before training.

| build | 4-class | 8-class | step | p (McNemar, uncorrected) |
|---|---|---|---|---|
| PTQ(baseline) | 59.58 | 50.83 | — | — |
| **PTQ(fine-tuned)** | **65.42** | 55.42 | **+5.83** | **0.013** |
| QAT | 70.00 | 60.42 | +4.58 | **0.061** |

Identical on the delegated path (65.42 either way; PTQ baseline 60.00 there), so
this does not depend on execution path.

**A5's headline decomposes, and the part attributable to QAT is not significant.**
Of the +10.42 points QAT held over post-training quantization, **+5.83 comes from
the extra 15 epochs alone** (p = 0.013) and **+4.58 from quantization awareness**
— which at 29 discordant samples does not reach significance (p = 0.061
uncorrected, and worse under any correction). Fine-tuning and then post-training
quantizing captures most of the measurable benefit.

**The inference I had used to argue the confound away was wrong**, and the way it
was wrong is worth recording. The fine-tuned float model is *less* accurate than
the baseline (69.17% vs 70.83% 4-class), which I had taken as evidence that extra
training could not explain QAT's gain. But its INT8 conversion is 5.83 points
better. Extra training made the model **more quantizable without making it more
accurate** — quantizability and accuracy are different properties, and reasoning
from one to the other does not work.

Two things this does *not* say. It does not show QAT is useless: +4.58 points is
a real point estimate and 240 samples has little power to resolve it — "not
demonstrated" is not "no effect". And it does not touch A5's other result, that
QAT INT8 is statistically indistinguishable from FP32 (p_Holm = 1.0).

**What it changes for FER was tested and did not hold.** The case for rebuilding
EfficientNet-B0 in Keras for QAT (1–2 weeks) rested on QAT's margin over PTQ, and
most of that margin came from the fine-tuning rather than the quantization
awareness — suggesting fine-tune-then-PTQ as a cheap substitute. On FER it is
**worse than plain PTQ** on the deployment path (−4.57 points, p = 9.2e−5): see
A5-control-FER. The SER result does not transfer, and the rebuild question stays
open.

### A5-control-FER — fine-tune-then-PTQ does *not* transfer to FER

A5-control's SER result suggested a cheap alternative to a 1–2 week Keras
rebuild: fine-tune the existing PyTorch model, then post-training quantize.
`quant_10` does exactly that — same recipe as `quant_09` (15 epochs, Adam 1e-4,
batch 32, balanced class weights, early stopping on val loss, restore best),
architecture and preprocessing untouched, calibration identical to `quant_01`.
`quant_11` decomposes the result.

**It does not work, and the reference-kernel number would have said it did.**

| path | PTQ(baseline) | PTQ(fine-tuned) | raw gain | p |
|---|---|---|---|---|
| reference kernels | 61.24 (penalty −21.20) | 66.32 (penalty −18.89) | **+5.08** | 2.1e−4 |
| **XNNPACK** | 71.82 (penalty −10.63) | 67.25 (penalty −17.97) | **−4.57** | 9.2e−5 |

The decomposition explains the sign flip. Fine-tuning does two things at once:

1. **It produces a better float model** — 85.22% against 82.44%, **+2.77 points,
   p = 5.7e−5** on 176 discordant samples. A real improvement, obtained with
   model selection on the val split and no test leakage.
2. **It produces a model that quantizes worse.** Its INT8 penalty is −17.97
   points on the deployment path against the baseline's −10.63 — **7.34 points
   worse**.

On reference kernels those two effects sum positive (+2.77 − (−2.31) = +5.08) and
fine-tuning looks like a win. On the delegated path the baseline's penalty is
much smaller, so the same +2.77 cannot cover a 7.34-point deterioration, and the
fine-tuned build ends up **significantly worse than plain PTQ of the original
checkpoint**. Note the fine-tuned model's penalty barely differs between paths
(−18.89 vs −17.97) while the baseline's halves (−21.20 vs −10.63): whatever the
delegate rescues in the original checkpoint, fine-tuning destroys.

**Conclusion: fine-tune-then-PTQ is not a substitute for QAT on FER**, and the
recommendation this report made after A5-control — "try it before committing to
the rebuild" — was tested and is withdrawn. Trying it was still the right call:
it cost one GPU-hour against 1–2 weeks.

Two further notes, neither about quantization:

- **One epoch was the whole of the fine-tuning.** Early stopping fired at epoch 5
  and restored epoch 1; val_loss rose monotonically (0.358 → 0.658) while train
  accuracy went 0.90 → 0.99. FER's val split is the corrected notebook's own
  cluster-disjoint split, so unlike the SER control this stopping signal is
  trustworthy. The SER control, by contrast, ran all 15 epochs. That asymmetry
  weakens any direct SER-vs-FER comparison of "the same recipe".
- **The +2.77-point float improvement is a finding about the checkpoint, not
  about quantization.** One epoch with balanced class weights beat the published
  corrected model on its own frozen test set. It does not invalidate the
  published 82.49% — that is what the corrected notebook produced — but it does
  indicate the checkpoint was not fully converged, and it is worth a sentence in
  the paper rather than silence.

### 7.5 Paired significance tests (McNemar) — and a correction

Every format is evaluated on the **same** held-out samples, so the comparison is
paired. The bootstrap CIs reported in §7 are *marginal* — one interval per format
— which discards the pairing and is badly underpowered: two formats can differ
while their marginal intervals overlap almost completely. McNemar's test uses only
the discordant samples (one format right, the other wrong), which is the correct
paired test for two classifiers on one test set.

> The table below is computed on **reference-kernel** predictions, matching §7's
> accuracies. The same tests on delegated predictions are in §7.6 — no conclusion
> differs, but the magnitudes do, and a p-value must be quoted from the same path
> as the accuracy beside it.

`mcnemar_compare.py`, Holm-corrected across the 10 within-model comparisons, with
paired bootstrap CIs on the difference:

Δ pts here is **A − B** in the "comparison" column — the tool's argument order,
the opposite of §7's format-minus-fp32. FER dynrange is the case where the two
tables look contradictory: −0.05 in §7 (dynrange is 0.05 below fp32) and +0.05
here (fp32 is 0.05 above dynrange). Same measurement, both correct, stated in
opposite directions.

| model / view | comparison | Δ pts (A − B) | b | c | n disc | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| FER | fp32 vs dynrange | +0.05 | 44 | 43 | 87 | 1.0 | 1.0 | no difference |
| FER | fp32 vs int8 per-ch | +21.25 | — | — | — | 3.2e−57 | 2.2e−56 | significant |
| FER | per-ch vs per-tensor | +28.03 | 847 | 301 | 1148 | 3.2e−58 | 2.3e−57 | significant |
| SER 8-class | per-ch vs per-tensor | −1.25 | 3 | 6 | 9 | 0.508 | 1.0 | not significant |
| SER 4-class | fp32 vs dynrange | −0.42 | 3 | 4 | 7 | 1.0 | 1.0 | no difference |
| SER 4-class | fp32 vs int8 per-ch | +11.25 | — | — | — | 1.9e−05 | 1.7e−04 | significant |
| SER 4-class | per-ch vs per-tensor | −2.92 | 1 | 8 | 9 | **0.039** | **0.156** | **not significant** |

**The correction.** §7 originally read the SER per-channel/per-tensor gap
(62.50% vs 59.58%) as the ordering "reversing" between models, and called it a
headline A6 finding. **That claim does not survive a paired test and has been
withdrawn.** The difference rests on **9 discordant samples out of 240**, split
1 vs 8. Uncorrected it is p = 0.039 — nominally significant, which is exactly why
it looked convincing — but Holm across 10 comparisons puts it at 0.156. With 9
discordant samples the test has almost no power to begin with.

The defensible A6 statement is: **per-channel is essential on FER (28 points,
p ≈ 1e−57) and makes no demonstrable difference on SER.** The practical
recommendation is unchanged — state which variant you used, because on FER the
choice is worth 28 points — but the "two models disagree" framing was an artifact
of reading marginal numbers.

Two further results worth taking into the paper:

- **Dynamic range is statistically indistinguishable from FP32** on both models
  (p = 1.0, both views). "Preserves baseline behaviour" is now a tested
  non-difference rather than an eyeballed one — a much stronger sentence, and it
  is the core deployment recommendation.
- **FER per-channel and per-tensor agree on only 29.7% of samples.** They are not
  near-variants; per-tensor quantization changes what the model does.

Method note: exact binomial below 25 discordant pairs, chi-square with continuity
correction above. The 4-class SER view is tested separately because that is the
deployed protocol the paper publishes, and it is where the withdrawn claim lived.

Two bugs were fixed to get here: `mcnemar_compare.py` wrote its markdown with
Windows' default cp1252 codec and crashed on `Δ` (now explicit UTF-8), and
`quant_02`'s prediction caches carry no labels file, so
`quant/quant_07_stage_mcnemar_input.py` reconstructs labels **in the order the
predictions were computed** and stages both the raw and deployed-4-class views.
Every staged accuracy was checked against the A1 table before testing.

---

### 7.6 Execution path: reference kernels vs XNNPACK

**How this was found.** The x86 latencies in §8 had a suspicious shape — dynamic
range 120× slower than FP32, and 28 threads slower than 8. That is the signature
of delegate fallback, not of expensive kernels. It was checked directly rather
than inferred: create an interpreter, ask whether XNNPACK announces itself, and
A/B the same model with `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES`.

**`ai-edge-litert` 2.1.4 applies no delegate at all** unless
`experimental_default_delegate_latest_features=True` is passed. Every A1–A6
measurement therefore ran on TFLite's reference kernels. Version matters: the
2.1.6 build in A5's venv *does* delegate by default, which is why the QAT run
printed `Created TensorFlow Lite XNNPACK delegate for CPU.` and nothing else did.

**Latency, 8 threads, x86** (not a deployment measurement — Group B owns that).
Delegated figures are from `quant_08` over the full test set (1948 / 240 real
images); the no-delegate column is a 5–20 invocation microbenchmark on random
input, because a full non-delegated pass costs 20 minutes per format. **Run-to-run
variance on the delegated column is ±2 ms** — repeat measurements of `fer_fp32`
gave 10.3, 11.7 and 12.2 ms — so the speedup ratios are order-of-magnitude
statements, not precise ones.

| model | no delegate | XNNPACK (full test set) | speedup |
|---|---|---|---|
| fer_fp32 | 150.2 ms | 10.3 ms | ~13× |
| fer_fp16 | 115.5 ms | 12.4 ms | ~10× |
| fer_dynrange | 337.6 ms | **216.3 ms** | **1.6×** |
| fer_int8_full_perchannel | 600.6 ms | 7.7 ms | ~75× |
| fer_int8_full_pertensor | not measured | 7.8 ms | — |
| ser_fp32 | 14.4 ms | 4.6 ms | ~3× |
| ser_fp16 | not measured | 4.4 ms | — |
| ser_dynrange | 337.2 ms | 2.2 ms | ~130× |
| ser_int8_full_perchannel | 426.8 ms | 2.5 ms | ~170× |
| ser_int8_full_pertensor | not measured | 2.3 ms | — |

"not measured" means the no-delegate microbenchmark was never run for that
build, not that it is fast or slow. The delegated column is complete: all ten
figures come from `quant_format_evaluation_xnnpack.json`
(`ms_per_image_x86_xnnpack`). The three gaps are all in the ad-hoc
reference-kernel column, and none of them carries a claim in this report.

**fp16 buys size, not speed, on both models.** FER 12.4 ms vs fp32's 10.3 and
SER 4.4 vs 4.6 are all inside the ±2 ms run-to-run band — the format halves the
file (16.0→8.1 MB, 1.70→0.86 MB) and leaves latency where it was. That is
expected: TFLite fp16 stores weights at half width and dequantizes to fp32 at
load, so activations and arithmetic stay fp32. **Per-tensor is not faster than
per-channel either** (FER 7.8 vs 7.7 ms, SER 2.3 vs 2.5 ms, both within noise),
which matters for A6: per-tensor's 40.61-point accuracy loss on FER buys 0.73 MB
of file size and no measurable time.

The 1.6× on FER dynamic range is the whole story in one number: **in
`ai-edge-litert` 2.1.4**, XNNPACK's dynamic-range (hybrid) kernel coverage does
not extend to depthwise convolutions, which is most of EfficientNet-B0, so those
ops stay on reference kernels. SER's four plain Conv2Ds get the full ~130×.

**This is a coverage gap in one runtime version, not a property of dynamic-range
quantization.** XNNPACK's hybrid op coverage has broadened over releases, and if a
later build adds depthwise support, FER dynamic range should drop toward the
~10 ms range and finding #5 dissolves. Anyone repeating this must state their
litert/XNNPACK version beside the number; ours is **ai-edge-litert 2.1.4** (and
2.1.6 in A5's venv, which differs in delegate defaults but was not benchmarked).

#### A1/A6 on both paths

Δ here is **XNNPACK − reference** for the same build (the delegate's effect), not
a comparison against FP32. Accuracy against FP32 is read off the column itself:
82.49 − 71.82 = −10.68 for full INT8 per-channel.

| FER | XNNPACK | reference | Δ (xnn − ref) | agree w/ fp32 | ms |
|---|---|---|---|---|---|
| fp32 | 82.49 | 82.49 | +0.00 | — | 10.3 |
| fp16 | 82.55 | 82.55 | +0.00 | 0.9995 | 12.4 |
| dynrange | 82.80 | 82.44 | +0.36 | 0.9559 | 216.3 |
| int8_full_perchannel | **71.82** | 61.24 | **+10.57** | 0.7505 | 7.7 |
| int8_full_pertensor | 31.21 | 33.21 | −2.00 | 0.3537 | 7.8 |

| SER (4-class) | XNNPACK | reference | Δ (xnn − ref) | agree w/ fp32 | MB | ms |
|---|---|---|---|---|---|---|
| fp32 | 70.83 | 70.83 | +0.00 | — | 1.70 | 4.6 |
| fp16 | 70.83 | 70.83 | +0.00 | 1.0000 | 0.86 | 4.4 |
| dynrange | **71.25** | **71.25** | +0.00 | 0.9542 | 0.45 | 2.2 |
| int8_full_perchannel | 60.00 | 59.58 | +0.42 | 0.7625 | 0.45 | 2.5 |
| int8_full_pertensor | 62.92 | 62.50 | +0.42 | 0.7500 | 0.43 | 2.3 |

Agreement is measured on the 8-class head, before summation. SER moves by at
most 0.83 points on any format on either path (that maximum is on the 8-class
view; on the 4-class view nothing moves more than 0.42), while running up to
130× faster. **The float formats are
bit-identical across paths on both models** — fp32 reproduces 82.49% exactly —
which is the control that proves the harness is sound and isolates the divergence
to quantized graphs.

**Do not read SER's per-tensor row as beating per-channel.** The +2.92 apparent
advantage rests on 9 discordant samples and does not survive correction
(p_Holm = 0.458 delegated; see §7.5 and the McNemar table below). On SER the two
INT8 variants are a tested non-difference, which is itself the finding: A6's
"per-channel is essential" holds on FER and has no demonstrated effect on SER.

Two consequences. **Note which comparison each number makes** — the two differ by
basis, not by measurement:

- **A1's magnitude halves.** Full INT8 on FER costs **−10.68 points against
  FP32** (71.82 vs 82.49), not −21.25. The same build is **+10.57 points against
  its own reference-kernel result** (71.82 vs 61.24). Both appear in this report;
  the first is the quantization penalty, the second is the delegate's effect.
  Still a collapse worth reporting — nobody ships 71.82% when 82.80% costs the
  same — but half the published penalty was the fallback implementation.
- **A6 strengthens.** The per-channel/per-tensor gap widens from 28.03 to
  **40.61 points**, because the delegate rescues per-channel (+10.57 vs its own
  reference result) and mildly penalises per-tensor (−2.00). "Per-channel is
  essential on FER" is a stronger claim on the deployment path than on reference
  kernels.

#### Paired tests on the delegated path

§7.5's McNemar results use reference-kernel predictions. Quoting a p-value from
one execution path beside an accuracy from another is not defensible, so the
tests were re-run on delegated predictions (`mcnemar_*_xnnpack.json`):

| model / view | comparison | Δ pts (A − B) | p (Holm) | verdict |
|---|---|---|---|---|
| FER | fp32 vs dynrange | −0.31 | 1.0 | no difference |
| FER | fp32 vs int8 per-ch | +10.68 | 1.3e−21 | significant |
| FER | per-ch vs per-tensor | **+40.61** | 2.7e−116 | significant |
| SER 4-class | fp32 vs dynrange | −0.42 | 1.0 | no difference |
| SER 4-class | fp32 vs int8 per-ch | +10.83 | 0.0007 | significant |
| SER 4-class | fp32 vs qat_int8 | +0.83 | 1.0 | no difference |
| SER 4-class | per-ch vs qat_int8 | −10.00 | 0.0033 | significant |
| SER 4-class | per-ch vs per-tensor | −2.92 | 0.458 | not significant |

**No conclusion changes.** Dynamic range remains a tested non-difference from
FP32 on both models; the INT8 collapse remains significant; QAT remains
indistinguishable from FP32 and significantly better than PTQ; and the withdrawn
SER per-channel/per-tensor "reversal" stays withdrawn (p_Holm = 0.458 delegated,
0.156 on reference kernels — non-significant either way). Cite whichever path
your accuracies come from, and do not mix.

#### A4 on both paths

| | mean | SD | min | max | spread |
|---|---|---|---|---|---|
| XNNPACK | 70.36 | 2.68 | 65.45 | 74.23 | **8.78** |
| reference | 67.84 | 4.66 | 57.29 | 74.69 | 17.40 |

**A4's finding halves but survives.** 8.8 points of calibration spread is still
far too much to report a single full-INT8 number for FER, so "cite mean ± SD over
seeds" stands — with the magnitude corrected.

The correction is **not a uniform offset**: per-config deltas run from −1.95 to
+15.40 (mean +2.52). Every draw that scored below 63% on reference kernels gains
3.9 to 15.4 points; draws above 72% mostly dip slightly. The reference path has a
failure mode that badly-calibrated graphs trigger and the delegated path does not.
A1's own build was the worst-hit config in the entire sweep (61.24 → 71.82), which
is why the effect looked so dramatic when first measured on that one model. The
effect concentrates at n=200, where block spread collapses from 14.68 to 3.70; no
explanation is offered for that, because none was measured.

#### A3 on both paths

| k | MB | XNNPACK | reference | Δ (xnn − ref) |
|---|---|---|---|---|
| 3 | 4.891 | 72.18 | 63.19 | +8.99 |
| 5 | 4.898 | **72.79** | 63.14 | +9.65 |
| 10 | 4.960 | 72.59 | 62.68 | +9.91 |
| 15 | 5.210 | 72.18 | 62.83 | +9.35 |

**A3's conclusion sharpens.** Against delegated full INT8 (71.82%), the best
selective build buys **+0.98 points** — down from +1.95 — and the curve is flat to
within 0.61 points across k = 3 to 15. Exempting five times as many layers changes
nothing measurable, which states "the failure is distributed" more cleanly than
the reference-kernel version did.

The k=5 delegated figure was recorded as `null` until 2026-08-05 because XNNPACK
failed to build its runtime for that graph. **That failure was diagnosed and is
not a property of the model** — see below. The row is now measured, and its
delegate delta (+9.65) sits inside the +8.99…+9.91 band the other three occupy,
which is the check that it is a real number and not an artefact of the retry.

#### The k=5 delegate-prepare failure: diagnosed

Earlier revisions of this report called this failure "reproducible" and drew a
conclusion from it — that selective quantization can yield a model a deployment
runtime refuses to run. **That conclusion was wrong and is withdrawn.** What the
original sweep saw was real, but it is a flaky runtime defect, not a property of
the k=5 graph's structure.

The error is `failed to create XNNPACK runtime` / `Node number 255
(TfLiteXNNPackDelegate) failed to prepare`. Five findings, each measured:

1. **Node 255 is not a model layer — it is the XNNPACK delegate node itself.**
   k=5's delegated execution plan has 256 nodes (indices 0–255) and the delegate
   node is always last. k=3's is node 253, k=10's is node 265, each likewise the
   final index. The number encodes graph size, nothing else. Any reading of
   "node 255" as pointing at a specific op is a misreading of the message.
2. **k=5 is structurally unremarkable.** All three builds delegate into exactly
   **one partition with 170 inputs and 1 output** — identical shape. On every
   axis that distinguishes the builds, k=5 sits *between* k=3 and k=10:
   float32 4-D activation islands 9 / **13** / 26, QUANTIZE–DEQUANTIZE pairs
   4 / **5** / 10, float MEAN ops 1 / **2** / 3. There is no monotone structural
   property that singles it out.
3. **Three conditions must coincide.** Each was isolated by running every trial
   in its own process (the outcome is decided once per process and sticky
   afterwards, so trials sharing a process are not independent):

   | condition (1 thread) | k=5 | k=3 / k=10 / k=15 |
   |---|---|---|
   | prior interpreter built and dropped in-process | **18/30 fail** | 0/30, 0/30, 0/30 |
   | only interpreter in a fresh process | 0/30 | 0/30, 0/30, 0/30 |

   So it is specific to k=5's graph *and* probabilistic — both at once. A fresh
   process never failed, at any k or thread count tested.
4. **It is thread-dependent, in the direction that rules out capacity.** Under
   the failing pattern, *fewer* threads means *more* failures — pooled over
   three independent sweeps of 30 trials each:

   | threads | 1 | 2 | 4 | 8 | 16 |
   |---|---|---|---|---|---|
   | failure rate | **59%** | 31% | 18% | 4% | 10% |

   A threadpool or per-thread workspace shortage would do the opposite. Combined
   with the requirement that a previous interpreter be constructed and
   destroyed first, the signature is stale allocator state reused by XNNPACK's
   runtime creation.

   This also explains why it stayed hidden: at the repo's default of 8 threads
   it is close to the least likely configuration to show, and a batch of 30
   trials there can read as 0/30. **The rate itself is unstable** — six
   independent batches of 30 in the 1-thread poisoned cell gave 13, 14, 15, 18,
   19 and 20 failures (99/180 pooled, 55%) — so treat all of these as rough
   probabilities, not constants.
5. **When it does prepare, the result is exact.** Two independent runs returned
   0.7279260780287474 to the last digit. A partially-initialised runtime would
   not do that, so the 72.79% above is trustworthy.

**What this changes.** A3's substantive finding — selective quantization does not
recover full INT8 — is untouched and slightly strengthened, since the curve is now
complete with no missing point. What is withdrawn is the *deployment-risk*
claim built on top of it. The honest version: a build can hit a probabilistic
runtime-creation failure in **ai-edge-litert 2.1.4**, which is a bug to report
upstream and to retry around, not a reason to distrust mixed-precision graphs.

**Why k=5 and not the others is still open.** Everything above establishes what
the failure is *not* — not structural, not capacity, not a property of selective
quantization — and that it needs k=5's specific graph. It does not explain which
allocation in that graph collides with the freed arena. Settling that needs an
ASAN build of XNNPACK, which is outside this study. Stated as an open question
in §10 rather than papered over.

**The methodological lesson is the one worth keeping.** A failure observed three
times in one process, in a sweep that builds models in sequence, was recorded as
"reproducible" — and a conclusion was drawn from a sample that never varied the
one thing that mattered. Re-running in a fresh process took under a minute and
reversed the finding. Before a crash becomes evidence, vary the process, not just
the run.

Reproduce with `python quant/quant_12_delegate_prepare_diag.py --trials 15`.

#### SER on the delegated path — measured, not argued

A3 and A4 were initially re-run on FER only, on the argument that SER's A1
accuracies move ≤0.83 points. That was an inference, so it was checked:

| | XNNPACK | reference |
|---|---|---|
| A4-SER mean / SD / spread | 52.25 / 2.55 / **7.92** | 52.11 / 2.71 / **7.92** |
| A3-SER k=3 → k=15 | 50.42 → 58.75 | 50.83 → 58.75 |

**SER is path-invariant on both items.** A4's calibration spread is 7.92 points on
*both* paths — the same number to two decimals — with per-config deltas from
−1.25 to +2.50 (mean +0.14). A3's selective builds move by at most 0.83 points
and k=15 is identical. Every SER conclusion in this report holds on either path,
which is now a measurement rather than an argument.

The contrast with FER is the point: the delegate is worth up to +15.40 points on
a single FER calibration draw and essentially nothing on SER. Whatever the
reference kernels do badly, they do it to EfficientNet-B0's depthwise/SE
structure, not to a four-layer plain CNN.

No SER selective build hit the delegate-prepare failure that FER's k=5 produces,
which was the first hint that the fault belonged to one specific graph rather
than to mixed-precision builds as a class. The diagnosis above confirms it and
goes further: it is not a property of that graph either, but a runtime defect
the graph happens to trigger.

#### A2 cannot be measured on the delegated path

Not a gap in effort: `tf.lite.experimental.QuantizationDebugger` takes no delegate
argument — it constructs its own interpreters internally — and delegation fuses
away the intermediates it exists to read. Measured on
`fer_int8_full_perchannel`: the reference graph exposes **484 tensors**, the
delegated graph **415**. A forced version would profile a different, smaller layer
set that could not be compared with the committed ranking.

So A2's per-layer profile is inherently a reference-kernel measurement. That is a
property of per-layer profiling on a fused execution path, and it should be stated
in the paper rather than left for a reader to wonder about. Its practical role is
unaffected: A2's ranking supplies A3's denylist, and A3 shows on both paths that
those layers are not the cause.

#### What this changes about the study

The reference-kernel results are not wrong; they are measurements of TFLite's
reference implementation. But the paper is about **deployment**, so the delegated
numbers are the ones its claims should rest on. Both are committed —
`quant_format_evaluation.json` and `quant_format_evaluation_xnnpack.json`, and
likewise for A3 and A4 — and no committed reference-kernel number was overwritten
in producing them.

SER is unaffected throughout: every SER conclusion, including A5 and all McNemar
tests, rests on predictions the delegate barely perturbs.

---

## 8. Runtime characteristics

> **Read this section knowing what §7.6 established.** Every number below was
> measured with no delegate applied, which is why the ratios look pathological —
> "quantization makes it 120× slower" and "28 threads is slower than 8" are both
> the fallback path, not kernel cost. Under XNNPACK the same FER INT8 build runs
> at 7.7 ms. The table is kept because it is what the study actually ran on, and
> because the anomaly in it is what led to finding the delegate problem.

Quantized EfficientNet-B0 falls back to slow reference kernels on x86:

| build | ms/image (1 thread) | ms/image (28 threads) |
|---|---|---|
| fp32 | 68 | 241 |
| dynrange | 8243 | 1170 |
| int8_full_perchannel | 8831 | 1275 |

One FER format over 1948 images therefore costs ~70 minutes. The A4 sweep
measured this more directly: 18 configs took 6 h 33 min wall clock at the default
thread count (`min(8, cpu_count)` = 8), i.e. **~20 min per config including the
TFLite conversion**, or ~600 ms/image. That is *faster* than the 28-thread row
above — oversubscribing 28 cores to these kernels hurts — so the ~70 min figure
is the pessimistic end of the range, and the 21 h estimate for A4-FER in the
runbook was ~3× too high. Re-running `onnx2tf`
with graph simplification made no difference (245 ops either way), so this is
intrinsic to the quantized depthwise/SE kernels, not graph bloat.

**None of this is a deployment result** — x86 latency is meaningless for an
edge claim and belongs to Group B on the Pi. **Group B must check the delegate
before it measures anything**: if the Pi's runtime behaves like litert 2.1.4 and
applies no delegate, it will reproduce this pathology, conclude dynamic range is
unusably slow, and recommend against the format the paper recommends. Confirm the
`Created TensorFlow Lite XNNPACK delegate for CPU.` line appears, and report which
runtime build and version produced every latency number.

Latency aside, the delegate only shapes how the study is run:
`quant_02` caches per-format predictions keyed by the `.tflite` mtime and size,
so formats can be computed in parallel processes and merged by a final pass.

---

## 9. State of each item

| Item | State |
|---|---|
| A1 | **complete**, both models, **both execution paths** |
| A2 | **complete**, both models — reference kernels only, and necessarily so (§7.6) |
| A3 | **complete**, both models; FER also on the delegated path |
| A4 | **complete**, both models; FER also on the delegated path |
| A5 | **complete** for SER (isolated venv); FER QAT is the open decision |
| A6 | **complete**, both models, both paths — included in A1 |

Nothing was reported as done that was not measured. **Every item A1–A6 is now
run**, on both models except A5, which exists only for SER: QAT on FER would mean
rebuilding EfficientNet-B0 in Keras and retraining (1–2 weeks). That is an open
decision rather than a queued task, and **§7's A5-control sections narrow the
expected benefit considerably** — three results have accumulated against the
optimistic reading:

- **A5-control**: on SER, only +4.58 of QAT's +10.42 point margin is attributable
  to quantization awareness, and that part does not reach significance
  (p = 0.061). The demonstrated effect is from the extra training.
- **A5-control-FER**: the cheap version of that extra training — fine-tune, then
  post-training quantize — is *worse* than plain PTQ on FER's deployment path
  (−4.57, p = 9.2e−5). SER's pattern does not transfer, and the one
  FER-specific datum available says training before quantization can hurt this
  model rather than help it.
- **§7.6**: FER's full-INT8 penalty on the deployment path is −10.68 points, not
  −21.25, so there is roughly half as much to recover as the figure that
  motivated the rebuild.

None of that shows QAT would fail on FER — it has not been tried, and the
architectures differ. It does mean the rebuild is now a **speculative** 1–2 weeks
against a smaller gap than advertised, with the only cheap probe of the idea
having come back negative.

Five follow-ons were listed here. **Three have since been done and are struck
through below**, kept rather than deleted because two of them changed a
conclusion. Genuinely open: items 2 and 3, plus one new question raised by item
1's result — whether a real QAT rebuild helps FER, now that the cheap substitute
is ruled out.

1. ~~The QAT confound~~ and ~~fine-tune-then-PTQ on FER~~ — **both done**, see
   §7 A5-control and A5-control-FER. On SER, +5.83 of QAT's +10.42 is the extra
   training and QAT's own +4.58 is not significant. On FER the same recipe is
   **worse than plain PTQ** on the deployment path (−4.57, p = 9.2e−5), because
   it yields a better float model that quantizes worse. What remains open is
   whether a genuine QAT rebuild helps FER — the cheap substitute does not.
2. **The A4 seed count** — three seeds per block is enough to show the spread is
   large but too few to estimate per-block SDs, which is why the "variance shrinks
   at n=500" reading was left unclaimed.
3. **Early stopping's leaky validation split** in `quant_06` — augmented copies of
   one clip land on both sides of it. It never touches the frozen test set, but a
   speaker-disjoint validation carve would make the stopping point honest.
4. ~~The delegated McNemar tests~~ — **done**, see §7.6. Every paired test was
   re-run on delegated predictions and committed as `mcnemar_*_xnnpack.json`. No
   conclusion differs: dynamic range remains a tested non-difference from FP32 on
   both models, the INT8 collapse remains significant, QAT remains
   indistinguishable from FP32, and the withdrawn SER per-channel/per-tensor
   reversal stays withdrawn. A6's gap is +40.61 points (p = 2.7e−116).
5. ~~SER on the delegated path for A3/A4~~ — **done**, see §7.6. SER is
   path-invariant on both: A4's spread is 7.92 points on either path, A3's builds
   move ≤0.83. The argument held, but it is now measured.
6. **Which allocation in the k=5 graph collides with the freed arena.** §7.6
   establishes that the delegate-prepare failure is a runtime defect rather than
   a structural property, and that it needs k=5's specific graph plus a
   destroyed prior interpreter plus a low thread count. It does not identify the
   offending allocation — that needs an ASAN build of XNNPACK, which is outside
   this study. Worth an upstream bug report against `ai-edge-litert` 2.1.4 with
   `fer_int8_selective_k5.tflite` and `quant_12` attached as the reproducer.

---

## 10. Reproducing

```bash
pip install -r requirements-quant.txt
# point at the datasets via paths.local.json or FER_DATA_ROOT / RAVDESS_ROOT
python quant/quant_00_prepare.py          # ~7 min, verifies everything
python quant/quant_01_convert.py          # ~9 min, builds 10 tflite models
python quant/quant_02_evaluate_formats.py # A1 + A6
python quant/quant_03_layerwise_debug.py --model fer --top 15   # A2
python quant/quant_05_selective_quant.py --model fer --ks 3 5 10 15  # A3
python quant/quant_04_calibration_sensitivity.py --model ser    # A4
python quant/quant_04_calibration_sensitivity.py --model fer    # A4, ~6.5 h
```

The FER sweep checkpoints each config to
`artifacts/results/quant_calibration_partial_fer.json`; re-running the same
command resumes from there, and `--restart` discards it.

To reproduce the delegated path (§7.6), set `QUANT_XNNPACK=1` for any of the
above, or run the dedicated A1/A6 script:

```bash
python quant/quant_08_xnnpack_formats.py --model both   # A1/A6, both models
QUANT_XNNPACK=1 python quant/quant_05_selective_quant.py --model fer --ks 3 5 10 15
QUANT_XNNPACK=1 python quant/quant_04_calibration_sensitivity.py --model fer --restart
```

The k=5 selective build hits a probabilistic XNNPACK runtime-creation failure
(§7.6). It is not a property of the model — retry, or run in a fresh process.
To reproduce the diagnosis:

```bash
python quant/quant_12_delegate_prepare_diag.py --trials 30   # ~12 min
```

`--restart` is not optional on the A4 command above: the resume cache holds
reference-kernel configs and must not be mixed with delegated ones. The scripts
write to the same filenames on both paths, so back up (or rename) the
reference-kernel result first — the committed copies in `outputs/quant/` are the
safety net either way.

See `GROUP_A.md` for the parallel-execution recipe and the A5 isolated-venv
command.

---

## 11. Commits

| Commit | Contents |
|---|---|
| `0626490` | Group A implementation (A1–A6), `quant_00_prepare.py`, runbook, requirements; ONNX and Keras prerequisite fixes |
| `452b6bc` | A5 weight-transfer fix; isolated-env requirement documented |
| `9e41b42` | README Group A section |
| `2efd9cc` | A2 squeeze-and-excitation test made structural, with hypergeometric enrichment |
| `9f42133` | Results for A1, A6, A2 (both models), A3 (SER) published to `outputs/quant/` |
| `a50be52` | This report; A4 (SER) calibration results |
| `9c94a2f` | McNemar paired tests; the SER per-channel/per-tensor claim withdrawn |
| `e99e742` | `quant_05` reuses `quant_02`'s prediction cache |
| `8721ede` | A3 (FER) selective quantization results |
| `77f229c` | A4's sweep made resumable — each config checkpointed, so a 6.5 h run survives interruption |
| `e733f32` | A4 (FER) results; the 17.4-point calibration spread that qualifies A1's headline number |
| `9c4727f` | A3 (FER) written up here and in the runbook |
| `adfd511` | A5 (QAT, SER) results; `tfmot` BatchNorm fix; QAT added to the paired tests |
| `0d0c530` | REPORT brought up to date with the completed study |
| `46acd09` | Four overclaims corrected ("to the digit", the Δ sign convention, "preserves behaviour", the leaky-split disclosure) |
| `fd067b4` | §7.6 — the XNNPACK execution-path finding; A1/A6, A3, A4 re-run delegated; `quant_08`; cache keys made path-aware |
| `7331c8b` | §7.6 — SER fp16 and both per-tensor latencies added to the latency table; SER A1/A6 given its own both-paths table |
| `9b3ca51` | §7.6 — the k=5 delegate-prepare failure diagnosed (`quant_12`); the "runtime refuses to run this graph" claim withdrawn and the k=5 row measured at 72.79% |
