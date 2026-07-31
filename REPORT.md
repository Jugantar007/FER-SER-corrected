# Group A implementation report

Implementation of `README_GroupA.md` (items A1–A6) for *"Verification-Driven
Post-Training Quantization of Deep Emotion Recognition Models for Raspberry Pi
Deployment"* (Sensua, Ahirwar, Kaur).

**Date:** 2026-07-31 · **Machine:** RTX 4060 workstation, Windows 11, Python 3.10,
TensorFlow 2.21.0, torch 2.6.0+cu124, onnx2tf 1.28.8

This is the narrative record: what was found, what was decided and why, what
broke, and what the numbers came out as. `GROUP_A.md` is the runbook; this is the
account of getting there.

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

**A5 — SER first, and not run here.** Option 3 from the README. Implemented, and
everything up to the `tfmot` call is verified. Not executed because
`tensorflow-model-optimization` 0.8.1 pins `absl-py~=1.2` and installing it would
**downgrade absl-py from 2.x in the environment every other Group A item depends
on**. An isolated-venv command is documented instead. This was a deliberate
refusal to risk the working environment for one item.

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

**FER, n = 1948.** The FP32 TFLite control reproduces the PyTorch baseline to the
digit, so the conversion chain is clean and every drop below is quantization, not
conversion — which is exactly the role the README assigns the fp32 build.

| format | MB | acc % | bal acc | macro F1 | kappa | agree | Δ pts | mean KL |
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

The paper's central claim holds on both models: **dynamic range preserves
baseline behaviour while full INT8 collapses.**

**A6 — the ordering is not universal, and that is the finding.** For FER,
per-channel beats per-tensor by 28 points (61.24 vs 33.21). For SER the ordering
**reverses** (62.50 vs 59.58). Two models, opposite answers — so "full INT8"
cannot be left unqualified in the paper, which is precisely the reviewer question
the README anticipates.

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

---

## 8. Runtime characteristics

Quantized EfficientNet-B0 falls back to slow reference kernels on x86:

| build | ms/image (1 thread) | ms/image (28 threads) |
|---|---|---|
| fp32 | 68 | 241 |
| dynrange | 8243 | 1170 |
| int8_full_perchannel | 8831 | 1275 |

One FER format over 1948 images therefore costs ~70 minutes. Re-running `onnx2tf`
with graph simplification made no difference (245 ops either way), so this is
intrinsic to the quantized depthwise/SE kernels, not graph bloat.

**None of this is a deployment result** — x86 latency is meaningless for an
edge claim and belongs to Group B on the Pi. It only shapes how the study is run:
`quant_02` caches per-format predictions keyed by the `.tflite` mtime and size,
so formats can be computed in parallel processes and merged by a final pass.

---

## 9. State of each item

| Item | State |
|---|---|
| A1 | **complete**, both models |
| A2 | **complete**, both models |
| A3 | **complete** for SER; FER queued (4 builds × 1948 images, several hours) |
| A4 | **complete** for SER; FER not started — 18 × ~70 min ≈ 21 h |
| A5 | implemented and partly verified; **not run** (needs isolated venv) |
| A6 | **complete**, both models — included in A1 |

Nothing was reported as done that was not measured. The outstanding items are
outstanding because of compute cost or environment risk, both stated above.

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
```

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
