# Group A — Laptop-Only Experiments (A1–A6)

Quantization experiments for *"Verification-Driven Post-Training Quantization of Deep
Emotion Recognition Models for Raspberry Pi Deployment"* (Sensua, Ahirwar, Kaur).

**Everything here runs on the RTX 4060 workstation. No Raspberry Pi required.**
TFLite models execute on x86 via `tf.lite.Interpreter`, so accuracy, macro F1 and
confusion matrices per format need no special hardware. Only *latency, memory and
thermal* numbers require the Pi, and those live in Group B.

---

## Prerequisite — do not start until this is done

Group A depends on the corrected baselines. Run both notebooks first:

- `jugantarSER_corrected.ipynb` → `jugantarSER_corrected.keras`, `ser_test_set.npz`
- `jugantarFER_corrected.ipynb` → `jugantarFER_corrected.pth`, `jugantarFER_corrected.onnx`, `fer_split.json`

**Every experiment below must use the same held-out test sets those notebooks
produced.** Measuring quantized formats on a different split than the baseline
reintroduces exactly the inconsistency the correction was meant to remove. The old
checkpoints (`jugantarSER.keras`, `jugantarFER.pth`) are not valid inputs here — they
were trained on leaked splits.

---

## Item-to-script mapping

Scripts already delivered in `paper-fix.zip` implement five of the six items:

| Item | Script | Status |
|---|---|---|
| A1 | `quant/quant_01_convert.py` → `quant/quant_02_evaluate_formats.py` | ready |
| A2 | `quant/quant_03_layerwise_debug.py` | ready |
| A3 | `quant/quant_05_selective_quant.py` | ready |
| A4 | `quant/quant_04_calibration_sensitivity.py` | ready, **one variable needs changing — see A4** |
| A5 | — | **not written yet** |
| A6 | `quant_01` + `quant_02` (both variants are already in `FORMATS`) | ready |

---

## A1. Full test-set evaluation of all model formats

**Effort:** 2–3 days, mostly compute · **Essential — this is the single biggest upgrade**

Replaces the 8-sample and 24-sample controlled verification scores as the paper's primary
evidence.

### What to produce

Evaluate every format on the **complete held-out test sets** — not the validation sets,
and not the controlled samples:

| Format | Role |
|---|---|
| `fp32` | control build; the only artifact that separates conversion faults from quantization faults |
| `fp16` | half-precision weights |
| `dynrange` | INT8 weights, float activations — the paper's selected format |
| `int8_full_perchannel` | INT8 weights + activations, per-channel weights (TFLite default) |
| `int8_full_pertensor` | INT8 weights + activations, per-tensor weights (this is also A6) |

Per format, report: accuracy, balanced accuracy, macro F1, weighted F1, Cohen's kappa,
per-class precision/recall/F1/support, confusion matrix, and bootstrap 95% CI.

Additionally report **agreement with the FP32 control** — top-1 agreement rate, mean and
max absolute probability difference, mean KL divergence. This generalises the paper's
"controlled verification" idea to the full test set, which is what makes the small-sample
scores demotable rather than deletable.

### Run

```bash
python quant/quant_01_convert.py
python quant/quant_02_evaluate_formats.py
```

For SER, `quant_02` also reports the deployed 4-class summation view alongside the raw
8-class numbers. Publish the summation view.

### Done when

`results/quant_format_evaluation.json` contains all five formats for both models, and
every format's `n_samples` matches the test-set size from the corrected notebooks.

### In the paper

Rewrite Sections 5.1 and 5.2 around these numbers. Keep the 8/8 and 24/24 scores in
Section 4.3 as a **cheap diagnostic** — they are still the thing that isolates *which
stage* of the conversion chain broke — but they are no longer the result.

---

## A2. Layer-wise quantization error profiling

**Effort:** 2–3 days · **High analytical value**

Turns Section 5.3 from informed speculation into demonstrated evidence.

### What to produce

Per-layer error metrics (RMSE / scale ratio) for the full INT8 build of the FER model,
ranked descending, plus a horizontal bar chart of the top ~15 layers.

### Run

```bash
python quant/quant_03_layerwise_debug.py --model fer --top 15
python quant/quant_03_layerwise_debug.py --model ser --top 15
```

### What you are looking for

Whether the squeeze-and-excitation blocks in EfficientNet-B0 rank highest. Values above
roughly 0.5 on RMSE/scale are commonly quantization-sensitive.

**If they do rank highest:** Section 5.3's architectural explanation becomes evidence.
**If they do not:** say so and report what actually dominates. A surprising result that
you report honestly is worth more than a tidy one you assumed. Do not tune the
methodology until the hypothesis wins.

### Known fragility

`QuantizationDebugger`'s CSV column names shift across TensorFlow versions. The script
handles the common variants and logs which columns it found — check that log before
trusting the ranking. If it fell back to a heuristic column, fix the column mapping
rather than proceeding.

### In the paper

New figure: per-layer quantization error. Rewrite Section 5.3 around it.

---

## A3. Selective (mixed-precision) quantization

**Effort:** 2–3 days · **Potentially the paper's original contribution**

Depends on A2's ranked layer list.

### What to produce

Models that keep the top-*k* sensitive layers in float while quantizing everything else
to INT8, swept over k ∈ {3, 5, 10, 15}. Report size and full-test-set accuracy against
both full INT8 and dynamic range.

### Run

```bash
python quant/quant_05_selective_quant.py --model fer --ks 3 5 10 15
```

### The claim to test

If a selective build reaches FP32-level accuracy at a size near full INT8 (~4.7 MB for
FER), then full integer quantization is **recoverable**, and the paper's conclusion
changes from "avoid full INT8" to "full INT8 fails under naive PTQ, and here is the fix."
That is an original result, not a replication.

If it does not recover, that is still publishable — it strengthens the case for dynamic
range as the default and bounds how much of the failure is attributable to a few layers.

### In the paper

New subsection under Section 5, plus a point on the trade-off figure.

---

## A4. Calibration sensitivity study

**Effort:** 2–3 days

Explains the currently unexplained "4/8 to 7/8" variation.

### Variables

| Variable | Levels |
|---|---|
| Sample count | 50 / 200 / 500 |
| Class balance | balanced vs. natural distribution |
| Random seed | 3 seeds per configuration |

### One correction to the original plan

The third variable was specified as **"FER2013-only vs. mixed FER2013+ExpW."**
**Drop it.** The corrected FER model is trained on the Kaggle
`sujaykapadnis/emotion-recognition-dataset` only — FER2013 and ExpW are not part of this
model's training data, so calibrating from them would be sampling from a distribution the
model never saw. That would measure domain shift, not calibration sensitivity, and a
reviewer would catch it.

The seed sweep replaces it and answers the actual question better: **how much of the
4/8-to-7/8 spread was calibration noise rather than a property of the quantization
scheme?** Variance across seeds at fixed *n* is the direct measurement.

### Run

```bash
python quant/quant_04_calibration_sensitivity.py --model fer
python quant/quant_04_calibration_sensitivity.py --model ser
```

Calibration draws from the **training split only**. Never calibrate from val or test —
the quantization study leaks too.

### In the paper

Replace the vague "4/8 to 7/8 with recalibration" sentence with mean ± SD across seeds
and an error-bar figure.

---

## A5. Quantization-aware training

**Effort:** 1–2 weeks · **Biggest single upgrade — and the only item with no script yet**

### Read this before estimating

The 1–2 week figure assumes a wrinkle that the original plan glossed over: **your FER
model is PyTorch, and TFLite QAT is a TensorFlow tool** (`tensorflow_model_optimization`).
There is no clean PyTorch → QAT → TFLite path. Three options:

1. **Rebuild FER in Keras** using `tf.keras.applications.EfficientNetB0`, retrain on the
   same corrected split, then apply `tfmot.quantization.keras.quantize_model`. Cleanest
   route to a TFLite INT8 artifact, but it is a full retrain — this is where the 1–2
   weeks goes.
2. **PyTorch-native QAT** via `torch.ao.quantization`, reporting PyTorch INT8 results
   rather than TFLite. Faster, but you are then comparing across two runtimes, which
   weakens the comparison.
3. **Start with SER instead.** The SER model is already Keras, so `tfmot` applies
   directly with no rebuild. **Do this first** — it is 2–3 days rather than 2 weeks, and
   it tells you whether QAT rescues full INT8 at all before you commit to rebuilding
   EfficientNet-B0.

Recommendation: option 3, then option 1 only if SER shows QAT is worth it.

### What to produce

QAT-INT8 accuracy and size against PTQ-INT8 and dynamic range, on the same test sets.

### In the paper

Upgrades the conclusion from *"full INT8 fails, avoid it"* to *"full INT8 fails under
PTQ, and here is the validated recovery path."* This is what a Q1 reviewer means by
"comparison against an alternative method."

### Status

No script yet. Ask and I will write the SER QAT script first, since that is the cheap
proof of concept.

---

## A6. Per-channel vs per-tensor quantization

**Effort:** 1–2 days · Already covered by A1

Per-channel weight quantization is the standard literature fix for this exact failure
mode, so a reviewer will ask. Both variants are already in `quant_01`'s `FORMATS` list
and both appear in `quant_02`'s output — no extra run needed.

Note that recent TFLite versions default to per-channel; `int8_full_pertensor` disables
it via `converter._experimental_disable_per_channel = True`. State explicitly in the
paper which one the reported "full INT8" results correspond to, because the default has
changed across TensorFlow versions and the ambiguity is a legitimate reviewer question.

### In the paper

One row in the main results table plus two sentences in Section 5.3.

---

## Execution order

```
prerequisite:  both corrected notebooks
      |
      v
A1  quant_01_convert.py -> quant_02_evaluate_formats.py     [A6 included free]
      |
      +--> A4  quant_04_calibration_sensitivity.py           (independent, can run overnight)
      |
      v
A2  quant_03_layerwise_debug.py
      |
      v
A3  quant_05_selective_quant.py                              (needs A2's ranked list)

A5  QAT — independent of everything above, start SER version in parallel
```

A1 → A2 → A3 is a hard chain. A4 and A5 are independent and can run alongside.

### Timeline

| Path | Items | Duration | Paper level |
|---|---|---|---|
| Minimum | A1 (+A6) | ~3 days | 8/10 with Group B |
| Strong | A1, A2, A3, A4, A6 | ~2 weeks | 9/10 with Group B |
| Complete | all six | ~4 weeks | Q1 submission |

---

## Rules that apply to every item

- **Test sets are frozen.** Every experiment uses the exact test splits from the
  corrected notebooks. If a script produces a different `n_samples`, stop and find out
  why before recording the result.
- **Calibration comes from training data only.** Never val, never test.
- **Do not change the model architectures.** This study is about deployment formats. A
  changed architecture invalidates comparison with everything already run.
- **Do not tune until the hypothesis wins.** If A2 does not implicate squeeze-and-excitation
  blocks, or A3 fails to recover full INT8, report that. The paper's credibility rests on
  the honest failure analysis — a result reverse-engineered to match Section 5.3 would
  destroy the thing that makes it good.
- **Every result file records its environment** (Python, TF/PyTorch versions, GPU,
  timestamp) via `common.save_json`. Journals ask for this; keep using the helper.
- **These scripts have not been run against your data.** They are syntax-checked and
  logically complete, but expect first-run debugging on paths and library versions.
  `onnx2tf` is the most fragile link — `quant_01` verifies the ONNX export against
  PyTorch and warns if the max logit difference exceeds 1e-3. Do not proceed past that
  warning.

---

## What Group A does not cover

Latency, memory, throughput and thermal behaviour. Those need the Raspberry Pi and no
laptop measurement substitutes for them — x86 timings are meaningless for an
edge-deployment claim and a reviewer will reject them. See Group B
(`pi/pi_01_benchmark.py`), roughly 2–3 days of device time, not on the critical path for
anything here.
