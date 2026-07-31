# Group A — implementation runbook

Implementation of `README_GroupA.md` (items A1–A6) for *"Verification-Driven
Post-Training Quantization of Deep Emotion Recognition Models for Raspberry Pi
Deployment"*.

Everything here runs on the workstation. No Raspberry Pi required — TFLite models
execute on x86 via the LiteRT/`tf.lite` interpreter, so accuracy, macro F1 and
confusion matrices per format need no special hardware. Latency, memory and
thermal numbers are Group B and are deliberately **not** produced here; x86
timings are meaningless for an edge-deployment claim.

---

## Layout

```
config.py                  all paths, class orderings, frozen test-set sizes
common.py                  metrics, agreement, TFLite runner, Keras compat loader
quant/
  quant_00_prepare.py      stage + verify the corrected-notebook artifacts
  quant_01_convert.py      build all five formats           (A1, A6)
  quant_02_evaluate_formats.py  full test-set evaluation    (A1, A6)
  quant_03_layerwise_debug.py   per-layer error profiling   (A2)
  quant_04_calibration_sensitivity.py  calibration sweep    (A4)
  quant_05_selective_quant.py   mixed precision             (A3)
  quant_06_qat_ser.py      quantization-aware training      (A5)
requirements-quant.txt
artifacts/                 gitignored: models/, results/, figures/, cache/, logs/
```

`quant_00_prepare.py` is not in the original item-to-script table. It exists
because the delivered scripts expected artifacts from their own training
pipeline, whereas Group A must reuse the **corrected notebooks'** checkpoints and
splits. It is the bridge, and it verifies the wiring rather than assuming it.

## Item-to-script mapping

| Item | Script | Status |
|---|---|---|
| A1 | `quant_01_convert.py` → `quant_02_evaluate_formats.py` | implemented, run |
| A2 | `quant_03_layerwise_debug.py` | implemented, run |
| A3 | `quant_05_selective_quant.py` | implemented, run |
| A4 | `quant_04_calibration_sensitivity.py` | implemented, variable corrected |
| A5 | `quant_06_qat_ser.py` | implemented (SER first, per the recommendation) |
| A6 | both `int8_full_*` variants in `quant_01`/`quant_02` | implemented, run |

---

## Prerequisites

Both corrected notebooks must have been run; their artifacts are committed under
`outputs/`. Nothing in Group A retrains anything.

| Artifact | Used for |
|---|---|
| `outputs/FER/jugantarFER_corrected.pth` | FER weights, ONNX re-export |
| `outputs/FER/jugantarFER_corrected.onnx` | FER conversion chain |
| `outputs/FER/fer_split.json` | the frozen FER test split (1948 images) |
| `outputs/SER/jugantarSER_corrected.keras` | SER weights |
| `outputs/SER/ser_test_set.npz` | the frozen SER test set (240 samples) |

The raw datasets are **not** committed. Point the scripts at them with either
environment variables or a gitignored `paths.local.json` in the repo root:

```json
{
  "fer_data_root": "D:\\FER TFlite\\archive (1)\\dataset",
  "ravdess_root": "D:\\SER Tflite\\ravdess-emotional-speech-audio"
}
```

`FER_DATA_ROOT` / `RAVDESS_ROOT` environment variables override the file.

```bash
pip install -r requirements-quant.txt
```

---

## Run order

```
quant_00_prepare.py                        stage + verify        (~7 min)
      |
      v
quant_01_convert.py                        A1 + A6 builds        (~9 min)
      |
      v
quant_02_evaluate_formats.py               A1 headline           (see note)
      |
      +--> quant_04_calibration_sensitivity.py --model {fer,ser}   A4 (overnight for FER)
      |
      v
quant_03_layerwise_debug.py --model fer --top 15                   A2
      |
      v
quant_05_selective_quant.py --model fer --ks 3 5 10 15             A3

quant_06_qat_ser.py                        A5, independent of the above
```

A1 → A2 → A3 is a hard chain. A4 and A5 are independent and can run alongside.

### Note on runtime

Quantized EfficientNet-B0 graphs fall back to slow reference kernels on x86:
FP32 runs at ~68 ms/image but dynamic-range and INT8 take ~2.2 s/image at 9
threads. A single FER format over the 1948-image test set therefore costs about
70 minutes. This says nothing about deployment — Pi latency is Group B — but it
does shape how you run things:

- `quant_02` caches per-format predictions under `artifacts/cache/preds_*.npy`,
  keyed by the `.tflite` file's mtime and size. Re-runs are then seconds.
- Formats can be computed in parallel processes and merged by a final full run:

```bash
for f in dynrange int8_full_perchannel int8_full_pertensor; do
  QUANT_NUM_THREADS=9 python quant/quant_02_evaluate_formats.py --models fer --formats $f &
done; wait
python quant/quant_02_evaluate_formats.py          # merges from cache, writes the JSON
```

`QUANT_NUM_THREADS` sets interpreter threads (default `min(8, cpu_count)`).

---

## Decisions made while implementing

**A4 — the corrected variable.** The originally planned third variable,
"FER2013-only vs. mixed FER2013+ExpW calibration", is dropped. The corrected FER
model is trained on the Kaggle `sujaykapadnis/emotion-recognition-dataset` only,
so calibrating from FER2013/ExpW would sample a distribution the model never saw
— that measures domain shift, not calibration sensitivity. A 3-seed sweep
replaces it and answers the actual question (how much of the 4/8→7/8 spread was
calibration noise). The substitution is recorded in the results JSON under
`dropped_variable_note` so the paper can cite the reasoning.

**A2 — column mapping, not a heuristic.** `QuantizationDebugger`'s CSV columns
move across TensorFlow versions. In TF 2.21 it emits `mean_squared_error` and
`scale` and *no* `rmse` column, so the delivered fallback would have silently
ranked layers on an arbitrary numeric column. `quant_03` now resolves the columns
explicitly, derives RMSE as `sqrt(mean_squared_error)`, records the resolution in
`column_resolution`, and **exits** rather than guessing. `quant_05` refuses to
consume a ranking produced by the heuristic path.

**A2 — honest hypothesis reporting.** The script tags squeeze-and-excitation
layers by name and reports `hypothesis_se_blocks_rank_highest` as true/false, or
`null` with `se_tagging_reliable: false` when no SE layer could be identified by
name at all. A null is not a refutation; it means the ranking cannot settle the
question and the top layers need checking against the architecture by hand.

**A5 — SER first.** Option 3 from the README. The SER model is already Keras, so
`tfmot` applies with no rebuild. `quant_06` fine-tunes from the corrected
checkpoint (architecture unchanged) and reports QAT-INT8 against PTQ-INT8 and
dynamic range on the same frozen test set. Its verdict field states whether QAT
recovers full INT8 well enough to justify the 1–2 week FER rebuild (option 1).
Because `tfmot` targets Keras 2, the script sets `TF_USE_LEGACY_KERAS=1` and
rebuilds the architecture layer-for-layer, transferring the checkpoint's weights
and **verifying they reproduce the published baseline before training starts**.

**Frozen test sets are enforced, not assumed.** `config.EXPECTED_N_TEST` holds
1948 (FER) and 240 (SER); `check_frozen()` aborts on any mismatch, per rule 1.

**Calibration is train-only, enforced.** `quant_00` asserts the SER calibration
pool contains no val/test actor, and the FER representative set is drawn from
`fer_split.json`'s train list. `quant_00` also proves the three FER splits are
pairwise disjoint by path.

---

## Two defects found in the prerequisites

**1. The committed FER ONNX was unusable.** `outputs/FER/jugantarFER_corrected.onnx`
was 605 KB and stored its weights in an external `jugantarFER_corrected.onnx.data`
sidecar that was never committed, so it could not be loaded on its own —
`onnx.load` failed on `features.0.0.weight`. The `.pth` checkpoint is complete, so
`quant_01` re-exports a self-contained ONNX (`export_params=True`) and verifies it
against PyTorch. **The repository artifact has been replaced** with that
self-contained export (16.0 MB, 3,991,664 parameters, loads standalone).

**2. The SER checkpoint needs a forward-compat shim.** `jugantarSER_corrected.keras`
was written by a newer Keras (Colab) than the workstation's 3.12.2, and embeds a
`quantization_config` key in its layer configs that older Keras rejects outright.
`common.load_keras_compat()` rewrites `config.json` inside the `.keras` zip to drop
the unknown key and loads the patched copy; weights are untouched. The results
JSON records `keras_config_patched: true` whenever this path was taken.

---

## Verification performed by `quant_00_prepare.py`

Recorded in `artifacts/results/quant_00_prepare.json`:

| Check | Result |
|---|---|
| All 13,008 FER split paths resolve on this machine | pass |
| FER splits pairwise disjoint | pass (0 shared paths) |
| FER test n = 1948 | matches the notebook |
| PIL preprocessing vs torchvision eval transform | max abs diff **0.0** |
| FER baseline reproduced from checkpoint | 82.44% vs 82.49% published (−0.05 pt) |
| SER test n = 240 | matches the notebook |
| SER calibration pool actors | 1–16 only, no val/test actor |
| SER training features rebuilt | 2880 samples, matches the notebook |
| SER baseline reproduced | 60.83% 8-class / 70.83% 4-class — exact match |

The preprocessing check matters more than it looks: `torchvision.transforms.Resize`
defaults to **bilinear** while PIL's `Image.resize` defaults to **bicubic**. Using
PIL's default would have evaluated every quantized format on subtly different
pixels than the notebook reported. `config.FER_RESIZE_FILTER` pins it.

---

## What Group A does not cover

Latency, memory, throughput and thermal behaviour. Those need the Raspberry Pi
(Group B, `pi/pi_01_benchmark.py`) and no laptop measurement substitutes for
them.
