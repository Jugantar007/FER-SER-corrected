# FER-SER-corrected

Corrected training pipelines for two emotion-recognition models, replacing earlier
notebooks (`Untitled43.ipynb`, `jugantarSER.ipynb`) whose reported accuracy came in part
from evaluation-protocol leakage rather than the model. Each notebook documents the
specific fault, the fix, and the resulting (lower, defensible) number.

## Contents

| Notebook | Task | Model | Dataset |
|---|---|---|---|
| `jugantarFER_corrected.ipynb` | Facial emotion recognition, 4 classes (Angry/Happy/Neutral/Sad) | EfficientNet-B0 fine-tune | Kaggle `sujaykapadnis/emotion-recognition-dataset` |
| `jugantarSER_corrected.ipynb` | Speech emotion recognition, 8-class and 4-class-by-summation | Custom CNN on mel-spectrograms | RAVDESS (`uwrfkaggler/ravdess-emotional-speech-audio`) |

`outputs/` holds the artifacts produced by an executed run of each notebook, so results
can be inspected or reproduced without re-running training.

## FER: what changed and why

| # | Original fault | Fix |
|---|---|---|
| F1 | Manuscript claimed FER2013 + ExpW + supplementary data; notebook actually trained on the Kaggle `emotion-recognition-dataset` only. | Dataset provenance recorded explicitly in the results JSON. |
| F2 | Checkpoint saved and reported at the same `best_val_acc` (85.29%, the max of a noisy plateau). | Checkpoint selected on **minimum validation loss**; a disjoint **test** split is reported instead. |
| F3 | Random `torch.randperm` split with no duplicate control; near-identical frames could straddle train/val/test. | Near-duplicates clustered by **perceptual hash** (Hamming distance ≤ 5); whole clusters assigned to one split. |
| F4 | Severe overfitting (train acc 99.48%, rising val loss). | Early stopping on val loss + class weights. |

**Result:** test accuracy **82.49%** (macro F1 0.819, 95% CI [80.80, 84.09]) vs. the
original 85.29% — a **+2.14 point** selection-optimism gap, quantified in
`outputs/FER/fer_corrected_results.json`.

## SER: what changed and why

| # | Original fault | Fix |
|---|---|---|
| S1 | Clean and noise-augmented spectrograms for the same clip were both generated before `train_test_split`, so ~80% of validation samples had a near-identical twin in training. | Split by **actor first**; augmentation applied to training actors only. |
| S2 | All 24 actors appeared in both train and validation. | Actor-disjoint partition (train 1–16, val 17–20, test 21–24), asserted at runtime. |
| S3 | RAVDESS's two takes per line could be separated by the random split. | Eliminated as a side effect of actor-disjoint splitting. |
| S4 | 4-class Neutral score used `max(p_neutral, p_calm)`, not matching the deployed system. | Summation over all eight classes, matching deployment. |
| S5 | `mask = y_true <= 4` discarded fearful/disgust/surprised (340 of 576 samples used). | All eight classes retained and folded into four. |

Evaluation reports three numbers side by side: 8-class, 4-class-by-summation (**the
deployed protocol — report this one**), and the legacy max()+filtered protocol kept only
to measure `protocol_inflation_points` against the original 94.41% figure.

**Result:** test accuracy (4-class, summation) **70.83%** (macro F1 0.695, 95% CI [65.00,
76.67]) vs. the original 94.41% — see `outputs/SER/ser_corrected_results.json` for the
full breakdown, including the legacy-protocol comparison (70.14%, i.e. actor-disjoint
splitting alone removed the inflation the old random split introduced).

## Reproducing a run

Both notebooks assume a Colab (or similar) environment with GPU access:

1. Upload a Kaggle API token as `kaggle.json` into the working directory.
2. Run all cells top to bottom. Each notebook downloads its own dataset, so no manual
   data prep is needed.
3. Outputs (`.pth`/`.keras`, ONNX export, split manifest, results JSON) are written to the
   working directory — copy them into `outputs/<task>/` to match this repo's layout.

## Quantization study (Group A)

`quant/` implements items A1–A6 of `README_GroupA.md` — the deployment-format
evaluation for the quantization paper. It reuses these notebooks' checkpoints and
**frozen** test splits; it does not retrain anything. See **`GROUP_A.md`** for the
runbook, the decisions taken while implementing, and the verification results.

| Item | What it produces | Script |
|---|---|---|
| A1, A6 | full test-set metrics for fp32 / fp16 / dynrange / INT8 per-channel / INT8 per-tensor, plus agreement with the FP32 control | `quant_01` → `quant_02` |
| A2 | per-layer quantization error, ranked | `quant_03` |
| A3 | selective (mixed-precision) quantization sweep | `quant_05` |
| A4 | calibration sensitivity (count × balance × seed) | `quant_04` |
| A5 | quantization-aware training, SER first | `quant_06` |
| — | A1/A6 re-evaluated with the XNNPACK delegate engaged | `quant_08` |

Latency, memory and thermal numbers are Group B and need the Raspberry Pi; nothing
here substitutes for them.

### Limitations to carry into the paper

These belong in the manuscript's limitations, not only in `REPORT.md`. This work
exists because an evaluation protocol was wrong; anything of that class found in
the correction has to be declared rather than left for a reviewer.

- **A5's early stopping used a validation split with augmentation leakage.** The
  10% validation set is drawn at random from the 2880 augmented training samples,
  so augmented copies of the same clip fall on both sides of it and validation
  accuracy saturates at 100% by the second epoch. It selected only the stopping
  epoch — every reported accuracy comes from the frozen actor-21-24 test set, and
  the train/val/test actor split itself is disjoint — but the stopping point was
  chosen on an optimistic signal. A speaker-disjoint carve would remove it.
- **"Dynamic range preserves the baseline" means accuracy, not behaviour.** There
  is no statistically detectable accuracy difference from FP32 (p = 1.0, paired
  McNemar), but the FER build disagrees with FP32 on 4.9% of test images. State
  the agreement rate alongside the accuracy.
- **Every quantized number depends on which kernels executed it.** TFLite applies
  no XNNPACK delegate unless asked (litert 2.1.4; 2.1.6 does by default), and the
  study's original measurements ran on reference kernels. On FER that is worth
  +10.6 points for full INT8. Both paths are committed — `*_xnnpack.json` beside
  the originals — and the delegated ones are what deployment claims should rest
  on. Any latency or accuracy figure must state which path produced it.
- **Full-INT8 accuracy on FER is a distribution, not a number.** It moves 8.8
  points across 18 calibration draws on the deployment path (17.4 on reference
  kernels), so cite mean ± SD over seeds rather than a single run
  (`quant_calibration_sensitivity_fer_xnnpack.json`).
- **The best format for FER depends on which axis you optimise — and on the
  runtime version.** Dynamic range is the most accurate (82.80%) and the
  *slowest* (216 ms); full INT8 is fastest (7.7 ms) at −10.68 points; fp16 matches
  FP32 accuracy at FP32 speed and half the file size, but dequantizes to fp32 in
  RAM. There is no single winner, and the paper should not imply one. The latency
  ordering is specific to **ai-edge-litert 2.1.4**, whose XNNPACK lacks hybrid
  depthwise coverage; quote the version with the number.
- **A better float model can quantize worse, and reference kernels hid it.** On
  FER, fine-tuning before post-training quantization produces a float model 2.77
  points *better* (p = 5.7e−5) whose INT8 build is 4.57 points *worse* on the
  deployment path (p = 9.2e−5) — while looking 5.08 points better on reference
  kernels. Float accuracy is not a proxy for quantized accuracy, and neither is
  the non-delegated path a proxy for deployment.
- **Most of QAT's advantage is the extra training, not quantization awareness.**
  The control (same model, same 15 epochs, plain fine-tuning then post-training
  quantization) reaches 65.42% against QAT's 70.00% and PTQ's 59.58%: +5.83 points
  from training (p = 0.013), +4.58 from QAT itself (p = 0.061, not significant at
  n = 240). Report QAT's margin over a *fine-tuned* PTQ baseline, not over an
  untrained one.

`REPORT.md` §9 lists what a follow-on study should measure to close each of these.

## Citing results

The `*_corrected_results.json` file in each `outputs/` subfolder is the source of truth
for the numbers to cite; it also records exactly why the originally reported figure
should not be used.

Quantization results land in `artifacts/results/` (gitignored until they are ready to
cite). `quant_format_evaluation.json` is the source of truth for A1/A6.
