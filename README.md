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

Latency, memory and thermal numbers are Group B and need the Raspberry Pi; nothing
here substitutes for them.

## Citing results

The `*_corrected_results.json` file in each `outputs/` subfolder is the source of truth
for the numbers to cite; it also records exactly why the originally reported figure
should not be used.

Quantization results land in `artifacts/results/` (gitignored until they are ready to
cite). `quant_format_evaluation.json` is the source of truth for A1/A6.
