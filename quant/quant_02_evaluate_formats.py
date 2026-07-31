"""
QUANT-02  THE HEADLINE EXPERIMENT (README item A1, includes A6).

Evaluates every TFLite format on the FULL held-out TEST set and reports, per format:
  - accuracy, balanced accuracy, macro F1, weighted F1, Cohen's kappa,
    per-class precision/recall/F1/support, confusion matrix, bootstrap 95% CI
  - top-1 agreement with the FP32 control, mean/max absolute probability
    difference, mean KL divergence
  - accuracy delta vs the FP32 control

This replaces the paper's 8/8 and 24/24 controlled-verification scores as
primary evidence. Those small sets stay in the paper as a cheap diagnostic
(they isolate WHICH stage of the conversion chain broke), not as the result.

For SER the deployed 4-class summation view is reported alongside the raw
8-class numbers. Publish the summation view.

Output: artifacts/results/quant_format_evaluation.json
        artifacts/figures/quant_cm_<model>_<fmt>.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    agreement_metrics, save_confusion, print_report, run_tflite,
                    four_class_summation, labels_8_to_4)

log = get_logger("quant_02_eval")

MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)


def load_fer_test():
    from PIL import Image
    items = json.loads((C.CACHE / "fer_split.json").read_text())["files"]["test"]
    resample = getattr(Image, C.FER_RESIZE_FILTER)
    X = np.empty((len(items), 224, 224, 3), dtype=np.float32)
    y = np.empty(len(items), dtype=np.int64)
    step = max(1, len(items) // 20)
    for i, (p, lbl) in enumerate(items):
        a = np.asarray(Image.open(p).convert("RGB").resize((224, 224), resample),
                       dtype=np.float32) / 255.0
        X[i] = (a - MEAN) / STD
        y[i] = lbl
        if (i + 1) % step == 0:
            print(f"\r    load FER test: {i+1}/{len(items)}", end="", flush=True)
    print(f"\r    load FER test: {len(items)}/{len(items)}", flush=True)
    return X, y, C.FER_CLASSES


def load_ser_test():
    d = np.load(C.CACHE / "ser_test.npz")
    return d["X"].astype(np.float32) / 255.0, d["y"], C.EMOTION8


def check_frozen(tag, n):
    """GROUP_A rule 1: if a script produces a different n_samples, stop."""
    expected = C.EXPECTED_N_TEST[tag]
    if n != expected:
        log.error("%s test set is %d, expected %d -- STOP and find out why",
                  tag.upper(), n, expected)
        sys.exit(1)
    log.info("%s test set n=%d matches the corrected notebook", tag.upper(), n)


def cached_predictions(tag, fmt, X):
    """Quantized FER inference costs minutes per format on x86, so predictions are
    cached. This lets formats run in parallel processes and makes a re-run cheap.
    The cache key includes the model file's mtime+size, so a rebuilt .tflite
    invalidates it automatically."""
    path = C.MODELS / f"{tag}_{fmt}.tflite"
    st = path.stat()
    cache = C.CACHE / f"preds_{tag}_{fmt}_{int(st.st_mtime)}_{st.st_size}.npy"
    if cache.exists():
        P = np.load(cache)
        if len(P) == len(X):
            log.info("%s/%s: using cached predictions", tag, fmt)
            return P
    P = run_tflite(path, X, f"{tag}/{fmt}")
    np.save(cache, P)
    return P


def evaluate_model_family(tag, X, y, class_names, formats):
    results, baseline_prob, baseline_acc = {}, None, None

    for fmt in formats:
        path = C.MODELS / f"{tag}_{fmt}.tflite"
        if not path.exists():
            log.warning("%s missing, skipping", path.name)
            continue

        P = cached_predictions(tag, fmt, X)
        pred = P.argmax(1)
        m = full_metrics(y, pred, class_names, y_prob=P)
        m.update(bootstrap_ci(y, pred))
        m["size_bytes"] = path.stat().st_size
        m["size_mb"] = round(path.stat().st_size / 1e6, 3)

        if fmt == "fp32":
            baseline_prob, baseline_acc = P, m["accuracy"]
            m["agreement_with_fp32"] = {"top1_agreement": 1.0, "mean_abs_prob_diff": 0.0,
                                        "max_abs_prob_diff": 0.0, "mean_kl_divergence": 0.0}
            m["accuracy_delta_vs_fp32_points"] = 0.0
        else:
            m["agreement_with_fp32"] = (agreement_metrics(baseline_prob, P)
                                        if baseline_prob is not None else None)
            m["accuracy_delta_vs_fp32_points"] = (
                round((m["accuracy"] - baseline_acc) * 100, 2)
                if baseline_acc is not None else None)

        # SER: also report the deployed 4-class summation view.
        if tag == "ser":
            y4 = labels_8_to_4(y)
            P4 = four_class_summation(P)
            m4 = full_metrics(y4, P4.argmax(1), C.CLASS4, y_prob=P4)
            m4.update(bootstrap_ci(y4, P4.argmax(1)))
            m["four_class_summation"] = m4
            save_confusion(m4["confusion_matrix"], C.CLASS4,
                           f"SER 4-class {fmt}",
                           C.FIGURES / f"quant_cm_ser4_{fmt}.png")

        results[fmt] = m
        print_report(f"{tag.upper()} / {fmt}  ({m['size_mb']} MB)", m)
        if tag == "ser":
            print(f"  4-class (summation, REPORT THIS): "
                  f"{m['four_class_summation']['accuracy']*100:.2f}%  "
                  f"macro F1 {m['four_class_summation']['macro_f1']:.3f}")
        save_confusion(m["confusion_matrix"], class_names,
                       f"{tag.upper()} {fmt}", C.FIGURES / f"quant_cm_{tag}_{fmt}.png")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["fer", "ser"], default=["fer", "ser"])
    ap.add_argument("--formats", nargs="+", default=C.FORMATS)
    args = ap.parse_args()

    out = {"note": ("Full held-out TEST set evaluation. Replaces the 8/8 and 24/24 "
                    "controlled-verification scores as the paper's primary evidence."),
           "baselines_from_corrected_notebooks": {}}

    if C.FER_RESULTS_JSON.exists():
        out["baselines_from_corrected_notebooks"]["fer_pytorch_test_accuracy"] = \
            load_json(C.FER_RESULTS_JSON)["test_accuracy_REPORT_THIS"]
    if C.SER_RESULTS_JSON.exists():
        s = load_json(C.SER_RESULTS_JSON)
        out["baselines_from_corrected_notebooks"]["ser_keras_test_accuracy_8class"] = \
            s["test_8class_accuracy"]
        out["baselines_from_corrected_notebooks"]["ser_keras_test_accuracy_4class"] = \
            s["test_4class_accuracy_summation_REPORT_THIS"]

    if "fer" in args.models and (C.MODELS / "fer_fp32.tflite").exists():
        X, y, cn = load_fer_test()
        check_frozen("fer", len(y))
        out["fer"] = evaluate_model_family("fer", X, y, cn, args.formats)
        del X

    if "ser" in args.models and (C.MODELS / "ser_fp32.tflite").exists():
        X, y, cn = load_ser_test()
        check_frozen("ser", len(y))
        out["ser"] = evaluate_model_family("ser", X, y, cn, args.formats)

    save_json(out, C.RESULTS / "quant_format_evaluation.json")
    log.info("saved -> %s", C.RESULTS / "quant_format_evaluation.json")

    print("\n" + "=" * 86)
    print(f"{'model':6}{'format':24}{'size MB':>9}{'acc':>9}{'macroF1':>9}"
          f"{'agree':>8}{'d_acc':>8}{'KL':>10}")
    print("=" * 86)
    for tag in ("fer", "ser"):
        for fmt, m in out.get(tag, {}).items():
            ag = (m.get("agreement_with_fp32") or {})
            print(f"{tag:6}{fmt:24}{m['size_mb']:>9.3f}{m['accuracy']*100:>8.2f}%"
                  f"{m['macro_f1']:>9.3f}{ag.get('top1_agreement', float('nan')):>8.3f}"
                  f"{m['accuracy_delta_vs_fp32_points']:>8.2f}"
                  f"{ag.get('mean_kl_divergence', float('nan')):>10.4f}")
    if "ser" in out:
        print("\nSER 4-class summation (deployed protocol -- publish this):")
        for fmt, m in out["ser"].items():
            f4 = m["four_class_summation"]
            print(f"  {fmt:24}{f4['accuracy']*100:>8.2f}%  macro F1 {f4['macro_f1']:.3f}"
                  f"  CI95 [{f4['ci95_low']*100:.2f}, {f4['ci95_high']*100:.2f}]")


if __name__ == "__main__":
    main()
