"""
QUANT-04  Calibration sensitivity study (README item A4).

Explains the paper's currently unexplained "4/8 to 7/8" variation by sweeping
the representative dataset used for full INT8 quantization:

    sample count {50, 200, 500}  x  class balance {balanced, natural}  x  3 seeds

THE ONE CORRECTION TO THE ORIGINAL PLAN
    The third variable was originally specified as "FER2013-only vs. mixed
    FER2013+ExpW". It is DROPPED, and the seed sweep replaces it. The corrected
    FER model is trained on the Kaggle `sujaykapadnis/emotion-recognition-dataset`
    only -- FER2013 and ExpW are not part of this model's training data, so
    calibrating from them would sample a distribution the model never saw. That
    measures domain shift, not calibration sensitivity, and a reviewer would
    catch it.

    Variance across seeds at fixed n answers the actual question: how much of the
    4/8-to-7/8 spread was calibration noise rather than a property of the
    quantization scheme?

Calibration draws from the TRAINING split only. Never val, never test -- the
quantization study leaks too.

Output: artifacts/results/quant_calibration_sensitivity_<model>.json
        artifacts/figures/quant_calibration_<model>.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config as C
from common import (get_logger, save_json, agreement_metrics, run_tflite,
                    load_keras_compat, keras_saved_model_dir, four_class_summation,
                    labels_8_to_4)
from quant_01_convert import fer_representative_dataset, ser_representative_dataset
from quant_02_evaluate_formats import load_fer_test, load_ser_test, check_frozen

log = get_logger("quant_04_calib")

DROPPED_VARIABLE_NOTE = (
    "The originally planned third variable (FER2013-only vs mixed FER2013+ExpW "
    "calibration sources) was dropped: this model never saw FER2013 or ExpW, so "
    "that contrast would measure domain shift rather than calibration "
    "sensitivity. A 3-seed sweep replaces it.")


def build_converter(tag):
    if tag == "fer":
        return tf.lite.TFLiteConverter.from_saved_model(str(C.MODELS / "fer_saved_model"))
    sm = C.MODELS / "ser_saved_model"
    if not sm.exists():
        model, _ = load_keras_compat(C.MODELS / "ser_baseline.keras")
        sm = keras_saved_model_dir(model, "ser")
    return tf.lite.TFLiteConverter.from_saved_model(str(sm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fer", "ser"], required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=[50, 200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pertensor", action="store_true")
    ap.add_argument("--float-io", action="store_true")
    args = ap.parse_args()
    tag = args.model
    int8_io = not args.float_io

    X, y, class_names = (load_fer_test() if tag == "fer" else load_ser_test())
    check_frozen(tag, len(y))
    rep_fn = fer_representative_dataset if tag == "fer" else ser_representative_dataset

    base_path = C.MODELS / f"{tag}_fp32.tflite"
    if not base_path.exists():
        log.error("run quant_01_convert.py first")
        sys.exit(1)
    base_prob = run_tflite(base_path, X, "fp32 baseline")
    base_acc = float((base_prob.argmax(1) == y).mean())
    log.info("FP32 baseline accuracy: %.2f%%", base_acc * 100)

    y4 = labels_8_to_4(y) if tag == "ser" else None
    base_acc4 = (float((four_class_summation(base_prob).argmax(1) == y4).mean())
                 if tag == "ser" else None)

    records = []
    total = len(args.counts) * 2 * len(args.seeds)
    done = 0
    for n in args.counts:
        for balanced in (True, False):
            for seed in args.seeds:
                done += 1
                conv = build_converter(tag)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.representative_dataset = rep_fn(n=n, balanced=balanced, seed=seed)
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                if int8_io:
                    conv.inference_input_type = tf.int8
                    conv.inference_output_type = tf.int8
                if args.pertensor:
                    conv._experimental_disable_per_channel = True
                try:
                    blob = conv.convert()
                except Exception as e:
                    log.error("n=%d balanced=%s seed=%d failed: %s", n, balanced, seed, e)
                    continue

                P = run_tflite(blob, X, f"[{done}/{total}] n={n} bal={balanced} s={seed}")
                acc = float((P.argmax(1) == y).mean())
                ag = agreement_metrics(base_prob, P)
                rec = {"n_samples": n, "balanced": balanced, "seed": seed,
                       "accuracy": acc,
                       "accuracy_delta_points": round((acc - base_acc) * 100, 2), **ag}
                if tag == "ser":
                    rec["accuracy_4class"] = float(
                        (four_class_summation(P).argmax(1) == y4).mean())
                records.append(rec)
                log.info("n=%3d balanced=%-5s seed=%d | acc %.2f%% (%+.2f pts) | agree %.3f",
                         n, balanced, seed, acc * 100, (acc - base_acc) * 100,
                         ag["top1_agreement"])

    summary = {}
    for n in args.counts:
        for balanced in (True, False):
            sel = [r for r in records if r["n_samples"] == n and r["balanced"] == balanced]
            if not sel:
                continue
            a = np.array([r["accuracy"] for r in sel])
            g = np.array([r["top1_agreement"] for r in sel])
            summary[f"n{n}_{'balanced' if balanced else 'natural'}"] = {
                "accuracy_mean": float(a.mean()), "accuracy_std": float(a.std(ddof=0)),
                "accuracy_min": float(a.min()), "accuracy_max": float(a.max()),
                "accuracy_spread_points": float((a.max() - a.min()) * 100),
                "agreement_mean": float(g.mean()), "agreement_std": float(g.std(ddof=0)),
                "n_runs": len(sel)}

    all_acc = np.array([r["accuracy"] for r in records]) if records else np.array([])
    out = {"model": tag,
           "fp32_baseline_accuracy": base_acc,
           "fp32_baseline_accuracy_4class": base_acc4,
           "counts": args.counts, "seeds": args.seeds,
           "per_channel": not args.pertensor,
           "calibration_source": "TRAINING split only (never val/test)",
           "dropped_variable_note": DROPPED_VARIABLE_NOTE,
           "runs": records, "summary": summary,
           "overall_spread_points": (float((all_acc.max() - all_acc.min()) * 100)
                                     if all_acc.size else None),
           "interpretation": ("Spread across seeds at fixed n quantifies how much of the "
                              "originally reported 4/8-to-7/8 variation was calibration "
                              "noise rather than a property of the quantization scheme.")}
    save_json(out, C.RESULTS / f"quant_calibration_sensitivity_{tag}.json")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for balanced, marker, colour in ((True, "o", "#4d4d4d"), (False, "s", "#999999")):
        xs, ys, es = [], [], []
        for n in args.counts:
            k = f"n{n}_{'balanced' if balanced else 'natural'}"
            if k in summary:
                xs.append(n)
                ys.append(summary[k]["accuracy_mean"] * 100)
                es.append(summary[k]["accuracy_std"] * 100)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=marker, capsize=4, color=colour,
                        label="balanced" if balanced else "natural")
    ax.axhline(base_acc * 100, ls="--", color="black", lw=1, label="FP32 baseline")
    ax.set_xlabel("calibration samples")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"{tag.upper()} full-INT8 calibration sensitivity "
                 f"(mean $\\pm$ SD over {len(args.seeds)} seeds)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"quant_calibration_{tag}.png", dpi=300)
    plt.close(fig)
    log.info("saved -> %s", C.RESULTS / f"quant_calibration_sensitivity_{tag}.json")


if __name__ == "__main__":
    main()
