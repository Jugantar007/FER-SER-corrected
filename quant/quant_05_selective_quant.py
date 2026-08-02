"""
QUANT-05  Selective (mixed-precision) quantization (README item A3).

Takes the sensitive layers ranked by quant_03 and keeps the top-k in float while
quantizing everything else to INT8. Sweeps k so the accuracy/size trade-off is
visible.

THE CLAIM TO TEST
    If a selective build reaches FP32-level accuracy at a size near full INT8
    (~4.7 MB for FER), then full integer quantization is RECOVERABLE and the
    paper's conclusion changes from "avoid full INT8" to "full INT8 fails under
    naive PTQ, and here is the fix". That is an original result, not a replication.

    If it does not recover, that is still publishable: it strengthens the case
    for dynamic range as the default and bounds how much of the failure is
    attributable to a few layers. Report whichever happens.

Depends on: quant_03_layerwise_debug.py --model <model>

Output: artifacts/models/<model>_int8_selective_k<k>.tflite
        artifacts/results/quant_selective_<model>.json
        artifacts/figures/quant_selective_<model>.png
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
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    agreement_metrics, run_tflite, load_keras_compat,
                    keras_saved_model_dir, four_class_summation, labels_8_to_4,
                    DelegatePrepareError)
from quant_01_convert import fer_representative_dataset, ser_representative_dataset
from quant_02_evaluate_formats import (load_fer_test, load_ser_test, check_frozen,
                                       cached_predictions)

log = get_logger("quant_05_selective")


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
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10, 15])
    ap.add_argument("--calib-n", type=int, default=C.CALIB_SAMPLES)
    args = ap.parse_args()
    tag = args.model

    lw_path = C.RESULTS / f"quant_layerwise_{tag}.json"
    if not lw_path.exists():
        log.error("run quant_03_layerwise_debug.py --model %s first", tag)
        sys.exit(1)
    lw = load_json(lw_path)
    suspects = lw["top_suspects"]
    if lw.get("column_resolution", {}).get("source") == "HEURISTIC_FALLBACK":
        log.error("the ranked layer list came from a heuristic column and is not "
                  "trustworthy -- fix the column mapping in quant_03 first")
        sys.exit(1)
    log.info("%d ranked suspect layers available (metric %s)", len(suspects), lw["metric"])

    X, y, class_names = (load_fer_test() if tag == "fer" else load_ser_test())
    check_frozen(tag, len(y))
    rep_fn = fer_representative_dataset if tag == "fer" else ser_representative_dataset
    y4 = labels_8_to_4(y) if tag == "ser" else None

    # Reuse quant_02's per-format prediction cache: the FP32 and reference-format
    # predictions over this exact test set were already computed for A1, and on
    # FER each one costs ~70 minutes to recompute.
    base_prob = cached_predictions(tag, "fp32", X)
    base_acc = float((base_prob.argmax(1) == y).mean())
    log.info("FP32 baseline accuracy: %.2f%%", base_acc * 100)

    ref = {}
    for fmt in ("int8_full_perchannel", "int8_full_pertensor", "dynrange"):
        p = C.MODELS / f"{tag}_{fmt}.tflite"
        if p.exists():
            P = cached_predictions(tag, fmt, X)
            ref[fmt] = {"accuracy": float((P.argmax(1) == y).mean()),
                        "size_mb": round(p.stat().st_size / 1e6, 3),
                        **agreement_metrics(base_prob, P)}
    log.info("reference points: %s",
             {k: round(v["accuracy"] * 100, 2) for k, v in ref.items()})

    runs = []
    for k in args.ks:
        if k > len(suspects):
            log.warning("k=%d exceeds the %d ranked layers, skipping", k, len(suspects))
            continue
        denylist = suspects[:k]
        conv = build_converter(tag)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_fn(n=args.calib_n)
        try:
            opts = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_nodes=denylist)
            dbg = tf.lite.experimental.QuantizationDebugger(
                converter=conv, debug_dataset=rep_fn(n=args.calib_n),
                debug_options=opts)
            blob = dbg.get_nondebug_quantized_model()
        except Exception as e:
            log.error("k=%d selective build failed: %s", k, e)
            continue

        out = C.MODELS / f"{tag}_int8_selective_k{k}.tflite"
        out.write_bytes(blob)
        # Cached too, so an interrupted sweep resumes instead of restarting.
        try:
            P = cached_predictions(tag, f"int8_selective_k{k}", X)
        except DelegatePrepareError as e:
            # A mixed-precision graph the deployment runtime will not execute is
            # a result, not a crash: record it and keep sweeping.
            log.error("k=%d: XNNPACK failed to prepare this build -- %s", k, e)
            runs.append({"k_layers_kept_float": k, "denylisted": denylist,
                         "size_mb": round(out.stat().st_size / 1e6, 3),
                         "accuracy": None,
                         "delegate_prepare_failed": True,
                         "note": ("XNNPACK accepted the graph then failed to "
                                  "prepare its runtime; this build does not run "
                                  "on the delegated path")})
            continue
        pred = P.argmax(1)
        m = full_metrics(y, pred, class_names)
        m.update(bootstrap_ci(y, pred))
        rec = {"k_layers_kept_float": k, "denylisted": denylist,
               "size_mb": round(out.stat().st_size / 1e6, 3),
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "ci95_low": m["ci95_low"], "ci95_high": m["ci95_high"],
               "accuracy_delta_vs_fp32_points": round((m["accuracy"] - base_acc) * 100, 2),
               **agreement_metrics(base_prob, P)}
        if tag == "ser":
            rec["accuracy_4class"] = float((four_class_summation(P).argmax(1) == y4).mean())
        runs.append(rec)
        log.info("k=%2d | %.3f MB | acc %.2f%% (%+.2f pts) | agree %.3f",
                 k, rec["size_mb"], m["accuracy"] * 100,
                 rec["accuracy_delta_vs_fp32_points"], rec["top1_agreement"])

    # --- did selective quantization recover full INT8? --------------------
    verdict = "no selective build completed"
    # Builds the delegate refused to prepare carry accuracy=None and are excluded
    # from the ranking -- they are reported, but they have no accuracy to compare.
    scored = [r for r in runs if r.get("accuracy") is not None]
    if scored:
        full_int8 = ref.get("int8_full_perchannel", {}).get("accuracy")
        best = max(scored, key=lambda r: r["accuracy"])
        within_1pt = abs(best["accuracy"] - base_acc) * 100 <= 1.0
        if full_int8 is not None:
            gained = (best["accuracy"] - full_int8) * 100
            verdict = (
                f"best selective build (k={best['k_layers_kept_float']}) reaches "
                f"{best['accuracy']*100:.2f}% at {best['size_mb']:.3f} MB, "
                f"{gained:+.2f} points vs full INT8 ({full_int8*100:.2f}%) and "
                f"{(best['accuracy']-base_acc)*100:+.2f} points vs FP32. "
                + ("Full INT8 is RECOVERABLE -- this is the paper's original result."
                   if within_1pt else
                   "Full INT8 is NOT recovered to FP32 level; report this honestly and "
                   "use it to bound how much of the failure a few layers explain."))
        log.info("%s", verdict)

    save_json({"model": tag, "fp32_baseline_accuracy": base_acc,
               "ranked_layers_source": str(lw_path),
               "reference_formats": ref, "selective_runs": runs,
               "verdict": verdict,
               "claim_to_check": ("If a selective build reaches FP32-level accuracy at a "
                                  "size near full INT8, full integer quantization is "
                                  "recoverable and this is a novel result for the paper.")},
              C.RESULTS / f"quant_selective_{tag}.json")

    if scored:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot([r["size_mb"] for r in scored], [r["accuracy"] * 100 for r in scored],
                "o-", color="#4d4d4d", label="selective INT8")
        for r in scored:
            ax.annotate(f"k={r['k_layers_kept_float']}",
                        (r["size_mb"], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        for name, mk in (("int8_full_perchannel", "s"), ("dynrange", "^"),
                         ("int8_full_pertensor", "v")):
            if name in ref:
                ax.plot(ref[name]["size_mb"], ref[name]["accuracy"] * 100, mk,
                        ms=9, color="#999999", label=name)
        ax.axhline(base_acc * 100, ls="--", lw=1, color="black", label="FP32 baseline")
        ax.set_xlabel("model size (MB)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{tag.upper()} selective quantization trade-off", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(C.FIGURES / f"quant_selective_{tag}.png", dpi=300)
        plt.close(fig)
    log.info("saved -> %s", C.RESULTS / f"quant_selective_{tag}.json")


if __name__ == "__main__":
    main()
