"""
QUANT-08  A1/A6 re-evaluated with the XNNPACK delegate engaged.

WHY THIS EXISTS
    Every number in quant_02 was produced on a path where XNNPACK never applied:
    `ai-edge-litert` 2.1.4 creates no delegate unless asked, so the quantized
    graphs ran TFLite's reference kernels. That is not what a deployment runtime
    does, and on FER it is not a small difference -- full INT8 scores 61.24% on
    reference kernels and 71.82% under XNNPACK, disagreeing on 32.7% of test
    images. Roughly half of the "INT8 collapses by 21 points" headline is the
    fallback implementation rather than quantization.

    This script re-runs A1/A6 with the delegate on so both paths can be reported.
    It does NOT overwrite quant_02's results; it writes its own file, and the
    committed reference-kernel numbers stay exactly where they are.

    The delegate is enabled with `experimental_default_delegate_latest_features`,
    which is what litert 2.1.6 does by default. Verified: FP32 accuracy is
    identical under both paths (82.49%), so the delegate is numerically neutral
    where every op is supported; the quantized graphs are where it diverges.

Output: artifacts/results/quant_format_evaluation_xnnpack.json
        artifacts/cache/xnn_preds_<model>_<format>.npy
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ai_edge_litert.interpreter import Interpreter

import config as C
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    agreement_metrics, four_class_summation, labels_8_to_4,
                    tflite_size_mb)
from quant_02_evaluate_formats import load_fer_test, load_ser_test, check_frozen

log = get_logger("quant_08_xnnpack")


def run_delegated(path, X, desc="", num_threads=8):
    """run_tflite's logic, but with the XNNPACK delegate engaged."""
    interp = Interpreter(model_path=str(path), num_threads=num_threads,
                         experimental_default_delegate_latest_features=True)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    n = len(X)
    probs = np.empty((n, out["shape"][-1]), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        x = X[i:i + 1]
        if inp["dtype"] in (np.int8, np.uint8):
            scale, zp = inp["quantization"]
            lo, hi = (-128, 127) if inp["dtype"] == np.int8 else (0, 255)
            x = np.clip(np.round(x / scale + zp), lo, hi).astype(inp["dtype"]) \
                if scale else x.astype(inp["dtype"])
        else:
            x = x.astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] in (np.int8, np.uint8):
            scale, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        probs[i] = y[0].astype(np.float64)
    ms = (time.perf_counter() - t0) / n * 1000

    if not np.allclose(probs.sum(1), 1.0, atol=1e-2):
        e = np.exp(probs - probs.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
    return probs, ms


def evaluate(tag):
    X, y, class_names = (load_fer_test() if tag == "fer" else load_ser_test())
    check_frozen(tag, len(y))
    y4 = labels_8_to_4(y) if tag == "ser" else None

    committed = load_json(C.RESULTS / "quant_format_evaluation.json").get(tag, {})
    out = {}
    baseline_prob = None

    for fmt in C.FORMATS:
        path = C.MODELS / f"{tag}_{fmt}.tflite"
        if not path.exists():
            log.warning("missing %s", path.name)
            continue
        P, ms = run_delegated(path, X, f"{tag}/{fmt}")
        np.save(C.CACHE / f"xnn_preds_{tag}_{fmt}.npy", P)
        pred = P.argmax(1)

        m = full_metrics(y, pred, class_names, y_prob=P)
        m.update(bootstrap_ci(y, pred))
        m["size_mb"] = tflite_size_mb(path)
        m["ms_per_image_x86_xnnpack"] = round(ms, 2)
        if fmt == "fp32":
            baseline_prob = P
        else:
            m["agreement_with_fp32"] = agreement_metrics(baseline_prob, P)

        if tag == "ser":
            P4 = four_class_summation(P)
            m4 = full_metrics(y4, P4.argmax(1), C.CLASS4, y_prob=P4)
            m4.update(bootstrap_ci(y4, P4.argmax(1)))
            m["four_class_summation"] = m4

        prev = committed.get(fmt, {})
        ref_acc = prev.get("accuracy")
        m["reference_kernel_accuracy"] = ref_acc
        if ref_acc is not None:
            m["delegate_delta_points"] = round((m["accuracy"] - ref_acc) * 100, 2)
        out[fmt] = m

        log.info("%s/%-22s acc %.4f (reference %.4f, %+.2f pts)  %6.1f ms",
                 tag, fmt, m["accuracy"], ref_acc if ref_acc else float("nan"),
                 m.get("delegate_delta_points", float("nan")), ms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fer", "ser", "both"], default="both")
    args = ap.parse_args()

    result = {"note": ("A1/A6 with the XNNPACK delegate engaged. quant_02's "
                       "committed numbers are the same models on TFLite reference "
                       "kernels, which is what runs when no delegate applies."),
              "delegate": "XNNPACK via experimental_default_delegate_latest_features",
              "threads": 8}
    for tag in (["fer", "ser"] if args.model == "both" else [args.model]):
        result[tag] = evaluate(tag)
    save_json(result, C.RESULTS / "quant_format_evaluation_xnnpack.json")
    log.info("saved -> %s", C.RESULTS / "quant_format_evaluation_xnnpack.json")


if __name__ == "__main__":
    main()
