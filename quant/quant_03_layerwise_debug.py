"""
QUANT-03  Layer-wise quantization error profiling (README item A2).

Turns the paper's squeeze-and-excitation hypothesis from speculation into
evidence. Uses tf.lite.experimental.QuantizationDebugger to report per-layer
RMSE / scale statistics for the fully quantized model, ranked descending.

Emits the ranked layer list consumed by quant_05_selective_quant.py.

READ BEFORE TRUSTING THE RANKING
    QuantizationDebugger's CSV column names shift across TensorFlow versions.
    This script resolves them explicitly and records which names it found in the
    output JSON. If it cannot resolve them it FAILS rather than silently ranking
    on an arbitrary numeric column -- a heuristic fallback produces a plausible
    looking ranking that means nothing, which is worse than no ranking at all.
    (Observed in TF 2.21: the debugger emits `mean_squared_error` and `scale`,
    and no `rmse` column at all, so rmse is derived as sqrt(mse).)

WHAT YOU ARE LOOKING FOR
    Whether the squeeze-and-excitation blocks in EfficientNet-B0 rank highest.
    Values above roughly 0.5 on RMSE/scale are commonly quantization-sensitive.
    If they do NOT rank highest, say so and report what actually dominates.
    Do not tune the methodology until the hypothesis wins.

Output: artifacts/results/quant_layerwise_<model>.csv/.json
        artifacts/figures/quant_layerwise_<model>.png
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config as C
from common import get_logger, save_json, load_keras_compat, keras_saved_model_dir
from quant_01_convert import fer_representative_dataset, ser_representative_dataset

log = get_logger("quant_03_layerwise")

# Column-name candidates, most-specific first. Extend here when a new
# TensorFlow version renames something rather than adding a silent fallback.
MSE_COLS = ["mean_squared_error", "mse", "mean_squared_err"]
RMSE_COLS = ["rmse", "root_mean_squared_error"]
SCALE_COLS = ["scale", "quant_scale", "scales", "tensor_scale"]
RATIO_COLS = ["rmse/scale", "rmse_per_scale"]
NAME_COLS = ["tensor_name", "name", "op_name", "tensor_idx"]

# EfficientNet squeeze-and-excitation blocks. onnx2tf keeps ONNX node names, so
# these are matched loosely; se_tagging_reliable records whether it found any.
SE_PATTERN = re.compile(r"(squeeze|excit|[/_.]se[/_.\d]|\bse\b)", re.IGNORECASE)


def build_converter(tag):
    if tag == "fer":
        sm = C.MODELS / "fer_saved_model"
        if not sm.exists():
            log.error("missing %s -- run quant_01_convert.py first", sm)
            sys.exit(1)
        return tf.lite.TFLiteConverter.from_saved_model(str(sm))
    model, _ = load_keras_compat(C.MODELS / "ser_baseline.keras")
    sm = C.MODELS / "ser_saved_model"
    if not sm.exists():
        sm = keras_saved_model_dir(model, "ser")
    return tf.lite.TFLiteConverter.from_saved_model(str(sm))


def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_error_metric(df, allow_heuristic=False):
    """Return (series, description dict). Fails loudly if columns are unknown."""
    resolved = {"columns_present": list(df.columns)}

    ratio_col = pick(df, RATIO_COLS)
    if ratio_col:
        resolved.update(source="direct_ratio_column", ratio_column=ratio_col)
        return df[ratio_col], resolved

    scale_col = pick(df, SCALE_COLS)
    rmse_col = pick(df, RMSE_COLS)
    mse_col = pick(df, MSE_COLS)

    if scale_col and (rmse_col or mse_col):
        if rmse_col:
            rmse = df[rmse_col].astype(float)
            resolved.update(rmse_column=rmse_col)
        else:
            rmse = np.sqrt(df[mse_col].astype(float))
            resolved.update(rmse_derived_from=mse_col)
        scale = df[scale_col].astype(float).replace(0, np.nan)
        resolved.update(source="rmse_over_scale", scale_column=scale_col)
        return rmse / scale, resolved

    msg = (f"cannot resolve the error columns from {list(df.columns)}; "
           "add the new names to MSE_COLS/SCALE_COLS in this script")
    if not allow_heuristic:
        log.error(msg)
        log.error("refusing to rank on an arbitrary column -- see the module docstring")
        sys.exit(1)
    num = df.select_dtypes(include=[np.number]).columns
    log.warning("%s; falling back to %s BECAUSE --allow-heuristic-columns", msg, num[-1])
    resolved.update(source="HEURISTIC_FALLBACK", column=str(num[-1]))
    return df[num[-1]].astype(float), resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fer", "ser"], required=True)
    ap.add_argument("--top", type=int, default=15, help="layers to denylist downstream")
    ap.add_argument("--calib-n", type=int, default=C.CALIB_SAMPLES)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="RMSE/scale above which a layer is called sensitive")
    ap.add_argument("--pertensor", action="store_true",
                    help="profile the per-tensor build instead of per-channel")
    ap.add_argument("--allow-heuristic-columns", action="store_true",
                    help="rank on an arbitrary numeric column if resolution fails "
                         "(produces a meaningless ranking; do not use for the paper)")
    args = ap.parse_args()
    tag = args.model

    rep_fn = fer_representative_dataset if tag == "fer" else ser_representative_dataset

    conv = build_converter(tag)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_fn(n=args.calib_n)
    if args.pertensor:
        conv._experimental_disable_per_channel = True

    log.info("running QuantizationDebugger on %s (this takes a few minutes)", tag)
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=conv, debug_dataset=rep_fn(n=args.calib_n))
    debugger.run()

    csv_path = C.RESULTS / f"quant_layerwise_{tag}.csv"
    with open(csv_path, "w") as f:
        debugger.layer_statistics_dump(f)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    log.info("debugger columns: %s", list(df.columns))

    metric, resolved = resolve_error_metric(df, args.allow_heuristic_columns)
    log.info("error metric resolved via: %s", resolved.get("source"))
    df["rmse_per_scale"] = metric

    name_col = pick(df, NAME_COLS) or df.columns[0]
    df = df.sort_values("rmse_per_scale", ascending=False)
    df.to_csv(csv_path, index=False)

    ranked = df.dropna(subset=["rmse_per_scale"])
    top = ranked.head(args.top)
    suspects = top[name_col].astype(str).tolist()
    top_vals = [float(v) for v in top["rmse_per_scale"].values]

    # --- honest reporting of the Section 5.3 hypothesis -------------------
    all_names = ranked[name_col].astype(str)
    se_all = [n for n in all_names if SE_PATTERN.search(n)]
    se_in_top = [n for n in suspects if SE_PATTERN.search(n)]
    se_reliable = len(se_all) > 0
    n_above = int((ranked["rmse_per_scale"] > args.threshold).sum())

    if not se_reliable:
        verdict = ("SE blocks could not be identified by name in this graph, so the "
                   "Section 5.3 hypothesis cannot be judged from the ranking alone. "
                   "Inspect the top layers manually against the architecture.")
    elif se_in_top:
        verdict = (f"{len(se_in_top)} of the top {len(suspects)} layers match the "
                   "squeeze-and-excitation pattern -- Section 5.3's architectural "
                   "explanation is supported by this ranking.")
    else:
        verdict = ("NO squeeze-and-excitation layer appears in the top "
                   f"{len(suspects)}. Section 5.3's hypothesis is NOT supported; "
                   "report what actually dominates instead of tuning until it wins.")
    log.info("%s", verdict)

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(top))))
    ax.barh(range(len(top)), top["rmse_per_scale"].values[::-1], color="#4d4d4d")
    ax.set_yticks(range(len(top)),
                  [s[-52:] for s in top[name_col].astype(str).values[::-1]], fontsize=7)
    ax.axvline(args.threshold, ls="--", lw=1, color="black")
    ax.set_xlabel("RMSE / scale (higher = more quantization-sensitive)")
    ax.set_title(f"{tag.upper()} layer-wise quantization error", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"quant_layerwise_{tag}.png", dpi=300)
    plt.close(fig)

    save_json({
        "model": tag,
        "n_layers": int(len(df)),
        "n_ranked": int(len(ranked)),
        "metric": "rmse_per_scale",
        "column_resolution": resolved,
        "name_column": name_col,
        "per_channel": not args.pertensor,
        "calibration_samples": args.calib_n,
        "threshold": args.threshold,
        "n_layers_above_threshold": n_above,
        "top_suspects": suspects,
        "top_values": top_vals,
        "se_pattern": SE_PATTERN.pattern,
        "se_tagging_reliable": se_reliable,
        "se_layers_total": len(se_all),
        "se_layers_in_top": se_in_top,
        "hypothesis_se_blocks_rank_highest": bool(se_in_top) if se_reliable else None,
        "verdict": verdict,
    }, C.RESULTS / f"quant_layerwise_{tag}.json")

    log.info("%d/%d layers exceed RMSE/scale %.2f", n_above, len(ranked), args.threshold)
    log.info("top %d suspect layers:", args.top)
    for s, v in zip(suspects, top_vals):
        log.info("   %-58s %.4f", s[-58:], v)


if __name__ == "__main__":
    main()
