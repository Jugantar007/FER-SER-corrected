#!/usr/bin/env python3
"""
McNemar paired significance tests for quantization format comparison.

WHY THIS EXISTS
---------------
Every TFLite format is evaluated on the SAME held-out test samples. Comparing
them with independent confidence intervals throws away that pairing and is badly
underpowered - two formats can differ significantly while their marginal 95% CIs
overlap heavily. McNemar's test uses only the DISCORDANT samples (where one
format is right and the other wrong), which is the correct paired test for two
classifiers on one test set.

This matters concretely for A6: on SER the per-channel vs per-tensor difference
(59.58% vs 62.50%, n=240) has hugely overlapping marginal CIs. Only a paired test
can say whether the "reversal" is real or noise.

WHAT IT REPORTS
---------------
Per format pair:
  b, c            discordant counts (b = A right & B wrong, c = A wrong & B right)
  n_discordant    b + c   (the effective sample size of the test)
  statistic, p    exact binomial when b+c < 25, else chi-square with continuity
                  correction
  p_holm          Holm-Bonferroni adjusted across all pairs within a model
  acc_diff        (b - c) / n, the paired accuracy difference
  ci95            PAIRED bootstrap CI on that difference (not marginal CIs)
  agreement       top-1 agreement rate between the two formats

USAGE
-----
  python mcnemar_compare.py --preds-dir outputs/quant/cache --model both
  python mcnemar_compare.py --preds-dir outputs/quant/cache --model fer --alpha 0.01

EXPECTED INPUT
--------------
Cached per-format predictions from quant_02. The loader accepts, per model:
  <model>_<format>_probs.npy    (n, n_classes) float   -> argmax taken
  <model>_<format>_preds.npy    (n,) int
  <model>_labels.npy            (n,) int
  ... or a single <model>_predictions.npz containing arrays named by format
      plus 'labels' / 'y' / 'y_true'.

If nothing is found the script prints exactly which filenames it looked for and
exits, rather than guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

FORMATS = ["fp32", "fp16", "dynrange", "int8_full_perchannel", "int8_full_pertensor",
           "qat_int8", "ptq_finetuned_int8"]

# Comparisons the paper leans on. Flagged in the console summary.
KEY_PAIRS = [
    ("int8_full_perchannel", "int8_full_pertensor"),   # A6 - the reversal claim
    ("fp32", "dynrange"),                              # "preserves baseline"
    ("fp32", "int8_full_perchannel"),                  # "full INT8 collapses"
    ("dynrange", "int8_full_perchannel"),              # the deployment decision
    ("qat_int8", "int8_full_perchannel"),              # A5 - does QAT rescue INT8
    ("fp32", "qat_int8"),                              # A5 - does QAT match float
    ("ptq_finetuned_int8", "int8_full_perchannel"),    # A5-control - the fine-tuning step
    ("qat_int8", "ptq_finetuned_int8"),                # A5-control - QAT's own margin
]

EXACT_THRESHOLD = 25          # b+c below this -> exact binomial


# --------------------------------------------------------------------- loading
def _as_preds(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.argmax(1).astype(np.int64)
    if arr.ndim == 1:
        return arr.astype(np.int64)
    raise ValueError(f"unexpected prediction array shape {arr.shape}")


def load_model_predictions(preds_dir: Path, model: str):
    """Returns (labels, {format: preds}). Raises with a useful message if absent."""
    tried = []
    preds: dict[str, np.ndarray] = {}
    labels = None

    # ---- option 1: a single npz per model
    for name in (f"{model}_predictions.npz", f"{model}_probs.npz", f"{model}.npz"):
        p = preds_dir / name
        tried.append(str(p))
        if p.exists():
            d = np.load(p, allow_pickle=False)
            for key in ("labels", "y", "y_true", "y_test"):
                if key in d:
                    labels = np.asarray(d[key]).astype(np.int64)
                    break
            for fmt in FORMATS:
                for key in (fmt, f"{model}_{fmt}", f"{fmt}_probs", f"{fmt}_preds"):
                    if key in d:
                        preds[fmt] = _as_preds(d[key])
                        break
            if preds and labels is not None:
                return labels, preds

    # ---- option 2: one file per format
    for key in ("labels", "test_labels", "y_test", "y_true"):
        p = preds_dir / f"{model}_{key}.npy"
        tried.append(str(p))
        if p.exists():
            labels = np.load(p).astype(np.int64)
            break

    for fmt in FORMATS:
        for suffix in ("probs", "preds", "predictions", "logits"):
            p = preds_dir / f"{model}_{fmt}_{suffix}.npy"
            tried.append(str(p))
            if p.exists():
                preds[fmt] = _as_preds(np.load(p))
                break

    # ---- option 3: loose glob, last resort
    if not preds:
        for fmt in FORMATS:
            hits = sorted(preds_dir.glob(f"*{model}*{fmt}*.np[yz]"))
            if hits:
                arr = np.load(hits[0])
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr[list(arr.keys())[0]]
                preds[fmt] = _as_preds(arr)

    if not preds or labels is None:
        print(f"\nERROR: could not load cached predictions for '{model}'.", file=sys.stderr)
        print(f"Searched in: {preds_dir.resolve()}", file=sys.stderr)
        print("Looked for (among others):", file=sys.stderr)
        for t in tried[:14]:
            print(f"  {t}", file=sys.stderr)
        print("\nEither point --preds-dir at the quant_02 cache, or rename the arrays "
              "to <model>_<format>_probs.npy plus <model>_labels.npy.", file=sys.stderr)
        raise SystemExit(2)

    n = len(labels)
    for fmt, p in preds.items():
        if len(p) != n:
            raise SystemExit(
                f"ERROR: {model}/{fmt} has {len(p)} predictions but there are {n} labels. "
                "The frozen-test-set rule is violated - stop and find out why.")
    return labels, preds


# ------------------------------------------------------------------- statistics
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Two-sided McNemar. Exact binomial for small discordance, else chi-square
    with Edwards' continuity correction."""
    b = int(np.sum(correct_a & ~correct_b))     # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))     # A wrong, B right
    n_disc = b + c

    if n_disc == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "test": "none",
                "statistic": None, "p_value": 1.0,
                "note": "formats produce identical predictions on every sample"}

    if n_disc < EXACT_THRESHOLD:
        p = float(binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
        return {"b": b, "c": c, "n_discordant": n_disc, "test": "exact binomial",
                "statistic": float(min(b, c)), "p_value": p}

    stat = (abs(b - c) - 1.0) ** 2 / n_disc     # continuity-corrected
    p = float(chi2.sf(stat, df=1))
    return {"b": b, "c": c, "n_discordant": n_disc,
            "test": "chi-square (continuity corrected)",
            "statistic": float(stat), "p_value": p}


def paired_bootstrap_diff(correct_a, correct_b, n_boot=10000, seed=42):
    """PAIRED CI on the accuracy difference. Resamples SAMPLES, not formats, so
    the correlation between the two classifiers is preserved."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo * 100), float(hi * 100)


def holm_bonferroni(pvals: list[float], alpha: float):
    """Returns adjusted p-values and reject flags, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj.tolist(), [bool(a <= alpha) for a in adj]


# ----------------------------------------------------------------------- report
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def analyse(model: str, labels, preds: dict, alpha: float, n_boot: int):
    correct = {f: (p == labels) for f, p in preds.items()}
    acc = {f: float(c.mean()) for f, c in correct.items()}

    available = [f for f in FORMATS if f in preds]
    rows = []
    for a, b_fmt in combinations(available, 2):
        ca, cb = correct[a], correct[b_fmt]
        r = mcnemar(ca, cb)
        lo, hi = paired_bootstrap_diff(ca, cb, n_boot=n_boot)
        rows.append({
            "format_a": a, "format_b": b_fmt,
            "accuracy_a": acc[a], "accuracy_b": acc[b_fmt],
            "acc_diff_points": round((acc[a] - acc[b_fmt]) * 100, 3),
            "paired_ci95_points": [round(lo, 3), round(hi, 3)],
            "agreement": float((preds[a] == preds[b_fmt]).mean()),
            **r,
        })

    adj, rej = holm_bonferroni([r["p_value"] for r in rows], alpha)
    for r, a_, j in zip(rows, adj, rej):
        r["p_holm"] = float(a_)
        r["significant_holm"] = j
        r["is_key_pair"] = (r["format_a"], r["format_b"]) in KEY_PAIRS or \
                           (r["format_b"], r["format_a"]) in KEY_PAIRS

    return {
        "model": model,
        "n_test_samples": int(len(labels)),
        "alpha": alpha,
        "multiple_comparison_correction": "Holm-Bonferroni within model",
        "n_comparisons": len(rows),
        "accuracy_by_format": {f: round(v * 100, 3) for f, v in acc.items()},
        "comparisons": rows,
        "method_note": (
            "McNemar's test on paired predictions over the same held-out test set. "
            "Exact binomial used when the discordant count is below "
            f"{EXACT_THRESHOLD}, otherwise chi-square with continuity correction. "
            "Confidence intervals are paired bootstrap intervals on the accuracy "
            "DIFFERENCE, not marginal intervals on each format."),
    }


def to_markdown(res: dict) -> str:
    lines = [
        f"### {res['model'].upper()} — McNemar paired comparisons "
        f"(n = {res['n_test_samples']})", "",
        "| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["comparisons"]:
        ci = f"[{r['paired_ci95_points'][0]:+.2f}, {r['paired_ci95_points'][1]:+.2f}]"
        mark = " **←**" if r["is_key_pair"] else ""
        lines.append(
            f"| {r['format_a']} | {r['format_b']} | {r['accuracy_a']*100:.2f} | "
            f"{r['accuracy_b']*100:.2f} | {r['acc_diff_points']:+.2f} | {ci} | "
            f"{r['b']} | {r['c']} | {r['p_value']:.2e} | {r['p_holm']:.2e} | "
            f"{stars(r['p_holm'])}{mark} |")
    lines += ["", "`b` = A correct & B wrong; `c` = A wrong & B correct. "
                  "Only discordant samples carry information.",
              "Significance stars use Holm-adjusted p-values. "
              "**←** marks comparisons the paper's claims rest on.", ""]
    return "\n".join(lines)


def to_latex(res: dict) -> str:
    head = (r"\begin{table}[htbp]\centering" "\n"
            r"\caption{" + res["model"].upper() + r" paired format comparisons "
            r"(McNemar, Holm-corrected, $n=" + str(res["n_test_samples"]) + r"$).}" "\n"
            r"\begin{tabular}{llrrrrl}" "\n" r"\hline" "\n"
            r"Format A & Format B & Acc A & Acc B & $\Delta$ & $p_{\mathrm{Holm}}$ & Sig. \\"
            "\n" r"\hline" "\n")
    body = ""
    for r in res["comparisons"]:
        body += (f"{r['format_a'].replace('_', ' ')} & {r['format_b'].replace('_', ' ')} & "
                 f"{r['accuracy_a']*100:.2f} & {r['accuracy_b']*100:.2f} & "
                 f"{r['acc_diff_points']:+.2f} & {r['p_holm']:.2e} & "
                 f"{stars(r['p_holm'])} \\\\\n")
    return head + body + r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n"


def console(res: dict):
    print("\n" + "=" * 96)
    print(f"{res['model'].upper()}  |  n = {res['n_test_samples']}  |  "
          f"{res['n_comparisons']} comparisons, Holm-corrected at alpha = {res['alpha']}")
    print("=" * 96)
    print(f"{'A':<22}{'B':<22}{'dAcc':>8}{'b':>6}{'c':>6}{'p':>11}{'p_Holm':>11}{'':>5}")
    print("-" * 96)
    for r in res["comparisons"]:
        mark = "  <<<" if r["is_key_pair"] else ""
        print(f"{r['format_a']:<22}{r['format_b']:<22}{r['acc_diff_points']:>+8.2f}"
              f"{r['b']:>6}{r['c']:>6}{r['p_value']:>11.2e}{r['p_holm']:>11.2e}"
              f"  {stars(r['p_holm'])}{mark}")

    print("\nKEY PAIRS — read these into the paper:")
    for r in res["comparisons"]:
        if not r["is_key_pair"]:
            continue
        ci = r["paired_ci95_points"]
        verdict = ("SIGNIFICANT" if r["significant_holm"] else
                   "NOT significant - do not claim a difference")
        print(f"  {r['format_a']} vs {r['format_b']}: "
              f"{r['acc_diff_points']:+.2f} pts, paired CI [{ci[0]:+.2f}, {ci[1]:+.2f}], "
              f"p_Holm = {r['p_holm']:.3g}  ->  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", required=True,
                    help="directory holding quant_02's cached predictions")
    ap.add_argument("--model", choices=["fer", "ser", "both"], default="both")
    ap.add_argument("--out-dir", default="outputs/quant")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    models = ["fer", "ser"] if args.model == "both" else [args.model]
    md_all = ["# Paired significance tests (McNemar)", "",
              "Formats are evaluated on the same held-out test samples, so marginal "
              "confidence intervals are the wrong tool - they discard the pairing and "
              "are underpowered. These are paired tests on the discordant samples.", ""]

    for m in models:
        labels, preds = load_model_predictions(preds_dir, m)
        print(f"\nloaded {m}: n = {len(labels)}, formats = {sorted(preds)}")
        res = analyse(m, labels, preds, args.alpha, args.n_boot)
        console(res)

        # encoding is explicit: the tables contain non-cp1252 characters and
        # Windows' default codec raises UnicodeEncodeError on them.
        (out_dir / f"mcnemar_{m}.json").write_text(json.dumps(res, indent=2),
                                                   encoding="utf-8")
        (out_dir / "tables" / f"mcnemar_{m}.tex").write_text(to_latex(res),
                                                             encoding="utf-8")
        md_all.append(to_markdown(res))

    (out_dir / "tables" / "mcnemar.md").write_text("\n".join(md_all), encoding="utf-8")
    print(f"\nsaved -> {out_dir}/mcnemar_*.json and {out_dir}/tables/mcnemar.*")


if __name__ == "__main__":
    main()
