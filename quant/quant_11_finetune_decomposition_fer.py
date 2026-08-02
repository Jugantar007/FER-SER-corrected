"""
QUANT-11  Does fine-tune-then-PTQ actually help FER? Decomposed, on both paths.

quant_10 reports one number: PTQ(fine-tuned) beats PTQ(baseline) by +5.08 points
on reference kernels. That number hides two effects pulling in different
directions, and on the deployment path they do not cancel the same way:

    raw gain = (better float model) + (change in quantization penalty)

The fine-tuned model starts 2.77 points higher in float, so a raw comparison
against the baseline's INT8 build credits fine-tuning for accuracy that has
nothing to do with quantizability. What matters for a deployment decision is the
*penalty* each model pays when quantized -- and that reverses between paths.

Reuses every cached prediction; only the float models (seconds on GPU) and the
fine-tuned INT8 build on the delegated path (~45 s) need computing.

Output: artifacts/results/quant_finetune_decomposition_fer.json
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(r"D:\Research FER and SER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))
import config as C
from common import save_json
from ai_edge_litert.interpreter import Interpreter
from quant_02_evaluate_formats import load_fer_test, check_frozen

X, y, _ = load_fer_test()
check_frozen("fer", len(y))


def float_preds(ckpt):
    import torch
    import torch.nn as nn
    from torchvision import models
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(C.FER_CLASSES))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.to(dev).eval()
    out = np.empty(len(y), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(y), 64):
            xb = torch.from_numpy(X[i:i + 64]).to(dev).permute(0, 3, 1, 2).float()
            out[i:i + 64] = m(xb).argmax(1).cpu().numpy()
    return out


def run_delegated(path):
    it = Interpreter(model_path=str(path), num_threads=8,
                     experimental_default_delegate_latest_features=True)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    P = np.empty((len(X), out["shape"][-1]), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(len(X)):
        x = X[i:i + 1]
        s, z = inp["quantization"]
        x = np.clip(np.round(x / s + z), -128, 127).astype(inp["dtype"])
        it.set_tensor(inp["index"], x)
        it.invoke()
        v = it.get_tensor(out["index"])[0]
        s2, z2 = out["quantization"]
        P[i] = (v.astype(np.float64) - z2) * s2
    print(f"  (delegated pass {1000*(time.perf_counter()-t0)/len(X):.1f} ms/img)")
    return P.argmax(1)


def mcnemar(a, b, label):
    ca, cb = (a == y), (b == y)
    bb, cc = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = bb + cc
    p = binomtest(bb, n, 0.5).pvalue if n else 1.0
    print(f"  {label:<44} d={100*(bb-cc)/len(y):+6.2f} pts  b={bb} c={cc} "
          f"n_disc={n}  p={p:.3g}")
    return {"delta_points": round(100 * (bb - cc) / len(y), 2), "b": bb, "c": cc,
            "n_discordant": n, "p_value": float(p)}


cache = C.CACHE
f_base = float_preds(C.MODELS / "fer_baseline.pth")
f_ft = float_preds(C.MODELS / "fer_finetuned.pth")
acc_fb, acc_ft = float(np.mean(f_base == y)), float(np.mean(f_ft == y))
print(f"FLOAT: baseline {acc_fb:.4f}  fine-tuned {acc_ft:.4f}")
float_test = mcnemar(f_ft, f_base, "float fine-tuned vs float baseline")
out = {"model": "fer",
       "purpose": ("decompose quant_10's raw gain into 'better float model' and "
                   "'changed quantization penalty', on both execution paths"),
       "float_baseline_accuracy_torch": acc_fb,
       "float_finetuned_accuracy_torch": acc_ft,
       "note_torch_vs_tflite": ("float accuracies here are from PyTorch; the TFLite "
                               "fp32 control reads 0.8249 against torch's 0.8244, a "
                               "one-image difference consistent with GPU "
                               "non-determinism (REPORT §6)"),
       "float_finetuned_vs_baseline": float_test,
       "paths": {}}

paths = {
    "reference kernels": (
        np.load(cache / "preds_fer_int8_full_perchannel_1785496891_4891392.npy").argmax(1),
        np.load(cache / "preds_fer_finetuned_int8_control.npy").argmax(1)),
    "XNNPACK": (
        np.load(cache / "xnn_preds_fer_int8_full_perchannel.npy").argmax(1),
        run_delegated(C.MODELS / "fer_finetuned_int8.tflite")),
}

for tag, (p_base, p_ft) in paths.items():
    a_base, a_ft = float(np.mean(p_base == y)), float(np.mean(p_ft == y))
    pen_base, pen_ft = 100 * (a_base - acc_fb), 100 * (a_ft - acc_ft)
    print(f"\n=== {tag} ===")
    print(f"  PTQ(baseline)   {a_base:.4f}   penalty {pen_base:+.2f} pts vs its float")
    print(f"  PTQ(fine-tuned) {a_ft:.4f}   penalty {pen_ft:+.2f} pts vs its float")
    print(f"  raw gain {100*(a_ft-a_base):+.2f} = "
          f"{100*(acc_ft-acc_fb):+.2f} better float + {pen_ft-pen_base:+.2f} smaller penalty")
    test = mcnemar(p_ft, p_base, "PTQ(fine-tuned) vs PTQ(baseline)")
    out["paths"][tag] = {
        "ptq_baseline_accuracy": a_base, "ptq_finetuned_accuracy": a_ft,
        "penalty_baseline_points": round(pen_base, 2),
        "penalty_finetuned_points": round(pen_ft, 2),
        "raw_gain_points": round(100 * (a_ft - a_base), 2),
        "from_better_float_points": round(100 * (acc_ft - acc_fb), 2),
        "from_changed_penalty_points": round(pen_ft - pen_base, 2),
        "ptq_finetuned_vs_baseline": test}

out["verdict"] = (
    "Fine-tuning yields a better float model (+%.2f pts, p=%.3g) that quantizes "
    "WORSE. On reference kernels the two effects happen to sum positive (+%.2f); "
    "on the delegated deployment path they do not (%.2f, p=%.3g). "
    "Fine-tune-then-PTQ is not a substitute for QAT on FER."
    % (100 * (acc_ft - acc_fb), float_test["p_value"],
       out["paths"]["reference kernels"]["raw_gain_points"],
       out["paths"]["XNNPACK"]["raw_gain_points"],
       out["paths"]["XNNPACK"]["ptq_finetuned_vs_baseline"]["p_value"]))
print("\n" + out["verdict"])
save_json(out, C.RESULTS / "quant_finetune_decomposition_fer.json")
print("saved ->", C.RESULTS / "quant_finetune_decomposition_fer.json")
