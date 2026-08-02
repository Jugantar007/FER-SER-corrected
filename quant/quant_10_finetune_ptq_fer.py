"""
QUANT-10  Fine-tune-then-PTQ on FER — the cheap alternative to a QAT rebuild.

WHY
    A5-control showed that on SER, +5.83 of QAT's +10.42 point advantage came
    from the extra 15 epochs of fine-tuning rather than from quantization
    awareness, and QAT's own +4.58 did not reach significance. That matters most
    for FER, where QAT would mean rebuilding EfficientNet-B0 in Keras and
    retraining (1-2 weeks) because tfmot is a TensorFlow tool and this model is
    PyTorch.

    Fine-tuning the existing PyTorch model and then post-training quantizing it
    needs no rebuild, no tfmot and no architecture change. If SER's pattern
    holds, most of the recoverable gap is available for hours of work.

WHAT IS HELD FIXED
    Architecture, preprocessing (bilinear resize + ImageNet normalisation, per
    config.FER_RESIZE_FILTER), the frozen 1948-image test set, and the training
    recipe from quant_09: 15 epochs, Adam at 1e-4, batch 32, balanced class
    weights, early stopping on val loss (patience 4, restore best).

    One thing is better here than in the SER control: FER's validation split is
    the corrected notebook's own cluster-disjoint split, so the stopping signal
    carries no augmentation leakage.

    Calibration for PTQ draws from TRAIN only, n=200 balanced at seed 42 --
    identical to quant_01, so the comparison against the committed
    int8_full_perchannel build differs only by the fine-tuning.

Output: artifacts/models/fer_finetuned.pth, fer_finetuned_int8.tflite
        artifacts/results/quant_finetune_ptq_fer.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import config as C
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    run_tflite, agreement_metrics)
from quant_02_evaluate_formats import load_fer_test, check_frozen

log = get_logger("quant_10_finetune_fer")

MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)


def load_split_u8(which):
    """Decode a split once into RAM as uint8; 9108 JPEGs per epoch is otherwise
    the bottleneck. Normalisation happens per batch on the GPU."""
    from PIL import Image
    items = json.loads((C.CACHE / "fer_split.json").read_text())["files"][which]
    resample = getattr(Image, C.FER_RESIZE_FILTER)
    X = np.empty((len(items), 224, 224, 3), dtype=np.uint8)
    y = np.empty(len(items), dtype=np.int64)
    step = max(1, len(items) // 10)
    for i, (p, lbl) in enumerate(items):
        X[i] = np.asarray(Image.open(p).convert("RGB").resize((224, 224), resample))
        y[i] = lbl
        if (i + 1) % step == 0:
            log.info("  load %s: %d/%d", which, i + 1, len(items))
    return X, y


def fer_representative_dataset(n, seed):
    """quant_01's FER representative set: TRAIN split only, class balanced."""
    from PIL import Image
    items = json.loads((C.CACHE / "fer_split.json").read_text())["files"]["train"]
    rng = np.random.default_rng(seed)
    resample = getattr(Image, C.FER_RESIZE_FILTER)
    by_cls = {c: [p for p, lbl in items if lbl == c] for c in range(len(C.FER_CLASSES))}
    per = max(1, n // len(C.FER_CLASSES))
    chosen = []
    for c in sorted(by_cls):
        pool = by_cls[c]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        chosen += [pool[i] for i in idx]

    def gen():
        for p in chosen:
            a = np.asarray(Image.open(p).convert("RGB").resize((224, 224), resample),
                           dtype=np.float32) / 255.0
            yield [((a - MEAN) / STD)[None, ...].astype(np.float32)]
    return gen


def finetune(args):
    import torch
    import torch.nn as nn
    from torchvision import models

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", dev)

    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(C.FER_CLASSES))
    m.load_state_dict(torch.load(C.MODELS / "fer_baseline.pth", map_location="cpu"))
    m.to(dev)

    Xtr, ytr = load_split_u8("train")
    Xva, yva = load_split_u8("val")
    log.info("train %s  val %s", Xtr.shape, Xva.shape)

    counts = np.bincount(ytr, minlength=len(C.FER_CLASSES)).astype(np.float64)
    weights = torch.tensor((counts.sum() / (len(counts) * counts)),
                           dtype=torch.float32, device=dev)
    log.info("class weights: %s", np.round(weights.cpu().numpy(), 3).tolist())

    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    mean_t = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def batches(X, y, bs, shuffle):
        idx = np.random.default_rng(C.SEED).permutation(len(y)) if shuffle \
            else np.arange(len(y))
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            xb = torch.from_numpy(X[sel]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            xb = (xb - mean_t) / std_t
            yield xb, torch.from_numpy(y[sel]).to(dev)

    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    history = []
    for ep in range(1, args.epochs + 1):
        m.train()
        t0, tot, correct, loss_sum = time.perf_counter(), 0, 0, 0.0
        for xb, yb in batches(Xtr, ytr, args.batch, True):
            opt.zero_grad(set_to_none=True)
            out = m(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(yb)
            correct += int((out.argmax(1) == yb).sum())
            tot += len(yb)

        m.eval()
        v_loss, v_correct, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in batches(Xva, yva, args.batch, False):
                out = m(xb)
                v_loss += float(crit(out, yb)) * len(yb)
                v_correct += int((out.argmax(1) == yb).sum())
                v_tot += len(yb)
        vl, va = v_loss / v_tot, v_correct / v_tot
        history.append({"epoch": ep, "loss": loss_sum / tot, "accuracy": correct / tot,
                        "val_loss": vl, "val_accuracy": va})
        log.info("epoch %2d/%d  loss %.4f acc %.4f | val_loss %.4f val_acc %.4f | %.0fs",
                 ep, args.epochs, loss_sum / tot, correct / tot, vl, va,
                 time.perf_counter() - t0)

        if vl < best["val_loss"]:
            best = {"val_loss": vl, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
        elif ep - best["epoch"] >= args.patience:
            log.info("early stopping at epoch %d (best was %d)", ep, best["epoch"])
            break

    log.info("restoring best epoch %d (val_loss %.4f)", best["epoch"], best["val_loss"])
    m.load_state_dict(best["state"])
    torch.save(m.state_dict(), C.MODELS / "fer_finetuned.pth")
    return m.eval(), history, best["epoch"]


def to_tflite_int8(model, args):
    """Same chain as quant_01: torch -> ONNX -> SavedModel (onnx2tf) -> TFLite."""
    import torch
    import tensorflow as tf

    onnx_path = C.MODELS / "fer_finetuned.onnx"
    torch.onnx.export(
        model.cpu(), torch.randn(1, 3, 224, 224), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, export_params=True)
    log.info("ONNX -> %s (%.2f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)

    sm_dir = C.MODELS / "fer_finetuned_saved_model"
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path),
           "-o", str(sm_dir), "-nuo", "--non_verbose"]
    log.info("running: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("onnx2tf failed:\n%s\n%s", r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(1)

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = fer_representative_dataset(args.calib_n, C.SEED)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    out = C.MODELS / "fer_finetuned_int8.tflite"
    out.write_bytes(blob)
    log.info("INT8 -> %s (%.3f MB)", out.name, len(blob) / 1e6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--calib-n", type=int, default=C.CALIB_SAMPLES)
    args = ap.parse_args()

    X, y, class_names = load_fer_test()
    check_frozen("fer", len(y))

    model, history, best_epoch = finetune(args)

    # float accuracy of the fine-tuned model, on the frozen test set
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    preds = np.empty(len(y), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(y), 64):
            xb = torch.from_numpy(X[i:i + 64]).to(dev).permute(0, 3, 1, 2).float()
            preds[i:i + 64] = model(xb).argmax(1).cpu().numpy()
    float_acc = float((preds == y).mean())
    log.info("fine-tuned FLOAT accuracy on frozen test: %.4f (baseline 0.8249)", float_acc)

    tfl = to_tflite_int8(model, args)
    P = run_tflite(tfl, X, "fer/finetuned_int8")
    np.save(C.CACHE / "preds_fer_finetuned_int8_control.npy", P)
    m = full_metrics(y, P.argmax(1), class_names, y_prob=P)
    m.update(bootstrap_ci(y, P.argmax(1)))

    committed = load_json(C.RESULTS / "quant_format_evaluation.json")["fer"]
    ptq_base = committed["int8_full_perchannel"]["accuracy"]
    fp32_base = committed["fp32"]["accuracy"]
    gained = (m["accuracy"] - ptq_base) * 100

    verdict = (
        f"PTQ(baseline) {ptq_base*100:.2f}% -> PTQ(fine-tuned) {m['accuracy']*100:.2f}% "
        f"({gained:+.2f} pts), against FP32 {fp32_base*100:.2f}%. Fine-tuned float is "
        f"{float_acc*100:.2f}%. Reference-kernel path; see the results file for the "
        f"delegated numbers.")
    log.info("%s", verdict)

    save_json({"model": "fer",
               "purpose": ("fine-tune-then-PTQ, the cheap alternative to a Keras "
                           "rebuild for QAT (see A5-control)"),
               "epochs_requested": args.epochs, "epochs_run": len(history),
               "best_epoch": best_epoch, "learning_rate": args.lr,
               "batch": args.batch, "calibration_samples": args.calib_n,
               "calibration": "TRAIN split only, balanced, seed 42 (as quant_01)",
               "validation": "the corrected notebook's cluster-disjoint val split",
               "fp32_baseline_accuracy": fp32_base,
               "finetuned_float_accuracy": float_acc,
               "ptq_baseline_accuracy": ptq_base,
               "ptq_finetuned": m,
               "points_gained_vs_ptq_baseline": round(gained, 2),
               "history": history, "verdict": verdict},
              C.RESULTS / "quant_finetune_ptq_fer.json")
    log.info("saved -> %s", C.RESULTS / "quant_finetune_ptq_fer.json")


if __name__ == "__main__":
    main()
