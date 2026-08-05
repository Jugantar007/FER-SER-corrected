"""Shared metric, model-loading, TFLite and reporting helpers."""
from __future__ import annotations
import datetime as _dt
import json
import logging
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, balanced_accuracy_score,
                             cohen_kappa_score, roc_auc_score)

import config as C


# ------------------------------------------------------------------ logging
def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(C.LOGS / f"{name}.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ------------------------------------------------------------- environment
def environment_report() -> dict:
    """Captured into every results file. Journals ask for this."""
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    env = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": _v("numpy"), "tensorflow": _v("tensorflow"), "keras": _v("keras"),
        "torch": _v("torch"), "librosa": _v("librosa"), "sklearn": _v("sklearn"),
        "onnx2tf": _v("onnx2tf"), "onnxruntime": _v("onnxruntime"),
    }
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception:
        pass
    return env


# ------------------------------------------------------------------ metrics
def full_metrics(y_true, y_pred, class_names, y_prob=None) -> dict:
    """Every metric the paper needs, in one dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            cn: {"precision": float(p[i]), "recall": float(r[i]),
                 "f1": float(f1[i]), "support": int(sup[i])}
            for i, cn in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))).tolist(),
        "class_names": list(class_names),
    }
    if y_prob is not None:
        try:
            out["macro_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            out["macro_auc_ovr"] = None
    return out


def bootstrap_ci(y_true, y_pred, n_boot=2000, alpha=0.05, seed=C.SEED):
    """95% CI on accuracy. Reviewers at Q1 venues expect intervals, not point estimates."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return {"accuracy_mean": float(accs.mean()),
            "ci95_low": float(lo), "ci95_high": float(hi)}


def agreement_metrics(prob_ref, prob_test) -> dict:
    """How closely a converted/quantized model reproduces the baseline.
    This is the paper's 'verification' idea generalised to the full test set."""
    prob_ref = np.asarray(prob_ref, dtype=np.float64)
    prob_test = np.asarray(prob_test, dtype=np.float64)
    top1 = (prob_ref.argmax(1) == prob_test.argmax(1)).mean()
    eps = 1e-12
    kl = float(np.mean(np.sum(prob_ref * (np.log(prob_ref + eps) -
                                          np.log(prob_test + eps)), axis=1)))
    return {
        "top1_agreement": float(top1),
        "mean_abs_prob_diff": float(np.mean(np.abs(prob_ref - prob_test))),
        "max_abs_prob_diff": float(np.max(np.abs(prob_ref - prob_test))),
        "mean_kl_divergence": kl,
    }


def four_class_summation(prob8):
    """Deployed SER protocol: SUM the eight class probabilities into four.
    Never max(), never a filtered subset -- that was fault S4/S5."""
    prob8 = np.asarray(prob8, dtype=np.float64)
    return np.stack([prob8[:, idx].sum(1) for _, idx in sorted(C.GROUPS_4.items())], 1)


def labels_8_to_4(y8):
    return np.array([C.MAP_8_TO_4[int(v)] for v in np.asarray(y8)])


# ------------------------------------------------------------------- output
def save_json(obj: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(obj)
    payload.setdefault("_environment", environment_report())
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_confusion(cm, class_names, title, path: Path, normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Greys", vmin=0)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thr else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def print_report(name, m):
    print(f"\n{'='*62}\n{name}\n{'='*62}")
    print(f"  n              : {m['n_samples']}")
    print(f"  accuracy       : {m['accuracy']*100:.2f}%")
    print(f"  balanced acc   : {m['balanced_accuracy']*100:.2f}%")
    print(f"  macro F1       : {m['macro_f1']:.4f}")
    print(f"  weighted F1    : {m['weighted_f1']:.4f}")
    print(f"  Cohen's kappa  : {m['cohen_kappa']:.4f}")
    print(f"  {'class':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>7}")
    for cn, d in m["per_class"].items():
        print(f"  {cn:<14}{d['precision']:>8.3f}{d['recall']:>8.3f}"
              f"{d['f1']:>8.3f}{d['support']:>7d}")


# ---------------------------------------------------------------- Keras load
# Colab wrote the checkpoint with a newer Keras than may be installed locally;
# newer versions embed keys (e.g. `quantization_config`) that older ones reject.
# Stripping them is safe: they are all None for this model, and the weights --
# which is what actually matters -- are untouched.
_FORWARD_COMPAT_KEYS = ("quantization_config",)


def _strip_keys(obj):
    if isinstance(obj, dict):
        return {k: _strip_keys(v) for k, v in obj.items()
                if k not in _FORWARD_COMPAT_KEYS}
    if isinstance(obj, list):
        return [_strip_keys(v) for v in obj]
    return obj


def load_keras_compat(path, compile=False):
    """Load a .keras file, tolerating a forward-version config mismatch.

    Returns (model, was_patched). Raises if the model cannot be loaded at all.
    """
    import keras
    path = Path(path)
    try:
        return keras.models.load_model(path, compile=compile), False
    except Exception:
        pass

    patched = C.CACHE / f"{path.stem}__compat.keras"
    with zipfile.ZipFile(path) as zin, \
            zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(_strip_keys(json.loads(data))).encode()
            zout.writestr(item, data)
    return keras.models.load_model(patched, compile=compile), True


def keras_saved_model_dir(model, tag):
    """Keras 3 models are not always accepted by TFLiteConverter.from_keras_model;
    exporting to a SavedModel first is the reliable path."""
    out = C.MODELS / f"{tag}_saved_model"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    model.export(str(out))
    return out


# -------------------------------------------------------------------- TFLite
import os as _os

# Quantized EfficientNet graphs fall back to slow reference kernels on x86, so
# thread count dominates wall-clock here. This has no bearing on any reported
# number -- deployment latency is measured on the Pi (Group B), never on x86.
DEFAULT_THREADS = int(_os.environ.get("QUANT_NUM_THREADS", min(8, _os.cpu_count() or 4)))


# ai-edge-litert 2.1.4 applies NO delegate unless asked, so quantized graphs run
# TFLite's reference kernels -- which is not what a deployment runtime does. On
# FER that is worth +10.6 points and ~80x speed on full INT8. Set QUANT_XNNPACK=1
# to evaluate on the delegated path instead. Results from the two paths are NOT
# interchangeable; keep them in separate files.
USE_XNNPACK = _os.environ.get("QUANT_XNNPACK", "0") == "1"


def _interpreter(model_path=None, model_content=None, num_threads=None):
    """Prefer LiteRT when available; tf.lite.Interpreter is deprecated but works."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter
    nt = num_threads or DEFAULT_THREADS
    kw = {"num_threads": nt}
    if USE_XNNPACK:
        kw["experimental_default_delegate_latest_features"] = True
    if model_content is not None:
        return Interpreter(model_content=model_content, **kw)
    return Interpreter(model_path=str(model_path), **kw)


class DelegatePrepareError(RuntimeError):
    """XNNPACK accepted the graph then failed to prepare its runtime.

    Seen on one selective (mixed-precision) FER build: k=5 fails at "node 255"
    while k=3, k=10 and k=15 prepare fine.

    This is a FLAKY RUNTIME DEFECT, not a property of the model. quant_12
    pins it down: node 255 is the delegate node itself (always the last index in
    the delegated plan), k=5 is structurally unremarkable, and the failure needs
    a prior interpreter destroyed in-process plus a low thread count -- a fresh
    process is 0/30 at every thread count. Retrying in a new process works; the
    k=5 accuracy was recovered that way. Do NOT read this exception as "the
    deployment runtime refuses to run this model".

    Raised as its own type so callers can retry or record the build as unmeasured
    instead of crashing -- and so it is never silently downgraded to a
    reference-kernel run, which would mix execution paths inside one result file.
    """


def run_tflite(model, X, desc="", progress=True, num_threads=None):
    """Run a TFLite model over X (NHWC float32) and return float probabilities.

    Handles float and fully-quantized (int8/uint8 in/out) interpreters alike, and
    applies softmax when the graph emits logits rather than probabilities.
    """
    if isinstance(model, (bytes, bytearray)):
        interp = _interpreter(model_content=bytes(model), num_threads=num_threads)
    else:
        interp = _interpreter(model_path=model, num_threads=num_threads)
    try:
        interp.allocate_tensors()
    except RuntimeError as e:
        if USE_XNNPACK and ("XNNPACK" in str(e) or "delegate" in str(e).lower()):
            raise DelegatePrepareError(f"{desc or model}: {e}") from e
        raise
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    n = len(X)
    probs = np.empty((n, out["shape"][-1]), dtype=np.float64)
    step = max(1, n // 20)
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
        if progress and desc and (i + 1) % step == 0:
            print(f"\r    {desc}: {i+1}/{n}", end="", flush=True)
    if progress and desc:
        print(f"\r    {desc}: {n}/{n}", flush=True)

    if not np.allclose(probs.sum(1), 1.0, atol=1e-2):     # logits -> softmax
        e = np.exp(probs - probs.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
    return probs


def tflite_size_mb(path):
    return round(Path(path).stat().st_size / 1e6, 3)
