"""
QUANT-01  Produce every deployment format for both models (README items A1, A6).

FER: PyTorch -> ONNX -> TensorFlow SavedModel (onnx2tf) -> TFLite
SER: Keras -> SavedModel -> TFLite

Formats produced per model:
  fp32                 control build; isolates conversion faults from quantization faults
  fp16                 half-precision weights
  dynrange             INT8 weights, float activations          (the paper's selected format)
  int8_full_perchannel INT8 weights + activations, per-channel weights (TFLite default)
  int8_full_pertensor  INT8 weights + activations, per-tensor weights  (ablation for A6)

The two int8_full variants are A6: recent TFLite defaults to per-channel, and
`_experimental_disable_per_channel` turns it off. State in the paper which one
the reported "full INT8" numbers correspond to -- the default has moved across
TensorFlow versions and the ambiguity is a legitimate reviewer question.

Calibration uses CALIB_SAMPLES drawn from the TRAINING split only -- never from
val or test, or the quantization study leaks too.

Output: artifacts/models/{fer,ser}_<format>.tflite
        artifacts/results/quant_conversion_manifest.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

import config as C
from common import get_logger, save_json, load_keras_compat, keras_saved_model_dir

log = get_logger("quant_01_convert")

MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)

ONNX_TOLERANCE = 1e-3


# ------------------------------------------------------------ representative sets
def fer_representative_dataset(n=None, balanced=True, seed=C.SEED):
    """NHWC, ImageNet-normalised, drawn from the TRAINING split only."""
    from PIL import Image
    n = n or C.CALIB_SAMPLES
    items = json.loads((C.CACHE / "fer_split.json").read_text())["files"]["train"]
    rng = np.random.default_rng(seed)
    resample = getattr(Image, C.FER_RESIZE_FILTER)

    if balanced:
        by_cls = {c: [p for p, lbl in items if lbl == c] for c in range(len(C.FER_CLASSES))}
        per = max(1, n // len(C.FER_CLASSES))
        chosen = []
        for c in sorted(by_cls):
            pool = by_cls[c]
            idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
            chosen += [pool[i] for i in idx]
    else:
        idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
        chosen = [items[i][0] for i in idx]

    def gen():
        for p in chosen:
            a = np.asarray(Image.open(p).convert("RGB").resize((224, 224), resample),
                           dtype=np.float32) / 255.0
            yield [((a - MEAN) / STD)[None, ...].astype(np.float32)]
    return gen


def ser_representative_dataset(n=None, balanced=True, seed=C.SEED):
    """Drawn from the rebuilt TRAINING actors (1-16) only."""
    n = n or C.CALIB_SAMPLES
    d = np.load(C.CACHE / "ser_train.npz")
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(seed)

    if balanced:
        per = max(1, n // 8)
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum())),
                       replace=False)
            for c in range(8) if (y == c).any()])
    else:
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    def gen():
        for i in idx:
            yield [(X[i].astype(np.float32) / 255.0)[None, ...]]
    return gen


# ------------------------------------------------------------------ FER path
def load_fer_torch():
    import torch
    import torch.nn as nn
    from torchvision import models
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(C.FER_CLASSES))
    m.load_state_dict(torch.load(C.MODELS / "fer_baseline.pth", map_location="cpu"))
    return m.eval()


def export_fer_onnx(dest):
    """Re-export a SELF-CONTAINED ONNX from the .pth checkpoint.

    The ONNX committed by the notebook stores its weights in an external
    `.onnx.data` sidecar that was never committed, so it cannot be loaded on its
    own. The .pth is complete, so the export is regenerated here with
    export_params=True and everything inlined.
    """
    import torch
    m = load_fer_torch()
    torch.onnx.export(
        m, torch.randn(1, 3, 224, 224), str(dest),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, export_params=True,
    )
    log.info("exported self-contained ONNX -> %s (%.2f MB)",
             dest.name, dest.stat().st_size / 1e6)
    return dest


def onnx_is_self_contained(path):
    """True if the file carries its own weights (no external .data sidecar)."""
    try:
        import onnx
        onnx.load(str(path))
        return True
    except Exception as e:
        log.warning("%s is not self-contained: %s", Path(path).name, str(e)[:160])
        return False


def verify_onnx(onnx_path):
    """The ONNX export is the first link in the FER chain. Verify it against
    PyTorch before quantizing anything -- do not proceed past a divergence."""
    import torch
    import onnxruntime as ort

    m = load_fer_torch()
    rng = np.random.default_rng(C.SEED)
    x = rng.standard_normal((4, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(onnx_path),
                               providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    log.info("ONNX max |logit diff| vs PyTorch: %.3e", diff)
    return diff


def onnx_to_saved_model(onnx_path, force=False):
    """onnx2tf converts NCHW -> NHWC, which TFLite needs."""
    out_dir = C.MODELS / "fer_saved_model"
    if out_dir.exists() and not force:
        log.info("reusing %s (pass --force to rebuild)", out_dir)
        return out_dir
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path),
           "-o", str(out_dir), "-nuo", "--non_verbose"]
    log.info("running: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("onnx2tf failed:\n%s\n%s", r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)
    log.info("SavedModel -> %s", out_dir)
    return out_dir


def verify_saved_model(saved_model_dir, onnx_path):
    """onnx2tf transposes the graph; confirm it still computes the same function."""
    import onnxruntime as ort
    rng = np.random.default_rng(C.SEED + 1)
    x_nchw = rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
    ref = ort.InferenceSession(str(onnx_path),
                               providers=["CPUExecutionProvider"]).run(
                                   None, {"input": x_nchw})[0]
    loaded = tf.saved_model.load(str(saved_model_dir))
    fn = loaded.signatures["serving_default"]
    x_nhwc = tf.constant(x_nchw.transpose(0, 2, 3, 1))
    got = list(fn(x_nhwc).values())[0].numpy()
    diff = float(np.abs(ref - got).max())
    log.info("SavedModel max |logit diff| vs ONNX: %.3e", diff)
    return diff


# ------------------------------------------------------------- conversion
def apply_format(conv, fmt, rep_ds=None, int8_io=True):
    if fmt == "fp32":
        pass
    elif fmt == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif fmt == "dynrange":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif fmt in ("int8_full_perchannel", "int8_full_pertensor"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_ds
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if int8_io:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        if fmt == "int8_full_pertensor":
            # A6: disables TFLite's default per-channel weight quantization.
            conv._experimental_disable_per_channel = True
    else:
        raise ValueError(fmt)
    return conv.convert()


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def convert_family(tag, converter_factory, rep, manifest, int8_io=True):
    entry = manifest["models"].setdefault(tag, {})
    entry["tflite"] = {}
    for fmt in C.FORMATS:
        out = C.MODELS / f"{tag}_{fmt}.tflite"
        try:
            blob = apply_format(converter_factory(), fmt, rep, int8_io=int8_io)
        except Exception as e:
            log.error("%s %s conversion failed: %s", tag.upper(), fmt, e)
            entry["tflite"][fmt] = None
            continue
        out.write_bytes(blob)
        size = out.stat().st_size
        entry["tflite"][fmt] = {"bytes": size, "mb": round(size / 1e6, 3),
                                "sha256_16": sha256(out)}
        log.info("%s %-22s -> %8.3f MB", tag.upper(), fmt, size / 1e6)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["fer", "ser"], default=["fer", "ser"])
    ap.add_argument("--calib-n", type=int, default=C.CALIB_SAMPLES)
    ap.add_argument("--force", action="store_true", help="rebuild the FER SavedModel")
    ap.add_argument("--float-io", action="store_true",
                    help="keep float32 in/out on the full-INT8 builds "
                         "(default is int8 in/out, which is what the Pi deploys)")
    ap.add_argument("--allow-onnx-drift", action="store_true",
                    help="continue even if the ONNX export diverges from PyTorch")
    args = ap.parse_args()
    int8_io = not args.float_io

    manifest = {
        "formats": C.FORMATS,
        "calibration": {"n_samples": args.calib_n, "balanced": True,
                        "source": "TRAINING split only (never val/test)"},
        "int8_inference_io": "int8" if int8_io else "float32",
        "per_channel_note": ("int8_full_perchannel is the TFLite default; "
                             "int8_full_pertensor sets "
                             "converter._experimental_disable_per_channel = True"),
        "models": {},
    }

    if "fer" in args.models:
        log.info("--- FER ---")
        onnx_path = C.MODELS / "fer_baseline.onnx"
        reexported = False
        if args.force or not onnx_path.exists() or not onnx_is_self_contained(onnx_path):
            export_fer_onnx(onnx_path)
            reexported = True

        onnx_diff = verify_onnx(onnx_path)
        if onnx_diff > ONNX_TOLERANCE:
            log.warning("ONNX export diverges from PyTorch (%.3e > %.0e)",
                        onnx_diff, ONNX_TOLERANCE)
            if not args.allow_onnx_drift:
                log.error("stopping: do not quantize a graph that already disagrees "
                          "with the baseline. Re-export, or pass --allow-onnx-drift.")
                sys.exit(1)

        sm = onnx_to_saved_model(onnx_path, force=args.force)
        sm_diff = verify_saved_model(sm, onnx_path)

        manifest["models"]["fer"] = {
            "pytorch_mb": round((C.MODELS / "fer_baseline.pth").stat().st_size / 1e6, 3),
            "onnx_mb": round(onnx_path.stat().st_size / 1e6, 3),
            "onnx_reexported_from_checkpoint": reexported,
            "onnx_max_logit_diff_vs_pytorch": onnx_diff,
            "saved_model_max_logit_diff_vs_onnx": sm_diff,
        }
        rep = fer_representative_dataset(n=args.calib_n)
        convert_family("fer", lambda: tf.lite.TFLiteConverter.from_saved_model(str(sm)),
                       rep, manifest, int8_io=int8_io)

    if "ser" in args.models:
        log.info("--- SER ---")
        keras_path = C.MODELS / "ser_baseline.keras"
        if not keras_path.exists():
            log.error("missing %s -- run quant_00_prepare.py first", keras_path)
            sys.exit(1)
        model, patched = load_keras_compat(keras_path)
        sm = keras_saved_model_dir(model, "ser")
        manifest["models"]["ser"] = {
            "keras_kb": round(keras_path.stat().st_size / 1e3, 1),
            "keras_config_patched": patched,
        }
        rep = ser_representative_dataset(n=args.calib_n)
        convert_family("ser", lambda: tf.lite.TFLiteConverter.from_saved_model(str(sm)),
                       rep, manifest, int8_io=int8_io)

    save_json(manifest, C.RESULTS / "quant_conversion_manifest.json")
    log.info("saved -> %s", C.RESULTS / "quant_conversion_manifest.json")


if __name__ == "__main__":
    main()
