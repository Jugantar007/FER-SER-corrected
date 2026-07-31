"""
QUANT-00  Stage the corrected-notebook artifacts for the quantization study.

The Group A scripts must reuse the EXACT test sets the corrected notebooks
produced (GROUP_A rule 1: "test sets are frozen"). This script is the bridge
between `outputs/` (what the notebooks committed) and `artifacts/` (what
quant_01..06 consume), and it verifies the wiring rather than assuming it.

It performs no training and modifies nothing under outputs/.

Produces
  artifacts/models/fer_baseline.pth      copy of the corrected checkpoint
  artifacts/models/fer_baseline.onnx     copy of the corrected ONNX export
  artifacts/models/ser_baseline.keras    copy of the corrected Keras checkpoint
  artifacts/cache/fer_split.json         split manifest, paths resolved on this machine
  artifacts/cache/ser_test.npz           frozen SER test set (from the notebook)
  artifacts/cache/ser_train.npz          SER TRAINING features, rebuilt for calibration
  artifacts/results/quant_00_prepare.json

Verifications (all recorded in the JSON)
  * every path in fer_split.json resolves under FER_DATA_ROOT
  * split sizes match the corrected results JSON, and splits are disjoint
  * SER actor-disjointness holds for the rebuilt training features
  * the PIL preprocessing reproduces torchvision's eval transform
  * the baselines reproduce the notebooks' published test accuracy
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
from common import (get_logger, save_json, load_json, full_metrics,
                    four_class_summation, labels_8_to_4, load_keras_compat)

log = get_logger("quant_00_prepare")

MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)


# --------------------------------------------------------------------- FER
def resolve_fer_paths():
    """fer_split.json stores Colab-relative paths ('emotion_data/dataset/<Cls>/<f>').
    Re-root them onto this machine and prove every one of them exists."""
    if not C.FER_SPLIT_JSON.exists():
        log.error("missing %s -- run jugantarFER_corrected.ipynb first", C.FER_SPLIT_JSON)
        sys.exit(1)
    raw = load_json(C.FER_SPLIT_JSON)
    root = C.FER_DIR
    if not root.exists():
        log.error("FER dataset root does not exist: %s", root)
        log.error("set FER_DATA_ROOT or paths.local.json (see GROUP_A.md)")
        sys.exit(1)

    marker = "emotion_data/dataset/"
    out, missing = {}, []
    for split, items in raw.items():
        resolved = []
        for p, label in items:
            rel = p.split(marker, 1)[1] if marker in p else Path(p).name
            f = root / rel
            if not f.exists():
                missing.append(str(f))
            resolved.append([str(f), int(label)])
        out[split] = resolved
        log.info("FER %-5s : %5d images", split, len(resolved))

    if missing:
        log.error("%d split paths did not resolve, e.g. %s", len(missing), missing[:3])
        sys.exit(1)
    return out


def check_fer_split(split):
    sizes = {k: len(v) for k, v in split.items()}
    n_test = sizes.get("test", 0)
    if n_test != C.EXPECTED_N_TEST["fer"]:
        log.error("FER test set is %d, expected %d -- STOP (rule 1: test sets are frozen)",
                  n_test, C.EXPECTED_N_TEST["fer"])
        sys.exit(1)

    sets = {k: {p for p, _ in v} for k, v in split.items()}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sets[a] & sets[b]
        overlaps[f"{a}_{b}"] = len(common)
        if common:
            log.error("%s/%s share %d images -- the split leaks", a, b, len(common))
            sys.exit(1)
    log.info("FER splits are disjoint and test n=%d matches the notebook", n_test)
    return {"sizes": sizes, "pairwise_overlap": overlaps}


def fer_load_images(items, desc="FER"):
    """PIL implementation of the notebook's eval transform:
    Resize((224,224), bilinear) -> ToTensor -> Normalize(ImageNet).
    Returns NHWC float32 (TFLite layout)."""
    from PIL import Image
    resample = getattr(Image, C.FER_RESIZE_FILTER)
    X = np.empty((len(items), 224, 224, 3), dtype=np.float32)
    y = np.empty(len(items), dtype=np.int64)
    step = max(1, len(items) // 20)
    for i, (p, label) in enumerate(items):
        img = Image.open(p).convert("RGB").resize((224, 224), resample)
        a = np.asarray(img, dtype=np.float32) / 255.0
        X[i] = (a - MEAN) / STD
        y[i] = label
        if (i + 1) % step == 0:
            print(f"\r    load {desc}: {i+1}/{len(items)}", end="", flush=True)
    print(f"\r    load {desc}: {len(items)}/{len(items)}", flush=True)
    return X, y


def verify_preprocessing(items, n=16):
    """The PIL path must match torchvision's, or every downstream number is
    measured on subtly different pixels than the notebook reported."""
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
    except Exception as e:
        log.warning("skipping preprocessing check (%s)", e)
        return None

    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(list(C.IMAGENET_MEAN), list(C.IMAGENET_STD)),
    ])
    ours, ref = [], []
    for p, _ in items[:n]:
        img = Image.open(p).convert("RGB")
        ref.append(tf_eval(img).numpy())
        resample = getattr(Image, C.FER_RESIZE_FILTER)
        a = np.asarray(img.resize((224, 224), resample), dtype=np.float32) / 255.0
        ours.append(((a - MEAN) / STD).transpose(2, 0, 1))
    diff = float(np.abs(np.asarray(ours) - np.asarray(ref)).max())
    log.info("preprocessing max|diff| vs torchvision over %d images: %.3e", n, diff)
    if diff > 1e-5:
        log.warning("preprocessing does NOT match torchvision (%.3e) -- check "
                    "FER_RESIZE_FILTER in config.py", diff)
    return diff


def verify_fer_baseline(split):
    """Reproduce the notebook's published test accuracy from the checkpoint."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except Exception as e:
        log.warning("skipping FER baseline check (%s)", e)
        return None

    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(C.FER_CLASSES))
    m.load_state_dict(torch.load(C.FER_CHECKPOINT, map_location="cpu"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.eval().to(dev)

    X, y = fer_load_images(split["test"], "FER test")
    preds = np.empty(len(y), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(X), 64):
            xb = torch.from_numpy(X[s:s + 64].transpose(0, 3, 1, 2)).to(dev)
            preds[s:s + 64] = m(xb).argmax(1).cpu().numpy()

    acc = float((preds == y).mean())
    published = load_json(C.FER_RESULTS_JSON)["test_accuracy_REPORT_THIS"]
    log.info("FER baseline: %.4f (notebook published %.4f, delta %+.4f)",
             acc, published, acc - published)
    if abs(acc - published) > 0.005:
        log.warning("FER baseline deviates from the notebook by more than 0.5 points")
    return {"accuracy": acc, "published": published, "delta": acc - published,
            "metrics": full_metrics(y, preds, C.FER_CLASSES)}


# --------------------------------------------------------------------- SER
def ser_spectrogram(y, sr):
    """Byte-for-byte the notebook's get_spectrogram. Do NOT add denoise / trim /
    volume normalisation -- a mismatch here is what broke the original deployment."""
    import librosa
    import cv2
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=C.N_MELS)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    img = melspec_db - melspec_db.min()
    img = (img / (img.max() if img.max() > 0 else 1.0) * 255).astype(np.uint8)
    img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    return np.stack((img,) * 3, axis=-1)


def build_ser_train_cache(augment=True, seed=C.SEED):
    """Rebuild the SER TRAINING features (actors 1-16).

    The notebook saved only the test set, but calibration must draw from training
    data (GROUP_A rule 2). Augmentation mirrors the notebook: training actors only.
    """
    import librosa
    root = C.RAVDESS_DIR
    if not root.exists():
        log.error("RAVDESS root does not exist: %s", root)
        log.error("set RAVDESS_ROOT or paths.local.json (see GROUP_A.md)")
        sys.exit(1)

    rng = np.random.default_rng(seed)

    def add_noise(d):
        amp = 0.035 * rng.uniform() * np.amax(np.abs(d))
        return d + amp * rng.standard_normal(d.shape[0])

    def time_shift(d, max_frac=0.15):
        return np.roll(d, int(rng.uniform(-max_frac, max_frac) * len(d)))

    pairs, actors = [], set()
    for adir in sorted(root.iterdir()):
        if not adir.is_dir():
            continue
        for fn in sorted(p.name for p in adir.glob("*.wav")):
            actor = int(fn.split("-")[6].split(".")[0])
            if actor in C.SER_TRAIN_ACTORS:
                pairs.append((adir, fn))
                actors.add(actor)

    held_out = set(C.SER_VAL_ACTORS) | set(C.SER_TEST_ACTORS)
    if actors & held_out:
        log.error("calibration pool contains held-out actors %s -- STOP", actors & held_out)
        sys.exit(1)
    log.info("SER training files: %d from actors %s", len(pairs), sorted(actors))

    X, y = [], []
    for i, (adir, fn) in enumerate(pairs):
        emotion = int(fn.split("-")[2]) - 1
        audio, sr = librosa.load(str(adir / fn), duration=C.DURATION, offset=C.OFFSET)
        if audio.size == 0:
            continue
        X.append(ser_spectrogram(audio, sr))
        y.append(emotion)
        if augment:
            X.append(ser_spectrogram(add_noise(audio), sr))
            y.append(emotion)
            X.append(ser_spectrogram(time_shift(audio), sr))
            y.append(emotion)
        if (i + 1) % 50 == 0:
            print(f"\r    SER train features: {i+1}/{len(pairs)} files", end="", flush=True)
    print(f"\r    SER train features: {len(pairs)}/{len(pairs)} files", flush=True)

    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int64)
    log.info("SER train tensor %s  class counts %s", X.shape, np.bincount(y, minlength=8).tolist())
    np.savez_compressed(C.CACHE / "ser_train.npz", X=X, y=y)
    return {"n_files": len(pairs), "n_samples": int(len(y)),
            "actors": sorted(actors), "augmented": augment,
            "class_counts": np.bincount(y, minlength=8).tolist()}


def verify_ser_baseline():
    d = np.load(C.CACHE / "ser_test.npz")
    X = d["X"].astype(np.float32) / 255.0
    y = d["y"]
    model, patched = load_keras_compat(C.SER_CHECKPOINT)
    if patched:
        log.info("SER checkpoint loaded via forward-compat config patch")
    prob8 = model.predict(X, verbose=0)
    y4 = labels_8_to_4(y)
    pred4 = four_class_summation(prob8).argmax(1)

    acc8 = float((prob8.argmax(1) == y).mean())
    acc4 = float((pred4 == y4).mean())
    pub = load_json(C.SER_RESULTS_JSON)
    log.info("SER baseline: 8-class %.4f (published %.4f) | 4-class %.4f (published %.4f)",
             acc8, pub["test_8class_accuracy"],
             acc4, pub["test_4class_accuracy_summation_REPORT_THIS"])
    if abs(acc4 - pub["test_4class_accuracy_summation_REPORT_THIS"]) > 0.005:
        log.warning("SER 4-class baseline deviates from the notebook by >0.5 points")
    return {"accuracy_8class": acc8, "accuracy_4class_summation": acc4,
            "published_8class": pub["test_8class_accuracy"],
            "published_4class": pub["test_4class_accuracy_summation_REPORT_THIS"],
            "keras_config_patched": patched,
            "metrics_4class": full_metrics(y4, pred4, C.CLASS4)}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["fer", "ser"], default=["fer", "ser"])
    ap.add_argument("--skip-baseline-check", action="store_true",
                    help="skip reproducing the notebooks' published accuracy")
    ap.add_argument("--rebuild-ser-train", action="store_true",
                    help="rebuild ser_train.npz even if cached")
    ap.add_argument("--no-augment", action="store_true",
                    help="build the SER calibration pool from clean audio only")
    args = ap.parse_args()

    report = {"note": "Stages corrected-notebook artifacts for Group A. No training.",
              "paths": {"fer_data_root": str(C.FER_DIR),
                        "ravdess_root": str(C.RAVDESS_DIR),
                        "outputs": str(C.OUTPUTS)}}

    if "fer" in args.models:
        log.info("--- FER ---")
        for src, dst in ((C.FER_CHECKPOINT, "fer_baseline.pth"),
                         (C.FER_ONNX, "fer_baseline.onnx")):
            if not src.exists():
                log.error("missing %s -- run jugantarFER_corrected.ipynb first", src)
                sys.exit(1)
            shutil.copyfile(src, C.MODELS / dst)
            log.info("staged %s -> %s", src.name, dst)

        split = resolve_fer_paths()
        report["fer_split"] = check_fer_split(split)
        (C.CACHE / "fer_split.json").write_text(json.dumps(
            {"files": split, "data_root": str(C.FER_DIR),
             "source": str(C.FER_SPLIT_JSON)}))
        report["fer_preprocessing_max_diff_vs_torchvision"] = \
            verify_preprocessing(split["test"])
        if not args.skip_baseline_check:
            report["fer_baseline"] = verify_fer_baseline(split)

    if "ser" in args.models:
        log.info("--- SER ---")
        if not C.SER_CHECKPOINT.exists() or not C.SER_TEST_NPZ.exists():
            log.error("missing SER artifacts -- run jugantarSER_corrected.ipynb first")
            sys.exit(1)
        shutil.copyfile(C.SER_CHECKPOINT, C.MODELS / "ser_baseline.keras")
        shutil.copyfile(C.SER_TEST_NPZ, C.CACHE / "ser_test.npz")

        d = np.load(C.CACHE / "ser_test.npz")
        n_test = int(len(d["y"]))
        if n_test != C.EXPECTED_N_TEST["ser"]:
            log.error("SER test set is %d, expected %d -- STOP (rule 1)",
                      n_test, C.EXPECTED_N_TEST["ser"])
            sys.exit(1)
        log.info("SER test set n=%d matches the notebook", n_test)
        report["ser_test"] = {"n_samples": n_test,
                              "class_counts": np.bincount(d["y"], minlength=8).tolist()}

        train_cache = C.CACHE / "ser_train.npz"
        if args.rebuild_ser_train or not train_cache.exists():
            report["ser_train_cache"] = build_ser_train_cache(augment=not args.no_augment)
        else:
            z = np.load(train_cache)
            log.info("reusing cached ser_train.npz (%d samples)", len(z["y"]))
            report["ser_train_cache"] = {"n_samples": int(len(z["y"])), "cached": True}

        if not args.skip_baseline_check:
            report["ser_baseline"] = verify_ser_baseline()

    save_json(report, C.RESULTS / "quant_00_prepare.json")
    log.info("saved -> %s", C.RESULTS / "quant_00_prepare.json")
    log.info("prepare complete -- quant_01_convert.py can now run")


if __name__ == "__main__":
    main()
