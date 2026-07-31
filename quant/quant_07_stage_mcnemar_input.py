"""
QUANT-07  Stage quant_02's cached predictions for mcnemar_compare.py.

quant_02 caches predictions as `preds_<model>_<format>_<mtime>_<size>.npy` and
never writes a labels file, because it already has the labels in memory.
`mcnemar_compare.py` expects `<model>_<format>_probs.npy` plus
`<model>_labels.npy`. This script does that rename, and reconstructs the labels
in the SAME order the predictions were computed in:

  FER  the order of fer_split.json["files"]["test"]  (quant_02.load_fer_test)
  SER  the order stored in ser_test.npz              (quant_02.load_ser_test)

Two views are staged, because SER is reported two ways:

  <out>/raw/       FER 4-class, SER *8-class* (the model's native output)
  <out>/deployed4/ SER folded to 4 classes by SUMMATION -- the deployed protocol
                   and the one the paper publishes, so it is the view the A6
                   per-channel/per-tensor comparison must actually be tested on.

Run mcnemar_compare.py once per directory.

Output: artifacts/mcnemar_input/{raw,deployed4}/
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
from common import get_logger, four_class_summation, labels_8_to_4

log = get_logger("quant_07_stage")


def newest_cache(tag, fmt):
    """The cache key embeds the .tflite mtime and size; take the most recent."""
    hits = sorted(C.CACHE.glob(f"preds_{tag}_{fmt}_*.npy"),
                  key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def fer_labels():
    items = json.loads((C.CACHE / "fer_split.json").read_text())["files"]["test"]
    return np.array([int(lbl) for _, lbl in items], dtype=np.int64)


def ser_labels():
    return np.load(C.CACHE / "ser_test.npz")["y"].astype(np.int64)


def stage(out_root):
    raw = out_root / "raw"
    dep = out_root / "deployed4"
    for d in (raw, dep):
        d.mkdir(parents=True, exist_ok=True)

    manifest = {"views": {}}

    # ---------------------------------------------------------------- FER
    y_fer = fer_labels()
    if len(y_fer) != C.EXPECTED_N_TEST["fer"]:
        log.error("FER labels are %d, expected %d -- STOP", len(y_fer),
                  C.EXPECTED_N_TEST["fer"])
        sys.exit(1)
    np.save(raw / "fer_labels.npy", y_fer)
    staged_fer = []
    for fmt in C.FORMATS:
        src = newest_cache("fer", fmt)
        if src is None:
            log.warning("no cached predictions for fer/%s", fmt)
            continue
        P = np.load(src)
        if len(P) != len(y_fer):
            log.error("fer/%s has %d rows, expected %d -- STOP", fmt, len(P), len(y_fer))
            sys.exit(1)
        np.save(raw / f"fer_{fmt}_probs.npy", P)
        staged_fer.append(fmt)
        log.info("fer/%-22s acc %.4f  <- %s", fmt, (P.argmax(1) == y_fer).mean(), src.name)
    manifest["views"]["fer"] = {"dir": "raw", "n": int(len(y_fer)),
                                "formats": staged_fer, "classes": C.FER_CLASSES}

    # ---------------------------------------------------------------- SER
    y8 = ser_labels()
    if len(y8) != C.EXPECTED_N_TEST["ser"]:
        log.error("SER labels are %d, expected %d -- STOP", len(y8),
                  C.EXPECTED_N_TEST["ser"])
        sys.exit(1)
    y4 = labels_8_to_4(y8)
    np.save(raw / "ser_labels.npy", y8)
    np.save(dep / "ser_labels.npy", y4)

    staged_ser = []
    for fmt in C.FORMATS:
        src = newest_cache("ser", fmt)
        if src is None:
            log.warning("no cached predictions for ser/%s", fmt)
            continue
        P8 = np.load(src)
        if len(P8) != len(y8):
            log.error("ser/%s has %d rows, expected %d -- STOP", fmt, len(P8), len(y8))
            sys.exit(1)
        P4 = four_class_summation(P8)
        np.save(raw / f"ser_{fmt}_probs.npy", P8)
        np.save(dep / f"ser_{fmt}_probs.npy", P4)
        staged_ser.append(fmt)
        log.info("ser/%-22s 8-class %.4f | 4-class %.4f  <- %s", fmt,
                 (P8.argmax(1) == y8).mean(), (P4.argmax(1) == y4).mean(), src.name)
    manifest["views"]["ser_8class"] = {"dir": "raw", "n": int(len(y8)),
                                       "formats": staged_ser, "classes": C.EMOTION8}
    manifest["views"]["ser_4class_summation"] = {
        "dir": "deployed4", "n": int(len(y4)), "formats": staged_ser,
        "classes": C.CLASS4,
        "note": "deployed protocol -- the view the paper publishes"}

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("staged -> %s", out_root)
    return raw, dep


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (C.ARTIFACTS / "mcnemar_input")
    stage(out)
