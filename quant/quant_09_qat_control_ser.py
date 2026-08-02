"""
QUANT-09  The control A5 was missing: is QAT's gain quantization-awareness, or
just 15 more epochs of training?

WHY
    quant_06 compares a QAT model that received 15 epochs of fine-tuning against
    a post-training-quantized model that received none. The +10.42 point gap is
    therefore confounded. This script removes the confound the only way that
    settles it: fine-tune the SAME float model for the SAME 15 epochs with the
    SAME hyper-parameters, split and seed, then post-training quantize THAT.

    Three-way comparison, all on the frozen actor-21-24 test set:
        PTQ(baseline)      -- A5's reference, 59.58% 4-class
        PTQ(fine-tuned)    -- this script: does extra training alone rescue INT8?
        QAT                -- A5's result, 70.00% 4-class

    If PTQ(fine-tuned) stays near PTQ(baseline), the gain is quantization
    awareness. If it climbs toward QAT, A5's headline is partly extra training.

Everything about the training run is copied from quant_06 rather than
reimplemented, so the only difference is that no quantization wrappers are
applied before training.

Output: artifacts/results/quant_qat_control_ser.json
        artifacts/models/ser_ptq_finetuned_int8.tflite
"""
import argparse
import os
import sys
from pathlib import Path

# Match quant_06 exactly: tfmot targets Keras 2 and quant_06 sets this before
# importing TF, so the rebuilt architecture must be built the same way.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

import config as C
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    run_tflite, four_class_summation, labels_8_to_4)
from quant_06_qat_ser import (build_ser_architecture, load_baseline_weights,
                              load_data)

log = get_logger("quant_09_qat_control")


def representative_dataset(n, seed):
    """quant_01's SER representative set, rebuilt here to avoid importing the
    conversion module (which pulls in torch/onnx that this venv does not have)."""
    tr = np.load(C.CACHE / "ser_train.npz")
    X, y = tr["X"], tr["y"]
    rng = np.random.default_rng(seed)
    by_cls = {c: np.flatnonzero(y == c) for c in range(8)}
    per = max(1, n // 8)
    idx = np.concatenate([rng.choice(v, size=min(per, len(v)), replace=False)
                          for v in by_cls.values()])

    def gen():
        for i in idx:
            yield [(X[i:i + 1].astype(np.float32) / 255.0)]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--calib-n", type=int, default=C.CALIB_SAMPLES)
    args = ap.parse_args()

    from tensorflow.keras.utils import to_categorical
    from sklearn.utils.class_weight import compute_class_weight

    (Xtr_u8, ytr), (Xte, yte) = load_data()
    if len(yte) != C.EXPECTED_N_TEST["ser"]:
        log.error("SER test set is %d, expected %d -- STOP (rule 1)",
                  len(yte), C.EXPECTED_N_TEST["ser"])
        sys.exit(1)
    y4 = labels_8_to_4(yte)

    model = build_ser_architecture()
    load_baseline_weights(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    base_prob = model.predict(Xte, verbose=0)
    acc4_before = float((four_class_summation(base_prob).argmax(1) == y4).mean())
    published = load_json(C.SER_RESULTS_JSON)
    if abs(acc4_before - published["test_4class_accuracy_summation_REPORT_THIS"]) > 0.005:
        log.error("weight transfer did not reproduce the baseline -- STOP")
        sys.exit(1)
    log.info("float baseline before fine-tuning: 4-class %.4f", acc4_before)

    # ---- identical split, seed and schedule to quant_06 --------------------
    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(len(ytr))
    n_val = int(len(idx) * args.val_frac)
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xa = Xtr_u8[tr_idx].astype(np.float32) / 255.0
    Xv = Xtr_u8[va_idx].astype(np.float32) / 255.0
    ya = to_categorical(ytr[tr_idx], 8)
    yv = to_categorical(ytr[va_idx], 8)
    cw = compute_class_weight("balanced", classes=np.arange(8), y=ytr[tr_idx])

    hist = model.fit(
        Xa, ya, validation_data=(Xv, yv), epochs=args.epochs,
        batch_size=args.batch, verbose=2,
        class_weight={i: float(w) for i, w in enumerate(cw)},
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)])

    ft_prob = model.predict(Xte, verbose=0)
    acc8_ft = float((ft_prob.argmax(1) == yte).mean())
    acc4_ft = float((four_class_summation(ft_prob).argmax(1) == y4).mean())
    log.info("float AFTER fine-tuning: 8-class %.4f | 4-class %.4f", acc8_ft, acc4_ft)

    # ---- post-training quantize the fine-tuned model ----------------------
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(args.calib_n, C.SEED)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    out = C.MODELS / "ser_ptq_finetuned_int8.tflite"
    out.write_bytes(blob)
    log.info("PTQ(fine-tuned) -> %s (%.3f MB)", out.name, len(blob) / 1e6)

    P = run_tflite(out, Xte, "ser/ptq_finetuned_int8")
    m8 = full_metrics(yte, P.argmax(1), C.EMOTION8, y_prob=P)
    m8.update(bootstrap_ci(yte, P.argmax(1)))
    P4 = four_class_summation(P)
    m4 = full_metrics(y4, P4.argmax(1), C.CLASS4, y_prob=P4)
    m4.update(bootstrap_ci(y4, P4.argmax(1)))
    np.save(C.CACHE / "preds_ser_ptq_finetuned_int8_control.npy", P)

    # ---- the three-way comparison -----------------------------------------
    qat = load_json(C.RESULTS / "quant_qat_ser.json")
    fmt = load_json(C.RESULTS / "quant_format_evaluation.json")["ser"]
    ptq_base4 = fmt["int8_full_perchannel"]["four_class_summation"]["accuracy"]
    qat4 = qat["qat_int8_4class"]["accuracy"]

    from_training = (m4["accuracy"] - ptq_base4) * 100
    from_qat = (qat4 - m4["accuracy"]) * 100
    total = (qat4 - ptq_base4) * 100
    verdict = (
        f"PTQ(baseline) {ptq_base4*100:.2f}% -> PTQ(fine-tuned) {m4['accuracy']*100:.2f}% "
        f"-> QAT {qat4*100:.2f}% (4-class). Of A5's {total:+.2f} point gap, "
        f"{from_training:+.2f} is attributable to the extra 15 epochs and "
        f"{from_qat:+.2f} to quantization awareness itself.")
    log.info("%s", verdict)

    save_json({"model": "ser",
               "purpose": "control for A5's fine-tuning confound",
               "epochs_requested": args.epochs,
               "epochs_run": len(hist.history["loss"]),
               "learning_rate": args.lr,
               "calibration_samples": args.calib_n,
               "float_4class_before_finetuning": acc4_before,
               "float_4class_after_finetuning": acc4_ft,
               "float_8class_after_finetuning": acc8_ft,
               "ptq_finetuned_8class": m8, "ptq_finetuned_4class": m4,
               "ptq_baseline_4class": ptq_base4,
               "qat_4class": qat4,
               "points_from_extra_training": round(from_training, 2),
               "points_from_quantization_awareness": round(from_qat, 2),
               "verdict": verdict},
              C.RESULTS / "quant_qat_control_ser.json")
    log.info("saved -> %s", C.RESULTS / "quant_qat_control_ser.json")


if __name__ == "__main__":
    main()
