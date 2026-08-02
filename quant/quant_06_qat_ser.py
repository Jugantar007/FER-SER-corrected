"""
QUANT-06  Quantization-aware training for SER (README item A5, option 3).

WHY SER AND NOT FER
    The FER model is PyTorch and TFLite QAT is a TensorFlow tool
    (`tensorflow_model_optimization`); there is no clean PyTorch -> QAT -> TFLite
    path, so FER QAT means rebuilding EfficientNet-B0 in Keras and retraining
    (1-2 weeks). The SER model is already Keras, so tfmot applies directly with
    no rebuild -- 2-3 days instead of 2 weeks. Run this first: it tells you
    whether QAT rescues full INT8 *at all* before committing to the FER rebuild.

WHAT IT PRODUCES
    QAT-INT8 accuracy and size against PTQ-INT8 and dynamic range, on the same
    frozen test set (actors 21-24) the corrected notebook produced.

ARCHITECTURE IS UNCHANGED
    tfmot wraps the existing layers; it does not alter the topology. QAT starts
    from the trained checkpoint's weights (fine-tuning), which is both the
    standard recipe and much cheaper than training from scratch.

REQUIREMENTS
    pip install tensorflow-model-optimization tf-keras
    tfmot targets Keras 2, so this script sets TF_USE_LEGACY_KERAS=1 before
    importing TensorFlow. It must therefore be run as its own process.

Output: artifacts/models/ser_qat_int8.tflite
        artifacts/results/quant_qat_ser.json
        artifacts/figures/quant_qat_ser.png
"""
import argparse
import os
import sys
from pathlib import Path

# tfmot's quantization API is Keras-2 only. This must precede the TF import.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config as C
from common import (get_logger, save_json, load_json, full_metrics, bootstrap_ci,
                    agreement_metrics, run_tflite, save_confusion,
                    four_class_summation, labels_8_to_4)

log = get_logger("quant_06_qat_ser")


def make_bn_quantize_config(tfmot):
    """tfmot's default scheme rejects a standalone BatchNormalization layer.

    It only handles BN fused into a preceding Conv2D. The corrected architecture
    puts the ReLU *inside* Conv2D (`Conv2D(..., activation="relu")`) and BN after
    it, so no BN here is fusable and `quantize_model()` raises. Changing the layer
    order to Conv -> BN -> ReLU would fix that, but it is a different network and
    rule 3 forbids touching the architecture.

    So BN is quantized through an explicit config instead: its output is
    quantized to 8 bits, and gamma/beta stay float -- the same treatment tfmot
    gives other layers that carry no quantizable kernel.
    """
    qk = tfmot.quantization.keras

    class BNQuantizeConfig(qk.QuantizeConfig):
        def get_weights_and_quantizers(self, layer):
            return []

        def set_quantize_weights(self, layer, quantize_weights):
            pass

        def get_activations_and_quantizers(self, layer):
            return []

        def set_quantize_activations(self, layer, quantize_activations):
            pass

        def get_output_quantizers(self, layer):
            return [qk.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

        def get_config(self):
            return {}

    return BNQuantizeConfig


def quantize_ser_model(tfmot, model):
    """Annotate every layer, with the BN exception above, then apply.

    Verified to carry the transferred weights through: the wrapped layers are the
    original layer objects, and the caller re-checks accuracy afterwards.
    """
    qk = tfmot.quantization.keras
    BNQuantizeConfig = make_bn_quantize_config(tfmot)
    from tensorflow.keras.layers import BatchNormalization

    def annotate(layer):
        if isinstance(layer, BatchNormalization):
            return qk.quantize_annotate_layer(layer, BNQuantizeConfig())
        return qk.quantize_annotate_layer(layer)

    annotated = tf.keras.models.clone_model(model, clone_function=annotate)
    with qk.quantize_scope({"BNQuantizeConfig": BNQuantizeConfig}):
        return qk.quantize_apply(annotated)


def require_tfmot():
    try:
        import tensorflow_model_optimization as tfmot
        return tfmot
    except ImportError:
        log.error("tensorflow_model_optimization is not installed.")
        log.error("  pip install tensorflow-model-optimization tf-keras")
        sys.exit(1)


def build_ser_architecture(n_classes=8):
    """The corrected notebook's architecture, rebuilt layer-for-layer.

    Rebuilt rather than loaded because the checkpoint was written by Keras 3 and
    tfmot needs a Keras 2 model. Weights are transferred by position below, and
    the transfer is verified against the original predictions before training.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                         GlobalAveragePooling2D, Dense, Dropout,
                                         BatchNormalization)
    from tensorflow.keras.regularizers import l2
    return Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(n_classes, activation="softmax"),
    ])


def load_baseline_weights(model):
    """Transfer the corrected checkpoint's weights into the Keras-2 rebuild.

    A `.keras` archive stores weights in model.weights.h5 as
    `layers/<layer_name>/vars/<i>`, where i is already the order Keras expects
    (conv: kernel, bias; batchnorm: gamma, beta, moving_mean, moving_variance).
    Layer names match because the rebuild declares the layers in the same order.
    """
    import zipfile
    import h5py

    path = C.MODELS / "ser_baseline.keras"
    tmp = C.CACHE / "ser_baseline_weights.h5"
    with zipfile.ZipFile(path) as z:
        wname = next(n for n in z.namelist() if n.endswith(".weights.h5")
                     or n.endswith(".h5"))
        tmp.write_bytes(z.read(wname))

    transferred, missing = 0, []
    with h5py.File(tmp, "r") as f:
        root = f["layers"] if "layers" in f else f
        for ly in model.layers:
            if not ly.weights:
                continue
            if ly.name not in root or "vars" not in root[ly.name]:
                missing.append(ly.name)
                continue
            g = root[ly.name]["vars"]
            arrays = [np.array(g[k]) for k in sorted(g.keys(), key=int)]
            expected = [tuple(w.shape) for w in ly.weights]
            got = [a.shape for a in arrays]
            if got != expected:
                raise ValueError(
                    f"weight shape mismatch for {ly.name}: stored {got}, "
                    f"model expects {expected}")
            ly.set_weights(arrays)
            transferred += 1

    if missing:
        raise ValueError(f"no stored weights found for layers: {missing}")
    log.info("transferred weights for %d layers", transferred)
    return model


def load_data():
    tr = np.load(C.CACHE / "ser_train.npz")
    te = np.load(C.CACHE / "ser_test.npz")
    return (tr["X"], tr["y"]), (te["X"].astype(np.float32) / 255.0, te["y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="fine-tuning rate; QAT starts from the trained weights")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="carved out of TRAIN actors only -- never val/test actors")
    args = ap.parse_args()

    tfmot = require_tfmot()
    from tensorflow.keras.utils import to_categorical
    from sklearn.utils.class_weight import compute_class_weight

    (Xtr_u8, ytr), (Xte, yte) = load_data()
    log.info("train %s  test %s", Xtr_u8.shape, Xte.shape)
    if len(yte) != C.EXPECTED_N_TEST["ser"]:
        log.error("SER test set is %d, expected %d -- STOP (rule 1)",
                  len(yte), C.EXPECTED_N_TEST["ser"])
        sys.exit(1)

    model = build_ser_architecture()
    load_baseline_weights(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    # Verify the weight transfer reproduces the published baseline before QAT.
    base_prob = model.predict(Xte, verbose=0)
    y4 = labels_8_to_4(yte)
    acc8 = float((base_prob.argmax(1) == yte).mean())
    acc4 = float((four_class_summation(base_prob).argmax(1) == y4).mean())
    published = load_json(C.SER_RESULTS_JSON)
    log.info("rebuilt model: 8-class %.4f (published %.4f) | 4-class %.4f (published %.4f)",
             acc8, published["test_8class_accuracy"],
             acc4, published["test_4class_accuracy_summation_REPORT_THIS"])
    if abs(acc4 - published["test_4class_accuracy_summation_REPORT_THIS"]) > 0.005:
        log.error("weight transfer did not reproduce the baseline -- QAT would start "
                  "from the wrong model. Fix load_baseline_weights before continuing.")
        sys.exit(1)

    # ---- QAT fine-tune ------------------------------------------------
    q_model = quantize_ser_model(tfmot, model)
    q_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                    loss="categorical_crossentropy", metrics=["accuracy"])
    q_model.summary(print_fn=lambda s: log.info("%s", s))

    # The wrapping must not have lost the transferred weights. Fake-quant noise
    # makes this approximate, so it is reported rather than asserted -- but a
    # large drop here means the weights did not survive quantize_apply.
    q_prob = q_model.predict(Xte, verbose=0)
    q_acc4 = float((four_class_summation(q_prob).argmax(1) == y4).mean())
    log.info("after quantize_apply, before fine-tuning: 4-class %.4f "
             "(float baseline %.4f, agreement %.3f)",
             q_acc4, acc4, float((q_prob.argmax(1) == base_prob.argmax(1)).mean()))

    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(len(ytr))
    n_val = int(len(idx) * args.val_frac)
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xa = Xtr_u8[tr_idx].astype(np.float32) / 255.0
    Xv = Xtr_u8[va_idx].astype(np.float32) / 255.0
    ya = to_categorical(ytr[tr_idx], 8)
    yv = to_categorical(ytr[va_idx], 8)

    cw = compute_class_weight("balanced", classes=np.arange(8), y=ytr[tr_idx])
    hist = q_model.fit(
        Xa, ya, validation_data=(Xv, yv), epochs=args.epochs,
        batch_size=args.batch, verbose=2,
        class_weight={i: float(w) for i, w in enumerate(cw)},
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)])

    # ---- convert the QAT model to full INT8 ----------------------------
    conv = tf.lite.TFLiteConverter.from_keras_model(q_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = conv.convert()
    out = C.MODELS / "ser_qat_int8.tflite"
    out.write_bytes(blob)
    log.info("QAT INT8 -> %s (%.3f MB)", out.name, len(blob) / 1e6)

    P = run_tflite(out, Xte, "ser/qat_int8")
    pred = P.argmax(1)
    m8 = full_metrics(yte, pred, C.EMOTION8, y_prob=P)
    m8.update(bootstrap_ci(yte, pred))
    P4 = four_class_summation(P)
    m4 = full_metrics(y4, P4.argmax(1), C.CLASS4, y_prob=P4)
    m4.update(bootstrap_ci(y4, P4.argmax(1)))
    save_confusion(m4["confusion_matrix"], C.CLASS4, "SER 4-class QAT INT8",
                   C.FIGURES / "quant_cm_ser4_qat_int8.png")

    # ---- compare against PTQ from A1 -----------------------------------
    comparison = {}
    fe = C.RESULTS / "quant_format_evaluation.json"
    if fe.exists():
        prev = load_json(fe).get("ser", {})
        for fmt in ("fp32", "dynrange", "int8_full_perchannel", "int8_full_pertensor"):
            if fmt in prev:
                comparison[fmt] = {
                    "accuracy_8class": prev[fmt]["accuracy"],
                    "accuracy_4class": prev[fmt]["four_class_summation"]["accuracy"],
                    "size_mb": prev[fmt]["size_mb"]}
    comparison["qat_int8"] = {"accuracy_8class": m8["accuracy"],
                              "accuracy_4class": m4["accuracy"],
                              "size_mb": round(len(blob) / 1e6, 3)}

    ptq = comparison.get("int8_full_perchannel", {}).get("accuracy_4class")
    verdict = "no PTQ INT8 reference available -- run quant_02 first"
    if ptq is not None:
        d = (m4["accuracy"] - ptq) * 100
        verdict = (f"QAT INT8 reaches {m4['accuracy']*100:.2f}% 4-class vs PTQ INT8 "
                   f"{ptq*100:.2f}% ({d:+.2f} points) against an FP32 baseline of "
                   f"{acc4*100:.2f}%. "
                   + ("QAT recovers full INT8 -- worth rebuilding FER in Keras (option 1)."
                      if d > 2 else
                      "QAT does not meaningfully recover full INT8 here; the FER rebuild "
                      "is probably not worth 1-2 weeks. Report this."))
    log.info("%s", verdict)

    save_json({"model": "ser", "method": "quantization-aware training (tfmot)",
               "epochs_requested": args.epochs,
               "epochs_run": len(hist.history["loss"]),
               "learning_rate": args.lr,
               "architecture": "unchanged from the corrected notebook",
               "qat_init": "fine-tuned from the corrected checkpoint",
               "validation_source": "carved from TRAIN actors only",
               "float_baseline_4class": acc4,
               "qat_int8_8class": m8, "qat_int8_4class": m4,
               "comparison": comparison, "verdict": verdict},
              C.RESULTS / "quant_qat_ser.json")

    labels = [k for k in ("fp32", "dynrange", "int8_full_perchannel",
                          "int8_full_pertensor", "qat_int8") if k in comparison]
    if labels:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        vals = [comparison[k]["accuracy_4class"] * 100 for k in labels]
        ax.bar(range(len(labels)), vals,
               color=["#4d4d4d" if k != "qat_int8" else "#111111" for k in labels])
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
        ax.axhline(acc4 * 100, ls="--", lw=1, color="black")
        ax.set_ylabel("4-class accuracy (%)")
        ax.set_title("SER: QAT vs post-training quantization", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "quant_qat_ser.png", dpi=300)
        plt.close(fig)
    log.info("saved -> %s", C.RESULTS / "quant_qat_ser.json")


if __name__ == "__main__":
    main()
