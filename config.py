"""Central configuration for the Group A quantization study.

Every script imports from here so paths and class orderings are defined exactly once.

Data roots are resolved in this order:
    1. environment variable   (FER_DATA_ROOT / RAVDESS_ROOT)
    2. paths.local.json       (gitignored; machine-specific, see GROUP_A.md)
    3. portable default under ./data

The corrected notebooks are the single source of truth for model weights and
splits. Nothing here re-trains anything -- see GROUP_A.md.
"""
from pathlib import Path
import json
import os

ROOT = Path(os.environ.get("PAPER_FIX_ROOT", Path(__file__).resolve().parent))

# ------------------------------------------------------------------ local paths
_LOCAL = {}
_local_file = ROOT / "paths.local.json"
if _local_file.exists():
    try:
        _LOCAL = json.loads(_local_file.read_text())
    except Exception:
        _LOCAL = {}


def _resolve(env_key, local_key, default):
    v = os.environ.get(env_key) or _LOCAL.get(local_key)
    return Path(v) if v else Path(default)


DATA = _resolve("DATA_ROOT", "data_root", ROOT / "data")

# Artifacts produced by the corrected notebooks (committed in this repo).
OUTPUTS = ROOT / "outputs"
FER_OUT = OUTPUTS / "FER"
SER_OUT = OUTPUTS / "SER"

FER_CHECKPOINT = FER_OUT / "jugantarFER_corrected.pth"
FER_ONNX = FER_OUT / "jugantarFER_corrected.onnx"
FER_SPLIT_JSON = FER_OUT / "fer_split.json"
FER_RESULTS_JSON = FER_OUT / "fer_corrected_results.json"

SER_CHECKPOINT = SER_OUT / "jugantarSER_corrected.keras"
SER_TEST_NPZ = SER_OUT / "ser_test_set.npz"
SER_RESULTS_JSON = SER_OUT / "ser_corrected_results.json"

# Raw datasets. Not committed -- the notebooks download them from Kaggle.
FER_DIR = _resolve("FER_DATA_ROOT", "fer_data_root", DATA / "fer" / "dataset")
RAVDESS_DIR = _resolve("RAVDESS_ROOT", "ravdess_root",
                       DATA / "ravdess" / "audio_speech_actors_01-24")

# ------------------------------------------------------------------- artifacts
ARTIFACTS = Path(os.environ.get("ARTIFACTS_ROOT", ROOT / "artifacts"))
CACHE = ARTIFACTS / "cache"
MODELS = ARTIFACTS / "models"
RESULTS = ARTIFACTS / "results"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
LOGS = ARTIFACTS / "logs"

for _p in (CACHE, MODELS, RESULTS, FIGURES, TABLES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ------------------------------------------------------- SER: speaker split
# RAVDESS actors 1-24; odd = male, even = female.
# Held-out sets are gender balanced so results are not confounded by voice pitch.
SER_TRAIN_ACTORS = list(range(1, 17))          # 1-16  (8M / 8F)
SER_VAL_ACTORS = [17, 18, 19, 20]              #       (2M / 2F)
SER_TEST_ACTORS = [21, 22, 23, 24]             #       (2M / 2F)

# RAVDESS emotion code (filename field 3) minus 1 -> index
EMOTION8 = ["neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"]

# The DEPLOYED 4-class mapping (probability SUMMATION, not max()).
CLASS4 = ["Neutral", "Happy", "Sad", "Angry"]
GROUPS_4 = {0: [0, 1],   # Neutral  = neutral  + calm
            1: [2, 7],   # Happy    = happy    + surprised
            2: [3, 5],   # Sad      = sad      + fearful
            3: [4, 6]}   # Angry    = angry    + disgust
MAP_8_TO_4 = {c8: c4 for c4, group in GROUPS_4.items() for c8 in group}

# Audio feature extraction (must match training and Raspberry Pi inference EXACTLY).
# Do NOT add denoise / trim / volume normalisation here.
DURATION = 3.0
OFFSET = 0.5
N_MELS = 128
IMG_SIZE = 224

# ------------------------------------------------------------------- FER
# torchvision ImageFolder sorts class folders alphabetically.
# NOTE: this ordering is NOT the same as CLASS4 above. Never mix them.
FER_CLASSES = ["Angry", "Happy", "Neutral", "Sad"]
# torchvision transforms.Resize defaults to BILINEAR; PIL's Image.resize defaults
# to BICUBIC. The notebook used torchvision, so BILINEAR is the correct match.
FER_RESIZE_FILTER = "BILINEAR"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frozen test-set sizes from the corrected notebooks. Any deviation means a
# script is measuring a different split -- stop and find out why (GROUP_A rule 1).
EXPECTED_N_TEST = {"fer": 1948, "ser": 240}

# --------------------------------------------------------------- quantization
FORMATS = ["fp32", "fp16", "dynrange", "int8_full_perchannel", "int8_full_pertensor"]
CALIB_SAMPLES = 200
