"""Central configuration: paths, class maps, signal constants.

Everything reads paths from here so you only change them in one place when
you swap mock data for real recordings.
"""
from __future__ import annotations
import os

# ---- Repo-root-relative paths -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

for _d in (DATASET_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- A1: RF signal classifier -------------------------------------------------
# 4-class problem. Index order is the model's output order; keep it stable.
RF_CLASSES = {"noise": 0, "drone": 1, "wifi": 2, "bluetooth": 3}
RF_CLASS_NAMES = [name for name, _ in sorted(RF_CLASSES.items(), key=lambda kv: kv[1])]

# ---- A2: RF fingerprinting (exact drone model) --------------------------------
DRONE_MODELS = {"DJI_Mavic": 0, "DJI_Tello": 1, "Syma_X5C": 2, "Parrot_Anafi": 3}
DRONE_MODEL_NAMES = [n for n, _ in sorted(DRONE_MODELS.items(), key=lambda kv: kv[1])]

# ---- A3 / A5: deep-learning class maps ----------------------------------------
SPECTRO_CLASSES = {"noise": 0, "drone": 1, "wifi": 2, "bluetooth": 3}
ACOUSTIC_CLASSES = {"noise": 0, "drone": 1, "motor": 2}

# ---- A4: vision ---------------------------------------------------------------
VISION_CLASSES = ["drone", "bird", "plane"]

# ---- SDR / signal constants ---------------------------------------------------
SAMPLE_RATE = 2.4e6          # RTL-SDR default 2.4 MS/s
CENTER_FREQ = 2.44e9         # 2.44 GHz (2.4 GHz ISM band)
IQ_SAMPLES_PER_FILE = 65536  # samples written per mock capture

# ---- Model artifact paths -----------------------------------------------------
A1_RF_MODEL = os.path.join(MODELS_DIR, "a1_rf_clf.pkl")
A1_RF_SCALER = os.path.join(MODELS_DIR, "a1_rf_scaler.pkl")
A1_MODEL_META = os.path.join(MODELS_DIR, "a1_meta.json")

A2_FP_MODEL = os.path.join(MODELS_DIR, "a2_fingerprint_svm.pkl")
A2_FP_SCALER = os.path.join(MODELS_DIR, "a2_fingerprint_scaler.pkl")

A3_CNN_MODEL = os.path.join(MODELS_DIR, "a3_spectrogram_cnn.pt")
A5_CNN_MODEL = os.path.join(MODELS_DIR, "a5_acoustic_cnn.pt")
A4_YOLO_MODEL = os.path.join(MODELS_DIR, "a4_best_drone.pt")

A6_ISO_MODEL = os.path.join(MODELS_DIR, "a6_isolation_forest.pkl")

REGISTRY_PATH = os.path.join(DATASET_DIR, "registry.json")
