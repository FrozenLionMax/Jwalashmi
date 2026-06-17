"""
Solar Flare Early Warning System — Configuration
All paths, constants, and instrument mappings.
"""
import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"c:\Users\Acer\Desktop\ISRO")

# ── Data Paths ────────────────────────────────────────────────
SOLEXS_RAW = PROJECT_ROOT / "Solexs" / "extracted"
HEL1OS_RAW = PROJECT_ROOT / "Helios" / "extracted"
HEL1OS_ZIPS = PROJECT_ROOT / "Helios"
GOES_DATA = PROJECT_ROOT / "data" / "goes"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── Output Paths ──────────────────────────────────────────────
CATALOG_CSV = PROCESSED / "flare_catalog.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# ── SoLEXS Constants ─────────────────────────────────────────
SOLEXS_CADENCE = 1.0          # seconds
SOLEXS_ENERGY_RANGE = (2, 22) # keV
SOLEXS_DETECTOR = "SDD2"      # SDD1 saturates during solar max
SOLEXS_CHANNELS = 340         # spectral channels in PI files

# SoLEXS SDD2 approximate counts → GOES class mapping
# (derived from data analysis: quiet median~80, C~1000, M~10000, X~25000)
SOLEXS_CLASS_THRESHOLDS = {
    "B": 200,      # counts/sec
    "C": 1000,
    "M": 8000,
    "X": 20000,
}

# ── HEL1OS Constants ─────────────────────────────────────────
HEL1OS_CADENCE = 1.0          # seconds

# CdTe energy bands (soft hard X-ray)
CDTE_BANDS = [
    "5.00KEV_TO_20.00KEV",
    "20.00KEV_TO_30.00KEV",
    "30.00KEV_TO_40.00KEV",
    "40.00KEV_TO_60.00KEV",
    "1.80KEV_TO_90.00KEV",
]

# CZT energy bands (hard X-ray)
CZT_BANDS = [
    "20.00KEV_TO_40.00KEV",
    "40.00KEV_TO_60.00KEV",
    "60.00KEV_TO_80.00KEV",
    "80.00KEV_TO_150.00KEV",
    "18.00KEV_TO_160.00KEV",
]

# HEL1OS CdTe approximate counts → GOES class mapping
HELOS_CLASS_THRESHOLDS = {
    "B": 10,
    "C": 100,
    "M": 1000,
    "X": 5000,
}

# ── Nowcasting Parameters ─────────────────────────────────────
BG_WINDOW = 600           # 10-min rolling median for background
PEAK_SIGMA = 5.0          # detection threshold: bg + N*sigma
MIN_PEAK_DISTANCE = 300   # min 5 min between separate flares
MIN_FLARE_DURATION = 30   # min 30 sec to be a real flare

# ── Tier 2: Tactical Forecasting (30-60 min) ─────────────────
WINDOW_SIZE = 3600        # 60-min input window (seconds)
STRIDE = 300              # 5-min stride for sliding windows
FORECAST_HORIZON = 3600   # predict 60 min ahead (doubled from 30)
N_CLASSES = 5             # [none, B, C, M, X]
CLASS_NAMES = ["None", "B", "C", "M", "X"]
CLASS_WEIGHTS = [1.0, 2.0, 5.0, 30.0, 80.0]  # boosted M/X weights

# ── Tier 1: Strategic Forecasting (5-10 hours) ───────────────
STRATEGIC_HORIZON = 36000   # look 10 hours ahead
STRATEGIC_WINDOW = 21600    # 6-hour input window (seconds)
STRATEGIC_STRIDE = 1800     # 30-min stride
STRATEGIC_DOWNSAMPLE = 60   # downsample to 1-min cadence

# Model architecture (Tier 2 — Tactical)
CNN_CHANNELS = [32, 64, 128]
CNN_KERNELS = [7, 5, 3]
ATTENTION_HEADS = 4
DROPOUT = 0.3
HIDDEN_DIM = 64

# Model architecture (Tier 1 — Strategic)
STRATEGIC_CNN_CHANNELS = [32, 64]
STRATEGIC_CNN_KERNELS = [7, 5]
STRATEGIC_HIDDEN_DIM = 32

# Training
LEARNING_RATE_PRETRAIN = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
BATCH_SIZE = 32
EPOCHS_PRETRAIN = 50      # increased from 20
EPOCHS_FINETUNE = 50
PATIENCE = 15             # increased from 7

# ── Ensure directories exist ──────────────────────────────────
for d in [PROCESSED, MODEL_DIR, PLOTS_DIR, GOES_DATA]:
    d.mkdir(parents=True, exist_ok=True)
