<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/ECharts-5.0-AA344D?style=flat-square" />
  <img src="https://img.shields.io/badge/GSAP-3.12-88CE02?style=flat-square&logo=greensock" />
  <img src="https://img.shields.io/badge/Astropy-FITS-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<h1 align="center">☀️ JWALASHMI &nbsp; ज्वालाश्मि</h1>
<h3 align="center">AI-Powered Solar Flare Early Warning System</h3>
<p align="center"><em>Real-time X-ray flux monitoring and flare prediction using ISRO's Aditya-L1 satellite data</em></p>
<p align="center"><strong>SoLEXS (1–8 keV) • HEL1OS (10–150 keV) • 5-Model Ensemble • Temperature-Calibrated</strong></p>

---

## 🌍 Why JWALASHMI?

Solar flares release enormous bursts of electromagnetic radiation that can:
- 🛰️ **Damage satellites** — Degrading solar panels and electronics in orbit
- 📡 **Disrupt communications** — HF radio blackouts lasting hours
- ⚡ **Cause power grid failures** — Geomagnetically induced currents in transformers
- 🧑‍🚀 **Endanger astronauts** — Elevated radiation doses during EVA
- ✈️ **Reroute aviation** — Polar flights diverted due to radiation risk

**JWALASHMI** provides **advance warning** of these events by analyzing real-time X-ray data from India's first solar observatory mission, **Aditya-L1**, positioned at the L1 Lagrange point 1.5 million km from Earth.

---

## Overview

**JWALASHMI** (Sanskrit: *Jwala* — flame, *Rashmi* — ray of light) is an end-to-end solar flare prediction platform that ingests soft and hard X-ray data from ISRO's **Aditya-L1** mission instruments — **SoLEXS** (Solar Low Energy X-ray Spectrometer, 1–8 keV) and **HEL1OS** (High Energy L1 Orbiting X-ray Spectrometer, 10–150 keV) — to provide real-time nowcasting and short-term forecasting of solar flare events.

The system classifies flares across the standard **B / C / M / X** GOES scale and provides probabilistic predictions with calibrated uncertainty estimates, enabling early warning for space weather events.

### Dual-Tier Prediction

| Tier | Horizon | Purpose | Architecture |
|------|---------|---------|-------------|
| **Tactical** | 30–60 min | Imminent flare warning | 5-model CNN+Attention ensemble |
| **Strategic** | 5–10 hours | Activity outlook | Single-model deep forecaster |

---

## ✨ Features

### 🖥️ Mission Control Dashboard
- **GOES-Standard Flux Chart** — Logarithmic Y-axis (10⁻⁹ to 10⁻² W/m²) with colored B/C/M/X classification bands, real IST timestamps, zoom/pan via mouse wheel, and peak annotations with class labels
- **Time Window Toggles** — Switch between 1-hour, 6-hour, and 24-hour views with a rolling 86,400-point circular buffer
- **SMA & Derivative Overlays** — 5-minute Simple Moving Average and dF/dt rate-of-change analysis for trend detection
- **Attention Heatmap Overlay** — Toggle to visualize WHERE the model focuses in the time series (Gaussian-weighted around detected peaks with derivative-boosted regions)
- **Dual Instrument View** — Toggle between SoLEXS only, HEL1OS only, or combined display
- **Feature Importance Chart** — Real-time gradient-attribution bar chart showing which of the 9 physics features most influenced the current prediction
- **NOAA Scale Indicators** — R (Radio), S (Solar Radiation), G (Geomagnetic) storm severity levels
- **Real-time Alert Engine** — THREE-TIER alert system:
  - 🟢 **GREEN** — All clear, routine monitoring
  - 🟡 **YELLOW** — Watch, elevated activity detected
  - 🔴 **RED** — Warning, high-probability flare imminent
- **SDO Live Solar Images** — Real-time NASA Solar Dynamics Observatory images cycling through AIA 193Å (corona), 171Å (loops), 304Å (chromosphere), and HMI (photosphere) every 30 seconds
- **Flare Event Log** — Timestamped catalog with class, peak flux (W/m²), duration, active region, and model confidence
- **Confusion Matrix Heatmap** — Interactive ECharts heatmap of model performance with per-class click details
- **Prediction Trend Chart** — Rolling confidence history across last 20 predictions
- **Probability Distribution Bars** — Real-time B/C/M/X probability bars with dominant class highlighting
- **System Console** — Filterable log (All/OK/Info/Warn/Err) showing every prediction cycle with latency, confidence, and alert level
- **Loading Skeleton** — Professional shimmer-animated skeleton screens while data initializes
- **Mobile Responsive** — Full 4→2→1 column grid layout for desktop, tablet, and phone
- **Keyboard Shortcuts** — `1` Combined, `2` SoLEXS, `3` HEL1OS, `D` Derivative, `A` Attention, `S` SMA

### 🧠 ML Pipeline
- **Ensemble Forecasting** — 5-model ensemble with temperature-scaled calibration for reliable probability estimates
- **Physics-Informed Features** — 9 engineered features per timestep:

  | # | Feature | Description |
  |---|---------|-------------|
  | 1 | SoLEXS Flux | Soft X-ray intensity (1–8 keV) |
  | 2 | HEL1OS Flux | Hard X-ray intensity (10–150 keV) |
  | 3 | dF/dt | Flux derivative (rate of change) |
  | 4 | Temperature | Estimated coronal temperature |
  | 5 | Emission Measure | Thermal plasma density proxy |
  | 6 | QPP Index | Quasi-Periodic Pulsation indicator |
  | 7 | Normalized Rate | Background-subtracted count rate |
  | 8 | Slope | Windowed linear trend coefficient |
  | 9 | Acceleration | Second derivative of flux |

- **Multi-Head Architecture** — Simultaneous class prediction (5-class softmax) + lead-time regression (minutes to flare) with temporal attention
- **Data Augmentation** — 10× dataset multiplication via:
  - Gaussian noise injection (σ = 5% of feature std)
  - Temporal shifting (±300 samples)
  - Amplitude scaling (0.8× to 1.2×)
  - Smooth time warping (σ = 0.2)
- **Ordinal-Aware Loss** — Combined: `0.7×CE + 0.2×EMD + 0.1×MSE` respecting B < C < M < X ordering
- **Confidence Thresholds** — Only alert when max probability exceeds 60%, reducing false alarm rate

### 📊 Data Pipeline
- **FITS Loader** — Handles both SoLEXS (`.lc.gz` gzipped light curves) and HEL1OS (`lightcurve_*.fits`) formats using Astropy
- **Nowcasting Detector** — Real-time flare onset detection using:
  - Rolling median background estimation (3600-sample window)
  - Dynamic σ-threshold peak detection
  - Automatic start/stop boundary finding
  - Class estimation from net counts above background
- **Unified Catalog Builder** — Cross-matches detections from both instruments within a 5-minute window
- **Windowed Feature Engineering** — 60-minute sliding windows at 1-second cadence (3600 × 9 feature tensors)

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     JWALASHMI Platform v2.2                    │
├────────────────┬───────────────────┬──────────────────────────┤
│   Data Layer   │  Intelligence     │  Presentation Layer      │
│                │  Layer            │                          │
│  ┌──────────┐  │  ┌─────────────┐  │  ┌────────────────────┐  │
│  │ SoLEXS   │──│─►│ 9-Feature   │  │  │ Mission Control    │  │
│  │ 1-8 keV  │  │  │ Physics     │  │  │ Dashboard          │  │
│  └──────────┘  │  │ Engine      │  │  │                    │  │
│  ┌──────────┐  │  └──────┬──────┘  │  │ ├─ Flux Chart      │  │
│  │ HEL1OS   │──│─────────┘         │  │ ├─ Alert Engine    │  │
│  │ 10-150keV│  │  ┌─────────────┐  │  │ ├─ Attention Map   │  │
│  └──────────┘  │  │ Ensemble    │──│─►│ ├─ Feature Imp.    │  │
│  ┌──────────┐  │  │ Forecaster  │  │  │ ├─ Event Log       │  │
│  │ GOES XRS │  │  │ 5×CNN+Attn  │  │  │ ├─ SDO Live       │  │
│  │ pretrain  │──│─►│ T-calibrated│  │  │ ├─ Confusion Mat.  │  │
│  └──────────┘  │  │ Conf>0.60   │  │  │ └─ Console        │  │
│                │  └─────────────┘  │  └────────────────────┘  │
├────────────────┴───────────────────┴──────────────────────────┤
│                    Flask REST API Server                       │
│   /api/predict  /api/catalog  /api/metrics  /api/health       │
└───────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Jwalashmi/
├── dashboard/
│   └── index.html              # Mission Control UI (single-file SPA, ~65KB)
├── src/
│   ├── data/
│   │   ├── fits_loader.py      # FITS file parser for SoLEXS & HEL1OS
│   │   ├── extract_all.py      # Batch extraction pipeline
│   │   └── goes_downloader.py  # GOES-16/17 XRS historical data fetcher
│   ├── features/
│   │   ├── physics_features.py # 9-feature physics engineering
│   │   └── windowing.py        # 60-min sliding window generator
│   ├── model/
│   │   ├── architecture.py     # FlareForecaster (CNN+Attention+MLP)
│   │   ├── ensemble.py         # 5-model Ensemble + Temperature Scaling
│   │   ├── augmentation.py     # 4-type time-series augmentation (10×)
│   │   ├── train.py            # Training loop with ordinal-aware loss
│   │   └── evaluate.py         # TSS, HSS, Brier, reliability diagrams
│   └── nowcasting/
│       └── detector.py         # Real-time flare onset detection
├── server.py                   # Flask API server (5 endpoints)
├── app.py                      # Application entry point
├── config.py                   # Centralized configuration & thresholds
├── run_pipeline.py             # End-to-end pipeline runner (CLI)
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip
- ~2GB RAM for model inference

### Installation

```bash
git clone https://github.com/FrozenLionMax/Jwalashmi.git
cd Jwalashmi
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python server.py
```

```
============================================================
  JWALASHMI — Solar Flare Early Warning System
  AI-Powered Mission Control Dashboard
============================================================

Loading models...
Dashboard: http://localhost:5000
API:       http://localhost:5000/api/predict
Health:    http://localhost:5000/api/health
```

Open **http://localhost:5000** in your browser.

> **Note:** The dashboard runs in simulation mode by default, generating realistic synthetic flux data with proper flare profiles (quadratic rise + exponential decay + quiet-sun baseline noise). Connect real Aditya-L1 data by placing FITS files in the `Helios/` and `Solexs/` directories.

### Train the Model

```bash
# Full pipeline: ensemble + both tiers + augmentation
python run_pipeline.py

# Quick mode: single model, no augmentation
python run_pipeline.py --quick

# Only tactical tier (30-60 min prediction)
python run_pipeline.py --tactical

# Only strategic tier (5-10 hour outlook)
python run_pipeline.py --strategic
```

---

## 📡 Data Sources

| Instrument | Satellite | Energy Range | Cadence | Format | Purpose |
|------------|-----------|-------------|---------|--------|---------|
| **SoLEXS** | Aditya-L1 (ISRO) | 1–8 keV | 1 sec | `.lc.gz` | Soft X-ray flux (primary) |
| **HEL1OS** | Aditya-L1 (ISRO) | 10–150 keV | 1 sec | `.fits` | Hard X-ray flux (impulsive phase) |
| **XRS** | GOES-16/17 (NOAA) | 0.1–0.8 nm | 1 sec | NetCDF | Pre-training labels (50 years) |
| **AIA/HMI** | SDO (NASA) | Multi-λ | 12 sec | JPEG | Live solar disk images |

### GOES Flare Classification Scale

| Class | Peak Flux (W/m²) | Frequency | Impact |
|-------|-----------------|-----------|--------|
| **A** | < 10⁻⁷ | Background | None |
| **B** | 10⁻⁷ – 10⁻⁶ | ~100/day | Minimal |
| **C** | 10⁻⁶ – 10⁻⁵ | ~10/day | Minor radio |
| **M** | 10⁻⁵ – 10⁻⁴ | ~1/day | HF blackout |
| **X** | > 10⁻⁴ | ~10/year | Major disruption |

---

## 🧠 Model Details

### Architecture: FlareForecaster

```
Input: (batch, 3600, 9) — 60 min @ 1s, 9 features
  │
  ├─► Conv1D(9→64, k=7) + BN + ReLU + MaxPool(4)
  ├─► Conv1D(64→128, k=5) + BN + ReLU + MaxPool(4)
  ├─► Conv1D(128→256, k=3) + BN + ReLU + AdaptiveAvgPool(32)
  │
  ├─► MultiHeadAttention(256, 4 heads) + LayerNorm
  ├─► GlobalAveragePooling
  │
  ├─► Head 1: Linear(256→128→5) → Class logits (None/B/C/M/X)
  ├─► Head 2: Linear(256→64→1) → Lead time (minutes)
  └─► Output: Attention weights (for visualization)
```

- **Parameters**: ~850K per model
- **Ensemble**: 5 models × 850K = **4.25M total parameters**
- **Inference**: <50ms per prediction on CPU

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | 0.7×CE + 0.2×EMD + 0.1×Lead-MSE |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Epochs | 50 per fold |
| Batch size | 32 |
| Augmentation | 10× (noise + shift + scale + warp) |
| Calibration | Temperature scaling (T≈2.04) |

### Performance Metrics

| Metric | Tactical | Strategic |
|--------|----------|-----------|
| Accuracy | 53.6% | 84.9% |
| Weighted F1 | 71.6% | — |
| M+X Recall | 100% | 99.3% |
| TSS (M+) | ≥0.65 | — |
| HSS | ≥0.55 | — |
| Mean Lead Time | 32.9 min | — |
| FPR | 41.7% | 45.6% |
| X-class AUC | 1.000 | — |
| M-class AUC | 0.997 | — |

---

## 🔌 API Reference

### `GET /api/predict`
Returns the current prediction state with flux data.

```json
{
  "timestamp": "2026-06-17T18:00:00Z",
  "tactical": {
    "predicted_class": "M",
    "confidence": 0.82,
    "alert_level": "RED",
    "lead_time_min": 23.4,
    "probabilities": {"None": 0.03, "B": 0.05, "C": 0.08, "M": 0.82, "X": 0.02},
    "uncertainty": 0.04,
    "true_class": "M"
  },
  "strategic": {
    "class_name": "M",
    "confidence": 0.71,
    "probabilities": [0.02, 0.05, 0.12, 0.71, 0.10]
  },
  "flux_solexs": [3.2e-7, 3.4e-7, "... 3600 points at 1s cadence"],
  "flux_helios": [8.1e-8, 8.5e-8, "... 3600 points"],
  "system": {
    "ensemble_models": 5,
    "temperature": 2.04
  }
}
```

### `GET /api/catalog`
Returns detected flare events from SoLEXS data.

```json
{
  "events": [
    {
      "date": "2024-03-15",
      "peak_time": "2024-03-15T14:23:45+00:00",
      "class": "M",
      "peak_counts": 8542.3,
      "background": 120.5,
      "duration_sec": 1240.0,
      "confidence": 0.87
    }
  ],
  "total": 47,
  "days_analyzed": 25,
  "source": "solexs"
}
```

### `GET /api/metrics`
Returns model performance metrics.

### `GET /api/feature_importance`
Returns gradient-based feature attribution scores.

### `GET /api/health`
Returns system health and model loading status.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5, CSS3, JavaScript | Single-page application |
| **Charts** | ECharts 5.5 | Flux chart, confusion matrix, trend |
| **Animation** | GSAP 3.12 | Smooth UI transitions |
| **Fonts** | Space Grotesk, JetBrains Mono | Typography |
| **Backend** | Python 3.10+, Flask | REST API server |
| **ML** | PyTorch 2.0+ | Model training & inference |
| **Science** | NumPy, SciPy, Pandas | Numerical computing |
| **Evaluation** | scikit-learn | Metrics & cross-validation |
| **Data** | Astropy | FITS file parsing |
| **Images** | NASA SDO API | Real-time solar disk images |

---

## 📚 Scientific Background

### Solar Flare Physics
Solar flares are sudden brightenings on the Sun caused by magnetic reconnection in the corona. They release energy across the electromagnetic spectrum, with X-ray emission being the primary diagnostic tool.

**Key observables used by JWALASHMI:**
- **Soft X-ray flux** (SoLEXS 1–8 keV) — Thermal emission from heated plasma, defines flare class
- **Hard X-ray flux** (HEL1OS 10–150 keV) — Non-thermal bremsstrahlung from accelerated electrons, marks impulsive phase
- **GOES ratio** (hard/soft) — Indicator of spectral hardness, correlates with particle acceleration
- **Neupert effect** — Hard X-ray time integral correlates with soft X-ray peak, used for lead-time estimation

### Aditya-L1 Mission
- **Launch**: September 2, 2023, by ISRO
- **Orbit**: Halo orbit around Sun-Earth L1 Lagrange point
- **Distance**: ~1.5 million km from Earth
- **Advantage**: Uninterrupted solar observation without eclipses

---

## 🗺️ Roadmap

- [x] GOES-standard flux chart with log scale
- [x] Dual-instrument support (SoLEXS + HEL1OS)
- [x] 5-model ensemble with temperature calibration
- [x] Attention heatmap visualization
- [x] Feature importance attribution
- [x] Live NASA SDO solar imagery
- [x] Mobile-responsive mission control layout
- [ ] GOES 50-year pre-training pipeline
- [ ] Reliability diagram (calibration curves)
- [ ] PDF space weather bulletin export
- [ ] Gunicorn/Nginx production deployment
- [ ] WebSocket real-time streaming

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>JWALASHMI</strong> — <em>ज्वालाश्मि — The intelligence that reads the Sun's rays to predict its flares</em> ☀️
</p>
<p align="center">
  Built with ❤️ for space weather safety
</p>
