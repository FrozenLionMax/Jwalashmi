<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/ECharts-5.0-AA344D?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<h1 align="center">JWALASHMI &nbsp;&#9728;&nbsp; ज्वालाश्मि</h1>
<h3 align="center">AI-Powered Solar Flare Early Warning System</h3>
<p align="center"><em>Real-time X-ray flux monitoring and flare prediction using ISRO's Aditya-L1 satellite data</em></p>

---

## Overview

**JWALASHMI** (Sanskrit: *Jwala* — flame, *Rashmi* — ray) is an end-to-end solar flare prediction platform that ingests soft and hard X-ray data from ISRO's **Aditya-L1** mission instruments — **SoLEXS** (1–8 keV) and **HEL1OS** (10–150 keV) — to provide real-time nowcasting and short-term forecasting of solar flare events.

The system classifies flares across the standard **B / C / M / X** scale and provides probabilistic predictions with calibrated uncertainty estimates, enabling early warning for space weather events that can impact satellite operations, power grids, and communication systems.

---

## Features

### Mission Control Dashboard
- **GOES-Standard Flux Chart** — Logarithmic Y-axis (10⁻⁹ to 10⁻² W/m²) with colored B/C/M/X classification bands, real IST timestamps, zoom/pan, and peak annotations
- **Time Window Toggles** — Switch between 1-hour, 6-hour, and 24-hour views with rolling data buffer
- **SMA & Derivative Overlays** — 5-minute Simple Moving Average and dF/dt rate-of-change analysis
- **Dual Instrument View** — Toggle between SoLEXS, HEL1OS, or combined display
- **NOAA Scale Indicators** — R (Radio), S (Solar Radiation), G (Geomagnetic) severity levels
- **Real-time Alert Engine** — GREEN / YELLOW / RED alert states with actionable recommendations
- **Flare Event Log** — Timestamped catalog with class, peak flux, duration, and active region
- **Confusion Matrix Heatmap** — Live model performance visualization
- **Prediction Trend Chart** — Rolling confidence history across last 20 predictions
- **Mobile Responsive** — Full 4→2→1 column grid layout for desktop, tablet, and phone

### ML Pipeline
- **Ensemble Forecasting** — 5-model ensemble with temperature-scaled calibration
- **Physics-Informed Features** — GOES ratio, thermal energy proxy, emission measure derivative, Neupert correlation
- **Multi-Head Architecture** — Simultaneous class prediction + lead-time regression with attention
- **Data Augmentation** — Time warping, amplitude scaling, Gaussian noise injection, temporal shifting
- **Ordinal-Aware Loss** — Combined cross-entropy + Earth Mover's Distance respecting B < C < M < X ordering

### Data Pipeline
- **FITS Loader** — Handles both SoLEXS (`.lc.gz`) and HEL1OS (`lightcurve_*.fits`) formats
- **Nowcasting Detector** — Real-time onset detection using rolling derivative + threshold logic
- **Windowed Feature Engineering** — 60-minute sliding windows at 1-second cadence (3600 × 9 features)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    JWALASHMI Platform                     │
├──────────────┬──────────────────┬────────────────────────┤
│  Data Layer  │  Intelligence    │  Presentation          │
│              │  Layer           │  Layer                 │
│  SoLEXS ──┐ │                  │                        │
│            ├─│─► Feature ──►   │  Mission Control       │
│  HEL1OS ──┘ │   Engine    ▼   │  Dashboard             │
│              │  Ensemble  ──►  │  ├── Flux Chart        │
│  GOES ────── │  Forecaster     │  ├── Alert Engine      │
│  (pretrain)  │  ├── 5 models   │  ├── Prob Bars         │
│              │  ├── Calibrated │  ├── Event Log         │
│              │  └── Confidence │  └── Console           │
│              │       Thresholds│                        │
├──────────────┴──────────────────┴────────────────────────┤
│                  Flask REST API                          │
│                  /api/predict                             │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Jwalashmi/
├── dashboard/
│   └── index.html          # Mission Control UI (single-file SPA)
├── src/
│   ├── data/
│   │   ├── fits_loader.py  # FITS file parser for SoLEXS & HEL1OS
│   │   ├── extract_all.py  # Batch extraction pipeline
│   │   └── goes_downloader.py  # GOES historical data fetcher
│   ├── features/
│   │   ├── physics_features.py  # Physics-informed feature engineering
│   │   └── windowing.py    # Sliding window generator
│   ├── model/
│   │   ├── architecture.py # FlareForecaster (CNN+Attention+MLP)
│   │   ├── ensemble.py     # Ensemble + Temperature Scaling
│   │   ├── augmentation.py # Time-series data augmentation
│   │   ├── train.py        # Training loop with ordinal loss
│   │   └── evaluate.py     # TSS, HSS, reliability diagrams
│   └── nowcasting/
│       └── detector.py     # Real-time onset detection
├── server.py               # Flask API server
├── app.py                  # Application entry point
├── config.py               # Centralized configuration
├── run_pipeline.py         # End-to-end pipeline runner
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

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

Open **http://localhost:5000** in your browser.

> The dashboard runs in simulation mode by default, generating realistic synthetic flux data with proper flare profiles. Connect real Aditya-L1 data by placing FITS files in the `Helios/` and `Solexs/` directories.

### Train the Model

```bash
# Place SoLEXS and HEL1OS FITS files in respective directories
# Then run:
python run_pipeline.py
```

---

## Data Sources

| Instrument | Satellite | Energy Range | Cadence | Format |
|------------|-----------|-------------|---------|--------|
| **SoLEXS** | Aditya-L1 | 1–8 keV | 1 sec | `.lc.gz` |
| **HEL1OS** | Aditya-L1 | 10–150 keV | 1 sec | `.fits` |
| **XRS** | GOES-16/17 | 0.1–0.8 nm | 1 sec | NetCDF (for pre-training) |

---

## Model Details

### Architecture
- **Backbone**: 1D CNN (3 layers, 64→128→256 filters) with batch normalization
- **Attention**: Multi-head self-attention (4 heads) over temporal features
- **Heads**: Classification (5-class softmax) + Lead-time regression (linear)
- **Parameters**: ~850K per model, 5-model ensemble = ~4.25M total

### Training
- **Loss**: 0.7 × CrossEntropy + 0.2 × Earth Mover's Distance + 0.1 × Lead-time MSE
- **Augmentation**: 10× dataset multiplication via noise, shift, scale, warp
- **Calibration**: Post-hoc temperature scaling on validation logits

### Performance Metrics
| Metric | Value |
|--------|-------|
| True Skill Statistic (M+) | ≥0.65 |
| Heidke Skill Score | ≥0.55 |
| Brier Skill Score | ≥0.30 |
| Mean Lead Time | ~25 min |

---

## API Reference

### `GET /api/predict`

Returns the current prediction state.

```json
{
  "tactical": {
    "predicted_class": "M",
    "confidence": 0.82,
    "alert_level": "RED",
    "lead_time_min": 23.4,
    "probabilities": {"None": 0.03, "B": 0.05, "C": 0.08, "M": 0.82, "X": 0.02},
    "uncertainty": 0.04
  },
  "strategic": {
    "class_name": "M",
    "confidence": 0.71
  },
  "flux_solexs": [3.2e-7, 3.4e-7, ...],
  "flux_helios": [8.1e-8, 8.5e-8, ...]
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript, ECharts 5, GSAP |
| **Backend** | Python, Flask |
| **ML** | PyTorch, NumPy, SciPy, scikit-learn |
| **Data** | Astropy (FITS), NetCDF4 |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>JWALASHMI</strong> — <em>The intelligence that reads the Sun's rays to predict its flares</em> &#9728;
</p>
