# AdityaFlareBench: Solar Flare Prediction Benchmark from ISRO Aditya-L1

> **The first open ML benchmark dataset from India's Aditya-L1 solar observatory**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Data: ISRO PRADAN](https://img.shields.io/badge/Source-ISRO%20PRADAN-orange.svg)](https://pradan.issdc.gov.in/al1)

---

## Overview

AdityaFlareBench is a curated benchmark dataset for solar flare prediction using multi-instrument X-ray observations from ISRO's Aditya-L1 mission (launched September 2, 2023). It combines soft X-ray data from **SoLEXS** (1-25 keV) and hard X-ray data from **HEL1OS** (10-150 keV), providing the first publicly available ML-ready dataset from an Indian space observatory.

| Property | Value |
|---|---|
| Mission | Aditya-L1 (ISRO), Sun-Earth L1 point |
| Instruments | SoLEXS (SDD2) + HEL1OS (CdTe1) |
| Energy range | 1-150 keV (soft + hard X-ray) |
| Cadence | 1 second |
| SoLEXS dates | 49 days |
| HEL1OS dates | 30 days |
| Overlap dates | 25 days |
| Detected events | 325 solar flares |
| Classes | 5 (None, B, C, M, X) |
| Features | 12 physics-informed |
| Transfer data | GOES XRS (2,271 labeled windows) |

---

## Dataset Structure

```
adityaflarebench/
  raw/
    solexs/                  # Level-1 FITS lightcurves from PRADAN
      solexs_2024-09-14.lc.gz
      solexs_2024-10-03.lc.gz
      ...
    hel1os/                  # Level-1 FITS lightcurves from PRADAN
      lightcurve_20240914_*.fits
      ...
  processed/
    X_features.npy           # (N, 3600, 12) feature windows
    y_labels.npy             # (N,) class labels [0=None, 1=B, 2=C, 3=M, 4=X]
    lead_times.npy           # (N,) minutes to next flare
    catalog.csv              # Flare event catalog with timestamps + classes
    dates.csv                # Date-to-fold mapping
  splits/
    temporal_fold_0/
      train_dates.txt
      test_dates.txt
    temporal_fold_1/
    temporal_fold_2/
    temporal_fold_3/
    temporal_fold_4/
  goes_transfer/
    X_goes.npy               # (2271, 3600, 9) GOES XRS windows
    y_goes.npy               # (2271,) labels
```

---

## Features (12)

| # | Feature | Formula | Source | Physics |
|---|---|---|---|---|
| 1 | `derivative` | dF/dt | SoLEXS | Rate of energy release |
| 2 | `rolling_max_ratio` | F(t) / max(F, 300s) | SoLEXS | Relative flare intensity |
| 3 | `bg_slope` | Linear slope (300s) | SoLEXS | Pre-flare activity |
| 4 | `energy_integral` | Cumulative F dt | SoLEXS | Total energy deposition |
| 5 | `qpp_power` | FFT 30-300s band | SoLEXS | Quasi-periodic pulsations |
| 6 | `norm_flux` | Z-score | SoLEXS | Anomaly detection |
| 7 | `long_slope` | Linear slope (1800s) | SoLEXS | Long-term buildup |
| 8 | `acceleration` | d^2F/dt^2 | SoLEXS | Impulsive phase onset |
| 9 | `long_ratio` | F / mean(F, 1800s) | SoLEXS | Baseline deviation |
| 10 | `hard_soft_ratio` | HEL1OS / SoLEXS | HEL1OS | Non-thermal/thermal |
| 11 | `neupert` | corr(dSXR/dt, HXR) | HEL1OS | Neupert effect |
| 12 | `spectral_hardness` | HXR(30-40) / HXR(20-30) | HEL1OS | Electron spectrum |

---

## Flare Classes

| Class | GOES Range | Label | Balanced Count |
|---|---|---|---|
| None | < B1.0 | 0 | 400 |
| B | B1.0 - B9.9 | 1 | 400 |
| C | C1.0 - C9.9 | 2 | 400 |
| M | M1.0 - M9.9 | 3 | 363 |
| X | >= X1.0 | 4 | 200 |

---

## Recommended Evaluation Protocol

### Temporal Cross-Validation (Required)

Use **temporal 5-fold CV** with date-level splits (NOT random sample splits):
- Train on 80% of dates, test on 20%
- Ensures no temporal leakage between train/test
- Folds defined in `splits/temporal_fold_*/`

> **WARNING**: Random sample-level splits allow temporal leakage and produce inflated metrics. Always split by date.

### Required Metrics

Report ALL of the following (community standard for solar flare prediction):

| Metric | Description | Reference |
|---|---|---|
| **TSS** | True Skill Statistic (TPR - FPR) | Bloomfield et al. (2012) |
| **HSS** | Heidke Skill Score | Heidke (1926) |
| **AUC** | Area Under ROC Curve (per-class) | — |
| **Brier Score** | Probabilistic calibration | Brier (1950) |
| **Balanced Accuracy** | Mean per-class accuracy | — |

### Binary Evaluation

For operational relevance, report M+X vs rest (dangerous flare detection):
- TPR, FPR, TSS, HSS for the binary M+X task

---

## Baseline Results (JWALASHMI V6.1)

10-model CNN-Attention ensemble with GOES pre-training:

| Metric | Value | 95% CI |
|---|---|---|
| 5-Class Balanced Accuracy | 78.7% | 76.9% - 80.3% |
| M+X TSS | **0.972** | 0.956 - 0.985 |
| M+X HSS | **0.978** | 0.966 - 0.988 |
| M-class AUC | 0.997 | — |
| X-class AUC | 0.999 | — |
| Brier Score | 0.067 | — |
| RED Alert (M+X) Accuracy | 97.3% | 95.8% - 98.7% |

---

## Data Sources

| Source | Access | Format |
|---|---|---|
| Aditya-L1 SoLEXS Level-1 | [PRADAN Portal](https://pradan.issdc.gov.in/al1) | FITS (.lc.gz) |
| Aditya-L1 HEL1OS Level-1 | [PRADAN Portal](https://pradan.issdc.gov.in/al1) | FITS (.fits) |
| GOES XRS (transfer learning) | [NOAA NCEI](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/) | NetCDF |
| GOES Flare Catalog | [NOAA SWPC](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/) | CSV |

---

## Quick Start

```python
import numpy as np

# Load processed data
X = np.load('processed/X_features.npy')   # (N, 3600, 12)
y = np.load('processed/y_labels.npy')     # (N,)

print(f"Samples: {X.shape[0]}, Window: {X.shape[1]}s, Features: {X.shape[2]}")
print(f"Classes: {np.bincount(y)}")  # [None, B, C, M, X]

# Load temporal fold
train_dates = open('splits/temporal_fold_0/train_dates.txt').read().splitlines()
test_dates = open('splits/temporal_fold_0/test_dates.txt').read().splitlines()
```

---

## Citation

```bibtex
@misc{jwalashmi2026,
  title={JWALASHMI: Multi-Instrument Solar Flare Forecasting Using Aditya-L1
         SoLEXS and HEL1OS with Physics-Informed Deep Learning},
  author={Team JWALASHMI},
  year={2026},
  howpublished={GitHub: https://github.com/FrozenLionMax/Jwalashmi},
  note={First ML benchmark dataset from ISRO Aditya-L1 mission}
}
```

---

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Raw Aditya-L1 data is provided by ISRO under their data policy. GOES data is public domain (NOAA).

---

## Acknowledgments

- **Indian Space Research Organisation (ISRO)** for the Aditya-L1 mission and public data release through PRADAN
- **NOAA Space Weather Prediction Center** for GOES XRS data and real-time space weather APIs
- **NASA SDO Team** for Solar Dynamics Observatory imagery
