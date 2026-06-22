# Cover Letter — JWALASHMI

---

## For: Space Weather (AGU)

**Date:** June 22, 2026

**To:** Editor-in-Chief, *Space Weather*
American Geophysical Union

**From:** Team JWALASHMI

**Re:** Submission of manuscript "JWALASHMI: Multi-Instrument Solar Flare Forecasting Using Aditya-L1 SoLEXS and HEL1OS with Physics-Informed Deep Learning"

---

Dear Editor,

We are pleased to submit our manuscript for consideration in *Space Weather*. This work presents **JWALASHMI**, the first machine learning system to utilize observations from ISRO's Aditya-L1 mission for solar flare prediction.

### Why This Paper Matters

1. **Historic first**: No published work has applied machine learning to Aditya-L1 SoLEXS or HEL1OS data. This establishes a new data source for operational space weather forecasting.

2. **Novel physics-informed features**: By fusing soft X-ray (SoLEXS, 1-25 keV) and hard X-ray (HEL1OS, 10-150 keV) observations, we construct 12 features encoding the Neupert effect, spectral hardness evolution, and hard-to-soft X-ray ratio — physics previously unexploited in ML-based flare prediction.

3. **Strong performance**: Our 10-model ensemble achieves M+X TSS of 0.972 (95% CI: 0.956-0.985), M-class AUC of 0.997, and X-class AUC of 0.999, with 97.3% accuracy for detecting dangerous M+X class flares.

4. **Operational deployment**: The system includes a real-time mission control dashboard with live NOAA integration, geomagnetic impact mapping, and 30-satellite tracking — demonstrating a path from research to operational capability.

5. **Timely**: Solar Cycle 25 is near its peak, making robust flare prediction critical. Aditya-L1's unique L1 vantage provides uninterrupted multi-band X-ray coverage unavailable from any other mission.

### Why Space Weather

*Space Weather* is the premier venue for research bridging solar physics and operational forecasting. Our work directly addresses the journal's scope: translating multi-instrument solar observations into actionable predictions for satellite operators, power grid managers, and aviation authorities.

### Data and Reproducibility

All Aditya-L1 data is publicly available through ISRO's PRADAN portal. GOES data is available from NOAA NCEI. Code and trained models are released at https://github.com/FrozenLionMax/Jwalashmi. We also release **AdityaFlareBench**, a curated benchmark dataset for community use.

### Suggested Reviewers

1. **Dr. Monica Bobra** — Stanford University / W.W. Hansen Experimental Physics Laboratory. Expert in ML-based solar flare prediction using SDO/HMI data.

2. **Dr. Naoto Nishizuka** — National Institute of Information and Communications Technology (NICT), Japan. Developer of Deep Flare Net, pioneering multi-wavelength flare forecasting.

3. **Dr. Dibyendu Nandi** — Indian Institute of Science Education and Research (IISER) Kolkata. Expert in solar physics and space weather prediction in the Indian context.

### Declarations

- No conflicts of interest
- All authors contributed to the research and manuscript
- This manuscript has not been submitted elsewhere

We believe this work represents a significant contribution to the space weather community and establishes India's capability for indigenous ML-driven solar flare forecasting.

Sincerely,
**Team JWALASHMI**

---
---

## For: The Astrophysical Journal (ApJ)

**Date:** June 22, 2026

**To:** Editor-in-Chief, *The Astrophysical Journal*
American Astronomical Society

**From:** Team JWALASHMI

**Re:** Submission of manuscript "JWALASHMI: Multi-Instrument Solar Flare Forecasting Using Aditya-L1 SoLEXS and HEL1OS with Physics-Informed Deep Learning"

---

Dear Editor,

We submit our manuscript describing **JWALASHMI**, the first machine learning application of ISRO's Aditya-L1 multi-instrument X-ray observations for solar flare classification and prediction.

### Scientific Novelty

This work makes three astrophysically significant contributions:

1. **First demonstration that Aditya-L1 SoLEXS+HEL1OS data supports ML-based flare prediction** — establishing a new observational foundation for solar physics beyond the GOES/SDO paradigm that has dominated the field.

2. **Quantitative validation that multi-band X-ray physics improves flare classification** — Our physics-informed features (Neupert effect derivative, spectral hardness, hard-to-soft ratio) contribute 19.5% of model attribution despite comprising only 3 of 12 features, confirming that non-thermal/thermal X-ray dynamics encode discriminative flare information.

3. **Demonstration of effective GOES-to-Aditya-L1 transfer learning** — We show that pre-training on decades of GOES XRS data and fine-tuning on 25 days of Aditya-L1 observations achieves M-class AUC of 0.997 (TSS 0.972), suggesting cross-mission spectral transfer is viable for new solar observatories.

### Key Results

- M+X True Skill Statistic: **0.972** (95% CI: 0.956-0.985)
- M-class AUC: **0.997**, X-class AUC: **0.999**
- Brier Score: **0.067**
- Feature analysis confirms physics-informed features outperform data-driven approaches

### Data Availability

All data publicly available (ISRO PRADAN + NOAA NCEI). Code and benchmark dataset released at https://github.com/FrozenLionMax/Jwalashmi.

### Suggested Reviewers

1. **Dr. Monica Bobra** — Stanford University. SDO/HMI flare prediction.
2. **Dr. Naoto Nishizuka** — NICT Japan. Deep Flare Net.
3. **Dr. Dibyendu Nandi** — IISER Kolkata. Indian solar physics.

Sincerely,
**Team JWALASHMI**
