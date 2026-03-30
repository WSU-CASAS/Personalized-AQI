# Personalizing an Air Quality Score using Mobile Sensing and Machine Learning

This repository contains code for creating a **personalized air quality (AQ) score** using mobile sensing, multi-source environmental data, and machine learning.

We propose a **novel framework** that integrates:
- Continuously collected smartwatch data (location and mobility)
- External air quality data from fixed-site monitors and APIs
- Machine learning models for perception prediction

The system predicts **in-the-moment perceived air quality** and derives a **composite personalized AQ score** using feature attribution techniques. This enables **continuous, context-aware air quality assessment**, improving subjective exposure estimation and supporting individualized risk awareness—especially in regions with limited monitoring infrastructure.

---

## Overview of Pipeline

1. Generate location traces from smartwatch data
2. Derive location-based behavioral features
3. Integrate external air quality data
4. Train machine learning model to predict perceived air quality
5. Compute personalized AQ score using feature attribution

---

## Generate Location Traces

The `trace.py` script generates location traces from smartwatch data.

### Input
- `<input.csv>` containing:
  - `timestamp`
  - `latitude`
  - `longitude`

### Output
- `<input_navtrace.csv>` containing:
  - `timestamp`
  - `latitude`
  - `longitude`

Each location is assumed to remain valid until the next timestamp.

### EMA Integration
The script also processes EMA (Ecological Momentary Assessment) responses to:

> *“How clean does the air seem to you?”*

#### EMA Input
- Fields:
  - `Device`
  - `Response_date`
  - `Response_time`
  - `Response`

#### Output
- `ema_locations.csv` containing:
  - `Device`
  - `Response_timestamp`
  - `Response`
  - `Latitude`
  - `Longitude`

Each EMA response is matched to the user’s location at that time.

---

## Generate Location Markers

The `locmarkers.py` script extracts behavioral features from smartwatch data.

### Key Functionality
- Infers a user’s **home location**
- Computes daily behavioral metrics

### Output
- `<input>_daily.csv` (stored in `daily/` directory)

### Key Feature
- `time_at_home_ratio`:
  Fraction of time spent at home per day
  → Useful for modeling **personal exposure to outdoor air**

---

## Integrate External Air Quality Data

The `aq.py` script retrieves air quality data from external APIs.

### Input
- `ema_locations.csv`

### Output
- `ema_trace_pam_joined.csv`

This file combines:
- EMA responses
- User location
- External air quality metrics at corresponding time/location

---

## Create Personalized Air Quality Score

The `predict.py` script builds and interprets the machine learning model.

### Input
- `ema_trace_pam_joined.csv`

### Features Used
- External air quality data
- Distance to nearest PurpleAir monitor
- Daily behavioral features (e.g., time at home)

### Steps
1. Train and validate model to predict perceived air cleanliness
2. Perform SHAP (SHapley Additive exPlanations) analysis
3. Rank feature contributions
4. Compute weighted sum of features

### Output
- **Personalized Air Quality Score**

This score dynamically reflects:
- Individual exposure
- Environmental conditions
- Personal perception

---

## Key Contributions

- 📱 Mobile sensing via smartwatch data
- 🌍 Integration of heterogeneous environmental data
- 🤖 Machine learning for perception modeling
- 📊 Feature attribution (SHAP) for interpretability
- 🎯 Personalized, dynamic AQ scoring

---

## Acknowledgements

This code was developed by:

- **Dr. Diane Cook** (djcook@wsu.edu)
- FAU LakeO Team
- WSU CASAS Lab: https://casas.wsu.edu

### Funding
- NIH/NIA: R01AG083925, R035AG071451
- NSF/IIS: 1954372
