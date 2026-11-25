# Occupational class and stroke risk prediction (HRS)

Analysis code for the paper *"Does Occupational Class Improve Stroke Risk Prediction? A 30-Year Cohort Study with Temporal Validation of Survival Machine Learning Models."*

The pipeline builds a stroke-free working cohort from the RAND HRS Longitudinal File, compares nested predictor sets (M0–M4) with Cox regression, random survival forests and gradient-boosted survival models, and evaluates them with internal (repeated cross-validation) and temporal validation.

## Data

This repository contains **code only**. It does not include HRS data or any participant-level output.

To reproduce the analysis, register with the Health and Retirement Study (https://hrsdata.isr.umich.edu) and download the **RAND HRS Longitudinal File 2022 (V2)** in SPSS (`.sav`) format. HRS data must be used in accordance with the HRS Conditions of Use, including its policy on AI and large language models.

## Scripts

| Script | Purpose | Approx. runtime |
|---|---|---|
| `01_build_cohort.py` | Development (1992) and temporal-validation (1998) cohorts, outcomes, feasibility report | 2–3 min |
| `02_models.py` | Nested models M0–M4; Cox, RSF, GBS; CV + temporal validation; calibration, decision curves, hazard ratios, cumulative incidence, SHAP | 1–3 h |
| `03_sensitivity.py` | Hypertension PH fix (stratified and piecewise Cox); Fine–Gray competing-risk models; competing-risk-aware calibration | 15–25 min |
| `04_table1_and_tuning.py` | Baseline characteristics (Table 1); hyperparameter tuning of RSF/GBS with temporal evaluation | 30–60 min |

## Usage

```bash
pip install -r requirements.txt
python 01_build_cohort.py --sav randhrs1992_2022v2.sav --out results
python 02_models.py --data results --out results/models
python 03_sensitivity.py --data results --out results/sensitivity
python 04_table1_and_tuning.py --data results --out results/extras
```

`results/cohort_*.parquet` files are participant-level and must stay on your machine. All other outputs are aggregate tables and figures; small cells (<11) are suppressed in the feasibility report and Table 1.

## Acknowledgment

The HRS (Health and Retirement Study) is sponsored by the National Institute on Aging (grant number NIA U01AG009740) and the Social Security Administration and is conducted by the University of Michigan.

## License

MIT (see `LICENSE`).
