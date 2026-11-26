# Occupational class and stroke risk prediction (HRS, SHARE, ELSA)

Analysis code for the paper *"Does Occupational Class Improve Stroke Risk Prediction? Equivalence Evidence from Temporal and Cross-National External Validation in Three Ageing Cohorts."* Scripts 01–04 also reproduce the earlier HRS-only conference version.

The pipeline:

1. builds stroke-free working cohorts aged 50–64 from three ageing studies: HRS (USA), SHARE (12 European countries) and ELSA (England)
2. harmonises occupation into four classes from 1980 U.S. Census, ISCO-88 and NS-SEC codes
3. compares nested predictor sets with Cox regression, random survival forests and gradient-boosted survival models
4. develops the models in HRS, validates them temporally in a later HRS cohort, and transports them unchanged to SHARE and ELSA
5. tests whether occupation adds to prediction, using equivalence tests and random-effects meta-analysis

## Data

This repository contains **code only**. It includes no survey data and no participant-level output. Each dataset needs its own registration:

| Study | Where to register | File used |
|---|---|---|
| HRS | https://hrsdata.isr.umich.edu | RAND HRS Longitudinal File 2022 (V2), SPSS `.sav` |
| SHARE | https://share-eric.eu (data access) | Harmonized SHARE Version G (`GH_SHARE_g.dta`), from the Gateway to Global Aging Data |
| ELSA | UK Data Service, SN 5050 | Harmonized ELSA Version H (`gh_elsa_h.dta`) |

Use each dataset under its own conditions of use. This includes the HRS policy on AI and large language models: never upload HRS data, or anything participant-level derived from it, to an AI tool.

## Scripts

| Script | Purpose | Approx. runtime |
|---|---|---|
| `01_build_cohort.py` | HRS development (1992) and temporal-validation (1998) cohorts, outcomes, feasibility report | 2–3 min |
| `02_models.py` | Nested sets M0–M4; Cox, RSF and GBS; cross-validation and temporal validation; calibration, decision curves, hazard ratios, cumulative incidence, SHAP | 1–3 h |
| `03_sensitivity.py` | Hypertension proportional-hazards fix (stratified and piecewise Cox); Fine–Gray models; competing-risk-aware calibration | 15–25 min |
| `04_table1_and_tuning.py` | Baseline characteristics (Table 1); RSF/GBS hyperparameter tuning with temporal evaluation | 30–60 min |
| `05_share_cohort.py` | SHARE cohort (baseline Wave 1, follow-up Waves 2 and 4–9); ISCO-88 mapping; feasibility and missingness reports | 2–5 min |
| `07_elsa_cohort.py` | ELSA cohort (baseline Wave 2, follow-up Waves 3–10); NS-SEC mapping; recoding of negative missing-value codes; feasibility report | 2–5 min |
| `06_external_validation.py` | Part A: models trained on pooled HRS and validated in SHARE or ELSA (common sets C0–C3; discrimination, calibration, recalibration, ΔC). Part B: job-strain and effort–reward items within SHARE or ELSA | 20–40 min per cohort |
| `08_journal_analyses.py` | J1–J10: ΔC with equivalence tests, random-effects meta-analysis, mediation, survey weights, multiple imputation, sex stratification, employees only, external Fine–Gray models, decision curves | 1–2 h |

Script 07 builds the ELSA cohort, which 06 needs for its ELSA run. It is numbered 07 only because it was written after 06.

## Usage

```bash
pip install -r requirements.txt

# HRS
python 01_build_cohort.py --sav randhrs1992_2022v2.sav --out results
python 02_models.py --data results --out results/models
python 03_sensitivity.py --data results --out results/sensitivity
python 04_table1_and_tuning.py --data results --out results/extras

# SHARE and ELSA cohorts
python 05_share_cohort.py --dta share_data/GH_SHARE_g.dta --out results_share
python 07_elsa_cohort.py  --dta elsa_data/gh_elsa_h.dta  --out results_elsa

# External validation
python 06_external_validation.py --hrs results --ext results_share/cohort_share.parquet --name SHARE --out results_external
python 06_external_validation.py --hrs results --ext results_elsa/cohort_elsa.parquet  --name ELSA  --out results_external

# Journal analyses (the raw-file arguments are optional; without them the survey-weight analysis J5 is skipped)
python 08_journal_analyses.py \
    --hrs results --hrs_sav randhrs1992_2022v2.sav \
    --share results_share/cohort_share.parquet --share_dta share_data/GH_SHARE_g.dta \
    --elsa results_elsa/cohort_elsa.parquet --elsa_dta elsa_data/gh_elsa_h.dta \
    --out results_journal
```

On a Mac, prefix long runs with `caffeinate -i` so the machine does not sleep. If a run is interrupted, `08_journal_analyses.py --only J4,J5` restarts selected analyses, and `02_models.py --skip-cv` skips the cross-validation step.

## Outputs and data protection

- **Stays on your machine:** the `cohort_*.parquet` files, which are participant-level.
- **Aggregate, safe to share:** every other output (CSV tables, figures, run logs). Cells with fewer than 11 participants are suppressed in the feasibility reports and Table 1.
- **Git protection:** `.gitignore` blocks `.sav`, `.dta`, `.parquet`, `.csv` and the results folders, so data cannot be committed by accident.

## Acknowledgments

- **HRS:** sponsored by the National Institute on Aging (grant number NIA U01AG009740) and conducted by the University of Michigan.
- **SHARE:** uses data from SHARE Waves 1, 2, 4, 5, 6, 7, 8 and 9, release 9.0.0 (DOIs: 10.6103/SHARE.w1.900, 10.6103/SHARE.w2.900, 10.6103/SHARE.w4.900, 10.6103/SHARE.w5.900, 10.6103/SHARE.w6.900, 10.6103/SHARE.w7.900, 10.6103/SHARE.w8.900, 10.6103/SHARE.w9.900); see Börsch-Supan et al. (2013) for methodological details. The full SHARE funding acknowledgement is given in the paper and at www.share-eric.eu.
- **ELSA:** funded by the National Institute on Aging (R01AG017644) and by UK Government departments coordinated by the National Institute for Health and Care Research.
- **Harmonized files:** Harmonized SHARE and Harmonized ELSA are produced by the Gateway to Global Aging Data, funded by the National Institute on Aging (R01 AG030153).

## License

MIT (see `LICENSE`).
