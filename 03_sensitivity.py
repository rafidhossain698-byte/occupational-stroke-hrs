"""
03_sensitivity.py
Sensitivity analyses for the occupational-class stroke paper.

  S1. Proportional-hazards fix for hypertension
      (a) Cox model stratified by hypertension (separate baseline hazards)
      (b) Time-varying hypertension effect: piecewise HRs for 0-10, 10-20, >20 years
  S2. Competing risk of death: Fine-Gray subdistribution hazard model
      (IPCW-weighted Cox on an expanded data set; Geskus 2011 approach)
  S3. Competing-risk-aware temporal validation
      - Observed risk by Aalen-Johansen (death as competing event) vs Kaplan-Meier
      - Calibration (O/E) of cause-specific Cox vs Fine-Gray predicted cumulative incidence
      - Discrimination of Fine-Gray M2 vs M3 (paired bootstrap)

Input : results/cohort_development.parquet, results/cohort_temporal_validation.parquet (LOCAL ONLY)
Output: results/sensitivity/  -- aggregate tables only (safe to share)

Usage (Mac), from the same folder as 02_models.py:
  python3 03_sensitivity.py --data results --out results/sensitivity
Runtime: roughly 5-15 minutes.
"""
import argparse
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

warnings.filterwarnings("ignore")
SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))

# reuse variable coding and predictor sets from 02_models.py (single source of truth)
_spec = importlib.util.spec_from_file_location("models02", os.path.join(HERE, "02_models.py"))
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)
SETS, PRETTY, prepare = m02.SETS, m02.PRETTY, m02.prepare

OCC = ["occ_professional", "occ_routine_nonmanual", "occ_service_manual"]
KEY = ["hibpe", "diabe", "hearte", "smoken", "asinh_wealth"] + OCC + ["jstres"]


def impute(dev, val, feats):
    med = dev[feats].median()
    return dev[feats].fillna(med), val[feats].fillna(med)


def tidy(cph, label, keep=None):
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "lo", "hi", "p"]
    s.index = [str(i) for i in s.index]
    if keep is not None:
        s = s.loc[[k for k in keep if k in s.index]]
    s.insert(0, "variable", [PRETTY.get(i, i) for i in s.index])
    s.insert(0, "model", label)
    return s.reset_index(drop=True)


# ----------------------------------------------------------------------------
# S1. Proportional hazards fix for hypertension
# ----------------------------------------------------------------------------
def s1_ph_fix(dev, out, log):
    feats = SETS["M3"]
    X, _ = impute(dev, dev, feats)
    d = X.assign(time=dev["time"].values, stroke=dev["stroke"].astype(int).values)

    main = CoxPHFitter().fit(d, "time", "stroke")
    res = [tidy(main, "Main M3 Cox", KEY)]

    # (a) stratified by hypertension
    ds = d.copy()
    ds["hibpe"] = ds["hibpe"].round().astype(int)
    strat = CoxPHFitter().fit(ds, "time", "stroke", strata=["hibpe"])
    res.append(tidy(strat, "M3 Cox stratified by hypertension", KEY))
    ph = proportional_hazard_test(strat, ds, time_transform="rank").summary
    ph = ph.assign(variable=[PRETTY.get(str(i), str(i)) for i in ph.index])[["variable", "test_statistic", "p"]]
    ph.round(4).to_csv(os.path.join(out, "S1_ph_test_stratified_model.csv"), index=False)
    log(f"  S1a stratified: global PH tests with p<0.05: {(ph.p < 0.05).sum()} of {len(ph)}")

    # (b) piecewise hypertension effect (episode split at 10 and 20 years)
    cuts = [0, 10, 20, np.inf]
    rows = []
    base = d.drop(columns=["hibpe"]).copy()
    base["id"] = np.arange(len(base))
    for k in range(3):
        a, b = cuts[k], cuts[k + 1]
        seg = base[base["time"] > a].copy()
        seg["start"] = a
        seg["stop"] = np.minimum(seg["time"], b)
        seg["stroke"] = ((seg["stroke"] == 1) & (seg["time"] <= b)).astype(int)
        for j in range(3):
            seg[f"hibpe_p{j + 1}"] = d["hibpe"].values[seg["id"]] * (j == k)
        rows.append(seg)
    ep = pd.concat(rows, ignore_index=True).drop(columns=["time"])
    pw = CoxPHFitter().fit(ep, duration_col="stop", event_col="stroke", entry_col="start",
                           cluster_col="id", robust=True)
    pw_keep = ["hibpe_p1", "hibpe_p2", "hibpe_p3"] + OCC
    t = tidy(pw, "M3 Cox, piecewise hypertension effect", pw_keep)
    t["variable"] = t["variable"].replace({"hibpe_p1": "Hypertension, years 0-10",
                                           "hibpe_p2": "Hypertension, years 10-20",
                                           "hibpe_p3": "Hypertension, years >20"})
    res.append(t)
    out_df = pd.concat(res, ignore_index=True).round(4)
    out_df.to_csv(os.path.join(out, "S1_ph_fix_hazard_ratios.csv"), index=False)
    log("  S1 done")
    return out_df


# ----------------------------------------------------------------------------
# S2. Fine-Gray subdistribution hazards via IPCW-weighted Cox
# ----------------------------------------------------------------------------
def finegray_expand(X, time, status, step=0.25):
    """Expanded data for Fine-Gray. Subjects with the competing event (status 2)
    stay in the risk set after their death with weight G(t)/G(T_i), where G is the
    Kaplan-Meier estimate of the censoring distribution (piecewise on a time grid)."""
    km = KaplanMeierFitter().fit(time, event_observed=(status == 0))
    tmax = time.max()
    grid = np.arange(step, tmax + step, step)
    G_grid = np.clip(km.survival_function_at_times(grid).values, 1e-8, None)
    base = X.copy()
    base["id"] = np.arange(len(X))
    base["start"] = 0.0
    base["stop"] = time
    base["event"] = (status == 1).astype(int)
    base["w"] = 1.0
    comp = np.where(status == 2)[0]
    G_T = np.clip(km.survival_function_at_times(time[comp]).values, 1e-8, None)
    ids, starts, stops, ws = [], [], [], []
    for i, Ti, gT in zip(comp, time[comp], G_T):
        k = np.searchsorted(grid, Ti, side="right")
        if k >= len(grid):
            continue
        g = grid[k:]
        w = G_grid[k:] / gT
        keep = w > 1e-3
        n = int(keep.sum())
        if n == 0:
            continue
        st = np.concatenate([[Ti], g[:-1]])[keep]
        ids.append(np.full(n, i)); starts.append(st); stops.append(g[keep]); ws.append(w[keep])
    if not ids:
        return base
    ids = np.concatenate(ids)
    rep = X.iloc[ids].reset_index(drop=True).copy()
    rep["id"], rep["start"], rep["stop"] = ids, np.concatenate(starts), np.concatenate(stops)
    rep["event"], rep["w"] = 0, np.concatenate(ws)
    return pd.concat([base, rep], ignore_index=True)


def fit_finegray(X, time, status):
    ex = finegray_expand(X, time, status)
    fg = CoxPHFitter(penalizer=1e-4).fit(ex, duration_col="stop", event_col="event",
                                         entry_col="start", weights_col="w",
                                         cluster_col="id", robust=True,
                                         fit_options={"step_size": 0.5})
    return fg, ex


def finegray_cif(fg, ex, X_new, h):
    """Predicted cumulative incidence at horizon h from a Fine-Gray fit on expanded data.
    Breslow baseline computed by hand with the correct (left-truncated, weighted) risk sets;
    lifelines' built-in baseline ignores entry times and would count every expanded row
    as at risk from time 0."""
    cols = list(X_new.columns)
    ph_tr = fg.predict_partial_hazard(ex[cols]).values.ravel() * ex["w"].values
    start, stop = ex["start"].values, ex["stop"].values
    ev_t = np.sort(np.unique(stop[(ex["event"].values == 1) & (stop <= h)]))
    ev_all = stop[ex["event"].values == 1]
    H0 = 0.0
    for t in ev_t:
        d = np.sum(ev_all == t)
        at_risk = (start < t) & (stop >= t)
        H0 += d / ph_tr[at_risk].sum()
    ph_new = fg.predict_partial_hazard(X_new).values.ravel()
    return 1 - np.exp(-H0 * ph_new)


def s2_finegray(dev, val, out, log):
    res, models = [], {}
    t, s = dev["time"].values, dev["event_cr"].values
    for sname in ["M2", "M3"]:
        feats = SETS[sname]
        Xd, Xv = impute(dev, val, feats)
        fg, ex = fit_finegray(Xd.reset_index(drop=True), t, s)
        nrows = len(ex)
        models[sname] = (fg, Xd, Xv, ex)
        res.append(tidy(fg, f"Fine-Gray {sname}", KEY))
        log(f"  S2 Fine-Gray {sname}: fitted on {nrows:,} expanded rows")
    out_df = pd.concat(res, ignore_index=True).round(4)
    out_df.to_csv(os.path.join(out, "S2_finegray_subdistribution_HR.csv"), index=False)
    return out_df, models


# ----------------------------------------------------------------------------
# S3. Competing-risk-aware temporal validation
# ----------------------------------------------------------------------------
def aj_risk(time, status, h):
    aj = AalenJohansenFitter(calculate_variance=False).fit(time, status, event_of_interest=1)
    ci = aj.cumulative_density_
    return float(ci[ci.index <= h].iloc[-1, 0]) if (ci.index <= h).any() else 0.0


def km_risk(time, stroke, h):
    km = KaplanMeierFitter().fit(time, event_observed=stroke)
    return float(1 - km.survival_function_at_times(h).values[0])


def s3_validation(dev, val, fg_models, out, log, n_boot=500):
    rows = []
    tv, sv = val["time"].values, val["event_cr"].values
    # cause-specific Cox (death censored), M3
    feats = SETS["M3"]
    Xd, Xv = impute(dev, val, feats)
    cs = CoxPHFitter().fit(Xd.assign(time=dev["time"].values,
                                     stroke=dev["stroke"].astype(int).values), "time", "stroke")
    for h in (10, 20):
        obs_aj, obs_km = aj_risk(tv, sv, h), km_risk(tv, val["stroke"].values, h)
        p_cs = 1 - cs.predict_survival_function(Xv, times=[h]).values.ravel()
        fg, _, Xv_fg, ex = fg_models["M3"]
        p_fg = finegray_cif(fg, ex, Xv_fg, h)
        rows += [
            {"horizon": h, "model": "Cause-specific Cox M3", "mean_predicted": p_cs.mean(),
             "observed_AJ": obs_aj, "observed_KM": obs_km,
             "O_E_vs_AJ": obs_aj / p_cs.mean(), "O_E_vs_KM": obs_km / p_cs.mean()},
            {"horizon": h, "model": "Fine-Gray M3", "mean_predicted": p_fg.mean(),
             "observed_AJ": obs_aj, "observed_KM": obs_km,
             "O_E_vs_AJ": obs_aj / p_fg.mean(), "O_E_vs_KM": obs_km / p_fg.mean()},
        ]
    calib = pd.DataFrame(rows).round(4)
    calib.to_csv(os.path.join(out, "S3_competing_risk_calibration.csv"), index=False)

    # discrimination of Fine-Gray M2 vs M3 in temporal validation (paired bootstrap)
    y_dev = Surv.from_arrays(dev["stroke"].values, dev["time"].values)
    y_val = Surv.from_arrays(val["stroke"].values, np.minimum(tv, dev["time"].max()))
    risk = {k: fg_models[k][0].predict_partial_hazard(fg_models[k][2]).values.ravel()
            for k in ("M2", "M3")}
    point = {k: concordance_index_ipcw(y_dev, y_val, r, tau=20)[0] for k, r in risk.items()}
    rng = np.random.default_rng(SEED)
    diffs, cs2, cs3 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(val), len(val))
        try:
            c2 = concordance_index_ipcw(y_dev, y_val[idx], risk["M2"][idx], tau=20)[0]
            c3 = concordance_index_ipcw(y_dev, y_val[idx], risk["M3"][idx], tau=20)[0]
        except ValueError:
            continue
        cs2.append(c2); cs3.append(c3); diffs.append(c3 - c2)
    disc = pd.DataFrame([
        {"model": "Fine-Gray M2", "C_uno": point["M2"], "lo": np.quantile(cs2, .025), "hi": np.quantile(cs2, .975)},
        {"model": "Fine-Gray M3", "C_uno": point["M3"], "lo": np.quantile(cs3, .025), "hi": np.quantile(cs3, .975)},
        {"model": "Delta M3 - M2", "C_uno": point["M3"] - point["M2"],
         "lo": np.quantile(diffs, .025), "hi": np.quantile(diffs, .975)},
    ]).round(4)
    disc.to_csv(os.path.join(out, "S3_finegray_temporal_discrimination.csv"), index=False)
    log("  S3 done")
    return calib, disc


# ----------------------------------------------------------------------------
def write_summary(s1, s2, calib, disc, out):
    def row(df, model, var):
        r = df[(df.model == model) & (df.variable == PRETTY.get(var, var))]
        return "n/a" if r.empty else f"{r.HR.iloc[0]:.2f} ({r.lo.iloc[0]:.2f}-{r.hi.iloc[0]:.2f}), p={r.p.iloc[0]:.3f}"
    L = ["# Sensitivity analyses summary (aggregate only)\n",
         "## Service/manual vs managerial (stroke)",
         f"- Main cause-specific Cox M3: {row(s1, 'Main M3 Cox', 'occ_service_manual')}",
         f"- Stratified by hypertension: {row(s1, 'M3 Cox stratified by hypertension', 'occ_service_manual')}",
         f"- Piecewise hypertension model: {row(s1, 'M3 Cox, piecewise hypertension effect', 'occ_service_manual')}",
         f"- Fine-Gray M3 (subdistribution HR): {row(s2, 'Fine-Gray M3', 'occ_service_manual')}",
         "\n## Hypertension effect over follow-up (piecewise)"]
    for lab in ["Hypertension, years 0-10", "Hypertension, years 10-20", "Hypertension, years >20"]:
        r = s1[s1.variable == lab]
        if not r.empty:
            L.append(f"- {lab}: HR {r.HR.iloc[0]:.2f} ({r.lo.iloc[0]:.2f}-{r.hi.iloc[0]:.2f})")
    L += ["\n## Competing-risk calibration (temporal validation)", calib.to_string(index=False),
          "\n## Fine-Gray discrimination (temporal validation, Uno's C, tau=20)", disc.to_string(index=False)]
    with open(os.path.join(out, "sensitivity_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="results/sensitivity")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run_log.txt"), "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    dev = prepare(pd.read_parquet(os.path.join(a.data, "cohort_development.parquet"))).reset_index(drop=True)
    val = prepare(pd.read_parquet(os.path.join(a.data, "cohort_temporal_validation.parquet"))).reset_index(drop=True)
    log(f"Development n={len(dev)} | Validation n={len(val)}")
    log("== S1 proportional-hazards fix")
    s1 = s1_ph_fix(dev, a.out, log)
    log("== S2 Fine-Gray")
    s2, fg_models = s2_finegray(dev, val, a.out, log)
    log("== S3 competing-risk-aware validation")
    calib, disc = s3_validation(dev, val, fg_models, a.out, log, a.boot)
    write_summary(s1, s2, calib, disc, a.out)
    log("Done. Share the contents of " + a.out + " (aggregate only).")


if __name__ == "__main__":
    main()
