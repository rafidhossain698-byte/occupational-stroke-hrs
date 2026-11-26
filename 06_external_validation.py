"""
06_external_validation.py
  Part A. External validation: models trained on HRS (1992 + 1998 cohorts pooled) are
          applied unchanged to an external cohort (SHARE now; ELSA later, same file format).
          Common predictor sets (variables available in both studies):
            C0 demographics : age, female
            C1 + clinical   : hypertension, diabetes, heart disease, BMI, ever/current smoking
            C2 + SES        : education (3-level harmonized ISCED), income and wealth
                              percentiles, married
            C3 + occupation : occupational class (3 dummies), weekly hours, job tenure
          Primary comparison: C3 vs C2 (does occupation add value in a new population?).
          Metrics: Uno's C (tau=15 y), AUC at 10/15 y, O/E and calibration slope at 10 y,
                   paired bootstrap for C3-C2, C-index by country.
  Part B. Within-SHARE analysis of richer work measures not available in HRS:
          job-strain / effort-reward items and supervisory responsibility, added on top of C3
          (+ country). Repeated stratified 5-fold CV, paired fold-wise differences, and
          country-stratified Cox hazard ratios.

Input : results/cohort_development.parquet, results/cohort_temporal_validation.parquet (HRS)
        results_share/cohort_share.parquet (SHARE)              -- all LOCAL ONLY
Output: results_external/  -- aggregate tables only (safe to share)

Usage (Mac):
  python3 06_external_validation.py --hrs results --ext results_share/cohort_share.parquet \
          --name SHARE --out results_external
Runtime: roughly 20-40 minutes.
"""
import argparse
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore")
SEED = 2026
TAU = 15
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("m02", os.path.join(HERE, "02_models.py"))
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)
Preproc = m02.Preproc

OCC = ["occ_professional", "occ_routine_nonmanual", "occ_service_manual"]
C0 = ["age", "female"]
C1 = C0 + ["hibpe", "diabe", "hearte", "bmi", "smokev", "smoken"]
C2 = C1 + ["educ3", "income_pct", "wealth_pct", "married"]
C3 = C2 + OCC + ["jhours", "jcten"]
CSETS = {"C0": C0, "C1": C1, "C2": C2, "C3": C3}
STRAIN = ["timepre", "ltlfree", "demand", "deserve", "salary", "promot", "secure",
          "support", "nskills", "satjob", "jsprvs"]


# ----------------------------------------------------------------------------
def harmonize(df, income_col, wealth_col, group_col=None):
    """Common coding for HRS and external cohorts (RAND-style source variables)."""
    d = pd.DataFrame(index=df.index)
    d["time"] = df["time_years"].astype(float)
    d["stroke"] = df["event"] == 1
    d["event_cr"] = df["event"].astype(int)
    d["occ_group"] = df["occ_hierarchy"]
    d["age"] = df["age"]
    d["female"] = (df["sex"] == 2).astype(float)
    for v in ["hibpe", "diabe", "hearte", "smokev", "smoken"]:
        d[v] = (df[v] == 1).astype(float).where(df[v].isin([0, 1]))
    d["bmi"] = df["bmi"].where(df["bmi"].between(12, 70))
    if "educl" in df:            # Gateway harmonized 3-level education (SHARE, ELSA)
        d["educ3"] = df["educl"].where(df["educl"].between(1, 3))
    else:                        # RAND HRS raeduc: 1 <HS, 2 GED, 3 HS, 4 some college, 5 college+
        d["educ3"] = df["educ"].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3})
    d["married"] = df["mstat"].isin([1, 2, 3]).astype(float).where(df["mstat"].notna())
    for i, v in enumerate(OCC):
        d[v] = (df["occ_hierarchy"].str[0] == str(i + 2)).astype(float)
    d["jhours"] = df["jhours"].where(df["jhours"].between(1, 112))
    d["jcten"] = df["jcten"].where(df["jcten"] >= 0)
    g = df[group_col] if group_col else pd.Series(0, index=df.index)
    d["income_pct"] = df[income_col].groupby(g).rank(pct=True)
    d["wealth_pct"] = df[wealth_col].groupby(g).rank(pct=True)
    return d


def build_model(name):
    if name == "Cox":
        return CoxPHSurvivalAnalysis(alpha=1.0, ties="efron")
    if name == "RSF":   # settings selected by tuning in 04_table1_and_tuning.py
        return RandomSurvivalForest(n_estimators=300, min_samples_leaf=80, max_features="sqrt",
                                    n_jobs=-1, random_state=SEED, low_memory=False)
    return GradientBoostingSurvivalAnalysis(n_estimators=250, learning_rate=0.1, max_depth=2,
                                            min_samples_leaf=40, subsample=0.5, random_state=SEED)


def ysurv(d):
    return Surv.from_arrays(d["stroke"].values, d["time"].values, "stroke", "time")


def risk_at(model, X, h):
    fns = model.predict_survival_function(X)
    return np.array([1 - f(h) for f in fns])


def obs_aj(d, h):
    aj = AalenJohansenFitter(calculate_variance=False).fit(d["time"], d["event_cr"], event_of_interest=1)
    ci = aj.cumulative_density_
    return float(ci[ci.index <= h].iloc[-1, 0])


def obs_km(d, h):
    return float(1 - KaplanMeierFitter().fit(d["time"], d["stroke"]).survival_function_at_times(h).values[0])


def cal_slope(pred, d, h=10):
    """Calibration slope: Cox regression of outcome on log(-log(1-pred)) (1 = ideal), and
    recalibrated mean 10-y risk from that model (baseline hazard re-estimated in the new cohort)."""
    lp = np.log(-np.log(np.clip(1 - pred, 1e-9, 1 - 1e-9)))
    x = pd.DataFrame({"lp": lp, "time": d["time"].values, "stroke": d["stroke"].astype(int).values})
    cph = CoxPHFitter().fit(x, "time", "stroke")
    recal = 1 - cph.predict_survival_function(x[["lp"]], times=[h]).values.ravel()
    return float(cph.params_["lp"]), float(recal.mean())


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def part_a(hrs, ext, name, out, n_boot, log):
    y_tr = ysurv(hrs)
    ext = ext.copy()
    ext.loc[ext["time"] > y_tr["time"].max(), "time"] = y_tr["time"].max()
    y_te = ysurv(ext)
    rows, risks, preds10 = [], {}, {}
    for sname, feats in CSETS.items():
        pp = Preproc().fit(hrs[feats])
        Xtr, Xte = pp.transform(hrs[feats]), pp.transform(ext[feats])
        for mname in ("Cox", "RSF", "GBS"):
            m = build_model(mname).fit(Xtr, y_tr)
            r = m.predict(Xte)
            p10 = risk_at(m, Xte, 10)
            risks[(sname, mname)], preds10[(sname, mname)] = r, p10
            row = {"set": sname, "model": mname,
                   "C_uno": concordance_index_ipcw(y_tr, y_te, r, tau=TAU)[0]}
            for h in (10, 15):
                try:
                    row[f"AUC_{h}y"] = cumulative_dynamic_auc(y_tr, y_te, r, [h])[0][0]
                except ValueError:
                    row[f"AUC_{h}y"] = np.nan
            o_aj, o_km = obs_aj(ext, 10), obs_km(ext, 10)
            row.update({"pred_10y": p10.mean(), "obs_10y_AJ": o_aj, "obs_10y_KM": o_km,
                        "OE_AJ": o_aj / p10.mean(), "OE_KM": o_km / p10.mean(),
                        })
            slope, recal_mean = cal_slope(p10, ext)
            row.update({"cal_slope_10y": slope, "pred_10y_recal": recal_mean,
                        "OE_AJ_recal": o_aj / recal_mean})
            rows.append(row)
            log(f"  [A] {name} {sname}-{mname}: C={row['C_uno']:.3f}, O/E(AJ)={row['OE_AJ']:.2f}")
    perf = pd.DataFrame(rows)

    # bootstrap CIs for C and paired C3-C2
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(ext), len(ext)) for _ in range(n_boot)]
    ci_rows, delta_rows = [], []
    for mname in ("Cox", "RSF", "GBS"):
        bc = {s: [] for s in CSETS}
        dd = []
        for idx in boots:
            try:
                cs = {s: concordance_index_ipcw(y_tr, y_te[idx], risks[(s, mname)][idx], tau=TAU)[0] for s in CSETS}
            except ValueError:
                continue
            for s in CSETS:
                bc[s].append(cs[s])
            dd.append(cs["C3"] - cs["C2"])
        for s in CSETS:
            ci_rows.append({"set": s, "model": mname, "C_lo": np.quantile(bc[s], .025), "C_hi": np.quantile(bc[s], .975)})
        point = perf[(perf.model == mname) & (perf.set == "C3")].C_uno.iloc[0] - \
            perf[(perf.model == mname) & (perf.set == "C2")].C_uno.iloc[0]
        delta_rows.append({"model": mname, "delta_C_C3_minus_C2": point,
                           "lo": np.quantile(dd, .025), "hi": np.quantile(dd, .975)})
    perf = perf.merge(pd.DataFrame(ci_rows), on=["set", "model"])
    perf.round(4).to_csv(os.path.join(out, f"A_external_{name}_performance.csv"), index=False)
    pd.DataFrame(delta_rows).round(4).to_csv(os.path.join(out, f"A_external_{name}_C3_vs_C2.csv"), index=False)

    # discrimination by country (Cox, C2 and C3), Harrell's C, countries with >=20 strokes
    if "country" in ext:
        crow = []
        for c, sub in ext.groupby("country"):
            idx = sub.index.values
            ev = ext.loc[idx, "stroke"].values
            if ev.sum() < 20:
                continue
            pos = ext.index.get_indexer(idx)
            r = {"country_code": c, "n": len(idx), "strokes": int(ev.sum())}
            for s in ("C2", "C3"):
                r[f"C_{s}"] = concordance_index_censored(ev, ext.loc[idx, "time"].values,
                                                         risks[(s, "Cox")][pos])[0]
            crow.append(r)
        pd.DataFrame(crow).round(4).to_csv(os.path.join(out, f"A_external_{name}_by_country.csv"), index=False)

    # observed incidence by occupational group (Aalen-Johansen), external cohort
    inc = []
    for g, sub in ext.groupby("occ_group"):
        for h in (10, 15):
            inc.append({"group": g, "horizon": h, "n": len(sub), "cum_incidence_AJ": obs_aj(sub, h)})
    pd.DataFrame(inc).round(4).to_csv(os.path.join(out, f"A_external_{name}_incidence_by_group.csv"), index=False)


# ----------------------------------------------------------------------------
# Part B: richer work measures inside SHARE
# ----------------------------------------------------------------------------
def part_b(ext_raw, ext, name, out, n_rep, log):
    have = [v for v in STRAIN if v in ext_raw]
    if not have:
        log("  [B] no job-strain variables found; skipping Part B")
        return
    d = ext.copy()
    for v in have:
        d[v] = ext_raw[v].astype(float)
        if v != "jsprvs":
            d.loc[~d[v].between(1, 4), v] = np.nan   # Likert 1-4; anything else = missing
    if "jsprvs" in d:
        d["jsprvs"] = (d["jsprvs"] == 1).astype(float).where(d["jsprvs"].isin([0, 1]))
    if "country" in ext_raw:
        cdum = pd.get_dummies(ext_raw["country"].astype(int), prefix="cty", drop_first=True).astype(float)
    else:
        cdum = pd.DataFrame(index=d.index)
    d = pd.concat([d, cdum], axis=1)
    base = C3 + list(cdum.columns)
    sets = {"C3+country": base, "C3+country+strain": base + have}   # "country" adds nothing for single-country cohorts
    y = ysurv(d)
    rows = []
    for rep in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(d, d["event_cr"])):
            ytr = y[tr]
            yte = m02.clip_to_train(y[te], ytr)
            for sname, feats in sets.items():
                pp = Preproc().fit(d.iloc[tr][feats])
                for mname in ("Cox", "GBS"):
                    m = build_model(mname).fit(pp.transform(d.iloc[tr][feats]), ytr)
                    c = concordance_index_ipcw(ytr, yte, m.predict(pp.transform(d.iloc[te][feats])), tau=TAU)[0]
                    rows.append({"rep": rep, "fold": fold, "set": sname, "model": mname, "C_uno": c})
            log(f"  [B] rep{rep} fold{fold} done")
    cv = pd.DataFrame(rows)
    summ = cv.groupby(["model", "set"]).C_uno.agg(["mean", "std"]).reset_index()
    piv = cv.pivot_table(index=["model", "rep", "fold"], columns="set", values="C_uno").reset_index()
    piv["diff"] = piv["C3+country+strain"] - piv["C3+country"]
    inc = piv.groupby("model")["diff"].agg(mean_diff="mean", sd_diff="std",
                                           folds_improved=lambda x: f"{(x > 0).sum()}/{len(x)}").reset_index()
    summ.round(4).to_csv(os.path.join(out, f"B_{name}_strain_cv_summary.csv"), index=False)
    inc.round(4).to_csv(os.path.join(out, f"B_{name}_strain_increment.csv"), index=False)

    # country-stratified Cox hazard ratios
    feats = C3 + have
    x = d[feats].fillna(d[feats].median())
    x = x.assign(time=d["time"].values, stroke=d["stroke"].astype(int).values)
    if "country" in ext_raw:
        x["country"] = ext_raw["country"].astype(int).values
        cph = CoxPHFitter(penalizer=0.0).fit(x, "time", "stroke", strata=["country"])
    else:
        cph = CoxPHFitter(penalizer=0.0).fit(x, "time", "stroke")
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["HR", "lo", "hi", "p"]
    s.round(4).to_csv(os.path.join(out, f"B_{name}_strain_hazard_ratios.csv"))
    log("  [B] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrs", default="results")
    ap.add_argument("--ext", required=True)
    ap.add_argument("--name", default="SHARE")
    ap.add_argument("--out", default="results_external")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"run_log_{a.name}.txt"), "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    hrs_raw = pd.concat([pd.read_parquet(os.path.join(a.hrs, f)) for f in
                         ("cohort_development.parquet", "cohort_temporal_validation.parquet")],
                        ignore_index=True)
    ext_raw = pd.read_parquet(a.ext).reset_index(drop=True)
    hrs = harmonize(hrs_raw, "h_itot", "h_atotb")
    ext = harmonize(ext_raw, "h_itot", "h_atotb", group_col="country" if "country" in ext_raw else None)
    if "country" in ext_raw:
        ext["country"] = ext_raw["country"].values
    log(f"HRS training n={len(hrs)}, strokes={int(hrs.stroke.sum())} | "
        f"{a.name} n={len(ext)}, strokes={int(ext.stroke.sum())}")
    part_a(hrs, ext, a.name, a.out, a.boot, log)
    part_b(ext_raw, ext, a.name, a.out, a.reps, log)
    log("Done. Share the contents of " + a.out + " (aggregate only).")


if __name__ == "__main__":
    main()
