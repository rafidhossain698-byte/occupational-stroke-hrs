"""
02_models.py
Nested predictor-set survival modelling for incident stroke (HRS).

Input : results/cohort_development.parquet, results/cohort_temporal_validation.parquet
        (made by 01_build_cohort.py -- row-level, keep LOCAL)
Output: results/models/  -- aggregate tables (.csv) and figures (.png) only.
        These are safe to share; nothing row-level is written.

Estimand: cause-specific hazard of first stroke. Death without stroke is treated
as censoring for model fitting/evaluation (cause-specific), and is handled
explicitly in (a) Aalen-Johansen cumulative-incidence curves and (b) a
competing-risk sensitivity analysis reported by 03 later.

Models : ridge-penalised Cox, Random Survival Forest, gradient-boosted survival.
Sets   : M0 demographics -> M1 +clinical -> M2 +SES -> M3 +occupation/work -> M4 +lifestyle/psych
Primary comparison: M3 vs M2.

Validation:
  internal  = repeated stratified 5-fold CV on the development cohort (paired by fold)
  temporal  = models fitted on full development cohort, evaluated on the 1998 entry cohort,
              with bootstrap 95% CIs (resampling validation rows)

Usage (Mac):
  pip3 install pandas numpy scikit-learn scikit-survival lifelines shap matplotlib pyarrow
  python3 02_models.py --data results --out results/models            # full run
  python3 02_models.py --data results --out results/models --quick    # fast smoke test
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

warnings.filterwarnings("ignore")
SEED = 2026
HORIZONS = [10, 20]           # years; supported by event counts in the feasibility report
IBS_GRID = np.arange(1, 20.01, 0.5)

# ----------------------------------------------------------------------------
# Predictor sets (prespecified)
# ----------------------------------------------------------------------------
M0 = ["age", "female", "race_black", "race_other", "hispanic"]
M1 = M0 + ["hibpe", "diabe", "hearte", "bmi", "smokev", "smoken"]
M2 = M1 + ["edyrs", "log_income", "asinh_wealth", "married"]
M3 = M2 + ["occ_professional", "occ_routine_nonmanual", "occ_service_manual",
           "jstres", "jphys", "jlift", "jstoop", "jhours", "jcten", "slfemp"]
M4 = M3 + ["drink", "vigact", "shlt", "psyche"]
SETS = {"M0": M0, "M1": M1, "M2": M2, "M3": M3, "M4": M4}
# fsize / union dropped a priori: 9-17% missing and not central to the hypothesis.

PRETTY = {
    "age": "Age", "female": "Female", "race_black": "Race: Black", "race_other": "Race: other",
    "hispanic": "Hispanic", "hibpe": "Hypertension", "diabe": "Diabetes",
    "hearte": "Heart disease", "bmi": "BMI", "smokev": "Ever smoked", "smoken": "Current smoker",
    "edyrs": "Years of education", "log_income": "Household income (log)",
    "asinh_wealth": "Household wealth (asinh)", "married": "Married/partnered",
    "occ_professional": "Occ: professional (vs managerial)",
    "occ_routine_nonmanual": "Occ: routine non-manual (vs managerial)",
    "occ_service_manual": "Occ: service/manual (vs managerial)",
    "jstres": "Job stress (1=strongly agree .. 4)", "jphys": "Job: physical effort",
    "jlift": "Job: heavy lifting", "jstoop": "Job: stooping", "jhours": "Weekly work hours",
    "jcten": "Job tenure (years)", "slfemp": "Self-employed", "drink": "Drinks alcohol",
    "vigact": "Vigorous activity 3+/wk", "shlt": "Self-rated health (1=excellent..5)",
    "psyche": "Psychiatric condition",
}


def prepare(df):
    """Recode raw RAND values into model-ready predictors."""
    d = pd.DataFrame(index=df.index)
    d["time"] = df["time_years"].astype(float)
    d["stroke"] = (df["event"] == 1)
    d["death"] = (df["event"] == 2)
    d["event_cr"] = df["event"].astype(int)
    d["occ_group"] = df["occ_hierarchy"]

    d["age"] = df["age"]
    d["female"] = (df["sex"] == 2).astype(float)
    d["race_black"] = (df["racem"] == 2).astype(float).where(df["racem"].notna())
    d["race_other"] = (df["racem"] == 3).astype(float).where(df["racem"].notna())
    d["hispanic"] = (df["hispan"] == 1).astype(float).where(df["hispan"].notna())
    for v in ["hibpe", "diabe", "hearte", "smokev", "smoken", "slfemp",
              "drink", "vigact", "psyche"]:
        d[v] = (df[v] == 1).astype(float).where(df[v].isin([0, 1]))
    d["bmi"] = df["bmi"].where(df["bmi"].between(12, 70))
    d["edyrs"] = df["edyrs"]
    d["log_income"] = df["log_income"]
    d["asinh_wealth"] = df["asinh_wealth"]
    d["married"] = df["mstat"].isin([1, 2, 3]).astype(float).where(df["mstat"].notna())
    occ = df["occ_hierarchy"]
    d["occ_professional"] = (occ == "2_Professional").astype(float)
    d["occ_routine_nonmanual"] = (occ == "3_Routine_nonmanual").astype(float)
    d["occ_service_manual"] = (occ == "4_Service_manual").astype(float)
    d["jstres"] = df["jstres"].where(df["jstres"].between(1, 4))
    for v in ["jphys", "jlift", "jstoop"]:          # 1=all/almost all time .. 4=none
        d[v] = df[v].where(df[v].between(1, 4))
    d["jhours"] = df["jhours"].where(df["jhours"].between(1, 112))
    d["jcten"] = df["jcten"].where(df["jcten"] >= 0)
    d["shlt"] = df["shlt"].where(df["shlt"].between(1, 5))
    return d


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_model(name, quick=False):
    n_trees = 100 if quick else 300
    if name == "Cox":
        return CoxPHSurvivalAnalysis(alpha=1.0, ties="efron")
    if name == "RSF":
        return RandomSurvivalForest(n_estimators=n_trees, min_samples_leaf=40,
                                    max_features="sqrt", n_jobs=-1, random_state=SEED,
                                    low_memory=False)
    if name == "GBS":
        return GradientBoostingSurvivalAnalysis(n_estimators=60 if quick else 200,
                                                learning_rate=0.05, max_depth=3,
                                                min_samples_leaf=40, subsample=0.5,
                                                random_state=SEED)
    raise ValueError(name)


class Preproc:
    """Median/mode imputation + scaling, fitted on training data only (no leakage)."""
    def fit(self, X):
        self.imp = SimpleImputer(strategy="median").fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        return self

    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))


def surv_matrix(model, X, grid):
    fns = model.predict_survival_function(X)
    return np.vstack([f(grid) for f in fns])


def evaluate(y_tr, y_te, risk, surv_grid, grid):
    """Uno's C (tau=20), time-dependent AUC at horizons, IBS on grid."""
    out = {}
    tau = min(20, y_te["time"].max() - 1e-6)
    out["C_uno"] = concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0]
    for h in HORIZONS:
        try:
            out[f"AUC_{h}y"] = cumulative_dynamic_auc(y_tr, y_te, risk, [h])[0][0]
        except ValueError:
            out[f"AUC_{h}y"] = np.nan
    ok = grid < y_te["time"].max()
    out["IBS"] = integrated_brier_score(y_tr, y_te, surv_grid[:, ok], grid[ok])
    return out


def clip_to_train(y_te, y_tr):
    """sksurv's IPCW needs test times <= max train time."""
    y = y_te.copy()
    tmax = y_tr["time"].max()
    over = y["time"] > tmax
    y["time"][over] = tmax
    y["stroke"][over] = False
    return y


# ----------------------------------------------------------------------------
# Internal validation: repeated stratified CV
# ----------------------------------------------------------------------------
def run_cv(dev, models, n_rep, quick, log):
    rows, oof = [], {}
    y_all = Surv.from_arrays(dev["stroke"].values, dev["time"].values, "stroke", "time")
    for rep in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(skf.split(dev, dev["event_cr"])):
            y_tr, y_te = y_all[tr], clip_to_train(y_all[te], y_all[tr])
            for sname, feats in SETS.items():
                pp = Preproc().fit(dev.iloc[tr][feats])
                Xtr, Xte = pp.transform(dev.iloc[tr][feats]), pp.transform(dev.iloc[te][feats])
                for mname in models:
                    t0 = time.time()
                    m = make_model(mname, quick).fit(Xtr, y_tr)
                    risk = m.predict(Xte)
                    res = evaluate(y_tr, y_te, risk, surv_matrix(m, Xte, IBS_GRID), IBS_GRID)
                    rows.append({"rep": rep, "fold": fold, "set": sname, "model": mname, **res})
                    if rep == 0:
                        oof.setdefault((sname, mname), np.full(len(dev), np.nan))[te] = risk
                    log(f"  CV rep{rep} fold{fold} {sname}-{mname}: C={res['C_uno']:.3f} "
                        f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), oof


def summarise_cv(cv):
    metrics = [c for c in cv.columns if c.startswith(("C_", "AUC", "IBS"))]
    s = cv.groupby(["model", "set"])[metrics].agg(["mean", "std"]).round(4)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    # paired fold-wise increments vs previous set (primary: M3 - M2)
    deltas = []
    order = list(SETS)
    for mname, g in cv.groupby("model"):
        piv = {m: g.pivot_table(index=["rep", "fold"], columns="set", values=m) for m in metrics}
        for a, b in zip(order[:-1], order[1:]):
            for m in metrics:
                diff = piv[m][b] - piv[m][a]
                deltas.append({"model": mname, "comparison": f"{b} - {a}", "metric": m,
                               "mean_diff": diff.mean(), "sd_diff": diff.std(),
                               "folds_improved": f"{(diff > 0).sum() if m != 'IBS' else (diff < 0).sum()}/{len(diff)}"})
    return s.reset_index(), pd.DataFrame(deltas).round(4)


# ----------------------------------------------------------------------------
# Temporal validation with bootstrap CIs
# ----------------------------------------------------------------------------
def run_temporal(dev, val, models, n_boot, quick, log):
    y_dev = Surv.from_arrays(dev["stroke"].values, dev["time"].values, "stroke", "time")
    y_val = clip_to_train(Surv.from_arrays(val["stroke"].values, val["time"].values,
                                           "stroke", "time"), y_dev)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(val), len(val)) for _ in range(n_boot)]
    rows, fitted = [], {}
    for sname, feats in SETS.items():
        pp = Preproc().fit(dev[feats])
        Xd, Xv = pp.transform(dev[feats]), pp.transform(val[feats])
        for mname in models:
            m = make_model(mname, quick).fit(Xd, y_dev)
            risk = m.predict(Xv)
            S = surv_matrix(m, Xv, IBS_GRID)
            point = evaluate(y_dev, y_val, risk, S, IBS_GRID)
            bs = []
            for idx in boots:
                try:
                    bs.append(evaluate(y_dev, y_val[idx], risk[idx], S[idx], IBS_GRID))
                except ValueError:
                    continue
            bs = pd.DataFrame(bs)
            row = {"set": sname, "model": mname}
            for k, v in point.items():
                row[k] = v
                row[f"{k}_lo"], row[f"{k}_hi"] = bs[k].quantile([.025, .975]).values
            rows.append(row)
            fitted[(sname, mname)] = (m, pp, risk, S)
            log(f"  Temporal {sname}-{mname}: C={point['C_uno']:.3f}")
    # paired bootstrap for primary comparison M3 vs M2 (per model)
    paired = []
    for mname in models:
        r2, r3 = fitted[("M2", mname)][2], fitted[("M3", mname)][2]
        d = []
        for idx in boots:
            try:
                c2 = concordance_index_ipcw(y_dev, y_val[idx], r2[idx], tau=20)[0]
                c3 = concordance_index_ipcw(y_dev, y_val[idx], r3[idx], tau=20)[0]
                d.append(c3 - c2)
            except ValueError:
                pass
        d = np.array(d)
        paired.append({"model": mname, "delta_C_M3_minus_M2": d.mean(),
                       "lo": np.quantile(d, .025), "hi": np.quantile(d, .975),
                       "p_boot_le0": (d <= 0).mean()})
    return pd.DataFrame(rows).round(4), pd.DataFrame(paired).round(4), fitted


# ----------------------------------------------------------------------------
# Calibration and decision curves (temporal validation, 10-year horizon)
# ----------------------------------------------------------------------------
def km_risk(time_, event_, h):
    if event_.sum() == 0:
        return 0.0
    t, s = kaplan_meier_estimator(event_, time_)
    s_h = s[t <= h][-1] if (t <= h).any() else 1.0
    return 1 - s_h


def calibration(val, fitted, models, out, h=10):
    gi = int(np.argmin(np.abs(IBS_GRID - h)))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    rows = []
    for ax, mname in zip(axes[0], models):
        for sname, style in [("M2", "o--"), ("M3", "s-")]:
            S = fitted[(sname, mname)][3]
            pred = 1 - S[:, gi]
            q = pd.qcut(pred, 10, labels=False, duplicates="drop")
            xs, ys = [], []
            for g in np.unique(q):
                idx = q == g
                xs.append(pred[idx].mean())
                ys.append(km_risk(val["time"].values[idx], val["stroke"].values[idx], h))
            obs_all = km_risk(val["time"].values, val["stroke"].values, h)
            rows.append({"model": mname, "set": sname, "horizon": h,
                         "mean_predicted": pred.mean(), "observed_KM": obs_all,
                         "O_E_ratio": obs_all / pred.mean()})
            ax.plot(xs, ys, style, label=sname, ms=4)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color="grey", lw=.8)
        ax.set_title(f"{mname}: {h}-year calibration")
        ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed risk (KM)")
        ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_calibration_{h}y.png"), dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows).round(4)


def decision_curve(val, fitted, mname, out, h=10):
    gi = int(np.argmin(np.abs(IBS_GRID - h)))
    thr = np.linspace(0.01, 0.20, 40)
    n = len(val)
    t, e = val["time"].values, val["stroke"].values
    prev = km_risk(t, e, h)
    fig, ax = plt.subplots(figsize=(5, 4))
    rows = []
    for sname in ["M2", "M3", "M4"]:
        pred = 1 - fitted[(sname, mname)][3][:, gi]
        nb = []
        for p in thr:
            pos = pred >= p
            if pos.sum() == 0:
                nb.append(0.0); continue
            r = km_risk(t[pos], e[pos], h)
            tp, fp = r * pos.sum() / n, (1 - r) * pos.sum() / n
            nb.append(tp - fp * p / (1 - p))
        ax.plot(thr, nb, label=sname)
        rows += [{"set": sname, "threshold": a, "net_benefit": b} for a, b in zip(thr, nb)]
    treat_all = [prev - (1 - prev) * p / (1 - p) for p in thr]
    ax.plot(thr, treat_all, "k--", lw=.8, label="Treat all")
    ax.axhline(0, color="k", lw=.8, label="Treat none")
    ax.set_ylim(-0.01, max(prev, 0.01) * 1.2)
    ax.set_xlabel(f"Threshold {h}-year risk"); ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve ({mname}, temporal validation)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_dca_{h}y.png"), dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows).round(5)


# ----------------------------------------------------------------------------
# Interpretation: hazard ratios, cumulative incidence, SHAP, subgroups
# ----------------------------------------------------------------------------
def hazard_ratios(dev, out):
    from lifelines import CoxPHFitter
    res = []
    for sname in ["M2", "M3", "M4"]:
        feats = SETS[sname]
        d = dev[feats + ["time", "stroke"]].copy()
        d[feats] = SimpleImputer(strategy="median").fit_transform(d[feats])
        cph = CoxPHFitter(penalizer=0.0).fit(d, "time", "stroke")
        s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
        s = s.rename(columns={"exp(coef)": "HR", "exp(coef) lower 95%": "lo",
                              "exp(coef) upper 95%": "hi"})
        s["set"], s["variable"] = sname, [PRETTY.get(i, i) for i in s.index]
        res.append(s.reset_index(drop=True))
        if sname == "M3":
            ph = []
            try:
                from lifelines.statistics import proportional_hazard_test
                t = proportional_hazard_test(cph, d, time_transform="rank").summary
                ph = t.assign(variable=[PRETTY.get(i, i) for i in t.index])
                ph.round(4).to_csv(os.path.join(out, "table_ph_test_M3.csv"), index=False)
            except Exception as ex:  # noqa
                print("  PH test skipped:", ex)
    pd.concat(res).round(4).to_csv(os.path.join(out, "table_hazard_ratios.csv"), index=False)


def cumulative_incidence(dev, out):
    from lifelines import AalenJohansenFitter
    fig, ax = plt.subplots(figsize=(5.5, 4))
    rows = []
    for g, sub in dev.groupby("occ_group"):
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(sub["time"], sub["event_cr"], event_of_interest=1)
        ci = aj.cumulative_density_
        ax.step(ci.index, ci.iloc[:, 0], where="post", label=g.split("_", 1)[1].replace("_", " "))
        for h in HORIZONS:
            v = ci[ci.index <= h].iloc[-1, 0]
            rows.append({"group": g, "horizon": h, "cumulative_incidence": v, "n": len(sub)})
    ax.set_xlim(0, 25); ax.set_xlabel("Years since baseline")
    ax.set_ylabel("Cumulative incidence of stroke\n(death as competing risk)")
    ax.set_title("Stroke incidence by occupational group (development)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cumulative_incidence.png"), dpi=300); plt.close(fig)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "table_cumulative_incidence.csv"), index=False)


def shap_summary(dev, fitted, out, mname="GBS", sname="M4", n_explain=400, n_bg=100):
    import shap
    m, pp = fitted[(sname, mname)][0], fitted[(sname, mname)][1]
    feats = SETS[sname]
    X = pp.transform(dev[feats])
    rng = np.random.default_rng(SEED)
    bg = X[rng.choice(len(X), n_bg, replace=False)]
    ex_idx = rng.choice(len(X), min(n_explain, len(X)), replace=False)
    explainer = shap.Explainer(m.predict, bg, feature_names=[PRETTY[f] for f in feats])
    sv = explainer(X[ex_idx])
    imp = pd.DataFrame({"feature": [PRETTY[f] for f in feats],
                        "mean_abs_shap": np.abs(sv.values).mean(0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.round(5).to_csv(os.path.join(out, f"table_shap_{sname}_{mname}.csv"), index=False)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_shap_{sname}_{mname}.png"), dpi=300)
    plt.close("all")


def subgroup_c(dev, oof, out, models):
    from sksurv.metrics import concordance_index_censored
    rows = []
    groups = {"Sex": dev["female"].map({0.0: "Male", 1.0: "Female"}),
              "Occupation": dev["occ_group"]}
    for (sname, mname), risk in oof.items():
        if sname not in ("M2", "M3"):
            continue
        for gname, g in groups.items():
            for lvl in g.dropna().unique():
                idx = (g == lvl).values & ~np.isnan(risk)
                ev = dev["stroke"].values[idx]
                if ev.sum() < 20:
                    continue
                c = concordance_index_censored(ev, dev["time"].values[idx], risk[idx])[0]
                rows.append({"model": mname, "set": sname, "subgroup": gname, "level": lvl,
                             "n": int(idx.sum()), "strokes": int(ev.sum()), "C_harrell": c})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "table_subgroup_c.csv"), index=False)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="results/models")
    ap.add_argument("--quick", action="store_true", help="fewer trees/reps/bootstraps")
    ap.add_argument("--skip-cv", action="store_true",
                    help="reuse saved cross-validation results; run only temporal validation onward")
    ap.add_argument("--boot", type=int, default=None, help="number of bootstrap resamples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run_log.txt"), "a" if a.skip_cv else "w")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    dev = prepare(pd.read_parquet(os.path.join(a.data, "cohort_development.parquet")))
    val = prepare(pd.read_parquet(os.path.join(a.data, "cohort_temporal_validation.parquet")))
    log(f"Development n={len(dev)}, strokes={dev.stroke.sum()} | "
        f"Validation n={len(val)}, strokes={val.stroke.sum()}")

    models = ["Cox", "RSF", "GBS"]
    n_rep, n_boot = (1, 50) if a.quick else (2, 500)
    if a.boot:
        n_boot = a.boot

    cv_done = all(os.path.exists(os.path.join(a.out, f)) for f in
                  ["cv_fold_metrics.csv", "table_cv_summary.csv", "table_subgroup_c.csv"])
    if a.skip_cv and cv_done:
        log("== Resuming: cross-validation results found, skipping descriptives and CV")
    else:
        if a.skip_cv:
            log("[warn] --skip-cv given but saved CV results not found; running CV")
        log("== Descriptives")
        cumulative_incidence(dev, a.out)
        hazard_ratios(dev, a.out)

        log("== Internal validation (repeated 5-fold CV)")
        cv, oof = run_cv(dev, models, n_rep, a.quick, log)
        cv.round(4).to_csv(os.path.join(a.out, "cv_fold_metrics.csv"), index=False)
        summ, deltas = summarise_cv(cv)
        summ.to_csv(os.path.join(a.out, "table_cv_summary.csv"), index=False)
        deltas.to_csv(os.path.join(a.out, "table_cv_increments.csv"), index=False)
        subgroup_c(dev, oof, a.out, models)

    log("== Temporal validation")
    tv, paired, fitted = run_temporal(dev, val, models, n_boot, a.quick, log)
    tv.to_csv(os.path.join(a.out, "table_temporal_validation.csv"), index=False)
    paired.to_csv(os.path.join(a.out, "table_temporal_M3_vs_M2.csv"), index=False)
    for h in HORIZONS:
        calibration(val, fitted, models, a.out, h).to_csv(
            os.path.join(a.out, f"table_calibration_{h}y.csv"), index=False)
    decision_curve(val, fitted, "Cox", a.out, 10).to_csv(
        os.path.join(a.out, "table_dca_10y.csv"), index=False)

    log("== SHAP")
    shap_summary(dev, fitted, a.out)
    log("Done. Share the contents of " + a.out + " (aggregate only).")


if __name__ == "__main__":
    main()
