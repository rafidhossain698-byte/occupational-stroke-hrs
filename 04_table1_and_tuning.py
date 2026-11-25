"""
04_table1_and_tuning.py
  Part A. Table 1: baseline characteristics by occupational class (development cohort)
          plus the temporal validation cohort overall.
  Part B. Hyperparameter tuning of RSF and GBS.
          Tuning uses 3-fold CV on the DEVELOPMENT cohort only (selection by Uno's C);
          the selected settings are refitted on the full development cohort and evaluated
          ONCE on the temporal validation cohort, which never influences tuning.
          Settings are tuned on M3 and the same settings are used for M2, so the
          M3-vs-M2 comparison stays like-for-like.

Input : results/cohort_*.parquet (LOCAL ONLY)
Output: results/extras/  -- aggregate tables only (safe to share)

Usage (Mac), same folder as 02_models.py:
  python3 04_table1_and_tuning.py --data results --out results/extras
  python3 04_table1_and_tuning.py --data results --out results/extras --table1-only   # Part A only (~1 min)
Runtime for Part B: roughly 30-60 minutes.
"""
import argparse
import importlib.util
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

warnings.filterwarnings("ignore")
SEED = 2026
SMALL = 11
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("models02", os.path.join(HERE, "02_models.py"))
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)


# ----------------------------------------------------------------------------
# Part A: Table 1
# ----------------------------------------------------------------------------
def table1(raw_dev, raw_val, out, log):
    dev, val = m02.prepare(raw_dev), m02.prepare(raw_val)
    for d, r in ((dev, raw_dev), (val, raw_val)):
        d["income_k"] = r["h_itot"].values / 1000
        d["wealth_k"] = r["h_atotb"].values / 1000
        d["high_stress"] = (d["jstres"] <= 2).astype(float).where(d["jstres"].notna())

    cont = [("Age, years", "age", "mean"), ("Education, years", "edyrs", "mean"),
            ("Household income, $1000s", "income_k", "median"),
            ("Household wealth, $1000s", "wealth_k", "median"),
            ("BMI, kg/m2", "bmi", "mean"), ("Weekly work hours", "jhours", "mean"),
            ("Job tenure, years", "jcten", "mean")]
    binv = [("Female", "female"), ("Black", "race_black"), ("Hispanic", "hispanic"),
            ("Married/partnered", "married"), ("Hypertension", "hibpe"),
            ("Diabetes", "diabe"), ("Heart disease", "hearte"), ("Current smoker", "smoken"),
            ("High job stress (agree/strongly agree)", "high_stress"),
            ("Self-employed", "slfemp"), ("Vigorous activity 3+/week", "vigact")]

    groups = [(g.split("_", 1)[1].replace("_", " "), sub) for g, sub in dev.groupby("occ_group")]
    cols = groups + [("Development, all", dev), ("Validation, all", val)]

    def fmt_cont(x, how):
        x = x.dropna()
        if how == "mean":
            return f"{x.mean():.1f} ({x.std():.1f})"
        q = x.quantile([.25, .5, .75]).values
        return f"{q[1]:.0f} ({q[0]:.0f}-{q[2]:.0f})"

    def fmt_bin(x):
        x = x.dropna()
        n = int(x.sum())
        if 0 < n < SMALL:
            return f"<{SMALL}"
        return f"{n:,} ({100 * x.mean():.1f})"

    rows = [{"Characteristic": "N", **{c: f"{len(d):,}" for c, d in cols}, "p (across classes)": ""}]
    for lab, v, how in cont:
        r = {"Characteristic": lab + (", mean (SD)" if how == "mean" else ", median (IQR)")}
        for c, d in cols:
            r[c] = fmt_cont(d[v], how)
        samples = [sub[v].dropna() for _, sub in groups]
        p = stats.kruskal(*samples).pvalue if how == "median" else stats.f_oneway(*samples).pvalue
        r["p (across classes)"] = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append(r)
    for lab, v in binv:
        r = {"Characteristic": lab + ", n (%)"}
        for c, d in cols:
            r[c] = fmt_bin(d[v])
        tab = pd.crosstab(dev["occ_group"], dev[v].dropna())
        p = stats.chi2_contingency(tab)[1] if tab.shape[1] == 2 else np.nan
        r["p (across classes)"] = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append(r)
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(out, "table1_baseline_characteristics.csv"), index=False)
    log(f"[A] Table 1 written ({len(t1)} rows)")


# ----------------------------------------------------------------------------
# Part B: tuning
# ----------------------------------------------------------------------------
GRIDS = {
    "RSF": {"min_samples_leaf": [20, 40, 80], "max_features": ["sqrt", 0.33]},
    "GBS": {"learning_rate": [0.05, 0.1], "max_depth": [2, 3], "n_estimators": [100, 250]},
}


def build(name, params):
    if name == "RSF":
        return RandomSurvivalForest(n_estimators=300, n_jobs=-1, random_state=SEED,
                                    low_memory=True, **params)
    return GradientBoostingSurvivalAnalysis(subsample=0.5, min_samples_leaf=40,
                                            random_state=SEED, **params)


def tune(dev, feats, name, log):
    y = Surv.from_arrays(dev["stroke"].values, dev["time"].values, "stroke", "time")
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    folds = list(skf.split(dev, dev["event_cr"]))
    keys = list(GRIDS[name])
    results = []
    for combo in itertools.product(*[GRIDS[name][k] for k in keys]):
        params = dict(zip(keys, combo))
        cs, t0 = [], time.time()
        for tr, te in folds:
            pp = m02.Preproc().fit(dev.iloc[tr][feats])
            mdl = build(name, params).fit(pp.transform(dev.iloc[tr][feats]), y[tr])
            yte = m02.clip_to_train(y[te], y[tr])
            cs.append(concordance_index_ipcw(y[tr], yte, mdl.predict(pp.transform(dev.iloc[te][feats])), tau=20)[0])
        results.append({"model": name, **{k: str(v) for k, v in params.items()},
                        "cv_C_mean": np.mean(cs), "cv_C_sd": np.std(cs)})
        log(f"  [B] {name} {params}: C={np.mean(cs):.4f} ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(results).sort_values("cv_C_mean", ascending=False)
    best = res.iloc[0]
    best_params = {}
    for k in keys:
        v = best[k]
        best_params[k] = v if v == "sqrt" else (float(v) if "." in v else int(v))
    return res, best_params


def temporal_eval(dev, val, name, params, n_boot, log):
    y_dev = Surv.from_arrays(dev["stroke"].values, dev["time"].values, "stroke", "time")
    y_val = m02.clip_to_train(Surv.from_arrays(val["stroke"].values, val["time"].values,
                                               "stroke", "time"), y_dev)
    risks, rows = {}, []
    for sname in ("M2", "M3"):
        feats = m02.SETS[sname]
        pp = m02.Preproc().fit(dev[feats])
        mdl = build(name, params).fit(pp.transform(dev[feats]), y_dev)
        risks[sname] = mdl.predict(pp.transform(val[feats]))
    rng = np.random.default_rng(SEED)
    boot = {"M2": [], "M3": [], "d": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(val), len(val))
        try:
            c2 = concordance_index_ipcw(y_dev, y_val[idx], risks["M2"][idx], tau=20)[0]
            c3 = concordance_index_ipcw(y_dev, y_val[idx], risks["M3"][idx], tau=20)[0]
        except ValueError:
            continue
        boot["M2"].append(c2); boot["M3"].append(c3); boot["d"].append(c3 - c2)
    for sname in ("M2", "M3"):
        c = concordance_index_ipcw(y_dev, y_val, risks[sname], tau=20)[0]
        rows.append({"model": f"{name} (tuned)", "set": sname, "C_uno": c,
                     "lo": np.quantile(boot[sname], .025), "hi": np.quantile(boot[sname], .975)})
    d = rows[1]["C_uno"] - rows[0]["C_uno"]
    rows.append({"model": f"{name} (tuned)", "set": "Delta M3-M2", "C_uno": d,
                 "lo": np.quantile(boot["d"], .025), "hi": np.quantile(boot["d"], .975)})
    log(f"  [B] {name} tuned temporal: M2={rows[0]['C_uno']:.4f}, M3={rows[1]['C_uno']:.4f}, delta={d:+.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="results/extras")
    ap.add_argument("--table1-only", action="store_true")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run_log.txt"), "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    raw_dev = pd.read_parquet(os.path.join(a.data, "cohort_development.parquet")).reset_index(drop=True)
    raw_val = pd.read_parquet(os.path.join(a.data, "cohort_temporal_validation.parquet")).reset_index(drop=True)
    table1(raw_dev, raw_val, a.out, log)
    if a.table1_only:
        log("Done (Table 1 only)."); return

    dev, val = m02.prepare(raw_dev), m02.prepare(raw_val)
    grid_all, temporal, chosen = [], [], []
    for name in ("RSF", "GBS"):
        res, best = tune(dev, m02.SETS["M3"], name, log)
        grid_all.append(res)
        chosen.append({"model": name, **{k: str(v) for k, v in best.items()}})
        temporal += temporal_eval(dev, val, name, best, a.boot, log)
    pd.concat(grid_all).round(4).to_csv(os.path.join(a.out, "tuning_grid_results.csv"), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(a.out, "tuning_selected_params.csv"), index=False)
    pd.DataFrame(temporal).round(4).to_csv(os.path.join(a.out, "tuned_temporal_validation.csv"), index=False)
    log("Done. Share the contents of " + a.out + " (aggregate only).")


if __name__ == "__main__":
    main()
