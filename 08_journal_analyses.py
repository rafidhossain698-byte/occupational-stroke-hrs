"""
08_journal_analyses.py
Additional analyses for the three-cohort journal paper (HRS, SHARE, ELSA). Cox models throughout
(the primary model family; ML models gave the same conclusions in 02/06).

  J1  Common-predictor incremental value (C3 vs C2) in each cohort, with 95% and 90% bootstrap CIs
        HRS : train 1992 cohort -> test 1998 cohort (temporal)
        SHARE, ELSA : train pooled HRS -> test external cohort
  J2  Equivalence test (TOST): is the upper 90% bound of Delta-C below margins 0.01 and 0.02?
  J3  Random-effects meta-analysis (DerSimonian-Laird) of Delta-C and of the adjusted
      service/manual vs managerial hazard ratio across the three cohorts
  J4  Mediation (difference method): attenuation of occupational-class HRs after adjustment for
      SES, clinical factors, and both; bootstrap CIs
  J5  Survey-weighted occupational HRs and weighted 10-year incidence by class
  J6  Multiple imputation (m=5, chained equations) instead of median imputation: pooled HRs
      (Rubin's rules) and Delta-C averaged over imputations
  J7  Sex-stratified Delta-C and sex x occupation interaction
  J8  HRS employees only (comparability with SHARE, where self-employed lacked occupation codes)
  J9  Fine-Gray competing-risk models trained on HRS, validated externally (Delta-C, O/E vs Aalen-Johansen)
  J10 Decision curves (10-year, competing-risk-aware) in SHARE (as transported) and ELSA (recalibrated)

Inputs (all LOCAL ONLY): cohort parquet files from 01, 05, 07; raw files only for survey weights.
Output: results_journal/  -- aggregate tables and figures only (safe to share)

Usage (Mac):
  caffeinate -i python3 08_journal_analyses.py \
      --hrs results --hrs_sav randhrs1992_2022v2.sav \
      --share results_share/cohort_share.parquet \
      --share_dta share_data/GH_SHARE_g_rel9-0-0_ALL_datasets_stata/GH_SHARE_g.dta \
      --elsa results_elsa/cohort_elsa.parquet --elsa_dta elsa_data/gh_elsa_h.dta \
      --out results_journal
  (the three raw-file arguments are optional; without them J5 is skipped)
Runtime: roughly 1-2 hours.
"""
import argparse
import importlib.util
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

warnings.filterwarnings("ignore")
SEED = 2026
TAU = 15
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m02 = _load("m02", "02_models.py")
m03 = _load("m03", "03_sensitivity.py")
m06 = _load("m06", "06_external_validation.py")
Preproc, harmonize = m02.Preproc, m06.harmonize
C1, C2, C3, OCC = m06.C1, m06.C2, m06.C3, m06.OCC
SES = ["educ3", "income_pct", "wealth_pct", "married"]
CLIN = [v for v in C1 if v not in ("age", "female")]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def ysurv(d):
    return Surv.from_arrays(d["stroke"].values, d["time"].values, "stroke", "time")


def clip_time(test, train):
    t = test.copy()
    tmax = train["time"].max()
    over = t["time"] > tmax
    t.loc[over, "time"] = tmax
    t.loc[over, "stroke"] = False
    return t


def cox_fit_predict(train, test, feats):
    pp = Preproc().fit(train[feats])
    m = CoxPHSurvivalAnalysis(alpha=1.0, ties="efron").fit(pp.transform(train[feats]), ysurv(train))
    return m, pp, m.predict(pp.transform(test[feats]))


def delta_c(train, test, n_boot, rng, sub=None, featsA=C2, featsB=C3):
    """Delta-C (B - A) with paired bootstrap; returns dict with point, 95% and 90% CIs."""
    test = clip_time(test, train)
    _, _, rA = cox_fit_predict(train, test, featsA)
    _, _, rB = cox_fit_predict(train, test, featsB)
    idx_all = np.arange(len(test)) if sub is None else np.where(sub)[0]
    ytr, yte = ysurv(train), ysurv(test)
    cA = concordance_index_ipcw(ytr, yte[idx_all], rA[idx_all], tau=TAU)[0]
    cB = concordance_index_ipcw(ytr, yte[idx_all], rB[idx_all], tau=TAU)[0]
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        try:
            diffs.append(concordance_index_ipcw(ytr, yte[idx], rB[idx], tau=TAU)[0] -
                         concordance_index_ipcw(ytr, yte[idx], rA[idx], tau=TAU)[0])
        except ValueError:
            continue
    diffs = np.array(diffs)
    return {"C_A": cA, "C_B": cB, "delta": cB - cA, "se": diffs.std(ddof=1),
            "lo95": np.quantile(diffs, .025), "hi95": np.quantile(diffs, .975),
            "lo90": np.quantile(diffs, .05), "hi90": np.quantile(diffs, .95),
            "n": len(idx_all), "strokes": int(test["stroke"].values[idx_all].sum())}


def lifelines_cox(d, feats, strata=None, weights=None, robust=False):
    x = d[feats].copy()
    x = x.fillna(x.median())
    x["time"], x["stroke"] = d["time"].values, d["stroke"].astype(int).values
    kw = {}
    if strata is not None:
        x["_strata"] = strata.values if hasattr(strata, "values") else strata
        kw["strata"] = ["_strata"]
    if weights is not None:
        x["_w"] = weights
        kw["weights_col"] = "_w"
        robust = True
    return CoxPHFitter(penalizer=1e-4).fit(x, "time", "stroke", robust=robust, **kw)


def dl_meta(est, se):
    """DerSimonian-Laird random-effects meta-analysis."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - fe) ** 2)
    df = len(est) - 1
    tau2 = max(0.0, (q - df) / (w.sum() - (w ** 2).sum() / w.sum())) if df > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    re = np.sum(wr * est) / wr.sum()
    se_re = np.sqrt(1 / wr.sum())
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {"pooled": re, "lo": re - 1.96 * se_re, "hi": re + 1.96 * se_re,
            "lo90": re - 1.645 * se_re, "hi90": re + 1.645 * se_re, "se": se_re,
            "I2_pct": i2, "tau2": tau2, "Q": q}


def aj_risk(time, status, h):
    aj = AalenJohansenFitter(calculate_variance=False).fit(time, status, event_of_interest=1)
    ci = aj.cumulative_density_
    return float(ci[ci.index <= h].iloc[-1, 0]) if (ci.index <= h).any() else 0.0


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_all(a, log):
    dev_raw = pd.read_parquet(os.path.join(a.hrs, "cohort_development.parquet"))
    val_raw = pd.read_parquet(os.path.join(a.hrs, "cohort_temporal_validation.parquet"))
    sh_raw = pd.read_parquet(a.share).reset_index(drop=True)
    el_raw = pd.read_parquet(a.elsa).reset_index(drop=True)

    # survey weights (optional)
    dev_raw["w"] = val_raw["w"] = np.nan
    sh_raw["w"] = el_raw["w"] = np.nan
    try:
        import pyreadstat
        if a.hrs_sav:
            ws, _ = pyreadstat.read_sav(a.hrs_sav, usecols=["HHIDPN", "R1WTRESP", "R4WTRESP"])
            dev_raw["w"] = dev_raw["HHIDPN"].map(ws.set_index("HHIDPN")["R1WTRESP"])
            val_raw["w"] = val_raw["HHIDPN"].map(ws.set_index("HHIDPN")["R4WTRESP"])
        if a.share_dta:
            ws, _ = pyreadstat.read_dta(a.share_dta, usecols=["mergeid", "r1wtsamp"])
            sh_raw["w"] = sh_raw["id"].map(ws.set_index("mergeid")["r1wtsamp"])
        if a.elsa_dta:
            ws, _ = pyreadstat.read_dta(a.elsa_dta, usecols=["idauniq", "r2wtresp"])
            el_raw["w"] = el_raw["id"].map(ws.set_index("idauniq")["r2wtresp"])
    except Exception as ex:  # noqa
        log(f"[warn] survey weights not loaded: {ex}")

    dev = harmonize(dev_raw, "h_itot", "h_atotb")
    val = harmonize(val_raw, "h_itot", "h_atotb")
    hrs_raw = pd.concat([dev_raw, val_raw], ignore_index=True)
    hrs = harmonize(hrs_raw, "h_itot", "h_atotb")
    sh = harmonize(sh_raw, "h_itot", "h_atotb", group_col="country")
    el = harmonize(el_raw, "h_itot", "h_atotb")
    for d, r in ((dev, dev_raw), (val, val_raw), (hrs, hrs_raw), (sh, sh_raw), (el, el_raw)):
        d["w"] = r["w"].values
        d["slfemp"] = r["slfemp"].values if "slfemp" in r else np.nan
    sh["country"] = sh_raw["country"].values
    log(f"HRS dev n={len(dev)} | HRS 1998 n={len(val)} | SHARE n={len(sh)} | ELSA n={len(el)}")
    return dev, val, hrs, sh, el


# ----------------------------------------------------------------------------
# J1-J3
# ----------------------------------------------------------------------------
def j1_j3(dev, val, hrs, sh, el, out, n_boot, log):
    rng = np.random.default_rng(SEED)
    res = {"HRS (temporal)": delta_c(dev, val, n_boot, rng),
           "SHARE (external)": delta_c(hrs, sh, n_boot, rng),
           "ELSA (external)": delta_c(hrs, el, n_boot, rng)}
    rows = [{"cohort": k, **v} for k, v in res.items()]
    t = pd.DataFrame(rows)
    pooled = dl_meta(t["delta"], t["se"])
    t = pd.concat([t, pd.DataFrame([{"cohort": "Pooled (random effects)", "delta": pooled["pooled"],
                                     "se": pooled["se"], "lo95": pooled["lo"], "hi95": pooled["hi"],
                                     "lo90": pooled["lo90"], "hi90": pooled["hi90"]}])], ignore_index=True)
    for m in (0.01, 0.02):
        t[f"equivalent_margin_{m}"] = t["hi90"] < m
    t.round(4).to_csv(os.path.join(out, "J1_J2_deltaC_equivalence.csv"), index=False)
    pd.DataFrame([{"quantity": "Delta-C (C3-C2), Cox", **pooled}]).round(4).to_csv(
        os.path.join(out, "J3_meta_deltaC.csv"), index=False)
    log(f"  J1-J3 pooled Delta-C = {pooled['pooled']:+.4f} ({pooled['lo']:+.4f} to {pooled['hi']:+.4f}), I2={pooled['I2_pct']:.0f}%")

    # meta-analysis of adjusted service/manual HR (C2 covariates + occupation)
    hr_rows = []
    for name, d, strata in (("HRS", hrs, None), ("SHARE", sh, sh["country"]), ("ELSA", el, None)):
        f = lifelines_cox(d, C2 + OCC, strata=strata)
        for v in OCC:
            hr_rows.append({"cohort": name, "term": v, "logHR": f.params_[v],
                            "se": f.standard_errors_[v]})
    hr = pd.DataFrame(hr_rows)
    meta = []
    for v in OCC:
        s = hr[hr.term == v]
        mm = dl_meta(s["logHR"], s["se"])
        meta.append({"term": v, "pooled_HR": np.exp(mm["pooled"]), "lo": np.exp(mm["lo"]),
                     "hi": np.exp(mm["hi"]), "I2_pct": mm["I2_pct"]})
    hr["HR"], hr["lo"], hr["hi"] = np.exp(hr.logHR), np.exp(hr.logHR - 1.96 * hr.se), np.exp(hr.logHR + 1.96 * hr.se)
    hr.round(4).to_csv(os.path.join(out, "J3_occupation_HR_by_cohort.csv"), index=False)
    pd.DataFrame(meta).round(4).to_csv(os.path.join(out, "J3_meta_occupation_HR.csv"), index=False)
    forest(t, out)


def forest(t, out):
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        pooled = r["cohort"].startswith("Pooled")
        ax.plot([r.lo95, r.hi95], [yi, yi], color="black", lw=1)
        ax.plot(r.delta, yi, "D" if pooled else "s", color="#1f4e79" if pooled else "black", ms=6 if pooled else 4)
    ax.axvline(0, color="grey", lw=.8)
    for m in (-0.01, 0.01):
        ax.axvline(m, color="#c55a11", lw=.7, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(t["cohort"], fontsize=8)
    ax.set_xlabel("Change in Uno's C-index from adding occupation (C3 - C2)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_forest_deltaC.pdf")); fig.savefig(os.path.join(out, "fig_forest_deltaC.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# J4 mediation
# ----------------------------------------------------------------------------
def j4_mediation(hrs, sh, el, out, n_boot, log):
    base = ["age", "female"]
    models = {"M_base": base, "M_SES": base + SES, "M_clin": base + CLIN, "M_both": base + SES + CLIN}
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d, strat in (("HRS", hrs, "none"), ("SHARE", sh, "country"), ("ELSA", el, "none")):
        def fit_all(dd):
            s = dd["country"] if strat == "country" else None
            return {k: lifelines_cox(dd, f + OCC, strata=s).params_[OCC] for k, f in models.items()}
        point = fit_all(d)
        boots = []
        t0 = time.time()
        for b in range(n_boot):
            idx = rng.integers(0, len(d), len(d))
            try:
                boots.append(fit_all(d.iloc[idx].reset_index(drop=True)))
            except Exception:  # noqa
                continue
        for v in OCC:
            for k in models:
                bb = np.array([bt[k][v] for bt in boots])
                rows.append({"cohort": name, "term": v, "model": k, "HR": np.exp(point[k][v]),
                             "lo": np.exp(np.quantile(bb, .025)), "hi": np.exp(np.quantile(bb, .975))})
            for k in ("M_SES", "M_clin", "M_both"):
                att = lambda p: (p["M_base"][v] - p[k][v]) / p["M_base"][v] * 100 if abs(p["M_base"][v]) > 1e-6 else np.nan
                bb = np.array([att(bt) for bt in boots])
                rows.append({"cohort": name, "term": v, "model": f"attenuation_% ({k} vs base)",
                             "HR": att(point), "lo": np.nanquantile(bb, .025), "hi": np.nanquantile(bb, .975)})
        log(f"  J4 {name} done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).round(3).to_csv(os.path.join(out, "J4_mediation.csv"), index=False)


# ----------------------------------------------------------------------------
# J5 survey weights
# ----------------------------------------------------------------------------
def j5_weights(hrs, sh, el, out, log):
    rows, inc = [], []
    for name, d, strata in (("HRS", hrs, None), ("SHARE", sh, "country"), ("ELSA", el, None)):
        ok = d["w"].notna() & (d["w"] > 0)
        if ok.sum() < 100:
            log(f"  J5 {name}: weights unavailable, skipped"); continue
        dd = d[ok].reset_index(drop=True)
        s = dd["country"] if strata else None
        fw = lifelines_cox(dd, C2 + OCC, strata=s, weights=dd["w"].values / dd["w"].mean())
        fu = lifelines_cox(dd, C2 + OCC, strata=s)
        for v in OCC:
            rows.append({"cohort": name, "term": v,
                         "HR_unweighted": np.exp(fu.params_[v]),
                         "HR_weighted": np.exp(fw.params_[v]),
                         "lo_w": np.exp(fw.confidence_intervals_.loc[v].iloc[0]),
                         "hi_w": np.exp(fw.confidence_intervals_.loc[v].iloc[1]),
                         "n_weighted": int(ok.sum())})
        for g, sub in dd.groupby("occ_group"):
            km = KaplanMeierFitter().fit(sub["time"], sub["stroke"], weights=sub["w"])
            inc.append({"cohort": name, "group": g,
                        "weighted_10y_KM_risk": float(1 - km.survival_function_at_times(10).values[0])})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "J5_weighted_HR.csv"), index=False)
    pd.DataFrame(inc).round(4).to_csv(os.path.join(out, "J5_weighted_incidence.csv"), index=False)
    log("  J5 done")


# ----------------------------------------------------------------------------
# J6 multiple imputation
# ----------------------------------------------------------------------------
def mi_datasets(d, feats, m, seed):
    out = []
    aux = d[feats + ["time"]].copy()
    aux["stroke_i"] = d["stroke"].astype(float).values
    for i in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + i)
        arr = imp.fit_transform(aux)
        di = d.copy()
        di[feats] = arr[:, :len(feats)]
        for b in ("female", "hibpe", "diabe", "hearte", "smokev", "smoken", "married") + tuple(OCC):
            if b in feats:
                di[b] = di[b].clip(0, 1).round()
        out.append(di)
    return out


def j6_mi(dev, val, hrs, sh, el, out, m, log):
    rows, dc = [], []
    for name, d, strata in (("HRS", hrs, None), ("SHARE", sh, "country"), ("ELSA", el, None)):
        imps = mi_datasets(d, C3, m, SEED)
        est = {v: [] for v in OCC}
        var = {v: [] for v in OCC}
        for di in imps:
            f = lifelines_cox(di, C3, strata=di["country"] if strata else None)
            for v in OCC:
                est[v].append(f.params_[v]); var[v].append(f.standard_errors_[v] ** 2)
        for v in OCC:
            qbar = np.mean(est[v]); ubar = np.mean(var[v]); bvar = np.var(est[v], ddof=1)
            T = ubar + (1 + 1 / m) * bvar
            rows.append({"cohort": name, "term": v, "HR_MI": np.exp(qbar),
                         "lo": np.exp(qbar - 1.96 * np.sqrt(T)), "hi": np.exp(qbar + 1.96 * np.sqrt(T))})
    # Delta-C with MI (imputation fitted on training data only for each pair)
    for name, tr, te in (("HRS (temporal)", dev, val), ("SHARE (external)", hrs, sh), ("ELSA (external)", hrs, el)):
        ds = []
        for i in range(m):
            imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=SEED + i).fit(tr[C3])
            tri, tei = tr.copy(), te.copy()
            tri[C3], tei[C3] = imp.transform(tr[C3]), imp.transform(te[C3])
            tei = clip_time(tei, tri)
            _, _, rA = cox_fit_predict(tri, tei, C2)
            _, _, rB = cox_fit_predict(tri, tei, C3)
            ytr, yte = ysurv(tri), ysurv(tei)
            ds.append(concordance_index_ipcw(ytr, yte, rB, tau=TAU)[0] - concordance_index_ipcw(ytr, yte, rA, tau=TAU)[0])
        dc.append({"cohort": name, "deltaC_MI_mean": np.mean(ds), "deltaC_MI_min": np.min(ds), "deltaC_MI_max": np.max(ds)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "J6_MI_occupation_HR.csv"), index=False)
    pd.DataFrame(dc).round(4).to_csv(os.path.join(out, "J6_MI_deltaC.csv"), index=False)
    log("  J6 done")


# ----------------------------------------------------------------------------
# J7 sex-stratified, J8 employees only
# ----------------------------------------------------------------------------
def j7_sex(dev, val, hrs, sh, el, out, n_boot, log):
    rng = np.random.default_rng(SEED)
    rows, inter = [], []
    for name, tr, te in (("HRS (temporal)", dev, val), ("SHARE (external)", hrs, sh), ("ELSA (external)", hrs, el)):
        for sex, lab in ((0.0, "Men"), (1.0, "Women")):
            sub = (te["female"] == sex).values
            if te["stroke"].values[sub].sum() < 20:
                continue
            r = delta_c(tr, te, n_boot, rng, sub=sub)
            rows.append({"cohort": name, "sex": lab, **r})
    for name, d, strata in (("HRS", hrs, None), ("SHARE", sh, "country"), ("ELSA", el, None)):
        dd = d.copy()
        for v in OCC:
            dd[f"{v}_x_female"] = dd[v] * dd["female"]
        inter_terms = [f"{v}_x_female" for v in OCC]
        f = lifelines_cox(dd, C2 + OCC + inter_terms, strata=dd["country"] if strata else None)
        f0 = lifelines_cox(dd, C2 + OCC, strata=dd["country"] if strata else None)
        lr = 2 * (f.log_likelihood_ - f0.log_likelihood_)
        from scipy.stats import chi2
        inter.append({"cohort": name, "LR_chi2": lr, "df": 3, "p_interaction": 1 - chi2.cdf(lr, 3)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "J7_sex_stratified_deltaC.csv"), index=False)
    pd.DataFrame(inter).round(4).to_csv(os.path.join(out, "J7_sex_occupation_interaction.csv"), index=False)
    log("  J7 done")


def j8_employees(dev, val, out, n_boot, log):
    rng = np.random.default_rng(SEED)
    tr, te = dev[dev["slfemp"] == 0].reset_index(drop=True), val[val["slfemp"] == 0].reset_index(drop=True)
    r = delta_c(tr, te, n_boot, rng)
    both = pd.concat([tr, te], ignore_index=True)
    f = lifelines_cox(both, C2 + OCC)
    row = {**r}
    for v in OCC:
        row[f"HR_{v}"] = np.exp(f.params_[v])
        row[f"lo_{v}"] = np.exp(f.confidence_intervals_.loc[v].iloc[0])
        row[f"hi_{v}"] = np.exp(f.confidence_intervals_.loc[v].iloc[1])
    pd.DataFrame([row]).round(4).to_csv(os.path.join(out, "J8_HRS_employees_only.csv"), index=False)
    log(f"  J8 employees-only HRS: n_test={r['n']}, Delta-C={r['delta']:+.4f}")


# ----------------------------------------------------------------------------
# J9 Fine-Gray external, J10 decision curves
# ----------------------------------------------------------------------------
def j9_finegray(hrs, sh, el, out, n_boot, log):
    rng = np.random.default_rng(SEED)
    med = hrs[C3].median()
    fits = {}
    for sname, feats in (("C2", C2), ("C3", C3)):
        X = hrs[feats].fillna(med[feats]).reset_index(drop=True)
        fg, ex = m03.fit_finegray(X, hrs["time"].values, hrs["event_cr"].values)
        fits[sname] = (fg, ex, feats)
        log(f"  J9 Fine-Gray {sname} fitted ({len(ex):,} expanded rows)")
    rows = []
    ytr = ysurv(hrs)
    for name, te in (("SHARE", sh), ("ELSA", el)):
        te = clip_time(te, hrs)
        yte = ysurv(te)
        risk, cif = {}, {}
        for sname, (fg, ex, feats) in fits.items():
            Xt = te[feats].fillna(med[feats])
            risk[sname] = fg.predict_partial_hazard(Xt).values.ravel()
            cif[sname] = m03.finegray_cif(fg, ex, Xt, 10)
        obs = aj_risk(te["time"].values, te["event_cr"].values, 10)
        cs = {s: concordance_index_ipcw(ytr, yte, risk[s], tau=TAU)[0] for s in risk}
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(te), len(te))
            try:
                diffs.append(concordance_index_ipcw(ytr, yte[idx], risk["C3"][idx], tau=TAU)[0] -
                             concordance_index_ipcw(ytr, yte[idx], risk["C2"][idx], tau=TAU)[0])
            except ValueError:
                continue
        for s in ("C2", "C3"):
            rows.append({"cohort": name, "set": s, "C_uno": cs[s], "pred_10y_CIF": cif[s].mean(),
                         "obs_10y_AJ": obs, "OE": obs / cif[s].mean()})
        rows.append({"cohort": name, "set": "Delta C3-C2", "C_uno": cs["C3"] - cs["C2"],
                     "pred_10y_CIF": np.quantile(diffs, .025), "obs_10y_AJ": np.quantile(diffs, .975),
                     "OE": np.nan})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(out, "J9_finegray_external.csv"), index=False)
    log("  J9 done (Delta rows: pred_10y_CIF/obs_10y_AJ columns hold the 95% CI)")


def j10_dca(hrs, sh, el, out, log, h=10):
    thr = np.linspace(0.005, 0.10, 40)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    rows = []
    for ax, (name, te, recal) in zip(axes, (("SHARE", sh, False), ("ELSA", el, True))):
        te = te.reset_index(drop=True)
        n = len(te)
        prev = aj_risk(te["time"].values, te["event_cr"].values, h)
        for sname, feats in (("C2", C2), ("C3", C3)):
            m, pp, _ = cox_fit_predict(hrs, te, feats)
            p = np.array([1 - f(h) for f in m.predict_survival_function(pp.transform(te[feats]))])
            if recal:   # re-estimate baseline risk in the external cohort (slope kept free)
                lp = np.log(-np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
                x = pd.DataFrame({"lp": lp, "time": te["time"], "stroke": te["stroke"].astype(int)})
                cph = CoxPHFitter().fit(x, "time", "stroke")
                p = 1 - cph.predict_survival_function(x[["lp"]], times=[h]).values.ravel()
            nb = []
            for t in thr:
                pos = p >= t
                if pos.sum() < 10:
                    nb.append(np.nan); continue
                r = aj_risk(te["time"].values[pos], te["event_cr"].values[pos], h)
                nb.append(r * pos.mean() - (1 - r) * pos.mean() * t / (1 - t))
            ax.plot(thr, nb, label=sname)
            rows += [{"cohort": name, "set": sname, "threshold": a, "net_benefit": b} for a, b in zip(thr, nb)]
        ax.plot(thr, [prev - (1 - prev) * t / (1 - t) for t in thr], "k--", lw=.8, label="Treat all")
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylim(-0.005, max(prev, 0.005) * 1.3)
        ax.set_title(f"{name}{' (recalibrated)' if recal else ''}", fontsize=9)
        ax.set_xlabel("Threshold 10-year risk", fontsize=8); ax.set_ylabel("Net benefit", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_dca_external.png"), dpi=300); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "J10_dca_external.csv"), index=False)
    log("  J10 done")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrs", default="results")
    ap.add_argument("--hrs_sav", default=None)
    ap.add_argument("--share", default="results_share/cohort_share.parquet")
    ap.add_argument("--share_dta", default=None)
    ap.add_argument("--elsa", default="results_elsa/cohort_elsa.parquet")
    ap.add_argument("--elsa_dta", default=None)
    ap.add_argument("--out", default="results_journal")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot_med", type=int, default=200)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list, e.g. J1,J4 (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run_log.txt"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    want = set(s.strip().upper() for s in a.only.split(",") if s.strip()) or \
        {f"J{i}" for i in range(1, 11)}
    dev, val, hrs, sh, el = load_all(a, log)
    steps = [("J1", lambda: j1_j3(dev, val, hrs, sh, el, a.out, a.boot, log)),
             ("J4", lambda: j4_mediation(hrs, sh, el, a.out, a.boot_med, log)),
             ("J5", lambda: j5_weights(hrs, sh, el, a.out, log)),
             ("J6", lambda: j6_mi(dev, val, hrs, sh, el, a.out, a.m, log)),
             ("J7", lambda: j7_sex(dev, val, hrs, sh, el, a.out, a.boot_med, log)),
             ("J8", lambda: j8_employees(dev, val, a.out, a.boot, log)),
             ("J9", lambda: j9_finegray(hrs, sh, el, a.out, a.boot_med, log)),
             ("J10", lambda: j10_dca(hrs, sh, el, a.out, log))]
    for key, fn in steps:
        if key in want or (key == "J1" and want & {"J2", "J3"}):
            t0 = time.time()
            log(f"== {key}")
            try:
                fn()
            except Exception as ex:  # keep going; report the failure
                log(f"[error] {key} failed: {type(ex).__name__}: {ex}")
            log(f"   ({time.time() - t0:.0f}s)")
    log("Done. Share the contents of " + a.out + " (aggregate only).")


if __name__ == "__main__":
    main()
