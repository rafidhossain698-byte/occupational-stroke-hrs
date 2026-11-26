"""
05_share_cohort.py
External-validation cohort from the Gateway Harmonized SHARE (version G, release 9.0.0).

Run LOCALLY. Only the aggregate feasibility report is meant to be shared.

Design (variable names verified against the Gateway Harmonized SHARE G 2004-2022 codebook):
  Baseline : SHARE Wave 1 (2004/05)
  Inclusion: responded (r1iwstat=1), working for pay (r1work=1),
             valid ISCO-88 major group 1-9 (armed forces excluded),
             aged 50-64, no stroke reported at baseline (r1stroke=0)
  Follow-up: waves 2, 4, 5, 6, 7, 8, 9 (wave 3 = SHARELIFE, no health module)
  Outcome  : first wave where r{w}stroke ("ever had stroke") becomes 1;
             event time = midpoint between previous and current interview (decimal years)
  Competing: death without stroke (radyear/radmonth)
  Note     : in Wave 1 SHARE did not ask self-employed respondents their occupation,
             so the cohort is effectively employees/civil servants.

Occupational class (ISCO-88 major groups -> same 4 levels as the HRS analysis):
  1 Legislators/senior officials/managers -> 1_Managerial
  2 Professionals                         -> 2_Professional
  3 Technicians/associate prof., 4 Clerks -> 3_Routine_nonmanual
  5-9 Service/sales, agriculture, craft, operators, elementary -> 4_Service_manual

Usage (Mac):
  python3 05_share_cohort.py --dta share_data/gh/GH_SHARE_g.dta --out results_share
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("b01", os.path.join(HERE, "01_build_cohort.py"))
b01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b01)

FOLLOW_WAVES = [2, 4, 5, 6, 7, 8, 9]
B = 1  # baseline wave

STATIC = ["mergeid", "country", "ragender", "rabyear", "raedyrs", "raeducl", "radyear", "radmonth"]
EVERY_WAVE = ["iwstat", "iwy", "iwm", "agey", "stroke"]
BASELINE_R = ["work", "jisco", "hibpe", "diabe", "hearte", "hchole", "bmi", "smokev", "smoken",
              "drinkx", "vgactx", "shlt", "mstat", "jhours", "jcten", "slfemp", "jsprvs",
              # job quality / strain items (SHARE-specific)
              "timepre", "ltlfree", "demand", "deserve", "salary", "promot", "secure",
              "nskills", "support", "satjob", "jobsum"]
BASELINE_H = ["itot", "atotb"]

HIERARCHY = {1: "1_Managerial", 2: "2_Professional", 3: "3_Routine_nonmanual",
             4: "3_Routine_nonmanual", **{k: "4_Service_manual" for k in range(5, 10)}}


def wanted():
    cols = list(STATIC)
    for w in [B] + FOLLOW_WAVES:
        cols += [f"r{w}{s}" for s in EVERY_WAVE]
    cols += [f"r{B}{s}" for s in BASELINE_R] + [f"h{B}{s}" for s in BASELINE_H]
    return cols


def load(path):
    import pyreadstat
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    have = set(meta.column_names)
    cols = [c for c in wanted() if c in have]
    miss = [c for c in wanted() if c not in have]
    if miss:
        print(f"[warn] {len(miss)} expected columns not found, skipped: {miss}")
    df, _ = pyreadstat.read_dta(path, usecols=cols)   # Stata extended missings (.a, .d ...) -> NaN
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def dec_year(y, m):
    """Decimal year at mid-month; month missing -> mid-year."""
    m = pd.Series(np.asarray(m, dtype=float)).where(lambda s: s.between(1, 12), 6.5)
    return np.asarray(y, dtype=float) + (np.asarray(m, dtype=float) - 0.5) / 12.0


def build(df):
    flow = []
    d = df.copy()
    flow.append(("All Harmonized SHARE records", len(d)))
    d = d[d[f"r{B}iwstat"] == 1];            flow.append(("Responded at Wave 1", len(d)))
    d = d[d[f"r{B}work"] == 1];              flow.append(("Working for pay at baseline", len(d)))
    d = d[d[f"r{B}jisco"].between(1, 9)];    flow.append(("Valid ISCO-88 major group (1-9)", len(d)))
    d = d[d[f"r{B}agey"].between(50, 64)];   flow.append(("Aged 50-64 at baseline", len(d)))
    d = d[d[f"r{B}stroke"] == 0];            flow.append(("No stroke reported at baseline", len(d)))

    t0 = dec_year(d[f"r{B}iwy"], d[f"r{B}iwm"])
    n = len(d)
    event = np.zeros(n, dtype=int)
    t_end = t0.copy()
    last_iw = t0.copy()
    done = np.zeros(n, dtype=bool)
    for w in FOLLOW_WAVES:
        if f"r{w}iwstat" not in d:
            continue
        iw = dec_year(d[f"r{w}iwy"], d[f"r{w}iwm"])
        resp = (d[f"r{w}iwstat"] == 1).to_numpy() & ~done & ~np.isnan(iw)
        stroke = (d[f"r{w}stroke"] == 1).to_numpy()
        new_ev = resp & stroke
        t_end = np.where(new_ev, (last_iw + iw) / 2.0, t_end)
        event = np.where(new_ev, 1, event)
        done |= new_ev
        upd = resp & ~new_ev
        last_iw = np.where(upd, iw, last_iw)
        t_end = np.where(upd & ~done, iw, t_end)

    death = dec_year(d["radyear"], d["radmonth"])
    died = (~done) & ~np.isnan(death) & (death > last_iw)
    t_end = np.where(died, death, t_end)
    event = np.where(died, 2, event)

    out = pd.DataFrame({
        "id": d["mergeid"].values, "country": d["country"].values,
        "time_years": t_end - t0, "event": event,
        "age": d[f"r{B}agey"].values,
        "sex": d["ragender"].values,               # 1 man, 2 woman (same coding as RAND HRS)
        "edyrs": d["raedyrs"].values, "educl": d["raeducl"].values,
        "jisco": d[f"r{B}jisco"].values,
    })
    for s in BASELINE_R:
        if f"r{B}{s}" in d:
            out[s] = d[f"r{B}{s}"].values
    for s in BASELINE_H:
        if f"h{B}{s}" in d:
            out["h_" + s] = d[f"h{B}{s}"].values
    out["occ_hierarchy"] = out["jisco"].map(HIERARCHY)
    # income/wealth: within-country percentile ranks (currencies and price levels differ)
    for s in ("h_itot", "h_atotb"):
        if s in out:
            out[s + "_pct"] = out.groupby("country")[s].rank(pct=True)
    out = out[out["time_years"] > 0].reset_index(drop=True)
    flow.append(("With follow-up time > 0 (final cohort)", len(out)))
    return out, flow


def country_table(d, outdir):
    rows = []
    for c, sub in d.groupby("country"):
        ev = int((sub.event == 1).sum())
        rows.append({"country_code": c, "n": b01.suppress(len(sub)), "strokes": b01.suppress(ev),
                     "median_followup_y": round(sub.time_years.median(), 1)})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "share_by_country.csv"), index=False)


def missing_table(d, outdir):
    m = d.drop(columns=["id"]).isna().mean().mul(100).round(1).sort_values(ascending=False)
    m.rename("pct_missing").to_csv(os.path.join(outdir, "share_missingness.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dta", required=True, help="path to GH_SHARE_g.dta")
    ap.add_argument("--out", default="results_share")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.dta)
    cohort, flow = build(df)
    cohort.to_parquet(os.path.join(a.out, "cohort_share.parquet"), index=False)  # row-level: keep LOCAL
    rep = cohort.rename(columns={"id": "HHIDPN"}).assign(cohort="share").drop(columns=["country"])
    b01.feasibility_report({"SHARE_external_validation": rep},
                           {"SHARE_external_validation": flow}, a.out)
    country_table(cohort, a.out)
    missing_table(cohort, a.out)
    print(f"[ok] SHARE cohort: {len(cohort):,} people, {(cohort.event == 1).sum():,} strokes, "
          f"{(cohort.event == 2).sum():,} deaths without stroke")
    print(f"Share these files only: {a.out}/feasibility_report.md, share_by_country.csv, share_missingness.csv")


if __name__ == "__main__":
    main()
