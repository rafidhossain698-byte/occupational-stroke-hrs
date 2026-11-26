"""
07_elsa_cohort.py
External-validation cohort from the Gateway Harmonized ELSA (version H, 2002-2023; file gh_elsa_h.dta).

Run LOCALLY. Only the aggregate feasibility report is meant to be shared.

Design (variable names verified against the Gateway Harmonized ELSA H codebook):
  Baseline : ELSA Wave 2 (2004/05) -- first wave with NS-SEC occupational class and measured BMI
  Inclusion: responded (r2iwstat=1), working for pay (r2work=1), valid NS-SEC 8-class (1-7),
             aged 50-64, no stroke reported at baseline (r2stroke=0)
  Follow-up: waves 3-10
  Outcome  : first wave where r{w}stroke ("ever had stroke") becomes 1;
             event time = midpoint between previous and current interview
  Competing: death without stroke. radyear is temporarily withdrawn from the harmonized file,
             so death is taken from r{w}iwstat = 5 ("died since last wave") and placed at the
             midpoint between the last interview and that wave's median interview date.

Occupational class (NS-SEC 8-class -> approximate HRS-style levels):
  1 Higher managerial & professional            -> 1_Managerial
  2 Lower managerial & professional             -> 2_Professional
  3 Intermediate, 4 small employers/own account,
  5 lower supervisory & technical               -> 3_Routine_nonmanual
  6 Semi-routine, 7 routine                     -> 4_Service_manual
  (NS-SEC 1 mixes senior managers and higher professionals; report as an approximation.)

Usage (Mac):
  python3 07_elsa_cohort.py --dta elsa_data/gh_elsa_h.dta --out results_elsa
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

B = 2
FOLLOW_WAVES = list(range(3, 11))
STATIC = ["idauniq", "ragender", "rabyear", "raeducl"]
EVERY_WAVE = ["iwstat", "iwy", "iwm", "agey", "stroke"]
BASELINE_R = ["work", "nssec8", "hibpe", "diabe", "hearte", "mbmi", "smokev", "smoken", "drink",
              "shlt", "mstat", "jhours", "jcten", "slfemp",
              "timepre", "ltlfree", "demand", "deserve", "salary", "promot", "secure",
              "nskills", "support", "satjob", "jobsum"]
BASELINE_H = ["itot", "atotb"]
HIERARCHY = {1: "1_Managerial", 2: "2_Professional", 3: "3_Routine_nonmanual",
             4: "3_Routine_nonmanual", 5: "3_Routine_nonmanual",
             6: "4_Service_manual", 7: "4_Service_manual"}


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
    df, _ = pyreadstat.read_dta(path, usecols=cols)
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def dec_year(y, m):
    y = pd.Series(np.asarray(y, dtype=float)).where(lambda s: s.between(1990, 2030))
    m = pd.Series(np.asarray(m, dtype=float)).where(lambda s: s.between(1, 12), 6.5)
    return y.values + (m.values - 0.5) / 12.0


def build(df):
    flow = []
    # median interview date of each wave (used to place deaths reported as iwstat = 5)
    wave_mid = {}
    for w in FOLLOW_WAVES:
        if f"r{w}iwy" in df:
            t = dec_year(df[f"r{w}iwy"], df[f"r{w}iwm"])
            wave_mid[w] = np.nanmedian(t[(df[f"r{w}iwstat"] == 1).values])

    d = df.copy()
    flow.append(("All Harmonized ELSA records", len(d)))
    d = d[d[f"r{B}iwstat"] == 1];            flow.append(("Responded at Wave 2", len(d)))
    d = d[d[f"r{B}work"] == 1];              flow.append(("Working for pay at baseline", len(d)))
    d = d[d[f"r{B}nssec8"].between(1, 7)];   flow.append(("Valid NS-SEC class (1-7)", len(d)))
    d = d[d[f"r{B}agey"].between(50, 64)];   flow.append(("Aged 50-64 at baseline", len(d)))
    d = d[d[f"r{B}stroke"] == 0];            flow.append(("No stroke reported at baseline", len(d)))

    t0 = dec_year(d[f"r{B}iwy"], d[f"r{B}iwm"])
    n = len(d)
    event = np.zeros(n, dtype=int)
    t_end, last_iw = t0.copy(), t0.copy()
    done = np.zeros(n, dtype=bool)
    for w in FOLLOW_WAVES:
        if f"r{w}iwstat" not in d:
            continue
        st = d[f"r{w}iwstat"].to_numpy()
        iw = dec_year(d[f"r{w}iwy"], d[f"r{w}iwm"])
        resp = (st == 1) & ~done & ~np.isnan(iw)
        new_ev = resp & (d[f"r{w}stroke"] == 1).to_numpy()
        t_end = np.where(new_ev, (last_iw + iw) / 2.0, t_end)
        event = np.where(new_ev, 1, event)
        done |= new_ev
        upd = resp & ~new_ev
        last_iw = np.where(upd, iw, last_iw)
        t_end = np.where(upd & ~done, iw, t_end)
        died = (st == 5) & ~done & (w in wave_mid)
        if died.any():
            t_death = (last_iw + wave_mid[w]) / 2.0
            t_end = np.where(died, t_death, t_end)
            event = np.where(died, 2, event)
            done |= died

    out = pd.DataFrame({
        "id": d["idauniq"].values, "time_years": t_end - t0, "event": event,
        "age": d[f"r{B}agey"].values, "sex": d["ragender"].values, "educl": d["raeducl"].values,
        "nssec8": d[f"r{B}nssec8"].values,
    })
    for s in BASELINE_R:
        if f"r{B}{s}" in d:
            out["bmi" if s == "mbmi" else s] = d[f"r{B}{s}"].values
    for s in BASELINE_H:
        if f"h{B}{s}" in d:
            out["h_" + s] = d[f"h{B}{s}"].values
    # ELSA stores item non-response as negative codes (-1, -2, -3, -8, -9, -13, -23, ...):
    # set them to missing everywhere except wealth, which can legitimately be negative.
    for c in out.columns:
        if c not in ("id", "time_years", "event", "h_atotb") and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float).mask(out[c] < 0)   # cast first: int columns cannot hold NaN
    out["occ_hierarchy"] = out["nssec8"].map(HIERARCHY)
    out = out[out["time_years"] > 0].reset_index(drop=True)
    flow.append(("With follow-up time > 0 (final cohort)", len(out)))
    return out, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dta", required=True, help="path to gh_elsa_h.dta")
    ap.add_argument("--out", default="results_elsa")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cohort, flow = build(load(a.dta))
    cohort.to_parquet(os.path.join(a.out, "cohort_elsa.parquet"), index=False)   # row-level: keep LOCAL
    rep = cohort.rename(columns={"id": "HHIDPN"}).assign(cohort="elsa")
    b01.feasibility_report({"ELSA_external_validation": rep}, {"ELSA_external_validation": flow}, a.out)
    print(f"[ok] ELSA cohort: {len(cohort):,} people, {(cohort.event == 1).sum():,} strokes, "
          f"{(cohort.event == 2).sum():,} deaths without stroke")
    print(f"Share only: {a.out}/feasibility_report.md")


if __name__ == "__main__":
    main()
