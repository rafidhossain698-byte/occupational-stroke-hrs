"""
01_build_cohort.py
Occupational hierarchy & incident stroke -- cohort construction from the
RAND HRS Longitudinal File 2022 (V2).

Run this LOCALLY. Never upload the HRS data (or row-level outputs) to any AI tool:
HRS's AI/LLM policy forbids it. Only the aggregate feasibility report
(with small cells suppressed) is meant to be shared with collaborators.

Design (all variable names verified against the RAND HRS 2022 V2 codebook):
  Development cohort : HRS cohort   (HACOHORT=3),      baseline = Wave 1 (1992)
  Temporal validation: CODA + War Babies (HACOHORT=2,4), baseline = Wave 4 (1998)
  Both baselines use the same occupation coding (RwJCOCC, 1980 Census major groups).

  Inclusion at baseline: responded (RbIWSTAT=1), working for pay (RbWORK=1),
  valid occupation (RbJCOCC 1-16; Armed Forces excluded), aged 50-64,
  no prior stroke (RbSTROKE=0).

  Outcome: first follow-up wave where RwSTROKE ("ever had stroke") becomes 1.
           Event time = midpoint between the previous and current interview dates.
  Competing event: death without a reported stroke (RADDATE).
  Censoring: last interview with response.

Usage:
  pip install pandas numpy pyreadstat pyarrow
  python 01_build_cohort.py --sav path/to/randhrs1992_2022v2.sav --out results/
"""
import argparse
import os

import numpy as np
import pandas as pd

N_WAVES = 16
SMALL_CELL = 11  # suppress counts below this in shareable outputs

# --- time-invariant variables ------------------------------------------------
STATIC = ["HHIDPN", "HACOHORT", "RABYEAR", "RASEX", "RARACEM", "RAHISPAN",
          "RAEDYRS", "RAEDUC", "RADDATE"]

# --- per-wave variables (prefix R/H + wave + stem) ----------------------------
WAVE_STEMS_R = ["IWSTAT", "IWEND", "AGEY_B", "STROKE"]           # needed every wave
BASE_STEMS_R = [                                                 # needed at baseline only
    "WORK", "JCOCC",                                              # occupation
    "JSTRES", "JPHYS", "JLIFT", "JSTOOP", "JHOURS", "JCTEN",      # work characteristics
    "SLFEMP", "UNION", "FSIZE",
    "HIBPE", "DIABE", "HEARTE", "BMI",                           # clinical
    "SMOKEV", "SMOKEN", "DRINK", "VIGACT", "SHLT", "PSYCHE",     # lifestyle / health
    "MSTAT", "CENREG", "PROXY",                                  # social
]
BASE_STEMS_H = ["ITOT", "ATOTB"]                                 # household income, wealth

COHORTS = {
    "development": {"hacohort": [3], "baseline_wave": 1},
    "temporal_validation": {"hacohort": [2, 4], "baseline_wave": 4},
}

# 1980 Census major occupation groups (RwJCOCC codes) -> hierarchy level
HIERARCHY = {
    1: "1_Managerial",
    2: "2_Professional",
    3: "3_Routine_nonmanual", 4: "3_Routine_nonmanual",
    **{k: "4_Service_manual" for k in range(5, 17)},
}


def needed_columns():
    cols = list(STATIC)
    for w in range(1, N_WAVES + 1):
        cols += [f"R{w}{s}" for s in WAVE_STEMS_R]
    for b in {c["baseline_wave"] for c in COHORTS.values()}:
        cols += [f"R{b}{s}" for s in BASE_STEMS_R]
        cols += [f"H{b}{s}" for s in BASE_STEMS_H]
    return sorted(set(cols))


def load_sav(path):
    import pyreadstat
    cols = needed_columns()
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    missing = [c for c in cols if c not in meta.column_names]
    if missing:
        print(f"[warn] {len(missing)} expected columns not in file, skipped: {missing[:10]}...")
    cols = [c for c in cols if c in meta.column_names]
    df, _ = pyreadstat.read_sav(path, usecols=cols, user_missing=False)
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def build_one(df, name, hacohort, b):
    """Return (analysis dataframe, flow list) for one cohort."""
    flow = []
    d = df[df["HACOHORT"].isin(hacohort)].copy()
    flow.append(("In sample cohort(s) " + str(hacohort), len(d)))

    d = d[d[f"R{b}IWSTAT"] == 1];             flow.append((f"Responded at wave {b}", len(d)))
    d = d[d[f"R{b}WORK"] == 1];               flow.append(("Working for pay at baseline", len(d)))
    d = d[d[f"R{b}JCOCC"].between(1, 16)];    flow.append(("Valid civilian occupation code", len(d)))
    age = d[f"R{b}AGEY_B"]
    d = d[age.between(50, 64)];               flow.append(("Aged 50-64 at baseline", len(d)))
    d = d[d[f"R{b}STROKE"] == 0];             flow.append(("No stroke reported at baseline", len(d)))

    t0 = d[f"R{b}IWEND"]                      # SAS date (days since 1960-01-01)
    event = np.zeros(len(d), dtype=int)
    t_end = t0.copy().astype(float)
    last_iw = t0.copy().astype(float)
    done = np.zeros(len(d), dtype=bool)

    for w in range(b + 1, N_WAVES + 1):
        resp = (d[f"R{w}IWSTAT"] == 1).to_numpy() & ~done
        iw = d[f"R{w}IWEND"].to_numpy(dtype=float)
        stroke = (d[f"R{w}STROKE"] == 1).to_numpy()
        new_ev = resp & stroke & ~np.isnan(iw)
        # midpoint between previous interview and this one
        t_end = np.where(new_ev, (last_iw + iw) / 2.0, t_end)
        event = np.where(new_ev, 1, event)
        done |= new_ev
        upd = resp & ~new_ev & ~np.isnan(iw)
        last_iw = np.where(upd, iw, last_iw)
        t_end = np.where(upd & ~done, iw, t_end)

    # competing death: died after last stroke-free interview, no stroke reported
    death = d["RADDATE"].to_numpy(dtype=float)
    died = (~done) & ~np.isnan(death) & (death > last_iw)
    t_end = np.where(died, death, t_end)
    event = np.where(died, 2, event)

    d["time_years"] = (t_end - t0.to_numpy(dtype=float)) / 365.25
    d["event"] = event  # 0 censored, 1 stroke, 2 death without stroke
    d = d[d["time_years"] > 0]
    flow.append(("With follow-up time > 0 (final cohort)", len(d)))

    # ---- tidy baseline predictors (generic names, wave stripped) ----
    out = pd.DataFrame({"HHIDPN": d["HHIDPN"], "cohort": name,
                        "time_years": d["time_years"], "event": d["event"]})
    out["age"] = d[f"R{b}AGEY_B"]
    for s in ["RASEX", "RARACEM", "RAHISPAN", "RAEDYRS", "RAEDUC"]:
        out[s.lower()[2:]] = d[s]
    for s in BASE_STEMS_R:
        if f"R{b}{s}" in d:
            out[s.lower()] = d[f"R{b}{s}"]
    for s in BASE_STEMS_H:
        if f"H{b}{s}" in d:
            out["h_" + s.lower()] = d[f"H{b}{s}"]

    out["occ_hierarchy"] = out["jcocc"].map(HIERARCHY)
    out.loc[~out["jstres"].between(1, 4), "jstres"] = np.nan   # 5 = does not apply
    out["high_job_stress"] = (out["jstres"] <= 2).astype(float).where(out["jstres"].notna())
    out["log_income"] = np.log1p(out["h_itot"].clip(lower=0))
    out["asinh_wealth"] = np.arcsinh(out["h_atotb"] / 1000.0)
    return out, flow


def suppress(n):
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else n


def feasibility_report(cohorts, flows, outdir):
    lines = ["# Feasibility report (aggregate only, cells <%d suppressed)\n" % SMALL_CELL]
    for name, d in cohorts.items():
        lines.append(f"\n## {name}\n\n### Cohort flow\n")
        for step, n in flows[name]:
            lines.append(f"- {step}: {n:,}")
        fu = d["time_years"]
        lines.append(f"\n### Follow-up\n- Median follow-up: {fu.median():.1f} years "
                     f"(IQR {fu.quantile(.25):.1f}-{fu.quantile(.75):.1f}, max {fu.max():.1f})")
        lines.append(f"- Incident strokes: {suppress(int((d.event == 1).sum()))}")
        lines.append(f"- Deaths without stroke (competing): {suppress(int((d.event == 2).sum()))}")
        lines.append(f"- Censored: {suppress(int((d.event == 0).sum()))}")
        for h in (5, 10, 20):
            n_ev = int(((d.event == 1) & (d.time_years <= h)).sum())
            lines.append(f"- Strokes within {h} years: {suppress(n_ev)}")

        lines.append("\n### By occupational hierarchy\n")
        lines.append("| Group | N | Strokes | Crude rate /1000 person-years |")
        lines.append("|---|---|---|---|")
        for g, sub in d.groupby("occ_hierarchy"):
            n_ev = int((sub.event == 1).sum())
            rate = 1000 * n_ev / sub.time_years.sum()
            lines.append(f"| {g} | {suppress(len(sub))} | {suppress(n_ev)} | "
                         f"{rate:.2f} |" if n_ev >= SMALL_CELL else
                         f"| {g} | {suppress(len(sub))} | {suppress(n_ev)} | suppressed |")

        lines.append("\n### Missingness at baseline (%)\n")
        miss = d.drop(columns=["HHIDPN", "cohort"]).isna().mean().mul(100).round(1)
        lines.append("| Variable | % missing |\n|---|---|")
        for v, m in miss.sort_values(ascending=False).items():
            lines.append(f"| {v} | {m} |")
    path = os.path.join(outdir, "feasibility_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_sav(a.sav)
    cohorts, flows = {}, {}
    for name, cfg in COHORTS.items():
        cohorts[name], flows[name] = build_one(df, name, cfg["hacohort"], cfg["baseline_wave"])
        # row-level file stays on your machine only
        cohorts[name].to_parquet(os.path.join(a.out, f"cohort_{name}.parquet"), index=False)
        print(f"[ok] {name}: {len(cohorts[name]):,} people, "
              f"{int((cohorts[name].event == 1).sum()):,} strokes")
    feasibility_report(cohorts, flows, a.out)


if __name__ == "__main__":
    main()
