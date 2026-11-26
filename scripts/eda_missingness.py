"""Exploratory analysis and missing-data audit of the committed ILINet snapshot.

Reads only data/normalized.csv and writes to outputs/eda/ (or --output-dir).
It fits no models and does not change any frozen forecast outputs.

    python scripts/eda_missingness.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
# Approximate MMWR week-of-season positions for month labels (week 40 = 0).
MONTH_TICKS = [0, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48]
MONTH_LABELS = ["Oct", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def load(path):
    d = pd.read_csv(path)
    year, week = d.epiweek // 100, d.epiweek % 100
    start = np.where(week >= 40, year, year - 1)
    d["season"] = [f"{s}–{str(s + 1)[-2:]}" for s in start]
    d["region_num"] = d.region.str.replace("hhs", "").astype(int)
    d = d.sort_values(["region_num", "epiweek"]).reset_index(drop=True)
    d["season_week"] = d.groupby(["region", "season"]).cumcount()
    return d


def missingness(d):
    weeks = sorted(d.epiweek.unique())
    per_region = d.groupby("region").epiweek.nunique()
    week_end = pd.to_datetime(d.week_end)
    gaps = d.assign(week_end=week_end).groupby("region").week_end.apply(
        lambda s: sorted(set(s.sort_values().diff().dropna().dt.days))
    )
    blanks = d.isna().sum()
    return {
        "rows": int(len(d)),
        "regions": int(d.region.nunique()),
        "weeks": int(len(weeks)),
        "first_week": int(weeks[0]),
        "last_week": int(weeks[-1]),
        "expected_rows": int(d.region.nunique() * len(weeks)),
        "weeks_per_region": {k: int(v) for k, v in per_region.items()},
        "week_gaps_days": {k: [int(x) for x in v] for k, v in gaps.items()},
        "duplicate_region_weeks": int(d.duplicated(["region", "epiweek"]).sum()),
        "missing_wili": int(d.wili.isna().sum()),
        "zero_or_negative_wili": int((d.wili <= 0).sum()),
        "columns_entirely_blank": [c for c, n in blanks.items() if n == len(d)],
        "issue_range": [int(d.issue.min()), int(d.issue.max())],
        "lag_weeks_range": [int(d.lag.min()), int(d.lag.max())],
    }


def summaries(d):
    peaks = d.loc[d.groupby(["season", "region"]).wili.idxmax()]
    peak_table = peaks.pivot(index="season", columns="region_num", values="wili")
    by_season = peaks.groupby("season").wili.agg(["median", "min", "max"])
    wide = d.pivot(index="epiweek", columns="region", values="wili")[REGIONS]
    corr = wide.corr().to_numpy()[np.triu_indices(10, 1)]
    offseason = d[(d.epiweek % 100 >= 20) & (d.epiweek % 100 <= 39)].wili
    q = d.wili.quantile([0.25, 0.5, 0.75])
    return peaks, peak_table, {
        "wili_median": round(float(q[0.5]), 2),
        "wili_iqr": [round(float(q[0.25]), 2), round(float(q[0.75]), 2)],
        "wili_range": [round(float(d.wili.min()), 3), round(float(d.wili.max()), 2)],
        "offseason_mean_weeks_20_39": round(float(offseason.mean()), 2),
        "peak_season_week_median": float(peaks.season_week.median()),
        "share_peaks_season_weeks_10_25": round(float(peaks.season_week.between(10, 25).mean()), 3),
        "median_regional_peak_by_season": {k: round(float(v), 2) for k, v in by_season["median"].items()},
        "region_mean_wili": {k: round(float(v), 2) for k, v in d.groupby("region_num").wili.mean().items()},
        "between_region_corr_median": round(float(np.median(corr)), 2),
        "between_region_corr_min": round(float(corr.min()), 2),
        "lag2_autocorr_by_region": {r: round(float(wide[r].autocorr(2)), 2) for r in REGIONS},
    }


def figure(d, peak_table, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    curves = d.groupby(["season", "season_week"]).wili.median().reset_index()
    highlight = {"2017–18": "#B22222", "2011–12": "#1F5FA8"}
    for s in sorted(d.season.unique()):
        g = curves[curves.season == s]
        if s in highlight:
            a.plot(g.season_week, g.wili, color=highlight[s], lw=2.2, label=s, zorder=3)
        else:
            a.plot(g.season_week, g.wili, color="#9A9A9A", lw=1, alpha=0.8, zorder=1)
    a.plot([], [], color="#9A9A9A", lw=1, label="Other seasons")
    a.set_xticks(MONTH_TICKS)
    a.set_xticklabels(MONTH_LABELS)
    a.set_xlim(0, 52)
    a.set_ylabel("wILI (%), median across 10 HHS regions")
    a.set_title("A. Seasonal curves, 2010–11 to 2018–19", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9)
    a.spines[["top", "right"]].set_visible(False)

    b = ax[1]
    im = b.imshow(peak_table.to_numpy(), cmap="Reds", aspect="auto")
    b.set_xticks(range(10))
    b.set_xticklabels([str(c) for c in peak_table.columns])
    b.set_yticks(range(len(peak_table)))
    b.set_yticklabels(peak_table.index)
    b.set_xlabel("HHS region")
    b.set_title("B. Peak wILI (%) by season and region", loc="left", fontsize=11)
    for i in range(peak_table.shape[0]):
        for j in range(peak_table.shape[1]):
            v = peak_table.iat[i, j]
            b.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                   color="white" if v > 7 else "black")
    fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "normalized.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "eda"))
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = load(args.data)
    miss = missingness(d)
    assert miss["rows"] == miss["expected_rows"], "Incomplete region-week panel"
    assert miss["missing_wili"] == 0 and miss["duplicate_region_weeks"] == 0
    peaks, peak_table, summ = summaries(d)

    figure(d, peak_table, out / "eda_seasonality.png")
    peak_table.round(3).to_csv(out / "season_region_peaks.csv")
    peaks[["season", "region", "epiweek", "season_week", "wili"]].to_csv(
        out / "season_region_peak_weeks.csv", index=False)
    (out / "eda_summary.json").write_text(json.dumps({"missingness": miss, "summary": summ}, indent=2,
                                                     ensure_ascii=False))
    print(json.dumps({"missingness": miss, "summary": summ}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
