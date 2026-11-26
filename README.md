# Regional influenza-like illness forecasting

**BIOSTAT 707 · Fall 2026 · Final analysis through v0.5**

Bowen (Bofan) Zhang · Yufei Wang · Leslie (Jiaxiang) Xue · Lanbu Xi · Michael (Shuyang) Yu

We forecast CDC ILINet population-weighted ILI visit percentages in ten HHS regions from 12 weeks of history, region indicators, and calendar features. The target is two observation weeks after the origin. This is retrospective research on 2010–2019 data, not an operational forecast service.

## Read the submission

- **Final English report:** [PDF](reports/final-report.pdf), [editable DOCX](reports/final-report.docx), or [GitHub-readable report](reports/final-report.md).
- [Original proposal](707-proposal.md), retained as submitted. The final study follows the instructor's recommendations and supersedes the proposal's planned two-season-only diagnostics.
- [Submission verification and scope](docs/submission/README.md), including what was checked in this checkout.
- [Data sources and attribution](DATA_SOURCES.md).

Presentation slides and speaker scripts are outside this update. No presentation files are included or modified.

## Main findings

| Evaluation | Sample per model | Main result |
| --- | ---: | --- |
| Fixed pilot (v0.1) | 1,020 | Ridge had lowest test Macro-MAE: 0.417, 14.9% below persistence. |
| Six-season rolling (v0.2–v0.3) | 3,010 | HGB had lowest Macro-MAE (0.320); Ridge had lowest RMSE (0.618). |
| Residual-quantile forecasts (v0.4) | 3,010 | HGB had lowest WIS (0.238), 18.6% below persistence. Its 95% interval coverage fell from 94.1% overall to 87.6% in 2017–18. |
| Historical vintages (v0.5) | 2,540 | Ridge had lowest WIS in both matched regimes. AsOf WIS was 9.5%, 10.6%, and 12.1% higher than MatchedRevised for Ridge, HGB, and LSTM. |

MAE, RMSE, and WIS are in percentage points; lower is better. These samples and training masks differ, so scores are not pooled across versions. Rankings are descriptive, not universal. Bootstrap intervals condition on the observed seasons and fitted models. The vintage schedule leaves **7 days from scheduled issuance to target week end**, despite the t+2 observation-week target.

## Completed stages and saved evidence

| Stage | Methods and reproduction notes | Saved report | Predictions and summary scores |
| --- | --- | --- | --- |
| Exploratory analysis and missing-data audit | [Script](scripts/eda_missingness.py); report section "Data, models, and chronological evaluation" | [Figure](outputs/eda/eda_seasonality.png) | [peaks by season and region](outputs/eda/season_region_peaks.csv), [summary](outputs/eda/eda_summary.json) |
| v0.1 Pilot | [Pilot specification](SPEC.md), [pilot notes](README_PILOT.md) | [HTML](outputs/demo_v0_1/demo_report.html) | [predictions](outputs/demo_v0_1/predictions.csv), [metrics](outputs/demo_v0_1/metrics.csv) |
| v0.2 Rolling origin | [README](README_ROLLING.md) | [HTML](outputs/rolling_v0_2/rolling_report.html) | [predictions](outputs/rolling_v0_2/predictions.csv), [metrics](outputs/rolling_v0_2/metrics.csv) |
| v0.3 Frozen diagnostics | [README](README_DIAGNOSTICS.md) | [HTML](outputs/diagnostics_v0_3/diagnostics_report.html) | [bootstrap intervals](outputs/diagnostics_v0_3/bootstrap_intervals.csv), [CORP](outputs/diagnostics_v0_3/calibration_components.csv) |
| v0.4 Probability forecasts | [README](README_PROBABILISTIC.md) | [HTML](outputs/probabilistic_v0_4/probabilistic_report.html) | [quantiles](outputs/probabilistic_v0_4/quantiles_wide.csv), [metrics](outputs/probabilistic_v0_4/metrics.csv) |
| v0.5 Vintage replay | [README](README_VINTAGE.md) | [HTML](outputs/vintage_v0_5/vintage_report.html) | [quantiles](outputs/vintage_v0_5/quantiles_wide.csv), [metrics](outputs/vintage_v0_5/metrics.csv) |

Download self-contained HTML files to view them in a browser; GitHub's source view does not render the reports. Standalone figures, region/season summaries, model files, selection histories, provenance, and original run/verification records remain under each output folder. Large CSV files may require downloading rather than GitHub preview.

## Verify the committed results without retraining

The analysis environment was Python 3.11.5 on macOS arm64. Dependencies are pinned in [requirements.txt](requirements.txt). A fresh installation on other machines has not been verified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests tests_rolling tests_diagnostics tests_probabilistic tests_vintage
python scripts/verify_submission.py
```

The submission verifier checks imported file hashes, preserved original repository files (except the intentionally updated README), presentation exclusion, and independently reconstructs overall point-error and WIS summaries. It does not fit models or modify frozen outputs. See [the actual checkout check record](docs/submission/checks.json) and [test log](docs/submission/tests.txt).

## Reproduce analyses into new directories

The committed data caches are sufficient; a new download is not required. Use new output directories to retain frozen evidence.

```bash
python run_demo.py --config config/demo.json --output-dir outputs/demo_reproduction
python run_rolling.py --config config/rolling.json --output-dir outputs/rolling_reproduction
python run_diagnostics.py --config config/diagnostics.json --output-dir outputs/diagnostics_reproduction
python run_probabilistic.py --config config/probabilistic.json --output-dir outputs/probabilistic_reproduction
python run_vintage.py --config config/vintage.json --output-dir outputs/vintage_reproduction
python scripts/eda_missingness.py --output-dir outputs/eda_reproduction
```

Stage configurations explicitly select their input files. The last three commands use the committed frozen inputs by default; they do not automatically consume the preceding command's new output. Adjust a copied config to chain new runs. The vintage command refits models under each data regime. Original `verify_*.py` scripts include historical archive-preservation assertions and expect the original companion archive layout; their saved verification records are retained unchanged. For this formal repository, use `scripts/verify_submission.py` as documented above. Do not run verifiers with `python -O`.

The original [data downloader](scripts/download_ili_data.py) remains available, including `--verify-only`. Downloading again can replace the snapshot with revised observations and should be done only in a separate checkout when intentionally constructing a new dataset.

Report authoring is separate from model dependencies: `scripts/build_final_report.py` requires `python-docx` and rebuilds the DOCX from `reports/source/report_content.json`. The PDF was rendered and visually checked using LibreOffice; reports/source/facts.json retains primary-table source rows and hashes. Model artifacts use Python serialization; load only trusted files.

## Limitations and AI disclosure

The study contains six historical outer seasons, one small LSTM configuration/seed, dependent regional series, and retrospective protocol development after viewing the pilot. Residual distributions pool regions and reuse the validation season used for model selection. Historical vintage gaps exclude 15.6% of the original test windows, and release dates approximate rather than prove intraday availability. Full limitations and conditional interpretations are in the final report.

OpenAI Codex assisted with code, data retrieval, training workflows, debugging, automated checks, statistical summaries, and writing. AI also prepared this repository update. This is substantive assistance, not language editing alone. Automated review is not independent human validation; authors remain responsible for the claims. Individual contributions and personal learning outcomes are not fabricated by the repository update.

Data are CDC ILINet via [CMU Delphi](https://cmu-delphi.github.io/delphi-epidata/api/fluview.html); Delphi identifies the source as public-domain U.S. Government data. This student project is not endorsed by CDC, HHS, or the U.S. Government.
