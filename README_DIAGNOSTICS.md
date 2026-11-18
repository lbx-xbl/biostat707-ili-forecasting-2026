# ILI v0.3 — Frozen-forecast uncertainty and mean reliability

This stage analyzes the unchanged v0.2 saved point forecasts. It does not load or retrain forecasting models, replace forecasts, or alter previous reports. Primary inputs are 15,050 rows: 3,010 paired samples × five methods from six rolling test seasons. All source attribution and revised-data caveats from DATA_SOURCES.md still apply.

## Reproduce

From the project directory, use the existing `.venv` and locked `requirements.txt` (Python 3.11.5, macOS arm64 tested):

```bash
source .venv/bin/activate
python -m pytest -q tests_diagnostics
python run_diagnostics.py --config config/diagnostics.json --output-dir outputs/diagnostics_reproduction
python verify_diagnostics.py --output-dir outputs/diagnostics_reproduction
```

Use a fresh output directory: the delivered `outputs/diagnostics_v0_3` is nonempty and will not be overwritten. For a new environment: `python3.11 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt`. The analysis needs no network or additional packages. Prior archives and all files referenced in `docs/diagnostics_preservation_baseline.json` are included in the delivery; retain the ZIP's relative folder layout to run preservation checks. Other operating systems have not been tested.

Open `outputs/diagnostics_v0_3/diagnostics_report.html`: seven embedded figures, original-scale metric-difference intervals, decomposition tables and explicit limitations. Report-only regeneration is:

```bash
MPLCONFIGDIR=.cache/diagnostics-matplotlib python -c "from diagnostics.report import render; render('outputs/diagnostics_v0_3')"
```

## Prespecified methods

Full protocol: docs/diagnostics_protocol_v0.3.md, saved before calculation; config: config/diagnostics.json. Primary moving blocks L=8 weeks, sensitivity L=4 and L=13, B=5000, seed=2026. Draw nonwrapping overlapping blocks within each season, concatenate and truncate to its original week count. All regions and models share sampled week indices. Difference = method minus Persistence: negative favors the method. Use original point differences, 95% percentile endpoints, and report replicate-mean shifts. Overall Macro-MAE averages regions equally; RMSE pools squared errors before square root. No row-wise bootstrap, region resampling, season resampling, p-values or multiplicity-adjusted significance claims.

The intervals condition on the observed six seasons, ten regions, revised data and fitted models. They exclude training/selection uncertainty and do not imply future-season coverage. Seasonal nonstationarity and shared training across folds limit validity. Noncircular block sampling underweights season edges; inclusion frequencies and centering shifts are saved and plotted. All predeclared lengths are reported, not selected after results.

Mean reliability uses nondecreasing least-squares isotonic regression with ties. With original squared loss S, diagnostic isotonic loss S_iso and outcome variance UNC (ddof=0): MCB=S-S_iso, DSC=UNC-S_iso, S=MCB-DSC+UNC. MCB is a score difference, not generally mean((forecast-isotonic)^2). Squared bias is unconditional MCB; the remainder is conditional MCB. Components are pp²; sqrt(MCB) is pp. Skill against the evaluation group's empirical mean may be negative; constant outcomes yield NA. The evaluation mean is a diagnostic reference, not an operational baseline.

These are same-sample descriptive mean diagnostics, potentially optimistic. No isotonic-calibrated replacement predictions or independent performance gains are claimed. No calibration tests or confidence bands are generated. Mean/squared-loss components do not decompose MAE; no arbitrary binary classification or AUROC/AUPRC is introduced.

## Machine-readable artifacts

- `input_predictions.csv`: byte-identical saved rolling predictions.
- `bootstrap_intervals.csv`: 408 rows, overall/season/region × four comparators × two metrics × three lengths; marginal intervals and centering shifts.
- `bootstrap_indices.npz`: each key `L8_2013-2014` etc stores B×n indices into sorted target dates for that season; indices are shared across regions/models.
- `bootstrap_replicates_L*.npz`: `differences` has axes replicate×group×comparator×metric; `scopes`, `groups`, `models`, `metrics` and original `point_estimates` define axes. All 2,040,000 difference values are saved.
- `bootstrap_week_inclusion.csv`: mean sampling multiplicity per target week and length.
- `calibration_components.csv`: 85 decompositions (overall, six seasons, ten regions × five models).
- `calibration_diagnostic_points.csv`: 45,150 rows retaining keys, forecasts, outcomes and isotonic diagnostic values for the three scopes. These diagnostic values are not new forecasting outputs.
- `run_summary.json`, `verification_summary.json`, `review_summary.txt`, `execution_log.txt`: actual environment, timing, independent checks and review.

The independent verifier uses direct index resampling rather than the producer's count matrix, reconstructs all interval endpoints and original differences, and implements a separate tie-aware weighted PAV routine to check isotonic fits. It checks every saved component, identities, frozen inputs and previous artifact hashes. Unit fixtures cover nonmonotone outcomes, exact ties, constants, bias, block contiguity, shared pairing and pooled RMSE.

## Sources and remaining work

Gneiting & Resin (2023), Electronic Journal of Statistics 17:3226–3286, https://doi.org/10.1214/23-EJS2180 (section 3.3, eqs 26–31); Künsch (1989), https://doi.org/10.1214/aos/1176347265; scikit-learn IsotonicRegression documentation. Exact source URLs and formula choices are in the protocol. No copyrighted paper PDF is bundled.

Quantile forecasts/WIS and real-time vintage reconstruction remain later stages. All existing forecast accuracy results were already viewed before this diagnostic plan; do not describe it as a fully untouched confirmatory study. Codex implemented, executed and checked this analysis and report. The team should disclose actual AI assistance and add its own reflections about understanding, benefits and limitations.
