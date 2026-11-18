# ILI v0.5 — Historical vintage replay and matched revised-data control

This stage actually downloaded **212,710 Delphi ILINet vintage records**, reconstructed availability by `release_date`, and fit the existing five-method pipeline under two data regimes across six outer seasons. All v0.1–v0.4 files and archives are preserved byte-for-byte. New results are in `outputs/vintage_v0_5/vintage_report.html` (self-contained English report, Chinese interpretation, six figures).

## What “real time” means here

For an input origin week ending Saturday t, issue the forecast at **end of Saturday t+1** (seven days later), using only vintage records with `release_date <= issuance_date`. Select the latest eligible release for each region/observation week; break same-day ties by largest issue. The target remains t+2, matching earlier experiments. Consequently the target week ends **seven days after scheduled issuance**, not fourteen. This is a two-observation-week forecast, not a two-week release-time horizon.

The archive has daily release dates, not complete intraday availability or a record of historical service outages. The exercise is an archive-available information replay. It cannot prove exactly which value was visible to every user at every instant. Early summer observations were sometimes first entered much later; future values are never used to fill those gaps.

The first scheduled issuance of the preceding validation season fixes the selection training snapshot. Validation features use their own issuance vintages; validation labels use versions available at the next outer season's initial fit date. At that outer cutoff select the original Ridge grid / LSTM epoch and refit on eligible prior history. Scalers use only unique observations available in the appropriate training period. Historic training inputs may include revisions already available when fitting, which is intentional.

## Controlled comparison and eligibility

- **AsOf**: available values/labels at the relevant historical cutoffs; no test-target access during fitting, selection or residual estimation.
- **MatchedRevised**: identical train/validation/refit/test sample IDs, scaler-observation IDs, seasonal-reference masks, issuance schedule, feature definitions, seeds and hyperparameter rules, substituting values from the frozen 2026-10-20 reference snapshot. Parameters are chosen separately, so this compares the complete data-version-dependent pipeline. It is a hindsight counterfactual, not deployable.
- **OriginalFrozen**: v0.2 point / v0.4 probability outputs restricted to the same test IDs. Its historical training masks differ, so it is context rather than the controlled vintage comparison.

All models and regimes share complete ten-region origin groups. Require all twelve history weeks, required pre-test labels, and an available historical same-week Seasonal reference (only the previous week53 fallback is permitted). No arbitrary interpolation or final-data imputation. Six seasonal test counts per region are **20,47,46,48,46,47**, giving **2,540 samples per model/regime**, or 84.4% of the original 3,010. The first season is especially restricted; report the conditional evaluation population rather than claiming complete real-time coverage of all original dates. `eligibility.csv` records every candidate and exclusion reason.

## Actual comparison on the common 2,540 samples

| Method | MatchedRevised MAE | AsOf MAE | MatchedRevised WIS | AsOf WIS |
|---|---:|---:|---:|---:|
| Persistence | 0.416558 | 0.473800 | 0.302637 | 0.333531 |
| Seasonal | 0.547944 | 0.549242 | 0.401682 | 0.403149 |
| Ridge | 0.344202 | 0.383772 | 0.250554 | 0.274366 |
| HGB | 0.335766 | 0.379331 | 0.251435 | 0.278031 |
| LSTM | 0.342317 | 0.390641 | 0.253095 | 0.283661 |

Units are percentage points. AsOf Ridge has the lowest WIS and point RMSE; HGB has the lowest point MAE. Ridge/HGB/LSTM WIS increases about 9.5%/10.6%/12.1% relative to the matched revised controls. These overall changes do not mean every season or metric worsens. Do not compare the new ranking directly against the v0.4 full-sample ranking as an isolated vintage effect: the eligible sample and historical training masks also changed. Both controlled groups favor Ridge on overall WIS here.

Probability forecasts use the same fixed 23-level residual-quantile wrapper as v0.4, estimated from each regime's preceding validation predictions. AsOf residual labels are available at the outer fit date. Validation selection reuse, changing refit training size, pooled residual assumptions and sparse dependent tails remain limitations; no conformal or independent-calibration guarantee is claimed. WIS, point errors, median errors, all eleven interval coverage/width summaries are provided overall/by season/by region. Evaluation truth is the same frozen revised snapshot for every regime, not an immutable biological ground truth.

No new bootstrap confidence intervals are produced because missing-availability dates can create gaps; old consecutive-week block indices would not describe this sample. Differences are descriptive, not significance tests or evidence of universal model superiority. Earlier point results had already been viewed, so this remains exploratory retrospective work.

## Reproduce

Use the packaged cache; no network is required for the analysis or verifier. Existing Python 3.11.5 macOS arm64 environment, maximum four compute threads, seed2026; original seven Ridge penalties, fixed HGB and LSTM maximum60 epochs unchanged. On a fresh machine install the existing requirements in a virtual environment first; fresh-machine installation is not tested in this run.

```bash
.venv/bin/python -m pytest -q tests tests_rolling tests_diagnostics tests_probabilistic tests_vintage
.venv/bin/python run_vintage.py --config config/vintage.json --output-dir outputs/vintage_reproduction
.venv/bin/python verify_vintage.py --output-dir outputs/vintage_reproduction
```

Use a fresh output directory. Verification uses assertions: do not run with Python `-O`. Keep prior archives beside the project folder as packaged for relative preservation checks. The actual model run completed in 130.47 seconds with exit0; verification/report/package time is additional.

`vintage/download.py` records URLs, UTC retrieval times, result codes, hashes and compressed raw bytes. It uses system curl with TLS verification and two workers. The official anonymous API allows two multiselect parameters, so each query has one region, an observation-week range and an issue range. Ten permitted queries retrieved the full archive; no key was needed and rate limits were not bypassed. Initial three-multiselect401 and local urllib certificate errors were diagnosed; no global dependency/security setting was changed. Cached download files are reused by default. The module also regenerates normalized CSV from raw responses; exact decompressed CSV bytes were checked against the analysis input.

```bash
.venv/bin/python -m vintage.download
```

Do not delete the supplied raw cache just to rerun the analysis. Current upstream API availability may change; V3 no longer collects new observations, but returned the requested historical records during this run.

## Inspection artifacts

- `data/vintage/raw/*.json.gz`, companion metadata and download manifest: actual upstream responses, including issues and release dates.
- `data/vintage/records.csv.gz`: 212,710 normalized version records; raw-to-normalized conversion is reproducible.
- `asof_snapshots/`: every historical cutoff used in feature/training construction.
- `folds/<season>/<regime>/`: train/validation/refit/test windows with individual feature-week/issue/release provenance and label cutoffs; selected histories and scalers; model choices/curves; saved selection and final models; validation predictions, residual offsets and label-free frozen forecasts.
- `eligibility.csv`, `availability_summary.csv`: all candidates, included masks, exclusion reasons and stage/season counts.
- `archive_input_hashes.json`, frozen config/protocol, `forecast_freeze.json`: fixed-input and pre-evaluation forecast hashes.
- `forecasts_before_evaluation.csv`, `quantiles_wide.csv`, `scores.csv`: point/quantile forecasts and scored outcomes, including OriginalFrozen context. **Row orders differ across tables; join by sample_id, model and regime, not row position.**
- `metrics.csv`, `coverage.csv`, `asof_minus_revised.csv`, `feature_revision_summary.csv`: grouped scores, coverage/width and matched differences.
- `vintage_report.html`, `figures/`: self-contained report plus six standalone PNGs.
- `run_summary.json`, `verification_summary.json`, `review_summary.txt`, `execution_log.txt`: actual run, independent checks and review. ZIP verification and SHA-256 are saved beside the archive.

The independent verifier rebuilds snapshots directly from raw releases, reconstructs eligibility for included/excluded origins, checks every window's provenance and scaler masks, reloads all saved selection/refit models, independently interpolates residual quantiles, computes pinball WIS and grouped comparisons, and checks every prior file hash. A CSV row-order assumption in the first verifier was found and fixed using explicit key alignment and three regression tests; forecast values/models were unchanged. A new test filename collision was fixed by renaming only that new test file. Both diagnostic logs are retained.

## Sources and AI use

- Delphi vintage semantics: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
- ILINet source/report times: https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html
- Anonymous limits: https://cmu-delphi.github.io/delphi-epidata/api/api_keys.html
- CDC ILINet via Delphi: public-domain source attribution in DATA_SOURCES.md; existing WIS references remain in prior reports.

Codex designed and implemented this replay, downloaded data, fitted models, ran tests/verification, interpreted results and prepared reports. Team members should disclose this scope and provide their own account of what they independently checked and understood; no personal reflections are invented. AI accelerated implementation but does not remove uncertainty about archive completeness, methodological choices or coding errors.
