# ILI v0.2 — Six-season rolling evaluation

This is the approved first extension following professor feedback. The original v0.1 pilot remains unchanged. Read `docs/rolling_evaluation_design_v0.2.md` for the approved protocol and `config/rolling.json` for frozen settings.

## Run

Use the existing `.venv` (Python 3.11.5 on macOS arm64, pinned in requirements.txt). From the project directory:

```bash
source .venv/bin/activate
python -m pytest -q tests tests_rolling
python run_rolling.py --config config/rolling.json --output-dir outputs/rolling_v0_2
python verify_rolling.py --output-dir outputs/rolling_v0_2
```

The delivered directory contains real results. For reproduction choose a fresh output directory, e.g. `outputs/rolling_reproduction`; existing nonempty directories are rejected. From scratch, `python3.11 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt` provision the pinned dependencies. No additional packages are needed for v0.2. Other platforms are untested. Run scripts from the project root. No network is used during analysis: the exact supplied historical snapshot is required.

Open `outputs/rolling_v0_2/rolling_report.html`. All five figures are embedded. The report retains pilot results separately, shows test-key coverage differences, and reports overall, region, season and region×season metrics. `predictions.csv` has one row per model/outer sample; `metrics.csv` includes MAE, RMSE, bias (prediction minus observation), negative counts and improvement versus both baselines. Macro-MAE weights regions equally; pooled quantities weight observations equally. The 53-week season therefore contributes an additional eligible week. Macro bias is the mean of regional biases.

## What changed

- Six outer seasons 2013–14 to 2018–19, five methods: Persistence, Seasonal, Ridge, HGB, LSTM.
- Each fold fits candidate preprocessing/weights before the preceding validation season; validation selects Ridge alpha and LSTM epoch. Final preprocessing and weights are then fitted from scratch on all complete pre-test seasons, including validation. LSTM runs exactly the selected epoch count, with no test/validation early stopping during final refitting.
- Ridge grid is [0.1,1,10,100,1000,10000,100000]. No outer-result-driven search. HGB/LSTM settings, features, CPU/thread cap and seed remain as in the pilot.
- Seasonal baseline averages the same CDC week in all complete earlier seasons. Genuine earlier week53 observations are used when present; otherwise historical weeks52/1 are averaged. It never updates from the current test season.
- Both origin and target must lie within the outer test season; earlier history remains available. All five methods share identical keys. Final refitting restores valid windows across the former train/validation boundary because the expanded fitting interval is continuous.

## Saved evidence

`folds/<season>/` includes selected candidate models, validation predictions/metrics, full Ridge candidate scores, selection curve, final refit loss curve, both scalers, seasonal lookup tables, frozen model hashes, final weights and forecasts. `forecasts_before_evaluation.csv` is saved before outer labels are joined. All six folds are fitted/frozen before combined outer metrics are generated. `model_selection_summary.csv` and `split_summary.csv` collect choices, counts and boundaries. `run_summary.json`, `verification_summary.json` and `execution_log.txt` record actual checks. `verify_rolling.py` independently checks every raw-data window, boundary, scaler, seasonal prediction, epoch count, metrics and model reload without refitting. Original pilot hashes are recorded in `docs/pilot_preservation_baseline.json`.

For report-only regeneration (no models or forecasts changed):

```bash
MPLCONFIGDIR=.cache/rolling-matplotlib python -c "from rolling.report import render; render('outputs/rolling_v0_2')"
```

Only load trusted local joblib/PyTorch artifacts. ZIP delivery excludes `.venv`, Git metadata and caches, includes the original pilot outputs and original ZIP sibling to satisfy the preservation manifest, and retains source/license documentation.

## Interpretation and deferred stages

This is revised-history retrospective forecasting, not an as-of information-set backtest. Earlier outer seasons were used as training data in the pilot and its last two test seasons were already viewed. New predictions are not fully untouched confirmatory tests. The protocol was fixed before new results and has not been changed to improve rankings. Dependence across regions/weeks and few seasons limit broad generalization. Bias diagnostics and subgroup summaries are descriptive.

This completed phase contains point forecasts only. Moving-block bootstrap, isotonic/score decomposition, quantile/WIS forecasts and vintage reconstruction are not claimed as completed; they require the subsequent planned analysis stages. No arbitrary classification outcomes or AUROC/AUPRC are added.

Codex implemented, ran and tested the analysis and prepared the report; independent review and visual checks are recorded separately. Team members should add their own account of understanding, verification and the benefits/drawbacks of AI assistance. No personal learning statements are fabricated, and nothing is submitted to the course or published.
