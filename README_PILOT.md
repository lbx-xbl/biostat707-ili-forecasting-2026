# Regional ILI Forecasting Demo v0.1

A local retrospective forecasting demo using revised CDC ILINet data via CMU Delphi. The binding specification is `SPEC.md`. No website deployment or course submission is performed.

## Reproduce

Tested here on macOS arm64 with Python 3.11.5. Other operating systems have not been tested. From this project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py --config config/demo.json --output-dir outputs/demo_v0_1
```

The delivered `outputs/demo_v0_1` already contains the real run. For a new run use `--output-dir outputs/reproduction_01`; nonempty output directories are rejected to prevent mixing runs. The included raw snapshot is reused by default and its SHA-256 is checked. If the raw file is absent, the loader downloads the same public endpoint and creates a new source manifest; revised data may differ. Network access is needed only for package installation or a fresh download.

Open `outputs/demo_v0_1/demo_report.html` in a browser. It embeds all four figures and needs no external assets. `predictions.csv` and `metrics.csv` provide machine-readable results. `data/windows.csv` contains all windows in oldest-to-newest lag order. `model_selection.json` records validation selection and preprocessing. `models/parameters.json` plus the model files support loading through `src.models.load_models`.

## Fixed experiment

Ten numerically ordered HHS regions; 2010w40–2019w39; CDC/MMWR weeks; 12 observations to predict origin+2 weeks. Train ends 2016w39; Validation is 2016w40–2017w39; Test is 2017w40–2019w39. Origin and target must both belong to the same split; input history may cross its start. Later Test windows can use earlier Test observations without retraining.

Ridge uses exactly four alpha candidates (0.1, 1, 10, 100). HGB has a fixed configuration. A one-layer 16-unit LSTM uses Adam, batch size 32, at most 60 epochs, validation MAE patience 8, and a 0.00001 percentage-point improvement threshold. The final weights are the best validation checkpoint. Three learned models share history, region one-hot, and known target-date annual features. Persistence copies the raw origin value. CPU, seed 2026, at most four compute threads. No training or preprocessing fit uses Validation or Test; Validation is used only for prescribed selection/stopping. Test metrics are computed after the selection JSON is frozen. No refit, clipping, extra seed, or search.

## Checks and provenance

`tests/` checks calendar continuity and week 53; exact increasing-series labels; region isolation; boundary exclusion/history retention; duplicate/missing/invalid data rejection; train-only scaling; hand-calculated metrics; prediction-key failures; LSTM dimensions and independent windows. The runner additionally checks real panel counts and boundaries, CSV metric reconstruction, unchanged predictions after evaluation-label perturbation, and reload agreement to absolute tolerance 1e-6. `run_summary.json` records exact runtime, environment, checks and status. Initial test failures and subsequent green test logs are included as implementation evidence.

Only trusted local model files should be loaded (joblib uses Python object serialization). The ZIP excludes the venv, caches and unrelated coursework. Sources and public-domain attribution are in `DATA_SOURCES.md`.

## Interpretation and assistance

ILI is an outpatient syndrome indicator, not laboratory-confirmed influenza or population infection prevalence. Error is in percentage points. This experiment uses revised historical observations, not contemporary data releases; +2 weeks is relative to observation week. Regional and temporal dependence preclude treating windows as independent replicates. Results cover one split/seed before the pandemic. Predicted-versus-observed plots are bias diagnostics, not full probabilistic calibration; AUROC/AUPRC do not apply.

Codex wrote the implementation and tests, installed an isolated environment, downloaded the real data, ran the fixed experiment, inspected artifacts and documented results. If incorporated in coursework, disclose this actual assistance according to course rules; do not describe it as brainstorming only.

Next discussion, without extending this run: (1) real-time release/vintage design, (2) a regression-appropriate discrimination/calibration design for course requirements, (3) a pre-specified seasonal/seed robustness evaluation.
