# ILI v0.2: rolling-origin evaluation design

Date: 2026-11-17. Status: approved by user on 2026-11-17; direct execution authorized. This supersedes the pilot's no-refit restriction only for the new analysis. Pilot code, configuration, source snapshot, weights, forecasts, report and ZIP remain unchanged.

## Purpose and selected direction

Implement the professor's priority recommendations: evaluate across seasons, expand the Ridge penalty grid, and add a seasonal baseline. Keep the original pilot alongside the extension. Primary outcome remains region-equal MAE in percentage points; report pooled RMSE and mean signed error (prediction minus observation), by region and season as well as overall. Do not require LSTM to win.

Alternatives: continuing only frozen two-season diagnostics does not address the professor's main recommendation. Implementing probabilistic and vintage evaluation simultaneously increases methodological and data scope. Proposed approach: complete six-season point-forecast evaluation first, then its uncertainty/calibration diagnostics, then optional probabilistic and vintage extensions. This sequencing is Codex's proposal, not a claim that every statistical detail was approved previously.

## Data and preservation

Reuse the exact 4690-row CDC/Delphi snapshot, 10 numerically ordered HHS regions, 2010w40–2019w39, CDC/MMWR weeks and real week53. Use the existing Python environment and pinned libraries. Keep lookback 12, horizon 2, same region/calendar inputs, CPU <=4 threads and seed 2026. No new download or new input signals is required for this phase. Before implementation record hashes of pilot source/configuration/data/results; extension files and output directories must be separate. Do not touch the unrelated PhysioNet Checkpoint project.

## Six outer folds

Target seasons: 2013–14 through 2018–19 inclusive. For each target season S:

1. Validation is the complete immediately preceding season S-1.
2. Candidate training comprises seasons from 2010–11 through S-2. Fit all candidate preprocessing only on unique region-week observations in that training period. Fit weights only on training windows.
3. Select Ridge alpha and LSTM stopping epoch using Validation Macro-MAE only.
4. Freeze the choices, then refit each learned model from scratch on all complete seasons before S, including the validation season. Recompute preprocessing on that expanded fitting period. Train LSTM for exactly the previously selected epoch count; do not early-stop or inspect losses against S. Use the same seed rule for every fold.
5. Freeze final weights before predicting/evaluating S. No retraining within S; later origins may use earlier observations from S.

Example first fold: candidate fit 2010–11 and 2011–12; validation 2012–13; final fit 2010–11 through 2012–13; test 2013–14. Last fold: validation 2017–18; final fit through 2017–18; test 2018–19.

Build histories on the continuous calendar. During selection both origin and target must be within their assigned fitting or validation interval. During outer testing both must be within the target season; earlier history is allowed. For final refitting the combined pre-test history is a single continuous interval, permitting windows that crossed the former training/validation boundary. This differs deliberately from the pilot's continuous two-season Test partition; report exact dates and counts, and do not silently compare unmatched sample sets.

## Models and predeclared choices

- Persistence: raw origin value, unchanged definition.
- Ridge: alpha in [0.1, 1, 10, 100, 1000, 10000, 100000]; select smallest validation Macro-MAE, larger alpha on exact ties. A selected endpoint is reported, not a reason to extend the grid after outer results are seen.
- HGB: unchanged fixed pilot settings, squared-error loss and no internal early stopping.
- LSTM: unchanged pilot architecture/optimizer/batch size/60-epoch cap/patience/threshold; select epoch on the preceding validation season, then refit as above.
- Seasonal baseline: for each region and target CDC week number, average matching week observations from complete seasons strictly before the target season. Fix this baseline for that entire target season. For week53, average earlier week53 observations if any exist; if none exist, use the mean of historical week52 and week1 values from those complete seasons. Record which fallback was used. This rule is fixed in advance and uses no current-season outcomes.

## Evaluation and interpretation

All five models must share identical outer-test keys. Save fold/region/origin/target/season, true value and each untrimmed forecast. Report negative predictions. Show each of six seasons, region-specific performance, and pooled summaries with weighting stated. Retain the pilot as its own analysis, not a seventh independent fold.

These are expanded retrospective evaluations of revised data, not real-time vintage forecasts or fully untouched confirmatory holdouts. Earlier outer-test seasons previously contributed to the pilot's training, and 2017–19 pilot outcomes were viewed. Freeze this new protocol before inspecting new fold results; do not adapt it to model rankings.

## Subsequent evaluation work

After saving the new forecasts, evaluate uncertainty with paired moving blocks of weeks, resampling all regions/models together and preserving season boundaries. Specify block length, replicate count and estimands before calculating intervals. Treat intervals as conditional on observed seasons, not a guarantee across future seasons.

For continuous calibration/discrimination, first verify Gneiting–Resin (2023) definitions, target functional and scoring rule. Use isotonic curves as diagnostic fits on evaluation data; do not replace forecasts or claim independent performance gains from those fits. Any score decomposition must be linked to the appropriate functional, rather than applying squared-error mean calibration mechanically to MAE.

Quantile/WIS forecasts and Delphi as-of reconstruction are later optional stages, requiring their own documented prediction/calibration/data-availability rules before evaluation. They are not silently included in this initial implementation.

## Acceptance and deliverables

Create a separate rolling runner/config/modules and fresh outputs. Tests must establish calendar continuity, information boundaries for every fold and refit, train-only scaling during selection, fixed selected epochs at refit, historical-only seasonal baseline including week53, and identical keys across models. Save each fold's validation candidates, choices, scaler, training dates, weights, predictions and metrics. Reload models, recompute metrics from CSV, inspect all plots, compare pilot hashes, and report actual commands, versions, runtime and failures. Deliver a self-contained English HTML report and ZIP; explain findings in Chinese. Disclose actual AI assistance and ask team members to supply their own learning/reflection accounts rather than inventing them.
