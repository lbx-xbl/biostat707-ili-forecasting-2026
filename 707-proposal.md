# Two-Week-Ahead Regional Influenza-Like Illness Forecasting

*A Retrospective Comparison of Ridge Regression, Gradient Boosting, and LSTM*

**BIOSTAT 707 Final Project Proposal — Fall 2026**  
**Team:** Bowen Zhang, Yufei Wang, Jiaxiang Xue, Lanbu Xi, Shuyang Yu  
**Date:** October 20, 2026

## 1. Research question and motivation

Can a regularized linear model, a tree-based model, or a small recurrent neural network improve two-week-ahead forecasts of regional influenza-like illness (ILI) over a persistence baseline when they receive the same historical information? How do their accuracy, calibration, and ability to distinguish different levels of ILI vary across regions and influenza seasons?

We will study continuous, population-weighted ILI visit percentages in the ten U.S. Department of Health and Human Services (HHS) regions. The project emphasizes reproducible temporal validation and explanations of model failures, with no claim about individual infection risk or clinical benefit.

A completed pilot has downloaded and loaded the data and produced forecasts from all four models. The final project will add exploratory diagnostics to this comparison; its value does not depend on an LSTM outperforming simpler models.

## 2. Dataset, access, license, and analytic cohort

We use CDC's U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet), accessed through the CMU Delphi FluView API. The outcome, `wili`, is the population-weighted percentage of outpatient visits associated with ILI. Errors are expressed in percentage points. This is a syndromic surveillance measure, not laboratory-confirmed influenza incidence [1].

Delphi identifies this source as **Public Domain US Government** [1]. The pilot downloaded it without credentials, a data-use agreement, or competition terms. We will retain attribution to CDC and CMU Delphi, the source URL, the raw snapshot, and its checksum. Source data are available free of charge; this student analysis is not endorsed by CDC, HHS, or the U.S. Government [2].

The actual downloaded panel covers **2010w40–2019w39**, with **469 weekly observations per region and 4,690 region-week observations overall**. It has been read and checked: all ten regions contain the outcome and required time identifiers, with no missing `wili`, duplicate region-week keys, or gaps in weekly spacing. The genuine CDC/MMWR week 2014w53 is retained.

The prediction unit is a region–forecast-origin window. We require 12 consecutive historical observations and an observed target two weeks after the origin. No imputation is needed for the fixed snapshot. Reproduction stops on unexpected missing values, duplicates, or calendar gaps so that source changes are investigated before modeling; the pipeline does not silently fill targets or interpolate from future observations.

Exploratory analysis will summarize distributions, seasonal trajectories, regional variation, missingness, and the largest observed forecast errors. Outcome-based descriptions of the pilot test will be labeled exploratory.

## 3. Predictors and models

For region r at origin week t, the input consists of the 12 ILI values from t−11 through t. The target is the ILI value at t+2. The three learned models also receive a region indicator and sine/cosine features for the target date's annual cycle; calendar information is known in advance. No observed values after t enter the predictors.

All learned models pool the ten regions. The tabular models receive 12 lags, ten region indicators, and two calendar features. The LSTM receives the same 12-week sequence, with region/calendar features concatenated to its final sequence representation.

| Model | Role and retained pilot specification |
| --- | --- |
| Persistence | Simple baseline: forecast the value observed at t for t+2. |
| Ridge regression | Regularized linear comparison; alpha selected from 0.1, 1, 10, and 100 using validation Macro-MAE. |
| Histogram gradient boosting | Tree-based comparison; squared-error loss, 100 iterations, learning rate 0.05, at most seven leaves, minimum 20 samples per leaf, and L2 regularization 1. |
| LSTM | One recurrent layer with 16 hidden units and standardized squared-error training; early stopping uses validation Macro-MAE, patience eight, and at most 60 epochs. |

Ridge penalizes coefficient magnitude, which can stabilize estimates from correlated predictors such as neighboring lag values [3]. The more flexible models allow nonlinear relationships, but their greater flexibility does not guarantee better performance in this setting.

The seed is 2026; complete hyperparameters are recorded in `config/demo.json`. Scaling uses unique Train region-week observations, not duplicated appearances within windows. Broader tuning and additional random seeds are outside the core work proposed here.

## 4. Temporal validation and information boundaries

We retain the pilot's chronological split:

| Partition | Surveillance-week period | Eligible region–origin windows |
| --- | --- | ---: |
| Train | 2010w40–2016w39 | 3,000 |
| Validation | 2016w40–2017w39 | 500 |
| Pilot test | 2017w40–2019w39 | 1,020 |

An origin and its t+2 target must both belong to the assigned partition. Historical inputs may extend into an earlier partition. Consequently, the pilot test contains 102 target weeks across ten regions: 50 weeks in the 2017–18 season and 52 in 2018–19. Its first eligible target is 2017w42.

Model parameters remain fixed during evaluation while the 12-week input window advances at successive origins. This yields repeated t+2 forecasts across the evaluation period.

In the completed pilot, validation selected Ridge's alpha and the LSTM checkpoint. There was no Train-plus-Validation refit before testing. All four models use identical evaluation keys, with no random row split.

**Revision-time limitation.** The snapshot contains revised historical observations. Although features precede their targets in surveillance time, the values are not reconstructed as they were known on each historical publication date. Therefore, this is a retrospective t+2 comparison on a revised snapshot, not a validated real-time two-week forecast. A release-date/vintage backtest would require a separate specification of issuance dates and available data [1,4] and is outside the core scope.

**Pilot exposure.** Pilot-test targets were not used to fit model parameters or select hyperparameters/checkpoints. We retain the saved results as a descriptive out-of-sample comparison of that frozen specification. However, the test results have now been viewed and inform this proposal. Additional analyses are exploratory; these observations will not be described as a new untouched holdout, nor used to revise models and claim independent confirmation.

## 5. Completed pilot and feasibility

The completed pilot provides exploratory retrospective evidence. MAE and RMSE are in percentage points; improvement refers to pilot-test Macro-MAE versus persistence.

| Model | Validation Macro-MAE | Pilot-test Macro-MAE | Pilot-test pooled RMSE | MAE improvement |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.3890 | 0.4903 | 0.8503 | 0.00% |
| Ridge | 0.3234 | 0.4174 | 0.7780 | 14.87% |
| Gradient boosting | 0.2963 | 0.4311 | 0.9119 | 12.07% |
| LSTM | 0.3030 | 0.4357 | 0.8483 | 11.13% |

Ridge had the lowest pilot-test MAE and RMSE, whereas gradient boosting had the lowest validation MAE. Ridge's overall test ranking was therefore not selected in advance by validation. Its alpha was 100, the largest candidate in the specified grid; the LSTM used its validation-selected checkpoint.

Regional results qualify the aggregate ranking: gradient boosting had lower MAE than Ridge in seven of ten regions, while Ridge's advantages in some higher-error regions produced a lower overall Macro-MAE. Gradient boosting's higher RMSE despite lower MAE than persistence motivates examining large misses, including underprediction during some peaks. These are descriptive findings, not evidence that one model family is universally superior.

Saved forecasts, configurations, metrics, and execution records establish feasibility. Independent recalculation agreed with the reported metrics. No models were retrained during proposal preparation.

## 6. Remaining analysis: accuracy, calibration, discrimination, and uncertainty

The remaining work uses the saved forecasts. The retained model specifications and checkpoints will stay fixed; the proposed calibration and resampling analyses have not yet been executed.

### Accuracy and heterogeneity

The primary metric remains **Macro-MAE**: calculate MAE separately for each region and then average the ten regional MAEs equally. This weights regions equally in percentage-point error, without normalizing by regional variability. Secondary metrics are pooled RMSE and mean prediction error, defined as the average of prediction minus observation.

We will report these measures overall, by region, and by pilot-test season. Trajectory plots and summaries of the largest errors will retain all eligible regions and weeks. Model rankings will be interpreted in relation to the loss being evaluated.

### Calibration and discrimination for a continuous outcome

For the course's discrimination/calibration requirement, we propose a continuous point-forecast framework [5,6]. Calibration assessment will include mean bias, regression of observed values on predictions (ideal intercept 0 and slope 1), and isotonic reliability curves. Pooled and regional diagnostics will distinguish overall behavior from regional differences.

For squared error, let S be the original mean squared error, S_iso the error after an isotonic diagnostic fit on the same evaluation sample, and U the mean squared deviation from that sample's observed mean. With the same observations and weights throughout:

- Miscalibration: MCB = S − S_iso.
- Discrimination: DSC = U − S_iso.
- Uncertainty: UNC = U.
- Decomposition: S = MCB − DSC + UNC.

Here, discrimination measures separation of outcome means after an in-sample monotone diagnostic fit; it is not binary-classification AUROC. Pooled and region-specific decompositions will be reported, since pooled discrimination can partly reflect regional baseline differences. The decomposition concerns **MSE**, not the primary MAE.

The isotonic fit describes the observed sample. It will not replace the frozen predictions, and its in-sample error reduction will not be presented as independently validated recalibration improvement.

### Dependence-aware uncertainty

We plan a paired moving-block bootstrap with **1,000 resamples and eight consecutive target weeks per block**. Blocks will be sampled separately within each pilot-test season and truncated to retain that season's original length. Each selected week will carry all ten regions and all four models together. Each resample will recompute MAE, RMSE, and paired differences versus persistence from the saved prediction–observation records.

The 2.5th and 97.5th percentiles will form exploratory bootstrap intervals conditional on the fixed models and two observed seasons. These preserve some dependence but do not guarantee exact coverage or capture retraining and future-season uncertainty. Calibration curves and decompositions will be descriptive, without bootstrap confidence bands.

## 7. Expected findings and limitations

Based on the completed pilot, we expect a regularized lag model to remain a useful benchmark and model rankings to depend on region, season, and error metric. We expect calibration diagnostics to help distinguish systematic underprediction from limited separation of high and low ILI levels. We do not predict that extra model complexity must improve performance.

Interpretation will account for overlapping windows, related regional epidemics, only six training seasons and two pilot-test seasons, limited tuning, one LSTM seed, and the use of a revised historical snapshot. The evaluation concerns later weeks in the same ten regions; it does not establish transport to new regions or post-2019 surveillance conditions. ILI is an aggregate visit-based measure influenced by reporting and care seeking [1]. Good average error alone does not establish reliable peak forecasts or public-health decision benefit.

The project contribution will be a transparent comparison and explanation of these tradeoffs, with clear boundaries between observed results, plausible explanations, and untested extensions.

## 8. Five-member work plan and reproducible deliverables

| Member responsibility | Main contribution |
| --- | --- |
| Bowen Zhang — Data and provenance lead | Source/license documentation, frozen snapshot, data dictionary, EDA, and missingness audit. |
| Yufei Wang — Temporal validation lead | CDC week handling, window eligibility, split boundaries, preprocessing checks, and reproducibility review. |
| Jiaxiang Xue — Baseline and tree-model lead | Persistence, Ridge, and gradient-boosting specifications; interpretation of regional and seasonal errors. |
| Lanbu Xi — Recurrent-model lead | LSTM architecture and training-curve explanation; checkpoint and model-reload documentation. |
| Shuyang Yu — Evaluation and report lead | Continuous calibration/discrimination diagnostics, paired bootstrap, figures, and report integration. |

All five members will review the shared conclusions and explain their own contribution. Work will proceed through three milestones: preserve and document the pilot; complete the specified exploratory evaluation; then review and finalize the report and reproducible repository.

Deliverables will include the proposal, data-loading script and small redistributable snapshot, source/license manifest, locked dependencies, configuration, model/evaluation code, saved forecasts, and final report. Existing pilot evidence is retained in `data/source_manifest.json`, `config/demo.json`, and `outputs/demo_v0_1/`.

The group will submit through a new GitHub repository with `707-proposal.md` at its top level, all five members as collaborators, and individual commits and contribution statements. If private, the repository will also include the instructor and teaching assistants as collaborators. The repository URL will be submitted in Canvas. AI assistance in coding or writing will be documented, and the group will verify and take responsibility for the submitted analysis.

## References

1. CMU Delphi. [FluView ILINet Outpatient Illness: source, definitions, license, and revision behavior](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html). Accessed October 20, 2026.
2. CDC. [Use of Agency Materials](https://www.cdc.gov/other/agencymaterials.html). Accessed October 20, 2026.
3. Scikit-learn developers. [Linear Models: Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification).
4. CMU Delphi. [FluView endpoint documentation and migration guidance](https://cmu-delphi.github.io/delphi-epidata/api/fluview.html). The pilot's exact request URL, acquisition metadata, and SHA-256 are preserved in `data/source_manifest.json`.
5. Gneiting, T., and Resin, J. (2023). Regression Diagnostics meets Forecast Evaluation: Conditional Calibration, Reliability Diagrams, and Coefficient of Determination. *Electronic Journal of Statistics*, 17, 3226–3286. [https://doi.org/10.1214/23-EJS2180](https://doi.org/10.1214/23-EJS2180). [Author manuscript](https://arxiv.org/abs/2108.03210).
6. Model Diagnostics documentation. [Scoring and the miscalibration–discrimination–uncertainty decomposition](https://lorentzenchr.github.io/model-diagnostics/api-reference/scoring/#model_diagnostics.scoring.decompose).

