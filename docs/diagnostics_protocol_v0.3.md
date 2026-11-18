# ILI v0.3: frozen-forecast uncertainty and mean-calibration diagnostics

Date: 2026-11-17. User authorized direct execution of the next stage. This protocol is fixed before running the new diagnostics. Existing v0.1/v0.2 results were already viewed; this is not a new confirmatory design. No forecasting model, weight, hyperparameter, input feature, saved forecast or previous report is changed.

## Scope and alternatives

Use the 15050 saved rolling forecast rows (3010 matched test keys × five models). Main evaluation covers six seasons 2013–14 through 2018–19. The two-season pilot remains a separate prior analysis and is not pooled or resampled as additional observations. Produce paired loss-difference uncertainty and conditional-mean calibration/discrimination diagnostics overall, by season and by region. Region×season point metrics remain available in v0.2; do not multiply inferential subgroup claims. Quantile forecasts, WIS, vintage reconstruction and deploying recalibrated forecasts remain out of scope.

A row-wise bootstrap would destroy dependence; resampling regions or whole seasons would change the target and leave only six seasonal units. The selected approach holds regions/seasons fixed and resamples time blocks within each season, pairing all models and all ten regions. These are implementation defaults under the user's instruction to proceed, not professor-specified constants.

## Paired moving-block bootstrap

- Primary block length: 8 consecutive eligible target weeks. Sensitivity: 4 and 13 weeks; all results reported, no choosing a favorable length. These span short, intermediate and roughly quarter-season dependence; no claim of optimality. Primary eight-week length is a practical a priori choice, not estimated from these errors.
- 5000 replicates per length, seed 2026, deterministic SeedSequence keyed by block length and season start. NumPy PCG64 generator. CPU <=4 threads.
- For each season with n eligible weeks (50 or 51), sample ceil(n/L) starting indices independently and uniformly from 0,...,n-L. Concatenate overlapping blocks of length L, truncate to n. No wrapping, no block crossing seasons, no joining originally nonconsecutive weeks within a block. Sample the same weekly indices for all regions and models. Use the same draws for MAE and RMSE, overall, regional and seasonal summaries.
- Preserve each season's original n and each of ten regions, yielding the same overall observation weights as v0.2. Compute Macro-MAE as the mean of ten regional MAEs and pooled RMSE as sqrt(mean squared error). For a single region MAE is ordinary regional MAE. Overall RMSE is computed from pooled squared errors, never by averaging seasonal RMSEs.
- Difference = model metric minus Persistence metric. Negative favors the model. Methods compared: Seasonal, Ridge, HGB, LSTM. Save the draws/replicates for reproducibility. Point estimate is the original sample difference, not the average bootstrap difference.
- 95% percentile intervals at [0.025,0.975] using linear sample quantiles; report bootstrap mean minus original estimate as a diagnostic of centering/edge effects. Intervals are marginal descriptive intervals, not multiplicity-adjusted tests. No p-values, winner probabilities or significance ranking. Primary claims are overall differences; subgroup intervals are exploratory.
- Limitations: fixed models/data-vintage/seasons/regions; no training/selection uncertainty and no future-season or geographic population sampling. Shared training across folds, longer dependence, and seasonal nonstationarity are not fully captured. Noncircular moving blocks overrepresent interior weeks and can shift centering; record draws' marginal inclusion frequencies and show sensitivity. Approximate coverage is not guaranteed. Do not equate resampled block joins with new physically consecutive observations.

## CORP mean diagnostics

Verify Gneiting & Resin (2023), Electronic Journal of Statistics 17:3226–3286, DOI 10.1214/23-EJS2180, section 3.3, eqs 26–28, before implementation. Evaluate existing point forecasts as candidates for conditional means because learned models were fitted with squared-error objectives. MAE stays the primary accuracy measure but has a different median-optimal functional; no MAE decomposition is inferred from mean isotonic fits.

For each model and subgroup, fit non-decreasing isotonic least-squares y on the saved forecast x. Use sklearn IsotonicRegression(increasing=True), equal sample weights, exact forecast ties sharing a fitted value; no binning, no increasing='auto'. Fit only to describe that same evaluation subgroup. The map is never applied to replace forecasts or evaluated as a deployable improved model.

S = mean((x-y)^2); S_iso = mean((z-y)^2), z the isotonic fitted values;
UNC = mean((y-mean(y))^2), ddof=0;
MCB = S-S_iso; DSC = UNC-S_iso; S = MCB-DSC+UNC.
Compute MCB by score differences, not by mean((x-z)^2), which need not equal the empirical CORP component. Optional refinement fixed here: unconditional MCB = mean(x-y)^2; conditional MCB = MCB-unconditional MCB. Report sqrt(MCB) on the original percentage-point scale. Components themselves use squared percentage points. Check nonnegativity up to numerical tolerance and both decomposition identities; do not silently clip substantial failures.

Report 1-S/UNC as skill relative to the evaluation subgroup's empirical mean; negative values allowed. When UNC=0, mark skill NA with an explanation. This constant uses evaluation outcomes and is a decomposition reference, not an operational seasonal baseline or genuine out-of-sample reference forecast.

Produce overall, by-region and by-season reliability curves and component tables. The pooled plot can conceal subgroup errors; show both. Increasing fit constrains the estimated relation; it does not prove the population relation is monotone. Extreme forecasts with few observations are unstable. These are finite-sample descriptive fits, potentially optimistic; no iid calibration tests, bootstrap bands or formal calibration-null claims are added. Naive bootstrap inference for shape-constrained fits is deliberately not introduced.

## Sources verified

- Gneiting & Resin: https://doi.org/10.1214/23-EJS2180 ; open author/institution copy https://publikationen.bibliothek.kit.edu/1000169184/152434570 . Empirical formulas 26–28, nonnegative components theorem 3.3; optional bias refinement 29–30; out-of-sample skill can be negative (31).
- Künsch (1989), The Jackknife and the Bootstrap for General Stationary Observations, DOI 10.1214/aos/1176347265. Moving-block motivation; application to seasonal, nonstationary forecasts is approximate and explicitly limited.
- scikit-learn IsotonicRegression: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html . Installed version remains pinned in requirements.txt.

## Preservation, tests and delivery

Before new analysis record hashes of existing project files and prior ZIPs; write only new diagnostics files and outputs/diagnostics_v0_3. Run hand-calculable decomposition tests (ties, nonmonotone outcomes, constant forecast/truth, positive bias) and block-resampling tests (contiguity, boundaries, synchrony, exact pairing, deterministic reproducibility, pooled RMSE). Independently reconstruct intervals and components from saved intermediates. Verify previous file hashes. Inspect figures, perform fresh review, and package code, protocol/config, saved inputs, intervals, loss-difference replicates/draws, isotonic diagnostic values, report and checks. All statuses and results must be from actual execution; English self-contained report, Chinese final explanation.
