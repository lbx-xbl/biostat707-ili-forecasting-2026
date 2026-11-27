# Individual contribution statements

BIOSTAT 707 · Fall 2026 · Regional influenza-like illness forecasting

Each statement describes the member's own work. Commits were made from two accounts (lbx-xbl and bz2357-create); see "Team roles and commit history" in the [README](README.md) and the [final report](reports/final-report.md).

## Bowen (Bofan) Zhang · Data and Provenance Lead

I prepared the CDC ILINet download script and documented the source and verification of our saved data snapshot. For the historical vintage analysis, my focus was data provenance: where the archived values came from and which observations were available at each scheduled forecast date.

## Yufei Wang · Temporal Validation Lead

I led the design of the chronological splits and helped extend the two-season pilot to a six-season rolling evaluation. I reviewed the timing rules in the AI-assisted historical vintage analysis (v0.5), including which data could enter training and what was available when each forecast was scheduled to be issued.

## Leslie (Jiaxiang) Xue · Baseline and Tree-Model Lead

I led the comparison of persistence, the same-week seasonal average, ridge regression, and histogram gradient boosting under the shared evaluation design. I helped interpret why gradient boosting had the lowest mean absolute error in the rolling evaluation while ridge had the lowest root mean squared error.

## Lanbu Xi · Recurrent-Model Lead

I led the recurrent-model component, which used a small long short-term memory (LSTM) network to forecast regional ILI from the previous 12 weeks. I compared its results with the simpler models under the same evaluation design and helped explain why its ranking changed with the choice of error metric.

## Michael (Shuyang) Yu · Evaluation and Report Lead

I reviewed and interpreted the AI-assisted evaluation outputs, including the paired moving-block bootstrap intervals and CORP calibration decomposition in v0.3, and the weighted interval score (WIS) and empirical interval coverage in v0.4. I brought these findings into the final report and explained how an overall coverage result could hide weaker coverage in a severe season.
