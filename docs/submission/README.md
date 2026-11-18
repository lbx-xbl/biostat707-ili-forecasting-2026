# Formal submission update — 18 November 2026

The previous repository contained the proposal and frozen v0.1 pilot. This update adds the completed v0.2–v0.5 analyses and final English report. The original proposal is historical: its prospective work plan is not rewritten to imply it predicted later decisions. Presentation slides, slide sources, and speaker scripts are excluded. Existing proposal, data, code, and pilot output files remain byte-identical; README is intentionally updated.

## Included evidence

- All five stages' analysis code, pinned dependencies, configurations, and tests.
- Revised and historical-vintage raw caches, normalized data, window provenance, fitted model files, point/quantile predictions, scores, figures, and reports.
- Original execution logs and methodological verification summaries, retained as historical records rather than new checkout checks.
- A final English report in Markdown, DOCX, and PDF. Only repository-specific reproducibility wording differs from the previously delivered synthesis; scientific results are unchanged.
- `import_manifest.json`: SHA-256 of scientific files copied from the completed local project.
- `original_repository_manifest.json`: original tracked file hashes before this update.
- `checks.json` and `tests.txt`: fresh checks performed from this checkout using the existing analysis environment, without retraining or redownloading data.

## Verification boundary

The standalone original `verify_rolling.py`, `verify_diagnostics.py`, `verify_probabilistic.py`, and `verify_vintage.py` include historical preservation assertions. Those reference the original project's README/.gitignore, excluded scratch logs, and/or sibling ZIP files. They are preserved as implementation evidence, but their archive-layout assertions do not describe this formal repository. No assertions have been disabled or scientific outputs altered to make them pass.

`scripts/verify_submission.py` provides the repository-level check: import checksums, original repository preservation, presentation exclusion, source snapshot checksum, point-score reconstruction from prediction errors, WIS reconstruction from saved per-observation scores, and final report source consistency. The 70-test analytical suite separately covers temporal boundaries, metrics, resampling, probabilities, and vintage logic. This update does not claim a fresh-machine dependency installation, a rerun of model fitting, or independent human validation.

Bulk ZIP archives, virtual environments, caches, presentation materials, and internal planning files are not committed. The analysis files are directly accessible instead of being hidden in a nested ZIP. Root scratch installation/red-green logs are omitted; relevant version-specific execution and verification records remain in `docs/` and `outputs/`.
