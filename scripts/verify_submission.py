#!/usr/bin/env python3
"""Read-only verification of this submission's frozen artifacts; no model fitting."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(actual, expected, label):
    require(np.isclose(actual, expected, rtol=1e-9, atol=1e-10),
            f'{label}: reconstructed {actual}, saved {expected}')

def main():
    imported = json.loads((ROOT / 'docs/submission/import_manifest.json').read_text())
    original = json.loads((ROOT / 'docs/submission/original_repository_manifest.json').read_text())
    for path, digest in imported.items():
        require(sha(ROOT / path) == digest, f'Imported artifact changed: {path}')
    for path, digest in original.items():
        if path != 'README.md':
            require(sha(ROOT / path) == digest, f'Original repository file changed: {path}')
    # This update excludes slides, speaker scripts and presentation authoring sources.
    excluded = re.compile(r'(presentation|speaker.?script|slides\.json|build_slides|\.pptx$)', re.I)
    submitted = [p for p in ROOT.rglob('*') if p.is_file() and not any(
        part in {'.git', '.venv', '.cache', '.pytest_cache', '__pycache__'} for part in p.relative_to(ROOT).parts)]
    require(not [str(p.relative_to(ROOT)) for p in submitted if excluded.search(str(p.relative_to(ROOT)))],
            'Presentation-related file found')
    source = json.loads((ROOT / 'data/source_manifest.json').read_text())
    require(sha(ROOT / source['snapshot_path']) == source['sha256'], 'Pilot snapshot checksum')
    facts = json.loads((ROOT / 'reports/source/facts.json').read_text())
    for item in facts['sources'].values():
        require(sha(ROOT / item['path']) == item['sha256'], f"Report source: {item['path']}")
    point_groups = 0
    for version in ['demo_v0_1', 'rolling_v0_2']:
        folder = ROOT / 'outputs' / version
        pred = pd.read_csv(folder / 'predictions.csv')
        met = pd.read_csv(folder / 'metrics.csv')
        keys = ['sample_id', 'model']
        require(not pred.duplicated(keys).any(), f'{version}: duplicate predictions')
        for row in met[met.scope == 'overall'].itertuples():
            subset = pred[pred.model == row.model]
            if hasattr(row, 'split'):
                subset = subset[subset.split == row.split]
            error = subset.y_pred - subset.y_true
            close(error.abs().groupby(subset.region).mean().mean(), row.macro_mae, 'Macro-MAE')
            close(np.sqrt(np.mean(error ** 2)), row.pooled_rmse, 'RMSE')
            if hasattr(row, 'bias'):
                close(error.mean(), row.bias, 'Bias')
            require(len(subset) == row.n, 'Point sample count')
            point_groups += 1
    probability_groups = 0
    wis_rows = 0
    for version in ['probabilistic_v0_4', 'vintage_v0_5']:
        folder = ROOT / 'outputs' / version
        scores = pd.read_csv(folder / 'scores.csv')
        wide = pd.read_csv(folder / 'quantiles_wide.csv')
        met = pd.read_csv(folder / 'metrics.csv')
        keys = ['sample_id', 'model'] + (['regime'] if 'regime' in scores else [])
        require(not scores.duplicated(keys).any() and not wide.duplicated(keys).any(), 'Duplicate probability keys')
        require(set(map(tuple, scores[keys].values)) == set(map(tuple, wide[keys].values)), 'Probability key mismatch')
        qcols = sorted([c for c in wide if c.startswith('q_')], key=lambda c: float(c[2:]))
        tau = np.array([float(c[2:]) for c in qcols])
        require(len(tau) == 23, 'Quantile count')
        q = wide[qcols].to_numpy()
        require(np.all(np.diff(q, axis=1) >= -1e-12), 'Crossed quantiles')
        error = wide.y_true.to_numpy()[:, None] - q
        wide['recomputed_wis'] = 2 * np.mean(np.maximum(tau * error, (tau - 1) * error), axis=1)
        aligned = scores.merge(wide[keys + ['recomputed_wis']], on=keys, validate='one_to_one')
        require(np.allclose(aligned.wis, aligned.recomputed_wis, rtol=1e-9, atol=1e-10), 'WIS pinball identity')
        wis_rows += len(aligned)
        for row in met[met.scope == 'overall'].itertuples():
            subset = aligned[aligned.model == row.model]
            if hasattr(row, 'regime'):
                subset = subset[subset.regime == row.regime]
            close(subset.recomputed_wis.groupby(subset.region).mean().mean(), row.macro_wis, 'Macro-WIS')
            require(len(subset) == row.n, 'Probability sample count')
            if hasattr(row, 'point_macro_mae'):
                error = subset.y_pred - subset.y_true
                close(error.abs().groupby(subset.region).mean().mean(), row.point_macro_mae, 'Vintage point MAE')
                close(np.sqrt(np.mean(error ** 2)), row.point_rmse, 'Vintage point RMSE')
                close(error.mean(), row.point_bias, 'Vintage point bias')
            probability_groups += 1
    result = dict(status='pass', imported_files_unchanged=len(imported),
                  original_files_preserved_excluding_readme=len(original)-1,
                  presentation_files_added=0, point_summary_groups=point_groups,
                  probability_summary_groups=probability_groups,
                  independent_pinball_wis_rows=wis_rows,
                  report_source_hashes=len(facts['sources']),
                  model_training_performed=False, fresh_dependency_installation_tested=False,
                  python=sys.version.split()[0])
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
