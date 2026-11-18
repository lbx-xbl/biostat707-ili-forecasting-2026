"""Descriptive point-forecast evaluation; no fitting occurs here."""
import numpy as np
import pandas as pd
MODELS=['Persistence','Seasonal','Ridge','HGB','LSTM']

def score(d):
    error=d.y_pred.to_numpy()-d.y_true.to_numpy()
    grouped=d.assign(error=error,abs_error=np.abs(error)).groupby('region')[['error','abs_error']].mean()
    return dict(n=len(d),macro_mae=float(grouped.abs_error.mean()),pooled_mae=float(np.mean(np.abs(error))),
                pooled_rmse=float(np.sqrt(np.mean(error**2))),bias=float(np.mean(error)),macro_bias=float(grouped.error.mean()),negative_predictions=int((d.y_pred<0).sum()))

def validate(d,expected):
    if set(d.model)!=set(MODELS): raise ValueError('Model coverage mismatch')
    if d.duplicated(['sample_id','model']).any(): raise ValueError('Duplicate sample/model')
    if not np.isfinite(d[['y_true','y_pred']].to_numpy()).all(): raise ValueError('Nonfinite predictions/truth')
    columns=[c for c in ['sample_id','region','origin_week','origin_date','target_week','target_date','target_season','y_true'] if c in expected.columns]
    reference=expected[columns].sort_values('sample_id').reset_index(drop=True)
    if reference.sample_id.duplicated().any(): raise ValueError('Nonunique reference keys')
    for model,g in d.groupby('model'):
        actual=g[columns].sort_values('sample_id').reset_index(drop=True)
        try: pd.testing.assert_frame_equal(actual,reference,check_dtype=False,check_exact=True)
        except AssertionError as e: raise ValueError(f'Prediction keys/metadata mismatch: {model}') from e

def metrics(d):
    groups=[('overall','all',d)]
    groups += [('season',s,g) for s,g in d.groupby('target_season',sort=True)]
    groups += [('region',r,g) for r,g in d.groupby('region',sort=False)]
    groups += [('season_region',f'{s}/{r}',g) for (s,r),g in d.groupby(['target_season','region'],sort=False)]
    rows=[]
    for scope,group,g in groups:
        b=score(g[g.model=='Persistence'])['macro_mae']; seasonal=score(g[g.model=='Seasonal'])['macro_mae']
        for model in MODELS:
            v=score(g[g.model==model]); v.update(scope=scope,group=group,model=model,units='percentage points',
                 improvement_pct=100*(b-v['macro_mae'])/b if b else float('nan'),
                 improvement_vs_seasonal_pct=100*(seasonal-v['macro_mae'])/seasonal if seasonal else float('nan'))
            rows.append(v)
    return pd.DataFrame(rows)
