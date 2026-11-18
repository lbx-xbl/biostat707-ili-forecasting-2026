#!/usr/bin/env python3
"""Independent read-only scientific checks of saved rolling artifacts, no training."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='4'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from epiweeks import Week
from threadpoolctl import threadpool_limits
from src.windows import features
from rolling.models import load_models,predict
from rolling.evaluate import metrics

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(out):
    out=Path(out); cfg=json.loads((out/'config_frozen.json').read_text())
    baseline=json.loads(Path('docs/pilot_preservation_baseline.json').read_text())
    assert all(Path(p).exists() and sha(p)==h for p,h in baseline.items()),'Pilot modified'
    info=json.loads(Path('data/source_manifest.json').read_text()); rawpath=Path(info['snapshot_path'])
    assert sha(rawpath)==info['sha256']
    raw=json.loads(rawpath.read_text()); assert raw['result']==1 and raw['message']=='success'
    panel=pd.DataFrame(raw['epidata']); assert len(panel)==4690 and not panel.duplicated(['region','epiweek']).any()
    lookup={(r.region,int(r.epiweek)):r.wili for r in panel.itertuples()}
    w=pd.read_csv(out/'all_windows.csv'); assert len(w)==4560
    for row in w.itertuples():
        origin=Week.fromstring(str(row.origin_week),system='cdc'); target=origin+2
        assert int(str(target))==row.target_week
        assert str(origin.enddate())==row.origin_date and str(target.enddate())==row.target_date
        assert row.y_true==lookup[row.region,row.target_week]
        for j in range(12): assert getattr(row,f'lag_{j}')==lookup[row.region,int(str(origin-11+j))]
    predictions=pd.read_csv(out/'predictions.csv'); met=pd.read_csv(out/'metrics.csv')
    assert len(predictions)==15050
    assert not predictions.duplicated(['model','sample_id']).any()
    choice=pd.read_csv(out/'model_selection_summary.csv'); summary=pd.read_csv(out/'split_summary.csv')
    torch.set_num_threads(4)
    reload_errors={}; seasonal_fallback_counts={}
    with threadpool_limits(limits=4):
        for s in cfg['test_season_starts']:
            name=f'{s}-{s+1}'; path=out/'folds'/name
            sel=json.loads((path/'model_selection.json').read_text()); frozen=json.loads((path/'refit_frozen.json').read_text())
            assert sha(path/'model_selection.json')==frozen['selection_sha256']
            assert all(sha(path/'models'/f)==h for f,h in frozen['final_model_sha256'].items())
            curve=pd.read_csv(path/'selection_curve.csv'); refit=pd.read_csv(path/'refit_curve.csv')
            best=np.inf; epoch=0; stale=0
            for r in curve.itertuples():
                if r.validation_macro_mae<best-cfg['lstm']['min_delta']: best=r.validation_macro_mae; epoch=r.epoch; stale=0
                else: stale+=1
                assert stale<cfg['lstm']['patience'] or r.epoch==len(curve)
            assert epoch==sel['lstm_best_epoch']==len(refit)==frozen['final_refit_epochs']
            assert len(curve)==60 or stale==cfg['lstm']['patience']
            candidates=sel['ridge_candidates']; a=min(candidates,key=lambda r:(r['mae'],-r['alpha']))['alpha']
            assert a==sel['ridge_alpha'] and [v['alpha'] for v in candidates]==cfg['ridge_alphas']
            for transform,upper in [(sel['selection_transform'],(s-1)*100+39),(sel['refit_transform'],s*100+39)]:
                values=panel[panel.epiweek.between(201040,upper)].wili
                assert transform['n_unique_observations']==len(values)
                np.testing.assert_allclose([transform['mu'],transform['sigma']],[values.mean(),values.std(ddof=0)],atol=1e-12,rtol=0)
            bounds={'train':(201040,(s-1)*100+39),'validation':((s-1)*100+40,s*100+39),'refit':(201040,s*100+39),'test':(s*100+40,(s+1)*100+39)}
            for stage,(lo,hi) in bounds.items():
                expected=w[w.origin_week.between(lo,hi)&w.target_week.between(lo,hi)]
                for region,g in expected.groupby('region'):
                    z=summary[(summary.fold==name)&(summary.stage==stage)&(summary.region==region)].iloc[0]
                    assert [z.n,z.origin_min,z.origin_max,z.target_min,z.target_max]==[len(g),g.origin_week.min(),g.origin_week.max(),g.target_week.min(),g.target_week.max()]
            lo,hi=bounds['test']; test=w[w.origin_week.between(lo,hi)&w.target_week.between(lo,hi)].reset_index(drop=True)
            assert len(test)==(510 if s==2014 else 500)
            g=predictions[predictions.fold==name]
            for model in ['Persistence','Seasonal','Ridge','HGB','LSTM']:
                q=g[g.model==model].reset_index(drop=True)
                assert q.sample_id.tolist()==test.sample_id.tolist()
                for col in ['region','origin_week','origin_date','target_week','target_date','target_season','y_true']:
                    assert q[col].tolist()==test[col].tolist()
            np.testing.assert_array_equal(g[g.model=='Persistence'].y_pred.to_numpy(),test.lag_11.to_numpy())
            # Independently reconstruct every seasonal value, including fallback.
            hist=panel[panel.epiweek.between(201040,s*100+39)]
            expected_seasonal=[]; fallback=0
            for row in test.itertuples():
                r=hist[hist.region==row.region]; h=r[r.epiweek%100==row.target_week%100]
                if h.empty:
                    assert row.target_week%100==53; h=r[(r.epiweek%100).isin([52,1])]; fallback+=1
                expected_seasonal.append(h.wili.mean())
            np.testing.assert_allclose(g[g.model=='Seasonal'].y_pred,expected_seasonal,rtol=0,atol=1e-12)
            seasonal_fallback_counts[name]=fallback
            t=sel['refit_transform']; x=features(test.drop(columns='y_true'),t,cfg['regions']); bundle=load_models(path/'models')
            values=predict(bundle,x,t); err={}
            for model in values:
                saved=g[g.model==model].y_pred.to_numpy(); np.testing.assert_allclose(values[model],saved,rtol=0,atol=1e-6)
                err[model]=float(np.max(np.abs(values[model]-saved)))
            reload_errors[name]=err
            # Changing evaluator labels with fixed features cannot change inference or saved selection.
            altered=g.copy(); altered['y_true']+=1000; metrics(altered)
            again=predict(bundle,x,t)
            for model in values: np.testing.assert_array_equal(again[model],values[model])
            assert sha(path/'model_selection.json')==frozen['selection_sha256']
    # Independent arithmetic for every published row (not only the producer's metrics function).
    for row in met.itertuples():
        g=predictions
        if row.scope=='season': g=g[g.target_season==row.group]
        elif row.scope=='region': g=g[g.region==row.group]
        elif row.scope=='season_region':
            season,region=row.group.split('/'); g=g[(g.target_season==season)&(g.region==region)]
        elif row.scope!='overall': raise AssertionError('Unknown scope')
        q=g[g.model==row.model]; e=q.y_pred.to_numpy()-q.y_true.to_numpy()
        mae_by_region=[np.abs(r.y_pred-r.y_true).mean() for _,r in q.groupby('region')]
        b=g[g.model=='Persistence']; bmae=np.mean([np.abs(r.y_pred-r.y_true).mean() for _,r in b.groupby('region')])
        actual=[len(q),np.mean(mae_by_region),np.abs(e).mean(),np.sqrt(np.mean(e**2)),e.mean(),100*(bmae-np.mean(mae_by_region))/bmae]
        np.testing.assert_allclose(actual,[row.n,row.macro_mae,row.pooled_mae,row.pooled_rmse,row.bias,row.improvement_pct],atol=1e-12,rtol=1e-12)
    assert all(sha(p)==h for p,h in baseline.items())
    report=(out/'rolling_report.html').read_text(); assert report.count('src="data:image/png;base64,')==5
    result={'status':'PASS','raw_rows':len(panel),'windows_checked_against_raw':len(w),'outer_samples_per_model':3010,'prediction_rows':len(predictions),'pilot_files_unchanged':len(baseline),'metrics_rows_independently_recomputed':len(met),'reload_max_errors':reload_errors,'seasonal_week53_fallback_predictions':seasonal_fallback_counts,'fixed_refit_epochs':'PASS','selection_and_refit_ranges_and_scalers':'PASS','fixed_feature_label_perturbation':'PASS','embedded_report_figures':5}
    (out/'verification_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default='outputs/rolling_v0_2'); args=ap.parse_args(); verify(args.output_dir)
