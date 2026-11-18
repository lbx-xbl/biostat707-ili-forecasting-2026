#!/usr/bin/env python3
"""Independent archive lineage, matched masks, saved-model inference and scoring audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='4'
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from epiweeks import Week
from datetime import timedelta
from src.windows import features
from rolling.models import load_models,predict

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,rtol=tol,atol=tol,equal_nan=True)
def align_scores(w,s):
    keys=["sample_id","model","regime"]; expected=pd.MultiIndex.from_frame(w[keys]); actual=pd.MultiIndex.from_frame(s[keys])
    if not expected.is_unique or not actual.is_unique or set(expected)!=set(actual):raise ValueError("Unpaired score keys")
    return s.set_index(keys).reindex(expected).reset_index()

def verify(out,write=True):
    out=Path(out)
    inputs=json.loads((out/'archive_input_hashes.json').read_text());assert all(sha(f)==h for f,h in inputs.items())
    cfg=json.loads((out/'config_frozen.json').read_text());old=json.loads(Path('docs/vintage_preservation_baseline.json').read_text());assert all(sha(f)==h for f,h in old.items())
    manifests=json.loads(Path('data/vintage/download_manifest.json').read_text());rows=[]
    for m in manifests:
        blob=Path(m['file']).read_bytes();raw=gzip.decompress(blob);assert sha(m['file'])==m['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256'];j=json.loads(raw);assert j['result']==1 and len(j['epidata'])==m['rows'];rows.extend(j['epidata'])
    records=pd.DataFrame(rows);cols=['region','epiweek','issue','release_date','wili'];records=records[cols].sort_values(['region','epiweek','issue']).reset_index(drop=True)
    normalized=pd.read_csv(cfg['vintage_records'])[cols].sort_values(['region','epiweek','issue']).reset_index(drop=True);pd.testing.assert_frame_equal(records,normalized,check_exact=False,rtol=1e-12,atol=1e-12)
    assert len(records)==212710 and not records.duplicated(['region','epiweek','issue']).any() and records.release_date.notna().all()
    snaps={}
    for path in sorted((out/'asof_snapshots').glob('*.csv.gz')):
        cut=path.name[:10];actual=pd.read_csv(path);v=records[records.release_date<=cut];expected=v.sort_values(['region','epiweek','release_date','issue']).groupby(['region','epiweek'],sort=False).tail(1)
        a=actual.set_index(['region','epiweek']).sort_index();e=expected.set_index(['region','epiweek']).sort_index();pd.testing.assert_frame_equal(a[e.columns],e,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12);assert actual.release_date.max()<=cut
        snaps[cut]=a
    final=pd.read_csv(cfg['reference_snapshot']).set_index(['region','epiweek']);template=pd.read_csv('outputs/rolling_v0_2/all_windows.csv');elig=pd.read_csv(out/'eligibility.csv');freeze=json.loads((out/'forecast_freeze.json').read_text());assert all(sha(out/p)==h for p,h in freeze['sha256'].items())
    run=json.loads((out/'run_summary.json').read_text());assert sha(out/'protocol_frozen.md')==run['protocol_sha256'] and sha(out/'config_frozen.json')==run['config_sha256']
    torch.set_num_threads(4);checked_windows=0;reload_max=0.;eligibility_groups=0
    forecasts=pd.read_csv(out/'forecasts_before_evaluation.csv');assert 'y_true' not in forecasts
    for year in cfg['test_season_starts']:
        fold=f'{year}-{year+1}';cuts={'selection':(Week(year-1,40).enddate()+timedelta(days=7)).isoformat(),'refit':(Week(year,40).enddate()+timedelta(days=7)).isoformat()};frames={};histories={}
        ranges={'train':(201040,(year-1)*100+39),'validation':((year-1)*100+40,year*100+39),'refit':(201040,year*100+39),'test':(year*100+40,(year+1)*100+39)}
        for regime in cfg['regimes']:
            folder=out/'folds'/fold/regime;trans=json.loads((folder/'transforms.json').read_text());assert json.loads((folder/'cutoffs.json').read_text())==cuts
            histories[regime]={};frames[regime]={}
            for part,upper in [('selection',(year-1)*100+39),('refit',year*100+39)]:
                h=pd.read_csv(folder/f'{part}_history.csv.gz');histories[regime][part]=h
                source=snaps[cuts[part]];source=source[(source.index.get_level_values('epiweek')>=201040)&(source.index.get_level_values('epiweek')<=upper)]
                a=h.set_index(['region','epiweek']);assert set(a.index)==set(source.index)
                expected=source if regime=='AsOf' else final.reindex(source.index);close(a.wili.reindex(expected.index),expected.wili)
                close([trans[part]['mu'],trans[part]['sigma']],[h.wili.mean(),h.wili.std(ddof=0)]);assert trans[part]['n_unique_observations']==len(h)
            for stage,(lo,hi) in ranges.items():
                g=pd.read_csv(folder/f'{stage}_windows.csv.gz');frames[regime][stage]=g;checked_windows+=len(g)
                assert g.sample_id.is_unique and g.groupby('origin_date').region.nunique().eq(10).all();assert (g.origin_week.between(lo,hi)&g.target_week.between(lo,hi)).all()
                assert (pd.to_datetime(g.target_date)-pd.to_datetime(g.origin_date)).dt.days.eq(14).all()
                for cut,sub in g.groupby('feature_cutoff'):
                    source=snaps[cut] if regime=='AsOf' else final
                    if regime=='AsOf':
                        if stage in ['train','refit']:assert cut==cuts['selection' if stage=='train' else 'refit']
                        else:assert ((pd.to_datetime(sub.origin_date)+pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')==cut).all()
                    for j in range(12):
                        expected_week=(pd.to_datetime(sub.origin_date)+pd.Timedelta(weeks=j-11)).map(lambda t:int(str(Week.fromdate(t.date()))));np.testing.assert_array_equal(sub[f'lag_{j}_week'],expected_week)
                        key=pd.MultiIndex.from_arrays([sub.region,expected_week]);v=source.reindex(key);assert v.wili.notna().all();close(sub[f'lag_{j}'],v.wili)
                        np.testing.assert_array_equal(sub[f'lag_{j}_issue'],v.issue);np.testing.assert_array_equal(sub[f'lag_{j}_release'],v.release_date)
                if stage=='test':assert 'y_true' not in g
                else:
                    lc=cuts['selection' if stage=='train' else 'refit'];source=snaps[lc] if regime=='AsOf' else final;key=pd.MultiIndex.from_arrays([g.region,g.target_week]);v=source.reindex(key);close(g.y_true,v.wili);np.testing.assert_array_equal(g.label_release,v.release_date);np.testing.assert_array_equal(g.label_issue,v.issue)
                    if regime=='AsOf':assert g.label_cutoff.eq(lc).all() and (g.label_release<=lc).all()
            selection=json.loads((folder/'model_selection.json').read_text());candidates=selection['ridge_candidates'];assert selection['ridge_alpha']==min(candidates,key=lambda c:(c['mae'],-c['alpha']))['alpha'];assert set(c['alpha'] for c in candidates)==set(cfg['ridge_alphas'])
            curve=pd.read_csv(folder/'selection_curve.csv');best=float('inf');epoch=0
            for r in curve.itertuples():
                if r.validation_macro_mae<best-cfg['lstm']['min_delta']:best=r.validation_macro_mae;epoch=r.epoch
            assert epoch==selection['lstm_best_epoch'] and len(curve)<=60
            for stage,part,modeldir,predfile in [('validation','selection','selection_models','validation_predictions.csv'),('test','refit','models','forecasts_before_evaluation.csv')]:
                frame=frames[regime][stage];saved=pd.read_csv(folder/predfile);bundle=load_models(folder/modeldir);x=features(frame.drop(columns='y_true',errors='ignore'),trans[part],cfg['regions']);values=predict(bundle,x,trans[part]);values['Persistence']=frame.lag_11.to_numpy()
                h=histories[regime][part].copy();h['wk']=h.epiweek%100;table=h.groupby(['region','wk']).wili.mean().to_dict();fallback=h[h.wk.isin([52,1])].groupby('region').wili.mean().to_dict();values['Seasonal']=np.array([table.get((r.region,r.target_week%100),fallback.get(r.region,np.nan) if r.target_week%100==53 else np.nan) for r in frame.itertuples()])
                for model,value in values.items():
                    g=saved[saved.model==model].set_index('sample_id').reindex(frame.sample_id);close(g.y_pred,value,1e-6);reload_max=max(reload_max,float(np.max(np.abs(g.y_pred-value))))
                assert bundle['Ridge'].alpha==selection['ridge_alpha']
                if stage=='validation':
                    aligned=saved.merge(frame[['sample_id','y_true']],on='sample_id',suffixes=('_saved','_frame'));close(aligned.y_true_saved,aligned.y_true_frame)
            vp=pd.read_csv(folder/'validation_predictions.csv');offset=pd.read_csv(folder/'residual_quantiles.csv');pred=pd.read_csv(folder/'forecasts_before_evaluation.csv')
            for model,g in vp.groupby('model'):
                errors=np.sort((g.y_true-g.y_pred).to_numpy());levels=np.array(cfg['quantiles']);pos=(len(errors)-1)*levels;lower=np.floor(pos).astype(int);upper=np.ceil(pos).astype(int);q=errors[lower]+(pos-lower)*(errors[upper]-errors[lower]);off=offset[offset.model==model].sort_values('quantile');close(off.residual_quantile,q)
                test=pred[pred.model==model];expected=np.clip(test.y_pred.to_numpy()[:,None]+q,0,100);close(test[[f'q_{p:.3f}' for p in levels]],expected)
        for stage,(lo,hi) in ranges.items():
            a=frames['AsOf'][stage];b=frames['MatchedRevised'][stage];assert a.sample_id.tolist()==b.sample_id.tolist()
            # Independent availability reconstruction for every candidate origin, including excluded ones.
            cand=template[template.origin_week.between(lo,hi)&template.target_week.between(lo,hi)];keep=set()
            for origin,g in cand.groupby('origin_date'):
                cut=cuts['selection' if stage=='train' else 'refit'] if stage in ['train','refit'] else (pd.Timestamp(origin)+pd.Timedelta(days=7)).date().isoformat();source=snaps[cut];available=set(source.index);lab=set(snaps[cuts['selection' if stage=='train' else 'refit']].index);valid=True
                for r in g.itertuples():
                    w=Week.fromstring(str(r.origin_week));valid&=all((r.region,int(str(w+j))) in available for j in range(-11,1))
                    if stage!='test':valid&=(r.region,r.target_week) in lab
                    if stage in ['validation','test']:
                        part='selection' if stage=='validation' else 'refit';h=histories['AsOf'][part];wk=r.target_week%100;sub=h[(h.region==r.region)&((h.epiweek%100)==wk)]
                        if sub.empty and wk==53:sub=h[(h.region==r.region)&(h.epiweek%100).isin([52,1])]
                        valid&=not sub.empty
                if valid:keep.update(g.sample_id)
                eligibility_groups+=1
            assert keep==set(a.sample_id)
            e=elig[(elig.fold==fold)&(elig.stage==stage)];assert set(e[e.included].sample_id)==keep and set(e.sample_id)==set(cand.sample_id)
    w=pd.read_csv(out/'quantiles_wide.csv');s=align_scores(w,pd.read_csv(out/'scores.csv'));met=pd.read_csv(out/'metrics.csv');cov=pd.read_csv(out/'coverage.csv');d=pd.read_csv(out/'asof_minus_revised.csv');levels=np.array(cfg['quantiles']);q=w[[f'q_{p:.3f}' for p in levels]].to_numpy();y=w.y_true.to_numpy();assert (np.diff(q,axis=1)>=0).all() and np.min(q)>=0 and np.max(q)<=100
    original=pd.read_csv('outputs/rolling_v0_2/predictions.csv').set_index(['sample_id','model']);key=pd.MultiIndex.from_frame(w[['sample_id','model']]);close(y,original.y_true.reindex(key))
    e=y[:,None]-q;wis=2*np.where(e>=0,levels*e,(levels-1)*e).mean(axis=1);close(s.wis,wis);close(s['median'],q[:,11]);close(s.median_error,q[:,11]-y)
    assert w[['sample_id','model','regime']].equals(s[['sample_id','model','regime']]);assert not w.duplicated(['sample_id','model','regime']).any()
    for row in met.itertuples():
        g=s[(s.regime==row.regime)&(s.model==row.model)]
        if row.scope!='overall':g=g[g['region' if row.scope=='region' else 'target_season']==row.group]
        assert len(g)==row.n;macro=lambda x:pd.DataFrame({'r':g.region,'x':np.asarray(x)}).groupby('r').x.mean().mean();err=g.y_pred-g.y_true
        close([row.macro_wis,row.point_macro_mae,row.point_rmse,row.point_bias,row.median_mae,row.median_rmse],[macro(g.wis),macro(abs(err)),np.sqrt(np.mean(err**2)),err.mean(),macro(abs(g.median_error)),np.sqrt(np.mean(g.median_error**2))])
    for row in cov.itertuples():
        g=w[(w.regime==row.regime)&(w.model==row.model)]
        if row.scope!='overall':g=g[g['region' if row.scope=='region' else 'target_season']==row.group]
        p=(1-row.nominal_coverage)/2;l=g[f'q_{p:.3f}'];u=g[f'q_{1-p:.3f}'];covered=(l<=g.y_true)&(g.y_true<=u);close([row.coverage,row.mean_width],[covered.mean(),(u-l).mean()])
    a=met[met.regime=='AsOf'].set_index(['scope','group','model']);b=met[met.regime=='MatchedRevised'].set_index(['scope','group','model'])
    for col in [c for c in d if c.startswith('delta_')]:close(d.set_index(['scope','group','model'])[col],a[col[6:]]-b[col[6:]])
    frozen=pd.read_csv('outputs/probabilistic_v0_4/quantiles_wide.csv').set_index(['sample_id','model']);g=w[w.regime=='OriginalFrozen'];ix=pd.MultiIndex.from_frame(g[['sample_id','model']]);close(g[[f'q_{p:.3f}' for p in levels]],frozen.reindex(ix)[[f'q_{p:.3f}' for p in levels]]);close(g.y_pred,frozen.y_pred.reindex(ix))
    for regime,g in w.groupby('regime'):assert set(g.sample_id)==set(w[w.regime=='AsOf'].sample_id)
    assert all(sha(f)==h for f,h in old.items());assert (out/'vintage_report.html').read_text().count('src="data:image/png;base64,')==6
    result={'status':'PASS','preserved_prior_files':len(old),'raw_vintage_records':len(records),'asof_snapshots_independently_verified':len(snaps),'window_rows_and_provenance_verified':checked_windows,'candidate_origin_groups_verified':eligibility_groups,'paired_test_samples_per_model':w[w.regime=='AsOf'].sample_id.nunique(),'independent_wis_rows':len(s),'metric_rows':len(met),'coverage_rows':len(cov),'model_reload_max_abs_error':reload_max,'matching_training_and_scaler_masks':'PASS','no_post_cutoff_values_in_AsOf':'PASS','original_frozen_context_matches':'PASS','embedded_figures':6}
    if write:(out/'verification_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='outputs/vintage_v0_5');a=ap.parse_args();verify(a.output_dir)
