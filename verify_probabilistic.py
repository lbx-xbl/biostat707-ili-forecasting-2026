#!/usr/bin/env python3
"""Independent order-statistic, pinball, interval and direct-resampling verification."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='4'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-11,equal_nan=True)
def mean_region(d,x):
    return float(pd.Series(np.asarray(x),index=d.index).groupby(d.region).mean().mean())
def verify(out,write=True):
    out=Path(out);cfg=json.loads((out/'config_frozen.json').read_text());p=np.array(cfg['quantiles']);k=11;models=cfg['models'];regions=cfg['regions'];seasons=cfg['seasons']
    assert len(p)==23 and np.all(np.diff(p)>0) and p[11]==.5;close(p,1-p[::-1])
    prior=json.loads(Path('docs/probabilistic_preservation_baseline.json').read_text());assert all(sha(f)==h for f,h in prior.items())
    run=json.loads((out/'run_summary.json').read_text());assert sha(out/'config_frozen.json')==run['config_sha256'];assert sha(out/'protocol_frozen.md')==run['protocol_sha256']
    assert sha(out/'protocol_frozen.md')==sha('docs/probabilistic_protocol_v0.4.md')
    sources=json.loads((out/'input_manifest.json').read_text())
    for source,h in sources.items():
        assert sha(source)==h
        if source.endswith('/predictions.csv'):copy=out/'inputs/original_point_predictions.csv'
        elif source.endswith('bootstrap_indices.npz'):copy=out/'inputs/bootstrap_indices.npz'
        else:copy=out/'inputs'/Path(source).parent.name/Path(source).name
        assert sha(copy)==h
    freeze=json.loads((out/'forecast_freeze.json').read_text());assert freeze['outcomes_joined_after_freeze'];assert all(sha(out/f)==h for f,h in freeze['sha256'].items())
    raw=pd.read_csv(out/'quantiles_before_evaluation.csv');wide=pd.read_csv(out/'quantiles_wide.csv');offset=pd.read_csv(out/'residual_quantiles.csv');resid=pd.read_csv(out/'validation_residuals.csv');audit=pd.read_csv(out/'temporal_audit.csv')
    original=pd.read_csv(out/'inputs/original_point_predictions.csv');keys=['sample_id','model'];idx=pd.MultiIndex.from_frame(wide[keys]);assert idx.is_unique and len(wide)==15050
    assert len(offset)==6*5*23 and len(audit)==6 and len(resid)==15050
    assert 'y_true' not in raw
    a=wide.set_index(keys).sort_index();z=original.set_index(keys).sort_index();assert a.index.equals(z.index)
    pd.testing.assert_frame_equal(a[z.columns],z,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    pd.testing.assert_frame_equal(raw,wide.drop(columns='y_true'),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    maximum_quantile_error=0.
    for season in seasons:
        src=out/'inputs'/season;v=pd.read_csv(src/'validation_predictions.csv');f=pd.read_csv(src/'forecasts_before_evaluation.csv');yr=int(season[:4])
        assert 'y_true' not in f and set(v.target_season)=={f'{yr-1}-{yr}'} and set(v.fold)=={season}
        assert v.target_date.max()<f.origin_date.min() and not(set(v.sample_id)&set(f.sample_id))
        selection=json.loads((src/'model_selection.json').read_text());assert selection['selection_transform']['fit_last_week']==(yr-1)*100+39
        assert selection['refit_transform']['fit_last_week']==yr*100+39
        assert v.target_week.max()<=yr*100+39 and f.origin_week.min()>=yr*100+40
        for model in models:
            vm=v[v.model==model];sorted_errors=np.sort((vm.y_true-vm.y_pred).to_numpy());h=(len(sorted_errors)-1)*p;lo=np.floor(h).astype(int);hi=np.ceil(h).astype(int)
            estimated=sorted_errors[lo]+(h-lo)*(sorted_errors[hi]-sorted_errors[lo])
            saved=offset[(offset.fold==season)&(offset.model==model)].sort_values('quantile');close(saved.residual_quantile,estimated)
            rr=resid[(resid.fold==season)&(resid.model==model)].sort_values('sample_id');vv=vm.sort_values('sample_id');close(rr.residual, vv.y_true.to_numpy()-vv.y_pred.to_numpy())
            g=wide[(wide.fold==season)&(wide.model==model)];expected_raw=g.y_pred.to_numpy()[:,None]+estimated;expected=np.minimum(100,np.maximum(0,expected_raw))
            actual=g[[f'q_{t:.3f}' for t in p]].to_numpy();close(actual,expected);close(g[[f'raw_q_{t:.3f}' for t in p]],expected_raw)
            maximum_quantile_error=max(maximum_quantile_error,float(np.max(np.abs(actual-expected))))
            assert set(vm.region)==set(regions) and vm.groupby('region').size().nunique()==1
            ar=audit[audit.fold==season].iloc[0];assert ar.validation_rows_per_model==len(vm) and ar.outer_rows_per_model==len(g)
        # Actual-label perturbation check for construction, separately from independent arithmetic above.
        from probabilistic.core import forecast
        offsets=offset[offset.fold==season]
        pd.testing.assert_frame_equal(forecast(f,offsets,cfg),forecast(f.assign(y_true=np.arange(len(f))*100.),offsets,cfg))
    q=wide[[f'q_{t:.3f}' for t in p]].to_numpy();y=wide.y_true.to_numpy();assert np.isfinite(q).all() and np.min(q)>=0 and np.max(q)<=100 and (np.diff(q,axis=1)>=0).all()
    e=y[:,None]-q;pinball=np.where(e>=0,p*e,(p-1)*e);independent_wis=2*pinball.mean(axis=1)
    s=pd.read_csv(out/'scores.csv');pd.testing.assert_frame_equal(s[keys],wide[keys]);close(s.wis,independent_wis)
    close(s['median'],q[:,11]);close(s.median_error,q[:,11]-y)
    widths=[];overs=[];unders=[]
    for j in range(11):
        alpha=2*p[j];low=q[:,j];high=q[:,-1-j]
        widths.append(alpha/2*(high-low)/11.5);overs.append(np.where(y<low,low-y,0)/11.5);unders.append(np.where(y>high,y-high,0)/11.5)
    close(s.dispersion,np.sum(widths,axis=0));close(s.overprediction,np.sum(overs,axis=0));close(s.underprediction,np.sum(unders,axis=0));close(s.median_component,np.abs(y-q[:,11])/23)
    close(s.wis,s[['dispersion','overprediction','underprediction','median_component']].sum(axis=1))
    long=pd.read_csv(out/'quantile_predictions.csv');assert len(long)==346150 and not long.duplicated(keys+['quantile']).any()
    for tau in p:
        l=long[long['quantile']==tau].set_index(keys).reindex(idx)
        close(l.value,wide[f'q_{tau:.3f}']);close(l.raw_value,wide[f'raw_q_{tau:.3f}']);close(l.y_true,y)
        assert np.array_equal(l.clipped.to_numpy(),(l.value!=l.raw_value).to_numpy())
    ints=pd.read_csv(out/'interval_scores.csv');assert len(ints)==165550
    for j,tau in enumerate(p[:11]):
        c=round(1-2*tau,6);d=ints[ints.nominal_coverage==c].set_index(keys).reindex(idx);low=q[:,j];high=q[:,-1-j]
        close(d.lower,low);close(d.upper,high);close(d.width,high-low);close(d.interval_score,high-low+(np.maximum(low-y,0)+np.maximum(y-high,0))/tau)
        assert np.array_equal(d.covered.to_numpy(),(low<=y)&(y<=high));assert np.array_equal(d.below_lower.to_numpy(),y<low);assert np.array_equal(d.above_upper.to_numpy(),y>high)
    met=pd.read_csv(out/'metrics.csv');cov=pd.read_csv(out/'coverage.csv');rel=pd.read_csv(out/'quantile_reliability.csv');assert len(met)==85 and len(cov)==935 and len(rel)==1955
    def subset(d,scope,group,model):
        a=d[d.model==model]
        return a if scope=='overall' else a[a['region' if scope=='region' else 'target_season']==group]
    for row in met.itertuples():
        d=subset(s,row.scope,row.group,row.model);assert len(d)==row.n
        close(row.macro_wis,mean_region(d,d.wis));close(row.median_mae,mean_region(d,np.abs(d.median_error)));close(row.median_rmse,np.sqrt(np.mean(d.median_error**2)));close(row.median_bias,mean_region(d,d.median_error))
        for col in ['dispersion','overprediction','underprediction','median_component']:close(getattr(row,col),mean_region(d,d[col]))
        baseline=subset(s,row.scope,row.group,'Persistence');base=mean_region(baseline,baseline.wis)
        close(row.wis_difference_vs_persistence,row.macro_wis-base);close(row.relative_wis_reduction,(base-row.macro_wis)/base)
    for row in cov.itertuples():
        d=subset(ints,row.scope,row.group,row.model);d=d[d.nominal_coverage==row.nominal_coverage];assert len(d)==row.n
        for col,saved in [('covered','coverage'),('width','mean_width'),('interval_score','mean_interval_score'),('below_lower','below_lower'),('above_upper','above_upper')]:close(getattr(row,saved),mean_region(d,d[col]))
    for row in rel.itertuples():
        d=subset(wide,row.scope,row.group,row.model);actual=mean_region(d,d.y_true<=d[f'q_{row.quantile:.3f}']);close(row.observed_cdf,actual);close(row.calibration_gap,actual-row.quantile)
    bs=pd.read_csv(out/'wis_bootstrap_intervals.csv');draws=np.load(out/'inputs/bootstrap_indices.npz');cubes={};dates={}
    for season in seasons:
        d=s[s.target_season==season];dates[season]=sorted(d.target_date.unique());grid=pd.MultiIndex.from_product([dates[season],regions,models],names=['target_date','region','model']);cubes[season]=d.set_index(['target_date','region','model']).wis.reindex(grid).to_numpy().reshape(-1,10,5)
        actual_dates=pd.to_datetime(dates[season]);assert ((actual_dates[1:]-actual_dates[:-1]).days==7).all()
    checked=0;n_all=sum(len(c) for c in cubes.values())
    for length in cfg['block_lengths']:
        saved=np.load(out/f'wis_bootstrap_replicates_L{length}.npz');a={}
        for season in seasons:
            x=cubes[season];n=len(x);ix=draws[f'L{length}_{season}'];assert ix.shape==(5000,n)
            rng=np.random.default_rng(np.random.SeedSequence([2026,length,int(season[:4])]))
            starts=rng.integers(0,n-length+1,size=(5000,int(np.ceil(n/length))));expected=(starts[:,:,None]+np.arange(length)).reshape(5000,-1)[:,:n];np.testing.assert_array_equal(ix,expected)
            # Independent direct indexing, not the count-matrix producer.
            a[season]=np.concatenate([x[ix[j:j+200]].mean(axis=1) for j in range(0,5000,200)])
        overall=sum(a[s]*len(cubes[s]) for s in seasons)/n_all
        means=[overall.mean(axis=1)]+[a[s].mean(axis=1) for s in seasons]+[overall[:,i,:] for i in range(10)]
        delta=np.stack([x[:,1:]-x[:,:1] for x in means],axis=1);close(saved['differences'],delta);checked+=delta.size
        names=['all']+seasons+regions;scopes=['overall']+['season']*6+['region']*10
        for gi,(scope,group) in enumerate(zip(scopes,names)):
            for mi,model in enumerate(models[1:]):
                row=bs[(bs.block_length==length)&(bs.scope==scope)&(bs.group==group)&(bs.model==model)].iloc[0];d=delta[:,gi,mi];bounds=np.quantile(d,[.025,.975],method='linear');point=met[(met.scope==scope)&(met.group==group)&(met.model==model)].wis_difference_vs_persistence.iloc[0]
                close([row.estimate,row.ci_lower,row.ci_upper,row.bootstrap_mean,row.bootstrap_center_shift,row.bootstrap_sd],[point,*bounds,d.mean(),d.mean()-point,d.std(ddof=1)]);close(saved['point_estimates'][gi,mi],point)
    assert len(bs)==204 and checked==1020000
    html=(out/'probabilistic_report.html').read_text();assert html.count('src="data:image/png;base64,')==7
    assert all(sha(f)==h for f,h in prior.items())
    result={'status':'PASS','preserved_prior_files':len(prior),'point_forecasts_unchanged':15050,'empirical_distributions_independently_recomputed':30,'quantile_values_verified':346150,'max_quantile_error':maximum_quantile_error,'independent_pinball_wis_rows':len(s),'interval_scores_verified':len(ints),'group_metrics_verified':len(met),'coverage_rows_verified':len(cov),'reliability_rows_verified':len(rel),'bootstrap_interval_rows_verified':len(bs),'bootstrap_difference_values_verified':checked,'outer_label_perturbation':'PASS','residual_targets_before_outer_origins':'PASS','embedded_figures':7,'point_models_retrained':0}
    if write:(out/'verification_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='outputs/probabilistic_v0_4');args=ap.parse_args();verify(args.output_dir)
