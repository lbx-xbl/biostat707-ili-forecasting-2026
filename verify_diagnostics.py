#!/usr/bin/env python3
"""Independent arithmetic, PAV, direct-index bootstrap and preservation checks."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[name]='4'
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_pav(x,y):
    # Exact ties first grouped, then a separately implemented weighted PAV stack.
    unique,inverse=np.unique(x,return_inverse=True)
    counts=np.bincount(inverse); sums=np.bincount(inverse,weights=y)
    blocks=[]
    for i,(n,total) in enumerate(zip(counts,sums)):
        blocks.append([i,i+1,int(n),float(total)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            b=blocks.pop(); a=blocks.pop(); blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    z=np.empty(len(unique))
    for a,b,n,total in blocks: z[a:b]=total/n
    return z[inverse]

def verify(out,write=True):
    out=Path(out); cfg=json.loads((out/'config_frozen.json').read_text()); summary=json.loads((out/'run_summary.json').read_text())
    assert sha(out/'config_frozen.json')==summary['config_sha256']
    assert sha(out/'protocol_frozen.md')==summary['protocol_sha256']
    assert sha(out/'input_predictions.csv')==summary['input_predictions_sha256']==sha(cfg['input_predictions'])
    old=json.loads(Path('docs/diagnostics_preservation_baseline.json').read_text())
    assert all(Path(p).exists() and sha(p)==h for p,h in old.items()),'Prior artifact modified'
    d=pd.read_csv(out/'input_predictions.csv'); rows=pd.read_csv(out/'bootstrap_intervals.csv'); c=pd.read_csv(out/'calibration_components.csv'); pts=pd.read_csv(out/'calibration_diagnostic_points.csv')
    assert len(d)==15050 and not d.duplicated(['sample_id','model']).any()
    regions=cfg['regions']; models=cfg['models']; seasons=cfg['seasons']; e={}; point_abs={}; point_sq={}; dates={}
    for season in seasons:
        g=d[d.target_season==season]; dates[season]=sorted(g.target_date.unique())
        assert (np.diff(pd.to_datetime(dates[season]).to_numpy()).astype('timedelta64[D]').astype(int)==7).all()
        index=pd.MultiIndex.from_product([dates[season],regions],names=['target_date','region'])
        y=g[g.model=='Persistence'].set_index(['target_date','region']).y_true.reindex(index).to_numpy().reshape(-1,10)
        pred=np.stack([g[g.model==m].set_index(['target_date','region']).y_pred.reindex(index).to_numpy().reshape(-1,10) for m in models],axis=-1)
        e[season]=pred-y[:,:,None]; point_abs[season]=np.abs(e[season]).sum(axis=0)[None]; point_sq[season]=(e[season]**2).sum(axis=0)[None]
    indexfile=np.load(out/'bootstrap_indices.npz'); inclusion=pd.read_csv(out/'bootstrap_week_inclusion.csv'); checked_replicates=0
    for length in [8,4,13]:
        arrays=np.load(out/f'bootstrap_replicates_L{length}.npz'); ba={}; bs={}
        for season in seasons:
            index=indexfile[f'L{length}_{season}']; n=len(e[season]); b=cfg['bootstrap_replicates']
            assert index.shape==(b,n) and index.min()>=0 and index.max()<n
            for start in range(0,n,length): assert (np.diff(index[:,start:min(start+length,n)],axis=1)==1).all()
            rng=np.random.default_rng([cfg['seed'],length,int(season[:4])]); starts=rng.integers(0,n-length+1,size=(b,(n+length-1)//length))
            expected=np.concatenate([starts[:,k:k+1]+np.arange(length) for k in range(starts.shape[1])],axis=1)[:,:n]
            np.testing.assert_array_equal(index,expected)
            inc=inclusion[(inclusion.season==season)&(inclusion.block_length==length)].sort_values('target_date')
            np.testing.assert_allclose(inc.mean_copies_per_replicate,np.bincount(index.ravel(),minlength=n)/b,atol=1e-14)
            # Direct resampling differs from producer's count-matrix multiplication.
            ba[season]=np.empty((b,10,5)); bs[season]=np.empty((b,10,5))
            for first in range(0,b,100):
                chunk=e[season][index[first:first+100]]
                ba[season][first:first+100]=np.abs(chunk).sum(axis=1)
                bs[season][first:first+100]=(chunk**2).sum(axis=1)
        def calc(a,s,scope,group):
            if scope=='season': aa=a[group]; ss=s[group]; n=len(e[group])
            else:
                aa=sum(a.values()); ss=sum(s.values()); n=sum(len(v) for v in e.values())
                if scope=='region':
                    r=regions.index(group); aa=aa[:,r:r+1,:]; ss=ss[:,r:r+1,:]
            ma=aa.mean(axis=1)/n; rm=np.sqrt(ss.mean(axis=1)/n)
            return np.stack([ma[:,1:]-ma[:,:1],rm[:,1:]-rm[:,:1]],axis=-1)
        for gi,(scope,group) in enumerate(zip(arrays['scopes'],arrays['groups'])):
            values=calc(ba,bs,scope,group); point=calc(point_abs,point_sq,scope,group)[0]
            np.testing.assert_allclose(values,arrays['differences'][:,gi],atol=2e-12,rtol=1e-12)
            np.testing.assert_allclose(point,arrays['point_estimates'][gi],atol=1e-12,rtol=1e-12)
            for mi,model in enumerate(models[1:]):
                for ki,metric in enumerate(['MAE','RMSE']):
                    r=rows[(rows.scope==scope)&(rows.group==group)&(rows.model==model)&(rows.metric==metric)&(rows.block_length==length)].iloc[0]
                    q=np.quantile(values[:,mi,ki],[.025,.975],method='linear')
                    np.testing.assert_allclose([r.estimate,r.ci_lower,r.ci_upper,r.bootstrap_mean,r.bootstrap_center_shift],[point[mi,ki],q[0],q[1],values[:,mi,ki].mean(),values[:,mi,ki].mean()-point[mi,ki]],atol=2e-12,rtol=1e-12)
            checked_replicates+=values.size
    max_pav_error=0.
    for row in c.itertuples():
        g=d[d.model==row.model]
        if row.scope=='season': g=g[g.target_season==row.group]
        elif row.scope=='region': g=g[g.region==row.group]
        x=g.y_pred.to_numpy(); y=g.y_true.to_numpy(); z=independent_pav(x,y)
        actual=pts[(pts.scope==row.scope)&(pts.group==row.group)&(pts.model==row.model)].set_index('sample_id').loc[g.sample_id]
        np.testing.assert_allclose(actual.y_pred,x,atol=1e-12,rtol=0)
        np.testing.assert_allclose(actual.y_true,y,atol=1e-12,rtol=0)
        np.testing.assert_allclose(actual.isotonic_diagnostic_mean,z,atol=1e-10,rtol=1e-12)
        max_pav_error=max(max_pav_error,float(np.max(np.abs(actual.isotonic_diagnostic_mean-z))))
        s=np.mean((x-y)**2); iso=np.mean((z-y)**2); unc=np.mean((y-y.mean())**2); bias=np.mean(x-y)
        np.testing.assert_allclose([row.mse,row.mse_isotonic_diagnostic,row.unc,row.mcb,row.dsc,row.bias,row.mcb_unconditional,row.mcb_conditional,row.root_mcb],[s,iso,unc,s-iso,unc-iso,bias,bias**2,s-iso-bias**2,np.sqrt(max(s-iso,0))],atol=1e-10,rtol=1e-12)
        assert row.mcb>=-1e-10 and row.dsc>=-1e-10 and row.mcb_conditional>=-1e-10
        assert abs(row.mse-row.mcb+row.dsc-row.unc)<1e-10
        if unc: assert abs(row.skill_vs_evaluation_mean-(1-s/unc))<1e-10
        else: assert np.isnan(row.skill_vs_evaluation_mean)
        # Group PAV block residual sums must vanish at each fitted mean level.
        for val in np.unique(z): assert abs((y[z==val]-val).sum())<1e-8
    assert len(c)==85 and len(rows)==408 and len(pts)==45150
    for (scope,group),g in c.groupby(['scope','group']): assert g.unc.max()-g.unc.min()<1e-12
    oldmetrics=pd.read_csv('outputs/rolling_v0_2/metrics.csv')
    for row in rows[(rows.scope=='overall')&(rows.block_length==8)].itertuples():
        original=oldmetrics[oldmetrics.scope=='overall'].set_index('model'); metric='macro_mae' if row.metric=='MAE' else 'pooled_rmse'
        assert abs(row.estimate-(original.loc[row.model,metric]-original.loc['Persistence',metric]))<1e-12
    report=(out/'diagnostics_report.html').read_text(); assert report.count('src="data:image/png;base64,')==7
    assert all(sha(p)==h for p,h in old.items())
    result={'status':'PASS','prior_files_unchanged':len(old),'original_forecast_rows':len(d),'interval_rows_recomputed':len(rows),'replicate_difference_values_recomputed':checked_replicates,'decompositions_independently_checked':len(c),'diagnostic_points_checked':len(pts),'maximum_independent_pav_error':max_pav_error,'paired_seasonal_blocks_and_deterministic_seeds':'PASS','original_point_metrics_match_v0_2':'PASS','protocol_input_hashes':'PASS','embedded_figures':7,'forecast_models_retrained':0}
    if write: (out/'verification_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); return result
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default='outputs/diagnostics_v0_3'); args=ap.parse_args(); verify(args.output_dir)
