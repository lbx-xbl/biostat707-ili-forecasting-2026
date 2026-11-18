"""Scores and descriptive diagnostics for the fixed probabilistic forecasts."""
from pathlib import Path
import numpy as np
import pandas as pd
from probabilistic.core import validate_levels

def groups(d):
    yield 'overall','all',d
    for col,scope in [('target_season','season'),('region','region')]:
        for name,g in d.groupby(col,sort=True):yield scope,name,g

def macro(g,col):return float(g.groupby('region')[col].mean().mean())

def summaries(scored,wide,levels):
    p=validate_levels(levels);k=len(p)//2;metrics=[]
    for scope,name,g in groups(scored):
        for model,a in g.groupby('model',sort=False):
            row=dict(scope=scope,group=name,model=model,n=len(a),n_regions=a.region.nunique(),macro_wis=macro(a,'wis'),median_mae=macro(a.assign(ae=a.median_error.abs()),'ae'),median_rmse=float(np.sqrt(np.mean(a.median_error**2))),median_bias=macro(a,'median_error'))
            row.update({col:macro(a,col) for col in ['dispersion','overprediction','underprediction','median_component']});metrics.append(row)
    met=pd.DataFrame(metrics)
    baseline=met[met.model=='Persistence'].set_index(['scope','group']).macro_wis
    base=baseline.reindex(pd.MultiIndex.from_frame(met[['scope','group']])).to_numpy()
    met['wis_difference_vs_persistence']=met.macro_wis-base
    met['relative_wis_reduction']=np.divide(base-met.macro_wis,base,out=np.full(len(met),np.nan),where=base!=0)
    intervals=[]
    meta=[col for col in ['sample_id','region','origin_date','target_date','target_season','model','y_true'] if col in wide]
    for tau in p[:k]:
        d=wide[meta].copy();alpha=2*tau;l=wide[f'q_{tau:.3f}'].to_numpy();u=wide[f'q_{1-tau:.3f}'].to_numpy();y=wide.y_true.to_numpy()
        d['nominal_coverage']=round(1-alpha,6);d['lower']=l;d['upper']=u;d['covered']=(l<=y)&(y<=u);d['below_lower']=y<l;d['above_upper']=y>u;d['width']=u-l
        d['interval_score']=u-l+2/alpha*(np.maximum(l-y,0)+np.maximum(y-u,0));intervals.append(d)
    ints=pd.concat(intervals,ignore_index=True);coverage=[]
    for scope,name,g in groups(ints):
        for (model,level),a in g.groupby(['model','nominal_coverage']):
            coverage.append(dict(scope=scope,group=name,model=model,nominal_coverage=level,n=len(a),coverage=macro(a,'covered'),mean_width=macro(a,'width'),mean_interval_score=macro(a,'interval_score'),below_lower=macro(a,'below_lower'),above_upper=macro(a,'above_upper')))
    reliability=[]
    for scope,name,g in groups(wide):
        for model,a in g.groupby('model'):
            for tau in p:
                freq=macro(a.assign(hit=a.y_true<=a[f'q_{tau:.3f}']),'hit')
                reliability.append(dict(scope=scope,group=name,model=model,quantile=tau,observed_cdf=freq,calibration_gap=freq-tau,n=len(a)))
    return met,pd.DataFrame(coverage),pd.DataFrame(reliability),ints

def paired_means(cube,indices):
    x=np.asarray(cube,float);idx=np.asarray(indices)
    if x.ndim!=3 or idx.ndim!=2 or idx.shape[1]!=len(x) or not np.issubdtype(idx.dtype,np.integer) or idx.min()<0 or idx.max()>=len(x):raise ValueError('Invalid cube/indices')
    count=np.zeros((len(idx),len(x)),dtype=np.int64)
    np.add.at(count,(np.arange(len(idx))[:,None],idx),1)
    return (count@x.reshape(len(x),-1)/len(x)).reshape(len(idx),*x.shape[1:])

def bootstrap(scored,cfg,out):
    out=Path(out);models=cfg['models'];regions=cfg['regions'];seasons=cfg['seasons'];cubes={}
    draws=np.load(cfg['bootstrap_indices_source'])
    for season in seasons:
        g=scored[scored.target_season==season];dates=sorted(g.target_date.unique())
        key=pd.MultiIndex.from_product([dates,regions,models],names=['target_date','region','model'])
        x=g.set_index(['target_date','region','model']).wis.reindex(key).to_numpy().reshape(len(dates),len(regions),len(models))
        if not np.isfinite(x).all():raise ValueError('Incomplete WIS panel')
        cubes[season]=x
    names=['all']+seasons+regions;scopes=['overall']+['season']*len(seasons)+['region']*len(regions)
    rows=[];n_total=sum(len(x) for x in cubes.values())
    for length in cfg['block_lengths']:
        samples={s:paired_means(cubes[s],draws[f'L{length}_{s}']) for s in seasons}
        overall=sum(samples[s]*len(cubes[s]) for s in seasons)/n_total
        point=sum(cubes[s].sum(axis=0) for s in seasons)/n_total
        means=[overall.mean(axis=1)]+[samples[s].mean(axis=1) for s in seasons]+[overall[:,r,:] for r in range(len(regions))]
        points=[point.mean(axis=0)]+[cubes[s].mean(axis=(0,1)) for s in seasons]+[point[r,:] for r in range(len(regions))]
        delta=np.stack([m[:,1:]-m[:,:1] for m in means],axis=1)
        pts=np.stack([m[1:]-m[:1] for m in points])
        for gi,(scope,name) in enumerate(zip(scopes,names)):
            for mi,model in enumerate(models[1:]):
                a=delta[:,gi,mi];lo,hi=np.quantile(a,[.025,.975],method='linear')
                rows.append(dict(scope=scope,group=name,model=model,metric='WIS',block_length=length,estimate=pts[gi,mi],ci_lower=lo,ci_upper=hi,bootstrap_mean=a.mean(),bootstrap_center_shift=a.mean()-pts[gi,mi],bootstrap_sd=a.std(ddof=1),replicates=len(a)))
        np.savez_compressed(out/f'wis_bootstrap_replicates_L{length}.npz',differences=delta,point_estimates=pts,models=np.array(models[1:]),groups=np.array(names),scopes=np.array(scopes))
    result=pd.DataFrame(rows);result.to_csv(out/'wis_bootstrap_intervals.csv',index=False);return result
