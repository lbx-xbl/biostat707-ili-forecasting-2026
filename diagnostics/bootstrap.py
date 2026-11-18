"""Paired, season-stratified noncircular moving-block loss resampling."""
from pathlib import Path
import numpy as np
import pandas as pd
from epiweeks import Week

META=['sample_id','region','origin_week','origin_date','target_week','target_date','target_season','y_true']
def make_cubes(d,models,regions):
    if set(d.model)!=set(models) or models[0]!='Persistence': raise ValueError('Invalid model set/order')
    if set(d.region)!=set(regions): raise ValueError('Invalid region set')
    if d.duplicated(['sample_id','model']).any(): raise ValueError('Duplicate forecast')
    if not np.isfinite(d[['y_true','y_pred']].to_numpy()).all(): raise ValueError('Nonfinite forecast/truth')
    ref=d[d.model=='Persistence'].sort_values('sample_id')[META].reset_index(drop=True)
    for model in models:
        actual=d[d.model==model].sort_values('sample_id')[META].reset_index(drop=True)
        try: pd.testing.assert_frame_equal(ref,actual,check_dtype=False,check_exact=True)
        except AssertionError as e: raise ValueError('Unpaired forecast metadata/truth') from e
    if ((pd.to_datetime(ref.target_date)-pd.to_datetime(ref.origin_date)).dt.days!=14).any(): raise ValueError('Invalid horizon')
    for datecol,weekcol in [('origin_date','origin_week'),('target_date','target_week')]:
        expected=pd.to_datetime(ref[datecol]).map(lambda date:int(str(Week.fromdate(date.date(),system='cdc'))))
        if not np.array_equal(expected.to_numpy(),ref[weekcol].to_numpy()): raise ValueError('Calendar mismatch')
    cubes={}
    for season,g in d.groupby('target_season',sort=True):
        dates=sorted(g.target_date.unique()); dateidx=pd.to_datetime(dates)
        if len(dates)<2 or not ((dateidx[1:]-dateidx[:-1]).days==7).all(): raise ValueError('Nonconsecutive weeks')
        keys=pd.MultiIndex.from_product([dates,regions],names=['target_date','region'])
        base=g[g.model=='Persistence'].set_index(['target_date','region'])
        if base.index.duplicated().any() or set(base.index)!=set(keys): raise ValueError('Incomplete region-week panel')
        y=base.y_true.reindex(keys).to_numpy().reshape(len(dates),len(regions))
        forecasts=np.stack([g[g.model==m].set_index(['target_date','region']).y_pred.reindex(keys).to_numpy().reshape(y.shape) for m in models],axis=-1)
        if not np.isfinite(forecasts).all(): raise ValueError('Incomplete forecasts')
        cubes[season]={'errors':forecasts-y[:,:,None],'dates':dates,'y':y,'forecasts':forecasts}
    return cubes

def block_indices(n,length,replicates,seed):
    if not (isinstance(n,int) and 1<=length<=n and replicates>=1): raise ValueError('Invalid block arguments')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-length+1,size=(replicates,int(np.ceil(n/length))))
    return (starts[:,:,None]+np.arange(length)).reshape(replicates,-1)[:,:n]

def loss_sums(errors,indices):
    errors=np.asarray(errors,dtype=float); indices=np.asarray(indices)
    if errors.ndim!=3 or indices.ndim!=2 or indices.min()<0 or indices.max()>=len(errors): raise ValueError('Invalid resampling array')
    b,n=indices.shape; counts=np.zeros((b,len(errors)),dtype=float)
    np.add.at(counts,(np.arange(b)[:,None],indices),1)
    a=(counts@np.abs(errors).reshape(len(errors),-1)).reshape(b,*errors.shape[1:])
    s=(counts@(errors**2).reshape(len(errors),-1)).reshape(b,*errors.shape[1:])
    return a,s

def metric_differences(absolute_sums,squared_sums,n_weeks):
    a=np.asarray(absolute_sums); s=np.asarray(squared_sums)
    mae=(a/n_weeks).mean(axis=1)
    rmse=np.sqrt(s.sum(axis=1)/(n_weeks*s.shape[1]))
    return mae[:,1:]-mae[:,:1],rmse[:,1:]-rmse[:,:1]

def interval_summary(replicates,point):
    values=np.asarray(replicates,dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite bootstrap replicate')
    lo,hi=np.quantile(values,[.025,.975],method='linear')
    return {'estimate':float(point),'ci_lower':float(lo),'ci_upper':float(hi),
            'bootstrap_mean':float(values.mean()),'bootstrap_center_shift':float(values.mean()-point),
            'bootstrap_sd':float(values.std(ddof=1))}

def run_bootstrap(cubes,cfg,out):
    out=Path(out); seasons=list(cubes); models=cfg['models']; regions=cfg['regions']; b=cfg['bootstrap_replicates']
    groups=[('overall','all')]+[('season',s) for s in seasons]+[('region',r) for r in regions]
    lengths=[cfg['primary_block_length']]+cfg['sensitivity_block_lengths']; rows=[]; inclusion=[]; draws={}; repfiles=[]
    point_a={s:np.abs(cubes[s]['errors']).sum(axis=0)[None,:,:] for s in seasons}
    point_s={s:(cubes[s]['errors']**2).sum(axis=0)[None,:,:] for s in seasons}
    total_n=sum(len(cubes[s]['dates']) for s in seasons)
    def group_arrays(a,s,scope,group):
        if scope=='season': return a[group],s[group],len(cubes[group]['dates'])
        aa=sum(a.values()); ss=sum(s.values())
        if scope=='region':
            r=regions.index(group); aa=aa[:,r:r+1,:]; ss=ss[:,r:r+1,:]
        return aa,ss,total_n
    for length in lengths:
        aa={}; ss={}
        for season,cube in cubes.items():
            n=len(cube['dates']); index=block_indices(n,length,b,[cfg['seed'],length,int(season[:4])])
            draws[f'L{length}_{season}']=index.astype(np.int16)
            aa[season],ss[season]=loss_sums(cube['errors'],index)
            frequency=np.bincount(index.ravel(),minlength=n)/b
            for date,count in zip(cube['dates'],frequency): inclusion.append({'block_length':length,'season':season,'target_date':date,'mean_copies_per_replicate':float(count),'uniform_reference':1.})
        values=np.zeros((b,len(groups),len(models)-1,2)); point_values=np.zeros((len(groups),len(models)-1,2))
        for gi,(scope,group) in enumerate(groups):
            ga,gs,n=group_arrays(aa,ss,scope,group); ma,rm=metric_differences(ga,gs,n)
            pa,ps,pn=group_arrays(point_a,point_s,scope,group); pm,pr=metric_differences(pa,ps,pn)
            values[:,gi,:,0]=ma; values[:,gi,:,1]=rm
            point_values[gi,:,0]=pm[0]; point_values[gi,:,1]=pr[0]
            for mi,model in enumerate(models[1:]):
                for ki,metric in enumerate(['MAE','RMSE']):
                    rows.append(dict(scope=scope,group=group,model=model,metric=metric,block_length=length,
                         analysis_role='primary' if length==cfg['primary_block_length'] else 'sensitivity',
                         replicates=b,n_weeks=n,n_regions=1 if scope=='region' else len(regions),units='percentage points',
                         difference='model minus Persistence',**interval_summary(values[:,gi,mi,ki],point_values[gi,mi,ki])))
        name=f'bootstrap_replicates_L{length}.npz'; np.savez_compressed(out/name,differences=values,point_estimates=point_values,
             scopes=np.array([s for s,g in groups]),groups=np.array([g for s,g in groups]),models=np.array(models[1:]),metrics=np.array(['MAE','RMSE']))
        repfiles.append(name)
    np.savez_compressed(out/'bootstrap_indices.npz',**draws)
    pd.DataFrame(inclusion).to_csv(out/'bootstrap_week_inclusion.csv',index=False)
    result=pd.DataFrame(rows); result.to_csv(out/'bootstrap_intervals.csv',index=False)
    return result,repfiles
