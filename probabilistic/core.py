"""No outer labels enter distribution estimation or forecast construction."""
import numpy as np
import pandas as pd
from diagnostics.bootstrap import make_cubes

META=['sample_id','region','origin_week','origin_date','target_week','target_date','target_season','model','y_pred','fold']
def validate_levels(levels):
    p=np.asarray(levels,dtype=float)
    if p.ndim!=1 or len(p)<3 or len(p)%2!=1 or not np.isfinite(p).all() or not ((p>0)&(p<1)).all() or not (np.diff(p)>0).all():
        raise ValueError('Quantile levels must be finite, unique, sorted and inside (0,1)')
    if p[len(p)//2]!=.5 or not np.allclose(p,1-p[::-1],atol=1e-12,rtol=0):raise ValueError('Symmetric levels with one median required')
    return p

def score(y,q,levels):
    p=validate_levels(levels); y=np.asarray(y,dtype=float);q=np.asarray(q,dtype=float)
    if y.ndim!=1 or q.shape!=(len(y),len(p)) or not np.isfinite(y).all() or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():raise ValueError('Invalid outcome or quantile array')
    k=len(p)//2;denom=k+.5;l=q[:,:k];u=q[:,-1:-k-1:-1];weights=p[:k]
    dispersion=((u-l)*weights).sum(axis=1)/denom
    over=np.maximum(l-y[:,None],0).sum(axis=1)/denom
    under=np.maximum(y[:,None]-u,0).sum(axis=1)/denom
    med=.5*np.abs(y-q[:,k])/denom
    return {'wis':dispersion+over+under+med,'dispersion':dispersion,'overprediction':over,'underprediction':under,'median_component':med,'median':q[:,k],'median_error':q[:,k]-y}

def origins_only(origins,cfg):
    if not set(META).issubset(origins):raise ValueError('Missing forecast metadata')
    d=origins[META].copy()
    make_cubes(d.assign(y_true=0.),cfg['models'],cfg['regions'])
    if not (d.fold==d.target_season).all():raise ValueError('Forecast fold mismatch')
    return d

def residual_quantiles(validation,origins,cfg,season):
    p=validate_levels(cfg['quantiles']);o=origins_only(origins,cfg)
    if set(o.target_season)!={season}:raise ValueError('Wrong outer season')
    make_cubes(validation,cfg['models'],cfg['regions'])
    year=int(season[:4]);prev=f'{year-1}-{year}'
    if set(validation.target_season)!={prev} or set(validation.fold)!={season}:raise ValueError('Wrong previous validation season/fold')
    if ((validation.y_true<0)|(validation.y_true>100)).any():raise ValueError('ILI truth outside percent support')
    if pd.to_datetime(validation.target_date).max()>=pd.to_datetime(o.origin_date).min():raise ValueError('Residual target not before test origin')
    if set(validation.sample_id)&set(o.sample_id):raise ValueError('Residual/test overlap')
    # Reuse exact chronological range definition, not only human-readable labels.
    for d,lo,hi in [(validation,(year-1)*100+40,year*100+39),(o,year*100+40,(year+1)*100+39)]:
        if not (d.origin_week.between(lo,hi)&d.target_week.between(lo,hi)).all():raise ValueError('Origin/target season boundary violation')
    rows=[]
    for model in cfg['models']:
        v=validation[validation.model==model];r=(v.y_true-v.y_pred).to_numpy()
        q=np.quantile(r,p,method='linear')
        for tau,value in zip(p,q):rows.append({'fold':season,'model':model,'quantile':float(tau),'residual_quantile':float(value),'n_residuals':len(v),'n_weeks':v.target_date.nunique(),'source_season':prev,'source_first_target':v.target_date.min(),'source_last_target':v.target_date.max(),'first_test_origin':o.origin_date.min()})
    return pd.DataFrame(rows)

def forecast(origins,offsets,cfg):
    p=validate_levels(cfg['quantiles']);d=origins_only(origins,cfg)
    if offsets.duplicated(['fold','model','quantile']).any():raise ValueError('Duplicate residual quantile')
    expected={(fold,model,float(tau)) for fold in d.fold.unique() for model in cfg['models'] for tau in p}
    if set(map(tuple,offsets[['fold','model','quantile']].to_numpy()))!=expected:raise ValueError('Residual quantile keys do not match')
    for _,g in offsets.groupby(['fold','model']):
        vals=g.sort_values('quantile').residual_quantile.to_numpy()
        if not np.isfinite(vals).all() or (np.diff(vals)<0).any():raise ValueError('Invalid residual quantiles')
    index=pd.MultiIndex.from_frame(d[['fold','model']]);lo,hi=cfg['bounds']
    if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('Invalid support bounds')
    for tau in p:
        shift=offsets[offsets['quantile']==tau].set_index(['fold','model']).residual_quantile.reindex(index).to_numpy()
        raw=d.y_pred.to_numpy()+shift
        d[f'raw_q_{tau:.3f}']=raw;d[f'q_{tau:.3f}']=np.clip(raw,lo,hi)
    return d
