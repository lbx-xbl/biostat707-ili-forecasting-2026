"""The sole definition of rolling windows and origin/target split membership."""
import numpy as np
import pandas as pd
from .data import end_date

LAGS=[f'lag_{i}' for i in range(12)]
def build_windows(d,cfg):
    rows=[]; lookback=cfg['lookback']; horizon=cfg['horizon']
    for region,g in d.groupby('region',sort=False):
        g=g.reset_index(drop=True)
        for i in range(lookback-1,len(g)-horizon):
            origin=g.iloc[i]; target=g.iloc[i+horizon]
            for split,(lo,hi) in cfg['splits'].items():
                if end_date(lo)<=origin.week_end<=target.week_end<=end_date(hi):
                    history=g.iloc[i-lookback+1:i+1]
                    assert history.week_end.max()==origin.week_end
                    assert (target.week_end-origin.week_end).days==7*horizon
                    row=dict(sample_id=f'{region}_{origin.epiweek}_{target.epiweek}',region=region,
                             origin_week=int(origin.epiweek),origin_date=origin.week_end,
                             target_week=int(target.epiweek),target_date=target.week_end,
                             target_season=target.season,split=split,y_true=float(target.wili))
                    row.update({f'lag_{j}':float(v) for j,v in enumerate(history.wili)})
                    rows.append(row)
    x=pd.DataFrame(rows)
    if x.sample_id.duplicated().any(): raise ValueError('Duplicate window IDs')
    summary=x.groupby(['split','region'],sort=False).agg(n=('sample_id','size'),origin_min=('origin_week','min'),origin_max=('origin_week','max'),target_min=('target_week','min'),target_max=('target_week','max')).reset_index()
    return x,summary

def fit_transform(d,cfg):
    lo,hi=cfg['splits']['Train']
    t=d[d.epiweek.between(lo,hi)]
    if t.duplicated(['region','epiweek']).any(): raise ValueError('Nonunique scaler observations')
    mu=float(t.wili.mean()); sigma=float(t.wili.std(ddof=0))
    if not np.isfinite(sigma) or sigma<=0: raise ValueError('Invalid train sigma')
    return dict(mu=mu,sigma=sigma,n_unique_observations=len(t),fit_split='Train',ddof=0)

def features(x,transform,regions):
    hist=(x[LAGS].to_numpy(dtype=float)-transform['mu'])/transform['sigma']
    idx=x.region.map({r:i for i,r in enumerate(regions)}).to_numpy()
    days=(pd.to_datetime(x.target_date)-pd.Timestamp('2000-01-01')).dt.days.to_numpy()
    phase=2*np.pi*days/365.2425
    static=np.column_stack([np.eye(10)[idx],np.sin(phase),np.cos(phase)])
    return np.column_stack([hist,static])
