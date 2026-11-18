"""Rolling folds reuse the unchanged pilot's CDC window construction."""
import copy
import numpy as np
import pandas as pd
from src.windows import build_windows,fit_transform

def all_windows(panel,cfg):
    c=copy.deepcopy(cfg); c['splits']={'All':[201040,201939]}
    return build_windows(panel,c)[0].drop(columns='split')

def fold_frames(w,season):
    ranges={'train':(201040,(season-1)*100+39),
            'validation':((season-1)*100+40,season*100+39),
            'refit':(201040,season*100+39),
            'test':(season*100+40,(season+1)*100+39)}
    return {k:w[w.origin_week.between(a,b) & w.target_week.between(a,b)].copy().reset_index(drop=True) for k,(a,b) in ranges.items()}

def scaler(panel,upper):
    t=fit_transform(panel,{'splits':{'Train':[201040,upper]}})
    t.update(fit_first_week=201040,fit_last_week=int(upper))
    return t

def seasonal_table(panel,season,regions):
    # Only complete seasons before this fold's test/validation season.
    hist=panel[panel.epiweek.between(201040,season*100+39)].copy()
    hist['week']=hist.epiweek%100
    rows=[]
    for region in regions:
        g=hist[hist.region==region]
        for week in range(1,54):
            a=g[g.week==week]; method='historical_same_week'
            if a.empty and week==53:
                a=g[g.week.isin([52,1])]; method='week52_week1_fallback'
            if a.empty: raise ValueError(f'No historical seasonal reference: {region}, week {week}')
            rows.append(dict(region=region,week=week,y_pred=float(a.wili.mean()),n_observations=len(a),
                             source_first_week=int(a.epiweek.min()),source_last_week=int(a.epiweek.max()),method=method))
    return pd.DataFrame(rows)

def seasonal_predict(frame,table):
    keys=pd.MultiIndex.from_arrays([frame.region,frame.target_week%100],names=['region','week'])
    result=table.set_index(['region','week']).y_pred.reindex(keys).to_numpy()
    if not np.isfinite(result).all(): raise ValueError('Unmatched seasonal predictions')
    return result
