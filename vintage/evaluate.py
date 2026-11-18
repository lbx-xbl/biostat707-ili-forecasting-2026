from pathlib import Path
import numpy as np
import pandas as pd
from probabilistic.core import score
from probabilistic.evaluate import summaries

def evaluate(out,cfg):
    out=Path(out);f=pd.read_csv(out/'forecasts_before_evaluation.csv');truth=pd.read_csv('outputs/rolling_v0_2/predictions.csv');truth=truth[['sample_id','model','y_true']]
    f=f.merge(truth,on=['sample_id','model'],validate='many_to_one');assert len(f)==pd.read_csv(out/'forecasts_before_evaluation.csv').shape[0]
    old=pd.read_csv('outputs/probabilistic_v0_4/quantiles_wide.csv');ids=set(f.sample_id);old=old[old.sample_id.isin(ids)].copy();old['regime']='OriginalFrozen';old['issuance_date']=(pd.to_datetime(old.origin_date)+pd.Timedelta(days=7)).dt.date.astype(str)
    qcols=[f'q_{p:.3f}' for p in cfg['quantiles']];meta=['sample_id','region','origin_week','origin_date','target_week','target_date','target_season','model','y_pred','fold','regime','issuance_date','y_true']
    both=pd.concat([f[meta+qcols],old[meta+qcols]],ignore_index=True);mets=[];covs=[];scores=[]
    for regime,g in both.groupby('regime',sort=False):
        d=g[meta].copy();vals=score(g.y_true,g[qcols].to_numpy(),cfg['quantiles'])
        for k,v in vals.items():d[k]=v
        met,cov,rel,intervals=summaries(d,g,cfg['quantiles']);met['regime']=regime;cov['regime']=regime
        for i,r in met.iterrows():
            s=d[d.model==r.model]
            if r.scope!='overall':s=s[s['region' if r.scope=='region' else 'target_season']==r.group]
            err=s.y_pred-s.y_true;met.loc[i,'point_macro_mae']=pd.DataFrame({'region':s.region,'ae':abs(err)}).groupby('region').ae.mean().mean();met.loc[i,'point_rmse']=np.sqrt(np.mean(err**2));met.loc[i,'point_bias']=err.mean()
        mets.append(met);covs.append(cov);scores.append(d)
    met=pd.concat(mets,ignore_index=True);cov=pd.concat(covs,ignore_index=True);scored=pd.concat(scores,ignore_index=True)
    both.to_csv(out/'quantiles_wide.csv',index=False);scored.to_csv(out/'scores.csv',index=False);met.to_csv(out/'metrics.csv',index=False);cov.to_csv(out/'coverage.csv',index=False)
    a=met[met.regime=='AsOf'].set_index(['scope','group','model']);b=met[met.regime=='MatchedRevised'].set_index(['scope','group','model']);cols=['point_macro_mae','point_rmse','point_bias','macro_wis','median_mae','median_rmse']
    delta=(a[cols]-b[cols]).add_prefix('delta_').reset_index();delta.to_csv(out/'asof_minus_revised.csv',index=False)
    return met
