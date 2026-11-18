"""Availability-constrained data, with no fallback to revised observations."""
from pathlib import Path
import numpy as np
import pandas as pd
from epiweeks import Week
META=['sample_id','region','origin_week','origin_date','target_week','target_date','target_season']
def asof(records,cutoff):
    d=records[records.release_date.notna()&(records.release_date<=str(cutoff))].copy()
    return d.sort_values(['release_date','issue'],kind='stable').drop_duplicates(['region','epiweek'],keep='last').sort_values(['region','epiweek']).reset_index(drop=True)

class Archive:
    def __init__(self,records,directory=None):self.records=records;self.cache={};self.directory=Path(directory) if directory else None
    def at(self,cutoff):
        cut=str(cutoff)
        if cut not in self.cache:
            d=asof(self.records,cut);self.cache[cut]=d
            if self.directory:
                self.directory.mkdir(parents=True,exist_ok=True);d.to_csv(self.directory/f'{cut}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        return self.cache[cut]

def windows(template,panel,cutoff,label_panel=None,label_cutoff=None):
    lookup={(r.region,int(r.epiweek)):r for r in panel.itertuples()}
    labels={(r.region,int(r.epiweek)):r for r in label_panel.itertuples()} if label_panel is not None else None
    rows=[];drops=[]
    for r in template[META].itertuples(index=False):
        w=Week.fromstring(str(r.origin_week));keys=[(r.region,int(str(w+i))) for i in range(-11,1)]
        reason=None
        if not all(key in lookup for key in keys):reason='missing_history'
        elif labels is not None and (r.region,r.target_week) not in labels:reason='missing_label'
        if reason:
            drops.append(dict(sample_id=r.sample_id,region=r.region,origin_date=r.origin_date,cutoff=str(cutoff),reason=reason));continue
        if (Week.fromstring(str(r.target_week)).enddate()-w.enddate()).days!=14:raise ValueError('Horizon mismatch')
        row=r._asdict();row['feature_cutoff']=str(cutoff)
        for i,key in enumerate(keys):
            obs=lookup[key];row[f'lag_{i}']=float(obs.wili);row[f'lag_{i}_week']=obs.epiweek;row[f'lag_{i}_issue']=obs.issue;row[f'lag_{i}_release']=obs.release_date
        if labels is not None:
            lab=labels[(r.region,r.target_week)];row.update(y_true=float(lab.wili),label_issue=lab.issue,label_release=lab.release_date,label_cutoff=str(label_cutoff))
        rows.append(row)
    return pd.DataFrame(rows),drops

def complete(frame,regions):
    if frame.empty:return frame
    if frame.duplicated(['origin_date','region']).any():raise ValueError('Duplicate origin-region')
    ok=frame.groupby('origin_date').region.agg(lambda x:set(x)==set(regions));return frame[frame.origin_date.isin(ok[ok].index)].copy().reset_index(drop=True)

def seasonal(frame,panel,upper):
    h=panel[panel.epiweek.between(201040,upper)].copy();h['week']=h.epiweek%100
    table=h.groupby(['region','week']).wili.mean().to_dict();fallback=h[h.week.isin([52,1])].groupby('region').wili.mean().to_dict()
    return np.array([table.get((r.region,r.target_week%100),fallback.get(r.region,np.nan) if r.target_week%100==53 else np.nan) for r in frame.itertuples()])

def replace_values(panel,final):
    keys=pd.MultiIndex.from_frame(panel[['region','epiweek']]);g=final.set_index(['region','epiweek']).reindex(keys).reset_index()
    if g.wili.isna().any():raise ValueError('Reference snapshot missing matched observation')
    return g

def transform(panel,upper):
    d=panel[panel.epiweek.between(201040,upper)];sigma=float(d.wili.std(ddof=0))
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Invalid scaler')
    return {'mu':float(d.wili.mean()),'sigma':sigma,'n_unique_observations':len(d),'fit_last_week':upper,'fit_first_week':201040,'ddof':0}
