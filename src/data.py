"""Official snapshot acquisition and strict complete-panel normalization."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from epiweeks import Week

URL='https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,hhs2,hhs3,hhs4,hhs5,hhs6,hhs7,hhs8,hhs9,hhs10&epiweeks=201040-201939'
def end_date(code):
    return pd.Timestamp(Week.fromstring(str(int(code)),system='cdc').enddate())

def load_data(cfg):
    p=Path('data/raw/fluview_201040_201939.json'); manifest=Path('data/source_manifest.json')
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urlopen(URL,timeout=120) as response:
            status=response.status; payload=response.read()
        p.write_bytes(payload)
        obj=json.loads(payload)
        info=dict(url=URL,fetched_at_utc=datetime.now(timezone.utc).isoformat(),http_status=status,
                  bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                  fields=sorted(obj.get('epidata',[{}])[0]),rows=len(obj.get('epidata',[])),
                  api_result=obj.get('result'),api_message=obj.get('message'),
                  source='CDC ILINet via CMU Delphi',license='Public Domain US Government',
                  license_reference='https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html',snapshot_path=str(p))
        manifest.write_text(json.dumps(info,indent=2))
    payload=p.read_bytes(); obj=json.loads(payload); info=json.loads(manifest.read_text())
    if hashlib.sha256(payload).hexdigest()!=info['sha256']: raise ValueError('Snapshot SHA-256 mismatch')
    if obj.get('result')!=1 or obj.get('message')!='success': raise ValueError(f'API error: {obj}')
    if info['http_status']!=200: raise ValueError('Non-200 source response')
    df=normalize(pd.DataFrame(obj['epidata']),cfg)
    audit=dict(status='PASS',rows=len(df),regions=cfg['regions'],weeks_per_region=df.groupby('region').size().to_dict(),
               first_week=int(df.epiweek.min()),last_week=int(df.epiweek.max()),duplicates=0,missing_wili=0,
               consecutive_day_gap=7,calendar='CDC/MMWR; Saturday week end',
               week53=sorted(int(v) for v in df.epiweek.unique() if v%100==53),
               units='wili percent; errors in percentage points',source_snapshot=info,
               release_date_range=[str(df.release_date.min()),str(df.release_date.max())],
               issue_range=[int(df.issue.min()),int(df.issue.max())])
    return df,info,audit

def normalize(raw,cfg):
    d=raw.copy()
    required={'region','epiweek','wili','release_date','issue'}
    if not required.issubset(d): raise ValueError(f'Missing fields: {required-set(d)}')
    if d.duplicated(['region','epiweek']).any(): raise ValueError('Duplicate region-week')
    if set(d.region)!=set(cfg['regions']): raise ValueError('Unexpected region coverage')
    d['wili']=pd.to_numeric(d.wili,errors='raise')
    if not np.isfinite(d.wili).all() or (d.wili<0).any(): raise ValueError('Invalid wili')
    d['week_end']=d.epiweek.map(end_date)
    start=end_date(cfg['splits']['Train'][0]); stop=end_date(cfg['splits']['Test'][1])
    expected=pd.date_range(start,stop,freq='7D')
    d['region_order']=d.region.map({r:i for i,r in enumerate(cfg['regions'])})
    d=d.sort_values(['region_order','week_end']).drop(columns='region_order').reset_index(drop=True)
    for r,g in d.groupby('region',sort=False):
        if not pd.DatetimeIndex(g.week_end).equals(expected): raise ValueError(f'Missing/extra weeks or calendar gap: {r}')
    d['season']=d.epiweek.map(lambda w:f'{w//100 if w%100>=40 else w//100-1}-{(w//100 if w%100>=40 else w//100-1)+1}')
    return d
