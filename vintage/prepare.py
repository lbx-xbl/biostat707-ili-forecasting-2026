"""Construct availability masks before any fit, shared by both regimes."""
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from epiweeks import Week
from rolling.data import fold_frames
from vintage.data import Archive,windows,complete,seasonal,replace_values,transform,META

def cutoff(year):return (Week(year,40).enddate()+timedelta(days=7)).isoformat()
def prepare(cfg,out):
    out=Path(out);records=pd.read_csv(cfg['vintage_records']);final=pd.read_csv(cfg['reference_snapshot']);template=pd.read_csv('outputs/rolling_v0_2/all_windows.csv')
    archive=Archive(records,out/'asof_snapshots');eligibility=[];plans={}
    for year in cfg['test_season_starts']:
        fold=f'{year}-{year+1}';cuts={'selection':cutoff(year-1),'refit':cutoff(year)};panels={k:archive.at(v) for k,v in cuts.items()};t=fold_frames(template,year);frames={}
        for stage in ['train','validation','refit','test']:
            rows=[];reasons={}
            groups=[(cuts['selection' if stage=='train' else 'refit'],t[stage])] if stage in ['train','refit'] else [((pd.Timestamp(origin)+pd.Timedelta(days=7)).date().isoformat(),g) for origin,g in t[stage].groupby('origin_date',sort=True)]
            for cut,g in groups:
                fp=archive.at(cut)
                lp=None if stage=='test' else panels['selection' if stage=='train' else 'refit'];lc=None if stage=='test' else cuts['selection' if stage=='train' else 'refit']
                f,dropped=windows(g,fp,cut,lp,lc);rows.append(f);reasons.update({r['sample_id']:r['reason'] for r in dropped})
            f=pd.concat(rows,ignore_index=True)
            if stage in ['validation','test'] and len(f):
                parent='selection' if stage=='validation' else 'refit';upper=(year-1 if stage=='validation' else year)*100+39
                good=np.isfinite(seasonal(f,panels[parent],upper));reasons.update({sid:'missing_seasonal_reference' for sid in f.loc[~good,'sample_id']});f=f[good].copy()
            f=complete(f,cfg['regions'])
            if f.empty:raise ValueError(f'No eligible {stage} windows for {fold}')
            ids=set(f.sample_id);frames[stage]=f
            for r in t[stage][META].itertuples(index=False):eligibility.append(dict(fold=fold,stage=stage,**r._asdict(),included=r.sample_id in ids,reason='included' if r.sample_id in ids else reasons.get(r.sample_id,'incomplete_region_group')))
        for regime in cfg['regimes']:
            folder=out/'folds'/fold/regime;folder.mkdir(parents=True,exist_ok=True);made={}
            for stage,f in frames.items():
                if regime=='AsOf':g=f.copy()
                else:g,_=windows(f,final,'hindsight',None if stage=='test' else final,None if stage=='test' else 'hindsight')
                assert g.sample_id.tolist()==f.sample_id.tolist();g.to_csv(folder/f'{stage}_windows.csv.gz',index=False,compression={'method':'gzip','mtime':0});made[stage]=g
            histories={}
            transforms={}
            for part,upper in [('selection',(year-1)*100+39),('refit',year*100+39)]:
                p=panels[part];p=p[p.epiweek.between(201040,upper)].copy()
                if regime!='AsOf':p=replace_values(p,final)
                p.to_csv(folder/f'{part}_history.csv.gz',index=False,compression={'method':'gzip','mtime':0});histories[part]=p;transforms[part]=transform(p,upper)
            plans[(fold,regime)]={'frames':made,'histories':histories,'transforms':transforms,'cutoffs':cuts}
        print('ELIGIBLE',fold,{k:len(v)//10 for k,v in frames.items()},flush=True)
    pd.DataFrame(eligibility).to_csv(out/'eligibility.csv',index=False)
    return plans
