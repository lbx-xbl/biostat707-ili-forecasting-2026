#!/usr/bin/env python3
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='4'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('.cache/vintage-matplotlib').resolve()))
import argparse,hashlib,json,platform,sys,time,traceback,shutil,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.windows import features
from rolling.models import select_models,refit_models,predict,save_models,load_models
from vintage.data import seasonal,META
from vintage.prepare import prepare
from vintage.evaluate import evaluate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2))
def preserve():
    m=json.loads(Path('docs/vintage_preservation_baseline.json').read_text());assert all(sha(p)==h for p,h in m.items());return {'status':'PASS','files':len(m)}
class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
    def flush(self):
        for f in self.streams:f.flush()
def infer(frame,bundle,tr,hist,upper,cfg,fold,regime):
    assert 'y_true' not in frame
    x=features(frame,tr,cfg['regions']);values=predict(bundle,x,tr);values.update(Persistence=frame.lag_11.to_numpy(),Seasonal=seasonal(frame,hist,upper));parts=[]
    for model,p in values.items():
        g=frame[META].copy();g['model']=model;g['y_pred']=p;g['fold']=fold;g['regime']=regime;g['issuance_date']=(pd.to_datetime(g.origin_date)+pd.Timedelta(days=7)).dt.date.astype(str);parts.append(g)
    return pd.concat(parts,ignore_index=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config/vintage.json');ap.add_argument('--output-dir',default='outputs/vintage_v0_5');args=ap.parse_args();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose fresh output directory')
    out.mkdir(parents=True,exist_ok=True);log=(out/'execution_log.txt').open('w');sys.stdout=Tee(sys.__stdout__,log);sys.stderr=Tee(sys.__stderr__,log);start=time.monotonic();summary={'status':'PARTIAL','checks':{},'unfinished':[],'completed':[],'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        print('COMMAND',sys.executable,*sys.argv);cfg=json.loads(Path(args.config).read_text());dump(out/'config_frozen.json',cfg);shutil.copyfile('docs/vintage_protocol_v0.5.md',out/'protocol_frozen.md');summary['protocol_sha256']=sha(out/'protocol_frozen.md');summary['config_sha256']=sha(out/'config_frozen.json');summary['checks']['prior_before']=preserve()
        archive_files=[Path(cfg['vintage_records']),Path('data/vintage/download_manifest.json')]+sorted(Path('data/vintage/raw').glob('*'))
        dump(out/'archive_input_hashes.json',{str(p):sha(p) for p in archive_files})
        summary['environment']={'python':sys.version,'platform':platform.platform(),'threads':4,'device':'cpu','versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','scikit-learn','matplotlib']}}
        plans=prepare(cfg,out);torch.set_num_threads(4);torch.set_num_interop_threads(1);allpred=[];allval=[];alloffset=[];choices=[];hashes={}
        with threadpool_limits(limits=4):
            for (fold,regime),plan in plans.items():
                print('FIT',fold,regime,flush=True);year=int(fold[:4]);folder=out/'folds'/fold/regime;frames=plan['frames'];tr,va,re,te=[frames[s] for s in ['train','validation','refit','test']];st=plan['transforms']['selection'];rt=plan['transforms']['refit'];dump(folder/'transforms.json',plan['transforms']);dump(folder/'cutoffs.json',plan['cutoffs'])
                tx=features(tr.drop(columns='y_true'),st,cfg['regions']);vx=features(va.drop(columns='y_true'),st,cfg['regions'])
                selected,sel,curve=select_models(tx,tr.y_true.to_numpy(),vx,va.y_true.to_numpy(),va.region.to_numpy(),st,cfg);dump(folder/'model_selection.json',sel);curve.to_csv(folder/'selection_curve.csv',index=False);save_models(selected,folder/'selection_models')
                vp=infer(va.drop(columns='y_true'),selected,st,plan['histories']['selection'],(year-1)*100+39,cfg,fold,regime).merge(va[['sample_id','y_true']],on='sample_id',validate='many_to_one');vp.to_csv(folder/'validation_predictions.csv',index=False);allval.append(vp)
                rx=features(re.drop(columns='y_true'),rt,cfg['regions']);ry=(re.y_true.to_numpy()-rt['mu'])/rt['sigma'];final,rc=refit_models(rx,ry,cfg,sel);rc.to_csv(folder/'refit_curve.csv',index=False);save_models(final,folder/'models')
                pred=infer(te,final,rt,plan['histories']['refit'],year*100+39,cfg,fold,regime)
                loaded=load_models(folder/'models');again=infer(te,loaded,rt,plan['histories']['refit'],year*100+39,cfg,fold,regime);np.testing.assert_allclose(pred.y_pred,again.y_pred,atol=1e-6,rtol=0)
                offsets=[]
                for model,g in vp.groupby('model'):
                    rq=np.quantile((g.y_true-g.y_pred).to_numpy(),cfg['quantiles'],method='linear');mask=pred.model==model
                    for p,v in zip(cfg['quantiles'],rq):
                        pred.loc[mask,f'q_{p:.3f}']=np.clip(pred.loc[mask,'y_pred']+v,0,100);offsets.append(dict(fold=fold,regime=regime,model=model,quantile=p,residual_quantile=v,n=len(g),label_cutoff=plan['cutoffs']['refit']))
                off=pd.DataFrame(offsets);off.to_csv(folder/'residual_quantiles.csv',index=False);alloffset.append(off);pred.to_csv(folder/'forecasts_before_evaluation.csv',index=False);allpred.append(pred)
                hashes[str((folder/'forecasts_before_evaluation.csv').relative_to(out))]=sha(folder/'forecasts_before_evaluation.csv')
                choices.append(dict(fold=fold,regime=regime,ridge_alpha=sel['ridge_alpha'],lstm_best_epoch=sel['lstm_best_epoch'],lstm_selection_epochs=sel['lstm_selection_total_epochs'],train_n=len(tr),validation_n=len(va),refit_n=len(re),test_n=len(te)))
                summary['completed'].append(f'{fold}/{regime}');dump(out/'run_summary.json',summary);print('FROZEN',fold,regime,'alpha',sel['ridge_alpha'],'epoch',sel['lstm_best_epoch'],flush=True)
        pred=pd.concat(allpred,ignore_index=True);pred.to_csv(out/'forecasts_before_evaluation.csv',index=False);hashes['forecasts_before_evaluation.csv']=sha(out/'forecasts_before_evaluation.csv');dump(out/'forecast_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes});pd.concat(allval,ignore_index=True).to_csv(out/'validation_predictions.csv',index=False);pd.concat(alloffset,ignore_index=True).to_csv(out/'residual_quantiles.csv',index=False);pd.DataFrame(choices).to_csv(out/'model_selection_summary.csv',index=False)
        met=evaluate(out,cfg)
        from vintage.report import render
        render(out)
        summary['checks'].update(prior_after=preserve(),reload_predictions='PASS',frozen_forecasts='PASS');assert all(sha(out/p)==h for p,h in hashes.items())
        summary.update(status='COMPLETE',exit_code=0,actual_vintage_records=len(pd.read_csv(cfg['vintage_records'])),test_samples_per_model_per_regime=int(len(pred)/10),regimes=cfg['regimes']);print(met[met.scope=='overall'][['regime','model','point_macro_mae','point_rmse','macro_wis']].to_string(index=False))
    except Exception as e:
        traceback.print_exc();summary.update(exit_code=1,unfinished=[f'{type(e).__name__}: {e}'])
    finally:
        summary['elapsed_seconds']=time.monotonic()-start;summary['finished_utc']=datetime.now(timezone.utc).isoformat();dump(out/'run_summary.json',summary);print('STATUS',summary['status'],'EXIT',summary['exit_code'],'SECONDS',summary['elapsed_seconds'])
    return summary['exit_code']
if __name__=='__main__':raise SystemExit(main())
