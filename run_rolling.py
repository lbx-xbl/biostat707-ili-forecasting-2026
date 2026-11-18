#!/usr/bin/env python3
"""Approved six-season rolling analysis; does not modify pilot artifacts."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='4'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('.cache/rolling-matplotlib').resolve()))
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from src.data import load_data
from src.windows import features,LAGS
from rolling.data import all_windows,fold_frames,scaler,seasonal_table,seasonal_predict
from rolling.models import select_models,refit_models,predict,save_models,load_models
from rolling.evaluate import validate,metrics,MODELS

def dump(p,obj): Path(p).write_text(json.dumps(obj,indent=2))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def preservation():
    baseline=json.loads(Path('docs/pilot_preservation_baseline.json').read_text())
    changed=[p for p,h in baseline.items() if not Path(p).exists() or sha(p)!=h]
    if changed: raise ValueError(f'Pilot files changed: {changed}')
    return {'files_checked':len(baseline),'changed':changed,'status':'PASS'}

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams: f.write(s); f.flush()
    def flush(self):
        for f in self.streams: f.flush()

def predictions(frame,bundle,mat,t,table,fold):
    # Model inference only sees features, not frame.y_true.
    values=predict(bundle,mat,t)
    values.update(Persistence=frame.lag_11.to_numpy().copy(),Seasonal=seasonal_predict(frame,table))
    parts=[]
    for model in MODELS:
        g=frame.drop(columns=LAGS+['y_true']).copy(); g['model']=model; g['y_pred']=values[model]; g['fold']=fold; parts.append(g)
    return pd.concat(parts,ignore_index=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/rolling.json'); ap.add_argument('--output-dir',default='outputs/rolling_v0_2'); args=ap.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing nonempty output directory; choose fresh --output-dir.')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); log=(out/'execution_log.txt').open('w')
    sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    summary={'status':'PARTIAL','stage':'setup','started_utc':datetime.now(timezone.utc).isoformat(),'checks':{},'folds_completed':[],'unfinished':[]}
    try:
        print('COMMAND:',sys.executable,*sys.argv,flush=True)
        cfg=json.loads(Path(args.config).read_text()); dump(out/'config_frozen.json',cfg)
        (out/'protocol_frozen.md').write_bytes(Path('docs/rolling_evaluation_design_v0.2.md').read_bytes())
        protocol_hash=sha(out/'protocol_frozen.md'); config_hash=sha(out/'config_frozen.json')
        summary['protocol_sha256']=protocol_hash; summary['config_sha256']=config_hash
        summary['environment']={'python':sys.version,'platform':platform.platform(),'device':'cpu','threads':4,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','epiweeks','matplotlib','pytest']}}
        summary['checks']['pilot_before']=preservation()
        t=subprocess.run([sys.executable,'-m','pytest','-q','tests','tests_rolling'],capture_output=True,text=True)
        print(t.stdout,t.stderr); summary['checks']['pytest']={'exit_code':t.returncode,'output':t.stdout.strip()}
        if t.returncode: raise RuntimeError('Tests failed')
        # This phase is explicitly bound to the existing snapshot, not an implicit fresh download.
        if not Path('data/raw/fluview_201040_201939.json').exists(): raise FileNotFoundError('Required pilot snapshot absent')
        panel,source,audit=load_data(cfg); dump(out/'data_audit.json',audit)
        w=all_windows(panel,cfg); w.to_csv(out/'all_windows.csv',index=False)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        all_preds=[]; all_truth=[]; fold_summaries=[]; choice_rows=[]; reload_errors={}; selection_hashes={}
        with threadpool_limits(limits=4):
            for season in cfg['test_season_starts']:
                name=f'{season}-{season+1}'; path=out/'folds'/name; path.mkdir(parents=True)
                summary['stage']=f'fit {name}'; print('START FOLD',name,flush=True)
                frames=fold_frames(w,season); tr,va,re,te=[frames[k] for k in ['train','validation','refit','test']]
                for stage,g in frames.items():
                    for region,rg in g.groupby('region',sort=False):
                        fold_summaries.append(dict(fold=name,stage=stage,region=region,n=len(rg),origin_min=int(rg.origin_week.min()),origin_max=int(rg.origin_week.max()),target_min=int(rg.target_week.min()),target_max=int(rg.target_week.max())))
                st=scaler(panel,(season-1)*100+39); rt=scaler(panel,season*100+39)
                tx=features(tr.drop(columns='y_true'),st,cfg['regions']); vx=features(va.drop(columns='y_true'),st,cfg['regions'])
                selected,sel,curve=select_models(tx,tr.y_true.to_numpy(),vx,va.y_true.to_numpy(),va.region.to_numpy(),st,cfg)
                sel.update(fold=name,validation_first_week=(season-1)*100+40,validation_last_week=season*100+39,
                           refit_transform=rt,selected_before_outer_evaluation=True,configuration_sha256=config_hash)
                dump(path/'model_selection.json',sel); selection_hashes[name]=sha(path/'model_selection.json')
                curve.to_csv(path/'selection_curve.csv',index=False); save_models(selected,path/'selection_models')
                vtable=seasonal_table(panel,season-1,cfg['regions']); vtable.to_csv(path/'validation_seasonal_reference.csv',index=False)
                vp=predictions(va,selected,vx,st,vtable,name).merge(va[['sample_id','y_true']],on='sample_id',validate='many_to_one')
                validate(vp,va); vp.to_csv(path/'validation_predictions.csv',index=False)
                metrics(vp).to_csv(path/'validation_metrics.csv',index=False)
                rx=features(re.drop(columns='y_true'),rt,cfg['regions']); ry=(re.y_true.to_numpy()-rt['mu'])/rt['sigma']
                final,refit_curve=refit_models(rx,ry,cfg,sel)
                refit_curve.to_csv(path/'refit_curve.csv',index=False); save_models(final,path/'models')
                table=seasonal_table(panel,season,cfg['regions']); table.to_csv(path/'seasonal_reference.csv',index=False)
                frozen={'selection_sha256':selection_hashes[name],'final_refit_epochs':len(refit_curve),'refit_transform':rt,
                        'final_model_sha256':{p.name:sha(p) for p in sorted((path/'models').iterdir())},
                        'frozen_utc':datetime.now(timezone.utc).isoformat(),'test_season':name,'test_first_week':season*100+40,'test_last_week':(season+1)*100+39}
                dump(path/'refit_frozen.json',frozen)
                test_features=te.drop(columns='y_true'); xx=features(test_features,rt,cfg['regions'])
                # Predict without outer labels. Evaluation of ALL outer seasons occurs only after ALL fits.
                pred=predictions(te,final,xx,rt,table,name)
                pred.to_csv(path/'forecasts_before_evaluation.csv',index=False)
                loaded=load_models(path/'models'); new=predict(loaded,xx,rt)
                err={}
                for model in ['Ridge','HGB','LSTM']:
                    saved=pd.read_csv(path/'forecasts_before_evaluation.csv'); old=saved[saved.model==model].y_pred.to_numpy()
                    np.testing.assert_allclose(new[model],old,rtol=0,atol=1e-6); err[model]=float(np.max(np.abs(new[model]-old)))
                reload_errors[name]=err
                all_preds.append(pred); all_truth.append(te)
                choice_rows.append(dict(fold=name,ridge_alpha=sel['ridge_alpha'],lstm_best_epoch=sel['lstm_best_epoch'],lstm_selection_epochs=sel['lstm_selection_total_epochs'],lstm_refit_epochs=len(refit_curve),train_n=len(tr),validation_n=len(va),refit_n=len(re),test_n=len(te),selection_mu=st['mu'],selection_sigma=st['sigma'],refit_mu=rt['mu'],refit_sigma=rt['sigma']))
                summary['folds_completed'].append(name)
                dump(out/'run_summary.json',summary)
                print('FROZEN FOLD',name,'alpha',sel['ridge_alpha'],'LSTM epoch',sel['lstm_best_epoch'],'selection epochs',sel['lstm_selection_total_epochs'],'refit n',len(re),'test n',len(te),flush=True)
            summary['threadpools']=threadpool_info()
        summary['stage']='outer evaluation after all folds frozen'
        truth=pd.concat(all_truth,ignore_index=True); raw=pd.concat(all_preds,ignore_index=True)
        pred=raw.merge(truth[['sample_id','y_true']],on='sample_id',validate='many_to_one')
        validate(pred,truth); pred.to_csv(out/'predictions.csv',index=False)
        met=metrics(pred); met.to_csv(out/'metrics.csv',index=False)
        for name,g in pred.groupby('fold'):
            g.to_csv(out/'folds'/name/'predictions.csv',index=False); metrics(g).to_csv(out/'folds'/name/'test_metrics.csv',index=False)
        pd.DataFrame(fold_summaries).to_csv(out/'split_summary.csv',index=False)
        pd.DataFrame(choice_rows).to_csv(out/'model_selection_summary.csv',index=False)
        # Read CSV again: serialization is part of the verification boundary.
        recomputed=metrics(pd.read_csv(out/'predictions.csv'))
        pd.testing.assert_frame_equal(recomputed,pd.read_csv(out/'metrics.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
        summary['checks']['metrics_recomputed']='PASS'; summary['checks']['identical_keys_metadata']='PASS'
        summary['checks']['reload_max_abs_errors']=reload_errors
        assert sha(out/'config_frozen.json')==config_hash and sha(out/'protocol_frozen.md')==protocol_hash
        for name,h in selection_hashes.items(): assert sha(out/'folds'/name/'model_selection.json')==h
        summary['checks']['frozen_protocol_and_selection']='PASS'; summary['checks']['pilot_after']=preservation()
        summary['test_n_per_model']=len(truth); summary['prediction_rows']=len(pred)
        from rolling.report import render
        render(out)
        summary['status']='COMPLETE'; summary['stage']='phase_one_complete'; summary['exit_code']=0
        summary['later_stages_not_run']=['paired block bootstrap','continuous calibration/decomposition','quantile forecasts/WIS','vintage reconstruction']
        print(met[met.scope=='overall'].to_string(index=False))
    except Exception as exc:
        traceback.print_exc(); summary['unfinished']=[f'{type(exc).__name__}: {exc}']; summary['exit_code']=1
    finally:
        summary['elapsed_seconds']=time.monotonic()-started; summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        dump(out/'run_summary.json',summary)
        print('STATUS',summary['status'],'EXIT',summary['exit_code'],'SECONDS',summary['elapsed_seconds'])
    return summary['exit_code']
if __name__=='__main__': raise SystemExit(main())
