#!/usr/bin/env python3
"""Run the fixed SPEC.md experiment. Existing output directories are rejected."""
import os
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[var]='4'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('.cache/matplotlib').resolve()))
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
from threadpoolctl import threadpool_limits,threadpool_info
from src.data import load_data
from src.windows import build_windows,fit_transform,features,LAGS
from src.models import train_models,predict_models,load_models
from src.evaluate import metrics,validate_predictions,render_report

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams: f.write(s); f.flush()
    def flush(self):
        for f in self.streams: f.flush()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='config/demo.json'); parser.add_argument('--output-dir',default='outputs/demo_v0_1'); args=parser.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing nonempty output directory; choose a fresh --output-dir.')
    out.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    log=(out/'execution_log.txt').open('w'); sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    result={'status':'PARTIAL','started_utc':datetime.now(timezone.utc).isoformat(),'command':' '.join(sys.argv),'checks':{},'unfinished':[]}
    print('COMMAND:',sys.executable,*sys.argv,flush=True)
    try:
        cfg=json.loads(Path(args.config).read_text()); (out/'config_used.json').write_text(json.dumps(cfg,indent=2))
        result['environment']={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'device':'cpu','threads':4,
            'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','epiweeks','matplotlib','pytest']}}
        test=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True)
        print(test.stdout,test.stderr); result['checks']['pytest']={'exit_code':test.returncode,'output':test.stdout.strip()}
        if test.returncode: raise RuntimeError('Tests failed')
        d,source,audit=load_data(cfg); x,split_summary=build_windows(d,cfg)
        (out/'data_audit.json').write_text(json.dumps(audit,indent=2)); split_summary.to_csv(out/'split_summary.csv',index=False)
        d.to_csv('data/normalized.csv',index=False); x.to_csv('data/windows.csv',index=False)
        counts=x.groupby('split').size().to_dict()
        assert counts=={'Train':3000,'Validation':500,'Test':1020}
        expected={'Train':(201051,201101,201637,201639),'Validation':(201640,201642,201737,201739),'Test':(201740,201742,201937,201939)}
        for name,g in x.groupby('split'):
            assert (g.origin_week.min(),g.target_week.min(),g.origin_week.max(),g.target_week.max())==expected[name]
        result['data_rows']=len(d); result['window_counts']=counts; result['checks']['calendar_boundaries_and_counts']='PASS'
        transform=fit_transform(d,cfg); changed=d.copy(); changed.loc[changed.epiweek>=201640,'wili']=-12345
        assert fit_transform(changed,cfg)==transform
        result['checks']['train_only_preprocessing']='PASS'
        train=x[x.split=='Train']; val=x[x.split=='Validation']; test_x=x[x.split=='Test'].drop(columns='y_true')
        trf=features(train.drop(columns='y_true'),transform,cfg['regions']); vf=features(val.drop(columns='y_true'),transform,cfg['regions'])
        with threadpool_limits(limits=4):
            bundle,selection,curve=train_models(trf,train.y_true.to_numpy(),vf,val.y_true.to_numpy(),val.region.to_numpy(),transform,cfg,out/'models')
            (out/'model_selection.json').write_text(json.dumps(selection,indent=2)); curve.to_csv(out/'lstm_training_curve.csv',index=False)
            frozen_hash=hashlib.sha256((out/'model_selection.json').read_bytes()).hexdigest()
            print('Model selection frozen before Test prediction:',frozen_hash)
            tf=features(test_x,transform,cfg['regions'])
            # Test labels are first consumed below, by the evaluation table.
            parts=[]
            for name,frame,mat in [('Validation',val,vf),('Test',x[x.split=='Test'],tf)]:
                predictions=predict_models(bundle,mat,transform)
                predictions['Persistence']=frame.lag_11.to_numpy().copy()
                for model,pred in predictions.items():
                    part=frame.drop(columns=LAGS).copy(); part['model']=model; part['y_pred']=pred; parts.append(part)
            pred=pd.concat(parts,ignore_index=True)
            validate_predictions(pred,x[x.split!='Train'].sample_id.tolist())
            assert np.array_equal(pred[pred.model=='Persistence'].y_pred.to_numpy(),pd.concat([val,x[x.split=='Test']]).lag_11.to_numpy())
            result['checks']['same_evaluation_keys_and_persistence']='PASS'
            pred.to_csv(out/'predictions.csv',index=False)
            met=metrics(pred); met.to_csv(out/'metrics.csv',index=False)
            saved=pd.read_csv(out/'predictions.csv'); recomputed=metrics(saved); saved_metrics=pd.read_csv(out/'metrics.csv')
            pd.testing.assert_frame_equal(recomputed,saved_metrics,check_dtype=False,atol=1e-12,rtol=1e-12)
            result['checks']['metrics_recomputed_from_saved_csv']='PASS'
            reloaded=load_models(out/'models'); rp=predict_models(reloaded,tf,transform)
            errors={}
            for model,p in rp.items():
                previous=saved[(saved.split=='Test') & (saved.model==model)].y_pred.to_numpy()
                errors[model]=float(np.max(np.abs(p-previous)))
                np.testing.assert_allclose(p,previous,atol=1e-6,rtol=0)
            result['checks']['model_reload_max_abs_error']=errors
            altered=pred.copy(); altered.loc[altered.split=='Test','y_true']+=1000; metrics(altered)
            after=predict_models(bundle,tf,transform)
            for model in rp: np.testing.assert_allclose(after[model],rp[model],atol=1e-6,rtol=0)
            assert hashlib.sha256((out/'model_selection.json').read_bytes()).hexdigest()==frozen_hash
            result['checks']['test_labels_changed_at_fixed_features_no_effect']='PASS'
            result['threadpools']=threadpool_info()
        render_report(pred,met,curve,selection,split_summary,audit,out)
        result['negative_predictions']=met[met.scope=='overall'][['split','model','negative_predictions']].to_dict('records')
        result['status']='COMPLETE'; result['exit_code']=0
        print(met[met.scope=='overall'].to_string(index=False))
    except Exception as e:
        traceback.print_exc(); result['unfinished']=[f'{type(e).__name__}: {e}']; result['exit_code']=1
    finally:
        result['elapsed_seconds']=time.monotonic()-started; result['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'run_summary.json').write_text(json.dumps(result,indent=2))
        print('STATUS:',result['status'],'EXIT:',result['exit_code'],'SECONDS:',result['elapsed_seconds'])
    return result['exit_code']

if __name__=='__main__': raise SystemExit(main())
