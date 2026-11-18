#!/usr/bin/env python3
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='4'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('.cache/diagnostics-matplotlib').resolve()))
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from diagnostics.bootstrap import make_cubes,run_bootstrap
from diagnostics.calibration import run_calibration
from diagnostics.report import render

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def preservation():
    baseline=json.loads(Path('docs/diagnostics_preservation_baseline.json').read_text())
    changed=[p for p,h in baseline.items() if not Path(p).exists() or sha(p)!=h]
    if changed: raise ValueError(f'Prior artifact changed: {changed}')
    return {'status':'PASS','files_checked':len(baseline),'changed':changed}
class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams: f.write(s); f.flush()
    def flush(self):
        for f in self.streams: f.flush()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/diagnostics.json'); ap.add_argument('--output-dir',default='outputs/diagnostics_v0_3'); args=ap.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing nonempty output directory')
    out.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    log=(out/'execution_log.txt').open('w'); sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    summary={'status':'PARTIAL','started_utc':datetime.now(timezone.utc).isoformat(),'checks':{},'unfinished':[]}
    try:
        print('COMMAND:',sys.executable,*sys.argv,flush=True)
        cfg=json.loads(Path(args.config).read_text())
        (out/'config_frozen.json').write_text(json.dumps(cfg,indent=2))
        (out/'protocol_frozen.md').write_bytes(Path('docs/diagnostics_protocol_v0.3.md').read_bytes())
        summary['config_sha256']=sha(out/'config_frozen.json'); summary['protocol_sha256']=sha(out/'protocol_frozen.md')
        summary['preservation_before']=preservation()
        summary['environment']={'python':sys.version,'platform':platform.platform(),'threads':4,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','pytest']}}
        t=subprocess.run([sys.executable,'-m','pytest','-q','tests_diagnostics'],capture_output=True,text=True)
        print(t.stdout,t.stderr); summary['checks']['tests']={'exit_code':t.returncode,'output':t.stdout.strip()}
        if t.returncode: raise RuntimeError('Diagnostics tests failed')
        src=Path(cfg['input_predictions']); summary['input_predictions_sha256']=sha(src)
        (out/'input_predictions.csv').write_bytes(src.read_bytes())
        d=pd.read_csv(out/'input_predictions.csv'); cubes=make_cubes(d,cfg['models'],cfg['regions'])
        assert list(cubes)==cfg['seasons'] and len(d)==15050
        assert [len(c['dates']) for c in cubes.values()]==[50,51,50,50,50,50]
        summary['checks']['complete_matched_week_region_model_panel']='PASS'
        with threadpool_limits(limits=4):
            print('BOOTSTRAP: 5000 replicates for each fixed L=8,4,13; paired all regions/models',flush=True)
            intervals,files=run_bootstrap(cubes,cfg,out)
            print('CALIBRATION: grouped mean isotonic diagnostics; no forecast replacement',flush=True)
            components,points=run_calibration(d,cfg['models'],out)
        assert np.allclose(components.mse,components.mcb-components.dsc+components.unc,atol=1e-12,rtol=1e-12)
        assert np.allclose(components.mcb,components.mcb_unconditional+components.mcb_conditional,atol=1e-12,rtol=1e-12)
        summary['checks']['decomposition_identities']='PASS'
        assert sha(src)==summary['input_predictions_sha256'] and sha(out/'input_predictions.csv')==sha(src)
        assert sha(out/'config_frozen.json')==summary['config_sha256'] and sha(out/'protocol_frozen.md')==summary['protocol_sha256']
        render(out)
        summary['preservation_after']=preservation()
        summary['interval_rows']=len(intervals); summary['decomposition_rows']=len(components); summary['diagnostic_point_rows']=len(points)
        summary['forecast_models_trained']=0; summary['saved_forecasts_changed']=0
        summary['status']='COMPLETE'; summary['exit_code']=0
        print(intervals[(intervals.scope=='overall')&(intervals.block_length==8)][['model','metric','estimate','ci_lower','ci_upper','bootstrap_center_shift']].to_string(index=False))
        print(components[components.scope=='overall'][['model','mse','mcb','dsc','unc','root_mcb','bias']].to_string(index=False))
    except Exception as exc:
        traceback.print_exc(); summary['unfinished']=[f'{type(exc).__name__}: {exc}']; summary['exit_code']=1
    finally:
        summary['elapsed_seconds']=time.monotonic()-start; summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
        print('STATUS',summary['status'],'EXIT',summary['exit_code'],'SECONDS',summary['elapsed_seconds'])
    return summary['exit_code']
if __name__=='__main__': raise SystemExit(main())
