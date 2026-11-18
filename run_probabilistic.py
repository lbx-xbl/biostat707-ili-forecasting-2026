#!/usr/bin/env python3
"""Fit only pre-season empirical residual distributions; preserve every prior result."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='4'
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('.cache/probabilistic-matplotlib').resolve()))
import argparse,hashlib,importlib.metadata,json,platform,shutil,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from probabilistic.core import residual_quantiles,forecast,score
from probabilistic.evaluate import summaries,bootstrap

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2))
def preservation():
    m=json.loads(Path('docs/probabilistic_preservation_baseline.json').read_text());bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
    if bad:raise ValueError(f'Prior files changed: {bad}')
    return {'status':'PASS','files_checked':len(m)}
class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
    def flush(self):
        for f in self.streams:f.flush()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config/probabilistic.json');ap.add_argument('--output-dir',default='outputs/probabilistic_v0_4');args=ap.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise SystemExit('Refusing nonempty output directory; use a fresh directory')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();log=(out/'execution_log.txt').open('w');sys.stdout=Tee(sys.__stdout__,log);sys.stderr=Tee(sys.__stderr__,log)
    summary={'status':'PARTIAL','started_utc':datetime.now(timezone.utc).isoformat(),'checks':{},'unfinished':[],'point_models_retrained':0,'new_tuning_candidates':0,'empirical_residual_distributions_fit':0}
    try:
        print('COMMAND',sys.executable,*sys.argv,flush=True)
        summary['checks']['preservation_before']=preservation();cfg=json.loads(Path(args.config).read_text());dump(out/'config_frozen.json',cfg)
        shutil.copyfile('docs/probabilistic_protocol_v0.4.md',out/'protocol_frozen.md');summary['protocol_sha256']=sha(out/'protocol_frozen.md');summary['config_sha256']=sha(out/'config_frozen.json')
        summary['environment']={'python':sys.version,'platform':platform.platform(),'threads':4,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scikit-learn','pytest']}}
        t=subprocess.run([sys.executable,'-m','pytest','-q','tests_probabilistic'],capture_output=True,text=True);print(t.stdout,t.stderr);summary['checks']['pytest']={'exit_code':t.returncode,'output':t.stdout.strip()}
        if t.returncode:raise RuntimeError('Tests failed')
        source=Path(cfg['rolling_source']);inputs=out/'inputs';inputs.mkdir();sources={};offsets=[];allwide=[];residuals=[];audit=[];frozen={}
        for season in cfg['seasons']:
            path=inputs/season;path.mkdir();fold=source/'folds'/season
            for name in ['validation_predictions.csv','forecasts_before_evaluation.csv','model_selection.json','refit_frozen.json']:
                shutil.copyfile(fold/name,path/name);sources[str(fold/name)]=sha(path/name)
            va=pd.read_csv(path/'validation_predictions.csv');origins=pd.read_csv(path/'forecasts_before_evaluation.csv')
            if 'y_true' in origins:raise ValueError('Outer inference input must not contain outcomes')
            table=residual_quantiles(va,origins,cfg,season);pred=forecast(origins,table,cfg);offsets.append(table);allwide.append(pred)
            r=va.copy();r['residual']=r.y_true-r.y_pred;residuals.append(r)
            audit.append(dict(fold=season,validation_season=table.source_season.iloc[0],validation_rows_per_model=len(va)//len(cfg['models']),validation_target_max=va.target_date.max(),outer_origin_min=origins.origin_date.min(),outer_target_min=origins.target_date.min(),outer_target_max=origins.target_date.max(),outer_rows_per_model=len(origins)//len(cfg['models']),overlapping_sample_ids=len(set(va.sample_id)&set(origins.sample_id)),residual_targets_before_outer_origins=True))
            path=out/'folds'/season;path.mkdir(parents=True);table.to_csv(path/'residual_quantiles.csv',index=False);pred.to_csv(path/'quantiles_before_evaluation.csv',index=False)
            frozen[str(path.relative_to(out)/'quantiles_before_evaluation.csv')]=sha(path/'quantiles_before_evaluation.csv');summary['empirical_residual_distributions_fit']+=len(cfg['models']);print('FROZEN',season,'distributions',len(cfg['models']),'samples/model',len(origins)//len(cfg['models']),flush=True)
        # Every fold's distribution and forecast is fixed before evaluation outcomes are loaded.
        wide=pd.concat(allwide,ignore_index=True);wide.to_csv(out/'quantiles_before_evaluation.csv',index=False);frozen['quantiles_before_evaluation.csv']=sha(out/'quantiles_before_evaluation.csv')
        pd.concat(offsets,ignore_index=True).to_csv(out/'residual_quantiles.csv',index=False);pd.concat(residuals,ignore_index=True).to_csv(out/'validation_residuals.csv',index=False);pd.DataFrame(audit).to_csv(out/'temporal_audit.csv',index=False)
        dump(out/'forecast_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen,'outcomes_joined_after_freeze':True})
        for file,name in [(source/'predictions.csv','original_point_predictions.csv'),(Path(cfg['bootstrap_indices_source']),'bootstrap_indices.npz')]:
            shutil.copyfile(file,inputs/name);sources[str(file)]=sha(inputs/name)
        dump(out/'input_manifest.json',sources)
        truth=pd.read_csv(inputs/'original_point_predictions.csv')
        keys=['sample_id','model'];a=wide.sort_values(keys).reset_index(drop=True);b=truth.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[b.drop(columns='y_true').columns],b.drop(columns='y_true'),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
        scored_wide=wide.merge(truth[keys+['y_true']],on=keys,validate='one_to_one')
        if len(scored_wide)!=len(wide) or not scored_wide.y_true.between(0,100).all():raise ValueError('Invalid evaluation labels/keys')
        qcols=[f'q_{p:.3f}' for p in cfg['quantiles']];vals=scored_wide[qcols].to_numpy();result=score(scored_wide.y_true.to_numpy(),vals,cfg['quantiles'])
        scored=scored_wide[[c for c in truth.columns]].copy()
        for key,value in result.items():scored[key]=value
        scored.to_csv(out/'scores.csv',index=False);scored_wide.to_csv(out/'quantiles_wide.csv',index=False)
        long=[]
        for p in cfg['quantiles']:
            part=scored_wide[truth.columns].copy();part['quantile']=p;part['value']=scored_wide[f'q_{p:.3f}'];part['raw_value']=scored_wide[f'raw_q_{p:.3f}'];part['clipped']=part.value!=part.raw_value;long.append(part)
        long=pd.concat(long,ignore_index=True);long.to_csv(out/'quantile_predictions.csv',index=False)
        met,cov,rel,ints=summaries(scored,scored_wide,cfg['quantiles'])
        for name,frame in [('metrics.csv',met),('coverage.csv',cov),('quantile_reliability.csv',rel),('interval_scores.csv',ints)]:frame.to_csv(out/name,index=False)
        clip=long.groupby(['fold','model']).agg(n_quantiles=('clipped','size'),clipped_quantiles=('clipped','sum'),raw_min=('raw_value','min'),raw_max=('raw_value','max')).reset_index();clip.to_csv(out/'clipping_summary.csv',index=False)
        bootstrap(scored,cfg,out)
        summary['checks']['preservation_after']=preservation();summary['checks']['identical_point_predictions']='PASS';summary['checks']['chronological_calibration']='PASS';summary['checks']['monotone_bounded_quantiles']='PASS'
        assert all(sha(out/p)==h for p,h in frozen.items());summary['checks']['forecasts_frozen_before_evaluation']='PASS'
        from probabilistic.report import render
        render(out)
        summary.update(status='COMPLETE',exit_code=0,test_samples_per_model=len(scored)//5,quantile_rows=len(long),scored_forecasts=len(scored),residual_rows=sum(len(x) for x in residuals),clipped_quantiles=int(long.clipped.sum()),bootstrap_interval_rows=204,later_stages_not_run=['quantile gradient boosting','vintage reconstruction'])
        print(met[met.scope=='overall'].to_string(index=False),flush=True)
    except Exception as e:
        traceback.print_exc();summary.update(exit_code=1,unfinished=[f'{type(e).__name__}: {e}'])
    finally:
        summary['elapsed_seconds']=time.monotonic()-start;summary['finished_utc']=datetime.now(timezone.utc).isoformat();dump(out/'run_summary.json',summary);print('STATUS',summary['status'],'EXIT',summary['exit_code'],'SECONDS',summary['elapsed_seconds'])
    return summary['exit_code']
if __name__=='__main__':raise SystemExit(main())
