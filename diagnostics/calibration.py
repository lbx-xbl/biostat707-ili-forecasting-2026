"""Empirical CORP decomposition for means under squared-error scoring only."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def decompose(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)==0 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Invalid calibration arrays')
    fitted=IsotonicRegression(increasing=True,out_of_bounds='nan').fit_transform(x,y)
    mse=float(np.mean((x-y)**2)); iso=float(np.mean((fitted-y)**2)); unc=float(np.mean((y-y.mean())**2))
    mcb=mse-iso; dsc=unc-iso; bias=float(np.mean(x-y)); uncond=bias**2; cond=mcb-uncond
    tol=1e-10*max(1,mse,unc)
    if min(mcb,dsc,cond)<-tol: raise ValueError('Invalid negative CORP component')
    score={'n':len(x),'mse':mse,'mse_isotonic_diagnostic':iso,'mcb':mcb,'dsc':dsc,'unc':unc,
           'bias':bias,'mcb_unconditional':uncond,'mcb_conditional':cond,'root_mcb':float(np.sqrt(max(mcb,0))),
           'skill_vs_evaluation_mean':1-mse/unc if unc>0 else float('nan'),'empirical_outcome_mean':float(y.mean()),
           'unique_forecasts':len(np.unique(x)),'isotonic_levels':len(np.unique(fitted)),
           'identity_error':mse-(mcb-dsc+unc),'functional':'mean','score':'squared error','component_units':'squared percentage points'}
    return score,fitted

def run_calibration(predictions,models,out):
    groups=[('overall','all',predictions)]
    groups += [('season',s,g) for s,g in predictions.groupby('target_season',sort=True)]
    groups += [('region',r,g) for r,g in predictions.groupby('region',sort=False)]
    components=[]; points=[]
    for scope,group,g in groups:
        for model in models:
            q=g[g.model==model].copy(); result,z=decompose(q.y_pred,q.y_true)
            components.append({'scope':scope,'group':group,'model':model,**result})
            q=q[['sample_id','region','target_date','target_season','y_pred','y_true']].copy()
            q['scope']=scope; q['group']=group; q['model']=model; q['isotonic_diagnostic_mean']=z
            points.append(q)
    table=pd.DataFrame(components); detail=pd.concat(points,ignore_index=True)
    table.to_csv(out/'calibration_components.csv',index=False)
    detail.to_csv(out/'calibration_diagnostic_points.csv',index=False)
    return table,detail
