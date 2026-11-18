"""Selection uses Train/Validation; final refit takes no validation/test data."""
import copy
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from src.models import SmallLSTM

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def choose_alpha(candidates):
    return min(candidates,key=lambda c:(c['mae'],-c['alpha']))['alpha']

def macro(y,p,regions):
    return float(pd.DataFrame({'r':regions,'e':np.abs(y-p)}).groupby('r').e.mean().mean())

def fit_lstm(x,y,cfg,epochs,validation=None):
    """x/y are standardized. validation=(x, raw_y, regions, scaler), selection only."""
    seed_all(cfg['seed']); net=SmallLSTM(); c=cfg['lstm']
    optimizer=torch.optim.Adam(net.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    tx=torch.tensor(x,dtype=torch.float32); ty=torch.tensor(y,dtype=torch.float32)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=c['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
    curve=[]; best=float('inf'); stale=0; best_epoch=0; state=None
    for epoch in range(1,epochs+1):
        net.train(); total=0.
        for bx,by in loader:
            optimizer.zero_grad(); loss=((net(bx)-by)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),c['clip_norm']); optimizer.step()
            total+=float(loss.detach())*len(bx)
        row={'epoch':epoch,'train_mse_standardized':total/len(tx)}
        if validation is not None:
            vx,vy,vr,t=validation
            net.eval()
            with torch.no_grad(): vp=net(torch.tensor(vx,dtype=torch.float32)).numpy().astype(float)*t['sigma']+t['mu']
            value=macro(vy,vp,vr); row['validation_macro_mae']=value
            if value<best-c['min_delta']:
                best=value; best_epoch=epoch; stale=0; state=copy.deepcopy(net.state_dict())
            else: stale+=1
        curve.append(row)
        if validation is not None and stale>=c['patience']: break
    if validation is not None: net.load_state_dict(state)
    else: best_epoch=epochs
    net.eval()
    return net,curve,best_epoch

def select_models(x,y,vx,vy,regions,transform,cfg):
    seed_all(cfg['seed']); z=(y-transform['mu'])/transform['sigma']
    candidates=[]; ridge_models={}
    for a in cfg['ridge_alphas']:
        m=Ridge(alpha=a,fit_intercept=True).fit(x,z); ridge_models[a]=m
        p=m.predict(vx)*transform['sigma']+transform['mu']
        candidates.append({'alpha':a,'mae':macro(vy,p,regions)})
    alpha=choose_alpha(candidates)
    hgb=HistGradientBoostingRegressor(**cfg['hgb']).fit(x,z)
    net,curve,epoch=fit_lstm(x,z,cfg,epochs=cfg['lstm']['max_epochs'],validation=(vx,vy,regions,transform))
    selection={'seed':cfg['seed'],'ridge_candidates':candidates,'ridge_alpha':alpha,
               'ridge_grid_boundary':alpha in (min(cfg['ridge_alphas']),max(cfg['ridge_alphas'])),
               'lstm_best_epoch':epoch,'lstm_selection_total_epochs':len(curve),
               'selection_transform':transform,'selection_uses':'Train weights; Validation MAE only'}
    return {'Ridge':ridge_models[alpha],'HGB':hgb,'LSTM':net},selection,pd.DataFrame(curve)

def refit_models(x,y,cfg,selection):
    """y is standardized using the expanded-history scaler. No target-season argument."""
    seed_all(cfg['seed'])
    ridge=Ridge(alpha=selection['ridge_alpha'],fit_intercept=True).fit(x,y)
    hgb=HistGradientBoostingRegressor(**cfg['hgb']).fit(x,y)
    net,curve,epoch=fit_lstm(x,y,cfg,epochs=selection['lstm_best_epoch'])
    if epoch!=selection['lstm_best_epoch']: raise ValueError('Refit epoch mismatch')
    return {'Ridge':ridge,'HGB':hgb,'LSTM':net},pd.DataFrame(curve)

def predict(bundle,x,transform):
    pred={}
    for name in ['Ridge','HGB']: pred[name]=bundle[name].predict(x)*transform['sigma']+transform['mu']
    bundle['LSTM'].eval()
    with torch.no_grad(): pred['LSTM']=bundle['LSTM'](torch.tensor(x,dtype=torch.float32)).numpy().astype(float)*transform['sigma']+transform['mu']
    return pred

def save_models(bundle,path):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle['Ridge'],path/'ridge.joblib'); joblib.dump(bundle['HGB'],path/'hgb.joblib')
    torch.save(bundle['LSTM'].state_dict(),path/'lstm.pt')

def load_models(path):
    path=Path(path); net=SmallLSTM(); net.load_state_dict(torch.load(path/'lstm.pt',map_location='cpu',weights_only=True)); net.eval()
    return {'Ridge':joblib.load(path/'ridge.joblib'),'HGB':joblib.load(path/'hgb.joblib'),'LSTM':net}
