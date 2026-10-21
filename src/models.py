"""Train-only model fitting; no test target argument exists in this module."""
import copy
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class SmallLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=torch.nn.LSTM(input_size=1,hidden_size=16,num_layers=1,batch_first=True,dropout=0,bidirectional=False)
        self.head=torch.nn.Linear(28,1)
    def forward(self,x):
        seq,_=self.lstm(x[:,:12].unsqueeze(-1))
        return self.head(torch.cat([seq[:,-1,:],x[:,12:]],dim=1)).squeeze(-1)

def macro(y,p,regions):
    return float(pd.DataFrame({'region':regions,'e':np.abs(y-p)}).groupby('region').e.mean().mean())

def train_models(train_x,train_y,val_x,val_y,val_regions,transform,cfg,model_dir):
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    path=Path(model_dir); path.mkdir(parents=True,exist_ok=True)
    mu=transform['mu']; sigma=transform['sigma']; z=(train_y-mu)/sigma
    candidates=[]; ridge_models={}
    for alpha in cfg['ridge_alphas']:
        model=Ridge(alpha=alpha,fit_intercept=True).fit(train_x,z)
        score=macro(val_y,model.predict(val_x)*sigma+mu,val_regions)
        candidates.append({'alpha':alpha,'validation_macro_mae':score}); ridge_models[alpha]=model
    best=min(candidates,key=lambda c:(c['validation_macro_mae'],-c['alpha']))
    ridge=ridge_models[best['alpha']]
    tree=HistGradientBoostingRegressor(**cfg['hgb']).fit(train_x,z)
    net=SmallLSTM(); conf=cfg['lstm']
    optimizer=torch.optim.Adam(net.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'])
    tx=torch.tensor(train_x,dtype=torch.float32); ty=torch.tensor(z,dtype=torch.float32)
    vx=torch.tensor(val_x,dtype=torch.float32)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=conf['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        probe=net(tx[:32]); probe_loss=((probe-ty[:32])**2).mean()
        assert probe.shape==(min(32,len(tx)),) and torch.isfinite(probe_loss)
    curve=[]; best_mae=float('inf'); stale=0; best_epoch=0; best_state=None
    for epoch in range(1,conf['max_epochs']+1):
        net.train(); total=0.
        for bx,by in loader:
            optimizer.zero_grad(); pred=net(bx); loss=((pred-by)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite LSTM loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),conf['clip_norm']); optimizer.step()
            total+=float(loss.detach())*len(bx)
        net.eval()
        with torch.no_grad(): vp=net(vx).numpy().astype(float)*sigma+mu
        mae=macro(val_y,vp,val_regions)
        curve.append(dict(epoch=epoch,train_mse_standardized=total/len(tx),validation_macro_mae=mae))
        print(f'Epoch {epoch:02d}: train MSE={total/len(tx):.6f}; validation Macro-MAE={mae:.6f}',flush=True)
        if mae<best_mae-conf['min_delta']:
            best_mae=mae; best_epoch=epoch; stale=0; best_state=copy.deepcopy(net.state_dict())
        else: stale+=1
        if stale>=conf['patience']: break
    net.load_state_dict(best_state); net.eval()
    bundle={'Ridge':ridge,'HGB':tree,'LSTM':net}
    joblib.dump(ridge,path/'ridge.joblib'); joblib.dump(tree,path/'hgb.joblib')
    torch.save(net.state_dict(),path/'lstm.pt')
    selection=dict(seed=seed,threads=cfg['threads'],device='cpu',transform=transform,
                   ridge_candidates=candidates,ridge_selected_alpha=best['alpha'],hgb=cfg['hgb'],
                   lstm={**conf,'input_size':1,'num_layers':1,'dropout':0,'bidirectional':False,
                         'best_epoch':best_epoch,'total_epochs':len(curve),'best_validation_macro_mae':best_mae,
                         'initial_batch_loss':float(probe_loss),'early_stop_triggered':stale>=conf['patience']},
                   feature_order=[f'lag_{i}' for i in range(12)]+cfg['regions']+['target_sin','target_cos'],
                   fit_split='Train only; no Train+Validation refit',selection_split='Validation only',
                   frozen_before_test=True)
    (path/'parameters.json').write_text(json.dumps(selection,indent=2))
    return bundle,selection,pd.DataFrame(curve)

def predict_models(bundle,x,transform):
    mu=transform['mu']; sigma=transform['sigma']
    result={'Persistence':x[:,11]*sigma+mu}
    for name in ['Ridge','HGB']: result[name]=bundle[name].predict(x)*sigma+mu
    bundle['LSTM'].eval()
    with torch.no_grad(): result['LSTM']=bundle['LSTM'](torch.tensor(x,dtype=torch.float32)).numpy().astype(float)*sigma+mu
    return result

def load_models(path):
    path=Path(path); net=SmallLSTM(); net.load_state_dict(torch.load(path/'lstm.pt',map_location='cpu',weights_only=True)); net.eval()
    return {'Ridge':joblib.load(path/'ridge.joblib'),'HGB':joblib.load(path/'hgb.joblib'),'LSTM':net}
