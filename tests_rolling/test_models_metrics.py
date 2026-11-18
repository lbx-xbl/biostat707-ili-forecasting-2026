import inspect
import numpy as np
import pandas as pd
import pytest
from rolling.models import choose_alpha,fit_lstm,select_models,refit_models
from rolling.evaluate import score,validate,metrics

def test_alpha_tie_prefers_larger_and_grid_can_exceed_100():
    assert choose_alpha([{'alpha':100,'mae':1.},{'alpha':1000,'mae':0.8},{'alpha':10000,'mae':0.8}])==10000

def test_refit_fixed_epoch_without_labels():
    x=np.zeros((35,24)); y=np.linspace(-1,1,35)
    cfg={'seed':2026,'lstm':{'learning_rate':.001,'weight_decay':.0001,'batch_size':32,'clip_norm':1.,'patience':8,'min_delta':1e-5}}
    m,curve,best=fit_lstm(x,y,cfg,epochs=2)
    assert len(curve)==2 and best==2
    assert all('validation_macro_mae' not in row for row in curve)
    assert list(inspect.signature(refit_models).parameters)==['x','y','cfg','selection']
    assert 'test_y' not in inspect.signature(select_models).parameters

def example():
    rows=[]
    for model in ['Persistence','Seasonal','Ridge','HGB','LSTM']:
        for i,r in enumerate(['hhs1','hhs1','hhs2']):
            rows.append(dict(model=model,sample_id=str(i),region=r,origin_week=201340+i,target_week=201342+i,target_season='2013-2014',y_true=0.,y_pred=[1.,3.,-6.][i]))
    return pd.DataFrame(rows)

def test_signed_bias_macro_and_rmse():
    d=example(); s=score(d[d.model=='Ridge'])
    assert s['macro_mae']==4
    assert s['pooled_mae']==pytest.approx(10/3)
    assert s['pooled_rmse']==pytest.approx(np.sqrt(46/3))
    assert s['bias']==pytest.approx(-2/3)
    assert s['macro_bias']==-2
    m=metrics(d)
    assert (m.improvement_pct==0).all()

@pytest.mark.parametrize('kind',['missing','duplicate','metadata','truth','infinite'])
def test_prediction_integrity(kind):
    d=example()
    if kind=='missing': d=d.iloc[1:]
    if kind=='duplicate': d=pd.concat([d,d.iloc[[0]]])
    if kind=='metadata': d.loc[0,'region']='hhs9'
    if kind=='truth': d.loc[0,'y_true']=999
    if kind=='infinite': d.loc[0,'y_pred']=np.inf
    with pytest.raises(ValueError): validate(d,example()[example().model=='Persistence'])
