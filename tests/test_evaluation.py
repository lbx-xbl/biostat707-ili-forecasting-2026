import numpy as np
import pandas as pd
import pytest
from src.evaluate import scores, validate_predictions, improvement

def test_hand_metrics():
    d=pd.DataFrame({'region':['hhs1','hhs1','hhs2'],'y_true':[0.,0.,0.],'y_pred':[1.,3.,6.]})
    m=scores(d)
    assert m['macro_mae']==4
    assert m['pooled_mae']==pytest.approx(10/3)
    assert m['pooled_rmse']==pytest.approx(np.sqrt(46/3))
    assert improvement(4,3)==25
    assert np.isnan(improvement(0,1))

@pytest.mark.parametrize('kind',['duplicate','missing','nonfinite'])
def test_invalid_predictions(kind):
    d=pd.DataFrame([dict(sample_id=s,model=m,y_pred=1.) for s in ['a','b'] for m in ['Persistence','Ridge','HGB','LSTM']])
    if kind=='duplicate': d=pd.concat([d,d.iloc[[0]]])
    if kind=='missing': d=d.iloc[1:]
    if kind=='nonfinite': d.loc[0,'y_pred']=np.inf
    with pytest.raises(ValueError): validate_predictions(d,['a','b'])
