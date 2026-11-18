import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import load_data
from rolling.data import all_windows,fold_frames,scaler,seasonal_table,seasonal_predict
CFG=json.loads(Path('config/rolling.json').read_text())
@pytest.fixture(scope='module')
def panel():
    return load_data(CFG)[0]

def test_first_last_fold_and_refit_bridge(panel):
    w=all_windows(panel,CFG)
    a=fold_frames(w,2013)
    assert len(a['train'])==910 and len(a['validation'])==500 and len(a['refit'])==1430 and len(a['test'])==500
    assert a['train'].target_week.max()==201239
    assert a['validation'].origin_week.min()==201240
    assert a['refit'].target_week.max()==201339
    assert a['test'].origin_week.min()==201340 and a['test'].target_week.min()==201342
    assert a['test'].origin_week.max()==201437 and a['test'].target_week.max()==201439
    ids=set(a['refit'].sample_id)-set(a['train'].sample_id)-set(a['validation'].sample_id)
    assert len(ids)==20
    b=fold_frames(w,2018)
    assert b['train'].target_week.max()==201739 and b['validation'].target_week.max()==201839
    assert b['refit'].target_week.max()==201839 and b['test'].target_week.max()==201939
    for s in range(2013,2019):
        test=fold_frames(w,s)['test']
        assert len(test)==(510 if s==2014 else 500)
        assert ((test.target_date-test.origin_date).dt.days==14).all()
        assert test.target_season.nunique()==1

def test_scaler_and_seasonal_no_future_leak(panel):
    old=scaler(panel,201239)
    changed=panel.copy(); changed.loc[changed.epiweek>201239,'wili']=1e6
    assert scaler(changed,201239)==old
    train=panel[panel.epiweek<=201239].wili
    assert old['mu']==pytest.approx(train.mean())
    assert old['sigma']==pytest.approx(train.std(ddof=0))
    a=seasonal_table(panel,2014,CFG['regions'])
    changed=panel.copy(); changed.loc[changed.epiweek>=201440,'wili']=1e6
    pd.testing.assert_frame_equal(a,seasonal_table(changed,2014,CFG['regions']))
    t=a[(a.region=='hhs1') & (a.week==53)].iloc[0]
    want=panel[(panel.region=='hhs1') & (panel.epiweek<201440) & (panel.epiweek.mod(100).isin([52,1]))].wili.mean()
    assert t.y_pred==pytest.approx(want) and t.method=='week52_week1_fallback'
    b=seasonal_table(panel,2016,CFG['regions']); t=b[(b.region=='hhs1') & (b.week==53)].iloc[0]
    assert t.y_pred==pytest.approx(panel[(panel.region=='hhs1') & (panel.epiweek==201453)].wili.iloc[0])
    assert t.method=='historical_same_week'

def test_seasonal_matches_region_week_not_row_order(panel):
    table=seasonal_table(panel,2014,CFG['regions'])
    f=pd.DataFrame({'region':['hhs10','hhs1'],'target_week':[201501,201453]})
    pred=seasonal_predict(f,table)
    assert pred[0]==table[(table.region=='hhs10') & (table.week==1)].y_pred.iloc[0]
    assert pred[1]==table[(table.region=='hhs1') & (table.week==53)].y_pred.iloc[0]
