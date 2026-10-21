import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from epiweeks import Week
from src.data import normalize
from src.windows import build_windows, fit_transform

CFG=json.loads(Path('config/demo.json').read_text())
def fixture():
    weeks=[]; w=Week(2010,40)
    while w<=Week(2019,39):
        weeks.append(w); w+=1
    return pd.DataFrame([dict(region=r,epiweek=int(str(w)),wili=float(i+j*1000),release_date=None,issue=202001) for j,r in enumerate(CFG['regions']) for i,w in enumerate(weeks)])

def test_calendar_sequence_and_splits():
    d=normalize(fixture(),CFG); x,s=build_windows(d,CFG)
    assert x.groupby('split').size().to_dict()=={'Train':3000,'Validation':500,'Test':1020}
    a=x[(x.region=='hhs1') & (x.split=='Train')].iloc[0]
    assert (a.origin_week,a.target_week)==(201051,201101)
    assert a[[f'lag_{i}' for i in range(12)]].tolist()==list(range(12))
    assert a.y_true==13
    assert ((pd.to_datetime(x.target_date)-pd.to_datetime(x.origin_date)).dt.days==14).all()
    for r,g in x.groupby('region'):
        offset=(int(r[3:])-1)*1000
        assert (g.lag_0>=offset).all() and (g.lag_11<offset+469).all()
    b=x[(x.region=='hhs1') & (x.origin_week==201452)].iloc[0]
    assert b.target_week==201501
    assert 201453 in d.epiweek.values
    bounds={'Train':(201051,201101,201637,201639),'Validation':(201640,201642,201737,201739),'Test':(201740,201742,201937,201939)}
    for split, expected in bounds.items():
        g=x[x.split==split]
        assert (g.origin_week.min(),g.target_week.min(),g.origin_week.max(),g.target_week.max())==expected
    b=x[(x.region=='hhs1') & (x.origin_week==201640)].iloc[0]
    assert b.lag_0 < d[(d.region=='hhs1') & (d.epiweek==201640)].wili.iloc[0]
    assert not x.origin_week.isin([201638,201639,201738,201739]).any()

@pytest.mark.parametrize('kind',['duplicate','gap','nan'])
def test_bad_panel_rejected(kind):
    d=fixture()
    if kind=='duplicate': d=pd.concat([d,d.iloc[[0]]])
    if kind=='gap': d=d.drop(20)
    if kind=='nan': d.loc[20,'wili']=np.nan
    with pytest.raises(ValueError): normalize(d,CFG)

def test_train_only_transform():
    d=normalize(fixture(),CFG); a=fit_transform(d,CFG)
    altered=d.copy(); altered.loc[altered.epiweek>=201640,'wili']=999999
    assert fit_transform(altered,CFG)==a
    train=d[d.epiweek<=201639].wili
    assert a['mu']==pytest.approx(train.mean())
    assert a['sigma']==pytest.approx(train.std(ddof=0))
