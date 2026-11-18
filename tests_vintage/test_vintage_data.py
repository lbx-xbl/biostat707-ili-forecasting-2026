import numpy as np
import pandas as pd
import pytest
from epiweeks import Week
from vintage.data import asof,windows,complete,seasonal

def test_release_before_selection_and_issue_tie():
    d=pd.DataFrame([dict(region='hhs1',epiweek=201440,issue=201440,release_date='2014-10-10',wili=1),dict(region='hhs1',epiweek=201440,issue=201441,release_date='2014-10-20',wili=9),dict(region='hhs1',epiweek=201440,issue=201442,release_date='2014-10-10',wili=2)])
    assert asof(d,'2014-10-10').wili.iloc[0]==2
    assert asof(d,'2014-10-09').empty
    assert asof(d,'2014-10-20').wili.iloc[0]==9
    changed=d.copy();changed.loc[1,'wili']=999
    pd.testing.assert_frame_equal(asof(d,'2014-10-10'),asof(changed,'2014-10-10'))

def fixture():
    w=Week(2014,53);weeks=[w+i for i in range(-11,3)]
    p=pd.DataFrame([dict(region='hhs1',epiweek=int(str(x)),week_end=x.enddate().isoformat(),issue=int(str(x)),release_date='2015-01-10',wili=float(i)) for i,x in enumerate(weeks)])
    t=pd.DataFrame([dict(sample_id='x',region='hhs1',origin_week=201453,origin_date=w.enddate().isoformat(),target_week=int(str(w+2)),target_date=(w+2).enddate().isoformat(),target_season='2014-2015',y_true=999,lag_0=999)])
    return t,p

def test_week53_window_and_no_final_values():
    t,p=fixture();f,drop=windows(t,p,'2015-01-10');assert not drop and len(f)==1
    np.testing.assert_array_equal(f[[f'lag_{i}' for i in range(12)]].iloc[0],np.arange(12))
    assert 'y_true' not in f
    changed=t.assign(y_true=-100,lag_0=-100);g,_=windows(changed,p,'2015-01-10');pd.testing.assert_frame_equal(f,g)
    f,_=windows(t,p,'2015-01-10',p,'2015-01-10');assert f.y_true.iloc[0]==13

def test_missing_lag_and_unavailable_label():
    t,p=fixture();f,drop=windows(t,p.iloc[1:],'2015-01-10');assert f.empty and drop[0]['reason']=='missing_history'
    f,drop=windows(t,p,'2015-01-10',p.iloc[:-1],'2015-01-10');assert f.empty and drop[0]['reason']=='missing_label'

def test_complete_ten_region_group_only():
    d=pd.DataFrame({'sample_id':['a','b','c'],'origin_date':['a','a','b'],'region':['hhs1','hhs2','hhs1']})
    keep=complete(d,['hhs1','hhs2']);assert keep.sample_id.tolist()==['a','b']

def test_seasonal_missing_and_53_fallback():
    p=pd.DataFrame({'region':['hhs1','hhs1'],'epiweek':[201352,201401],'wili':[2.,4.]})
    f=pd.DataFrame({'region':['hhs1','hhs1'],'target_week':[201453,201450]})
    x=seasonal(f,p,201439);assert x[0]==3 and np.isnan(x[1])

def test_null_release_not_assumed_known():
    d=pd.DataFrame([dict(region='hhs1',epiweek=201440,issue=201440,release_date=None,wili=1)])
    assert asof(d,'2015-01-01').empty
