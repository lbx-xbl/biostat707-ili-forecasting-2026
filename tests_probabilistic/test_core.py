import numpy as np
import pandas as pd
import pytest
from probabilistic.core import score,validate_levels,residual_quantiles,forecast

LEVELS=[.1,.5,.9]
def cfg(): return {'models':['Persistence','Ridge'],'regions':['hhs1','hhs2'],'quantiles':LEVELS,'bounds':[0.,100.]}
def panels():
    old=[];new=[]
    for model in cfg()['models']:
        for region in cfg()['regions']:
            for i,(week,date,target,tdate) in enumerate([(201337,'2013-09-14',201339,'2013-09-28'),(201336,'2013-09-07',201338,'2013-09-21')]):
                old.append(dict(sample_id=f'{region}_{week}',region=region,origin_week=week,origin_date=date,target_week=target,target_date=tdate,target_season='2012-2013',model=model,y_pred=1.,y_true=2.+i,fold='2013-2014'))
            for week,date,target,tdate in [(201340,'2013-10-05',201342,'2013-10-19'),(201341,'2013-10-12',201343,'2013-10-26')]:
                new.append(dict(sample_id=f'{region}_{week}',region=region,origin_week=week,origin_date=date,target_week=target,target_date=tdate,target_season='2013-2014',model=model,y_pred=1.,fold='2013-2014'))
    return pd.DataFrame(old),pd.DataFrame(new)

def test_wis_hand_example_and_components():
    r=score([6.],[[0.,2.,4.]],LEVELS)
    np.testing.assert_allclose(r['wis'],[4.4/1.5])
    np.testing.assert_allclose(r['dispersion'],[.4/1.5])
    np.testing.assert_allclose(r['underprediction'],[2./1.5])
    np.testing.assert_allclose(r['overprediction'],[0.])
    np.testing.assert_allclose(r['median_component'],[2./1.5])

def test_lower_tail_and_inclusive_endpoints():
    r=score([-2.,0.,4.],np.tile([0.,2.,4.],(3,1)),LEVELS)
    np.testing.assert_allclose(r['overprediction'],[2/1.5,0,0])
    np.testing.assert_allclose(r['underprediction'],0)

def test_degenerate_wis_is_absolute_error():
    levels=[.01,.025]+[i/20 for i in range(1,20)]+[.975,.99]
    r=score([0.,3.,10.],np.repeat([[2.],[3.],[5.]],len(levels),axis=1),levels)
    np.testing.assert_allclose(r['wis'],[2,0,5])

def test_wis_matches_independent_pinball():
    rng=np.random.default_rng(10);q=np.sort(rng.normal(size=(37,3)),axis=1);y=rng.normal(size=37)
    e=y[:,None]-q;check=2*np.maximum(np.array(LEVELS)*e,(np.array(LEVELS)-1)*e).mean(axis=1)
    np.testing.assert_allclose(score(y,q,LEVELS)['wis'],check)

@pytest.mark.parametrize('levels',[[.1,.9],[.1,.5,.8],[0,.5,1],[.1,.5,.5,.9],[.9,.5,.1],[.1,np.nan,.9]])
def test_invalid_quantile_levels(levels):
    with pytest.raises(ValueError): validate_levels(levels)

@pytest.mark.parametrize('q',[[[2,1,3]],[[1,np.nan,3]],[[1,2]]])
def test_invalid_forecast_arrays(q):
    with pytest.raises(ValueError):score([2],q,LEVELS)

def test_previous_residuals_and_test_label_invariance():
    old,new=panels();c=cfg();t=residual_quantiles(old,new,c,'2013-2014')
    np.testing.assert_allclose(t[t.model=='Ridge'].residual_quantile,[1.,1.5,2.])
    first=forecast(new,t,c)
    poisoned=new.assign(y_true=np.nan);again=forecast(poisoned,t,c)
    pd.testing.assert_frame_equal(first,again)
    np.testing.assert_allclose(first['q_0.500'],2.5)

def test_future_residuals_rejected():
    old,new=panels();old.loc[0,'target_date']='2013-10-19'
    with pytest.raises(ValueError):residual_quantiles(old,new,cfg(),'2013-2014')

def test_wrong_previous_season_rejected():
    old,new=panels();old['target_season']='2011-2012'
    with pytest.raises(ValueError):residual_quantiles(old,new,cfg(),'2013-2014')

@pytest.mark.parametrize('damage',['missing','duplicate','truth','gap'])
def test_bad_panels_rejected(damage):
    old,new=panels()
    if damage=='missing':old=old.iloc[1:]
    if damage=='duplicate':old=pd.concat([old,old.iloc[:1]])
    if damage=='truth':old.loc[0,'y_true']=9.
    if damage=='gap':new.loc[0,'target_date']='2013-11-02'
    with pytest.raises(ValueError):residual_quantiles(old,new,cfg(),'2013-2014')

def test_support_projection_and_no_crossing():
    old,new=panels();t=residual_quantiles(old,new,cfg(),'2013-2014')
    new['y_pred']=-2.;r=forecast(new,t,cfg());np.testing.assert_allclose(r[['q_0.100','q_0.500','q_0.900']],0)
    new['y_pred']=99.;r=forecast(new,t,cfg());np.testing.assert_allclose(r[['q_0.100','q_0.500','q_0.900']],100)
    assert r['raw_q_0.100'].eq(100).all()
