import numpy as np
import pandas as pd
import pytest
from diagnostics.bootstrap import block_indices,loss_sums,metric_differences,make_cubes,interval_summary

def test_blocks_contiguous_nonwrapping_and_deterministic():
    a=block_indices(11,4,25,123)
    assert a.shape==(25,11)
    assert a.min()>=0 and a.max()<11
    for start,end in [(0,4),(4,8),(8,11)]: assert (np.diff(a[:,start:end],axis=1)==1).all()
    np.testing.assert_array_equal(a,block_indices(11,4,25,123))
    with pytest.raises(ValueError): block_indices(3,4,2,123)

def test_joint_week_draws_and_rmse_not_average():
    # time x region x model. Perfect pairing with scaled errors.
    e=np.array([[[1.,0.],[3.,1.]],[[5.,2.],[7.,3.]],[[9.,4.],[11.,5.]]])
    indices=np.array([[2,0,2],[1,1,0]])
    a,s=loss_sums(e,indices)
    np.testing.assert_allclose(a[0],[[19,8],[25,11]])
    np.testing.assert_allclose(s[0],[[163,32],[251,51]])
    ma,rm=metric_differences(a,s,3)
    assert ma[0,0]==pytest.approx(19/6-44/6)
    assert rm[0,0]==pytest.approx(np.sqrt(83/6)-np.sqrt(414/6))
    identical=np.repeat(e[:,:,:1],2,axis=2)
    aa,ss=loss_sums(identical,indices)
    d1,d2=metric_differences(aa,ss,3)
    np.testing.assert_array_equal(d1,0); np.testing.assert_array_equal(d2,0)

def panel():
    return pd.DataFrame([dict(sample_id=f'{r}_{t}',region=r,origin_week=201340+t,target_week=201342+t,
             origin_date=str(pd.Timestamp('2013-10-05')+pd.Timedelta(weeks=t))[:10],
             target_date=str(pd.Timestamp('2013-10-19')+pd.Timedelta(weeks=t))[:10],
             target_season='2013-2014',model=m,y_true=float(t),y_pred=float(t+1))
             for m in ['Persistence','Ridge'] for t in range(3) for r in ['hhs1','hhs2']])

@pytest.mark.parametrize('mutation',['missing','duplicate','metadata','truth','gap','nonfinite'])
def test_cube_rejects_unpaired_or_nonconsecutive_data(mutation):
    d=panel()
    if mutation=='missing': d=d.iloc[1:]
    if mutation=='duplicate': d=pd.concat([d,d.iloc[[0]]])
    if mutation=='metadata': d.loc[0,'origin_week']=201339
    if mutation=='truth': d.loc[0,'y_true']=10
    if mutation=='gap': d.loc[d.target_week==201344,'target_date']='2013-11-09'
    if mutation=='nonfinite': d.loc[0,'y_pred']=np.nan
    with pytest.raises(ValueError): make_cubes(d,['Persistence','Ridge'],['hhs1','hhs2'])

def test_summary_uses_original_point_and_linear_quantiles():
    s=interval_summary(np.array([-3.,-2.,-1.,0.,1.]),point=0.5)
    assert s['estimate']==0.5 and s['ci_lower']==pytest.approx(-2.9) and s['ci_upper']==pytest.approx(.9)
    assert s['bootstrap_center_shift']==-1.5
