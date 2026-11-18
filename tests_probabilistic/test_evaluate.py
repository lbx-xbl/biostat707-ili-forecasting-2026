import numpy as np
import pandas as pd
from probabilistic.evaluate import summaries,paired_means

def test_equal_region_wis_and_pooled_median_rmse():
    d=pd.DataFrame({'sample_id':['a','b','c'],'model':['Persistence']*3,'region':['hhs1','hhs1','hhs2'],'target_season':['2013-2014']*3,'wis':[1.,3.,9.],'dispersion':[1.,3.,9.],'overprediction':[0.]*3,'underprediction':[0.]*3,'median_component':[0.]*3,'median_error':[1.,3.,9.],'median':[2.,4.,10.],'y_true':[1.]*3})
    w=d[['sample_id','model','region','target_season','y_true']].copy();w['q_0.100']=0.;w['q_0.500']=1.;w['q_0.900']=1.
    met,cov,rel,ints=summaries(d,w,[.1,.5,.9]);r=met[met.scope=='overall'].iloc[0]
    assert r.macro_wis==5.5
    np.testing.assert_allclose(r.median_rmse,np.sqrt(91/3))
    assert cov[cov.scope=='overall'].coverage.iloc[0]==1.
    assert rel[(rel.scope=='overall')&(rel['quantile']==.9)].observed_cdf.iloc[0]==1.

def test_paired_resampling_manual():
    cube=np.array([[[1.,2.],[3.,4.]],[[5.,7.],[9.,11.]],[[0.,1.],[2.,4.]]])
    idx=np.array([[0,0,2],[2,1,0]])
    actual=paired_means(cube,idx)
    np.testing.assert_allclose(actual,np.stack([cube[i].mean(axis=0) for i in idx]))
    same=np.repeat(cube[:,:,:1],2,axis=2);p=paired_means(same,idx)
    np.testing.assert_allclose(p[:,:,1]-p[:,:,0],0)
