import numpy as np
import pytest
from diagnostics.calibration import decompose

def test_empirical_score_difference_not_prediction_distance():
    s,z=decompose([0,1,2],[0,2,1])
    np.testing.assert_allclose(z,[0,1.5,1.5])
    assert s['mse']==pytest.approx(2/3)
    assert s['mse_isotonic_diagnostic']==pytest.approx(1/6)
    assert s['mcb']==pytest.approx(.5) and s['dsc']==pytest.approx(.5) and s['unc']==pytest.approx(2/3)
    assert s['mcb']!=pytest.approx(np.mean((np.array([0,1,2])-z)**2))
    assert s['mse']==pytest.approx(s['mcb']-s['dsc']+s['unc'])

def test_ties_constant_forecast_and_bias_split():
    s,z=decompose([0,0],[1,3]); np.testing.assert_allclose(z,[2,2])
    assert s['mse']==5 and s['mcb']==4 and s['dsc']==0 and s['unc']==1
    assert s['bias']==-2 and s['mcb_unconditional']==4 and s['mcb_conditional']==0
    assert s['skill_vs_evaluation_mean']==-4
    x=np.array([1,1,2,3]); y=np.array([0,2,3,2]); a,za=decompose(x,y)
    order=[3,1,0,2]; b,zb=decompose(x[order],y[order])
    np.testing.assert_allclose(za[order],zb)
    assert za[0]==za[1] and a['mcb']==pytest.approx(b['mcb'])

def test_perfect_forecast_and_constant_truth():
    s,z=decompose([1,2,3],[1,2,3])
    assert s['mcb']==0 and s['mse']==0 and s['skill_vs_evaluation_mean']==1
    s,z=decompose([0,1,2],[2,2,2])
    assert s['unc']==0 and np.isnan(s['skill_vs_evaluation_mean'])
    assert s['mcb_conditional']>=-1e-12

@pytest.mark.parametrize('x,y',[([],[]),([1,np.nan],[1,2]),([1],[1,2]),([np.inf],[1])])
def test_invalid_arrays(x,y):
    with pytest.raises(ValueError): decompose(x,y)
