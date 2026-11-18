import pandas as pd
import pytest
from verify_vintage import align_scores

def test_verifier_aligns_by_regime_model_sample_not_csv_row_order():
    w=pd.DataFrame({'sample_id':['a','a','b'],'model':['Ridge']*3,'regime':['AsOf','MatchedRevised','AsOf']})
    s=w.assign(wis=[1.,2.,3.]).iloc[[2,0,1]].reset_index(drop=True)
    assert align_scores(w,s).wis.tolist()==[1.,2.,3.]

@pytest.mark.parametrize('damage',['missing','duplicate'])
def test_verifier_rejects_missing_or_duplicate_score_keys(damage):
    w=pd.DataFrame({'sample_id':['a','b'],'model':['Ridge']*2,'regime':['AsOf']*2});s=w.assign(wis=[1.,2.])
    s=s.iloc[:1] if damage=='missing' else pd.concat([s,s.iloc[:1]])
    with pytest.raises(ValueError):align_scores(w,s)
