from copy import deepcopy
from itertools import product
import pytest
from training.r7_spatial_study import VARIANTS,SEEDS,spatial_settings,spatial_protocol,summarize_spatial
from data.download.continuous_pilot_replay import PINNED_CONTINUOUS_SHA256
from training.r7_experiment import canonical_digest,make_model
from test_r7_extended_control import report as mock_report


def records():
    return [dict(variant=v,seed=s,depth=k,updates=400,report=mock_report())
            for v,s,k in product(VARIANTS,SEEDS,(1,3))]


def test_predeclared_protocol_and_parameter_fairness():
    p=spatial_protocol(PINNED_CONTINUOUS_SHA256)
    signature=p.pop('protocol_sha256')
    assert canonical_digest(p)==signature and p['updates']==400
    assert not p['test_evaluated'] and not p['controller_fitted']
    for kind in ('generic','process'):
        counts=[]
        for mode in ('global','spatial'):
            k,cfg,_=spatial_settings(kind+'_'+mode)
            counts.append(sum(p.numel() for p in make_model(k,cfg).parameters()))
        assert counts[0]==counts[1]
    with pytest.raises(ValueError):spatial_settings('not-supported')
    with pytest.raises(ValueError):spatial_protocol('0'*64)


def test_complete_spatial_summary():
    rows=summarize_spatial(records(),mock_report())
    assert len(rows)==4*2*4*2
    assert all(r['seeds']==3 and r['seed_rmse_sd']==0 for r in rows)


@pytest.mark.parametrize('bad',['missing','duplicate','test','unpaired','nonfinite','wrong_endpoint'])
def test_incomplete_or_unpaired_study(bad):
    r=records()
    if bad=='missing':r.pop()
    elif bad=='duplicate':r.append(deepcopy(r[0]))
    elif bad=='test':r[0]['report']['split']='test'
    elif bad=='unpaired':r[0]['report']['evaluation_manifest_sha256']='different'
    elif bad=='nonfinite':r[0]['report']['initializations'][0]['mse'][0][0]=float('nan')
    elif bad=='wrong_endpoint':r[0]['updates']=800
    with pytest.raises(ValueError):summarize_spatial(r,mock_report())
