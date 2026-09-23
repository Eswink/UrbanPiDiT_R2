from copy import deepcopy
import numpy as np
import pytest
import torch
from training.r7_baseline_study import VARIANTS,control_settings,TARGET_PARAMETERS,summarize_controls
from training.r7_experiment import make_model
from test_r7_seasonal_study import records_fixture


@pytest.mark.parametrize('name',VARIANTS)
def test_controls_have_declared_budget_and_trainable_forecasts(name):
    torch.manual_seed(42)
    kind,cfg,pw=control_settings(name);model=make_model(kind,cfg)
    n=sum(p.numel() for p in model.parameters())
    assert abs(n/TARGET_PARAMETERS-1)<.1
    history=torch.randn(1,2,11,12,12)
    forecast=model({'coarse_history':history}).forecast
    assert forecast.shape==(1,11,12,12) and torch.isfinite(forecast).all()
    (forecast-history[:,-1]+.1).square().mean().backward()
    assert any(p.grad is not None and float(p.grad.norm())>0 for p in model.parameters())
    if name=='afno_small':
        assert any(float(layer.mix.w1.grad.norm())>0 for layer in model.layers)


def evidence_fixture():
    _,old=records_fixture()
    records=[dict(variant=name,split=split,report=deepcopy(next(r['report'] for r in old if r['split']==split)))
             for name in VARIANTS+('persistence',) for split in ('val','test')]
    for r in records:
        r['report']['lead_hours']=[6,12,24,72]
        import pandas as pd
        for c in r['report']['initializations']:
            c['valid_times']=[(pd.Timestamp(c['init_time'])+pd.Timedelta(hours=h)).isoformat() for h in [6,12,24,72]]
            c['mse']=[[1.],[2.],[3.],[4.]]
    profiles={name:dict(device='cpu',precision='fp32',cpu_threads=2,input_shape=[1,2,11,12,12],
        warmup_excluded=5,repetitions=20,cuda_memory={'peak_allocated_bytes':None},seconds_per_batch=[.01]*20,
        median_seconds_per_batch=.01,input_sha256='input',hardware='cpu',torch='version',python='version',parameters=83319) for name in VARIANTS}
    return records,profiles


def test_complete_paired_controls_and_real_duration_summary():
    records,profiles=evidence_fixture()
    rows=summarize_controls(records,profiles)
    assert len(rows)==7*2*4
    assert next(r for r in rows if r['variant']=='unet')['median_cpu_forward_ms']==10.
    assert next(r for r in rows if r['variant']=='persistence')['median_cpu_forward_ms'] is None


@pytest.mark.parametrize('bad',['missing','duplicate','input','median','gpu','parameters','pair'])
def test_invalid_control_evidence_refused(bad):
    records,profiles=evidence_fixture()
    if bad=='missing':records.pop()
    elif bad=='duplicate':records.append(deepcopy(records[0]))
    elif bad=='input':profiles['unet']['input_sha256']='other'
    elif bad=='median':profiles['unet']['median_seconds_per_batch']=.001
    elif bad=='gpu':profiles['unet']['cuda_memory']['peak_allocated_bytes']=10
    elif bad=='parameters':profiles['unet']['parameters']=1
    else:records[0]['report']['evaluation_manifest_sha256']='other'
    with pytest.raises(ValueError):summarize_controls(records,profiles)
