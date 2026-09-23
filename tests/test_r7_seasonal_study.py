"""Synthetic analytics fixtures, never scientific forecast evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from training.r7_cpu_study import StudyPlan, expected_runs, summarize_study
from training.r7_seasonal_study import season_summaries, adaptive_comparison


def records_fixture():
    plan=StudyPlan(seeds=(41,42),endpoints=(20,200),lead_hours=(6,),max_samples=24,sampling_profile='four-season')
    records=[]
    for variant,seed,endpoint,split in sorted(expected_runs(plan)):
        year=2019 if split=='val' else 2020
        cases=[]
        for month in (1,4,7,9):
            for i in range(6):
                init=pd.Timestamp(year,month,1)+pd.Timedelta(hours=6*(i+1))
                cases.append(dict(init_time=init.isoformat(),valid_times=[(init+pd.Timedelta(hours=6)).isoformat()],
                                  mse=[[float(month+seed-40)]],cumulative_reasoning_steps=[3]))
        report=dict(channels=['t2m'],units=['K'],lead_hours=[6],step_hours=6,split=split,
                    evaluation_manifest_sha256=split,n_evaluated=24,initializations=cases)
        records.append(dict(variant=variant,seed=seed,updates=endpoint,split=split,report=report))
    return plan,records


def test_season_summaries_use_every_month_without_changing_reports():
    plan,records=records_fixture()
    original=deepcopy(records)
    rows=season_summaries(records,plan)
    assert records==original and set(r['month'] for r in rows)=={1,4,7,9}
    r=next(r for r in rows if r['month']==4 and r['variant']=='process' and r['split']=='test')
    assert r['seed_rmse_mean']==pytest.approx((np.sqrt(5)+np.sqrt(6))/2)
    assert r['seeds']==2


def test_imbalanced_case_set_refused_even_with_twenty_four_cases():
    plan,records=records_fixture()
    for record in records:
        for c in record['report']['initializations']:
            # All existing times still valid/unique; no July/September coverage.
            t=pd.Timestamp(c['init_time'])
            if t.month==7:t=t.replace(month=2)
            if t.month==9:t=t.replace(month=3)
            c['init_time']=t.isoformat();c['valid_times']=[(t+pd.Timedelta(hours=6)).isoformat()]
    with pytest.raises(ValueError,match='season-balanced'): summarize_study(records,plan)


@pytest.mark.parametrize('plan',[StudyPlan(max_samples=24),StudyPlan(sampling_profile='four-season'),StudyPlan(sampling_profile='unknown')])
def test_profile_and_case_count_must_be_consistent(plan):
    with pytest.raises(ValueError): plan.validate()


def adaptive_pair():
    _,records=records_fixture()
    base=deepcopy(next(r['report'] for r in records if r['split']=='test'))
    base.update(checkpoint_sha256='parent',controller_sha256='controller',training_identity='training',
        halting_policy=dict(gain_threshold=0.,probability_threshold=.5,min_steps=1,max_steps=3,force_full_depth=True))
    dynamic=deepcopy(base)
    dynamic['halting_policy']['force_full_depth']=False
    dynamic['policy_selection_sha256']='validated-policy'
    for c in dynamic['initializations']:c['cumulative_reasoning_steps']=[1]
    return base,dynamic


def test_compute_reduction_and_heldout_failure_are_both_reported():
    base,dynamic=adaptive_pair()
    for c in dynamic['initializations']:c['mse'][0][0]*=4
    result=adaptive_comparison(base,dynamic)
    assert result['relative_step_reduction']==pytest.approx([2/3])
    assert result['test_tolerance_satisfied'] is False
    assert result['failed_variable_horizon_pairs']==1
    assert result['rows'][0]['selected_rmse']==pytest.approx(2*result['rows'][0]['fixed_rmse'])


@pytest.mark.parametrize('mode',['parent','manifest','case','time','count','selection','split'])
def test_invalid_adaptive_evidence_refused(mode):
    base,dynamic=adaptive_pair()
    if mode=='parent':dynamic['checkpoint_sha256']='other'
    elif mode=='manifest':dynamic['evaluation_manifest_sha256']='other'
    elif mode=='case':dynamic['initializations'][0]['init_time']='2020-01-30T00:00:00'
    elif mode=='time':dynamic['initializations'][0]['valid_times']=['2020-01-30T00:00:00']
    elif mode=='count':dynamic['initializations'][0]['cumulative_reasoning_steps']=[0]
    elif mode=='selection':dynamic.pop('policy_selection_sha256')
    elif mode=='split':dynamic['split']='val'
    with pytest.raises(ValueError):adaptive_comparison(base,dynamic)
