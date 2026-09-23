import copy
import json
import numpy as np
import pytest
from training.r7_policy_selection import select_validation_policy,save_selection,load_selection,validate_policy


def report(*,full=False,gain=0.,prob=.5,mse=4.,cost=1):
    return {'split':'val','checkpoint_sha256':'parent','controller_sha256':'controller',
        'training_identity':'data','evaluation_manifest_sha256':'validation',
        'channels':['t2m','u10'],'units':['K','m/s'],'lead_hours':[6,12],'step_hours':6,'n_evaluated':2,
        'halting_policy':dict(gain_threshold=gain,probability_threshold=prob,min_steps=1,max_steps=4,force_full_depth=full),
        'initializations':[{'init_time':f'2019-01-0{day}T00:00:00',
            'valid_times':[f'2019-01-0{day}T06:00:00',f'2019-01-0{day}T12:00:00'],
            'mse':[[mse,mse],[mse,mse]],'cumulative_reasoning_steps':[4,8] if full else [cost,2*cost]}
            for day in [1,2]]}


def test_selects_cheapest_feasible_without_unit_mixing():
    base=report(full=True)
    good=report(mse=4.04,cost=2)
    bad=report(gain=.1,mse=4.,cost=1)
    bad['initializations'][0]['mse'][1][1]=9.
    result=select_validation_policy(base,[bad,good],relative_rmse_tolerance=.01)
    assert result['selected_policy']==good['halting_policy']
    assert result['candidates'][1]['feasible'] is False
    assert result['scientific_claim'] is False


def test_order_independent_ties_and_case_order():
    base=report(full=True)
    a,b=report(prob=.3,cost=2),report(prob=.7,cost=2)
    first=select_validation_policy(base,[a,b])['selected_policy']
    b['initializations'].reverse()
    assert select_validation_policy(base,[b,a])['selected_policy']==first


def test_full_depth_fallback_and_zero_reference():
    base=report(full=True,mse=0.)
    candidate=report(mse=1e-12)
    result=select_validation_policy(base,[candidate],relative_rmse_tolerance=100.)
    assert result['selected_policy']['force_full_depth']
    candidate=report(mse=0.)
    assert not select_validation_policy(base,[candidate])['selected_policy']['force_full_depth']


def test_pool_squared_errors_before_sqrt():
    base=report(full=True,mse=5.)
    candidate=report(mse=0.)
    candidate['initializations'][1]['mse']=[[16.,16.],[16.,16.]]
    result=select_validation_policy(base,[candidate],relative_rmse_tolerance=0.)
    np.testing.assert_allclose(result['candidates'][1]['rmse'],np.sqrt(8.))
    assert result['selected_policy']['force_full_depth']


@pytest.mark.parametrize('field,value',[('split','test'),('split','train'),('checkpoint_sha256','other'),
    ('controller_sha256','other'),('training_identity','other'),('evaluation_manifest_sha256','other'),
    ('units',['K','knots']),('channels',['u10','t2m']),('step_hours',3)])
def test_rejects_mismatched_or_non_validation_reports(field,value):
    a,b=report(full=True),report()
    b[field]=value
    with pytest.raises(ValueError):
        select_validation_policy(a,[b])


def test_no_silent_intersection_or_duplicates():
    for mutate in ['missing','duplicate','valid_time','cost','negative','nan']:
        a,b=report(full=True),report()
        if mutate=='missing':
            b['initializations'].pop(); b['n_evaluated']=1
        elif mutate=='duplicate':
            b['initializations'][1]=copy.deepcopy(b['initializations'][0])
        elif mutate=='valid_time':
            b['initializations'][0]['valid_times'][0]='2019-01-01T07:00:00'
        elif mutate=='cost':
            b['initializations'][0]['cumulative_reasoning_steps']=[0,9]
        else:
            b['initializations'][0]['mse'][0][0]=-1. if mutate=='negative' else float('nan')
        with pytest.raises(ValueError):
            select_validation_policy(a,[b])
    with pytest.raises(ValueError,match='duplicate'):
        select_validation_policy(report(full=True),[report(),report()])


@pytest.mark.parametrize('tol',[-1.,float('nan'),True])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        select_validation_policy(report(full=True),[report()],relative_rmse_tolerance=tol)


def test_frozen_identity_digest_and_exclusive_output(tmp_path):
    result=select_validation_policy(report(full=True),[report()])
    path=tmp_path/'selection.json'
    save_selection(path,result)
    args=dict(parent_sha256='parent',controller_sha256='controller',training_identity='data')
    policy,sha=load_selection(path,**args)
    assert policy==result['selected_policy'] and len(sha)==64
    with pytest.raises(FileExistsError):
        save_selection(path,result)
    for name in args:
        changed=dict(args); changed[name]='wrong'
        with pytest.raises(ValueError,match='identity'):
            load_selection(path,**changed)
    tampered=copy.deepcopy(result); tampered['selected_policy']['gain_threshold']=100.
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='digest'):
        load_selection(path,**args)


@pytest.mark.parametrize('key,value',[('max_steps',True),('min_steps',0),('min_steps',5),
    ('gain_threshold',float('inf')),('gain_threshold',-1),('probability_threshold',1),('force_full_depth',1)])
def test_invalid_policy(key,value):
    p=report()['halting_policy']; p[key]=value
    with pytest.raises(ValueError):
        validate_policy(p)
