from copy import deepcopy
import numpy as np
import pytest
from training.r7_cpu_study import StudyPlan, expected_runs, model_settings, protocol_payload, summarize_study


@pytest.mark.parametrize('kwargs', [dict(seeds=(42,42)),dict(seeds=(True,)),dict(seeds=(-1,)),
    dict(endpoints=(20,201)),dict(endpoints=(200,20)),dict(endpoints=(0,)),dict(lead_hours=(7,)),
    dict(lead_hours=(12,6)),dict(max_samples=True),dict(max_samples=7)])
def test_study_boundaries(kwargs):
    with pytest.raises(ValueError):
        StudyPlan(**kwargs).validate()


def test_frozen_protocol_and_auxiliary_ablation():
    a = protocol_payload(StudyPlan(), 'a'*64)
    b = protocol_payload(StudyPlan(), 'a'*64)
    assert a == b
    assert a['test_endpoint'] == 200
    assert a['protocol_sha256'] != protocol_payload(StudyPlan(seeds=(42,)), 'a'*64)['protocol_sha256']
    p, cfg, pw = model_settings('process')
    q, other, qw = model_settings('process_no_aux')
    assert p == q == 'process' and cfg == other and pw == .1 and qw == 0.
    assert not any(k[-1]=='test' and k[2]==20 for k in expected_runs(StudyPlan()))


def fixture_records():
    plan = StudyPlan(seeds=(1,2), endpoints=(2,3), lead_hours=(6,), max_samples=2)
    records = []
    for variant, seed, endpoint, split in sorted(expected_runs(plan)):
        mse = [[[1.,10000.]], [[9.,90000.]]] if seed == 1 else [[[4.,40000.]], [[16.,160000.]]]
        report = dict(channels=['t2m','mslp'], units=['K','Pa'], lead_hours=[6], split=split,
            evaluation_manifest_sha256=split, n_evaluated=2, initializations=[
                dict(init_time=f'{split}-{i}',valid_times=[f'{split}-{i}+6'],mse=mse[i]) for i in range(2)])
        records.append(dict(variant=variant,seed=seed,updates=endpoint,split=split,report=report))
    return plan, records


def test_seed_statistics_do_not_pool_units_or_average_case_rmse():
    plan, records = fixture_records()
    raw, summary = summarize_study(records,plan)
    rows = [r for r in summary if r['variant']=='process' and r['split']=='test']
    temp, pressure = next(r for r in rows if r['variable']=='t2m'), next(r for r in rows if r['variable']=='mslp')
    assert temp['seed_rmse_mean'] == pytest.approx((np.sqrt(5)+np.sqrt(10))/2)
    assert temp['seed_rmse_sd'] == pytest.approx(np.std([np.sqrt(5),np.sqrt(10)],ddof=1))
    assert pressure['seed_rmse_mean'] == pytest.approx(100*temp['seed_rmse_mean'])
    assert all(r['seeds']==2 for r in summary)


@pytest.mark.parametrize('corrupt', ['missing_run','duplicate_run','nan','negative','case','unit','manifest','count','split'])
def test_summary_refuses_incomplete_or_unpaired_evidence(corrupt):
    plan, records = fixture_records()
    records = deepcopy(records)
    first = records[0]['report']
    if corrupt == 'missing_run': records.pop()
    elif corrupt == 'duplicate_run': records.append(deepcopy(records[0]))
    elif corrupt == 'nan': first['initializations'][0]['mse'][0][0] = float('nan')
    elif corrupt == 'negative': first['initializations'][0]['mse'][0][0] = -1
    elif corrupt == 'case': first['initializations'][0]['init_time'] = 'different'
    elif corrupt == 'unit': first['units'][0] = 'degC'
    elif corrupt == 'manifest': first['evaluation_manifest_sha256'] = 'different'
    elif corrupt == 'count': first['n_evaluated'] = 1
    elif corrupt == 'split': first['split'] = 'train'
    with pytest.raises(ValueError): summarize_study(records, plan)
