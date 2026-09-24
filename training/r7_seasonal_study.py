"""Fixed seasonal study plus train-only calibration and validation-frozen halting."""
from __future__ import annotations
from dataclasses import replace
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .r7_cpu_study import StudyPlan, run_study, summarize_study, protocol_payload, _report_matrix, _json, _csv
from .r7_policy_selection import digest, validate_policy
from .r7_paired_comparison import _rows

PLAN = StudyPlan(max_samples=24, sampling_profile='four-season')
CONTROLLER_PROTOCOL = dict(updates=120, batch_size=2, max_steps=3, hidden=64, lr=1e-3,
                           gain_thresholds=[0.], probability_thresholds=[.3,.5,.7],
                           relative_rmse_tolerance=.01)
MONTHS = (1,4,7,9)


def season_summaries(records, plan=PLAN):
    """Validate full balance/pairing before viewing subsets; no changed predictions."""
    summarize_study(records, plan)
    output = []
    for month in MONTHS:
        views = []
        for record in records:
            r = record['report']
            cases = [c for c in r['initializations'] if pd.Timestamp(c['init_time']).month == month]
            views.append(dict(record, report=dict(r, initializations=cases, n_evaluated=len(cases))))
        # A six-case statistics view uses the original run-set validator. This
        # does not reread data, change training/sampling, or create a new protocol.
        _, rows = summarize_study(views, replace(plan, max_samples=6, sampling_profile='january'))
        output.extend(dict(row, month=month) for row in rows)
    return output


def _counts(report):
    _rows(report)  # Includes exact valid_time = init_time + lead checks.
    policy = validate_policy(report['halting_policy'])
    step = report['step_hours']
    if isinstance(step,bool) or not isinstance(step,int) or step <= 0:
        raise ValueError('positive integer forecast step required')
    counts = []
    for case in report['initializations']:
        values = case['cumulative_reasoning_steps']
        if len(values) != len(report['lead_hours']):
            raise ValueError('reasoning count/horizon mismatch')
        previous_h = previous_c = 0
        for h,c in zip(report['lead_hours'], values):
            if h % step or isinstance(c,bool) or not isinstance(c,int):
                raise ValueError('integer reasoning counts and divisible horizons required')
            transitions = (h-previous_h)//step
            low = policy['max_steps'] if policy['force_full_depth'] else policy['min_steps']
            if not transitions*low <= c-previous_c <= transitions*policy['max_steps']:
                raise ValueError('actual counts contradict the policy')
            previous_h, previous_c = h,c
        counts.append(values)
    return np.asarray(counts,dtype=np.float64).mean(0)


def adaptive_comparison(fixed, selected, *, tolerance=.01):
    """Report accuracy failures as results, not exceptions or retuned thresholds."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid accuracy tolerance')
    if fixed['split'] != 'test' or selected['split'] != 'test':
        raise ValueError('this comparison is held-out test only')
    a, rmse_fixed = _report_matrix(fixed)
    b, rmse_selected = _report_matrix(selected)
    if a != b:
        raise ValueError('unpaired adaptive test reports')
    for key in ('checkpoint_sha256','controller_sha256','training_identity','step_hours'):
        if not fixed.get(key) or fixed[key] != selected.get(key):
            raise ValueError('adaptive model/data identity mismatch')
    if not fixed['halting_policy']['force_full_depth'] or not selected.get('policy_selection_sha256'):
        raise ValueError('fixed reference and validation-frozen selected policy required')
    full, adaptive = _counts(fixed), _counts(selected)
    passed = rmse_selected <= rmse_fixed*(1+tolerance)
    rows = [dict(lead_hours=h,variable=name,unit=fixed['units'][j],
                 fixed_rmse=float(rmse_fixed[i,j]),selected_rmse=float(rmse_selected[i,j]),
                 within_tolerance=bool(passed[i,j]))
            for i,h in enumerate(fixed['lead_hours']) for j,name in enumerate(fixed['channels'])]
    return dict(scientific_claim=False, tolerance=tolerance, test_tolerance_satisfied=bool(passed.all()),
        failed_variable_horizon_pairs=int((~passed).sum()), rows=rows,
        fixed_mean_cumulative_steps=full.tolist(),selected_mean_cumulative_steps=adaptive.tolist(),
        relative_step_reduction=(1-adaptive/full).tolist(),
        note='Reasoning counts are NOT wall-clock latency or FLOPs; test failures are retained')


def run_seasonal_study(source, receipt, output_dir, *, deadline_seconds=1080):
    from data.download.seasonal_pilot_replay import verify_seasonal_pilot
    from .r7_calibration_runner import run_calibration
    from .r7_policy_selection import run_policy_search
    from .r7_evaluate import evaluate_local
    audited = verify_seasonal_pilot(source, receipt)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(format='r7-seasonal-study-v1', forecast=protocol_payload(PLAN,audited['source_netcdf_sha256']),
        controller=CONTROLLER_PROTOCOL, months=list(MONTHS),
        selection='six fixed complete 72h cases per sampled month',
        test_status='exploratory; no test-driven checkpoint/threshold selection', scientific_claim=False)
    protocol['signature'] = digest(protocol)
    _json(out/'experiment_protocol.json',protocol)  # BEFORE training or calibration.
    started = time.monotonic()
    study = run_study(source, receipt, out/'forecasts', plan=PLAN)
    month_rows = season_summaries(study['records'])
    _csv(out/'season_seed_summary.csv',month_rows)
    manifests = out/'forecasts'/'manifests'
    controllers = []
    for seed in PLAN.seeds:
        if time.monotonic() - started > deadline_seconds:
            raise RuntimeError('seasonal study wall budget exhausted')
        checkpoint = out/'forecasts'/'training'/f'process_{seed}'/'update_0000200.pt'
        controller, calibration = run_calibration(checkpoint,manifests/'train.jsonl',
            output_dir=out/'controllers'/str(seed),updates=120,batch_size=2,max_steps=3,
            hidden=64,lr=1e-3,seed=seed,device_name='cpu')
        folder = out/'selection'/str(seed)
        selected = run_policy_search(manifests/'val_balanced.jsonl',checkpoint=checkpoint,
            controller_checkpoint=controller,output_dir=folder,
            gain_thresholds=(0.,),probability_thresholds=(.3,.5,.7),
            lead_hours=PLAN.lead_hours,max_samples=24,relative_rmse_tolerance=.01,device_name='cpu')
        common = dict(checkpoint=checkpoint,controller_checkpoint=controller,
                      lead_hours=PLAN.lead_hours,max_samples=24,device_name='cpu')
        fixed = evaluate_local(manifests/'test_balanced.jsonl',output_dir=out/'adaptive_test'/f'fixed_{seed}',
                               force_full_depth=True,**common)
        dynamic = evaluate_local(manifests/'test_balanced.jsonl',output_dir=out/'adaptive_test'/f'selected_{seed}',
                                 policy_selection=folder/'selection.json',**common)
        process_report = next(r['report'] for r in study['records'] if r['variant']=='process' and
                              r['seed']==seed and r['split']=='test')
        if (_report_matrix(fixed)[0] != _report_matrix(process_report)[0] or
            not np.allclose([r['mse'] for r in fixed['initializations']],
                            [r['mse'] for r in process_report['initializations']],rtol=1e-5,atol=1e-8)):
            raise ValueError('full-depth adapter disagrees with the fixed process reference')
        comparison = adaptive_comparison(fixed,dynamic)
        controllers.append(dict(seed=seed,calibration=calibration,selection=selected,
                                fixed_test=fixed,selected_test=dynamic,comparison=comparison))
    result = dict(format='r7-seasonal-study-results-v1',scientific_claim=False,gpu_used=False,
        protocol=protocol,study_result='forecasts/study_result.json',
        season_summary=month_rows,controllers=controllers,elapsed_seconds=time.monotonic()-started,
        limitations=['Small eight-day seasonal excerpts and one spatial tile, not representative regional verification',
            'Three seeds describe initialization variability, not independent weather event significance',
            'No test retuning; both full-depth fallbacks and held-out tolerance failures are valid outcomes',
            '200 forecaster updates and120 controller updates are bounded pilots, not convergence guarantees'])
    _json(out/'seasonal_study_result.json',result)
    return result
