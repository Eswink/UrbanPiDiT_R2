"""Predeclared, bounded CPU learning-budget study. No test-driven selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

VARIANTS = ('native', 'generic', 'process', 'process_no_aux')


@dataclass(frozen=True)
class StudyPlan:
    seeds: tuple[int, ...] = (41, 42, 43)
    endpoints: tuple[int, ...] = (20, 200)
    lead_hours: tuple[int, ...] = (6, 12, 24, 72)
    max_samples: int = 6
    sampling_profile: str = 'january'

    def validate(self):
        for name, values, cap in (('seeds', self.seeds, 3), ('endpoints', self.endpoints, 4),
                                   ('lead_hours', self.lead_hours, 5)):
            if not values or len(values) > cap or len(set(values)) != len(values):
                raise ValueError(f'invalid unique bounded {name}')
            if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
                raise ValueError(f'{name} must contain integers')
        if any(s < 0 or s > 2**31-1 for s in self.seeds):
            raise ValueError('invalid seeds')
        if tuple(sorted(self.endpoints)) != self.endpoints or not 1 <= self.endpoints[0] <= self.endpoints[-1] <= 200:
            raise ValueError('endpoints must increase and stay within 200 updates')
        if tuple(sorted(self.lead_hours)) != self.lead_hours or any(h <= 0 or h > 72 or h % 6 for h in self.lead_hours):
            raise ValueError('lead_hours must increase in six-hour increments through 72h')
        if self.sampling_profile not in ('january', 'four-season'):
            raise ValueError('unknown study sampling profile')
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int):
            raise ValueError('max_samples must be an integer')
        if self.sampling_profile == 'january' and not 1 <= self.max_samples <= 6:
            raise ValueError('January max_samples must be in [1,6]')
        if self.sampling_profile == 'four-season' and self.max_samples != 24:
            raise ValueError('four-season study requires exactly 24 balanced cases')
        return self


def model_settings(variant):
    if variant not in VARIANTS:
        raise ValueError('unknown study variant')
    kind = 'process' if variant == 'process_no_aux' else variant
    model = dict(in_channels=11, out_channels=11, history_steps=2, dim=32,
                 depth=2, heads=4, window_size=4, patch_size=2)
    if kind == 'generic':
        model.update(latent_tokens=16, default_reasoning_steps=3)
    if kind == 'process':
        model.update(anchored_processes=8, free_processes=8, default_reasoning_steps=3)
    return kind, model, .1 if variant == 'process' else 0.


def protocol_payload(plan, source_sha256):
    plan.validate()
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('full lowercase source SHA256 required')
    body = {'format': 'r7-cpu-learning-budget-v1', 'plan': asdict(plan),
            'source_sha256': source_sha256,
            'variants': {v: {'kind': model_settings(v)[0], 'model': model_settings(v)[1],
                            'process_weight': model_settings(v)[2]} for v in VARIANTS},
            'batch_size': 2, 'lr': 2e-4, 'steps': 3, 'device': 'cpu',
            'test_endpoint': plan.endpoints[-1], 'checkpoint_selection': 'fixed-final-endpoint; no test tuning',
            'test_status': 'exploratory previously inspected pilot, not pristine paper test',
            'equal_budget_scope': 'same updates/sample-order/settings; NOT matched FLOPs or wall time'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    return {**body, 'protocol_sha256': hashlib.sha256(encoded).hexdigest()}


def expected_runs(plan):
    plan.validate()
    keys = {(v, s, e, 'val') for v, s, e in product(VARIANTS, plan.seeds, plan.endpoints)}
    keys |= {(v, s, plan.endpoints[-1], 'test') for v, s in product(VARIANTS + ('process_K1',), plan.seeds)}
    return keys


def _report_matrix(report):
    channels, units, leads = report['channels'], report['units'], report['lead_hours']
    if not channels or len(set(channels)) != len(channels) or len(channels) != len(units):
        raise ValueError('invalid metric channel/unit schema')
    cases = report['initializations']
    if not cases or report['n_evaluated'] != len(cases):
        raise ValueError('initialization count mismatch')
    ids = [c['init_time'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate initialization')
    if any(len(c['valid_times']) != len(leads) for c in cases):
        raise ValueError('valid-time dimension mismatch')
    mse = np.asarray([c['mse'] for c in cases], dtype=np.float64)
    if mse.shape != (len(cases), len(leads), len(channels)) or not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError('invalid finite per-case MSE matrix')
    signature = (tuple(channels), tuple(units), tuple(leads), report['split'],
                 report['evaluation_manifest_sha256'],
                 tuple((c['init_time'], tuple(c['valid_times'])) for c in cases))
    return signature, np.sqrt(mse.mean(axis=0))


def summarize_study(records, plan):
    """Mean/SD of seed-level RMSE, separately for every variable and horizon.

    This is initialization sensitivity, not an uncertainty interval for weather
    events. Never average RMSE from different physical units or unpaired cases.
    """
    required = expected_runs(plan)
    found, signatures, groups, raw = set(), {}, {}, []
    for record in records:
        key = (record['variant'], record['seed'], record['updates'], record['split'])
        if key not in required or key in found:
            raise ValueError('unexpected or duplicate study run')
        found.add(key)
        report = record['report']
        if report['split'] != record['split'] or tuple(report['lead_hours']) != plan.lead_hours:
            raise ValueError('study split/horizon mismatch')
        signature, matrix = _report_matrix(report)
        if plan.sampling_profile == 'four-season':
            from collections import Counter
            import pandas as pd
            counts = Counter(pd.Timestamp(c['init_time']).month for c in report['initializations'])
            if dict(counts) != {1:6,4:6,7:6,9:6}:
                raise ValueError('study case selection is not season-balanced')
        if len(report['initializations']) != plan.max_samples:
            raise ValueError('study did not evaluate all predeclared cases')
        split = record['split']
        if split in signatures and signature != signatures[split]:
            raise ValueError('unpaired study cases, manifests or metric schemas')
        signatures[split] = signature
        for h, lead in enumerate(report['lead_hours']):
            for c, (name, unit) in enumerate(zip(report['channels'], report['units'])):
                group = (record['variant'], record['updates'], split, lead, name, unit)
                value = float(matrix[h, c])
                groups.setdefault(group, []).append(value)
                raw.append(dict(variant=record['variant'], seed=record['seed'], updates=record['updates'],
                                split=split, lead_hours=lead, variable=name, unit=unit, rmse=value))
    if found != required:
        raise ValueError(f'missing study runs: {sorted(required-found)}')
    summary = []
    for (variant, updates, split, lead, name, unit), values in sorted(groups.items()):
        summary.append(dict(variant=variant, updates=updates, split=split, lead_hours=lead,
                            variable=name, unit=unit, seeds=len(values),
                            seed_rmse_mean=float(np.mean(values)),
                            seed_rmse_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    return raw, summary


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def _csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_study(source, receipt, output_dir, *, plan=StudyPlan()):
    """Only pinned real pilot profiles, no network downloads or GPU fallback."""
    import torch
    from data.download.pressure_pilot_replay import copy_verified_pressure_pilot
    from data.download.earthmover_pilot import FIELDS
    from data.preprocess.r7_preflight import prepare_local
    from .r7_experiment import dataset_identity, make_model
    from .r7_local_runner import run_local_updates
    from .r7_evaluate import evaluate_local
    plan.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    copier = copy_verified_pressure_pilot
    if plan.sampling_profile == 'four-season':
        from data.download.seasonal_pilot_replay import copy_verified_seasonal_pilot
        copier = copy_verified_seasonal_pilot
    source, audited = copier(source, receipt,
        out/'source'/'era5_pressure_pilot.nc', out/'source'/'receipt.json')
    protocol = protocol_payload(plan, audited['source_netcdf_sha256'])
    _json(out/'protocol.json', protocol)  # Frozen BEFORE preprocessing/training.
    config = {'channels': [{'variable':n, 'name':n} for _, _, n in FIELDS],
              'split_years': {'train':[2018], 'val':[2019], 'test':[2020]},
              'history_steps':2, 'history_interval_hours':6, 'lead_time_hours':6,
              'sample_stride_hours':6, 'time_chunk':8, 'compute_process_targets':True}
    preparation = prepare_local(source, config, write=True, store_path=out/'cache.zarr',
        manifest_dir=out/'manifests', max_raw_gib=.01)
    manifests = {s:out/'manifests'/f'{s}.jsonl' for s in ('train', 'val', 'test')}
    if plan.sampling_profile == 'four-season':
        from data.download.seasonal_sampling import requested_times, write_balanced_manifest
        for split in ('val', 'test'):
            manifests[split] = write_balanced_manifest(manifests[split], requested_times('four-season'))
    identity, dataset = dataset_identity(manifests['train'])
    records, resources = [], []
    for variant, seed in product(VARIANTS, plan.seeds):
        kind, model_cfg, pw = model_settings(variant)
        params = sum(p.numel() for p in make_model(kind, model_cfg).parameters())
        previous = None
        for endpoint in plan.endpoints:
            checkpoint, training = run_local_updates(dataset, kind=kind, model_config=model_cfg,
                data_identity=identity, output_dir=out/'training'/f'{variant}_{seed}',
                total_updates=endpoint, batch_size=2, accumulation=1, steps=3, seed=seed,
                lr=2e-4, process_weight=pw, device_name='cpu', resume=previous)
            previous = checkpoint
            resources.append({'variant':variant, 'seed':seed, 'endpoint':endpoint, 'params':params,
                'updates_this_run':training['updates_this_run'],
                'elapsed_seconds_including_batch_reads':training['elapsed_seconds'],
                'samples_this_run':sum(r['samples'] for r in training['losses'])})
            splits = ('val', 'test') if endpoint == plan.endpoints[-1] else ('val',)
            for split in splits:
                folder = out/'evaluation'/f'{variant}_{seed}_{endpoint}_{split}'
                report = evaluate_local(manifests[split], output_dir=folder, checkpoint=checkpoint,
                    lead_hours=plan.lead_hours, max_samples=plan.max_samples, device_name='cpu')
                records.append(dict(variant=variant, seed=seed, updates=endpoint, split=split, report=report))
            if variant == 'process' and endpoint == plan.endpoints[-1]:
                report = evaluate_local(manifests['test'], output_dir=out/'evaluation'/f'process_K1_{seed}',
                    checkpoint=checkpoint, lead_hours=plan.lead_hours, max_samples=plan.max_samples,
                    device_name='cpu', reasoning_steps=1)
                records.append(dict(variant='process_K1', seed=seed, updates=endpoint, split='test', report=report))
            print(json.dumps({'variant':variant, 'seed':seed, 'endpoint':endpoint, 'phase':'evaluated'}), flush=True)
    baselines = {s:evaluate_local(manifests[s], output_dir=out/'evaluation'/f'persistence_{s}',
        lead_hours=plan.lead_hours, max_samples=plan.max_samples, device_name='cpu') for s in ('val', 'test')}
    raw, summary = summarize_study(records, plan)
    # Independently assert persistence used the same complete case lists.
    for s, baseline in baselines.items():
        first = next(r['report'] for r in records if r['split'] == s)
        if _report_matrix(baseline)[0] != _report_matrix(first)[0]:
            raise ValueError('persistence is not paired with the neural models')
    _csv(out/'seed_rmse.csv', raw)
    _csv(out/'seed_summary.csv', summary)
    result = dict(format='r7-cpu-study-result-v1', scientific_claim=False, gpu_used=False,
        protocol=protocol, preparation=preparation, resources=resources, records=records,
        persistence=baselines, summary=summary, elapsed_seconds=time.monotonic()-started,
        limitations=['Small sampled tile and correlated cases; not representative regional weather skill',
            'Mean/SD across seeds describe initialization variation, not statistical significance',
            'Training total includes auxiliary loss only for process; do not compare that scalar as forecast RMSE',
            'No controller retuning, checkpoint selection or hyperparameter search on test',
            'Equal update/sample budgets do not establish matched FLOPs or latency'])
    _json(out/'study_result.json', result)
    return result
