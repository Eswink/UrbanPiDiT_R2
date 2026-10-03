from __future__ import annotations
import copy, csv, json, math
from datetime import datetime, timedelta
import os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.r7_v2_utc_contract import (ARMS, FORMAT, HERE, HOURS, REFERENCE_SOURCES, KERNELS, LEADS, LIMITATIONS, REGIONS, SEEDS,
                          digest, jobs_for, pin, read_json, sha256_file, source_pins, write_csv, write_json)
from training.r7_v2_utc_frozen_stats import STATISTICS, pooled_statistics
from training.r7_v2_utc_statistics import bad_reasons
ROOT = None
COUNTS = {6: 8, 12: 7, 24: 6, 48: 5, 72: 3}
VARIABLES = [f'synthetic_v{i:02d}' for i in range(17)]
UNITS = ['K', 'm s-1', 'Pa', 'm2 s-2', 'kg kg-1'] * 3 + ['K', 'Pa']
STDS = [1.5 + i / 4 for i in range(17)]
def synthetic_row(case, variable_index, region, job, kind, all_zero=False):
    i, v, seed, arm, k = case, variable_index, job['seed'], job['arm'], job['reasoning_steps']
    t = float((i + 1) ** 2 * (1 + v / 20))
    p = float((i % 3 + 1) ** 2 * (1 + v / 20))
    if kind == 'model':
        p *= 1 + (seed - 40) / 10 + list(ARMS['B'] + ARMS['C']).index(arm) / 8 + k / 20
    r = -0.6 if i % 4 == 1 else (0.2 if i % 4 == 0 else 0.9)
    if v == 0 and i % 4 == 0:
        p = t = 0.0
    if v == 1 and i == 0:
        p = 0.0
    if kind == 'climatology':
        p = 0.0
    if all_zero:
        p = t = 0.0
    d = r * math.sqrt(p * t)
    region_scale = {'full': 1., 'interior': .7, 'edge_2': 1.3}[region]
    p, t, d = p * region_scale, t * region_scale, d * region_scale
    scale = STDS[v] ** 2
    physical = {'mse': (p + t - 2 * d) * scale, 'climatology_mse': t * scale,
                'acc_dot': d * scale, 'acc_forecast_energy': p * scale, 'acc_target_energy': t * scale}
    metrics = pooled_statistics([physical])
    row = {'region': region, 'margin_cells': 0 if region == 'full' else 2,
           'n_grid_points': {'full': 100, 'interior': 36, 'edge_2': 64}[region],
           'full_area_fraction': {'full': 1., 'interior': .36, 'edge_2': .64}[region],
           'lead_hours': job['lead'], 'variable': VARIABLES[v], 'unit': UNITS[v],
           'rmse': metrics['rmse'], 'mse': physical['mse'], 'rmse_climatology': metrics['rmse_climatology'],
           'climatology_mse': physical['climatology_mse'], 'mse_climatology': physical['climatology_mse'],
           'acc_dot': d, 'acc_forecast_energy': p, 'acc_target_energy': t,
           'acc_statistic_units': 'normalized_anomaly_squared', 'mse_skill': metrics['mse_skill'],
           'acc': metrics['pooled_acc'], 'acc_status': metrics['acc_status'],
           'skill_status': 'undefined_zero_climatology_energy' if metrics['mse_skill'] is None else 'defined',
           'n_initializations': 1, 'bad_reasons': bad_reasons(metrics)}
    if kind != 'model':
        row = {'baseline_kind': kind, 'zero_train_updates': 0, 'parameters': 0, 'trainable_parameters': 0, **row}
    return row
def fixture_stage(directory, stage, *, all_zero=False):
    directory.mkdir(parents=True)
    start = datetime(2001, 1, 1)
    cases = {str(lead): {'n_available': n, 'cases': [
        [(start + timedelta(hours=6 * i)).isoformat(), [(start + timedelta(hours=6 * i + lead)).isoformat()]]
        for i in range(n)]} for lead, n in COUNTS.items()}
    refs = {'origins': REFERENCE_SOURCES}
    files = {name: refs['origins'][name]['sha256'] for name in
             ('training/r7_v2_tables.py', 'training/r7_v2_results.py')}
    evaluations = jobs_for(stage)
    training = [{'phase': 'train', 'seed': seed, 'arm': arm, 'lead': None, 'reasoning_steps': 4}
                for seed in SEEDS[stage] for arm in ARMS[stage]]
    protocol = {'format': 'synthetic-r7-v2-schema-only', 'stage': stage, 'output': str(directory),
                'scientific_claim': False, 'limitations': ['SYNTHETIC ONLY: invented statistics, not weather or actual training.'],
                'test_read': False, 'jobs': training + evaluations, 'arm_configs': {arm: {} for arm in ARMS[stage]},
                'code': {'files': files, 'source_tree_sha256': digest(files),
                         'model_code_sha256': digest('SYNTHETIC_MODEL_' + stage), 'code_zip_sha256': digest('SYNTHETIC_ZIP_' + stage),
                         'base_commit': '616b029dce569b92bd08295737981512180a1ad3' if stage == 'B' else 'c' * 40},
                'sources': {'source_sha256': digest('SYNTHETIC_SOURCE'), 'val_manifest_sha256': digest('SYNTHETIC_VAL_METADATA')},
                'data': {'channels': VARIABLES, 'units': UNITS, 'normalization_std': STDS, 'test_read': False,
                         'data_identity': digest('SYNTHETIC_DATA'), 'evaluation_cases': cases}}
    protocol['protocol_sha256'] = digest(protocol)
    write_json(directory / 'protocol.json', protocol)
    (directory / 'workers').mkdir()
    pins = {'protocol.json': sha256_file(directory / 'protocol.json')}
    records = []
    for job in evaluations:
        seed, arm, lead, k = (job[key] for key in ('seed', 'arm', 'lead', 'reasoning_steps'))
        out = directory / f'seed{seed}/evaluation/{arm}/lead_{lead:03d}h/k{k}'
        out.mkdir(parents=True)
        initializations, reference_initializations, model_csv, baseline_csv = [], [], [], []
        for i, (init, valid) in enumerate(cases[str(lead)]['cases']):
            ident = {'sample_id': f'SYNTHETIC_CASE_{i:03d}', 'init_time': init, 'valid_times': valid}
            model = [synthetic_row(i, v, region, job, 'model', all_zero) for region in REGIONS for v in range(17)]
            references = [synthetic_row(i, v, region, job, kind, all_zero)
                          for kind in ('persistence', 'climatology') for region in REGIONS for v in range(17)]
            initializations.append({**ident, 'cumulative_reasoning_steps': [k * lead // 6], 'region_metrics': model})
            reference_initializations.append({**ident, 'region_metrics': references})
            model_csv.extend({**ident, 'valid_time': valid[0], 'reasoning_steps': k, **row} for row in model)
            baseline_csv.extend({**ident, 'valid_time': valid[0], 'reasoning_steps': None, **row} for row in references)
        provenance = {'format': 'r7-v2-evaluation-v1', 'scientific_claim': False,
                      'limitations': protocol['limitations'], 'test_read': False, 'split': 'val',
                      'channels': VARIABLES, 'units': UNITS, 'lead_hours': [lead], 'step_hours': 6,
                      'reasoning_steps': k, 'checkpoint_training_steps': 4, 'independently_trained_k1': False,
                      'inference_options': {'reasoning_steps': k}, 'model_code_sha256': protocol['code']['model_code_sha256'],
                      'training_identity': protocol['data']['data_identity'],
                      'evaluation_manifest_sha256': protocol['sources']['val_manifest_sha256'],
                      'n_evaluated': COUNTS[lead], 'n_available_windows': COUNTS[lead],
                      'checkpoint_sha256': digest(['SYNTHETIC_CHECKPOINT', stage, seed, arm]),
                      'initializations': initializations, 'baseline_initializations': reference_initializations}
        write_csv(out / 'per_case_metrics.csv', model_csv)
        write_csv(out / 'baseline_per_case_metrics.csv', baseline_csv)
        write_json(out / 'provenance.json', provenance)
        artifacts = {name: sha256_file(out / name) for name in
                     ('per_case_metrics.csv', 'baseline_per_case_metrics.csv', 'provenance.json')}
        receipt = {'status': 'success', 'job': job, 'scientific_claim': False, 'test_read': False,
                   'limitations': protocol['limitations'], 'protocol_sha256': protocol['protocol_sha256'],
                   'model_code_sha256': protocol['code']['model_code_sha256'], 'source_tree_sha256': protocol['code']['source_tree_sha256'],
                   'code_zip_sha256': protocol['code']['code_zip_sha256'], 'source_sha256': protocol['sources']['source_sha256'],
                   'data_identity': protocol['data']['data_identity'], 'n_evaluated': COUNTS[lead],
                   'n_available_windows': COUNTS[lead], 'checkpoint_sha256': provenance['checkpoint_sha256'],
                   'artifact_sha256': artifacts, 'per_case_metrics_csv': str(out / 'per_case_metrics.csv'),
                   'baseline_per_case_metrics_csv': str(out / 'baseline_per_case_metrics.csv'),
                   'provenance': str(out / 'provenance.json'), 'evaluation_provenance': provenance}
        name = f'evaluate_seed{seed}_{arm}_lead{lead:03d}h_k{k}.json'
        write_json(directory / 'workers' / name, receipt)
        records.append({'job': job, 'receipt': pin(directory / 'workers' / name),
                        'per_case_metrics': pin(out / 'per_case_metrics.csv'),
                        'baseline_per_case_metrics': pin(out / 'baseline_per_case_metrics.csv'), 'provenance': pin(out / 'provenance.json')})
        pins['workers/' + name] = sha256_file(directory / 'workers' / name)
        for file in artifacts:
            pins[(out / file).relative_to(directory).as_posix()] = artifacts[file]
    for job in training:
        receipt = {'status': 'success', 'job': job, 'scientific_claim': False, 'test_read': False,
                   'limitations': protocol['limitations'], 'protocol_sha256': protocol['protocol_sha256'],
                   'model_code_sha256': protocol['code']['model_code_sha256'],
                   'source_tree_sha256': protocol['code']['source_tree_sha256'],
                   'data_identity': protocol['data']['data_identity']}
        name = f"train_seed{job['seed']}_{job['arm']}_k4.json"
        write_json(directory / 'workers' / name, receipt)
        pins['workers/' + name] = sha256_file(directory / 'workers' / name)
    attempt = {'status': 'success', 'finalized': True, 'scientific_claim': False, 'test_read': False,
               'limitations': protocol['limitations'], 'protocol_sha256': protocol['protocol_sha256'], 'stage': stage,
               'jobs_completed': protocol['jobs'], 'jobs_planned': protocol['jobs'], 'partial': False,
               'budget_limited': False, 'owned_unreaped': False}
    write_json(directory / 'attempt.json', attempt)
    manifest = {'status': 'stage-sealed', 'scientific_claim': False, 'test_read': False,
                'limitations': protocol['limitations'], 'protocol_sha256': protocol['protocol_sha256'],
                'identity': {**protocol['code'], 'data_identity': protocol['data']['data_identity']},
                'files_sha256': pins, 'files_digest': digest(pins)}
    write_json(directory / 'artifact_manifest.json', manifest)
    return {'stage': stage, 'source_kind': 'r7-v2-stage-seal', 'protocol': pin(directory / 'protocol.json'),
            'artifact_manifest': pin(directory / 'artifact_manifest.json'), 'attempt': pin(directory / 'attempt.json'),
            'evaluations': records}
def changed_c_source(spec):
    directory = Path(spec['protocol']['path']).parent
    source = (HERE / 'training/r7_v2_utc_frozen_stats.py').read_text()
    original = REFERENCE_SOURCES['training/r7_v2_results.py']
    import ast
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    text = '# Synthetic changed full file; copied functions remain exact; never execute.\n'
    for name in original['functions']:
        node = nodes[name]
        text += '\n\n' + ''.join(lines[node.lineno - 1:node.end_lineno])
    path = directory / 'static_sources/training/r7_v2_results.py'
    path.parent.mkdir(parents=True)
    path.write_text(text)
    protocol_path = directory / 'protocol.json'
    protocol = read_json(protocol_path)
    old = protocol['protocol_sha256']
    protocol['code']['files']['training/r7_v2_results.py'] = sha256_file(path)
    protocol['code']['source_tree_sha256'] = digest(protocol['code']['files'])
    protocol['protocol_sha256'] = digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})
    replace_json(protocol_path, protocol)
    manifest_path = directory / 'artifact_manifest.json'
    manifest = read_json(manifest_path)
    manifest.update(protocol_sha256=protocol['protocol_sha256'], identity={**protocol['code'], 'data_identity': protocol['data']['data_identity']})
    for receipt_path in sorted((directory / 'workers').glob('*.json')):
        receipt = read_json(receipt_path)
        assert receipt['protocol_sha256'] == old
        receipt.update(protocol_sha256=protocol['protocol_sha256'], source_tree_sha256=protocol['code']['source_tree_sha256'])
        replace_json(receipt_path, receipt)
        manifest['files_sha256'][receipt_path.relative_to(directory).as_posix()] = sha256_file(receipt_path)
    for record in spec['evaluations']:
        record['receipt'] = pin(record['receipt']['path'])
    manifest['files_sha256']['protocol.json'] = sha256_file(protocol_path)
    manifest['files_digest'] = digest(manifest['files_sha256'])
    replace_json(manifest_path, manifest)
    attempt_path = directory / 'attempt.json'
    attempt = read_json(attempt_path)
    attempt['protocol_sha256'] = protocol['protocol_sha256']
    replace_json(attempt_path, attempt)
    spec.update(protocol=pin(protocol_path), artifact_manifest=pin(manifest_path), attempt=pin(attempt_path),
                static_statistics_sources={'training/r7_v2_results.py': pin(path)})
    return spec
def cpu_protocol(label, specs, *, planned=60.0, hard=120.0):
    directory = ROOT / 'protocols' / label
    directory.mkdir(parents=True)
    body = {'format': FORMAT, 'mode': 'synthetic', 'scientific_claim': False, 'test_read': False,
            'limitations': ['SYNTHETIC schema-only counterproof; zero actual B/C/weather reads.'],
            'frozen_before_statistics': True, 'source_files_sha256': source_pins(),
            'timestamp_contract': 'naive_means_UTC; aware_must_be_UTC; exact_6h_slots',
            'grouping': {'axes': ['init_utc_hour', 'valid_utc_hour'], 'hours': list(HOURS)},
            'planned_seconds': planned, 'hard_cap_seconds': hard, 'output': str(ROOT / 'runs' / label), 'stages': specs}
    write_json(directory / 'protocol.json', {**body, 'protocol_sha256': digest(body)})
    return directory / 'protocol.json'
def run(label, specs, *, accepted, planned=60., hard=120.):
    path = cpu_protocol(label, specs, planned=planned, hard=hard)
    cmd = [sys.executable, '-B', str(HERE / 'scripts/stats_r7_v2_utc.py'), '--protocol', str(path),
           '--protocol-file-sha256', sha256_file(path), '--mode', 'synthetic']
    stamp = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=hard + 20,
                            cwd=str(HERE), env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    (logs / (label + '.stdout.log')).write_text(result.stdout, encoding='utf-8')
    (logs / (label + '.stderr.log')).write_text(result.stderr, encoding='utf-8')
    if (result.returncode == 0) != accepted:
        raise AssertionError(f'{label} unexpected returncode={result.returncode}\n{result.stdout}\n{result.stderr}')
    print(json.dumps({'counterproof': label, 'expected_accept': accepted, 'returncode': result.returncode,
                      'seconds': time.perf_counter() - stamp}), flush=True)
    return {'label': label, 'accepted': accepted, 'returncode': result.returncode,
            'stdout': pin(logs / (label + '.stdout.log')), 'stderr': pin(logs / (label + '.stderr.log'))}
def oracle_check(output, specs, *, all_zero=False):
    with (output / 'utc_group_metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = read_json(output / 'group_identities.json')['groups']
    expected_rows = sum((len(jobs_for(spec['stage'])) + len(SEEDS[spec['stage']]) * 5 * 2) * 8 * 51 for spec in specs)
    assert len(rows) == expected_rows and len(groups) == expected_rows // 51
    mapping = {(g['stage'], g['entity'], g['arm'], g['seed'], g['K'], g['baseline_kind'],
                g['lead_hours'], g['group_axis'], g['utc_hour']): g for g in groups}
    nonmean_rmse, nonmean_acc, empty, undefined = False, False, 0, 0
    for row in rows:
        stage, entity, arm = row['stage'], row['entity'], row['arm'] or None
        seed, lead = int(row['seed']), int(row['lead_hours'])
        k = int(row['K']) if row['K'] else None
        kind = 'model' if entity == 'model' else row['baseline_kind']
        group = mapping[(stage, entity, arm, seed, k, row['baseline_kind'] or None, lead,
                         row['group_axis'], int(row['utc_hour']))]
        assert digest(group['cases']) == row['case_set_sha256']
        assert len(group['cases']) == int(row['n_initializations'])
        if entity == 'baseline':
            assert row['K'] == row['arm'] == '' and row['model_depth_applicable'] == 'False'
            assert row['zero_train_updates'] == row['parameters'] == row['trainable_parameters'] == '0'
        v = VARIABLES.index(row['variable'])
        assert row['unit'] == UNITS[v]
        job = {'seed': seed, 'arm': arm or ARMS[stage][0], 'lead': lead, 'reasoning_steps': k or 4}
        n = len(group['cases'])
        if not n:
            assert row['physicalRMSE'] == row['climateMSEskill'] == row['pooledACC'] == ''
            assert row['acc_status'] == 'undefined_empty_hour_group'
            empty += 1
            continue
        raw = [synthetic_row(int(case['sample_id'].rsplit('_', 1)[1]), v, row['region'], job, kind, all_zero)
               for case in group['cases']]
        physical = [{**{key: r[key] for key in STATISTICS[:2]},
                     **{key: r[key] * STDS[v] ** 2 for key in STATISTICS[2:]}} for r in raw]
        means = {key: math.fsum(r[key] for r in physical) / n for key in STATISTICS}
        mse, climate, dot, p, t = (means[key] for key in STATISTICS)
        expected = {'physicalRMSE': math.sqrt(mse), 'climateMSEskill': None if climate == 0 else 1 - mse / climate,
                    'pooledACC': None if p * t == 0 else dot / (math.sqrt(p) * math.sqrt(t))}
        for key, value in {**means, **expected}.items():
            if value is None:
                assert row[key] == ''
                undefined += 1
            else:
                assert math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-12), (key, row[key], value)
        assert json.loads(row['bad_case_counts'])['worse_than_climatology'] == sum(r['mse'] > r['climatology_mse'] for r in physical)
        if n > 1:
            mean_rmse = math.fsum(r['rmse'] for r in raw) / n
            nonmean_rmse |= not math.isclose(expected['physicalRMSE'], mean_rmse, rel_tol=1e-6)
            if all(r['acc'] is not None for r in raw):
                nonmean_acc |= not math.isclose(expected['pooledACC'], math.fsum(r['acc'] for r in raw) / n, rel_tol=1e-6)
    for spec in specs:
        stage_groups = [g for g in groups if g['stage'] == spec['stage']]
        partitions = {}
        for g in stage_groups:
            key = (g['entity'], g['arm'], g['seed'], g['K'], g['baseline_kind'], g['lead_hours'], g['group_axis'])
            partitions.setdefault(key, []).extend(g['cases'])
        for key, cases in partitions.items():
            assert len(cases) == COUNTS[key[5]]
            assert len({c['sample_id'] for c in cases}) == len(cases)
    if not all_zero:
        assert nonmean_rmse and nonmean_acc and empty and undefined
    summary = read_json(output / 'statistics_summary.json')
    if len(specs) == 2:
        ids = summary['actual_frozen_identities']
        assert ids[0]['actual_code']['model_code_sha256'] != ids[1]['actual_code']['model_code_sha256']
    pins = read_json(output / 'pins.json')
    for name, expected in pins['files_sha256'].items():
        assert sha256_file(output / name) == expected
    return {'rows_checked': len(rows), 'groups_checked': len(groups), 'empty_cells': empty,
            'undefined_values_checked': undefined, 'nonmean_RMSE_counterexample': nonmean_rmse,
            'nonmean_ACC_counterexample': nonmean_acc, 'all_zero': all_zero}
def replace_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def mutated(spec, mutate, *, field='per_case_metrics'):
    spec = copy.deepcopy(spec)
    record = spec['evaluations'][0]
    csv_path, receipt_path = Path(record[field]['path']), Path(record['receipt']['path'])
    manifest_path = Path(spec['artifact_manifest']['path'])
    original = {p: p.read_bytes() for p in (csv_path, receipt_path, manifest_path)}
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = read_json(receipt_path)
    receipt['artifact_sha256'][csv_path.name] = sha256_file(csv_path)
    replace_json(receipt_path, receipt)
    record[field], record['receipt'] = pin(csv_path), pin(receipt_path)
    manifest = read_json(manifest_path)
    root = Path(spec['protocol']['path']).parent
    for p in (csv_path, receipt_path):
        manifest['files_sha256'][p.relative_to(root).as_posix()] = sha256_file(p)
    manifest['files_digest'] = digest(manifest['files_sha256'])
    replace_json(manifest_path, manifest)
    spec['artifact_manifest'] = pin(manifest_path)
    return spec, original
def complement_fixture(spec):
    spec = copy.deepcopy(spec)
    root = Path(spec['protocol']['path']).parent
    attempt_path = root / 'attempt.json'
    original = attempt_path.read_bytes()
    failed = read_json(attempt_path)
    failed.update(status='failed', finalized=False)
    replace_json(attempt_path, failed)
    out = ROOT / 'complement_fixture'
    out.mkdir()
    protocol = read_json(root / 'protocol.json')
    sourcepins = {str(root / name): value for name, value in read_json(root / 'artifact_manifest.json')['files_sha256'].items()}
    sourcepins[str(attempt_path)] = sha256_file(attempt_path)
    common = {'source_kind': 'r7-v2-stats-complement', 'status': 'aggregation-complete', 'scientific_claim': False,
              'test_read': False, 'limitations': ['Synthetic complement identity chain only; no actual accepted result.'],
              'training_performed': False, 'evaluation_performed': False, 'gpu_used': False,
              'source_attempt_failed': True, 'source_protocol_sha256': protocol['protocol_sha256'],
              'source_attempt_sha256': sha256_file(attempt_path)}
    inventory = []
    for job in protocol['jobs']:
        name = (f"train_seed{job['seed']}_{job['arm']}_k4.json" if job['phase'] == 'train'
                else f"evaluate_seed{job['seed']}_{job['arm']}_lead{job['lead']:03d}h_k{job['reasoning_steps']}.json")
        inventory.append({'job': job, 'receipt': pin(root / 'workers' / name), 'qualified': True})
    complement_protocol = {'format': 'synthetic-complement-only', 'source_kind': 'r7-v2-stats-complement',
                           'scientific_claim': False, 'test_read': False, 'limitations': common['limitations'],
                           'training_performed': False, 'evaluation_performed': False, 'source_write': False,
                           'frozen_before_reaggregation': True, 'planned_seconds': 600., 'hard_cap_seconds': 1200.,
                           'round_started_perf_counter': 100., 'monotonic_boot_id': 'SYNTHETIC_BOOT_ID',
                           'source_protocol_sha256': protocol['protocol_sha256'],
                           'source_attempt_sha256': sha256_file(attempt_path)}
    complement_protocol['protocol_sha256'] = digest(complement_protocol)
    write_json(out / 'protocol.json', complement_protocol)
    common['complement_protocol_sha256'] = complement_protocol['protocol_sha256']
    write_json(out / 'provenance.json', {**common, 'source_inventory': inventory,
               'source_code': protocol['code'], 'source_failed_cost_reference': str(attempt_path)})
    ownpins = {'protocol.json': sha256_file(out / 'protocol.json'),
               'provenance.json': sha256_file(out / 'provenance.json')}
    write_json(out / 'artifact_manifest.json', {'source_kind': 'r7-v2-stats-complement',
               'scientific_claim': False, 'limitations': common['limitations'],
               'source_protocol_sha256': protocol['protocol_sha256'],
               'complement_protocol_sha256': complement_protocol['protocol_sha256'],
               'files_sha256': ownpins, 'files_digest': digest(ownpins),
               'source_files_sha256': sourcepins, 'source_files_digest': digest(sourcepins)})
    write_json(out / 'attempt.json', {**common, 'finalized': True, 'coverage_complete': True,
               'planned_seconds': 600., 'hard_cap_seconds': 1200., 'started_perf_counter': 100.,
               'ended_perf_counter': 200., 'whole_elapsed_seconds': 100., 'soft_overrun_seconds': 0.,
               'monotonic_boot_id': 'SYNTHETIC_BOOT_ID',
               'provenance_sha256': sha256_file(out / 'provenance.json'),
               'artifact_manifest_sha256': sha256_file(out / 'artifact_manifest.json')})
    spec.update(source_kind='r7-v2-stats-complement', source_attempt=pin(attempt_path),
                attempt=pin(out / 'attempt.json'), artifact_manifest=pin(out / 'artifact_manifest.json'),
                complement_protocol=pin(out / 'protocol.json'), complement_provenance=pin(out / 'provenance.json'))
    return spec, {attempt_path: original}
def complement_gate_counterproofs(spec, results):
    for label, field, patch in (
        ('complement_source_wrongstage', 'source_attempt', {'stage': 'C'}),
        ('complement_source_wrongprotocol', 'source_attempt', {'protocol_sha256': 'e' * 64}),
        ('complement_overhard_rejected', 'attempt', {'ended_perf_counter': 1301., 'whole_elapsed_seconds': 1201.,
                                                   'soft_overrun_seconds': 601.}),
        ('complement_budget_limited_rejected', 'attempt', {'budget_limited': True}),
        ('complement_frozen_wrongprotocol', 'complement_protocol', {'source_protocol_sha256': 'e' * 64}),
    ):
        bad = copy.deepcopy(spec)
        path = Path(bad[field]['path'])
        original = path.read_bytes()
        value = read_json(path)
        value.update(patch)
        if field == 'complement_protocol':
            value['protocol_sha256'] = digest({k: v for k, v in value.items() if k != 'protocol_sha256'})
        replace_json(path, value)
        bad[field] = pin(path)
        try:
            results.append(run(label, [bad], accepted=False))
        finally:
            path.write_bytes(original)
def main():
    ROOT.mkdir()
    specs = [fixture_stage(ROOT / ('fixture_' + stage), stage) for stage in ('B', 'C')]
    specs[1] = changed_c_source(specs[1])
    results = [run('normal_B_C', specs, accepted=True)]
    oracle = oracle_check(ROOT / 'runs/normal_B_C', specs)
    mutations = {
        'duplicate_model': lambda rows: rows.append(dict(rows[0])),
        'incorrect_unit': lambda rows: rows[0].update(unit='normalized'),
        'incorrect_n_initializations': lambda rows: rows[0].update(n_initializations='2'),
        'missing_case_cell': lambda rows: rows.pop(),
        'sample_substitution': lambda rows: rows[0].update(sample_id='SYNTHETIC_UNDECLARED_SAMPLE'),
        'valid_time_alias': lambda rows: rows[0].update(valid_time=rows[1]['init_time']),
        'incorrect_K': lambda rows: rows[0].update(reasoning_steps='1'),
        'unknown_ACC_units': lambda rows: rows[0].update(acc_statistic_units='physical_anomaly_squared'),
        'imputed_zero_energy_ACC': lambda rows: rows[0].update(acc='0'),
        'imputed_zero_energy_skill': lambda rows: rows[0].update(mse_skill='0'),
        'negative_flags_dropped': lambda rows: next(r for r in rows if 'negative_acc' in r['bad_reasons']).update(bad_reasons='[]'),
        'poison_per_case_cancelling': lambda rows: (rows[2].update(acc_dot=str(float(rows[2]['acc_dot']) + .01)),
                                                   rows[4 * 51 + 2].update(acc_dot=str(float(rows[4 * 51 + 2]['acc_dot']) - .01))),
    }
    for label, mutate in mutations.items():
        changed, originals = mutated(specs[0], mutate)
        try:
            results.append(run(label, [changed], accepted=False))
        finally:
            for path, value in originals.items():
                path.write_bytes(value)
    for label, mutate in {
        'duplicate_baseline': lambda rows: rows.append(dict(rows[0])),
        'baseline_pseudo_K': lambda rows: rows[0].update(reasoning_steps='4'),
        'baseline_nonzero_training': lambda rows: rows[0].update(zero_train_updates='1'),
        'baseline_missing_case': lambda rows: rows.pop(),
    }.items():
        changed, originals = mutated(specs[0], mutate, field='baseline_per_case_metrics')
        try:
            results.append(run(label, [changed], accepted=False))
        finally:
            for path, value in originals.items():
                path.write_bytes(value)
    failed_spec = copy.deepcopy(specs[0])
    path = Path(failed_spec['attempt']['path'])
    original = path.read_bytes()
    value = read_json(path)
    value.update(status='failed', finalized=False)
    replace_json(path, value)
    failed_spec['attempt'] = pin(path)
    try:
        results.append(run('failed_source_stage_rejected', [failed_spec], accepted=False))
    finally:
        path.write_bytes(original)
    changed, originals = complement_fixture(specs[0])
    try:
        results.append(run('qualified_complement_chain', [changed], accepted=True))
        complement_oracle = oracle_check(ROOT / 'runs/qualified_complement_chain', [changed])
        complement_gate_counterproofs(changed, results)
        bad = copy.deepcopy(changed)
        attempt_path = Path(bad['attempt']['path'])
        original_attempt = attempt_path.read_bytes()
        value = read_json(attempt_path)
        value['training_performed'] = True
        replace_json(attempt_path, value)
        bad['attempt'] = pin(attempt_path)
        try:
            results.append(run('complement_retraining_rejected', [bad], accepted=False))
        finally:
            attempt_path.write_bytes(original_attempt)
        bad = copy.deepcopy(changed)
        provenance_path = Path(bad['complement_provenance']['path'])
        manifest_path = Path(bad['artifact_manifest']['path'])
        saved = {p: p.read_bytes() for p in (provenance_path, manifest_path, attempt_path)}
        value = read_json(provenance_path)
        value['source_inventory'].pop()
        replace_json(provenance_path, value)
        manifest = read_json(manifest_path)
        manifest['files_sha256']['provenance.json'] = sha256_file(provenance_path)
        manifest['files_digest'] = digest(manifest['files_sha256'])
        replace_json(manifest_path, manifest)
        value = read_json(attempt_path)
        value.update(provenance_sha256=sha256_file(provenance_path), artifact_manifest_sha256=sha256_file(manifest_path))
        replace_json(attempt_path, value)
        bad.update(complement_provenance=pin(provenance_path), artifact_manifest=pin(manifest_path), attempt=pin(attempt_path))
        try:
            results.append(run('complement_missing_receipt_rejected', [bad], accepted=False))
        finally:
            for path, value in saved.items():
                path.write_bytes(value)
    finally:
        for path, value in originals.items():
            path.write_bytes(value)
    zero = fixture_stage(ROOT / 'fixture_all_zero', 'B', all_zero=True)
    results.append(run('all_zero_energy', [zero], accepted=True))
    zero_oracle = oracle_check(ROOT / 'runs/all_zero_energy', [zero], all_zero=True)
    results.append(run('soft_overrun_continues', [specs[0]], accepted=True, planned=.001))
    assert read_json(ROOT / 'runs/soft_overrun_continues/attempt.json')['soft_overrun_seconds'] > 0
    results.append(run('hard_cap_rejected', [specs[0]], accepted=False, planned=.0000001, hard=.0000002))
    assert read_json(ROOT / 'runs/hard_cap_rejected/attempt.json')['budget_limited'] is True
    frozen_path = ROOT / 'protocols/normal_B_C/protocol.json'
    old_pins = sha256_file(ROOT / 'runs/normal_B_C/pins.json')
    result = subprocess.run([sys.executable, '-B', str(HERE / 'scripts/stats_r7_v2_utc.py'), '--protocol', str(frozen_path),
                             '--protocol-file-sha256', sha256_file(frozen_path), '--mode', 'synthetic'],
                            capture_output=True, text=True, cwd=str(HERE))
    assert result.returncode != 0 and sha256_file(ROOT / 'runs/normal_B_C/pins.json') == old_pins
    (ROOT / 'logs' / 'exclusive_output.stderr.log').write_text(result.stderr)
    results.append({'label': 'exclusive_output_rejected', 'accepted': False, 'returncode': result.returncode})
    report = {'status': 'synthetic-counterproofs-verified', 'scientific_claim': False, 'actual_pass': False,
              'actual_run': 'NOT RUN; wait for independently frozen actual CPU protocol and accepted source chain',
              'limitations': ['Invented scalar schema fixtures only, not weather/training/evaluation evidence.', *LIMITATIONS],
              'source_files_sha256': source_pins(), 'results': results, 'normal_oracle': oracle,
              'zero_oracle': zero_oracle, 'complement_oracle': complement_oracle,
              'forbidden_repo_reads': ['actual outputs', 'weather/forecast fields', 'test data/tests', 'configs'],
              'repo_writes': False, 'network': False, 'gpu': False}
    write_json(ROOT / 'counterproof_report.json', report)
    print(json.dumps({'status': report['status'], 'checks': len(results), 'normal_oracle': oracle,
                      'actual_pass': False, 'report': str(ROOT / 'counterproof_report.json')}), flush=True)
def test_synthetic_counterproof_suite(tmp_path):
    global ROOT
    ROOT = tmp_path / 'utc_synthetic'
    main()
    report = read_json(ROOT / 'counterproof_report.json')
    probes = ["import socket; socket.create_connection(('127.0.0.1', 1))", "import socket; socket.socket().connect(('127.0.0.1', 1))"]
    for probe in probes:
        code = 'from scripts.stats_r7_v2_utc import deny_network; deny_network(); ' + probe
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                                cwd=str(HERE), env={**os.environ, 'PYTHONPATH': str(HERE), 'CUDA_VISIBLE_DEVICES': ''})
        assert result.returncode != 0 and 'network prohibited' in result.stderr
    code = ('import sys; from training import r7_v2_utc_contract, r7_v2_utc_statistics; '
            "assert not ({'torch','model','data'} & set(sys.modules))")
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                            cwd=str(HERE), env={**os.environ, 'PYTHONPATH': str(HERE), 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    assert report['status'] == 'synthetic-counterproofs-verified'
    assert report['actual_pass'] is False
    assert len(report['results']) == 30
    assert report['normal_oracle']['rows_checked'] == 87720
    assert report['normal_oracle']['groups_checked'] == 1720
    assert report['normal_oracle']['nonmean_RMSE_counterexample'] is True
    assert report['normal_oracle']['nonmean_ACC_counterexample'] is True
def test_early_claim_failures_sealed(tmp_path, monkeypatch):
    import pytest
    from scripts import stats_r7_v2_utc as cli
    cases = ('protocol_sha', 'helper_sha', 'wrong_boot', 'publication_marker', 'helper_zip', 'source_identity')
    for kind in cases:
        protocol_path = tmp_path / kind / 'freeze/protocol.json'
        protocol_path.parent.mkdir(parents=True)
        stage_root = tmp_path / kind / 'source'
        stage_root.mkdir()
        body = {'output': str(tmp_path / kind / 'output'), 'format': FORMAT, 'mode': 'synthetic', 'scientific_claim': False,
                'test_read': False, 'limitations': ['Synthetic early-claim counterproof; not actual evidence.'],
                'source_files_sha256': source_pins(), 'frozen_before_statistics': True,
                'timestamp_contract': 'naive_means_UTC; aware_must_be_UTC; exact_6h_slots',
                'grouping': {'axes': ['init_utc_hour', 'valid_utc_hour'], 'hours': list(HOURS)},
                'planned_seconds': 60., 'hard_cap_seconds': 120.,
                'stages': [{'stage': 'B', 'source_kind': 'r7-v2-stage-seal', 'protocol': {'path': str(stage_root / 'protocol.json'), 'sha256': 'e' * 64}}]}
        if kind == 'wrong_boot':
            body['monotonic_boot_id'] = 'SYNTHETIC_WRONG_BOOT'
        body['protocol_sha256'] = digest(body)
        write_json(protocol_path, body)
        expected = sha256_file(protocol_path)
        helper_path = protocol_path.parent / 'owned_helper.py'
        helper_path.write_text('SYNTHETIC_ORIGINAL\n')
        if kind == 'protocol_sha':
            protocol_path.write_bytes(protocol_path.read_bytes() + b'\n')
        if kind == 'helper_sha':
            helper_path.write_text('SYNTHETIC_TAMPER\n')
        if kind == 'publication_marker':
            write_json(stage_root / 'publication_failure.json', {'scientific_claim': False, 'status': 'failed'})
        def fake_inputs(spec, pins, mode):
            protocol = {'stage': 'B', 'protocol_sha256': 'e' * 64,
                        'code': {k: 'e' * 64 for k in ('base_commit', 'model_code_sha256', 'source_tree_sha256', 'code_zip_sha256')},
                        'data': {'data_identity': 'e' * 64}, 'sources': {'source_sha256': 'e' * 64}}
            return [(protocol, {})]
        with monkeypatch.context() as patch:
            patch.setattr(cli, 'deny_network', lambda: None)
            if kind == 'helper_sha':
                actual_sha = cli.file_sha256
                patch.setattr(cli, 'file_sha256', lambda path:'e'*64 if Path(path)==HERE / 'training/r7_v2_utc_statistics.py' and helper_path.read_text() == 'SYNTHETIC_TAMPER\n' else actual_sha(path))
            if kind in ('helper_zip', 'source_identity'):
                import training.r7_v2_utc_contract as contract
                import training.r7_v2_utc_statistics as statistics
                patch.setattr(contract, 'stage_inputs', fake_inputs)
                def poison(*args, **kwargs):
                    target = Path(body['output']) / ('helper_code.zip' if kind == 'helper_zip' else 'source_identity.json')
                    target.write_bytes(b'SYNTHETIC_POISON')
                    return [], []
                patch.setattr(statistics, 'process_stage', poison)
                patch.setattr(contract, 'write_csv', lambda path, rows: Path(path).write_text('synthetic\n'))
            argv = ['--protocol', str(protocol_path), '--protocol-file-sha256', expected, '--mode', 'synthetic']
            assert cli.main(argv) == 1
        output = Path(body['output'])
        attempt, pins = read_json(output / 'attempt.json'), read_json(output / 'pins.json')
        assert attempt['status'] == pins['status'] == 'failed' and attempt['finalized'] is False
        assert attempt['scientific_claim'] is False and attempt['actual_pass'] is False and attempt['failure_reason']
        reason = {'protocol_sha': 'file byte pin', 'helper_sha': 'source SHA256', 'wrong_boot': 'same-boot',
                  'publication_marker': 'publication_failure', 'helper_zip': 'helper ZIP/source identity',
                  'source_identity': 'JSONDecodeError'}[kind]
        assert reason in attempt['failure_reason']
        before = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        if kind == 'protocol_sha':
            protocol_path.write_bytes(protocol_path.read_bytes()[:-1])
        helper_path.write_text('SYNTHETIC_ORIGINAL\n')
        argv = ['--protocol', str(protocol_path), '--protocol-file-sha256', sha256_file(protocol_path), '--mode', 'synthetic']
        with pytest.raises(FileExistsError):
            cli.main(argv)
        assert before == {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        print(json.dumps({'early_claim_case': kind, 'sealed_failed': True, 'repeat_refused': True}), flush=True)
    for field in ('protocol', 'attempt'):
        marker_root = tmp_path / ('marker_' + field)
        marker_root.mkdir()
        write_json(marker_root / 'publication_failure.json', {'scientific_claim': False, 'status': 'failed'})
        with pytest.raises(ValueError, match='publication_failure'):
            cli.reject_publication_failures({'stages': [{field: {'path': str(marker_root / 'protocol.json')}}]})
