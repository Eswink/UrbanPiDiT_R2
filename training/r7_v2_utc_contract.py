"""Fail-closed, stdlib-only pins/metadata; reads explicit artifacts, never models or datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

DEADLINE = None


def wall_check():
    if DEADLINE is not None and time.perf_counter() >= DEADLINE:
        raise TimeoutError('CPU statistics hard whole-round deadline exceeded')


from .r7_v2_utc_frozen_stats import integer, number

REPO = Path(__file__).resolve().parents[1]
HERE = REPO
SOURCE_NAMES = ('training/r7_v2_utc_frozen_stats.py', 'training/r7_v2_utc_contract.py',
                'training/r7_v2_utc_statistics.py', 'scripts/stats_r7_v2_utc.py')
REFERENCE_SOURCES = {'training/r7_v2_evaluation.py': {'sha256': '303667e1f305b66d67a6e59f6526d48aae8bba73dad685682a3d72a652953c18'},
 'training/r7_v2_protocol.py': {'sha256': 'b79041c25baf4af8dfe0d0ece4ebb9380d0e5fb591b50f2640f17a088e393fa1'},
 'training/r7_v2_results.py': {'functions': {'_canonical_metric': {'first_line': 33,
                                                                   'last_line': 58,
                                                                   'source_sha256': '4c1fd92cf16cb59ed11217cddb2d3a6c3869546e9fbe1e13f96c79988a7a9b03'},
                                             '_list': {'first_line': 26,
                                                       'last_line': 30,
                                                       'source_sha256': 'f552f5cfff345cd0a3846b51a8645676a0698ad243682372b180b97b8d97a71f'}},
                               'sha256': '0e22a1d3506e762ef3a7a0b38d1e512715b53980f2bf72e759ee10edbec50de3'},
 'training/r7_v2_tables.py': {'functions': {'case_key': {'first_line': 50,
                                                         'last_line': 59,
                                                         'source_sha256': '8a998bcb558feba9c05f1c355b02cd7a4f0e458bc5ddef061bdce0b5ea76551b'},
                                            'close': {'first_line': 45,
                                                      'last_line': 47,
                                                      'source_sha256': '27bca5a2870d078fad229d506d9683fa366a111b5fe67792fb17e16b9d3406a6'},
                                            'integer': {'first_line': 38,
                                                        'last_line': 42,
                                                        'source_sha256': 'e0a0f93efaa357d6eabb10b28471d5d4a24a55405a71dfe2135ebd2c2823aa96'},
                                            'number': {'first_line': 26,
                                                       'last_line': 35,
                                                       'source_sha256': '2a5d0f4a4c2f855fc34fa1729614cae9087a75f0c811b4cd7f35d1ede93fdd87'},
                                            'pooled_statistics': {'first_line': 62,
                                                                  'last_line': 95,
                                                                  'source_sha256': '3c05ae300c2baad04561b637ca713e702a5f763274779e7ee32ecde52421fa9f'}},
                              'sha256': '8e039374596cbb2781ce83de814abd018176339b859867da5305c287f9044f8f'}}
LEADS = (6, 12, 24, 48, 72)
REGIONS = ('full', 'interior', 'edge_2')
HOURS = (0, 6, 12, 18)
ARMS = {'B': ('continue_l6', 'rollout_l6_l12', 'equal_compute_l6'),
        'C': ('old_ours', 'process', 'matched_generic')}
SEEDS = {'B': (41, 42), 'C': (41, 42, 43)}
KERNELS = {'B': (4,), 'C': (1, 2, 4)}
FORMAT = 'r7-utc-cpu-statistics-protocol-v1'
LIMITATIONS = [
    'Descriptive CPU statistics only; no scientific criteria, selection, significance or actual PASS.',
    'No forward, GPU, optimizer, model/checkpoint loading, weather arrays, test observations or network.',
    'Each variable/unit/lead/region/arm/seed/K stays separate; no mean RMSE/ACC or case intersection.',
    'Undefined and negative values remain in complete cohorts; empty hour buckets are explicit nulls.',
    'Baseline references are zero-training/no model K; repeated arm/K exports are verified then deduplicated.',
    'Naive source timestamps mean UTC only under the explicit frozen timestamp contract.',
    'Frozen five-statistic algebra is dependency-pruned, exact reviewed active code; no archive imports.',
    'Identity-bound numerical replay; no cross-platform bitwise reproducibility claim.',
    'Final receipts use a pre-serialization clock snapshot; terminal stdout includes completed publication.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def local(path):
    if not isinstance(path, (str, Path)) or '://' in str(path):
        raise ValueError('explicit local absolute path required')
    result = Path(path)
    if not result.is_absolute() or '..' in result.parts:
        raise ValueError('absolute paths without traversal required')
    if any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError('symlink paths/ancestors forbidden')
    if any('legacy' in p.lower() or p == 'tests' or p.startswith('test.')
           or p == 'test.jsonl' for p in result.parts):
        raise ValueError('legacy/test paths forbidden')
    if any(result.parts[i:i + 2] in (('data', 'raw'), ('data', 'interim'), ('data', 'processed'))
           for i in range(len(result.parts) - 1)):
        raise ValueError('weather/data arrays forbidden')
    return result


def output_path(path):
    result = local(path)
    if not result.is_relative_to(Path('/tmp')) or result == Path('/tmp') or result.is_relative_to(REPO):
        raise ValueError('writes require a new exclusive /tmp output; never repository files')
    return result


def sha256_file(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            wall_check()
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    return json.loads(local(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def write_json(path, value):
    with output_path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty required table')
    with output_path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False)
                             if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def pin(path):
    return {'path': str(local(path)), 'sha256': sha256_file(path)}


def check_pin(value, pins, *, name):
    if set(value) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise ValueError('explicit file SHA256 pin required')
    path = local(value['path'])
    if path.name != name or sha256_file(path) != value['sha256']:
        raise ValueError(f'input pin/name mismatch: {path}')
    previous = pins.setdefault(str(path), value['sha256'])
    if previous != value['sha256']:
        raise ValueError('conflicting input pins')
    return path


def verified_json(value, pins, *, name):
    return read_json(check_pin(value, pins, name=name))


def source_pins():
    return {name: sha256_file(HERE / name) for name in SOURCE_NAMES}


def job_key(job):
    return f"evaluate_seed{job['seed']}_{job['arm']}_lead{job['lead']:03d}h_k{job['reasoning_steps']}"


def jobs_for(stage):
    return [{'phase': 'evaluate', 'seed': seed, 'arm': arm, 'lead': lead, 'reasoning_steps': k}
            for seed in SEEDS[stage] for arm in ARMS[stage] for lead in LEADS for k in KERNELS[stage]]


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_cpu(cpu, mode):
    if (cpu.get('format') != FORMAT or cpu.get('mode') != mode or cpu.get('scientific_claim') is not False
            or cpu.get('test_read') is not False or not cpu.get('limitations')
            or cpu.get('frozen_before_statistics') is not True
            or cpu.get('timestamp_contract') != 'naive_means_UTC; aware_must_be_UTC; exact_6h_slots'
            or cpu.get('grouping') != {'axes': ['init_utc_hour', 'valid_utc_hour'], 'hours': list(HOURS)}
            or cpu.get('source_files_sha256') != source_pins()
            or cpu.get('protocol_sha256') != digest({k: v for k, v in cpu.items() if k != 'protocol_sha256'})):
        raise ValueError('independent frozen CPU protocol/mode/source/clock grouping mismatch')
    if mode == 'actual' and ('round_started_perf_counter' not in cpu or 'monotonic_boot_id' not in cpu):
        raise ValueError('actual CPU whole clock must include the independently frozen prepare entry')
    planned, hard = number(cpu['planned_seconds']), number(cpu['hard_cap_seconds'])
    if not 0 < planned < hard:
        raise ValueError('positive soft budget and larger hard whole-round cap required')
    output_path(cpu['output'])
    stages = [spec['stage'] for spec in cpu['stages']]
    if not stages or len(set(stages)) != len(stages) or any(stage not in ARMS for stage in stages):
        raise ValueError('explicit unique B/C stage list required')
    return cpu


def validate_actual_protocol(protocol, stage, mode):
    if (protocol.get('protocol_sha256') != digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})
            or protocol.get('stage') != stage or protocol.get('scientific_claim') is not False
            or protocol.get('test_read') is not False or not protocol.get('limitations')):
        raise ValueError('actual frozen protocol identity/flags mismatch')
    if mode == 'actual' and protocol.get('format') != 'r7-v2-remaining-protocol-v1':
        raise ValueError('actual requires existing B/C protocol, never a synthetic substitute')
    if mode == 'synthetic' and protocol.get('format') != 'synthetic-r7-v2-schema-only':
        raise ValueError('synthetic mode requires synthetic-labelled input')
    expected = jobs_for(stage)
    actual = [job for job in protocol['jobs'] if job['phase'] == 'evaluate']
    if actual != expected or set(protocol['arm_configs']) != set(ARMS[stage]):
        raise ValueError('complete exact B/C seed/arm/lead/K inventory required')
    data = protocol['data']
    if (len(data['channels']) != 17 or len(set(data['channels'])) != 17 or len(data['units']) != 17
            or len(data['normalization_std']) != 17 or data.get('test_read') is not False
            or set(data['evaluation_cases']) != {str(lead) for lead in LEADS}):
        raise ValueError('complete pinned 17-variable five-lead data metadata required')
    for unit, std in zip(data['units'], data['normalization_std']):
        if not isinstance(unit, str) or not unit.strip() or unit.strip().lower() in ('normalized', 'unknown') or number(std) <= 0:
            raise ValueError('explicit physical units and positive training std required')
    code = protocol['code']
    if (not valid_digest(code['model_code_sha256']) or not valid_digest(code['source_tree_sha256'])
            or digest(code['files']) != code['source_tree_sha256']):
        raise ValueError('actual static model/source identity malformed; current model is never guessed')
    return protocol


def verify_statistics_source(protocol, spec, pins):
    """New C model/source hashes are allowed; canonical/pool functions must be statically identical."""
    import ast
    references = REFERENCE_SOURCES
    supplied = spec.get('static_statistics_sources', {})
    for name in ('training/r7_v2_tables.py', 'training/r7_v2_results.py'):
        if protocol['code']['files'].get(name) == references[name]['sha256']:
            continue
        if name not in supplied:
            raise ValueError('changed statistics file needs explicit frozen source bytes; never guess current source')
        path = check_pin(supplied[name], pins, name=Path(name).name)
        if supplied[name]['sha256'] != protocol['code']['files'].get(name):
            raise ValueError('supplied static statistics source is not the actual frozen file')
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines(keepends=True)
        nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        for function, expected in references[name]['functions'].items():
            node = nodes.get(function)
            if node is None:
                raise ValueError('frozen statistic function is missing')
            exact = ''.join(lines[node.lineno - 1:node.end_lineno])
            if hashlib.sha256(exact.encode()).hexdigest() != expected['source_sha256']:
                raise ValueError('actual canonical/pool function changed; independent review required')
        if name.endswith('tables.py'):
            assignments = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'STATISTICS' for t in n.targets)]
            if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != (
                    'mse', 'climatology_mse', 'acc_dot', 'acc_forecast_energy', 'acc_target_energy'):
                raise ValueError('actual sufficient-statistic scope differs')


def verify_complement_clock(accepted, frozen):
    planned, hard = number(frozen['planned_seconds']), number(frozen['hard_cap_seconds'])
    started, ended = number(accepted['started_perf_counter']), number(accepted['ended_perf_counter'])
    elapsed = number(accepted['whole_elapsed_seconds'])
    if (planned != 600 or hard != 1200 or not 0 <= started <= ended < started + hard
            or accepted['planned_seconds'] != planned or accepted['hard_cap_seconds'] != hard
            or started != number(frozen['round_started_perf_counter'])
            or accepted.get('monotonic_boot_id') != frozen.get('monotonic_boot_id')
            or not isinstance(frozen.get('monotonic_boot_id'), str) or not frozen['monotonic_boot_id']
            or accepted.get('budget_limited', False) is not False or accepted.get('partial', False) is not False
            or accepted.get('owned_unreaped', False) is not False or accepted.get('failure_reason') is not None):
        raise ValueError('complement whole-clock hard-limited/failed/unsealed attempt cannot be accepted')
    if (not math.isclose(elapsed, ended - started, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(number(accepted['soft_overrun_seconds']), max(0., elapsed - planned),
                                rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError('complement continuous whole cost/soft-overrun differs from frozen clock')


def verify_complement(spec, protocol, pins):
    """Accept an independent qualified no-forward complement, never upgrade the failed source attempt."""
    root = local(protocol['output'])
    source_attempt = verified_json(spec['source_attempt'], pins, name='attempt.json')
    if (local(spec['source_attempt']['path']) != root / 'attempt.json'
            or source_attempt.get('status') != 'failed' or source_attempt.get('finalized') is not False
            or source_attempt.get('scientific_claim') is not False or source_attempt.get('test_read') is not False
            or not source_attempt.get('limitations') or source_attempt.get('protocol_sha256') != protocol['protocol_sha256']
            or source_attempt.get('stage') != protocol['stage']
            or source_attempt.get('jobs_completed') != protocol['jobs']
            or source_attempt.get('jobs_planned') != protocol['jobs']
            or source_attempt.get('partial') is not False or source_attempt.get('budget_limited') is not False
            or source_attempt.get('owned_unreaped') is not False):
        raise ValueError('stats complement requires the explicit unchanged failed full source attempt')
    manifest = verified_json(spec['artifact_manifest'], pins, name='artifact_manifest.json')
    accepted = verified_json(spec['attempt'], pins, name='attempt.json')
    provenance = verified_json(spec['complement_provenance'], pins, name='provenance.json')
    complement_root = local(spec['attempt']['path']).parent
    if (complement_root == root or complement_root.is_relative_to(root) or root.is_relative_to(complement_root)
            or local(spec['artifact_manifest']['path']).parent != complement_root
            or local(spec['complement_provenance']['path']).parent != complement_root):
        raise ValueError('complement must have its own independent pinned output root')
    complement_protocol = verified_json(spec['complement_protocol'], pins, name='protocol.json')
    if (local(spec['complement_protocol']['path']).parent != complement_root
            or complement_protocol.get('protocol_sha256') != digest({k: v for k, v in complement_protocol.items() if k != 'protocol_sha256'})
            or manifest.get('source_kind') != 'r7-v2-stats-complement' or manifest.get('scientific_claim') is not False
            or not manifest.get('limitations') or manifest.get('source_protocol_sha256') != protocol['protocol_sha256']
            or manifest.get('complement_protocol_sha256') != complement_protocol['protocol_sha256']
            or manifest['files_sha256'].get('protocol.json') != spec['complement_protocol']['sha256']):
        raise ValueError('complement independently frozen protocol/manifest identity mismatch')
    if (complement_protocol.get('source_kind') != 'r7-v2-stats-complement'
            or complement_protocol.get('scientific_claim') is not False or complement_protocol.get('test_read') is not False
            or not complement_protocol.get('limitations') or complement_protocol.get('training_performed') is not False
            or complement_protocol.get('evaluation_performed') is not False or complement_protocol.get('source_write') is not False
            or complement_protocol.get('frozen_before_reaggregation') is not True
            or complement_protocol.get('source_protocol_sha256') != protocol['protocol_sha256']
            or complement_protocol.get('source_attempt_sha256') != spec['source_attempt']['sha256']):
        raise ValueError('complement protocol must be independently frozen stats-only for this failed source')
    verify_complement_clock(accepted, complement_protocol)
    for value in (accepted, provenance):
        if (value.get('source_kind') != 'r7-v2-stats-complement' or value.get('status') != 'aggregation-complete'
                or value.get('scientific_claim') is not False or value.get('test_read') is not False
                or not value.get('limitations') or value.get('training_performed') is not False
                or value.get('evaluation_performed') is not False or value.get('gpu_used') is not False
                or value.get('source_attempt_failed') is not True
                or value.get('source_protocol_sha256') != protocol['protocol_sha256']
                or value.get('complement_protocol_sha256') != complement_protocol['protocol_sha256']
                or value.get('source_attempt_sha256') != spec['source_attempt']['sha256']):
            raise ValueError('independently accepted explicit no-forward stats complement required')
    if (accepted.get('finalized') is not True or accepted.get('coverage_complete') is not True
            or accepted.get('provenance_sha256') != spec['complement_provenance']['sha256']
            or accepted.get('artifact_manifest_sha256') != spec['artifact_manifest']['sha256']):
        raise ValueError('complement final accepted inventory/provenance/manifest seal required')
    if (manifest.get('files_digest') != digest(manifest['files_sha256'])
            or manifest.get('source_files_digest') != digest(manifest['source_files_sha256'])
            or manifest['files_sha256'].get('provenance.json') != spec['complement_provenance']['sha256']):
        raise ValueError('complement provenance/manifest pins mismatch')
    if (provenance.get('source_code') != protocol['code']
            or provenance.get('source_failed_cost_reference') != str(root / 'attempt.json')):
        raise ValueError('complement must preserve original source/code and failed cost reference')
    inventory = provenance['source_inventory']
    if [entry['job'] for entry in inventory] != protocol['jobs']:
        raise ValueError('all source train/evaluate receipts must be individually qualified by complement')
    for entry in inventory:
        job = entry['job']
        name = (f"train_seed{job['seed']}_{job['arm']}_k4.json" if job['phase'] == 'train'
                else job_key(job) + '.json')
        receipt = verified_json(entry['receipt'], pins, name=name)
        if (entry.get('qualified') is not True or local(entry['receipt']['path']) != root / 'workers' / name
                or receipt.get('status') != 'success' or receipt.get('job') != job
                or receipt.get('protocol_sha256') != protocol['protocol_sha256']
                or receipt.get('scientific_claim') is not False or receipt.get('test_read') is not False
                or not receipt.get('limitations')
                or receipt.get('model_code_sha256') != protocol['code']['model_code_sha256']
                or receipt.get('source_tree_sha256') != protocol['code']['source_tree_sha256']
                or receipt.get('data_identity') != protocol['data']['data_identity']):
            raise ValueError('unqualified/incomplete/changed source receipt inventory')
    for key in ('protocol', 'source_attempt'):
        if manifest['source_files_sha256'].get(spec[key]['path']) != spec[key]['sha256']:
            raise ValueError('complement source protocol/failed-attempt pin mismatch')
    for entry in inventory:
        if manifest['source_files_sha256'].get(entry['receipt']['path']) != entry['receipt']['sha256']:
            raise ValueError('complement qualified receipt pin mismatch')
    return {'source_kind': 'r7-v2-stats-complement', 'files_sha256': {
        str(Path(path).relative_to(root)): value for path, value in manifest['source_files_sha256'].items()
        if local(path).is_relative_to(root)}}


def verify_stage_seal(spec, protocol, pins):
    stage, root = protocol['stage'], local(protocol['output'])
    manifest = verified_json(spec['artifact_manifest'], pins, name='artifact_manifest.json')
    attempt = verified_json(spec['attempt'], pins, name='attempt.json')
    if (local(spec['artifact_manifest']['path']) != root / 'artifact_manifest.json'
            or local(spec['attempt']['path']) != root / 'attempt.json'):
        raise ValueError('actual completion receipts must belong to the same frozen root')
    if (manifest.get('status') != 'stage-sealed' or manifest.get('scientific_claim') is not False
            or manifest.get('test_read') is not False or not manifest.get('limitations')
            or manifest.get('protocol_sha256') != protocol['protocol_sha256']
            or manifest.get('files_digest') != digest(manifest['files_sha256'])):
        raise ValueError('complete frozen actual manifest required')
    if (attempt.get('status') not in ('success', 'paused') or attempt.get('finalized') is not True
            or attempt.get('scientific_claim') is not False or attempt.get('test_read') is not False
            or not attempt.get('limitations') or attempt.get('partial') is not False
            or attempt.get('budget_limited') is not False or attempt.get('owned_unreaped') is not False
            or attempt.get('protocol_sha256') != protocol['protocol_sha256']
            or attempt.get('stage') != stage or attempt.get('jobs_completed') != protocol['jobs']
            or attempt.get('jobs_planned') != protocol['jobs']):
        raise ValueError('final full actual completion required; pending/partial/cancelled never accepted')
    for field in ('model_code_sha256', 'source_tree_sha256', 'code_zip_sha256', 'base_commit'):
        if manifest['identity'][field] != protocol['code'][field]:
            raise ValueError('manifest actual code/model identity mismatch')
    if manifest['identity']['data_identity'] != protocol['data']['data_identity']:
        raise ValueError('manifest actual data identity mismatch')
    for value in (spec['protocol'],):
        if manifest['files_sha256'].get(Path(value['path']).relative_to(root).as_posix()) != value['sha256']:
            raise ValueError('actual manifest/protocol byte pin mismatch')
    return manifest


def stage_inputs(spec, pins, mode):
    stage = spec['stage']
    protocol = validate_actual_protocol(verified_json(spec['protocol'], pins, name='protocol.json'), stage, mode)
    root = local(protocol['output'])
    if local(spec['protocol']['path']) != root / 'protocol.json':
        raise ValueError('actual protocol root mismatch')
    verify_statistics_source(protocol, spec, pins)
    if spec.get('source_kind') == 'r7-v2-stage-seal':
        manifest = verify_stage_seal(spec, protocol, pins)
    elif spec.get('source_kind') == 'r7-v2-stats-complement':
        manifest = verify_complement(spec, protocol, pins)
    else:
        raise ValueError('explicit recognized sourcekind required; failed source stage is not success')
    records = spec['evaluations']
    if [entry['job'] for entry in records] != jobs_for(stage):
        raise ValueError('all frozen actual evaluation CSV jobs required, no intersection')
    for record in records:
        job = record['job']
        directory = root / f"seed{job['seed']}/evaluation/{job['arm']}/lead_{job['lead']:03d}h/k{job['reasoning_steps']}"
        receipt_name = job_key(job) + '.json'
        receipt = verified_json(record['receipt'], pins, name=receipt_name)
        if local(record['receipt']['path']) != root / 'workers' / receipt_name:
            raise ValueError('receipt exact job/root mismatch')
        for field, section in (('model_code_sha256', 'code'), ('source_tree_sha256', 'code'),
                               ('code_zip_sha256', 'code'), ('source_sha256', 'sources'), ('data_identity', 'data')):
            if receipt.get(field) != protocol[section][field]:
                raise ValueError('receipt actual source/model/data identity mismatch')
        if (receipt.get('status') != 'success' or receipt.get('job') != job
                or receipt.get('protocol_sha256') != protocol['protocol_sha256']
                or receipt.get('scientific_claim') is not False or receipt.get('test_read') is not False
                or not receipt.get('limitations')):
            raise ValueError('successful exact actual job receipt required')
        for field, name in (('per_case_metrics', 'per_case_metrics.csv'),
                            ('baseline_per_case_metrics', 'baseline_per_case_metrics.csv'), ('provenance', 'provenance.json')):
            path = check_pin(record[field], pins, name=name)
            if (path != directory / name or receipt['artifact_sha256'].get(name) != record[field]['sha256']
                    or local(receipt[field + '_csv' if name.endswith('.csv') else field]) != path):
                raise ValueError('receipt actual CSV/provenance byte identity mismatch')
        for field in ('receipt', 'per_case_metrics', 'baseline_per_case_metrics', 'provenance'):
            relative = Path(record[field]['path']).relative_to(root).as_posix()
            if manifest['files_sha256'].get(relative) != record[field]['sha256']:
                raise ValueError('actual manifest artifact pin mismatch')
        record = {**record, '_receipt': receipt, '_provenance': read_json(record['provenance']['path'])}
        if record['_provenance'] != receipt['evaluation_provenance']:
            raise ValueError('actual frozen embedded provenance mismatch')
        yield protocol, record
