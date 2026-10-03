"""Two separate fixed UTC hour partitions of complete per-case physical sufficient statistics."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .r7_v2_utc_contract import HOURS, LEADS, REGIONS, digest, local, read_json, write_csv
from .r7_v2_utc_frozen_stats import STATISTICS, _canonical_metric, case_key, close, integer, pooled_statistics


def utc_time(value):
    if not isinstance(value, str):
        raise ValueError('exact timestamp string required')
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() not in (None, timedelta(0)):
        raise ValueError('only UTC timestamps accepted; no local timezone guessing')
    if stamp.minute or stamp.second or stamp.microsecond or stamp.hour not in HOURS:
        raise ValueError('timestamps must occupy exact 0/6/12/18 UTC slots')
    return stamp.replace(tzinfo=timezone.utc)


def identity(item, lead):
    sample = item['sample_id']
    if not isinstance(sample, str) or not sample.strip():
        raise ValueError('explicit nonempty sample identity required')
    valid = item['valid_times']
    case_key(item['init_time'], valid, lead)
    init_stamp, valid_stamp = utc_time(item['init_time']), utc_time(valid[0])
    if valid_stamp - init_stamp != timedelta(hours=lead):
        raise ValueError('UTC valid/init lead mismatch')
    if item.get('valid_time', valid[0]) != valid[0]:
        raise ValueError('valid_time alias mismatch')
    return {'sample_id': sample, 'init_time': item['init_time'], 'valid_time': valid[0],
            'valid_times': valid, 'init_utc_hour': init_stamp.hour, 'valid_utc_hour': valid_stamp.hour}


def verify_provenance(protocol, record, mode):
    job, p, receipt = record['job'], record['_provenance'], record['_receipt']
    data, lead, k = protocol['data'], job['lead'], job['reasoning_steps']
    declared = data['evaluation_cases'][str(lead)]
    expected = declared['cases']
    if (p.get('scientific_claim') is not False or p.get('test_read') is not False or not p.get('limitations')
            or p.get('format') != 'r7-v2-evaluation-v1' or p.get('split') != 'val'
            or p.get('channels') != data['channels'] or p.get('units') != data['units']
            or p.get('lead_hours') != [lead] or p.get('step_hours') != 6 or p.get('reasoning_steps') != k
            or p.get('checkpoint_training_steps') != 4 or p.get('independently_trained_k1') is not False
            or p.get('inference_options') != {'reasoning_steps': k}
            or p.get('model_code_sha256') != protocol['code']['model_code_sha256']
            or p.get('training_identity') != data['data_identity']
            or p.get('evaluation_manifest_sha256') != protocol['sources']['val_manifest_sha256']
            or p.get('n_evaluated') != len(expected) or p.get('n_available_windows') != declared['n_available']
            or p.get('n_evaluated') != receipt.get('n_evaluated')
            or p.get('n_available_windows') != receipt.get('n_available_windows')
            or p.get('checkpoint_sha256') != receipt.get('checkpoint_sha256')):
        raise ValueError('frozen CSV provenance case/K/model/data identity mismatch')
    if mode == 'actual' and len(expected) != {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}[lead]:
        raise ValueError('actual complete per-lead 22/21/19/15/11 cohorts required')
    keys = [case_key(init, valid, lead) for init, valid in expected]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('nonempty unique frozen per-lead cohort required')
    inits = p['initializations']
    if sorted([[item['init_time'], item['valid_times']] for item in inits]) != sorted(expected):
        raise ValueError('provenance complete cohort mismatch; no count-only acceptance')
    cohort = {}
    sample_ids = set()
    cells = {(region, variable) for region in REGIONS for variable in data['channels']}
    embedded = {'model': {}, 'persistence': {}, 'climatology': {}}
    for item in inits:
        ident = identity(item, lead)
        if ident['init_time'] in cohort or ident['sample_id'] in sample_ids:
            raise ValueError('duplicate initialization/sample identities')
        cohort[ident['init_time']] = ident
        sample_ids.add(ident['sample_id'])
        if item.get('cumulative_reasoning_steps') != [k * lead // 6]:
            raise ValueError('recorded cumulative K identity mismatch')
        if len(item['region_metrics']) != 51:
            raise ValueError('complete embedded 17-variable three-region model cells required')
        for row in item['region_metrics']:
            key = (ident['init_time'], row['region'], row['variable'])
            if (row['region'], row['variable']) not in cells or key in embedded['model']:
                raise ValueError('duplicate/undeclared embedded model cell')
            embedded['model'][key] = row
    seen = set()
    for item in p['baseline_initializations']:
        ident = identity(item, lead)
        if ident != cohort.get(ident['init_time']) or ident['init_time'] in seen:
            raise ValueError('baseline exact full sample/init/valid identity mismatch')
        seen.add(ident['init_time'])
        if len(item['region_metrics']) != 102:
            raise ValueError('complete two-baseline embedded metric cells required')
        for row in item['region_metrics']:
            kind = row['baseline_kind']
            key = (ident['init_time'], row['region'], row['variable'])
            if kind not in ('persistence', 'climatology') or (row['region'], row['variable']) not in cells or key in embedded[kind]:
                raise ValueError('duplicate/undeclared embedded baseline cell')
            embedded[kind][key] = row
    if seen != set(cohort) or any(len(rows) != len(expected) * 51 for rows in embedded.values()):
        raise ValueError('complete per-case model and baseline embedded cohort required')
    return cohort, embedded


def csv_rows(path):
    with local(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('CSV requires unique explicit column names')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('malformed or incomplete CSV row')
            yield row


def bad_reasons(row):
    return [name for name, bad in (
        ('negative_mse_skill', row['mse_skill'] is not None and row['mse_skill'] < 0),
        ('negative_acc', row['pooled_acc'] is not None and row['pooled_acc'] < 0),
        ('undefined_mse_skill', row['mse_skill'] is None),
        ('undefined_acc', row['pooled_acc'] is None)) if bad]


def checked_metric(raw, data, ident, job, kind, expected):
    if integer(raw['lead_hours'], minimum=1) != job['lead'] or identity(
            {**raw, 'valid_times': json.loads(raw['valid_times'])} if isinstance(raw['valid_times'], str) else raw,
            job['lead']) != ident:
        raise ValueError('exact sample/init/valid/lead identity mismatch')
    if kind == 'model':
        if integer(raw['reasoning_steps'], minimum=1) != job['reasoning_steps']:
            raise ValueError('per-case actual K differs from frozen job')
    elif (raw.get('baseline_kind') != kind or raw.get('reasoning_steps') not in (None, '')
          or any(integer(raw[field]) != 0 for field in ('zero_train_updates', 'parameters', 'trainable_parameters'))):
        raise ValueError('baseline must remain zero-training/zero-parameter without a model K')
    canonical = _canonical_metric(raw, data)
    calculated = pooled_statistics([canonical])
    for key in ('rmse', 'rmse_climatology', 'mse_skill', 'pooled_acc', 'acc_status', 'skill_status'):
        value = canonical[key]
        target = calculated[key]
        if target is None:
            if value not in (None, ''):
                raise ValueError('undefined metric cannot be imputed, nonfinite or dropped')
        elif isinstance(target, str):
            if value != target:
                raise ValueError('metric undefined/defined status mismatch')
        else:
            close(value, target, key)
    encoded = raw['bad_reasons']
    reasons = json.loads(encoded) if isinstance(encoded, str) else encoded
    serialized = {key: None if canonical[key] in (None, '') else float(canonical[key])
                  for key in ('mse_skill', 'pooled_acc')}
    if reasons != bad_reasons(serialized):
        raise ValueError('negative/undefined per-case labels were filtered or changed')
    other = _canonical_metric({**expected, 'init_time': ident['init_time'], 'valid_times': ident['valid_times']}, data)
    for key in STATISTICS:
        close(canonical[key], other[key], 'CSV/embedded ' + key)
    for key in ('rmse', 'rmse_climatology', 'mse_skill', 'pooled_acc'):
        if calculated[key] is None:
            if other[key] not in (None, ''):
                raise ValueError('embedded undefined metric cannot be imputed')
        else:
            close(other[key], calculated[key], 'embedded ' + key)
    for key in ('region', 'variable', 'unit', 'lead_hours', 'n_initializations', 'acc_status', 'skill_status'):
        if canonical[key] != other[key]:
            raise ValueError('CSV/embedded metadata mismatch')
    if expected['bad_reasons'] != reasons:
        raise ValueError('CSV/embedded diagnostic mismatch')
    for key in ('margin_cells', 'n_grid_points', 'full_area_fraction'):
        close(raw[key], float(expected[key]), 'CSV/embedded geometry ' + key)
    if kind == 'climatology':
        close(canonical['mse'], canonical['climatology_mse'], 'zero-training climatology MSE')
        if canonical['acc_forecast_energy'] != 0 or canonical['acc_dot'] != 0 or calculated['pooled_acc'] is not None:
            raise ValueError('zero-training climate anomaly must retain undefined ACC')
    return {**ident, **{key: canonical[key] for key in STATISTICS}}


def read_complete_job(protocol, record, mode, tick):
    data, job = protocol['data'], record['job']
    cohort, embedded = verify_provenance(protocol, record, mode)
    cells = {(region, variable) for region in REGIONS for variable in data['channels']}
    expected_keys = {(init, region, variable) for init in cohort for region, variable in cells}
    buckets = {kind: {} for kind in ('model', 'persistence', 'climatology')}
    for field, baseline in (('per_case_metrics', False), ('baseline_per_case_metrics', True)):
        for raw in csv_rows(record[field]['path']):
            tick()
            kind = raw.get('baseline_kind') if baseline else 'model'
            key = (raw['init_time'], raw['region'], raw['variable'])
            if kind not in buckets or (not baseline and kind != 'model') or key not in expected_keys or key in buckets[kind]:
                raise ValueError('duplicate or undeclared CSV case/variable/region; no intersection')
            if baseline and kind == 'model':
                raise ValueError('a model is not a zero-trained reference')
            buckets[kind][key] = checked_metric(raw, data, cohort[raw['init_time']], job, kind, embedded[kind][key])
    if any(set(rows) != expected_keys for rows in buckets.values()):
        raise ValueError('missing CSV cases/cells; all 17 variables and three regions required')
    return cohort, buckets


def pool_hour(rows):
    if rows:
        return pooled_statistics(rows)
    return {**{key: None for key in STATISTICS}, 'rmse': None, 'rmse_climatology': None,
            'mse_skill': None, 'skill_status': 'undefined_empty_hour_group', 'pooled_acc': None,
            'acc_status': 'undefined_empty_hour_group', 'n_initializations': 0,
            'bad_case_counts': {key: 0 for key in ('worse_than_climatology', 'zero_climatology_mse',
                                                'negative_acc_dot', 'zero_anomaly_energy')}}


def summarize(protocol, job, kind, buckets, input_sources, cpu_digest, tick):
    data = protocol['data']
    table, groups = [], []
    dimensions = {'stage': protocol['stage'], 'entity': 'model' if kind == 'model' else 'baseline',
                  'arm': job['arm'] if kind == 'model' else None, 'seed': job['seed'],
                  'K': job['reasoning_steps'] if kind == 'model' else None,
                  'baseline_kind': None if kind == 'model' else kind, 'model_depth_applicable': kind == 'model',
                  'zero_train_updates': None if kind == 'model' else 0,
                  'parameters': None if kind == 'model' else 0, 'trainable_parameters': None if kind == 'model' else 0,
                  'lead_hours': job['lead'], 'protocol_sha256': protocol['protocol_sha256'],
                  'cpu_protocol_sha256': cpu_digest, 'model_code_sha256': protocol['code']['model_code_sha256'],
                  'source_tree_sha256': protocol['code']['source_tree_sha256'],
                  'actual_base_commit': protocol['code']['base_commit'], 'data_identity': data['data_identity'],
                  'scientific_claim': False, 'test_read': False}
    all_inits = sorted({key[0] for key in buckets})
    for axis in ('init_utc_hour', 'valid_utc_hour'):
        case_by_init = {key[0]: row for key, row in buckets.items()}
        for hour in HOURS:
            tick()
            inits = [init for init in all_inits if case_by_init[init][axis] == hour]
            cases = [{key: case_by_init[init][key] for key in
                      ('sample_id', 'init_time', 'valid_time', 'valid_times', 'init_utc_hour', 'valid_utc_hour')}
                     for init in inits]
            group = {**dimensions, 'group_axis': axis, 'utc_hour': hour, 'n_initializations': len(cases),
                     'case_set_sha256': digest(cases), 'cases': cases, 'input_sources': input_sources}
            group['group_sha256'] = digest(group)
            groups.append(group)
            for region in REGIONS:
                for variable, unit in zip(data['channels'], data['units']):
                    selected = [buckets[(init, region, variable)] for init in inits]
                    metrics = pool_hour(selected)
                    row = {**dimensions, 'group_axis': axis, 'utc_hour': hour, 'region': region,
                           'variable': variable, 'unit': unit, **metrics,
                           'physicalRMSE': metrics['rmse'], 'climateMSEskill': metrics['mse_skill'],
                           'pooledACC': metrics['pooled_acc'], 'case_set_sha256': group['case_set_sha256'],
                           'group_sha256': group['group_sha256'], 'input_sources_sha256': digest(input_sources),
                           'aggregation': 'equal-case physical sufficient statistics, then sqrt/division'}
                    row['row_sha256'] = digest(row)
                    table.append(row)
    return table, groups


def process_stage(protocol_records, cpu, tick):
    table, groups, baseline_registry, cohort_registry = [], [], {}, {}
    for protocol, record in protocol_records:
        tick()
        job = record['job']
        cohort, buckets = read_complete_job(protocol, record, cpu['mode'], tick)
        cohort_digest = digest(cohort)
        lead_key = job['lead']
        if cohort_registry.setdefault(lead_key, cohort_digest) != cohort_digest:
            raise ValueError('same-lead sample/init/valid identities differ across arms/seeds/K')
        sources = [{key: record[key] for key in ('receipt', 'provenance', 'per_case_metrics')}]
        model_table, model_groups = summarize(protocol, job, 'model', buckets['model'], sources, cpu['protocol_sha256'], tick)
        table.extend(model_table)
        groups.extend(model_groups)
        for kind in ('persistence', 'climatology'):
            key = (job['seed'], job['lead'], kind)
            semantic = digest([[list(k), v] for k, v in sorted(buckets[kind].items())])
            source = {key: record[key] for key in ('receipt', 'provenance', 'baseline_per_case_metrics')}
            source.update(reference_arm=job['arm'], reference_K=job['reasoning_steps'])
            if key in baseline_registry:
                if baseline_registry[key]['semantic'] != semantic:
                    raise ValueError('same-case zero-training baseline changes across arm/K exports')
                baseline_registry[key]['sources'].append(source)
            else:
                baseline_registry[key] = {'semantic': semantic, 'sources': [source], 'protocol': protocol,
                                          'job': job, 'kind': kind, 'rows': buckets[kind]}
    for value in baseline_registry.values():
        reference_table, reference_groups = summarize(value['protocol'], value['job'], value['kind'],
                                                      value['rows'], value['sources'], cpu['protocol_sha256'], tick)
        table.extend(reference_table)
        groups.extend(reference_groups)
    return table, groups
