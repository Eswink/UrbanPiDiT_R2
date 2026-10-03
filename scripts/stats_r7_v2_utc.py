"""No-forward CPU UTC statistics; claim a new output before trusting protocol or helper identities."""
from __future__ import annotations
import time
ENTRY_STARTED = time.perf_counter()  # Before parsing, helper imports, pins and statistics.

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BOOTSTRAP_SOURCES = ('training/r7_v2_utc_frozen_stats.py', 'training/r7_v2_utc_contract.py',
                     'training/r7_v2_utc_statistics.py', 'scripts/stats_r7_v2_utc.py')
BOOTSTRAP_LIMITATIONS = ['CPU statistics only, scientific_claim:false; no forward/GPU/weather/test/network.',
                        'A claimed failure is sealed, never retried or upgraded by restoring input/helper bytes.',
                        'Unvalidated declarations are diagnostic only; actual PASS is never asserted.']
MAX_JSON_BYTES = 16 * 1024 * 1024


def deny_network():
    import socket
    def denied(*args, **kwargs):
        raise RuntimeError('offline CPU statistics: network prohibited')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def safe_local(value):
    if not isinstance(value, str) or '://' in value:
        raise ValueError('explicit local absolute path required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink paths without traversal required')
    if any('legacy' in p.lower() or p == 'tests' or p.startswith('test.') or p == 'test.jsonl' for p in path.parts):
        raise ValueError('legacy/test paths forbidden')
    if any(path.parts[i:i + 2] in (('data', 'raw'), ('data', 'interim'), ('data', 'processed'))
           for i in range(len(path.parts) - 1)):
        raise ValueError('weather/data paths forbidden')
    return path


def minimal_json(path):
    """Bounded routing-only read: no pin/identity/model trust before the output claim."""
    deadline = ENTRY_STARTED + 30.
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('protocol must be a bounded regular JSON file')
    with path.open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('protocol JSON size exceeds the routing-read bound')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    body = json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if time.perf_counter() >= deadline:
        raise TimeoutError('minimal protocol routing read exceeded its bootstrap deadline')
    if not isinstance(body, dict):
        raise ValueError('protocol routing object required')
    return body, raw


def claim_output(protocol_path):
    path = safe_local(protocol_path)
    if path.name != 'protocol.json':
        raise ValueError('explicit protocol.json routing path required')
    body, raw = minimal_json(path)
    output = safe_local(body['output'])
    if output == Path('/tmp') or not output.is_relative_to(Path('/tmp')) or output.is_relative_to(ROOT):
        raise ValueError('writes require a new exclusive /tmp root outside the source tree')
    roots, pending, visited = [path.parent], [(body.get('stages', []), 0)], 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 30 or visited > 100000 or time.perf_counter() >= ENTRY_STARTED + 30.:
            raise ValueError('routing metadata structure exceeds bounded preclaim inspection')
        if isinstance(item, dict):
            if 'path' in item:
                roots.append(safe_local(item['path']).parent)
            pending.extend((value, depth + 1) for value in item.values())
        elif isinstance(item, list):
            pending.extend((value, depth + 1) for value in item)
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('exclusive output cannot overlap any frozen protocol/input root')
    output.mkdir(parents=True, exist_ok=False)  # Atomic ownership BEFORE every trusted validation/import.
    return path, body, raw, output


def emit_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def file_sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def clock_settings(cpu, mode):
    if mode == 'actual' and ('round_started_perf_counter' not in cpu or 'monotonic_boot_id' not in cpu):
        raise ValueError('actual whole clock requires the independently frozen earliest CPU prepare anchor')
    anchor, planned, hard = cpu.get('round_started_perf_counter', ENTRY_STARTED), cpu['planned_seconds'], cpu['hard_cap_seconds']
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (anchor, planned, hard)) or not 0 <= anchor <= ENTRY_STARTED or not 0 < planned < hard):
        raise ValueError('finite past prepare anchor and positive soft/hard CPU budget required')
    if 'monotonic_boot_id' in cpu and cpu['monotonic_boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('same-boot frozen CPU prepare clock required')
    return anchor, planned, hard


def reject_publication_failures(cpu):
    for spec in cpu['stages']:
        for field in ('protocol', 'artifact_manifest', 'attempt', 'source_attempt',
                      'complement_protocol', 'complement_provenance'):
            if field in spec:
                marker = safe_local(spec[field]['path']).parent / 'publication_failure.json'
                if marker.exists() or marker.is_symlink():
                    raise ValueError('source publication_failure marker forbids acceptance: ' + str(marker))


def execute_statistics(cpu, args, output, input_pins, tick, log):
    import zipfile
    from training.r7_v2_utc_contract import (HERE, LIMITATIONS, REFERENCE_SOURCES, SOURCE_NAMES, digest,
                                            read_json, sha256_file, source_pins, stage_inputs, write_csv, write_json)
    from training.r7_v2_utc_statistics import process_stage
    source = source_pins()
    with zipfile.ZipFile(output / 'helper_code.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_NAMES:
            tick()
            archive.writestr(name, (HERE / name).read_bytes())
    source_identity = {'source_files_sha256': source, 'source_sha256': digest(source),
                       'helper_code_zip_sha256': sha256_file(output / 'helper_code.zip'),
                       'reference_sources_sha256': digest(REFERENCE_SOURCES),
                       'scientific_claim': False, 'limitations': LIMITATIONS}
    write_json(output / 'source_identity.json', source_identity)
    log(f'mode={args.mode}; offline stdlib CPU; no GPU/forward/weather/test; frozen before statistics')
    table, groups, identities = [], [], []
    for spec in cpu['stages']:
        log(f"stage={spec['stage']}: verify complete final seals and all CSV byte pins")
        records = list(stage_inputs(spec, input_pins, args.mode))
        tick()
        protocol = records[0][0]
        identities.append({'stage': protocol['stage'], 'source_kind': spec['source_kind'],
                           'source_completion_pins': {key: spec[key] for key in ('artifact_manifest', 'attempt',
                                                      'source_attempt', 'complement_protocol', 'complement_provenance') if key in spec},
                           'actual_protocol_sha256': protocol['protocol_sha256'],
                           'actual_protocol_file_sha256': spec['protocol']['sha256'],
                           'actual_code': {key: protocol['code'][key] for key in
                                           ('base_commit', 'model_code_sha256', 'source_tree_sha256', 'code_zip_sha256')},
                           'data_identity': protocol['data']['data_identity'], 'source_sha256': protocol['sources']['source_sha256']})
        stage_table, stage_groups = process_stage(records, cpu, tick)
        table.extend(stage_table)
        groups.extend(stage_groups)
        log(f"stage={spec['stage']}: complete cells={len(stage_table)} groups={len(stage_groups)}")
    tick()
    write_csv(output / 'utc_group_metrics.csv', table)
    write_json(output / 'group_identities.json', {'scientific_claim': False, 'limitations': LIMITATIONS,
               'cpu_protocol_sha256': cpu['protocol_sha256'], 'groups': groups})
    for path, expected in input_pins.items():
        tick()
        if sha256_file(path) != expected:
            raise ValueError('frozen input changed during CPU statistics: ' + path)
    if source_pins() != source:
        raise ValueError('helper source changed during CPU statistics')
    if (read_json(output / 'source_identity.json') != source_identity
            or sha256_file(output / 'helper_code.zip') != source_identity['helper_code_zip_sha256']):
        raise ValueError('published helper ZIP/source identity changed during CPU statistics')
    tick()
    status = 'synthetic-statistics-complete' if args.mode == 'synthetic' else 'actual-statistics-complete'
    write_json(output / 'statistics_summary.json', {
        'status': status, 'scientific_claim': False, 'limitations': LIMITATIONS, 'test_read': False,
        'actual_pass': False, 'scientific_gate_evaluated': False, 'mode': args.mode,
        'rows': len(table), 'groups': len(groups), 'cpu_protocol_sha256': cpu['protocol_sha256'],
        'source_sha256': digest(source), 'actual_frozen_identities': identities,
        'input_files_sha256': input_pins, 'rows_sha256': digest(table), 'group_identities_sha256': digest(groups),
        'reproducibility_level': 'identity-bound numerical replay; no cross-platform bitwise claim'})
    return status, len(table), len(groups)


def seal_output(output, args, cpu, state, input_pins, log):
    from_context = state['validated']
    elapsed = time.perf_counter() - state['anchor']
    hard = state['hard']
    if hard is not None and elapsed >= hard:
        state.update(status='failed', error=state['error'] or 'TimeoutError: hard cap reached before final CPU seal')
    overrun = None if state['planned'] is None else max(0., elapsed - state['planned'])
    log(f"final status={state['status']}; soft_overrun_seconds={overrun}")
    protocol_digest = cpu.get('protocol_sha256')
    if not isinstance(protocol_digest, str) or len(protocol_digest) != 64:
        protocol_digest = None
    emit_json(output / 'attempt.json', {
        'format': 'r7-utc-cpu-attempt-v1', 'status': state['status'], 'mode': args.mode, 'actual_pass': False,
        'scientific_claim': False, 'limitations': BOOTSTRAP_LIMITATIONS, 'test_read': False,
        'failure_reason': state['error'], 'budget_limited': state['error'] is not None and 'TimeoutError' in state['error'],
        'partial': state['error'] is not None, 'finalized': state['error'] is None,
        'cpu_protocol_sha256': protocol_digest, 'cpu_protocol_file_sha256': args.protocol_file_sha256,
        'protocol_validated': from_context, 'entry_started_perf_counter': ENTRY_STARTED,
        'round_started_perf_counter': state['anchor'], 'ended_perf_counter': state['anchor'] + elapsed,
        'whole_elapsed_seconds': elapsed, 'planned_seconds': state['planned'], 'hard_cap_seconds': hard,
        'soft_overrun_seconds': overrun,
        'whole_clock_scope': 'frozen earliest prepare entry/imports/pins/freeze, prepare-run gap, CLI claim/validation/statistics/publication/postflight; final receipt serialization follows snapshot',
        'rows': state['rows'], 'groups': state['groups'], 'input_files_sha256': input_pins})
    files = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    files_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    emit_json(output / 'pins.json', {'scientific_claim': False, 'limitations': BOOTSTRAP_LIMITATIONS,
              'status': state['status'], 'cpu_protocol_sha256': protocol_digest, 'files_sha256': files,
              'files_digest': files_digest, 'excluded_self': 'pins.json'})
    terminal = time.perf_counter() - state['anchor']
    limited = hard is not None and terminal >= hard
    if limited:
        emit_json(output / 'publication_failure.json', {'status': 'failed', 'scientific_claim': False,
                  'limitations': BOOTSTRAP_LIMITATIONS, 'whole_elapsed_seconds': terminal,
                  'failure_reason': 'hard cap exceeded during final receipt/pins publication; not accepted'})
    print(json.dumps({'status': 'failed' if limited else state['status'], 'output': str(output), 'mode': args.mode,
                      'actual_pass': False, 'scientific_claim': False, 'failure_reason': state['error'],
                      'whole_elapsed_seconds_through_publication': terminal,
                      'cpu_protocol_sha256': protocol_digest}, allow_nan=False))
    return 1 if state['error'] or limited else 0


def main(argv=None):
    deny_network()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, help='Frozen independent CPU statistics protocol; absolute path')
    parser.add_argument('--protocol-file-sha256', required=True, help='Expected SHA256 of protocol file bytes')
    parser.add_argument('--mode', required=True, choices=('synthetic', 'actual'))
    args = parser.parse_args(argv)
    cpu_path, cpu, raw, output = claim_output(args.protocol)
    state = {'anchor': ENTRY_STARTED, 'planned': None, 'hard': None, 'validated': False,
             'status': 'failed', 'error': None, 'rows': 0, 'groups': 0}
    input_pins, contract = {}, None
    def tick():
        if state['hard'] is not None and time.perf_counter() >= state['anchor'] + state['hard']:
            raise TimeoutError('CPU statistics hard whole-round deadline exceeded')
    def log(message):
        with (output / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(f"{time.perf_counter() - state['anchor']:.6f}s {message}\n")
    try:
        emit_json(output / 'run_started.json', {'format': 'r7-utc-cpu-started-v1', 'mode': args.mode,
                  'scientific_claim': False, 'limitations': BOOTSTRAP_LIMITATIONS, 'test_read': False,
                  'cpu_protocol_file_sha256': args.protocol_file_sha256, 'entry_started_perf_counter': ENTRY_STARTED,
                  'output_exclusive': True, 'protocol_validated': False})
        with (output / 'protocol.json').open('xb') as stream:
            stream.write(raw)
        state['anchor'], state['planned'], state['hard'] = clock_settings(cpu, args.mode)
        tick()
        if hashlib.sha256(raw).hexdigest() != args.protocol_file_sha256:
            raise ValueError('frozen CPU protocol file byte pin mismatch')
        declared = cpu.get('source_files_sha256', {})
        if set(declared) != set(BOOTSTRAP_SOURCES):
            raise ValueError('frozen helper source identity inventory mismatch')
        for name in BOOTSTRAP_SOURCES:
            tick()
            if file_sha256(safe_local(str(ROOT / name))) != declared[name]:
                raise ValueError('frozen helper source SHA256 mismatch: ' + name)
        from training import r7_v2_utc_contract as contract
        contract.DEADLINE = state['anchor'] + state['hard']
        contract.check_pin({'path': args.protocol, 'sha256': args.protocol_file_sha256}, input_pins, name='protocol.json')
        contract.validate_cpu(cpu, args.mode)
        tick()
        state['validated'] = True
        reject_publication_failures(cpu)
        state['status'], state['rows'], state['groups'] = execute_statistics(cpu, args, output, input_pins, tick, log)
        reject_publication_failures(cpu)  # A marker published during consumption also invalidates the attempt.
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        log('FAILED ' + state['error'])
    finally:
        if contract is not None:
            contract.DEADLINE = None  # Retain failed evidence after hard truncation; never accept overtime.
    return seal_output(output, args, cpu, state, input_pins, log)


if __name__ == '__main__':
    raise SystemExit(main())
