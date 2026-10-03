"""Independent zero-GPU metadata complement for the exact sealed B validator failure.

prepare() binds old failed inputs and the narrowly corrected validator before
run() reads any forecast metrics. run() executes the archived validator in an
isolated import tree plus only the FP32 receipt correction, never current models,
never original finalize, and never alters the old failed attempt/protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

CLI_STARTED = time.perf_counter()

from .r7_v2_stats_sources import (
    CORRECTED, FORMAT, OBJECTIVE_HELPER, MAX_SOURCE_BYTES, archive_content, archive_members,
    bounded_bytes, corrected_source, digest, extract_corrected, path, read_json, seal_owned,
    sha256_file, source_metadata, verify_archive, verify_inventory, write_json,
)
from .r7_v2_stats_tables import publish

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILES = ("training/r7_v2_stats_complement.py", "training/r7_v2_stats_sources.py", "training/r7_v2_stats_tables.py",
                 OBJECTIVE_HELPER, "scripts/r7_m3_offline.py")
LIMITATIONS = [
    "scientific_claim:false; independent metadata acceptance, not a training/evaluation or scientific support claim",
    "original B attempt stays failed; its entire GPU/whole cost remains charged and referenced unchanged",
    "only exact serialized FP32 objective reconstruction differs; physical metric tolerances and scientific inequalities unchanged",
    "all numerical reaggregation consumes archived same-case CSV/provenance, never weather data getitem or sealed test",
    "two seeds and one winter region are descriptive consistency, not significance or GPU bitwise reproducibility",
]


def offline():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from scripts.r7_m3_offline import deny_network
    deny_network()


def output_path(value, source=None):
    requested = path(value)
    source = None if source is None else path(source)
    if ((source is not None and (requested == source or requested.is_relative_to(source) or source.is_relative_to(requested)))
            or any("legacy" in part.lower() for part in requested.parts)
            or any(requested.parts[i:i + 2] in (("data", "raw"), ("data", "processed"), ("data", "interim"))
                   for i in range(len(requested.parts) - 1))):
        raise ValueError("independent new metadata output cannot overlap old source/protected paths")
    return requested


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def clock_fields(anchor, protocol, *, clock=None):
    now = (time.perf_counter if clock is None else clock)()
    return {"ended_perf_counter": now, "whole_elapsed_seconds": now - anchor,
            "soft_overrun_seconds": max(0., now - anchor - protocol["planned_seconds"])}


def new_attempt(anchor, phase):
    return {"format": FORMAT, "source_kind": FORMAT, "status": "failed", "phase": phase,
            "finalized": False, "coverage_complete": False, "partial": True, "budget_limited": False,
            "owned_unreaped": False, "failure_reason": None, "scientific_claim": False, "limitations": LIMITATIONS,
            "test_read": False, "training_performed": False, "evaluation_performed": False,
            "gpu_used": False, "gpu_hours_charged": 0., "source_attempt_failed": True,
            "planned_seconds": 600., "hard_cap_seconds": 1200., "cleanup_reserve_seconds": 10.,
            "started_perf_counter": anchor}


def finish_owned(stream, output, attempt, started, error, *, success=False, clock=None):
    """Finalize only the claimed descriptor; final hashes/reap/serialization cannot overrun into success."""
    clock = time.perf_counter if clock is None else clock
    try:
        if success and error is None:
            _check(started, attempt, clock=clock)
            attempt.update(status="aggregation-complete", finalized=True, coverage_complete=True, partial=False)
        attempt.update(clock_fields(started, attempt, clock=clock))
        if error is not None:
            attempt.update(status="failed", finalized=False, coverage_complete=False, partial=True,
                           budget_limited=isinstance(error, TimeoutError), failure_reason=f"{type(error).__name__}: {error}")
        check = (lambda: _check(started, attempt, clock=clock)) if attempt["finalized"] else (lambda: None)
        seal_owned(stream, attempt, check=check)
        if attempt["finalized"]:
            _check(started, attempt, clock=clock)
            attempt.update(clock_fields(started, attempt, clock=clock))
            seal_owned(stream, attempt, check=check)
            _check(started, attempt, clock=clock)
    except BaseException as exc:
        error = error or exc
        attempt.update(status="failed", finalized=False, coverage_complete=False, partial=True,
                       budget_limited=isinstance(error, TimeoutError), failure_reason=f"{type(error).__name__}: {error}",
                       **clock_fields(started, attempt, clock=clock))
        try:
            seal_owned(stream, attempt)
        except BaseException:
            # An independent negative marker rejects a stale/partial success when final serialization fails.
            with (output / "publication_failure").open("x", encoding="ascii") as marker:
                marker.write("failed final publication; never accepted\n")
            raise
    return error


def _freeze(source_pins, output, *, planned_seconds, hard_cap_seconds, anchor, boot, check):
    """Internal preparation under the already-established whole CPU deadline."""
    if planned_seconds != 600 or hard_cap_seconds != 1200:
        raise ValueError("new stats complement freezes CPU soft600/hard1200, never old GPU budgets")
    pins_path = path(source_pins)
    pins = read_json(pins_path, check=check)
    source = path(pins["source_attempt"])
    output = output_path(output, source)
    check()
    original = read_json(source / "protocol.json", check=check)
    verify_inventory(source, pins["files_sha256"], check=check)
    _, attempt, _, inventory = source_metadata(source, original["protocol_sha256"], check=check)
    old_validator = verify_archive(source, original, check=check)
    correction = {"kind": "exact-FP32-loss-multiply-then-add-receipt-check",
                  "archived_validator_sha256": hashlib.sha256(old_validator).hexdigest(),
                  "corrected_validator_sha256": hashlib.sha256(corrected_source(old_validator)).hexdigest(),
                  "helper_path": str(ROOT / OBJECTIVE_HELPER), "helper_sha256": sha256_file(ROOT / OBJECTIVE_HELPER, check=check),
                  "allowed_changed_archive_members": [CORRECTED], "added_runtime_members": [OBJECTIVE_HELPER],
                  "metric_tolerance_changed": False, "scientific_threshold_changed": False}
    helpers = {name: sha256_file(ROOT / name, check=check) for name in SUPPORT_FILES}
    with (output / "complement_code.zip").open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in SUPPORT_FILES:
                content = bounded_bytes(ROOT / name, limit=MAX_SOURCE_BYTES, check=check)
                if hashlib.sha256(content).hexdigest() != helpers[name]:
                    raise ValueError("metadata helper changed while freezing archive")
                archive.writestr(name, content)
                check()
    body = {"format": FORMAT, "source_kind": FORMAT, "source_attempt": str(source), "output": str(output),
            "source_pins_path": str(pins_path), "source_pins_sha256": sha256_file(pins_path, check=check),
            "source_files_sha256": pins["files_sha256"], "source_files_digest": digest(pins["files_sha256"]),
            "source_protocol_sha256": original["protocol_sha256"], "source_protocol_file_sha256": sha256_file(source / "protocol.json", check=check),
            "source_attempt_sha256": sha256_file(source / "attempt.json", check=check), "source_code": original["code"],
            "source_status": attempt["status"], "source_inventory": inventory, "correction": correction,
            "helper_files_sha256": helpers, "helper_source_root": str(ROOT),
            "complement_code_zip_sha256": sha256_file(output / "complement_code.zip", check=check),
            "planned_seconds": planned_seconds, "hard_cap_seconds": hard_cap_seconds,
            "round_started_perf_counter": anchor, "monotonic_boot_id": boot, "cleanup_reserve_seconds": 10.,
            "gpu_hours_charged": 0., "training_performed": False, "evaluation_performed": False,
            "weather_getitem": False, "test_read": False, "source_write": False,
            "frozen_before_reaggregation": True, "no_retry_or_resurrection": True,
            "scientific_claim": False, "limitations": LIMITATIONS}
    protocol = {**body, "protocol_sha256": digest(body)}
    write_json(output / "protocol.json", protocol, check=check)
    check()
    return protocol


def prepare(source_pins, output, *, planned_seconds=600., hard_cap_seconds=1200.,
            round_started_perf_counter=None, monotonic_boot_id=None, clock=None):
    """Claim new directory before any trusted preflight; all own failures permanently seal it."""
    clock = time.perf_counter if clock is None else clock
    entry = clock()
    anchor = entry if round_started_perf_counter is None else round_started_perf_counter
    safe_anchor = anchor if type(anchor) in (int, float) and math.isfinite(anchor) and 0 <= anchor <= entry else entry
    output = output_path(output)
    if any((parent / "protocol.json").exists() and (parent / "attempt.json").exists() for parent in output.parents):
        raise ValueError("new metadata output must not write inside an existing sealed source")
    output.mkdir(parents=True, exist_ok=False)
    attempt = new_attempt(safe_anchor, "cpu-prepare")
    try:
        offline()
        boot = boot_id() if monotonic_boot_id is None else monotonic_boot_id
        attempt["monotonic_boot_id"] = boot
        if (planned_seconds != 600 or hard_cap_seconds != 1200 or safe_anchor != anchor
                or isinstance(anchor, bool) or boot != boot_id()):
            raise ValueError("valid earliest same-boot CPU anchor and soft600/hard1200 required")
        check = lambda: _check(anchor, attempt, clock=clock)
        check()
        protocol = _freeze(source_pins, output, planned_seconds=planned_seconds, hard_cap_seconds=hard_cap_seconds,
                           anchor=anchor, boot=boot, check=check)
        check()
        return protocol
    except BaseException as exc:
        with (output / "attempt.json").open("x+", encoding="utf-8") as stream:
            finish_owned(stream, output, attempt, safe_anchor, exc, clock=clock)
        raise


def _validate(protocol, *, check=lambda: None):
    check()
    if (protocol["format"] != FORMAT or protocol["source_kind"] != FORMAT
            or protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            or any(protocol[key] is not False for key in ("scientific_claim", "test_read", "training_performed", "evaluation_performed",
                                                         "weather_getitem", "source_write"))
            or protocol["planned_seconds"] != 600 or protocol["hard_cap_seconds"] != 1200 or protocol["cleanup_reserve_seconds"] != 10.
            or protocol["gpu_hours_charged"] != 0. or protocol["source_status"] != "failed"
            or protocol["frozen_before_reaggregation"] is not True or protocol["no_retry_or_resurrection"] is not True
            or set(protocol["helper_files_sha256"]) != set(SUPPORT_FILES)):
        raise ValueError("frozen independent stats-only complement protocol required")
    source, output = path(protocol["source_attempt"]), output_path(protocol["output"], protocol["source_attempt"])
    if (source / "publication_failure").exists() or (output / "publication_failure").exists():
        raise ValueError("failed final publication marker forbids metadata acceptance")
    if sha256_file(protocol["source_pins_path"], check=check) != protocol["source_pins_sha256"]:
        raise ValueError("original failed-source inventory receipt changed")
    source_pins = read_json(protocol["source_pins_path"], check=check)
    if (path(source_pins["source_attempt"]) != source or source_pins["files_sha256"] != protocol["source_files_sha256"]
            or digest(protocol["source_files_sha256"]) != protocol["source_files_digest"]):
        raise ValueError("exact frozen failed-source file identity required")
    for name, key in (("protocol.json", "source_protocol_file_sha256"), ("attempt.json", "source_attempt_sha256")):
        if (protocol["source_files_sha256"][name] != protocol[key]
                or sha256_file(source / name, check=check) != protocol[key]):
            raise ValueError("original source protocol/attempt file identity changed")
    original = read_json(source / "protocol.json", check=check)
    if (original["stage"] != "B" or original["protocol_sha256"] != protocol["source_protocol_sha256"]
            or digest({k: v for k, v in original.items() if k != "protocol_sha256"}) != protocol["source_protocol_sha256"]
            or original["code"] != protocol["source_code"]):
        raise ValueError("original canonical B protocol/code identity mismatch")
    if sha256_file(output / "complement_code.zip", check=check) != protocol["complement_code_zip_sha256"]:
        raise ValueError("complement source archive changed")
    with zipfile.ZipFile(output / "complement_code.zip") as archive:
        for item in archive_members(archive, protocol["helper_files_sha256"], check=check):
            content = archive_content(archive, item, output, check=check)
            if hashlib.sha256(content).hexdigest() != protocol["helper_files_sha256"][item.filename]:
                raise ValueError("frozen complement archive member changed")
    for name, expected in protocol["helper_files_sha256"].items():
        if sha256_file(path(protocol["helper_source_root"]) / name, check=check) != expected:
            raise ValueError("frozen metadata helper source changed")
    check()
    return source, output


def _check(started, protocol, *, clock=None):
    now = (time.perf_counter if clock is None else clock)()
    if now - started >= protocol["hard_cap_seconds"] - protocol["cleanup_reserve_seconds"]:
        raise TimeoutError("metadata complement whole CPU hard budget exhausted (cleanup/failure seal reserve)")


def _worker(protocol_path, started):
    """Fresh process sees only extracted original code plus corrected FP32 validator."""
    offline()
    protocol = read_json(protocol_path)
    check = lambda: _check(started, protocol)
    source, output = _validate(protocol, check=check)
    verify_inventory(source, protocol["source_files_sha256"], check=check)
    original, failed_attempt, execution, inventory = source_metadata(source, protocol["source_protocol_sha256"], check=check)
    if original["code"] != protocol["source_code"] or inventory != protocol["source_inventory"]:
        raise ValueError("frozen source code or full 36 receipt inventory changed")
    runtime_pins = {**original["code"]["files"], CORRECTED: protocol["correction"]["corrected_validator_sha256"],
                    OBJECTIVE_HELPER: protocol["correction"]["helper_sha256"]}
    verify_inventory(output / "archive_runtime", runtime_pins, check=check)
    # Import safety: launch sets cwd/runtime path to original extracted code only.
    from training.r7_v2_results import validate_full_set
    from training.r7_v2_tables import aggregate_metrics, paired_comparisons, candidate_selection
    from training.r7_v2_protocol import B_ARMS, B_SEEDS, LEADS
    imported = path(sys.modules["training.r7_v2_results"].__file__)
    if imported != output / "archive_runtime" / CORRECTED:
        raise ValueError("aggregation must use explicitly corrected original archived source, never current code")
    if sha256_file(imported, check=check) != protocol["correction"]["corrected_validator_sha256"]:
        raise ValueError("corrected archived validator bytes changed")
    training, evaluations, records = validate_full_set(source, original)
    check()
    table = aggregate_metrics(records, seeds=B_SEEDS, arms=B_ARMS, leads=LEADS, kernels=(4,))
    compared = paired_comparisons(table, pairs=(("rollout_l6_l12", "continue_l6"), ("rollout_l6_l12", "equal_compute_l6"),
                                               ("equal_compute_l6", "continue_l6")), kernels=(4,))
    selection = candidate_selection(original, compared)
    if len(training) != 6 or len(evaluations) != 30 or len(records) != 1530 or len(table) != 765 or any(len(p["cells"]) != 255 for p in compared.values()):
        raise ValueError("full strict B inventory/metric/pair coverage required")
    check()
    published = publish(output, protocol, original, training, evaluations, records, table, compared, selection, failed_attempt, check=check)
    provenance = {"format": FORMAT, "source_kind": FORMAT, "status": "aggregation-complete", "coverage_complete": True,
                  "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False, "training_performed": False,
                  "evaluation_performed": False, "gpu_used": False, "source_attempt_failed": True,
                  "source_protocol_sha256": original["protocol_sha256"], "complement_protocol_sha256": protocol["protocol_sha256"],
                  "source_attempt_sha256": protocol["source_attempt_sha256"], "source_inventory": inventory,
                  "source_failed_cost_reference": str(source / "attempt.json"), "source_code": original["code"],
                  "correction": protocol["correction"], "n_training": 6, "n_evaluations": 30, "n_metric_rows": 1530,
                  "n_aggregate_rows": 765, "candidate_selection": selection, "reproducibility_level": "identity-bound metadata reaggregation; no GPU bitwise claim"}
    write_json(output / "provenance.json", provenance, check=check)
    verify_inventory(source, protocol["source_files_sha256"], check=check)
    verify_inventory(output / "archive_runtime", runtime_pins, check=check)
    pins = {name: sha256_file(output / name, check=check) for name in ["protocol.json", "complement_code.zip", "provenance.json", "worker.log", *published]}
    pins.update({"archive_runtime/" + name: pin for name, pin in runtime_pins.items()})
    sources = {str(source / name): pin for name, pin in protocol["source_files_sha256"].items()}
    manifest = {"source_kind": FORMAT, "scientific_claim": False, "limitations": LIMITATIONS,
                "complement_protocol_sha256": protocol["protocol_sha256"], "source_protocol_sha256": original["protocol_sha256"],
                "source_files_sha256": sources, "source_files_digest": digest(sources),
                "files_sha256": pins, "files_digest": digest(pins), "excluded_recursive_or_final": ["artifact_manifest.json", "attempt.json"]}
    write_json(output / "artifact_manifest.json", manifest, check=check)
    return provenance


def reap_cpu_owned(process, deadline, *, clock=None):
    """Only this direct child handle, with bounded waits for every exception path."""
    clock = time.perf_counter if clock is None else clock
    if process is None:
        return {"owned_unreaped": False, "cleanup": "no-owned-child"}
    if process.poll() is not None:
        process.wait(timeout=0)
        return {"owned_unreaped": False, "cleanup": "already-exited"}
    process.kill()
    try:
        process.wait(timeout=max(0., deadline - clock()))
        return {"owned_unreaped": False, "cleanup": "killed-and-reaped-owned-CPU-child"}
    except subprocess.TimeoutExpired:
        return {"owned_unreaped": True, "cleanup": "owned-CPU-child-unreaped-at-hard-deadline"}


def run(protocol_path):
    """Exclusive claim precedes trusted validation; every own failure is permanent, never resurrected."""
    run_entry = time.perf_counter()
    protocol_path = path(protocol_path)
    output = output_path(protocol_path.parent)
    if protocol_path.name != "protocol.json":
        raise ValueError("independent complement protocol.json path required")
    started, attempt = run_entry, new_attempt(run_entry, "cpu-run")
    error, process = None, None
    # The exclusive descriptor is also the durable claim. Prior owners are never overwritten.
    with (output / "attempt.json").open("x+", encoding="utf-8") as stream:
        try:
            seal_owned(stream, attempt)
            offline()
            for item in output.iterdir():
                if item.name not in ("protocol.json", "complement_code.zip", "attempt.json"):
                    raise FileExistsError("complement already started/completed/failed; no retry or resurrection")
            protocol = read_json(protocol_path, check=lambda: _check(started, attempt))
            anchor = protocol["round_started_perf_counter"]
            if (protocol["monotonic_boot_id"] != boot_id() or type(anchor) not in (int, float)
                    or not math.isfinite(anchor) or not 0 <= anchor <= run_entry):
                raise ValueError("same-boot frozen whole CPU prepare anchor required; no clock reset")
            started = anchor
            attempt.update(started_perf_counter=started, run_entry_perf_counter=run_entry,
                           monotonic_boot_id=protocol["monotonic_boot_id"], source_protocol_sha256=protocol["source_protocol_sha256"],
                           complement_protocol_sha256=protocol["protocol_sha256"], source_attempt_sha256=protocol["source_attempt_sha256"],
                           whole_clock_scope="earliest CPU prepare/imports, freeze, prepare-run gap, aggregation/output/owned reap; no GPU")
            check = lambda: _check(started, attempt)
            source, declared_output = _validate(protocol, check=check)
            if output != declared_output:
                raise ValueError("protocol output must equal the exclusively claimed directory")
            original = read_json(source / "protocol.json", check=check)
            verify_archive(source, original, check=check)
            runtime = extract_corrected(source, original, output / "archive_runtime", protocol["correction"], check=check)
            bootstrap = ("import sys,types,pathlib,importlib.util; "
                         "p=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); "
                         "pkg=types.ModuleType('stats_support');pkg.__path__=[str(root/'training')];sys.modules['stats_support']=pkg; "
                         "spec=importlib.util.spec_from_file_location('stats_support.r7_v2_stats_complement',root/'training/r7_v2_stats_complement.py'); "
                         "m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m._worker(p,float(sys.argv[3]))")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(runtime), "PYTHONDONTWRITEBYTECODE": "1"}
            with (output / "worker.log").open("x", encoding="utf-8") as log:
                check()
                remaining = started + attempt["hard_cap_seconds"] - attempt["cleanup_reserve_seconds"] - time.perf_counter()
                process = subprocess.Popen([sys.executable, "-B", "-c", bootstrap, str(protocol_path), str(ROOT), repr(started)],
                                           cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=max(0., remaining))
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError("owned CPU complement child exhausted whole budget; bounded cleanup follows") from exc
            if code != 0:
                raise RuntimeError(f"CPU complement worker failed with exit {code}; inspect independent worker.log")
            result = read_json(output / "provenance.json", check=check)
            attempt.update(candidate_selection=result["candidate_selection"],
                           provenance_sha256=sha256_file(output / "provenance.json", check=check),
                           artifact_manifest_sha256=sha256_file(output / "artifact_manifest.json", check=check))
        except BaseException as exc:
            error = exc
        finally:
            try:
                attempt.update(reap_cpu_owned(process, started + attempt["hard_cap_seconds"]))
            except BaseException as additional:
                attempt.update(owned_unreaped=True, cleanup=f"owned cleanup failed: {type(additional).__name__}: {additional}")
                error = error or additional
            if attempt["owned_unreaped"]:
                error = error or RuntimeError("owned CPU child not reaped; complement not accepted")
            if error is None:
                try:
                    final_log = sha256_file(output / "worker.log", check=check)
                    manifest = read_json(output / "artifact_manifest.json", check=check)
                    if manifest["files_sha256"].get("worker.log") != final_log:
                        raise ValueError("owned worker log changed after manifest publication or reap")
                    attempt["worker_log_sha256"] = final_log
                except BaseException as additional:
                    error = additional
            error = finish_owned(stream, output, attempt, started, error, success=error is None)
    if error is not None:
        raise error
    return attempt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--source-pins", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    replay = sub.add_parser("run")
    replay.add_argument("--protocol", required=True, type=Path)
    options = parser.parse_args(argv)
    return prepare(options.source_pins, options.output, round_started_perf_counter=CLI_STARTED) if options.mode == "prepare" else run(options.protocol)


if __name__ == "__main__":
    main()
