"""Read-only failed-source pins, exact archive extraction and explicit validator correction."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

FORMAT = "r7-v2-stats-complement"
CORRECTED = "training/r7_v2_results.py"
OBJECTIVE_HELPER = "training/r7_v2_objective_receipt.py"
# Metadata/source transport bounds, not numerical/scientific acceptance tolerances.
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_INVENTORY_ENTRIES = 4096
OLD_OBJECTIVE = '''        if config["mode"] == "two_step":
            close(loss["loss"], number(loss["l6"]) + controls["lambda12"] * number(loss["l12"], nonnegative=True), "training objective")
        elif loss["l12"] is not None:
            raise ValueError("L6 training must not consume a future L12 loss")
'''
NEW_OBJECTIVE = '''        from .r7_v2_objective_receipt import verify_training_objective
        verify_training_objective(loss, mode=config["mode"], controls=controls, contract=contract)
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path(value):
    requested = Path(value).absolute()
    if ("://" in str(value) or ".." in requested.parts
            or any(item.is_symlink() for item in (requested, *requested.parents))):
        raise ValueError("local nonsymlink paths without traversal required")
    if requested.name == "test.jsonl":
        raise ValueError("sealed test manifest forbidden")
    return requested.resolve()


def sha256_file(value, *, check=lambda: None):
    check()
    result = hashlib.sha256()
    with path(value).open("rb") as stream:
        while True:
            check()
            block = stream.read(1024 * 1024)
            check()
            if not block:
                break
            result.update(block)
    check()
    return result.hexdigest()


def bounded_bytes(value, *, limit, check=lambda: None):
    check()
    requested = path(value)
    if not requested.is_file() or requested.stat().st_size > limit:
        raise ValueError("bounded regular metadata/source file required")
    data = bytearray()
    with requested.open("rb") as stream:
        while True:
            check()
            block = stream.read(min(1024 * 1024, limit + 1 - len(data)))
            check()
            if not block:
                break
            data.extend(block)
            if len(data) > limit:
                raise ValueError("metadata/source byte cap exceeded")
    return bytes(data)


def read_json(value, *, check=lambda: None):
    result = json.loads(bounded_bytes(value, limit=MAX_JSON_BYTES, check=check))
    check()
    return result


def dump_json(stream, payload, *, check=lambda: None):
    for piece in json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False).iterencode(payload):
        check()
        stream.write(piece)
    stream.write("\n")
    stream.flush()
    check()


def write_json(value, payload, *, check=lambda: None):
    check()
    with path(value).open("x", encoding="utf-8") as stream:
        dump_json(stream, payload, check=check)
    check()


def seal_owned(stream, attempt, *, check=lambda: None):
    """Update only an exclusive claimed descriptor; never overwrite another attempt."""
    stream.seek(0)
    stream.truncate()
    dump_json(stream, attempt, check=check)


def relative_file(root, name):
    relative = PurePosixPath(name)
    if (not name or str(relative) != name or relative.is_absolute() or "\\" in name
            or any(part in (".", "..") for part in relative.parts)):
        raise ValueError("unsafe inventory member")
    requested = path(Path(root) / name)
    if not requested.is_relative_to(path(root)) or requested.name == "test.jsonl":
        raise ValueError("inventory escape/sealed test forbidden")
    return requested


def verify_inventory(root, expected, *, check=lambda: None):
    check()
    root = path(root)
    if not root.is_dir() or not isinstance(expected, dict) or not 0 < len(expected) <= MAX_INVENTORY_ENTRIES:
        raise ValueError("bounded exact inventory required")
    actual, stack, count = set(), [root], 0
    while stack:
        check()
        with os.scandir(stack.pop()) as entries:
            for item in entries:
                check()
                count += 1
                if count > MAX_INVENTORY_ENTRIES or item.is_symlink():
                    raise ValueError("source exact whole inventory changed; no symlinks/extra entries")
                if item.is_dir(follow_symlinks=False):
                    stack.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):
                    actual.add(Path(item.path).relative_to(root).as_posix())
                else:
                    raise ValueError("source inventory special file forbidden")
    if actual != set(expected):
        raise ValueError("source exact whole inventory changed; no intersection or skipped files")
    for name, pin in expected.items():
        check()
        if sha256_file(relative_file(root, name), check=check) != pin:
            raise ValueError(f"source bytes changed: {name}")
    check()
    return len(actual)


def source_metadata(source, expected_protocol, *, check=lambda: None):
    """Only receipts and opaque hashes; no checkpoint deserialization or weather reads."""
    source = path(source)
    if (source / "publication_failure").exists():
        raise ValueError("failed final publication marker forbids source qualification")
    original = read_json(source / "protocol.json", check=check)
    if original["protocol_sha256"] != expected_protocol or digest({k: v for k, v in original.items() if k != "protocol_sha256"}) != expected_protocol:
        raise ValueError("original frozen protocol digest mismatch")
    attempt = read_json(source / "attempt.json", check=check)
    execution = read_json(source / "execution_attempt.json", check=check)
    for item in (attempt, execution):
        if (item["protocol_sha256"] != expected_protocol or item["stage"] != "B"
                or item["status"] != "failed" or item["finalized"] is not False
                or any(item[key] is not False for key in ("partial", "budget_limited", "owned_unreaped", "test_read", "scientific_claim"))):
            raise ValueError("both failed source seals must bind canonical B protocol and explicit false safety flags")
    jobs = original["jobs"]
    if (original["stage"] != "B" or len(jobs) != 36 or sum(job["phase"] == "train" for job in jobs) != 6
            or sum(job["phase"] == "evaluate" for job in jobs) != 30
            or attempt["status"] != "failed" or attempt["finalized"] is not False
            or execution["status"] != "failed" or execution["finalized"] is not False
            or attempt["failure_reason"] != "ValueError: same-case metric mismatch: training objective"
            or execution["failure_reason"] != attempt["failure_reason"]
            or attempt["jobs_planned"] != jobs or attempt["jobs_completed"] != jobs
            or execution["jobs_completed"] != jobs or attempt["partial"] is not False
            or attempt["budget_limited"] is not False or attempt["owned_unreaped"] is not False
            or attempt["test_read"] is not False or attempt["scientific_claim"] is not False):
        raise ValueError("only sealed complete-worker B with this exact validator failure qualifies")
    for key in ("protocol_sha256", "stage", "jobs_planned", "jobs_completed", "jobs_results", "headroom_checks",
                "first_gpu_spawn_started_perf_counter", "last_owned_gpu_reap_perf_counter", "started_perf_counter",
                "gpu_phase_elapsed_seconds", "gpu_hours_charged", "billing_scope"):
        if attempt[key] != execution[key]:
            raise ValueError(f"sealed source execution/attempt mismatch: {key}")
    first, last, started = (attempt[key] for key in ("first_gpu_spawn_started_perf_counter", "last_owned_gpu_reap_perf_counter", "started_perf_counter"))
    for item in (attempt, execution):
        ended = item["ended_perf_counter"]
        if (not started <= first <= last <= ended < started + original["hard_cap_seconds"]
                or not math.isclose(item["whole_elapsed_seconds"], ended - started, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(item["gpu_phase_elapsed_seconds"], last - first, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(item["gpu_hours_charged"], (last - first) / 3600, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(item["soft_overrun_seconds"], max(0., ended - started - original["planned_seconds"]), rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("source failed continuous/whole costs or overrun mismatch")
    verify_owned_timing(source, original, attempt, check=check)
    inventory = []
    for job in jobs:
        check()
        lead = "" if job["lead"] is None else f"_lead{job['lead']:03d}h"
        name = f"{job['phase']}_seed{job['seed']}_{job['arm']}{lead}_k{job['reasoning_steps']}"
        receipt_path = source / "workers" / (name + ".json")
        receipt = read_json(receipt_path, check=check)
        if (receipt["status"] != "success" or receipt["job"] != job or receipt["protocol_sha256"] != expected_protocol
                or receipt["scientific_claim"] is not False or receipt["test_read"] is not False):
            raise ValueError("every original exact worker must qualify; no failed/skipped/partial complement")
        inventory.append({"job": job, "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path, check=check)}, "qualified": True})
    return original, attempt, execution, inventory


def verify_owned_timing(source, original, attempt, *, check=lambda: None):
    """Accept complete owned timings under FAILED source status, never fabricate results-complete."""
    jobs = original["jobs"]
    if ([item["job"] for item in attempt["jobs_results"]] != jobs
            or [item["job"] for item in attempt["headroom_checks"]] != jobs):
        raise ValueError("failed source requires all exact result and pre-spawn headroom receipts")
    previous = attempt["first_gpu_spawn_started_perf_counter"]
    observed = 0.
    for index, (job, result, headroom) in enumerate(zip(jobs, attempt["jobs_results"], attempt["headroom_checks"])):
        check()
        lead = "" if job["lead"] is None else f"_lead{job['lead']:03d}h"
        key = f"{job['phase']}_seed{job['seed']}_{job['arm']}{lead}_k{job['reasoning_steps']}"
        receipt_path = path(Path(source) / "workers" / (key + ".json"))
        receipt = read_json(receipt_path, check=check)
        timing = read_json(Path(source) / "workers" / (key + ".timing.json"), check=check)
        snapshot = headroom["snapshot"]
        gpu = original["gpu"]
        required = max(gpu["estimated_peak_mib"], math.ceil(observed)) + gpu["headroom_margin_mib"]
        if (path(result["result"]) != receipt_path or result["elapsed_seconds"] != receipt["elapsed_seconds"]
                or result["peak_reserved_bytes"] != receipt["peak_reserved_bytes"]
                or snapshot["uuid"] != gpu["uuid"] or snapshot["read_only"] is not True
                or snapshot["required_free_mib"] != required or snapshot["free_mib"] < required
                or snapshot["observed_owned_peak_mib"] != observed
                or timing["job"] != job or timing["status"] != "success" or timing["returncode"] != 0
                or timing["protocol_sha256"] != original["protocol_sha256"] or timing["headroom"] != snapshot
                or timing["cleanup"] != "already-exited" or timing["scientific_claim"] is not False
                or timing["test_read"] is not False or not timing["limitations"]):
            raise ValueError("original result/headroom/owned timing identity mismatch")
        spawned, reaped = timing["spawned_perf_counter"], timing["last_owned_reap_perf_counter"]
        if (not previous <= spawned <= reaped <= attempt["last_owned_gpu_reap_perf_counter"]
                or (index == 0 and spawned != previous) or timing["ended_perf_counter"] < reaped):
            raise ValueError("original continuous first-spawn to last-reap timing mismatch")
        previous = reaped
        observed = max(observed, receipt["peak_reserved_bytes"] / 2 ** 20)


def corrected_source(original_bytes):
    text = original_bytes.decode("utf-8")
    if text.count(OLD_OBJECTIVE) != 1:
        raise ValueError("archived validator is not the exact diagnosed implementation")
    return text.replace(OLD_OBJECTIVE, NEW_OBJECTIVE).encode("utf-8")


def archive_members(archive, expected, *, check=lambda: None):
    check()
    members = archive.infolist()
    if (not 0 < len(members) <= MAX_INVENTORY_ENTRIES or len(members) != len(expected)
            or {item.filename for item in members} != set(expected)):
        raise ValueError("exact bounded archived source inventory required")
    total = 0
    for item in members:
        check()
        total += item.file_size
        if item.file_size > MAX_SOURCE_BYTES or total > MAX_ARCHIVE_BYTES:
            raise ValueError("archive member/expanded byte cap exceeded")
    return members


def archive_content(archive, item, root, *, check=lambda: None):
    check()
    relative_file(root, item.filename)
    if (item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
            or any(part in ("tests", "outputs", "manifests") or "legacy" in part for part in PurePosixPath(item.filename).parts)
            or item.file_size > MAX_SOURCE_BYTES):
        raise ValueError("unsafe archive symlink/special/protected/oversize member")
    data = bytearray()
    with archive.open(item) as stream:
        while True:
            check()
            block = stream.read(min(1024 * 1024, MAX_SOURCE_BYTES + 1 - len(data)))
            check()
            if not block:
                break
            data.extend(block)
            if len(data) > MAX_SOURCE_BYTES:
                raise ValueError("expanded archive byte cap exceeded")
    return bytes(data)


def verify_archive(source, original, *, check=lambda: None):
    check()
    code, archive_path = original["code"], path(Path(source) / "code.zip")
    if (archive_path.stat().st_size > MAX_ARCHIVE_BYTES or digest(code["files"]) != code["source_tree_sha256"]
            or sha256_file(archive_path, check=check) != code["code_zip_sha256"]):
        raise ValueError("original code tree/archive hash mismatch or byte cap")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive, code["files"], check=check)
        model, validator = hashlib.sha256(), None
        for item in sorted(members, key=lambda value: value.filename):
            content = archive_content(archive, item, source, check=check)
            if hashlib.sha256(content).hexdigest() != code["files"][item.filename]:
                raise ValueError("archived source member bytes changed")
            if item.filename == CORRECTED:
                validator = content
            if item.filename.startswith("model/") and item.filename.endswith(".py"):
                model.update(item.filename.removeprefix("model/").encode() + b"\0")
                model.update(content)
        if model.hexdigest() != code["model_code_sha256"] or validator is None:
            raise ValueError("original archived model/validator implementation digest mismatch")
        check()
        return validator


def extract_corrected(source, original, target, correction, *, check=lambda: None):
    check()
    target = path(target)
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path(Path(source) / "code.zip")) as archive:
        for item in archive_members(archive, original["code"]["files"], check=check):
            content = archive_content(archive, item, target, check=check)
            if hashlib.sha256(content).hexdigest() != original["code"]["files"][item.filename]:
                raise ValueError("source archive changed during extraction")
            requested = relative_file(target, item.filename)
            requested.parent.mkdir(parents=True, exist_ok=True)
            with requested.open("xb") as stream:
                stream.write(content)
            check()
    corrected = corrected_source(bounded_bytes(target / CORRECTED, limit=MAX_SOURCE_BYTES, check=check))
    if hashlib.sha256(corrected).hexdigest() != correction["corrected_validator_sha256"]:
        raise ValueError("correction contains changes outside exact FP32 receipt reconstruction")
    (target / CORRECTED).write_bytes(corrected)
    helper = bounded_bytes(correction["helper_path"], limit=MAX_SOURCE_BYTES, check=check)
    if hashlib.sha256(helper).hexdigest() != correction["helper_sha256"]:
        raise ValueError("frozen FP32 reconstruction helper changed")
    with (target / OBJECTIVE_HELPER).open("xb") as stream:
        stream.write(helper)
    check()
    return target
