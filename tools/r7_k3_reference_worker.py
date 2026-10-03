"""Stdlib-only archive worker bootstrap; CUDA evaluation only, no CPU fallback or verify loads."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from r7_k3_reference_support import (
    GPU_UUID, check_deadline, deny_network, job_key, local_path, preserve_error, require, sha256_file, write_json,
)
from r7_k3_reference_identity import assert_origins, isolate_archive, read_protocol, verify_extraction, verify_inputs


def zero_allocator_baseline(torch):
    """Same original M3 worker contract; stdlib bootstrap avoids importing its hardpinned trainer."""
    require(not torch.cuda.is_initialized(), "fresh CUDA process requires pre-init allocator measurement")
    device = torch.device("cuda:0")
    before = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device)}
    require(before == {"allocated_bytes": 0, "reserved_bytes": 0} and not torch.cuda.is_initialized(), "pre-init allocator must be measured0/0")
    torch.cuda.init()
    require(torch.cuda.device_count() == 1, "one bound visible CUDA device only")
    torch.cuda.set_device(device)
    after = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device)}
    require(after == before, "initialized allocator must remain0/0; no clearing or private cache calls")
    torch.cuda.reset_peak_memory_stats(device)
    return {"pre_init": before, "initialized": after, "device": "cuda:0", "pre_init_is_initialized": False,
            "memory_scope": "pre-init0/0, initialize only UUID-bound device without tensors/clearing, initialized0/0, then reset owned peak counters"}


def verify_companion(protocol, root, *, check=lambda: None):
    root = local_path(root)
    files = protocol["code"]["files"]
    require({p.name for p in root.iterdir() if p.is_file()} == set(files), "exact temporary companion files required")
    require(not any(p.is_symlink() or p.is_dir() for p in root.iterdir()), "companion source directory may contain only pinned files")
    for name, expected in files.items():
        require(sha256_file(root / name, check=check) == expected, "temporary companion bytes changed")
    require(sha256_file(root.parent / "companion_code.zip", check=check) == protocol["code"]["companion_zip_sha256"],
            "frozen companion archive bytes changed")
    from r7_k3_reference_support import read_json
    require(read_json(root.parent / "source_identity.json") == protocol["code"], "frozen source identity receipt changed")


def run_worker(protocol_path, job, deadline):
    deny_network()  # Before any torch/project imports; offline bootstrap is stdlib-only.
    check_deadline(deadline)
    protocol = read_protocol(protocol_path)
    require(deadline == protocol["clock"]["earliest_started_perf_counter"] + protocol["hard_cap_seconds"] - protocol["cleanup_reserve_seconds"],
            "worker must consume exact frozen same-boot whole-round hard deadline")
    key = job_key(job)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "exact fixed physical CUDA UUID required; no alternative-device/CPU fallback")
    root = local_path(Path(protocol["output"]) / "archived_code")
    companion = local_path(Path(protocol["output"]) / "companion_code")
    check = lambda: check_deadline(deadline)
    payload = {"status": "failed", "scientific_claim": False, "limitations": protocol["limitations"],
               "test_read": False, "job": job, "protocol_sha256": protocol["protocol_sha256"], "pid": os.getpid(),
               "started_perf_counter": time.perf_counter()}
    receipt = local_path(Path(protocol["output"]) / "workers" / (key + ".json"))
    error = None
    try:
        require(not receipt.exists(), "exclusive worker receipt required; no repeated jobs")
        verify_companion(protocol, companion, check=check)
        require(local_path(__file__).parent == companion, "worker must execute exclusively from frozen temporary companion root")
        verify_extraction(root, protocol["archive"]["files"], check=check)
        verify_inputs(protocol, check=check)
        isolate_archive(root, companion, protocol["repo"])
        import torch
        torch.set_num_threads(4)
        assert_origins(root, companion)
        payload["allocator"] = zero_allocator_baseline(torch)
        check()
        from r7_k3_reference_evaluate import evaluate_job
        report = evaluate_job(protocol, job, deadline)
        torch.cuda.synchronize(0)
        check()
        payload["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        payload["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        assert_origins(root, companion)
        verify_extraction(root, protocol["archive"]["files"], check=check)
        verify_inputs(protocol, check=check)
        verify_companion(protocol, companion, check=check)
        result = Path(protocol["output"]) / "cases" / (key + ".json")
        write_json(result, report)
        check()
        payload.update(status="success", result=str(result), result_sha256=sha256_file(result, check=check),
                       checkpoint_sha256=protocol["parents"][str(job["seed"])]["checkpoint_sha256"],
                       source_sha256=protocol["code"]["source_sha256"], archive_sha256=protocol["archive"]["sha256"],
                       model_code_sha256=protocol["archive"]["model_code_sha256"], data_identity=protocol["data"]["data_identity"],
                       companion_zip_sha256=protocol["code"]["companion_zip_sha256"],
                       import_origin_scope="data/model/training/scripts exclusively archived_code; companion exclusively temporary companion_code")
    except BaseException as exc:
        error = exc
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["ended_perf_counter"] = time.perf_counter()
        try:
            write_json(receipt, payload)
        except BaseException as exc:
            preserve_error(error, exc, "worker receipt publication failed")
        if error is None:
            check()  # Parent rejects nonzero exit if the final worker publication passed its hard deadline.
    return payload
