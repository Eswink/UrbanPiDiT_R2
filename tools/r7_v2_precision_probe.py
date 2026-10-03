"""Offline engineering probe; prepare/run/all, never a forecast-skill acceptance.
Only run/all launch CUDA. Each precision owns a fresh process and two independently
accounted two-update paths. Resume exercises the real rollout/checkpoint body, NOT
fine_tune's interrupted orchestration. Failed attempts cannot be resurrected.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/r7_v2_precision_probe_20261003_attempt01"
PLANNED_SECONDS, HARD_CAP_SECONDS, CLEANUP_SECONDS = 900., 1800., 10.
PRECISIONS = ("fp32", "bf16")
CONTROLS = {"seed": 41, "updates_per_path": 2, "paths_per_precision": 2,
            "batch_size": 2, "reasoning_steps": 4, "step_hours": 6, "lambda12": .5,
            "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "clip": 1.,
            "sampling": "first two valid train windows, repeated for both updates",
            "internal_k_detach": False, "physical_step_detach": False,
            "objective": "deep_supervised_latitude_area_mse", "internal_deep_supervision": True,
            "initial_draft_supervised": True, "final_weight": 2.,
            "resume_acceptance": "torch.equal weights, losses, optimizer and RNG; no tolerance",
            "deterministic_algorithms": True, "tf32": False, "calendar_phase_atol": 2e-6}
LIMITATIONS = [
    "Engineering only; no forecast-skill, generalization, significance or SOTA claim.",
    "Seed 41 and two updates per path are not convergence evidence or fixed-weather negative conclusions.",
    "Actual checkpoint-body resume, not fine_tune interrupted-orchestration acceptance.",
    "FP32/BF16 are checked independently; no exact numerical equality across precisions is claimed.",
    "Same-software, same-device exact resume is required; no cross-platform bitwise promise.",
    "Data identity pins metadata/norms plus opaque audited source bytes, not every atmospheric chunk.",
    "Shared GPU neighbors affect wall time; no private-cache clearing or neighbor interference.",
    "Socket denial prevents accidental connections; it is not a network sandbox.",
]


class BudgetLimited(TimeoutError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deny_network():
    def denied(*_args, **_kwargs):
        raise RuntimeError("offline precision probe: network connection denied")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def local_path(value):
    path = Path(value).absolute()
    require("://" not in str(value) and path.name != "test.jsonl",
            "local non-test path required")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink ancestor forbidden")
    return path.resolve()


def output_path(value, inputs=(), *, fresh=False):
    from tools.check_conventions import ARCHIVAL_PREFIXES
    path = local_path(value)
    protected = [ROOT / name for name in (*ARCHIVAL_PREFIXES, "data/raw", "data/interim", "data/processed")]
    require(not any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected), "protected output forbidden")
    require(not any(path.is_relative_to(local_path(p)) or local_path(p).is_relative_to(path) for p in inputs),
            "output overlaps parent/data artifacts")
    if fresh and path.exists():
        raise FileExistsError("new exclusive output required; old attempts are immutable")
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(value, check=lambda: None):
    result = hashlib.sha256()
    with local_path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            check()
            result.update(chunk)
    check()
    return result.hexdigest()


def publish(value, payload, original=None):
    try:
        write_json(value, payload)
    except BaseException as exc:
        if original is None:
            raise
        original.add_note(f"receipt publication failed: {value}: {type(exc).__name__}: {exc}")


def read_json(value):
    result = json.loads(local_path(value).read_text(encoding="utf-8"))
    digest(result)  # Reject NaN/Infinity, including nested values.
    return result


def write_json(value, payload):
    with local_path(value).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def check_budget(deadline, clock=time.perf_counter, reserve=CLEANUP_SECONDS):
    require(isinstance(deadline, (int, float)) and not isinstance(deadline, bool) and math.isfinite(deadline),
            "finite deadline required")
    if clock() >= deadline - reserve:
        raise BudgetLimited("whole-round hard cap exhausted; retain owned cleanup reserve")


def source_paths():
    pending = [Path(__file__), ROOT / "tests/test_r7_v2_precision_probe.py"]
    pending.extend(p for p in (ROOT / "model").rglob("*.py") if "legacy" not in str(p.relative_to(ROOT)))
    seen = set()
    while pending:
        path = local_path(pending.pop())
        if path in seen:
            continue
        relative = path.relative_to(ROOT)
        require(path.is_file() and not any("legacy" in part for part in relative.parts), "active source required")
        seen.add(path)
        package = ".".join(relative.with_suffix("").parts[:-1])
        for parent in path.parents:
            if parent == ROOT:
                break
            if (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                target = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
                names = [target] + [(target or "") + "." + a.name for a in node.names]
            for name in names:
                if not name or name.split(".")[0] not in ("training", "data", "model", "tools"):
                    continue
                base = ROOT.joinpath(*name.split("."))
                pending.extend(p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file())
    return sorted(p.relative_to(ROOT).as_posix() for p in seen) + [
        n for n in ("pyproject.toml", "requirements.txt", "requirements-r7-data.txt", "requirements-dev.txt") if (ROOT / n).is_file()]


def archive_code(output, check):
    from training.r7_experiment import model_code_digest
    files = {}
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_paths():
            check()
            content = local_path(ROOT / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    for name, command in (("code_commit.txt", ["git", "rev-parse", "HEAD"]),
                          ("code_status.txt", ["git", "status", "--short"])):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(subprocess.check_output(command, cwd=ROOT, text=True, timeout=5))
    return {"files": files, "source_tree_sha256": digest(files), "model_code_sha256": model_code_digest(),
            "artifacts": {n: sha256_file(output / n, check) for n in ("code.zip", "code_commit.txt", "code_status.txt")},
            "archive_scope": "actual active source import closure, all active model files, own CPU tests and config; dirty bytes included"}


def verify_code(protocol, check):
    from training.r7_experiment import model_code_digest
    code, output = protocol["code"], local_path(protocol["output"])
    require(code["source_tree_sha256"] == digest(code["files"]), "code identity missing or changed")
    for name, expected in code["artifacts"].items():
        require(sha256_file(output / name, check) == expected, "code artifact changed")
    with zipfile.ZipFile(output / "code.zip") as archive:
        require(sorted(archive.namelist()) == sorted(code["files"]), "archive inventory changed")
        for name, expected in code["files"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe code member")
            require(sha256_file(ROOT / name, check) == expected and hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    "active source/archive changed after freeze: " + name)
    require(model_code_digest() == code["model_code_sha256"], "current model digest changed")


def qualify_inputs(inputs, check):
    from training.r7_parent_import import import_parent, M3_PINS
    from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset, preflight_training_windows
    paths = {name: str(local_path(value)) for name, value in inputs.items()}
    require(set(paths) == {"checkpoint", "original_protocol", "codezip", "sidecar", "trainmanifest"}, "all five parent/input identities required")
    pins = {name: sha256_file(value, check) for name, value in paths.items()}
    original = read_json(paths["original_protocol"])
    spec = dict(next(a["model_config"] for a in original["arms"] if a["name"] == "aux_off"), detach_between_steps=False)
    # Guard referenced local paths before the importer opens any store/source.
    manifest = Path(paths["trainmanifest"])
    for record in (json.loads(line) for line in manifest.read_text().splitlines() if line.strip()):
        local_path(manifest.parent / record["store_path"])
    preflight = read_json(manifest.parent / "source_preflight.json")
    source = local_path(preflight["source_path"])
    marker = local_path(manifest.parent / "BUILD_COMPLETE.json")
    require(preflight.get("fingerprint", {}).get("scope") == "full-local-file" and marker.is_file(),
            "existing read-only real-source preflight/publication required")
    local_path(manifest.parent.parent.parent / "source_receipt.json")
    require(sha256_file(source, check) == preflight["fingerprint"]["sha256"], "audited real source bytes changed")
    check()
    model, report = import_parent(**paths, model_spec=spec)
    require(report["parent"]["contract"]["seed"] == 41 and report["data"]["sources"]["source_sha256"] == M3_PINS["source_sha256"], "accepted seed41 real parent required")
    summary = preflight_training_windows(manifest)
    require(all(e["reason"] == "t12_outside_train_split" for e in summary["exclusions"]), "only explicit train-boundary exclusions allowed; missing frames refused")
    dataset = ZarrAutoregressiveDataset(manifest, expected_exclusions=summary["excluded_sample_ids"])
    require(len(dataset) >= 2 and dataset.summary == summary, "two actual valid train windows required")
    selected = [{"sample_id": dataset.records[i]["sample_id"], "init_time": dataset.records[i]["init_time"],
                 "history_indices": dataset.records[i]["history_indices"],
                 "target_indices": [dataset.records[i]["target_index"], dataset.windows[i][1]["target_index"]]} for i in range(2)]
    require(spec["in_channels"] == spec["out_channels"] == 17, "real 17-channel model required")
    check()
    return {"paths": paths, "sha256": pins, "parent_import": report, "model_spec": spec,
            "windows": summary, "selected_windows": selected}, model, dataset


def verify_inputs(protocol, check):
    identity, model, dataset = qualify_inputs(protocol["inputs"]["paths"], check)
    require(identity == protocol["inputs"], "parent/source/sidecar/data/window pins changed")
    require(identity["parent_import"]["initialization_contract"]["model_code_sha256"] == protocol["code"]["model_code_sha256"], "new model/input identity mismatch")
    return model, dataset


def verify_protocol(value):
    p = read_json(value)
    require(p.get("protocol_sha256") == digest({k: v for k, v in p.items() if k != "protocol_sha256"}), "protocol digest changed")
    require(p.get("format") == "r7-v2-precision-probe-v1" and p.get("scientific_claim") is False
            and p.get("limitations") == LIMITATIONS and p.get("test_read") is False
            and p.get("frozen_before_any_step") is True and p.get("controls") == CONTROLS
            and p.get("precisions") == list(PRECISIONS) and p.get("planned_seconds") == PLANNED_SECONDS
            and p.get("hard_cap_seconds") == HARD_CAP_SECONDS, "frozen probe contract changed")
    paths = p["inputs"]["paths"]
    require(set(paths) == set(p["inputs"]["sha256"]) == {"checkpoint", "original_protocol", "codezip", "sidecar", "trainmanifest"},
            "all five parent/input pins required")
    require(local_path(value) == output_path(p["output"], [Path(v).parent for v in paths.values()]) / "protocol.json", "protocol/output mismatch")
    require(p["gpu"]["policy"] == "shared" and re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", p["gpu"]["uuid"])
            and type(p["gpu"]["estimated_peak_mib"]) is int and p["gpu"]["estimated_peak_mib"] > 0
            and p["gpu"]["headroom_margin_mib"] == 2048, "bound shared UUID/headroom required")
    require(type(p["round_started_perf_counter"]) in (int, float) and math.isfinite(p["round_started_perf_counter"])
            and 0 <= p["round_started_perf_counter"] <= time.perf_counter() and p["monotonic_boot_id"] == boot_id(), "same-boot prepare-entry clock required")
    for value in (p["code"]["model_code_sha256"], p["inputs"]["parent_import"]["data"]["data_identity"],
                  p["inputs"]["parent_import"]["data"]["sources"]["source_sha256"], *p["inputs"]["sha256"].values()):
        require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "complete SHA256 identity required")
    return p


def costs(started, first=None, last=None, clock=time.perf_counter):
    now = clock()
    whole, gpu = max(0., now - started), 0. if first is None else max(0., (now if last is None else last) - first)
    return {"whole_elapsed_seconds": whole, "gpu_phase_elapsed_seconds": gpu, "gpu_hours_charged": gpu / 3600.,
            "soft_overrun_seconds": max(0., whole - PLANNED_SECONDS), "ended_perf_counter": now}


def prepare(inputs, output=DEFAULT_OUTPUT, *, gpu_uuid, estimated_peak_mib=4096,
            started_perf_counter=None, clock=time.perf_counter):
    started = clock() if started_perf_counter is None else started_perf_counter
    require(type(started) in (int, float) and math.isfinite(started) and 0 <= started <= clock(), "prepare-entry anchor required")
    deny_network()
    output = output_path(output, [Path(v).absolute().parent for v in inputs.values()], fresh=True)
    output.mkdir(parents=True, exist_ok=False)
    anchor = {"round_started_perf_counter": started, "monotonic_boot_id": boot_id(),
              "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS}
    check = lambda: check_budget(started + HARD_CAP_SECONDS, clock)
    try:
        write_json(output / "prepare_entry.json", {**anchor, "scientific_claim": False, "limitations": LIMITATIONS})
        require(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", gpu_uuid or "") and type(estimated_peak_mib) is int
                and estimated_peak_mib > 0, "bound physical UUID and positive estimate required")
        check()
        identity, _, _ = qualify_inputs(inputs, check)  # Metadata/parent weights only; no __getitem__.
        code = archive_code(output, check)
        body = {"format": "r7-v2-precision-probe-v1", "scientific_claim": False, "limitations": LIMITATIONS,
                "test_read": False, "frozen_before_any_step": True, "output": str(output), **anchor,
                "inputs": identity, "code": code, "controls": CONTROLS, "precisions": list(PRECISIONS),
                "gpu": {"policy": "shared", "uuid": gpu_uuid, "estimated_peak_mib": estimated_peak_mib,
                        "headroom_margin_mib": 2048, "estimate_source": "explicit engineering estimate, not measured",
                        "billing": "first direct CUDA spawn through last owned reap, including gaps/failures/cleanup"},
                "whole_scope": "prepare entry, identity/freeze, prepare-run interval, startup, diagnostics and closeout"}
        write_json(output / "protocol.json", {**body, "protocol_sha256": digest(body)})
        protocol = verify_protocol(output / "protocol.json")
        check()
        write_json(output / "prepare_attempt.json", {"status": "prepared-not-run", "scientific_claim": False,
                   "limitations": LIMITATIONS, **costs(started, clock=clock)})
        return protocol
    except BaseException as exc:
        publish(output / "attempt.json", {"status": "failed", "phase": "prepare", "scientific_claim": False,
                "limitations": LIMITATIONS, "failure_reason": f"{type(exc).__name__}: {exc}",
                "budget_limited": isinstance(exc, BudgetLimited), **costs(started, clock=clock)}, exc)
        raise


def worker_command(protocol, precision, deadline):
    return [str(ROOT / ".venv/bin/python"), "-I", "-B", str(ROOT / "tools/r7_v2_precision_probe.py"), "run",
            "--protocol", str(Path(protocol["output"]) / "protocol.json"), "--worker", precision, "--deadline", repr(deadline)]


def run(output=DEFAULT_OUTPUT, *, snapshot_fn=None, popen_factory=None, clock=time.perf_counter):
    deny_network()
    from training.r7_m3_driver import gpu_snapshot, verify_headroom, reap_owned, worker_environment
    output = output_path(output)
    for name in ("run_started.json", "attempt.json", "workers", "closeout.json"):
        require(not local_path(output / name).exists(), "attempt already started/failed; no resurrection")
    raw = read_json(output / "protocol.json")
    write_json(output / "run_started.json", {"protocol_sha256": raw.get("protocol_sha256"), "scientific_claim": False,
               "limitations": LIMITATIONS, "run_entry_perf_counter": clock()})
    anchor = raw.get("round_started_perf_counter")
    started = anchor if type(anchor) in (int, float) and math.isfinite(anchor) and 0 <= anchor <= clock() else clock()
    first, last = None, None
    deadline = started + HARD_CAP_SECONDS
    check = lambda: check_budget(deadline, clock)
    snapshot_fn, popen_factory = snapshot_fn or gpu_snapshot, popen_factory or subprocess.Popen
    result = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
              "protocol_sha256": raw.get("protocol_sha256"), "workers": [], "budget_limited": False, "finalized": False}
    error, peak = None, 0.
    try:
        p = verify_protocol(output / "protocol.json")
        (output / "workers").mkdir(exist_ok=False)
        verify_code(p, check)
        verify_inputs(p, check)
        for precision in PRECISIONS:
            process, timing, own_error = None, {"precision": precision, "status": "failed"}, None
            try:
                check()
                timing["headroom"] = verify_headroom(snapshot_fn(p["gpu"]["uuid"]), p["gpu"], peak)
                check()
                env = worker_environment(p["gpu"]["uuid"])
                env.pop("PYTHONPATH", None)
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                with (output / "workers" / (precision + ".log")).open("x") as log:
                    spawned = clock()
                    first = spawned if first is None else first
                    process = popen_factory(worker_command(p, precision, deadline - CLEANUP_SECONDS),
                                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    timing.update(owned_pid=process.pid, spawned_perf_counter=spawned)
                    try:
                        code = process.wait(timeout=max(0., deadline - CLEANUP_SECONDS - clock()))
                    except subprocess.TimeoutExpired as exc:
                        raise BudgetLimited("owned precision worker timed out; no retry") from exc
                    require(code == 0, f"{precision} worker failed with exit {code}; stop attempt")
                    check()
                receipt = read_json(output / "workers" / (precision + ".json"))
                validate_worker_result(receipt, p, precision)
                result["workers"].append(receipt)
                peak = max(peak, receipt["peak_reserved_bytes"] / 2 ** 20)
                timing["status"] = "success"
            except BaseException as exc:
                own_error = exc
                timing["failure_reason"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if process is not None:
                    try:
                        timing["cleanup"] = reap_owned(process, deadline, clock=clock)
                    except BaseException as exc:
                        if own_error is None:
                            raise
                        own_error.add_note("owned cleanup failed/unreaped: " + str(exc))
                    finally:
                        last = clock()
                publish(output / "workers" / (precision + ".timing.json"), {**timing, "ended_perf_counter": clock()}, own_error)
        require(result["workers"][0]["input_tensor_sha256"] == result["workers"][1]["input_tensor_sha256"], "precision workers read different actual inputs")
        verify_code(p, check)
        verify_inputs(p, check)
        check()
        result.update(status="success", finalized=True)
    except BaseException as exc:
        error = exc
        result.update(failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, BudgetLimited))
    finally:
        result.update(costs(started, first, last, clock), cleanup_diagnostics=list(getattr(error, "__notes__", [])))
        if result["whole_elapsed_seconds"] >= HARD_CAP_SECONDS:
            error = error or BudgetLimited("whole-round hard cap exceeded during closeout")
            result.update(status="failed", finalized=False, budget_limited=True)
        try:
            write_json(output / "attempt.json", result)
            closing = {**costs(started, first, last, clock), "status": result["status"], "scientific_claim": False,
                       "limitations": LIMITATIONS, "attempt_sha256": sha256_file(output / "attempt.json"),
                       "cost_scope": "includes attempt publication; closeout serialization follows snapshot"}
            if closing["whole_elapsed_seconds"] >= HARD_CAP_SECONDS:
                error = error or BudgetLimited("whole-round hard cap exceeded publishing attempt")
                closing.update(status="failed", budget_limited=True)
            publish(output / "closeout.json", closing, error)
        except BaseException as exc:
            if error is None:
                raise
            error.add_note("attempt/closeout publication failed: " + str(exc))
    if error is not None:
        raise error
    return result


def fresh_cuda(torch, uuid, precision):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "worker UUID environment differs; no fallback")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA unavailable or topology differs; no CPU fallback")
    torch.cuda.set_device(0)
    torch.cuda.init()
    baseline = {"baseline_allocated_bytes": torch.cuda.memory_allocated(0), "baseline_reserved_bytes": torch.cuda.memory_reserved(0)}
    require(baseline == {"baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0}, "fresh allocator baseline must be exactly 0/0; no private clear")
    require(precision != "bf16" or torch.cuda.is_bf16_supported(), "BF16 unavailable; no FP32 fallback")
    torch.cuda.reset_peak_memory_stats(0)
    return baseline


def finite_tensors(torch, values, label):
    for name, value in values.items():
        require(torch.is_tensor(value), label + " missing tensor: " + name)
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), label + " nonfinite: " + name)


def exact_equal(torch, left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_equal(torch, v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def timed_update(torch, model, optimizer, batch, precision, deadline):
    from tools.r7_v2_precision_support import timed_update as update
    return update(torch, model, optimizer, batch, precision, deadline, sys.modules[__name__])


def trace_rollout(torch, model, batch, precision, *, one_step=False):
    from tools.r7_v2_precision_support import trace_rollout as trace
    return trace(torch, model, batch, precision, sys.modules[__name__], one_step=one_step)


def diagnostic_checks(torch, model, batch, precision):
    from tools.r7_v2_precision_support import diagnostic_checks as checks
    return checks(torch, model, batch, precision, sys.modules[__name__])


def resume_paths(torch, model, batch, p, precision, deadline):
    from tools.r7_v2_precision_support import resume_paths as compare
    return compare(torch, model, batch, p, precision, deadline, sys.modules[__name__])


def validate_worker_result(result, protocol, precision):
    require(result.get("model_code_sha256") == protocol["code"]["model_code_sha256"]
            and result.get("actual_optimizer_updates") == 4, "new-model/four-update accounting required")
    checks = result.get("checks", {})
    require(checks.get("poison_forecasts_equal") is True and checks.get("poison_loss_changed") is True
            and checks.get("two_step_forward_calls") == 2 and checks.get("one_step_forward_calls") == 1
            and checks.get("internal_k_calls") == 8 and len(checks.get("calendar_trace", [])) == 2
            and len(checks.get("valid_time_phase_trace", [])) == 2, "complete poison/calendar/K counts required")
    ownership = checks.get("l12_gradient_ownership", {})
    require(set(ownership) == {"first_forecast", "first_encoder", "first_reader", "first_internal_k", "encoder_parameters", "reader_parameters"},
            "full L12-only gradient ownership required")
    for norms in ownership.values():
        require(bool(norms) and all(type(n) in (int, float) and math.isfinite(n) and n > 0 for n in norms), "finite nonzero L12 gradient required")
    require([b.get("branch") for b in result.get("branches", [])] == ["uninterrupted", "intentional_resume"]
            and all(b.get("updates") == len(b.get("losses", [])) == 2 for b in result["branches"]), "both actual two-update branches required")
    require(result.get("status") == "success" and result.get("precision") == precision
            and result.get("protocol_sha256") == protocol["protocol_sha256"] and result.get("scientific_claim") is False
            and result.get("limitations") == LIMITATIONS and result.get("test_read") is False
            and result.get("baseline_allocated_bytes") == result.get("baseline_reserved_bytes") == 0
            and result.get("resume_exact") == {n: True for n in ("weights", "optimizer", "rng", "losses")}
            and result.get("checks", {}).get("full_gradient") is True and result.get("checks", {}).get("finite") is True,
            "complete finite fresh-CUDA engineering receipt required; no skip")
    for name in ("peak_reserved_bytes", "peak_allocated_bytes", "elapsed_seconds"):
        value = result.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite actual memory/timing required")


def worker(protocol_path, precision, deadline):
    deny_network()
    import torch
    from torch.utils.data import default_collate
    from training.r7_parent_import import tensor_sha256
    require(Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
            and Path(sys.executable).absolute() == ROOT / ".venv/bin/python", "repository .venv Python required")
    p = verify_protocol(protocol_path)
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "frozen deterministic CUDA workspace required")
    require(precision in PRECISIONS and deadline == p["round_started_perf_counter"] + HARD_CAP_SECONDS - CLEANUP_SECONDS,
            "worker deadline/precision differs from protocol")
    check = lambda: check_budget(deadline, reserve=0)
    output = Path(p["output"])
    require(read_json(output / "run_started.json")["protocol_sha256"] == p["protocol_sha256"], "owned run marker required")
    error = None
    started, result = time.perf_counter(), {"status": "failed", "precision": precision, "protocol_sha256": p["protocol_sha256"],
                                          "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False}
    try:
        verify_code(p, check)
        model, dataset = verify_inputs(p, check)
        result.update(fresh_cuda(torch, p["gpu"]["uuid"], precision))
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        batch = default_collate([dataset[0], dataset[1]])
        result["input_tensor_sha256"] = {n: tensor_sha256(v) for n, v in batch.items() if torch.is_tensor(v)}
        require(batch["coarse_history"].shape[:3] == (2, 2, 17) and batch["future_target"].shape[1] == 17, "actual first-two 17ch batch required")
        batch = {n: v.to("cuda:0") if torch.is_tensor(v) else v for n, v in batch.items()}
        model.to("cuda:0").train()
        result["branches"], result["resume_exact"] = resume_paths(torch, model, batch, p, precision, deadline)
        result["checks"] = diagnostic_checks(torch, model, batch, precision)
        torch.cuda.synchronize(0)
        verify_code(p, check)
        verify_inputs(p, check)
        check()
        result.update(status="success", actual_optimizer_updates=4, model_code_sha256=p["code"]["model_code_sha256"])
    except BaseException as exc:
        error = exc
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        if "baseline_allocated_bytes" in result:
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        publish(output / "workers" / (precision + ".json"), result, error)
    return result


def main(argv=None):
    started = time.perf_counter()  # Before parsing, project imports, pinning or CUDA startup.
    deny_network()
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "all"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--protocol")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--estimated-peak-mib", type=int, default=4096)
    for name in ("parent-checkpoint", "parent-protocol", "parent-codezip", "sidecar", "trainmanifest"):
        parser.add_argument("--" + name)
    parser.add_argument("--worker", choices=PRECISIONS, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        require(args.command == "run" and args.protocol is not None and args.deadline is not None, "internal worker arguments required")
        return worker(args.protocol, args.worker, args.deadline)
    if args.command in ("prepare", "all"):
        inputs = dict(checkpoint=args.parent_checkpoint, original_protocol=args.parent_protocol, codezip=args.parent_codezip,
                      sidecar=args.sidecar, trainmanifest=args.trainmanifest)
        require(all(inputs.values()) and args.gpu_uuid is not None, "prepare requires parent/source inputs and physical GPU UUID")
        prepare(inputs, args.output, gpu_uuid=args.gpu_uuid, estimated_peak_mib=args.estimated_peak_mib, started_perf_counter=started)
    return run(args.output) if args.command in ("run", "all") else None
if __name__ == "__main__":
    main()
