"""New C-package engineering probe: prepare/run/all, not B results or 1600-update retraining.

No historical checkpoint import. Two fresh Process CUDA workers (FP32/BF16),
each with independent uninterrupted2 and intentional-save1/resume-to2 paths.
Main owns real execution after source/configuration freeze, full CPU suite and CI.
"""
from __future__ import annotations

import time
CLI_ENTRY_PERF_COUNTER = time.perf_counter()

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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/r7_v2_package_precision_probe_20261003_attempt01"
PLANNED_SECONDS, HARD_CAP_SECONDS, CLEANUP_SECONDS = 900., 1800., 10.
PRECISIONS, SEEDS, ARMS = ("fp32", "bf16"), (41, 42, 43), ("old_ours", "process", "matched_generic")
BASE_MODEL = {"architecture": "window", "in_channels": 17, "out_channels": 17, "history_steps": 2,
              "dim": 192, "depth": 4, "heads": 4, "window_size": 4, "patch_size": 2,
              "default_reasoning_steps": 4, "detach_between_steps": False, "dropout": 0.,
              "use_forecast_feedback": True}
PACKAGE_FLAGS = ("known_context_inputs", "draft_query_feedback", "source_position_markers", "source_role_markers",
                 "local_solver_state", "solver_gate_proposal", "solver_state_recurrence", "spacetime_inputs",
                 "positional_process_readout")
CONTROLS = {"seed": 41, "updates_per_path": 2, "paths_per_precision": 2, "batch_size": 2, "reasoning_steps": 4,
            "step_hours": 6, "lambda12": .5, "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "clip": 1.,
            "sampling": "first two of exactly185 valid train windows; repeat for both updates", "native_hw": [65, 65],
            "internal_k_detach": False, "physical_step_detach": False, "objective": "deep_supervised_latitude_area_mse",
            "initial_draft_supervised": True, "final_weight": 2., "loss_dtype": "torch.float32",
            "resume_acceptance": "torch.equal weights, optimizer, RNG, loss sequence; no tolerance",
            "deterministic_algorithms": True, "tf32": False, "calendar_feature_atol": 2e-6}
LIMITATIONS = [
    "New-package engineering only; not B results, not repeated B1600 updates, no forecast-skill/scientific claim.",
    "Seed41 four actual optimizer steps per precision (two per independently accounted path) are not convergence evidence.",
    "Scratch same-anchor initialization; no archived M3 checkpoint, optimizer or historical score is imported.",
    "Only new Process is GPU-probed; mapped Generic/old pooled eligibility is CPU engineering, not C forecast acceptance.",
    "Intentional checkpoint-body restore, not fine_tune interrupted orchestration or failed-attempt resurrection.",
    "Exact resume is same-software/device; no cross-platform or across-precision numerical equivalence claim.",
    "Data pins bind manifest/metadata/norms and audited source bytes, not every atmospheric chunk.",
    "Shared GPU wall time includes startup/gaps/failures/owned cleanup; no neighbor signals or private cache clearing.",
    "Negative/failed engineering results remain; they are not fixed-weather negative forecast conclusions.",
    "Socket denial guards accidental connections, not an operating-system network sandbox.",
]


class BudgetLimited(TimeoutError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deny_network():
    def denied(*_args, **_kwargs):
        raise RuntimeError("offline package precision probe: connection denied")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied


def local_path(value):
    path = Path(value).absolute()
    require("://" not in str(value) and path.name != "test.jsonl", "local non-test path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink ancestor forbidden")
    return path.resolve()


def output_path(value, inputs=(), *, fresh=False):
    from tools.check_conventions import ARCHIVAL_PREFIXES
    path = local_path(value)
    prefixes = (*ARCHIVAL_PREFIXES, "data/raw", "data/interim", "data/processed")
    require(not any(path.is_relative_to(ROOT / p) or (ROOT / p).is_relative_to(path) for p in prefixes), "protected output forbidden")
    require(not any("legacy" in p or p in ("raw", "interim", "processed") for p in path.parts), "protected output token")
    require(path != ROOT and not ROOT.is_relative_to(path), "output must not contain source root")
    require(not any(path.is_relative_to(local_path(p)) or local_path(p).is_relative_to(path) for p in inputs), "output overlaps input")
    if path.is_relative_to(ROOT):
        require(path.is_relative_to(ROOT / "outputs") and
                re.fullmatch(r"r7_v2_package_precision_probe_\d{8}_attempt\d{2,}", path.relative_to(ROOT / "outputs").parts[0]),
                "only independent named package outputs allowed")
    for parent in path.parents:
        require(not any((parent / n).exists() for n in ("protocol.json", "attempt.json", "run_started.json")), "nested old attempt output refused")
    if fresh and path.exists():
        raise FileExistsError("new exclusive attempt/path required")
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


def read_json(value):
    result = json.loads(local_path(value).read_text(encoding="utf-8"))
    digest(result)
    return result


def write_json(value, payload):
    with local_path(value).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def publish(value, payload, original=None):
    try:
        write_json(value, payload)
    except BaseException as exc:
        if original is None:
            raise
        original.add_note(f"receipt publication failed: {value}: {exc}")


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def check_budget(deadline, clock=time.perf_counter, reserve=CLEANUP_SECONDS):
    require(type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")
    if clock() >= deadline - reserve:
        raise BudgetLimited("whole-round hard cap exhausted; retain owned cleanup reserve")


def validate_configuration(configuration):
    specs, initialization = configuration["model_specs"], configuration["initialization"]
    require(set(specs) == set(initialization["mapping"]) == set(ARMS), "three explicit complete C mappings required")
    require(initialization["anchor"] == specs["process"], "scratch full Process must be the same anchor")
    for arm, spec in specs.items():
        require(spec["kind"] == ("generic" if arm == "matched_generic" else "process"), "declared C model kind differs")
        model = spec["model"]
        require(all(model.get(k) == v and type(model.get(k)) is type(v) for k, v in BASE_MODEL.items()), "frozen C capacity/full-BPTT differs")
        enabled = arm != "old_ours"
        require(all(model.get(k, True if k in ("solver_gate_proposal", "solver_state_recurrence") else None) is
                    (enabled or k in ("solver_gate_proposal", "solver_state_recurrence")) for k in PACKAGE_FLAGS),
                "strict public package/old-pooled flags differ (old local subflags remain inert defaults)")
        require(model.get("pooled_readout_query", False) is False and model.get("spacetime_field_mode", "fields") == "fields"
                and model.get("activation_checkpointing", False) is False and model.get("spatial_solver_feedback", False) is False
                and model.get("periodic_width", False) is False,
                "unprobed configuration branch forbidden")
        tokens = model.get("latent_tokens") if arm == "matched_generic" else model.get("anchored_processes", 0) + model.get("free_processes", 0)
        require(tokens == 16 and bool(initialization["mapping"][arm]), "16 tokens and explicit mapping required")
    return configuration


def qualify_configuration(path, check):
    from training.r7_v2_profile import seeded_mapped_model
    configuration = validate_configuration(read_json(path))
    reports = {}
    for seed in SEEDS:
        reports[str(seed)] = {}
        for arm in ARMS:
            check()
            model, report, spec = seeded_mapped_model(configuration, seed, arm)
            require(report["uninitialized_target_keys"] == [] and report["target"]["model"] == spec, "complete same-anchor target required")
            reports[str(seed)][arm] = report
            del model
        require(len({r["anchor_state_sha256"] for r in reports[str(seed)].values()}) == 1, "seed arms use different anchors")
    check()
    return {"path": str(local_path(path)), "file_sha256": sha256_file(path, check), "value": configuration, "mapping_reports": reports}


def qualify_inputs(trainmanifest, sidecar, check):
    from training.r7_experiment import dataset_identity
    from training.r7_m3_identity import source_identity
    from data.preprocess.r7_process_scale_sidecar import load_process_scale_sidecar
    from data.r7_autoregressive_dataset import preflight_training_windows, ZarrAutoregressiveDataset
    manifest, sidecar = local_path(trainmanifest), local_path(sidecar)
    require(manifest.name == "train.jsonl", "explicit train manifest required")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        local_path(manifest.parent / record["store_path"])
    preflight = read_json(manifest.parent / "source_preflight.json")
    local_path(preflight["source_path"])
    local_path(manifest.parent.parent.parent / "source_receipt.json")
    data_identity, reader = dataset_identity(manifest)
    root = reader._store(reader.records[0])
    sources = source_identity(manifest.parent, root)
    metadata = load_process_scale_sidecar(sidecar)
    require(metadata["store"] == str(local_path(manifest.parent / records[0]["store_path"]))
            and metadata["train_sample_ids"] == [r["sample_id"] for r in records]
            and metadata["source_identity"]["path"] == sources["source_path"]
            and metadata["source_identity"]["bytes"] == sources["source_bytes"]
            and metadata["source_identity"]["source_preflight_sha256"] == sources["preflight_report_sha256"], "explicit sidecar source/train ownership differs")
    require(metadata["data_identity"] == data_identity and metadata["train_manifest"] == str(manifest)
            and metadata["source_identity"]["sha256"] == sources["source_sha256"]
            and metadata["train_manifest_sha256"] == sources["train_manifest_sha256"], "source/data/sidecar preflight identity mismatch")
    channels, units = list(root.attrs["channels"]), list(root.attrs["units"])
    require(len(channels) == len(set(channels)) == len(units) == 17 and list(root["state"].shape[1:]) == [17, 65, 65], "real native65 all17 channels required")
    windows = preflight_training_windows(manifest)
    require(windows["usable_windows"] == 185 and windows["state_fields_read"] is False and windows["test_read"] is False
            and all(e["reason"] == "t12_outside_train_split" for e in windows["exclusions"]), "185 exact train windows and frozen boundary exclusions required")
    dataset = ZarrAutoregressiveDataset(manifest, expected_exclusions=windows["excluded_sample_ids"])
    selected = [{"sample_id": dataset.records[i]["sample_id"], "init_time": dataset.records[i]["init_time"],
                 "history_indices": dataset.records[i]["history_indices"], "target_indices": [dataset.records[i]["target_index"], dataset.windows[i][1]["target_index"]]} for i in range(2)]
    check()
    return {"trainmanifest": str(manifest), "sidecar": str(sidecar), "data_identity": data_identity, "sources": sources,
            "channels": channels, "units": units, "windows": windows, "selected_windows": selected,
            "sidecar_identity": metadata["sidecar_identity"], "sidecar_sha256": sha256_file(sidecar, check),
            "sidecar_marker_sha256": sha256_file(sidecar.parent / "BUILD_COMPLETE.json", check)}, dataset


def verify_inputs(protocol, check):
    actual, dataset = qualify_inputs(protocol["inputs"]["trainmanifest"], protocol["inputs"]["sidecar"], check)
    require(actual == protocol["inputs"], "frozen real-source/data/sidecar/windows changed")
    configuration = qualify_configuration(protocol["configuration"]["path"], check)
    require(configuration == protocol["configuration"], "frozen C configuration/mapped anchor changed")
    return dataset


def source_paths():
    pending = [Path(__file__), ROOT / "tests/test_r7_v2_package_precision_probe.py"]
    pending.extend(p for p in (ROOT / "model").rglob("*.py") if "legacy" not in str(p.relative_to(ROOT)))
    seen = set()
    while pending:
        path = local_path(pending.pop())
        if path in seen:
            continue
        relative = path.relative_to(ROOT)
        require(path.is_file() and not any("legacy" in p or p in ("outputs", "manifests") for p in relative.parts), "active code-only source required")
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
                if name and name.split(".")[0] in ("model", "training", "data", "tools"):
                    base = ROOT.joinpath(*name.split("."))
                    pending.extend(p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file())
    return sorted(p.relative_to(ROOT).as_posix() for p in seen) + [n for n in
        ("pyproject.toml", "requirements.txt", "requirements-r7-data.txt", "requirements-dev.txt") if (ROOT / n).is_file()]


def archive_code(output, check):
    from training.r7_experiment import model_code_digest
    files = {}
    with zipfile.ZipFile(output / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_paths():
            check()
            content = local_path(ROOT / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True, timeout=5)
    for name, content in (("code_commit.txt", commit + "\n"), ("code_status.txt", status)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return {"base_commit": commit, "source_root": str(ROOT), "files": files, "source_tree_sha256": digest(files),
            "model_code_sha256": model_code_digest(), "artifacts": {n: sha256_file(output / n, check) for n in
            ("code.zip", "code_commit.txt", "code_status.txt")}, "scope": "exact active package closure/config/own tests; no weather/outputs/legacy"}


def verify_code(protocol, check):
    from training.r7_experiment import model_code_digest
    code, output = protocol["code"], local_path(protocol["output"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    require(commit == code["base_commit"] and code["source_root"] == str(ROOT), "runtime exact HEAD/source root differs")
    require(code["source_tree_sha256"] == digest(code["files"]) and set(source_paths()) == set(code["files"]), "code closure identity changed")
    for name, expected in code["artifacts"].items():
        require(sha256_file(output / name, check) == expected, "frozen code artifact changed")
    with zipfile.ZipFile(output / "code.zip") as archive:
        require(sorted(archive.namelist()) == sorted(code["files"]), "source archive inventory changed")
        for name, expected in code["files"].items():
            require(sha256_file(ROOT / name, check) == expected and hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    "active package code changed: " + name)
    require(model_code_digest() == code["model_code_sha256"], "current model digest differs")


def verify_protocol(value):
    p = read_json(value)
    require(p.get("protocol_sha256") == digest({k: v for k, v in p.items() if k != "protocol_sha256"}), "protocol digest changed")
    require(p.get("format") == "r7-v2-package-precision-v1" and p.get("scientific_claim") is False
            and p.get("limitations") == LIMITATIONS and p.get("test_read") is False and p.get("frozen_before_any_step") is True
            and p.get("controls") == CONTROLS and p.get("precisions") == list(PRECISIONS)
            and p.get("planned_seconds") == 900. and p.get("hard_cap_seconds") == 1800., "frozen package protocol differs")
    require(local_path(value) == output_path(p["output"], [Path(p["inputs"]["trainmanifest"]).parent,
            Path(p["inputs"]["sidecar"]).parent, Path(p["configuration"]["path"])]) / "protocol.json", "protocol/output/input overlap")
    require(type(p["round_started_perf_counter"]) in (int, float) and math.isfinite(p["round_started_perf_counter"])
            and 0 <= p["round_started_perf_counter"] <= time.perf_counter() and p["monotonic_boot_id"] == boot_id(), "same-boot prepare clock required")
    require(p["gpu"] == {"policy": "shared", "uuid": p["gpu"]["uuid"], "estimated_peak_mib": 4096, "headroom_margin_mib": 2048}
            and re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", p["gpu"]["uuid"]), "bound UUID/4096+2048 shared headroom required")
    validate_configuration(p["configuration"]["value"])
    for pin in (p["code"]["model_code_sha256"], p["configuration"]["file_sha256"], p["inputs"]["data_identity"],
                p["inputs"]["sources"]["source_sha256"], p["inputs"]["sidecar_sha256"]):
        require(isinstance(pin, str) and re.fullmatch("[0-9a-f]{64}", pin), "complete new package identities required")
    return p


def costs(started, first=None, last=None, clock=time.perf_counter):
    now = clock()
    whole, gpu = max(0., now - started), 0. if first is None else max(0., (now if last is None else last) - first)
    return {"whole_elapsed_seconds": whole, "gpu_phase_elapsed_seconds": gpu, "gpu_hours_charged": gpu / 3600.,
            "soft_overrun_seconds": max(0., whole - 900.), "ended_perf_counter": now}


def prepare(configuration, trainmanifest, sidecar, output=DEFAULT_OUTPUT, *, gpu_uuid, started_perf_counter=None, clock=time.perf_counter):
    started = clock() if started_perf_counter is None else started_perf_counter
    require(type(started) in (int, float) and math.isfinite(started) and 0 <= started <= clock(), "prepare-entry anchor required")
    deny_network()
    output = output_path(output, [Path(trainmanifest).parent, Path(sidecar).parent, configuration], fresh=True)
    output.mkdir(parents=True, exist_ok=False)
    check = lambda: check_budget(started + 1800., clock)
    try:
        from tools.r7_v2_package_precision_support import require_source_runtime
        require_source_runtime(sys.modules[__name__])
        anchor = {"round_started_perf_counter": started, "monotonic_boot_id": boot_id(), "planned_seconds": 900., "hard_cap_seconds": 1800.}
        write_json(output / "prepare_entry.json", {**anchor, "scientific_claim": False, "limitations": LIMITATIONS})
        check()
        configuration = qualify_configuration(configuration, check)
        inputs, _ = qualify_inputs(trainmanifest, sidecar, check)  # Metadata only; no __getitem__.
        code = archive_code(output, check)
        body = {"format": "r7-v2-package-precision-v1", "scientific_claim": False, "limitations": LIMITATIONS,
                "test_read": False, "frozen_before_any_step": True, "output": str(output), **anchor, "configuration": configuration,
                "inputs": inputs, "code": code, "controls": CONTROLS, "precisions": list(PRECISIONS),
                "gpu": {"policy": "shared", "uuid": gpu_uuid, "estimated_peak_mib": 4096, "headroom_margin_mib": 2048},
                "billing": "whole prepare entry including imports/pins/freeze/gaps/startup/closeout; GPU first spawn through last owned reap"}
        write_json(output / "protocol.json", {**body, "protocol_sha256": digest(body)})
        protocol = verify_protocol(output / "protocol.json")
        check()
        write_json(output / "prepare_attempt.json", {"status": "prepared-not-run", "scientific_claim": False,
                   "limitations": LIMITATIONS, **costs(started, clock=clock)})
        check()
        return protocol
    except BaseException as exc:
        publish(output / "attempt.json", {"status": "failed", "phase": "prepare", "scientific_claim": False,
                "limitations": LIMITATIONS, "failure_reason": f"{type(exc).__name__}: {exc}",
                "budget_limited": isinstance(exc, BudgetLimited), **costs(started, clock=clock)}, exc)
        raise


def worker_command(protocol, precision, deadline):
    return [str(ROOT / ".venv/bin/python"), "-I", "-B", str(ROOT / "tools/r7_v2_package_precision_probe.py"), "run",
            "--protocol", str(Path(protocol["output"]) / "protocol.json"), "--worker", precision, "--deadline", repr(deadline)]


def run(output=DEFAULT_OUTPUT, *, snapshot_fn=None, popen_factory=None, clock=time.perf_counter):
    deny_network()
    output = output_path(output)
    for name in ("run_started.json", "attempt.json", "workers", "closeout.json", "publication_failure.json"):
        require(not local_path(output / name).exists(), "attempt already started/failed; no resurrection")
    started, first, last = clock(), None, None
    write_json(output / "run_started.json", {"scientific_claim": False, "limitations": LIMITATIONS, "run_entry_perf_counter": started})
    check = lambda: check_budget(started + 1800., clock)
    popen_factory = popen_factory or subprocess.Popen
    result = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
              "protocol_sha256": None, "workers": [], "budget_limited": False, "finalized": False}
    error, peak = None, 0.
    try:
        from training.r7_m3_driver import gpu_snapshot, verify_headroom, reap_owned, worker_environment
        from tools.r7_v2_package_precision_support import validate_worker_result, require_source_runtime
        snapshot_fn = snapshot_fn or gpu_snapshot
        require_source_runtime(sys.modules[__name__])
        raw = read_json(output / "protocol.json")
        anchor = raw.get("round_started_perf_counter")
        if type(anchor) in (int, float) and math.isfinite(anchor) and 0 <= anchor <= started:
            started = anchor
        p = verify_protocol(output / "protocol.json")
        result["protocol_sha256"] = p["protocol_sha256"]
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
                with (output / "workers" / (precision + ".log")).open("x", encoding="utf-8") as log:
                    spawned = clock()
                    process = popen_factory(worker_command(p, precision, started + 1800. - 10.), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    first = spawned if first is None else first
                    timing.update(owned_pid=process.pid, spawned_perf_counter=spawned)
                    try:
                        code = process.wait(timeout=max(0., started + 1800. - 10. - clock()))
                    except subprocess.TimeoutExpired as exc:
                        raise BudgetLimited("owned package precision worker timeout; no retry") from exc
                    require(code == 0, f"{precision} worker failed with exit {code}")
                    check()
                receipt = read_json(output / "workers" / (precision + ".json"))
                validate_worker_result(receipt, p, precision, sys.modules[__name__])
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
                        timing["cleanup"] = reap_owned(process, started + 1800., clock=clock)
                    except BaseException as exc:
                        if own_error is None:
                            raise
                        own_error.add_note("owned cleanup failed/unreaped: " + str(exc))
                    finally:
                        last = clock()
                publish(output / "workers" / (precision + ".timing.json"), {**timing, "ended_perf_counter": clock()}, own_error)
        require(result["workers"][0]["input_tensor_sha256"] == result["workers"][1]["input_tensor_sha256"], "precision workers read different inputs")
        verify_code(p, check)
        verify_inputs(p, check)
        check()
        result.update(status="success", finalized=True)
    except BaseException as exc:
        error = exc
        result.update(failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, BudgetLimited))
    finally:
        result.update(costs(started, first, last, clock), cleanup_diagnostics=list(getattr(error, "__notes__", [])))
        if result["whole_elapsed_seconds"] >= 1800.:
            error = error or BudgetLimited("hard cap exceeded during closeout")
            result.update(status="failed", finalized=False, budget_limited=True)
        try:
            write_json(output / "attempt.json", result)
            attempt_hash = sha256_file(output / "attempt.json")
            try:
                check()
            except BudgetLimited as exc:
                error = error or exc
            closing = {**costs(started, first, last, clock), "status": "failed" if error else result["status"],
                       "scientific_claim": False, "finalized": error is None, "limitations": LIMITATIONS,
                       "attempt_sha256": attempt_hash, "failure_reason": None if error is None else str(error),
                       "budget_limited": isinstance(error, BudgetLimited),
                       "cost_scope": "all hashes/publication/cleanup before authoritative closeout; serialization follows snapshot"}
            publish(output / "closeout.json", closing, error)
            try:
                check_budget(started + HARD_CAP_SECONDS, clock, reserve=0)
            except BudgetLimited as exc:
                error = error or exc
                publish(output / "publication_failure.json", {**costs(started, first, last, clock),
                        "status": "failed", "finalized": False, "budget_limited": True,
                        "scientific_claim": False, "limitations": LIMITATIONS,
                        "protocol_sha256": result["protocol_sha256"], "failure_reason": str(exc),
                        "invalidates": ["attempt.json", "closeout.json"],
                        "authority": "terminal post-publication failure; consumers must reject candidate receipts"}, error)
        except BaseException as exc:
            if error is None:
                raise
            error.add_note("attempt/closeout publication failed: " + str(exc))
    if error is not None:
        raise error
    return result


def read_closeout(output):
    """Consumers must use this gate, not a candidate success receipt alone."""
    output = output_path(output)
    marker = local_path(output / "publication_failure.json")
    require(not marker.exists(), "terminal publication failure invalidates candidate closeout")
    return read_json(output / "closeout.json")


def fresh_cuda(torch, uuid, precision):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "worker UUID differs; no fallback")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA unavailable/topology differs; no CPU fallback")
    torch.cuda.set_device(0)
    torch.cuda.init()
    baseline = {"baseline_allocated_bytes": torch.cuda.memory_allocated(0), "baseline_reserved_bytes": torch.cuda.memory_reserved(0)}
    require(baseline == {"baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0}, "fresh allocator must be0/0; no private clear")
    require(precision != "bf16" or torch.cuda.is_bf16_supported(), "BF16 unavailable; no FP32 fallback")
    torch.cuda.reset_peak_memory_stats(0)
    return baseline


def worker(protocol_path, precision, deadline):
    deny_network()
    import torch
    from torch.utils.data import default_collate
    from training.r7_v2_profile import seeded_mapped_model, state_hash
    from tools.r7_v2_package_precision_support import resume_paths, diagnostic_checks, tensor_sha256, require_source_runtime
    require_source_runtime(sys.modules[__name__])
    require(Path(sys.prefix).resolve() == (ROOT / ".venv").resolve() and Path(sys.executable).absolute() == ROOT / ".venv/bin/python", "repository .venv Python required")
    p = verify_protocol(protocol_path)
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and precision in PRECISIONS
            and deadline == p["round_started_perf_counter"] + 1800. - 10., "frozen precision/deadline/workspace differs")
    check = lambda: check_budget(deadline, reserve=0)
    output, error = Path(p["output"]), None
    require(read_json(output / "run_started.json").get("scientific_claim") is False
            and read_json(output / "run_started.json").get("protocol_sha256", p["protocol_sha256"]) == p["protocol_sha256"]
            and not (output / "attempt.json").exists(),
            "owned active run marker required; no failed-attempt resurrection")
    started, result = time.perf_counter(), {"status": "failed", "precision": precision, "protocol_sha256": p["protocol_sha256"],
                                          "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False}
    try:
        verify_code(p, check)
        dataset = verify_inputs(p, check)
        result.update(fresh_cuda(torch, p["gpu"]["uuid"], precision))
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, initialization, _ = seeded_mapped_model(p["configuration"]["value"], 41, "process")
        require(initialization == p["configuration"]["mapping_reports"]["41"]["process"], "initial mapped scratch anchor changed")
        result["initial_state_sha256"] = state_hash(model.state_dict())
        batch = default_collate([dataset[0], dataset[1]])
        result["input_tensor_sha256"] = {n: tensor_sha256(v) for n, v in batch.items() if torch.is_tensor(v)}
        require(tuple(batch["coarse_history"].shape) == (2, 2, 17, 65, 65) and batch["future_target"].shape == (2, 17, 65, 65), "actual first2 native65/17ch required")
        batch = {n: v.to("cuda:0") if torch.is_tensor(v) else v for n, v in batch.items()}
        model.to("cuda:0").train()
        api = sys.modules[__name__]
        result["branches"], result["resume_exact"] = resume_paths(torch, model, batch, p, precision, deadline, api)
        result["checks"] = diagnostic_checks(torch, model, batch, precision, api)
        torch.cuda.synchronize(0)
        verify_code(p, check)
        verify_inputs(p, check)
        check()
        result.update(status="success", actual_optimizer_updates=4, model_code_sha256=p["code"]["model_code_sha256"], configuration_sha256=p["configuration"]["file_sha256"])
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
    started = CLI_ENTRY_PERF_COUNTER if argv is None else time.perf_counter()
    deny_network()
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "all"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    for name in ("configuration", "trainmanifest", "sidecar", "gpu-uuid", "protocol"):
        parser.add_argument("--" + name)
    parser.add_argument("--worker", choices=PRECISIONS, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        require(args.command == "run" and args.protocol is not None and args.deadline is not None, "internal worker arguments required")
        return worker(args.protocol, args.worker, args.deadline)
    if args.command in ("prepare", "all"):
        require(all((args.configuration, args.trainmanifest, args.sidecar, args.gpu_uuid)), "C configuration and existing data inputs required")
        prepare(args.configuration, args.trainmanifest, args.sidecar, args.output, gpu_uuid=args.gpu_uuid, started_perf_counter=started)
    return run(args.output) if args.command in ("run", "all") else None


if __name__ == "__main__":
    main()
