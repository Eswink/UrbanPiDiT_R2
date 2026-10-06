"""One fresh, externally bounded re-execution of the S3 v3 rollout recipe."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_ft_worker import (
    GPU_UUID, HARD_CAP_SECONDS, PER_SEED_SECONDS, PLANNED_SECONDS, write_exclusive,
)
from training.r7_study_process import bounded_process, worker_environment

DEFAULT_OUT = ROOT / "outputs/r7_s3_v3_rollout_ft_20261006_attempt02"
WORKER = ROOT / "scripts/study_r7_s3_v3_rollout_ft_worker.py"


def _command(phase, output, deadline, seed=None):
    arguments = [sys.executable, str(WORKER), "--phase", phase, "--out", str(output),
                 "--deadline", str(deadline)]
    if seed is not None:
        arguments += ["--seed", str(seed)]
    return arguments


def _cpu_spawn_gate(output, phase):
    gate = recipe._shared().gpu_gate(GPU_UUID, estimated_peak_bytes=0,
                                     margin_bytes=recipe.HEADROOM_MARGIN_BYTES)
    write_exclusive(output / f"{phase}_spawn_gate.json", {"phase": phase, "cpu_only": True, **gate})


def _collect_receipts(output, state, deadline, environment, processes):
    _cpu_spawn_gate(output, "reading")
    process = bounded_process(_command("reading", output, deadline), cwd=ROOT,
                              log_path=output / "reading.log", deadline=deadline, env=environment)
    processes.append({"phase": "reading", **process})
    write_exclusive(output / "reading_process.json", processes[-1])
    if process["status"] != "success":
        raise RuntimeError(f"paired reading did not succeed: {process['status']}")
    readings = json.loads((output / "readings.json").read_text(encoding="utf-8"))
    state["result"].update(readings)


def fresh_output(output):
    requested = Path(output).absolute()
    if ".." in requested.parts or any(path.is_symlink() for path in (requested, *requested.parents)):
        raise ValueError("symlink/traversal output forbidden")
    for prefix in ("data/raw", "data/interim", "data/processed", "legacy_v531_full",
                   "data/legacy_v531", "model/legacy_v531", "legacy_v6"):
        if requested.is_relative_to(ROOT / prefix):
            raise ValueError("protected output forbidden")
    if requested.is_relative_to(ROOT) and requested.parent != ROOT / "outputs":
        raise ValueError("repository attempt must be a fresh direct outputs child")
    if requested.exists():
        raise FileExistsError(f"fresh output only: {requested}")
    for parent in requested.parents:
        if any((parent / name).exists() for name in
               ("protocol.json", "failure.json", "attempt.json", "fine_tune_started.json")):
            raise ValueError("cannot nest an attempt inside old evidence")
    return requested


def run_attempt(output):
    shared = recipe._shared()
    from training.r7_experiment import canonical_digest
    output = fresh_output(output)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + HARD_CAP_SECONDS
    state = {"protocol": None, "result": {"format": "r7-s3-v3-rollout-isolated-result-v1",
                                          "scientific_claim": False, "test_read": False,
                                          "seeds": {}, "gpu_gates": []}}
    processes = []
    write_exclusive(output / "execution_start.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "started_perf_counter": started,
        "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
        "deadline_perf_counter": deadline, "gpu_uuid": GPU_UUID,
        "scientific_claim": False, "test_read": False})
    try:
        environment = worker_environment(GPU_UUID)
        _cpu_spawn_gate(output, "prepare")
        process = bounded_process(_command("prepare", output, deadline), cwd=ROOT,
                                  log_path=output / "prepare.log", deadline=deadline, env=environment)
        processes.append({"phase": "prepare", **process})
        write_exclusive(output / "prepare_process.json", processes[-1])
        if process["status"] != "success":
            raise RuntimeError(f"preparation worker did not succeed: {process['status']}")
        protocol = json.loads((output / "prepared_protocol.json").read_text(encoding="utf-8"))
        protocol["timing"] = {"started_perf_counter": started, "deadline_perf_counter": deadline,
                              "prepare_process_elapsed_seconds": process["elapsed_seconds"]}
        protocol["protocol_sha256"] = canonical_digest(protocol)
        write_exclusive(output / "protocol.json", protocol)
        state["protocol"] = protocol
        frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if frozen["protocol_sha256"] != canonical_digest(
                {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
            raise RuntimeError("protocol on disk differs from freeze")
        result = state["result"]
        result.update(protocol_sha256=protocol["protocol_sha256"],
                      train_data_identity=protocol["train_data_identity"],
                      val_data_identity=protocol["val_data_identity"],
                      flop_probe=protocol["flop_probe"], limitations=protocol["limitations"])
        print(json.dumps({"frozen_protocol_sha256": protocol["protocol_sha256"]}), flush=True)
        owned_peak = 0
        for seed in recipe.SEEDS:
            if time.perf_counter() >= deadline:
                raise TimeoutError("round deadline reached before seed spawn")
            gate = shared.gpu_gate(GPU_UUID, owned_reserved_peak_bytes=owned_peak,
                                   estimated_peak_bytes=recipe.ESTIMATED_PEAK_BYTES,
                                   margin_bytes=recipe.HEADROOM_MARGIN_BYTES)
            result["gpu_gates"].append({"seed": seed, **gate})
            seed_deadline = min(time.perf_counter() + PER_SEED_SECONDS, deadline)
            process = bounded_process(_command("seed", output, seed_deadline, seed), cwd=ROOT,
                                      log_path=output / f"seed{seed}.log", deadline=seed_deadline,
                                      env=environment)
            processes.append({"phase": "seed", "seed": seed, **process})
            write_exclusive(output / f"seed{seed}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(f"seed {seed} did not succeed: {process['status']}")
            receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
            if receipt["seed"] != seed:
                raise RuntimeError("worker receipt seed differs from the declared job")
            owned_peak = max(owned_peak, int(receipt["owned_cuda_reserved_peak_bytes"]))
            result["seeds"][str(seed)] = receipt
            write_exclusive(output / f"partial_after_seed{seed}.json", result)
            print(json.dumps({"seed_done": seed, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if sorted(result["seeds"]) != [str(seed) for seed in recipe.SEEDS]:
            raise RuntimeError("all frozen seeds must complete before reading")
        _collect_receipts(output, state, deadline, environment, processes)
        if time.perf_counter() >= deadline:
            raise TimeoutError("round hard cap exceeded during paired reading")
        elapsed = time.perf_counter() - started
        result.update(elapsed_seconds_total=elapsed,
                      soft_overrun_seconds=max(0.0, elapsed - PLANNED_SECONDS),
                      hard_overrun_seconds=max(0.0, elapsed - HARD_CAP_SECONDS),
                      processes=processes)
        write_exclusive(output / "result.json", result)
        write_exclusive(output / "attempt.json", {
            "status": "complete", "scientific_claim": False, "test_read": False,
            "protocol_sha256": protocol["protocol_sha256"], "decision": result["decision"],
            "elapsed_seconds_total": elapsed})
        print(json.dumps({"decision": result["decision"], "elapsed_seconds": elapsed,
                          "gate_failures": len(result["gate_pre_screen"]["failures"])}), flush=True)
        return result
    except BaseException as exc:
        import traceback
        elapsed = time.perf_counter() - started
        write_exclusive(output / "failure.json", {
            "status": "failed", "scientific_claim": False, "test_read": False,
            "protocol_sha256": None if state["protocol"] is None else state["protocol"]["protocol_sha256"],
            "elapsed_seconds_total": elapsed, "planned_seconds": PLANNED_SECONDS,
            "hard_cap_seconds": HARD_CAP_SECONDS,
            "soft_overrun_seconds": max(0.0, elapsed - PLANNED_SECONDS),
            "hard_overrun_seconds": max(0.0, elapsed - HARD_CAP_SECONDS),
            "seeds_completed": sorted(state["result"]["seeds"]), "processes": processes,
            "gpu_gates": state["result"]["gpu_gates"],
            "latest_saved_updates": {str(seed): max(
                (int(path.stem.split("_")[-1]) for path in
                 (output / f"seed{seed}/training/candidate").glob("update_*.pt")), default=0)
                for seed in recipe.SEEDS},
            "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "limitations": ["partial attempt, no three-seed verdict; old artifacts remain untouched"],
            "note": "failed attempt is not resumed in place; no automatic retry"})
        raise


def main(argv=None):
    from scripts.r7_m3_offline import deny_network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    deny_network()
    return run_attempt(args.out)


if __name__ == "__main__":
    main()
