"""Liveness-guard tests for the round tools (interruption hardening).

A stage is proved finished by its receipt and nothing else, but "no receipt" does not
mean "no work in flight": a stopped orchestrator can leave its worker running. These
tests pin the read-only ``/proc`` scan that sees such an orphan, and the refusal that
stops ``run_stage`` from spawning a second worker onto the same attempt directory.

Split from ``tests/test_r7_round_stages.py`` to stay small. No GPU, no network, no
``outputs/``: the proc tree is synthetic and every outside call is injected.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = "scripts/study_fake.py"
PROC_FILES = (
    "tools/r7_round_liveness.py",
    "tools/r7_round_support.py",
    "tools/r7_round_stages.py",
    "tools/r7_round.py",
)
#: A call that could reach another process. The round tools must contain none of these.
FORBIDDEN_NAMES = ("kill", "killall", "pkill", "terminate", "SIGTERM", "SIGKILL", "SIGSTOP", "SIGCONT")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def support():
    """The per-stage wrapper, which re-exports the shared names it needs."""
    return _load("r7_round_stages_under_test", "tools/r7_round_stages.py")


@pytest.fixture()
def liveness():
    """The read-only liveness module on its own, for the scan-level tests."""
    return _load("r7_round_liveness_under_test", "tools/r7_round_liveness.py")


def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


def _proc_tree(root: Path, entries: dict[str, list[str]]) -> Path:
    """A synthetic ``/proc``: one directory per pid, each holding a NUL-joined cmdline."""
    root.mkdir(parents=True, exist_ok=True)
    for name, tokens in entries.items():
        entry = root / name
        entry.mkdir()
        (entry / "cmdline").write_bytes("\0".join(tokens).encode("utf-8") + b"\0")
    return root


def _use_proc(support, monkeypatch, root: Path, entries: dict[str, list[str]]) -> Path:
    """Point the stage wrapper's scan at a synthetic proc tree."""
    proc = _proc_tree(root, entries)
    monkeypatch.setattr(support.liveness, "PROC_ROOT", proc)
    return proc


def _worker_tokens(output: Path, phase: str, seed: int | None = None, driver: str = DRIVER) -> list[str]:
    tokens = [sys.executable, str(output.parent.parent / driver), "--worker", phase, "--out", str(output)]
    if seed is not None:
        tokens += ["--seed", str(seed)]
    return tokens


def _gpu_runner(calls: list, free_mib: int = 24000, worker: dict | None = None):
    def runner(argv, *args, **kwargs):
        calls.append(list(argv))
        if argv and argv[0] == "nvidia-smi":
            return _ok(f"GPU-test, {free_mib}, 24576\n")
        return worker if worker is not None else _ok()

    return runner


def _protocol_file(output: Path, **overrides) -> Path:
    body = {"protocol_sha256": "a" * 64, "deadline_perf_counter": 10_000.0, "per_seed_seconds": 1800.0,
            "planned_seconds": 5400.0, "hard_cap_seconds": 7200.0, "gpu_uuid": "GPU-test",
            "estimated_peak_bytes": 2 ** 31, "headroom_margin_bytes": 2 ** 31}
    body.update(overrides)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "protocol.json"
    target.write_text(json.dumps(body), encoding="utf-8")
    return target


def _clock() -> float:
    """A frozen clock inside the synthetic protocol's window."""
    return 1000.0


def test_running_worker_pids_finds_the_matching_worker(liveness, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    proc = _proc_tree(tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 41)})
    found = liveness.running_worker_pids(DRIVER, "seed", output, 41, proc_root=proc)
    assert [item["pid"] for item in found] == [4242]


def test_running_worker_pids_counterproof_a_different_seed_never_matches(liveness, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    proc = _proc_tree(tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 42)})
    assert liveness.running_worker_pids(DRIVER, "seed", output, 41, proc_root=proc) == []


def test_running_worker_pids_counterproof_a_different_attempt_never_matches(liveness, tmp_path):
    other = tmp_path / "outputs/stem_attempt02"
    proc = _proc_tree(tmp_path / "proc", {"4242": _worker_tokens(other, "seed", 41)})
    mine = tmp_path / "outputs/stem_attempt01"
    assert liveness.running_worker_pids(DRIVER, "seed", mine, 41, proc_root=proc) == []


def test_running_worker_pids_counterproof_a_bare_mention_never_matches(liveness, tmp_path):
    """A shell that merely names the driver is not a worker of this stage."""
    output = tmp_path / "outputs/stem_attempt01"
    proc = _proc_tree(tmp_path / "proc", {
        "11": ["/bin/bash", "-c", f"grep {DRIVER} notes.txt"],
        "22": [sys.executable, str(tmp_path / DRIVER), "--worker", "seed", "--out", str(output)],
    })
    assert liveness.running_worker_pids(DRIVER, "seed", output, 41, proc_root=proc) == []


def test_running_worker_pids_counterproof_an_absent_proc_root_is_empty(liveness, tmp_path):
    assert liveness.running_worker_pids(DRIVER, "seed", tmp_path, 41, proc_root=tmp_path / "nope") == []


def test_running_worker_pids_ignores_non_pid_entries(liveness, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    proc = _proc_tree(tmp_path / "proc", {
        "self": _worker_tokens(output, "seed", 41),
        "4242": _worker_tokens(output, "seed", 41),
    })
    found = liveness.running_worker_pids(DRIVER, "seed", output, 41, proc_root=proc)
    assert [item["pid"] for item in found] == [4242]


def test_stage_liveness_lists_every_stage_and_flags_the_running_one(liveness, tmp_path):
    attempt = tmp_path / "outputs/stem_attempt01"
    proc = _proc_tree(tmp_path / "proc", {"4242": _worker_tokens(attempt, "seed", 42)})
    spec = {"driver": DRIVER, "seeds": (41, 42, 43)}
    report = liveness.stage_liveness(spec, attempt, proc_root=proc)
    assert [(row["phase"], row["seed"]) for row in report["running"]] == [("seed", 42)]
    assert len(report["stages"]) == 6


def test_stage_plan_reports_a_running_stage(support, monkeypatch, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    _use_proc(support, monkeypatch, tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 41)})
    plan = support.stage_plan(output, "seed", 41, driver_rel=DRIVER)
    assert plan["skip"] is False
    assert plan["already_running"] is True
    assert plan["running_pids"] == [4242]


def test_stage_plan_counterproof_without_a_driver_it_never_claims_running(support, tmp_path):
    """The scan needs the driver path; a plan that cannot scan must not claim idleness either."""
    output = tmp_path / "outputs/stem_attempt01"
    plan = support.stage_plan(output, "seed", 41)
    assert plan["already_running"] is False
    assert plan["running_pids"] == []


def test_stage_plan_counterproof_a_finished_stage_is_never_running(support, monkeypatch, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    output.mkdir(parents=True)
    (output / "seed41_receipt.json").write_text("{}", encoding="utf-8")
    _use_proc(support, monkeypatch, tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 41)})
    plan = support.stage_plan(output, "seed", 41, driver_rel=DRIVER)
    assert plan["skip"] is True
    assert plan["already_running"] is False


def test_run_stage_counterproof_a_live_worker_is_never_duplicated(support, monkeypatch, tmp_path):
    """The guard: with an orphan still training, no second worker may be spawned."""
    output = tmp_path / "outputs/stem_attempt01"
    output.mkdir(parents=True)
    _use_proc(support, monkeypatch, tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 41)})
    calls: list = []
    record = support.run_stage(output, DRIVER, "seed", seed=41, runner=_gpu_runner(calls))
    assert record["status"] == "already-running"
    assert record["passed"] is False
    assert calls == []


def test_run_stage_spawns_when_the_worker_is_gone(support, monkeypatch, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    _protocol_file(output)
    _use_proc(support, monkeypatch, tmp_path / "proc", {})
    calls: list = []
    record = support.run_stage(output, DRIVER, "seed", seed=41, runner=_gpu_runner(calls), clock=_clock)
    assert record["status"] == "success"
    assert any(call and call[0] != "nvidia-smi" for call in calls)


def test_run_stage_counterproof_another_seed_running_does_not_block_this_one(support, monkeypatch, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    _protocol_file(output)
    _use_proc(support, monkeypatch, tmp_path / "proc", {"4242": _worker_tokens(output, "seed", 42)})
    calls: list = []
    record = support.run_stage(output, DRIVER, "seed", seed=41, runner=_gpu_runner(calls), clock=_clock)
    assert record["status"] == "success"


def test_run_stage_gates_on_the_measured_peak_from_a_finished_seed(support, monkeypatch, tmp_path):
    """A finished seed's receipt outweighs the protocol's freeze-time guess."""
    output = tmp_path / "outputs/stem_attempt01"
    _protocol_file(output)
    measured = 14_552_137_728
    (output / "seed41_receipt.json").write_text(
        json.dumps({"seed": 41, "owned_cuda_reserved_peak_bytes": measured}), encoding="utf-8"
    )
    _use_proc(support, monkeypatch, tmp_path / "proc", {})
    calls: list = []
    record = support.run_stage(output, DRIVER, "seed", seed=42, runner=_gpu_runner(calls), clock=_clock)
    expected_mib = max(2 ** 31, measured) // 2 ** 20 + 2 ** 31 // 2 ** 20
    assert record["gate"]["required_bytes"] == expected_mib * 2 ** 20


def test_measured_peak_bytes_takes_the_largest_finished_seed(liveness, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    output.mkdir(parents=True)
    for seed, value in ((41, 9_000_000_000), (42, 14_552_137_728)):
        (output / f"seed{seed}_receipt.json").write_text(
            json.dumps({"seed": seed, "owned_cuda_reserved_peak_bytes": value}), encoding="utf-8"
        )
    assert liveness.measured_peak_bytes(output) == 14_552_137_728


def test_measured_peak_bytes_counterproof_absent_or_broken_receipts_are_zero(liveness, tmp_path):
    output = tmp_path / "outputs/stem_attempt01"
    output.mkdir(parents=True)
    (output / "seed41_receipt.json").write_text("{not json", encoding="utf-8")
    assert liveness.measured_peak_bytes(output) == 0


def _signal_references(path: Path) -> list[str]:
    """Names that could reach another process, found anywhere in the module."""
    hits = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import) and any(alias.name == "signal" for alias in node.names):
            hits.append("import signal")
        elif isinstance(node, ast.ImportFrom) and node.module == "signal":
            hits.append("from signal import ...")
        elif isinstance(node, ast.Attribute) and (
            node.attr in FORBIDDEN_NAMES or node.attr.startswith("SIG")
        ):
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            hits.append(node.id)
    return hits


def test_round_tools_never_signal_a_process():
    """The liveness scan is read-only: no signal call may enter these modules."""
    for relative in PROC_FILES:
        assert _signal_references(ROOT / relative) == [], relative


def test_signal_scanner_counterproof_it_flags_a_signal_call(tmp_path):
    """The scanner above must be able to fail, or it proves nothing (R-026)."""
    sample = tmp_path / "sample.py"
    sample.write_text("import os\n\nos.kill(1, 9)\n", encoding="utf-8")
    assert _signal_references(sample) == ["kill"]
