"""Confirmed child-deadline exit attribution and actual CPU baseline adapter closure."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from training import r7_v2_driver as driver
from training import r7_v2_protocol as protocol
from training import r7_v2_worker as worker
from test_r7_v2_driver import _attempt, _fake, _run, frozen
from test_r7_v2_evaluation import _run as evaluate_fixture
from test_r7_v2_evaluation import fixture as evaluation_fixture


@pytest.mark.parametrize("at_deadline", [False, True])
def test_owned_exit1_classifies_exact_frozen_deadline_and_preserves_reason(frozen, monkeypatch, at_deadline):
    fake = _fake(frozen, monkeypatch, fail_at=0, seconds=10779.75 if at_deadline else 1)
    clock, *_, original_popen, aggregate = fake
    reason = "TimeoutError: v2 worker whole-attempt deadline exhausted" if at_deadline else "ValueError: actual model failure"
    def popen(command, **kwargs):
        process = original_popen(command, **kwargs)
        process.code = 1
        entry = {"status": "failed", "job": frozen["jobs"][0], "protocol_sha256": frozen["protocol_sha256"],
                 "failure_reason": reason, "deadline_perf_counter": 10790.,
                 "monotonic_boot_id": frozen["monotonic_boot_id"], "budget_limited": at_deadline}
        protocol.write_json(protocol.worker_result_path(frozen["output"], frozen["jobs"][0]), entry, output=frozen["output"])
        return process
    fake = (*fake[:6], popen, aggregate)
    expected = driver.BudgetLimited if at_deadline else RuntimeError
    with pytest.raises(expected) as caught:
        _run(frozen, fake)
    attempt = _attempt(frozen)
    assert attempt["budget_limited"] is at_deadline and reason in attempt["failure_reason"]
    assert attempt["worker_failure_receipt"]["failure_reason"] == reason
    assert isinstance(caught.value, driver.BudgetLimited) is at_deadline
    assert len(fake[1]) == 1 and not fake[4] and fake[2][0].signals == [] and fake[2][0].done
    assert attempt["gpu_phase_elapsed_seconds"] == (10779.75 if at_deadline else 1)
    assert attempt["gpu_hours_charged"] == attempt["gpu_phase_elapsed_seconds"] / 3600
    assert attempt["whole_elapsed_seconds"] == clock.now
    timing = protocol.read_json(Path(frozen["output"]) / "workers" / (protocol.job_key(frozen["jobs"][0]) + ".timing.json"))
    assert timing["cleanup"] == "already-exited" and timing["budget_limited"] is at_deadline


@pytest.mark.parametrize("change", [None, "job", "protocol", "deadline", "boot", "flag"])
def test_only_exact_failure_receipt_can_attribute_child_deadline(frozen, monkeypatch, change):
    fake = _fake(frozen, monkeypatch, fail_at=0)
    original_popen = fake[6]
    def popen(command, **kwargs):
        process = original_popen(command, **kwargs)
        process.code = 1
        entry = {"status": "failed", "job": frozen["jobs"][0], "protocol_sha256": frozen["protocol_sha256"],
                 "failure_reason": "TimeoutError: actual frozen deadline", "deadline_perf_counter": 10790.,
                 "monotonic_boot_id": frozen["monotonic_boot_id"], "budget_limited": True}
        if change == "job": entry["job"] = frozen["jobs"][1]
        elif change == "protocol": entry["protocol_sha256"] = "0" * 64
        elif change == "deadline": entry["deadline_perf_counter"] = 1800.
        elif change == "boot": entry["monotonic_boot_id"] = "other-boot"
        elif change == "flag": entry["budget_limited"] = False
        protocol.write_json(protocol.worker_result_path(frozen["output"], frozen["jobs"][0]), entry, output=frozen["output"])
        return process
    fake = (*fake[:6], popen, fake[7])
    with pytest.raises(driver.BudgetLimited if change is None else RuntimeError):
        _run(frozen, fake)
    assert _attempt(frozen)["budget_limited"] is (change is None)
    assert len(fake[1]) == 1 and _attempt(frozen)["gpu_phase_elapsed_seconds"] == 1.


def test_actual_evaluator_provenance_passes_worker_baseline_guard(evaluation_fixture):
    provenance = evaluate_fixture(evaluation_fixture, lead=12, k=4, name="actual-baseline-adapter")
    value = {"baseline_reporting": protocol.BASELINE_REPORTING,
             "data": {"channels": provenance["channels"], "units": provenance["units"]}}
    job = {"phase": "evaluate", "seed": 41, "arm": "continue_l6", "lead": 12, "reasoning_steps": 4}
    worker.verify_baseline_provenance(value, job, provenance)
    assert provenance["baseline_definition"]["climatology"]["kind"] == "train-only-month-hour"
    assert provenance["climatology"]["kind"] == "train-only-month-hour-grid-mean-v1"
    assert provenance["n_evaluated"] == 21 and len(provenance["baseline_region_metrics"]) == 102
    directory = evaluation_fixture["tmp"] / "actual-baseline-adapter"
    assert all((directory / name).is_file() for name in ("baseline_region_metrics.csv", "baseline_per_case_metrics.csv"))
    changed = deepcopy(provenance)
    changed["baseline_definition"]["climatology"]["kind"] = changed["climatology"]["kind"]
    with pytest.raises(ValueError, match="mandatory zero-trained"):
        worker.verify_baseline_provenance(value, job, changed)
