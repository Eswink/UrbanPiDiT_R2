"""Tiny offline CPU counterproofs; B references are opaque metadata, never B artifacts."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from training import r7_v2_driver as driver
from training import r7_v2_protocol as protocol


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeProcess:
    def __init__(self, clock):
        self.clock, self.done, self.pid = clock, False, 12345

    def wait(self, timeout):
        if self.done:
            return 0
        assert timeout > 0
        self.clock.now += 1
        self.done = True
        return 0

    def poll(self):
        return 0 if self.done else None

    def terminate(self):
        pytest.fail("normal fake worker must already be reaped")

    def kill(self):
        pytest.fail("normal fake worker must already be reaped")


@pytest.fixture
def sealed_fixture(tmp_path, monkeypatch):
    output = tmp_path / "r7_v2_comparison_20261003_attempt01"
    output.mkdir()
    specs = {arm: {"kind": "generic" if arm == "matched_generic" else "process",
                   "model": {"detach_between_steps": False}} for arm in protocol.C_ARMS}
    configuration = {"mode": "l6", "model_specs": specs, "initialization": {
        "anchor": specs["process"], "mapping": {arm: {"weight": "weight"} for arm in specs}},
        "primary": {"variable": "t2m"}, "tolerances": {"degradation": 0},
        "case_unit_selection": "exact paired fixture case units", "adaptive_gate": {"start": False},
        "reference": {"B": "opaque failed archive metadata; not revived or read"},
        "selection_evidence": {"sha256": "e" * 64}}
    value = protocol.build_protocol(stage="C", output=output, manifests=tmp_path / "manifests",
        data={"data_identity": "d" * 64, "channels": [f"var{i}" for i in range(17)],
              "units": ["K"] * 17, "test_read": False, "evaluation_cases": {
                  str(lead): {"n_available": 1, "cases": [["fixture-init", ["fixture-valid"]]]}
                  for lead in protocol.LEADS}}, sources={"source_sha256": "s" * 64},
        sidecar={"identity": "i" * 64, "path": str(tmp_path / "scale_metadata.json")},
        windows={"excluded_sample_ids": [], "window_sha256": "f" * 64}, parents={}, profile={},
        code={"model_code_sha256": "a" * 64, "source_tree_sha256": "b" * 64, "code_zip_sha256": "c" * 64},
        gpu_uuid="GPU-fixture", round_started_perf_counter=0.0, boot_id=protocol.monotonic_boot_id(),
        configuration=configuration)
    protocol.write_json(output / "protocol.json", value, output=output)
    protocol.write_json(output / "prepare_attempt.json", {
        "status": "prepared-not-run", "protocol_sha256": value["protocol_sha256"]}, output=output)
    clock, spawned, queried, aggregated = FakeClock(), [], [], []

    def preflight(value, *, check):
        assert (output / "run_started.json").is_file()
        clock.now += 2
        check()

    monkeypatch.setattr(driver, "_runtime_preflight", preflight)
    monkeypatch.setattr(driver, "_runtime_postflight", lambda *args, **kwargs: None)

    def snapshot(uuid):
        queried.append(uuid)
        return {"uuid": uuid, "free_mib": 9999, "read_only": True, "neighbors": []}

    def popen(command, **kwargs):
        job = value["jobs"][len(spawned)]
        assert command[:4] == [str(protocol.ROOT / ".venv/bin/python"), "-B", "-m", "training.r7_v2_worker"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
        assert all(kwargs["env"][name] == "4" for name in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert "PYTHONPATH" not in kwargs["env"]
        spawned.append(job)
        result = {"status": "success", "job": job, "scientific_claim": False,
            "limitations": ["synthetic CPU engineering receipt, not weather truth"], "test_read": False,
            "protocol_sha256": value["protocol_sha256"], **value["code"],
            "data_identity": value["data"]["data_identity"], "source_sha256": value["sources"]["source_sha256"],
            "sidecar_identity": value["sidecar"]["identity"], "windows_sha256": protocol.digest(value["windows"]),
            "baseline": {"allocated_bytes": 0, "reserved_bytes": 0}, "elapsed_seconds": 1,
            "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
            "updates_run": value["arm_configs"][job["arm"]]["updates"]}
        protocol.write_json(protocol.worker_result_path(output, job), result, output=output)
        return FakeProcess(clock)

    def aggregate(folder, frozen, execution):
        assert execution["jobs_completed"] == value["jobs"]
        aggregated.append(deepcopy(execution))
        clock.now += 1
        return {"scientific_claim": False, "limitations": ["CPU engineering fixture only"], "paused": False}

    def run():
        return driver.run_bounded_round(output, clock=clock, snapshot_fn=snapshot,
                                        popen_factory=popen, finalize_fn=aggregate)

    return output, value, clock, spawned, queried, aggregated, run


def _inventory(output):
    return {str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*") if path.is_file()}


def _refuse_retry(fixture):
    output, _, _, spawned, queried, _, run = fixture
    saved, counts = _inventory(output), (len(spawned), len(queried))
    with pytest.raises(FileExistsError, match="no retry"):
        run()
    assert _inventory(output) == saved
    assert (len(spawned), len(queried)) == counts


@pytest.mark.parametrize("kind", ["digest", "malformed", "shape"])
def test_startup_protocol_tamper_seals_unqualified_and_restore_cannot_retry(sealed_fixture, kind):
    output, value, _, spawned, queried, _, run = sealed_fixture
    path = output / "protocol.json"
    original = path.read_bytes()
    changed = deepcopy(value)
    changed["code"]["source_tree_sha256"] = "0" * 64
    path.write_text(json.dumps(changed) if kind == "digest" else "{" if kind == "malformed" else "[]", encoding="utf-8")
    with pytest.raises((ValueError, AttributeError)) as caught:
        run()
    assert (output / "run_started.json").is_file()
    attempt = protocol.read_json(output / "attempt.json")
    assert attempt["status"] == "failed" and attempt["finalized"] is False
    assert attempt["protocol_identity_qualified"] is False and "protocol_sha256" not in attempt
    assert "jobs_planned" not in attempt and "hard_cap_seconds" not in attempt
    assert attempt["no_retry_or_resurrection"] is True and attempt["scientific_claim"] is False
    assert attempt["failure_reason"].startswith(type(caught.value).__name__)
    assert caught.value.v2_attempt == attempt and not spawned and not queried
    path.write_bytes(original)
    _refuse_retry(sealed_fixture)


@pytest.mark.parametrize("kind", ["failed", "digest", "malformed", "missing"])
def test_declared_invalid_prepare_is_sealed_after_claim_without_spawn(sealed_fixture, kind):
    output, value, _, spawned, queried, _, run = sealed_fixture
    path = output / "prepare_attempt.json"
    original = path.read_bytes()
    if kind == "missing":
        path.unlink()
    else:
        path.write_text("{" if kind == "malformed" else json.dumps({
            "status": "failed" if kind == "failed" else "prepared-not-run",
            "protocol_sha256": "0" * 64 if kind == "digest" else value["protocol_sha256"]}), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        run()
    assert (output / "run_started.json").is_file()
    attempt = protocol.read_json(output / "attempt.json")
    assert attempt["status"] == "failed" and not attempt["finalized"]
    assert attempt["protocol_sha256"] == value["protocol_sha256"]
    assert attempt["jobs_completed"] == [] and not spawned and not queried
    assert caught.value.v2_attempt["status"] == "failed"
    path.write_bytes(original)
    _refuse_retry(sealed_fixture)


@pytest.mark.parametrize("kind", ["cross_boot", "future_anchor", "constructor", "preflight"])
def test_same_boot_and_startup_helpers_fail_closed_after_claim(sealed_fixture, monkeypatch, kind):
    output, value, _, spawned, queried, _, run = sealed_fixture
    if kind == "cross_boot":
        monkeypatch.setattr(driver, "monotonic_boot_id", lambda: "different-boot")
    elif kind == "future_anchor":
        value["round_started_perf_counter"] = 100
        value["protocol_sha256"] = protocol.digest({key: item for key, item in value.items() if key != "protocol_sha256"})
        (output / "protocol.json").write_text(json.dumps(value), encoding="utf-8")
        (output / "prepare_attempt.json").write_text(json.dumps({
            "status": "prepared-not-run", "protocol_sha256": value["protocol_sha256"]}), encoding="utf-8")
    else:
        def refused(*args, **kwargs):
            raise ValueError("startup helper failure fixture")
        monkeypatch.setattr(driver, "_execution" if kind == "constructor" else "_runtime_preflight", refused)
    with pytest.raises(ValueError):
        run()
    attempt = protocol.read_json(output / "attempt.json")
    assert attempt["status"] == "failed" and not attempt["finalized"]
    assert (output / "run_started.json").is_file() and not spawned and not queried
    _refuse_retry(sealed_fixture)


@pytest.mark.parametrize("name", ["run_started.json", "attempt.json", "execution_attempt.json", "publication_failure.json"])
def test_existing_claims_and_seals_are_opaque_and_never_overwritten(sealed_fixture, monkeypatch, name):
    output, _, _, spawned, queried, _, run = sealed_fixture
    (output / name).write_bytes(b"opaque failed evidence; completed inventory does not revive it")
    saved = _inventory(output)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: pytest.fail("existing output must not be consumed"))
    with pytest.raises(FileExistsError, match="no retry"):
        run()
    assert _inventory(output) == saved and not spawned and not queried


@pytest.mark.parametrize("kind", ["serialization", "close"])
def test_claimed_marker_write_or_close_error_seals_unqualified(sealed_fixture, monkeypatch, kind):
    output, _, _, spawned, queried, _, run = sealed_fixture
    if kind == "serialization":
        def denied(*args, **kwargs):
            raise OSError("claimed marker serialization denied fixture")
        monkeypatch.setattr(driver.json, "dump", denied)
    else:
        original = Path.open

        class FailedClose:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, *args):
                self.stream.close()
                raise OSError("claimed marker close denied fixture")

        def opened(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            return FailedClose(stream) if path.name == "run_started.json" and args == ("x",) else stream

        monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(OSError, match="claimed marker") as caught:
        run()
    assert (output / "run_started.json").exists() and not spawned and not queried
    assert caught.value.v2_attempt["status"] == "failed"
    assert not caught.value.v2_attempt["protocol_identity_qualified"]
    if kind == "close":
        assert protocol.read_json(output / "attempt.json")["status"] == "failed"
    else:
        assert any("startup seal publication failed" in note for note in caught.value.__notes__)
    _refuse_retry(sealed_fixture)


def test_losing_exclusive_claim_never_validates_or_publishes(sealed_fixture, monkeypatch):
    output, _, _, spawned, queried, _, run = sealed_fixture
    original = driver.write_path

    def raced(path, root):
        result = original(path, root)
        if result.name == "run_started.json" and not result.exists():
            result.write_bytes(b"opaque concurrent winning claim")
        return result

    monkeypatch.setattr(driver, "write_path", raced)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: pytest.fail("losing claimant must not validate"))
    with pytest.raises(FileExistsError):
        run()
    assert (output / "run_started.json").read_bytes() == b"opaque concurrent winning claim"
    assert not (output / "attempt.json").exists() and not (output / "publication_failure.json").exists()
    assert not spawned and not queried


@pytest.mark.parametrize("boundary", ["serialization_at_cap", "serialization_past_cap", "last_hash"])
def test_late_final_publication_preserves_snapshot_but_authoritatively_fails(sealed_fixture, monkeypatch, boundary):
    output, value, clock, spawned, queried, aggregated, run = sealed_fixture
    original_write, original_hash = driver.write_json, protocol.sha256_file
    published = {}

    def write(path, receipt, **kwargs):
        result = original_write(path, receipt, **kwargs)
        if path.name == "attempt.json":
            published["bytes"] = path.read_bytes()
            if boundary.startswith("serialization"):
                clock.now = value["hard_cap_seconds"] + (boundary == "serialization_past_cap")
        return result

    def last_hash(path):
        result = original_hash(path)
        if Path(path).name == "attempt.json" and boundary == "last_hash":
            clock.now = value["hard_cap_seconds"] + 1
        return result

    monkeypatch.setattr(driver, "write_json", write)
    monkeypatch.setattr(driver, "sha256_file", last_hash, raising=False)
    with pytest.raises(driver.BudgetLimited) as caught:
        run()
    assert spawned == value["jobs"] and len(queried) == 144 and len(aggregated) == 1
    assert (output / "attempt.json").read_bytes() == published["bytes"]
    attempt = protocol.read_json(output / "attempt.json")
    assert attempt["status"] == "success" and attempt["finalized"] is True
    assert attempt["whole_elapsed_seconds"] < value["hard_cap_seconds"]
    failure = protocol.read_json(output / "publication_failure.json")
    assert failure["status"] == "failed" and failure["finalized"] is False
    assert failure["authoritative"] is True and failure["acceptance_refused"] is True
    assert failure["budget_limited"] is True and failure["no_retry_or_resurrection"] is True
    assert failure["scientific_claim"] is False and failure["protocol_sha256"] == value["protocol_sha256"]
    assert failure["receipt_sha256"]["attempt.json"] == original_hash(output / "attempt.json")
    assert failure["publication_checked_perf_counter"] >= value["hard_cap_seconds"]
    assert caught.value.v2_attempt["status"] == "failed" and not caught.value.v2_attempt["finalized"]
    assert caught.value.v2_attempt["budget_limited"] and caught.value.v2_publication_failure == failure
    clock.now = 0
    _refuse_retry(sealed_fixture)


@pytest.mark.parametrize("seconds", [0, 10801])
def test_early_success_and_soft_overrun_continue_real_driver_fake_popen(sealed_fixture, seconds):
    output, value, clock, spawned, queried, aggregated, run = sealed_fixture
    clock.now = seconds
    outcome = run()
    attempt = protocol.read_json(output / "attempt.json")
    assert spawned == value["jobs"] and len(queried) == 144 and len(aggregated) == 1
    assert outcome["scientific_claim"] is False and attempt["status"] == "success" and attempt["finalized"]
    assert attempt["whole_elapsed_seconds"] == seconds + 147
    assert attempt["soft_overrun_seconds"] == max(0, seconds + 147 - value["planned_seconds"])
    assert attempt["soft_budget_exceeded"] is (seconds != 0) and not attempt["budget_limited"]
    assert not (output / "publication_failure.json").exists()
    _refuse_retry(sealed_fixture)


def test_final_hash_error_has_authoritative_failure_and_never_returns_success(sealed_fixture, monkeypatch):
    output, _, _, _, _, _, run = sealed_fixture
    original = protocol.sha256_file

    def denied(path):
        if Path(path).name == "attempt.json":
            raise OSError("last receipt hash denied fixture")
        return original(path)

    monkeypatch.setattr(driver, "sha256_file", denied, raising=False)
    with pytest.raises(OSError, match="last receipt hash") as caught:
        run()
    failure = protocol.read_json(output / "publication_failure.json")
    assert failure["status"] == "failed" and failure["acceptance_refused"] and failure["authoritative"]
    assert not failure["budget_limited"] and not caught.value.v2_attempt["finalized"]
    _refuse_retry(sealed_fixture)


def test_failure_marker_io_diagnostic_does_not_mask_late_budget_error(sealed_fixture, monkeypatch):
    output, value, clock, _, _, _, run = sealed_fixture
    original = driver.write_json

    def write(path, receipt, **kwargs):
        if path.name == "publication_failure.json":
            raise OSError("failure marker denied fixture")
        result = original(path, receipt, **kwargs)
        if path.name == "attempt.json":
            clock.now = value["hard_cap_seconds"] + 1
        return result

    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(driver.BudgetLimited) as caught:
        run()
    assert caught.value.v2_attempt["status"] == "failed" and caught.value.v2_attempt["budget_limited"]
    assert any("failure marker denied" in note for note in caught.value.__notes__)
    assert (output / "run_started.json").exists()
    _refuse_retry(sealed_fixture)
