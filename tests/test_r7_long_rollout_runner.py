"""CPU synthetic contracts and worker counterproofs, not weather evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import pytest
import torch

from test_r7_m4_autoregressive_rollout import runner_options
from data.r7_long_rollout_dataset import preflight_long_rollout_windows
from training.r7_experiment import canonical_digest, load_checkpoint
from training import r7_long_rollout_runner as runner
from scripts import study_r7_s3_long_rollout as study


@pytest.fixture(autouse=True)
def test_single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _options(tmp_path, kind="process"):
    paths, options = runner_options(tmp_path, kind=kind)
    options.pop("bf16")
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=3)
    options["contract"]["autoregression"] = {"physical_steps": 3, "physical_weights": [1., .5, .5],
        "window_sha256": summary["window_sha256"], "excluded_sample_ids": summary["excluded_sample_ids"]}
    options["physical_weights"] = (1., .5, .5)
    return paths, options


@pytest.mark.parametrize("kind", ["process", "generic"])
def test_frozen_cpu_endpoint_and_ordinary_checkpoint(tmp_path, kind):
    paths, options = _options(tmp_path, kind)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    saved = load_checkpoint(checkpoint, expected=report["signature"])
    assert report["selected_update"] == saved["updates"] == 3
    assert report["resumed_from_updates"] == 0 and report["parent_optimizer_imported"] is False
    assert report["test_read"] is False and report["scientific_claim"] is False and report["limitations"]
    assert report["contract"]["mode"] == "long_rollout"
    assert report["contract"]["training_code_sha256"] == runner.training_code_digest()
    assert saved["contract"]["scientific_claim"] is False and saved["contract"]["test_read"] is False
    assert saved["contract"]["limitations"]
    assert report["contract"]["autoregression"]["physical_steps"] == 3
    assert len(report["losses"]) == 3
    for row in report["losses"]:
        assert len(row["per_step_losses"]) == 3
        assert row["loss"] == pytest.approx(sum(w * v for w, v in zip(options["physical_weights"], row["per_step_losses"])), rel=1e-6)
    with pytest.raises(FileExistsError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)


@pytest.mark.parametrize("field,value", [("physical_steps", 2), ("physical_weights", [1., 1., 1.]),
                                          ("window_sha256", "0" * 64), ("excluded_sample_ids", [])])
def test_training_frozen_metadata_mismatch_refused(tmp_path, field, value):
    paths, options = _options(tmp_path)
    options["contract"]["autoregression"][field] = value
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_changed_model_semantics_refused(tmp_path):
    paths, options = _options(tmp_path)
    options["model"].default_reasoning_steps += 1
    with pytest.raises(ValueError, match="configuration"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)


def test_expired_training_deadline_refused(tmp_path):
    paths, options = _options(tmp_path)
    with pytest.raises(RuntimeError, match="deadline"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", deadline=time.perf_counter() - 1, **options)
    assert not (tmp_path / "refused").exists()


def test_actual_training_error_keeps_failure_receipt(tmp_path, monkeypatch):
    paths, options = _options(tmp_path)
    def test_failure(*args, **kwargs):
        raise RuntimeError("injected long objective failure")
    monkeypatch.setattr(runner, "_update", test_failure)
    with pytest.raises(RuntimeError, match="injected"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "failed", **options)
    failure = json.loads((tmp_path / "failed/failed_attempt.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed" and failure["resume_permitted"] is False
    assert not (tmp_path / "failed/training_report.json").exists()


def _fake_round(tmp_path, monkeypatch, mode, *, known_peak=123, gate_calls=None):
    monkeypatch.setattr(study, "PER_SEED_SECONDS", .5)
    monkeypatch.setattr(study, "PLANNED_SECONDS", 10.)
    monkeypatch.setattr(study, "HARD_CAP_SECONDS", 20.)
    def _gate(uuid, **kwargs):
        if gate_calls is not None:
            gate_calls.append(kwargs)
        required = max(kwargs["estimated_peak_bytes"], kwargs["owned_reserved_peak_bytes"]) + kwargs["margin_bytes"]
        if mode == "gate-failed":
            raise RuntimeError("injected headroom rejection")
        return {"passed": True, "required_bytes": required}
    monkeypatch.setattr(study.recipe._shared(), "gpu_gate", _gate)
    source = tmp_path / "feasibility"
    source.mkdir()
    (source / "attempt.json").write_text(json.dumps({"status": "complete", "protocol_sha256": "a" * 64}), encoding="utf-8")
    (source / "feasibility_receipt.json").write_text(json.dumps({"status": "success", "owned_cuda_reserved_peak_bytes": known_peak}), encoding="utf-8")
    output = tmp_path / "round"
    def test_process(arguments, *, log_path, **kwargs):
        phase = arguments[arguments.index("--worker") + 1]
        seed = int(arguments[arguments.index("--seed") + 1]) if "--seed" in arguments else None
        if phase == "prepare":
            body = {"protocol_sha256": "a" * 64, "train_data_identity": "b" * 64,
                    "val_data_identity": "c" * 64, "limitations": ["Synthetic CPU fixture only"]}
            study.write_exclusive(output / "prepared_protocol.json", body)
        elif phase == "seed" and not (mode == "failed" and seed == 42):
            study.write_exclusive(output / f"seed{seed}_receipt.json", {"seed": seed, "owned_cuda_reserved_peak_bytes": 456})
        elif phase == "reading":
            study.write_exclusive(output / "readings.json", {"decision": "registered-negative"})
        return {"status": "failed" if mode == "failed" and seed == 42 else "success",
                "reaped": True, "returncode": 1 if mode == "failed" and seed == 42 else 0,
                "signals": [], "elapsed_seconds": .01}
    monkeypatch.setattr(study, "bounded_process", test_process)
    return output, source


def test_owned_round_failure_stops_before_seed43(tmp_path, monkeypatch):
    output, source = _fake_round(tmp_path, monkeypatch, "failed")
    with pytest.raises(RuntimeError, match="seed42"):
        study.run_attempt(output, code_archive=tmp_path / "archive.zip", code_commit="a" * 40,
                          code_sha256="b" * 64, feasibility_root=source)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["seeds_completed"] == ["41"]
    assert not (output / "seed43_receipt.json").exists() and not (output / "result.json").exists()
    assert all(p["reaped"] for p in failure["processes"])


def test_complete_fake_round_contains_exact_three_seeds(tmp_path, monkeypatch):
    output, source = _fake_round(tmp_path, monkeypatch, "success")
    result = study.run_attempt(output, code_archive=tmp_path / "archive.zip", code_commit="a" * 40,
                               code_sha256="b" * 64, feasibility_root=source)
    assert sorted(result["seeds"]) == ["41", "42", "43"] and result["test_read"] is False
    assert len(result["processes"]) == 6 and result["decision"] == "registered-negative"
    assert json.loads((output / "attempt.json").read_text(encoding="utf-8"))["status"] == "complete"


@pytest.mark.parametrize("known_peak", [123, 3 * 2**30])
def test_each_spawn_retains_declared_headroom(tmp_path, monkeypatch, known_peak):
    calls = []
    output, source = _fake_round(tmp_path, monkeypatch, "success", known_peak=known_peak, gate_calls=calls)
    study.run_attempt(output, code_archive=tmp_path / "archive.zip", code_commit="a" * 40,
                      code_sha256="b" * 64, feasibility_root=source)
    assert len(calls) == 6
    for call in calls:
        assert call["estimated_peak_bytes"] == 2**31 and call["margin_bytes"] == 2**31
        assert call["owned_reserved_peak_bytes"] >= known_peak
        assert max(call["estimated_peak_bytes"], call["owned_reserved_peak_bytes"]) + call["margin_bytes"] >= 2**32


def test_headroom_rejection_prevents_any_worker_spawn(tmp_path, monkeypatch):
    output, source = _fake_round(tmp_path, monkeypatch, "gate-failed")
    with pytest.raises(RuntimeError, match="headroom"):
        study.run_attempt(output, code_archive=tmp_path / "archive.zip", code_commit="a" * 40,
                          code_sha256="b" * 64, feasibility_root=source)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["processes"] == [] and failure["seeds_completed"] == []
    assert not (output / "protocol.json").exists()


def _protocol(tmp_path, monkeypatch):
    windows = {"physical_steps": 12, "input_windows": 100, "usable_windows": 90,
               "excluded_sample_ids": ["synthetic-boundary"], "window_sha256": "a" * 64}
    spec = {"in_channels": 17, "detach_between_steps": False}
    model_digest, training_digest = "b" * 64, "c" * 64
    monkeypatch.setattr(study, "execution_files", lambda: ["training/r7_experiment.py"])
    monkeypatch.setattr(study.recipe._shared(), "archived_process_spec", lambda: (spec, None, None))
    import training.r7_experiment as experiment
    import data.r7_long_rollout_dataset as data
    monkeypatch.setattr(experiment, "dataset_identity", lambda manifest: (study.recipe._shared().V3_DATA_IDENTITY if manifest == study.recipe.TRAIN_MANIFEST else "d" * 64, None))
    monkeypatch.setattr(experiment, "model_code_digest", lambda: model_digest)
    monkeypatch.setattr(runner, "training_code_digest", lambda: training_digest)
    monkeypatch.setattr(data, "preflight_long_rollout_windows", lambda *args, **kwargs: windows)
    preliminary = study.freeze(tmp_path, kind="probe", started=1., deadline=1. + study.PROBE_HARD_SECONDS,
                               code_archive=tmp_path / "archive.zip", code_commit="a" * 40, code_sha256="b" * 64)
    candidate = {"mode": "long_rollout", "updates": study.UPDATES, "lr": study.LR,
        "warmup": study.WARMUP, "physical_weights": list(study.PHYSICAL_WEIGHTS), "steps": 4,
        "weight_decay": 1e-4, "batch_size": 1, "clip": 1., "checkpoint_every": 20,
        "bf16": False, "selection": "frozen endpoint; no validation selection",
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}
    body = {**preliminary, "train_manifest": str(study.recipe.TRAIN_MANIFEST), "val_manifest": str(study.recipe.VAL_MANIFEST),
        "train_data_identity": study.recipe._shared().V3_DATA_IDENTITY, "val_data_identity": "d" * 64,
        "model_code_sha256": model_digest, "training_code_sha256": training_digest, "train_windows": windows,
        "execution_files_sha256": {"training/r7_experiment.py": hashlib.sha256((study.ROOT / "training/r7_experiment.py").read_bytes()).hexdigest()},
        "arms": {"candidate": candidate}}
    body["protocol_sha256"] = canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"})
    study.write_exclusive(tmp_path / "protocol.json", body)
    return body


@pytest.mark.parametrize("field,value", [("lr", 1e-4), ("updates", 201), ("warmup", 20), ("bf16", True),
                                          ("physical_weights", [1.] * 12), ("mode", "two_step")])
def test_frozen_protocol_changed_recipe_refused(tmp_path, monkeypatch, field, value):
    body = _protocol(tmp_path, monkeypatch)
    assert study.validate_protocol(tmp_path) == body
    body["arms"]["candidate"][field] = value
    body["protocol_sha256"] = canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"})
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate"):
        study.validate_protocol(tmp_path)


def test_probe_timeout_reaps_owned_worker_and_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "PROBE_HARD_SECONDS", .3)
    monkeypatch.setattr(study.recipe._shared(), "gpu_gate", lambda *args, **kwargs: {"passed": True})
    actual = study.bounded_process
    def _blocked(arguments, **kwargs):
        return actual([study.sys.executable, "-c", "import time;time.sleep(20)"], **kwargs)
    monkeypatch.setattr(study, "bounded_process", _blocked)
    with pytest.raises(RuntimeError, match="prepare"):
        study.run_attempt(tmp_path / "blocked", probe=True, code_archive=tmp_path / "archive.zip",
                          code_commit="a" * 40, code_sha256="b" * 64)
    failure = json.loads((tmp_path / "blocked/failure.json").read_text(encoding="utf-8"))
    process = failure["processes"][0]
    assert process["status"] == "deadline-exceeded" and process["reaped"] is True
    assert all(signal["pid"] == process["pid"] for signal in process["signals"])
    assert failure["elapsed_seconds_total"] >= .3 and failure["elapsed_seconds_total"] < 8
    assert not (tmp_path / "blocked/attempt.json").exists()


def test_contract_true_scientific_claim_is_refused(tmp_path):
    paths, options = _options(tmp_path)
    options["contract"]["scientific_claim"] = True
    with pytest.raises(ValueError, match="scientific_claim"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused/training_report.json").exists()
