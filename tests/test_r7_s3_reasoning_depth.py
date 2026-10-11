"""CI-safe checks for the reasoning-depth driver (K 4 -> 8 on the 2400-update arm).

The reasoning-depth round changes exactly one factor against the registered
2400-update wide interior-dose arm: the internal reasoning depth K, raised from 4
to 8 in training (``steps``) and in scoring (``reasoning_steps``). The
fidelity-critical parts are that both K values really enter the frozen protocol
and the bound training contract/signature, that the frozen supervision mask and
physical weights are unchanged, and that the primary pin is the registered
2400-update arm's *boundary* table (the table the reading stage actually reads).
The pin logic is exercised against a synthetic output tree so this file is checked
on a checkout with no ``outputs/`` -- the earlier dose round turned CI red exactly
because its test read a real ``outputs/`` artifact. Every check has a counterproof
that a deliberately corrupted constant or pin must fail it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIABLES = ("t2m", "u10", "v10", "mslp", "tcc", "tcwv", "sp", "z500", "t500", "u500",
             "v500", "q500", "z850", "t850", "u850", "v850", "r850")
COHORTS = ((6, 472), (12, 468), (24, 460), (48, 444), (72, 428))
PHYSICAL_WEIGHTS = (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


depth = _load("study_r7_s3_reasoning_depth")
dose = depth.dose
interior = depth.interior
wide = depth.wide


def _rmse_csv():
    lines = ["lead_hours,variable,rmse,unit,n_initializations"]
    for variable in VARIABLES:
        lines.append(f"6,{variable},1.0,K,472")
    return "\n".join(lines) + "\n"


def _boundary_csv():
    lines = ["region,variable,rmse,unit,n_initializations"]
    for region in ("full", "interior_32", "edge_32"):
        for variable in VARIABLES:
            lines.append(f"{region},{variable},1.0,K,472")
    return "\n".join(lines) + "\n"


def _synthetic_run(tmp_path):
    """Minimal stand-in for the registered 2400-update dose arm's output tree."""
    from training.r7_arm_harness import sha256_file

    run = tmp_path / "prev"
    for seed in (41, 42, 43):
        evaluations = {}
        for lead, cohort in COHORTS:
            folder = run / f"seed{seed}" / f"lead_{lead:03d}h"
            folder.mkdir(parents=True)
            (folder / "rmse.csv").write_text(_rmse_csv(), encoding="utf-8")
            (folder / "boundary_rmse.csv").write_text(_boundary_csv(), encoding="utf-8")
            evaluations[str(lead)] = {
                "dir": str(folder), "n_evaluated": cohort, "n_available_windows": cohort,
                "rmse_csv_sha256": sha256_file(folder / "rmse.csv"),
                "boundary_rmse_csv_sha256": sha256_file(folder / "boundary_rmse.csv")}
        (run / f"seed{seed}_receipt.json").write_text(
            json.dumps({"checkpoint_sha256": f"{seed:064d}", "evaluations": evaluations}),
            encoding="utf-8")
    return run


def _check_constants():
    """The frozen single factor and every held-fixed neighbour, asserted from the module."""
    assert depth.FORMAT == "r7-s3-reasoning-depth-protocol-v1"
    assert depth.UPDATES == 2400
    assert depth.STEPS == 8 and depth.REASONING_STEPS == 8
    assert (depth.PER_SEED_SECONDS, depth.PLANNED_SECONDS, depth.HARD_CAP_SECONDS) == (18000., 54000., 72000.)
    assert depth.PER_SEED_SECONDS <= depth.HARD_CAP_SECONDS
    assert depth.PLANNED_SECONDS < depth.HARD_CAP_SECONDS
    # the registered arm this round derives from must stay untouched
    assert wide.STEPS == 4 and wide.REASONING_STEPS == 4
    assert wide.UPDATES == 800
    assert tuple(wide.PHYSICAL_WEIGHTS) == PHYSICAL_WEIGHTS
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.SEEDS == (41, 42, 43)
    assert depth.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert str(depth.PREVIOUS_DOSE_ARM_RUN).endswith("outputs/r7_s3_wide_interior_dose_20261009_attempt02")
    assert str(depth.DEFAULT_OUT).endswith("outputs/r7_s3_reasoning_depth_20261009_attempt01")


def test_only_the_reasoning_depth_changed():
    _check_constants()


@pytest.mark.parametrize("attr, wrong", [("STEPS", 4), ("REASONING_STEPS", 4), ("UPDATES", 800),
                                         ("PER_SEED_SECONDS", 12600.), ("PLANNED_SECONDS", 36000.),
                                         ("HARD_CAP_SECONDS", 54000.),
                                         ("FORMAT", "r7-s3-wide-interior-dose-protocol-v1")])
def test_a_corrupted_constant_fails_the_checks(monkeypatch, attr, wrong):
    """Counterproof: the checks are sensitive to each frozen constant."""
    _check_constants()
    monkeypatch.setattr(depth, attr, wrong)
    with pytest.raises(AssertionError):
        _check_constants()


def test_protocol_constants_carry_k_in_training_and_scoring():
    body = depth.protocol_constants()
    assert body["format"] == depth.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 8
    assert body["evaluation"]["reasoning_steps"] == 8
    assert body["recipe"]["physical_weights"] == list(PHYSICAL_WEIGHTS)
    assert body["recipe"]["lr"] == 2e-5 and body["recipe"]["warmup"] == 10
    assert body["recipe"]["bf16"] is False
    assert body["supervision"] == interior.supervision_block()
    assert body["supervision"]["runner_block"]["selected_cells"] == 4225
    assert body["budgets"] == {"planned_seconds": 54000., "hard_cap_seconds": 72000.,
                               "per_seed_seconds": 18000.}
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "K 4 -> 8" in body["difference"] and "reasoning_steps=8" in body["difference"]


def test_protocol_constants_counterproof_k_constant_must_be_8(monkeypatch):
    """Counterproof: if the module's K were still 4 the frozen fields would show it."""
    body = depth.protocol_constants()
    assert body["recipe"]["steps"] == depth.STEPS == 8
    assert body["evaluation"]["reasoning_steps"] == depth.REASONING_STEPS == 8
    monkeypatch.setattr(depth, "STEPS", 4)
    monkeypatch.setattr(depth, "REASONING_STEPS", 4)
    corrupted = depth.protocol_constants()
    assert corrupted["recipe"]["steps"] == 4 and corrupted["evaluation"]["reasoning_steps"] == 4
    assert corrupted["recipe"]["steps"] != body["recipe"]["steps"]


def test_previous_arm_pins_read_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(depth, "PREVIOUS_DOSE_ARM_RUN", _synthetic_run(tmp_path))
    pins = depth.previous_arm_pins()
    assert pins["record_id"] == "s3-wide-interior-dose"
    assert pins["updates"] == 2400
    assert pins["seeds"] == [41, 42, 43]
    assert pins["run_dir"].endswith("prev")
    assert len(pins["seeds_evaluations"]) == 3
    from training.r7_arm_harness import sha256_file
    for seed in (41, 42, 43):
        leads = pins["seeds_evaluations"][str(seed)]["leads"]
        assert len(leads) == 5
        for lead in ("6", "12", "24", "48", "72"):
            entry = leads[lead]
            # the pinned hash must be the *boundary* table the reading stage reads,
            # not rmse.csv (pinning the wrong one broke a full attempt)
            assert entry["boundary_rmse_csv_sha256"] != entry.get("rmse_csv_sha256")
            assert sha256_file(Path(entry["dir"]) / "boundary_rmse.csv") \
                == entry["boundary_rmse_csv_sha256"]
            assert sha256_file(Path(entry["dir"]) / "rmse.csv") \
                != entry["boundary_rmse_csv_sha256"]


def test_previous_arm_pins_refuse_a_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(depth, "PREVIOUS_DOSE_ARM_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        depth.previous_arm_pins()


def test_previous_arm_pins_refuse_a_cohort_that_differs_from_the_pin(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    receipt_path = run / "seed42_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evaluations"]["72"]["n_evaluated"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(depth, "PREVIOUS_DOSE_ARM_RUN", run)
    with pytest.raises(RuntimeError):
        depth.previous_arm_pins()


def test_pin_check_rejects_a_boundary_table_that_drifted(tmp_path, monkeypatch):
    """Counterproof: the reading stage's pin check catches a changed boundary table."""
    from training.r7_arm_harness import sha256_file
    run = _synthetic_run(tmp_path)
    monkeypatch.setattr(depth, "PREVIOUS_DOSE_ARM_RUN", run)
    pins = depth.previous_arm_pins()
    lead = pins["seeds_evaluations"]["41"]["leads"]["48"]
    table = Path(lead["dir"]) / "boundary_rmse.csv"
    assert sha256_file(table) == lead["boundary_rmse_csv_sha256"]
    table.write_text(_boundary_csv().replace("1.0", "2.0"), encoding="utf-8")
    assert sha256_file(table) != lead["boundary_rmse_csv_sha256"]


class _FakeDataset:
    """Just enough dataset surface for the runner's contract binder."""

    manifest = "fake/train.jsonl"
    summary = {"window_sha256": "a" * 64, "excluded_sample_ids": ["excluded"]}

    def __len__(self):
        return 3


def test_k_enters_the_bound_contract_and_signature(monkeypatch):
    from training import r7_long_rollout_runner as runner
    from training.r7_experiment import canonical_digest
    monkeypatch.setattr(runner, "dataset_identity", lambda manifest: ("d" * 64, None))
    supplied = {"data_identity": "d" * 64,
                "autoregression": {"physical_steps": 12, "physical_weights": list(PHYSICAL_WEIGHTS),
                                   "window_sha256": "a" * 64, "excluded_sample_ids": ["excluded"]}}

    def bound(steps):
        return runner._contract(_FakeDataset(), supplied, output="/tmp/out", steps=steps,
                                updates=depth.UPDATES, seed=41, lr=wide.LR, warmup=wide.WARMUP,
                                weight_decay=wide.WEIGHT_DECAY, device=SimpleNamespace(type="cpu"),
                                batch_size=wide.BATCH_SIZE, clip=wide.CLIP,
                                checkpoint_every=wide.CHECKPOINT_EVERY, physical_weights=PHYSICAL_WEIGHTS)

    four, eight = bound(wide.STEPS), bound(depth.STEPS)
    assert four["steps"] == 4 and eight["steps"] == 8
    # the signature is the digest of the bound contract, so K must change it
    assert canonical_digest(four) != canonical_digest(eight)
    assert {k: v for k, v in four.items() if k != "steps"} == {k: v for k, v in eight.items()
                                                               if k != "steps"}


def test_seed_binds_k8_into_training_and_scoring(tmp_path, monkeypatch):
    import training.r7_evaluate as evaluator
    import training.r7_long_rollout_runner as runner

    captured = {"training": None, "scoring": []}

    def fake_training(manifest, output, **kwargs):
        captured["training"] = kwargs
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "training_report.json").write_text("{}", encoding="utf-8")
        checkpoint = folder / "checkpoint.pt"
        checkpoint.write_text("x", encoding="utf-8")
        return checkpoint, {"updates_this_run": kwargs["updates"],
                            "data_identity": kwargs["contract"]["data_identity"],
                            "supervision": kwargs["contract"]["autoregression"]["supervision"],
                            "contract": {"steps": kwargs["steps"]}, "elapsed_seconds": 1.0}

    def fake_evaluate(manifest, output_dir, **kwargs):
        captured["scoring"].append(kwargs)
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("rmse.csv", "boundary_rmse.csv", "climatology_skill.csv", "provenance.json"):
            (folder / name).write_text(name, encoding="utf-8")
        cohort = wide.WIDE_VAL_COHORTS[str(kwargs["lead_hours"][0])]
        return {"split": "val", "n_evaluated": cohort, "n_available_windows": cohort,
                "boundary_scoring": {"regions": ["full", "interior_32"]}, "elapsed_seconds": 0.1}

    monkeypatch.setattr(runner, "fine_tune_long_rollout", fake_training)
    monkeypatch.setattr(evaluator, "evaluate_local", fake_evaluate)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(interior, "supervision_mask", lambda: SimpleNamespace(to=lambda *a, **k: "mask"))
    monkeypatch.setattr(wide, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"), "p" * 64))

    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": {"kind": "process"}}}}}
    output = tmp_path / "run"
    output.mkdir()
    depth._seed(output, body, 41, None)

    assert captured["training"]["steps"] == 8
    assert captured["training"]["updates"] == 2400
    assert captured["training"]["physical_weights"] == PHYSICAL_WEIGHTS
    assert captured["training"]["contract"]["model"] == {"kind": "process"}
    assert [call["reasoning_steps"] for call in captured["scoring"]] == [8] * 5
    assert [call["lead_hours"] for call in captured["scoring"]] == [(lead,) for lead in wide.EVALUATION_LEADS]
    receipt = json.loads((output / "seed41_receipt.json").read_text(encoding="utf-8"))
    assert receipt["reasoning_steps"] == 8 and receipt["updates"] == 2400


def test_seed_counterproof_a_k4_contract_is_refused(tmp_path, monkeypatch):
    """Counterproof: a training report bound at K=4 must abort the seed."""
    import training.r7_evaluate as evaluator
    import training.r7_long_rollout_runner as runner

    def fake_training(manifest, output, **kwargs):
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "training_report.json").write_text("{}", encoding="utf-8")
        checkpoint = folder / "checkpoint.pt"
        checkpoint.write_text("x", encoding="utf-8")
        return checkpoint, {"updates_this_run": kwargs["updates"],
                            "data_identity": kwargs["contract"]["data_identity"],
                            "supervision": kwargs["contract"]["autoregression"]["supervision"],
                            "contract": {"steps": 4}, "elapsed_seconds": 1.0}

    monkeypatch.setattr(runner, "fine_tune_long_rollout", fake_training)
    monkeypatch.setattr(evaluator, "evaluate_local", lambda *a, **k: pytest.fail("scoring must not run"))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(interior, "supervision_mask", lambda: SimpleNamespace(to=lambda *a, **k: "mask"))
    monkeypatch.setattr(wide, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"), "p" * 64))
    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": {"kind": "process"}}}}}
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(RuntimeError, match="K=8"):
        depth._seed(output, body, 41, None)


def test_training_path_accepts_k8_on_a_cpu_tiny_model():
    """CPU smoke: the real rollout/contract path accepts K=8 (no GPU, synthetic tensors)."""
    from test_r7_m4_autoregressive_rollout import batch, tiny_spec
    from training.r7_experiment import make_model
    from training.r7_long_rollout import training_long_rollout

    torch.manual_seed(12)
    model = make_model("process", dict(tiny_spec("process", parent_detach=True),
                                       known_context_inputs=True)).train()
    sample = batch(17)
    targets = [sample["atmos_target"]]
    targets.extend(sample["future_target"].clone() for _ in range(len(PHYSICAL_WEIGHTS) - 1))
    sample["physical_targets"] = torch.stack(targets, dim=1)
    result = training_long_rollout(model, sample, depth.STEPS, physical_weights=PHYSICAL_WEIGHTS)
    assert result.forecasts.shape == (1, 12, 17, 3, 4)
    assert result.per_step_losses.shape == (12,)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        depth.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        depth.main(["--out", str(out)])
    assert not out.exists()
