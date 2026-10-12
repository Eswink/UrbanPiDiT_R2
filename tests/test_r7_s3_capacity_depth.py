"""CI-safe checks for the capacity-depth driver (trunk depth 4 -> 8 on the 2400 arm).

The capacity-depth round changes exactly one factor against the registered
2400-update wide interior-dose arm: the backbone trunk depth, raised from 4 to 8
in the bound model spec (training and scoring both read that spec). The
fidelity-critical parts are that only ``depth`` changes, that the frozen
supervision mask and K=4 are unchanged, that the depth-4 parent is expanded by the
frozen identity-gated rule (each added block mirrors block ``i - 4`` with zeroed
output projections) and the expansion is bit-identical on a fixed synthetic batch,
and that the primary pin is the registered 2400-update arm's *boundary* table (the
table the reading stage actually reads). Every check is exercised against
synthetic trees / tiny models so this file is checked on a checkout with no
``outputs/`` -- the earlier dose round turned CI red exactly because its test read
a real ``outputs/`` artifact. Each check has a counterproof.
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


capacity = _load("study_r7_s3_capacity_depth")
dose = capacity.dose
interior = capacity.interior
wide = capacity.wide


def _tiny_spec(depth):
    return dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=8,
                depth=depth, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, detach_between_steps=False,
                anchored_processes=2, free_processes=1)


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
    assert capacity.FORMAT == "r7-s3-capacity-depth-protocol-v1"
    assert capacity.UPDATES == 2400
    assert capacity.PARENT_DEPTH == 4 and capacity.DEPTH == 8
    assert capacity.STEPS == 4 and capacity.REASONING_STEPS == 4
    assert (capacity.PER_SEED_SECONDS, capacity.PLANNED_SECONDS, capacity.HARD_CAP_SECONDS) \
        == (28800., 86400., 115200.)
    assert capacity.PER_SEED_SECONDS <= capacity.HARD_CAP_SECONDS
    assert capacity.PLANNED_SECONDS < capacity.HARD_CAP_SECONDS
    assert capacity.ESTIMATED_PEAK_BYTES >= capacity.MIN_ESTIMATED_PEAK_BYTES > 2**31
    # the registered arm this round derives from must stay untouched
    assert wide.STEPS == 4 and wide.REASONING_STEPS == 4
    assert wide.UPDATES == 800
    assert tuple(wide.PHYSICAL_WEIGHTS) == PHYSICAL_WEIGHTS
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.SEEDS == (41, 42, 43)
    assert capacity.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert str(capacity.PREVIOUS_DOSE_ARM_RUN).endswith("outputs/r7_s3_wide_interior_dose_20261009_attempt02")
    assert str(capacity.DEFAULT_OUT).endswith("outputs/r7_s3_capacity_depth_20261012_attempt01")


def test_only_the_trunk_depth_changed():
    _check_constants()


@pytest.mark.parametrize("attr, wrong", [("DEPTH", 4), ("PARENT_DEPTH", 8), ("STEPS", 8),
                                         ("REASONING_STEPS", 8), ("UPDATES", 800),
                                         ("PER_SEED_SECONDS", 12600.), ("PLANNED_SECONDS", 36000.),
                                         ("HARD_CAP_SECONDS", 54000.),
                                         ("ESTIMATED_PEAK_BYTES", 2**31),
                                         ("FORMAT", "r7-s3-reasoning-depth-protocol-v1")])
def test_a_corrupted_constant_fails_the_checks(monkeypatch, attr, wrong):
    """Counterproof: the checks are sensitive to each frozen constant."""
    _check_constants()
    monkeypatch.setattr(capacity, attr, wrong)
    with pytest.raises(AssertionError):
        _check_constants()


def test_capacity_spec_changes_only_depth(monkeypatch):
    """Counterproof that the spec builder touches exactly one field."""
    base = _tiny_spec(4)
    monkeypatch.setattr(capacity.recipe._shared(), "archived_process_spec",
                        lambda: (dict(base), {}, {}))
    spec = capacity.capacity_spec()
    assert spec["depth"] == capacity.DEPTH == 8
    assert {k: v for k, v in spec.items() if k != "depth"} == {k: v for k, v in base.items() if k != "depth"}
    assert base["depth"] == 4, "the archived spec must not be mutated in place"


def test_capacity_spec_refuses_a_spec_that_is_not_depth4(monkeypatch):
    monkeypatch.setattr(capacity.recipe._shared(), "archived_process_spec",
                        lambda: (_tiny_spec(8), {}, {}))
    with pytest.raises(RuntimeError):
        capacity.capacity_spec()


def test_protocol_constants_carry_depth_but_keep_k4():
    body = capacity.protocol_constants()
    assert body["format"] == capacity.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(PHYSICAL_WEIGHTS)
    assert body["recipe"]["lr"] == 2e-5 and body["recipe"]["warmup"] == 10
    assert body["recipe"]["bf16"] is False
    assert body["supervision"] == interior.supervision_block()
    assert body["supervision"]["runner_block"]["selected_cells"] == 4225
    assert body["budgets"] == {"planned_seconds": 86400., "hard_cap_seconds": 115200.,
                               "per_seed_seconds": 28800.}
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "depth 4 -> 8" in body["difference"] and "K=4" in body["difference"]


def test_depth_expansion_block_is_the_frozen_rule():
    block = capacity.depth_expansion_block()
    assert block["parent_depth"] == 4 and block["target_depth"] == 8
    assert block["added_blocks"] == [4, 5, 6, 7]
    assert block["gate_keys"] == list(capacity.GATE_KEYS)
    assert block["equivalence"]["batch_shape"] == list(capacity.EQUIVALENCE_SHAPE)
    assert capacity.depth_expansion_block() == block


def test_expand_parent_state_is_bit_identical_and_deterministic(monkeypatch):
    """The depth-8 trunk must start at the registered depth-4 function, bit for bit."""
    monkeypatch.setattr(capacity, "EQUIVALENCE_SHAPE", (1, 2, 17, 8, 8))
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent = make_model("process", _tiny_spec(4))
    parent_state = parent.state_dict()
    model, state, receipt = capacity.expand_parent_state(parent_state, _tiny_spec(8))
    assert receipt["encoder_bit_identical"] is True
    assert receipt["added_blocks"] == [4, 5, 6, 7]
    assert receipt["parent_depth"] == 4 and receipt["target_depth"] == 8
    assert receipt["non_block_keys_unchanged"] is True
    # deterministic: expanding the same parent again gives the same state digest
    _, _, again = capacity.expand_parent_state(parent_state, _tiny_spec(8))
    assert again["expanded_state_sha256"] == receipt["expanded_state_sha256"]
    assert again["encoder_output_sha256"] == receipt["encoder_output_sha256"]
    # the expanded state loads strictly into the depth-8 model
    model.load_state_dict(state, strict=True)
    # the gated projections of the added blocks really are zero
    for block in (4, 5, 6, 7):
        for gate in capacity.GATE_KEYS:
            assert torch.count_nonzero(state[f"backbone.encoder.blocks.{block}.{gate}"]) == 0


def test_expand_parent_state_counterproof_ungated_blocks_are_not_identity(monkeypatch):
    """Counterproof: without zeroing the gates the expansion is not bit-identical."""
    monkeypatch.setattr(capacity, "EQUIVALENCE_SHAPE", (1, 2, 17, 8, 8))
    monkeypatch.setattr(capacity, "GATE_KEYS", ())
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(4)).state_dict()
    with pytest.raises(RuntimeError, match="bit-identical"):
        capacity.expand_parent_state(parent_state, _tiny_spec(8))


def test_expand_parent_state_counterproof_a_wrong_spec_is_refused(monkeypatch):
    monkeypatch.setattr(capacity, "EQUIVALENCE_SHAPE", (1, 2, 17, 8, 8))
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(4)).state_dict()
    with pytest.raises(RuntimeError):
        capacity.expand_parent_state(parent_state, _tiny_spec(6))


def test_previous_arm_pins_read_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "PREVIOUS_DOSE_ARM_RUN", _synthetic_run(tmp_path))
    pins = capacity.previous_arm_pins()
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
            assert sha256_file(Path(entry["dir"]) / "boundary_rmse.csv") \
                == entry["boundary_rmse_csv_sha256"]
            assert sha256_file(Path(entry["dir"]) / "rmse.csv") \
                != entry["boundary_rmse_csv_sha256"]


def test_previous_arm_pins_refuse_a_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "PREVIOUS_DOSE_ARM_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        capacity.previous_arm_pins()


def test_previous_arm_pins_refuse_a_cohort_that_differs_from_the_pin(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    receipt_path = run / "seed42_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evaluations"]["72"]["n_evaluated"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(capacity, "PREVIOUS_DOSE_ARM_RUN", run)
    with pytest.raises(RuntimeError):
        capacity.previous_arm_pins()


def test_depth_enters_the_bound_contract_and_signature():
    """The depth lives in the model spec the contract binds verbatim.

    The signature is the digest of the bound contract, so a depth-8 spec must change
    it while every other field stays identical.
    """
    from training.r7_experiment import canonical_digest
    four, eight = _tiny_spec(4), _tiny_spec(8)
    assert canonical_digest(four) != canonical_digest(eight)
    assert {k: v for k, v in four.items() if k != "depth"} == {k: v for k, v in eight.items()
                                                               if k != "depth"}


def test_seed_binds_depth8_into_training_and_scoring(tmp_path, monkeypatch):
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
    spec8 = _tiny_spec(8)
    expansion = {"encoder_bit_identical": True, "added_blocks": [4, 5, 6, 7]}
    monkeypatch.setattr(capacity, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"),
                                            "p" * 64, expansion))

    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": spec8}}}}
    output = tmp_path / "run"
    output.mkdir()
    capacity._seed(output, body, 41, None)

    assert captured["training"]["steps"] == 4
    assert captured["training"]["updates"] == 2400
    assert captured["training"]["physical_weights"] == PHYSICAL_WEIGHTS
    assert captured["training"]["contract"]["model"]["depth"] == 8
    assert [call["reasoning_steps"] for call in captured["scoring"]] == [4] * 5
    assert [call["lead_hours"] for call in captured["scoring"]] == [(lead,) for lead in wide.EVALUATION_LEADS]
    receipt = json.loads((output / "seed41_receipt.json").read_text(encoding="utf-8"))
    assert receipt["depth"] == 8 and receipt["updates"] == 2400 and receipt["reasoning_steps"] == 4
    assert receipt["depth_expansion"] == expansion


def test_seed_counterproof_a_depth4_contract_is_refused(tmp_path, monkeypatch):
    """Counterproof: a training report bound at K=8 (not the frozen K=4) must abort."""
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
                            "contract": {"steps": 8}, "elapsed_seconds": 1.0}

    monkeypatch.setattr(runner, "fine_tune_long_rollout", fake_training)
    monkeypatch.setattr(evaluator, "evaluate_local", lambda *a, **k: pytest.fail("scoring must not run"))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(interior, "supervision_mask", lambda: SimpleNamespace(to=lambda *a, **k: "mask"))
    monkeypatch.setattr(capacity, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"),
                                            "p" * 64, {"encoder_bit_identical": True}))
    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": _tiny_spec(8)}}}}
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(RuntimeError, match="depth-8"):
        capacity._seed(output, body, 41, None)


def test_training_path_accepts_depth8_on_a_cpu_tiny_model():
    """CPU smoke: the real rollout/contract path accepts a depth-8 trunk (no GPU)."""
    from test_r7_m4_autoregressive_rollout import batch, tiny_spec
    from training.r7_experiment import make_model
    from training.r7_long_rollout import training_long_rollout

    torch.manual_seed(12)
    model = make_model("process", dict(tiny_spec("process", parent_detach=True), depth=8)).train()
    sample = batch(17)
    targets = [sample["atmos_target"]]
    targets.extend(sample["future_target"].clone() for _ in range(len(PHYSICAL_WEIGHTS) - 1))
    sample["physical_targets"] = torch.stack(targets, dim=1)
    result = training_long_rollout(model, sample, capacity.STEPS, physical_weights=PHYSICAL_WEIGHTS)
    assert result.forecasts.shape == (1, 12, 17, 3, 4)
    assert result.per_step_losses.shape == (12,)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        capacity.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        capacity.main(["--out", str(out)])
    assert not out.exists()


def test_driver_exposes_the_round_driver_contract_names():
    """The campaign round wrapper refuses a driver missing these; keep them exposed."""
    for name in ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS"):
        assert hasattr(capacity, name), name
    assert tuple(capacity.SEEDS) == (41, 42, 43)
    assert tuple(capacity.SEEDS) == tuple(wide.SEEDS)
