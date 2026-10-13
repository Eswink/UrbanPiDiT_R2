"""CI-safe checks for the capacity-width driver (trunk dim 192 -> 256 on the 2400 arm).

The capacity-width round changes exactly one factor against the registered 2400-update
wide interior-dose arm: the backbone trunk width, raised from 192 to 256 in the bound
model spec (training and scoring both read that spec), with the trunk depth held at 4.
Unlike the depth round there is **no bit-identical width expansion**: the dim-192 parent
cannot load into a dim-256 trunk, so the parent is migrated by a frozen zero-pad rule and
the width is confounded with that initialization. The fidelity-critical parts are that
only ``dim`` changes, that the frozen supervision mask, K=4 and depth 4 are unchanged, that
the zero-pad expansion copies each parent tensor's overlapping slice and zeroes the rest
without any bit-identical claim, and that the primary pin is the registered 2400-update
arm's *boundary* table (the table the reading stage actually reads). Every check runs
against synthetic trees / tiny models so this file is checked on a checkout with no
``outputs/``. Each check has a counterproof.
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

from tools.r7_round_support import round_spec  # noqa: E402

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


capacity = _load("study_r7_s3_capacity_width")
dose = capacity.dose
interior = capacity.interior
wide = capacity.wide


def _tiny_spec(dim, depth=4):
    return dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=dim,
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
    assert capacity.FORMAT == "r7-s3-capacity-width-protocol-v1"
    assert capacity.UPDATES == 2400
    assert capacity.PARENT_DIM == 192 and capacity.DIM == 256 and capacity.DEPTH == 4
    assert capacity.STEPS == 4 and capacity.REASONING_STEPS == 4
    assert (capacity.PER_SEED_SECONDS, capacity.PLANNED_SECONDS, capacity.HARD_CAP_SECONDS) \
        == (28800., 86400., 115200.)
    assert capacity.PER_SEED_SECONDS <= capacity.HARD_CAP_SECONDS
    assert capacity.PLANNED_SECONDS < capacity.HARD_CAP_SECONDS
    assert capacity.ESTIMATED_PEAK_BYTES >= capacity.MIN_ESTIMATED_PEAK_BYTES > 2**31
    assert capacity.DEVICE_MODE == "dual"
    # the registered arm this round derives from must stay untouched
    assert wide.STEPS == 4 and wide.REASONING_STEPS == 4
    assert wide.UPDATES == 800
    assert tuple(wide.PHYSICAL_WEIGHTS) == PHYSICAL_WEIGHTS
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.SEEDS == (41, 42, 43)
    assert capacity.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert str(capacity.PREVIOUS_DOSE_ARM_RUN).endswith("outputs/r7_s3_wide_interior_dose_20261009_attempt02")
    assert str(capacity.DEFAULT_OUT).endswith("outputs/r7_s3_capacity_width_20261013_attempt01")


def test_only_the_trunk_width_changed():
    _check_constants()


@pytest.mark.parametrize("attr, wrong", [("DIM", 192), ("PARENT_DIM", 256), ("DEPTH", 8),
                                         ("STEPS", 8), ("REASONING_STEPS", 8), ("UPDATES", 800),
                                         ("PER_SEED_SECONDS", 12600.), ("PLANNED_SECONDS", 36000.),
                                         ("HARD_CAP_SECONDS", 54000.), ("ESTIMATED_PEAK_BYTES", 2**31),
                                         ("DEVICE_MODE", "single"),
                                         ("FORMAT", "r7-s3-capacity-depth-protocol-v1")])
def test_a_corrupted_constant_fails_the_checks(monkeypatch, attr, wrong):
    """Counterproof: the checks are sensitive to each frozen constant."""
    _check_constants()
    monkeypatch.setattr(capacity, attr, wrong)
    with pytest.raises(AssertionError):
        _check_constants()


def test_capacity_spec_changes_only_dim(monkeypatch):
    """Counterproof that the spec builder touches exactly one field."""
    base = _tiny_spec(192)
    monkeypatch.setattr(capacity.recipe._shared(), "archived_process_spec",
                        lambda: (dict(base), {}, {}))
    spec = capacity.capacity_spec()
    assert spec["dim"] == capacity.DIM == 256
    assert spec["depth"] == capacity.DEPTH == 4
    assert {k: v for k, v in spec.items() if k != "dim"} == {k: v for k, v in base.items() if k != "dim"}
    assert base["dim"] == 192, "the archived spec must not be mutated in place"


def test_capacity_spec_refuses_a_spec_that_is_not_dim192(monkeypatch):
    monkeypatch.setattr(capacity.recipe._shared(), "archived_process_spec",
                        lambda: (_tiny_spec(128), {}, {}))
    with pytest.raises(RuntimeError):
        capacity.capacity_spec()


def test_protocol_constants_carry_width_but_keep_depth_and_k4():
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
    assert "dim 192 -> 256" in body["difference"] and "depth 4" in body["difference"]
    assert "K=4" in body["difference"]
    assert any("no bit-identical width expansion" in text for text in body["limitations"])


def test_width_expansion_block_is_the_frozen_rule():
    block = capacity.width_expansion_block()
    assert block["parent_dim"] == 192 and block["target_dim"] == 256
    assert block["bit_identical"] is False
    assert block["confound"] and "LayerNorm" in block["confound"]
    assert block["shape_selfcheck"]["batch_shape"] == list(capacity.EQUIVALENCE_SHAPE)
    assert capacity.width_expansion_block() == block


def test_expand_parent_state_width_zero_pads_the_overlap(monkeypatch):
    """Counterproof: the overlap slice is copied, everything else is zero, no bit-identity."""
    monkeypatch.setattr(capacity, "EQUIVALENCE_SHAPE", (1, 2, 17, 8, 8))
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(8)).state_dict()
    model, state, receipt = capacity.expand_parent_state_width(parent_state, _tiny_spec(16))
    target_shapes = make_model("process", _tiny_spec(16)).state_dict()
    assert set(state) == set(target_shapes)
    for key, tgt in target_shapes.items():
        assert tuple(state[key].shape) == tuple(tgt.shape)
    # a 1-D parameter: the first 8 entries are copied, the high 8 are zero
    src = parent_state["backbone.encoder.norm.weight"]
    got = state["backbone.encoder.norm.weight"]
    assert torch.equal(got[:8], src) and torch.count_nonzero(got[8:]) == 0
    # a 2-D parameter: the overlapping block is copied, the rest is zero
    src2 = parent_state["backbone.encoder.blocks.0.qkv.weight"]
    got2 = state["backbone.encoder.blocks.0.qkv.weight"]
    assert torch.equal(got2[:src2.shape[0], :src2.shape[1]], src2)
    assert torch.count_nonzero(got2[src2.shape[0]:, :]) == 0
    assert torch.count_nonzero(got2[:, src2.shape[1]:]) == 0
    # the expanded state loads strictly and is deterministic
    model.load_state_dict(state, strict=True)
    _, _, again = capacity.expand_parent_state_width(parent_state, _tiny_spec(16))
    assert again["expanded_state_sha256"] == receipt["expanded_state_sha256"]
    assert receipt["parent_dim"] == 8 and receipt["target_dim"] == 16
    assert receipt["non_block_keys_unchanged"] is True
    # there is no bit-identical width expansion, so no equivalence claim may be made
    assert "encoder_bit_identical" not in receipt
    assert receipt["encoder_output_finite"] is True


def test_expand_parent_state_width_keeps_a_missing_key_at_random_init(monkeypatch):
    """Counterproof: a key the parent lacks is left at the target's random init."""
    monkeypatch.setattr(capacity, "EQUIVALENCE_SHAPE", (1, 2, 17, 8, 8))
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(8)).state_dict()
    del parent_state["backbone.lead_time.net.0.weight"]
    _, _, receipt = capacity.expand_parent_state_width(parent_state, _tiny_spec(16))
    assert "backbone.lead_time.net.0.weight" in receipt["left_at_random"]
    assert receipt["non_block_keys_unchanged"] is False


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


def test_dim_enters_the_bound_contract_and_signature():
    """The width lives in the model spec the contract binds verbatim."""
    from training.r7_experiment import canonical_digest
    narrow, wide_spec = _tiny_spec(192), _tiny_spec(256)
    assert canonical_digest(narrow) != canonical_digest(wide_spec)
    assert {k: v for k, v in narrow.items() if k != "dim"} == {k: v for k, v in wide_spec.items()
                                                               if k != "dim"}


def test_seed_binds_dim256_into_training_and_scoring(tmp_path, monkeypatch):
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
    spec256 = _tiny_spec(256)
    expansion = {"parent_dim": 192, "target_dim": 256, "encoder_output_finite": True}
    monkeypatch.setattr(capacity, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"),
                                            "p" * 64, expansion))

    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": spec256}}}}
    output = tmp_path / "run"
    output.mkdir()
    capacity._seed(output, body, 41, None)

    assert captured["training"]["steps"] == 4
    assert captured["training"]["updates"] == 2400
    assert captured["training"]["physical_weights"] == PHYSICAL_WEIGHTS
    assert captured["training"]["contract"]["model"]["dim"] == 256
    assert [call["reasoning_steps"] for call in captured["scoring"]] == [4] * 5
    assert [call["lead_hours"] for call in captured["scoring"]] == [(lead,) for lead in wide.EVALUATION_LEADS]
    receipt = json.loads((output / "seed41_receipt.json").read_text(encoding="utf-8"))
    assert receipt["dim"] == 256 and receipt["updates"] == 2400 and receipt["reasoning_steps"] == 4
    assert receipt["width_expansion"] == expansion
    assert "zero-pad dim 192 -> 256 expansion" in captured["training"]["contract"]["initialization"]["description"]


def test_seed_counterproof_a_wrong_k_contract_is_refused(tmp_path, monkeypatch):
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
                                            "p" * 64, {"parent_dim": 192}))
    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": _tiny_spec(256)}}}}
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(RuntimeError, match="dim-256"):
        capacity._seed(output, body, 41, None)


def _patch_validate(monkeypatch, candidate):
    """Neutralise the store/GPU reads so validate_protocol is exercised on a bare checkout."""
    import data.r7_long_rollout_dataset as dataset
    import training.r7_arm_harness as harness
    import training.r7_experiment as experiment
    import training.r7_long_rollout_runner as runner
    monkeypatch.setattr(experiment, "dataset_identity",
                        lambda manifest: ("train-id", "x") if str(manifest) == str(wide.TRAIN_MANIFEST)
                        else ("val-id", "x"))
    monkeypatch.setattr(experiment, "model_code_digest", lambda: "m" * 64)
    monkeypatch.setattr(runner, "training_code_digest", lambda: "t" * 64)
    monkeypatch.setattr(dataset, "preflight_long_rollout_windows", lambda *a, **k: {"stub": 1})
    monkeypatch.setattr(wide, "source_sha256", lambda: "s" * 64)
    monkeypatch.setattr(capacity.interior, "supervision_block", lambda: {"stub": 1})
    monkeypatch.setattr(capacity.devices, "assignment_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(capacity, "_expected_candidate", lambda: candidate)
    # ``execution_files()`` includes an outputs/ path that does not exist on a bare checkout,
    # so restrict the inventory to files that are actually present and stub the hasher: the
    # assertions under test are about the frozen fields, not about on-disk source hashing.
    real_files = [name for name in capacity.execution_files() if (ROOT / name).is_file()]
    monkeypatch.setattr(capacity, "execution_files", lambda: real_files)
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)


def _protocol_body(candidate):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = {
        "execution_files_sha256": {name: sha256_file(ROOT / name) for name in capacity.execution_files()},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": capacity.FORMAT,
        "recipe": {"updates": capacity.UPDATES, "steps": capacity.STEPS},
        "evaluation": {"reasoning_steps": capacity.REASONING_STEPS},
        "width_expansion": capacity.width_expansion_block(),
        "arms": {"candidate": candidate,
                 "narrow_control": {"record_id": wide.NARROW_CONTROL_RECORD, "updates": 800,
                                    "seeds": list(wide.SEEDS)},
                 "previous_dose_arm": {"record_id": capacity.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": capacity.UPDATES, "seeds": list(wide.SEEDS)}},
        "planned_seconds": capacity.PLANNED_SECONDS, "hard_cap_seconds": capacity.HARD_CAP_SECONDS,
        "per_seed_seconds": capacity.PER_SEED_SECONDS,
        "estimated_peak_bytes": capacity.ESTIMATED_PEAK_BYTES,
        "headroom_margin_bytes": capacity.HEADROOM_MARGIN_BYTES,
        "scientific_claim": False, "test_read": False,
        "test_manifest_never_read": str(wide.TEST_MANIFEST),
    }
    body["protocol_sha256"] = canonical_digest(body)
    return body


def _write_protocol(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_validate_protocol_accepts_the_frozen_candidate(tmp_path, monkeypatch):
    candidate = {"model": {"spec": _tiny_spec(256)}}
    _patch_validate(monkeypatch, candidate)
    body = capacity.validate_protocol(_write_protocol(tmp_path, _protocol_body(candidate)))
    assert body["format"] == capacity.FORMAT
    assert body["width_expansion"] == capacity.width_expansion_block()


@pytest.mark.parametrize("spec", [_tiny_spec(128), _tiny_spec(256, depth=8)])
def test_validate_protocol_refuses_a_wrong_dim_or_depth(tmp_path, monkeypatch, spec):
    candidate = {"model": {"spec": spec}}
    _patch_validate(monkeypatch, candidate)
    _write_protocol(tmp_path, _protocol_body(candidate))
    with pytest.raises(RuntimeError, match="dim 256 depth 4"):
        capacity.validate_protocol(tmp_path)


def test_validate_protocol_refuses_a_changed_width_expansion(tmp_path, monkeypatch):
    from training.r7_experiment import canonical_digest
    candidate = {"model": {"spec": _tiny_spec(256)}}
    _patch_validate(monkeypatch, candidate)
    body = _protocol_body(candidate)
    body["width_expansion"] = {**body["width_expansion"], "bit_identical": True}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    _write_protocol(tmp_path, body)
    with pytest.raises(RuntimeError, match="width-expansion"):
        capacity.validate_protocol(tmp_path)


def test_training_path_accepts_a_wide_trunk_on_a_cpu_tiny_model():
    """CPU smoke: the real rollout/contract path accepts a dim-16 trunk (no GPU)."""
    from test_r7_m4_autoregressive_rollout import batch, tiny_spec
    from training.r7_experiment import make_model
    from training.r7_long_rollout import training_long_rollout

    torch.manual_seed(12)
    model = make_model("process", dict(tiny_spec("process", parent_detach=True), dim=16)).train()
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


def _load_devices():
    from training import r7_device_selection as devices
    return devices


def _two_card_probe(uuids=("GPU-a", "GPU-b"), free_mib=30000):
    def probe(argv):
        assert argv[0] == "nvidia-smi"
        body = "".join(f"{uuid}, {free_mib}, 24576\n" for uuid in uuids)
        return {"exit_code": 0, "stdout": body, "stderr": ""}
    return probe


def test_round_spec_matches_the_driver_and_device_mode():
    """The width brief's round-spec must agree with the driver it names."""
    spec = round_spec(ROOT / "docs" / "goals" / "s3-capacity-width.md")
    assert spec["arm"] == "s3-capacity-width"
    assert spec["driver"] == "scripts/study_r7_s3_capacity_width.py"
    assert spec["control_record"] == capacity.PREVIOUS_DOSE_ARM_RECORD
    assert spec["seeds"] == list(capacity.SEEDS)
    assert spec["test_path"] == ["tests/test_r7_s3_capacity_width.py"]
    assert spec["device_mode"] == capacity.DEVICE_MODE == "dual"


def test_device_mode_is_dual_and_the_round_spec_agrees():
    """The driver and the round-spec must not disagree about how many cards the round uses."""
    devices = _load_devices()
    assert capacity.DEVICE_MODE == devices.MODE_DUAL


def test_single_mode_freezes_one_card_and_keeps_the_legacy_rule():
    """``single`` must stay byte-compatible with every already-registered arm."""
    devices = _load_devices()
    body = {"gpu_uuid": "GPU-a", **devices.protocol_fields("single", single_uuid="GPU-a")}
    assert body["gpu_uuids"] == ["GPU-a"] and body["device_mode"] == "single"
    assert devices.frozen_uuids(body) == ["GPU-a"]
    assert devices.assignment_is_valid(body, "GPU-a") is True
    assert devices.assignment_is_valid(body, "GPU-b") is False
    legacy = {"gpu_uuid": "GPU-a"}
    assert devices.assignment_is_valid(legacy, "GPU-a") is True
    assert devices.assignment_is_valid(legacy, "GPU-b") is False


def test_dual_mode_lists_the_declared_card_first_and_still_accepts_it():
    devices = _load_devices()
    body = {"gpu_uuid": "GPU-a",
            **devices.protocol_fields("dual", single_uuid="GPU-a", runner=_two_card_probe())}
    assert body["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert body["neighbour_reserve_bytes"] == devices.NEIGHBOUR_RESERVE_BYTES
    assert devices.assignment_is_valid(body, "GPU-a") is True
    assert devices.assignment_is_valid(body, "GPU-b") is False


def test_a_worker_refuses_a_card_outside_the_frozen_list(monkeypatch):
    """Counterproof: an unexpected CUDA_VISIBLE_DEVICES is a hard error, never a fallback."""
    devices = _load_devices()
    body = {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-a", "GPU-b"]}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    assert devices.assigned_uuid(body) == "GPU-b"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-z")
    with pytest.raises(RuntimeError, match="not in the frozen device list"):
        devices.assigned_uuid(body)
    with pytest.raises(RuntimeError, match="declares no device"):
        devices.assigned_uuid({})


def test_seed_order_rotates_over_the_frozen_cards():
    """Three seeds over two cards must not pile every seed onto the declared card."""
    devices = _load_devices()
    assert devices.seed_order(["GPU-a", "GPU-b"], 0) == ["GPU-a", "GPU-b"]
    assert devices.seed_order(["GPU-a", "GPU-b"], 1) == ["GPU-b", "GPU-a"]
    assert devices.seed_order(["GPU-a", "GPU-b"], 2) == ["GPU-a", "GPU-b"]
