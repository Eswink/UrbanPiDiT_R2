"""CI-safe checks for the H-A process-state input-conditioning driver (per-sample ``P_0``).

The H-A round changes exactly one factor against the registered 2400-update wide interior-dose
arm: the process-state initial value ``P_0``, made to depend on each sample's own context via
``process_init_conditioning`` False -> True, with the trunk (dim 192, depth 4), K=4, the input
region and the supervision mask held fixed. The pathway is built last under ``isolated_stream()``
so the "off" path is bit-identical to the previous implementation and the parent (dim 192, depth
4) loads unchanged except for the four new ``process_init_proj`` keys. The fidelity-critical parts
are that only ``process_init_conditioning`` changes, that the expansion adds exactly those four
keys (no more, no fewer), that the switch-off / zeroed-projection paths are bit-identical to the
registered arm, and that the frozen ``process_init`` block is pinned. Every check runs against
synthetic specs / tiny models so this file is checked on a checkout with no ``outputs/``. Each
check has a counterproof.
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


process_init = _load("study_r7_s3_process_init")
dose = process_init.dose
interior = process_init.interior
wide = process_init.wide


def _tiny_spec(dim, depth=4, process_init_conditioning=False):
    return dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=dim,
                depth=depth, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, detach_between_steps=False,
                anchored_processes=2, free_processes=1,
                process_init_conditioning=process_init_conditioning)


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
    assert process_init.FORMAT == "r7-s3-process-init-protocol-v1"
    assert process_init.UPDATES == 2400
    assert process_init.DIM == 192 and process_init.DEPTH == 4
    assert process_init.STEPS == 4 and process_init.REASONING_STEPS == 4
    assert process_init.NEW_PARAMS == 37440
    assert set(process_init.NEW_KEYS) == {"process_init_proj.0.weight", "process_init_proj.0.bias",
                                          "process_init_proj.1.weight", "process_init_proj.1.bias"}
    assert (process_init.PER_SEED_SECONDS, process_init.PLANNED_SECONDS, process_init.HARD_CAP_SECONDS) \
        == (28800., 86400., 115200.)
    assert process_init.PER_SEED_SECONDS <= process_init.HARD_CAP_SECONDS
    assert process_init.PLANNED_SECONDS < process_init.HARD_CAP_SECONDS
    assert process_init.ESTIMATED_PEAK_BYTES == 12198 * 2**20
    assert process_init.ESTIMATED_PEAK_BYTES >= process_init.MIN_ESTIMATED_PEAK_BYTES > 2**31
    assert process_init.DEVICE_MODE == "dual"
    # the registered arm this round derives from must stay untouched
    assert wide.STEPS == 4 and wide.REASONING_STEPS == 4
    assert wide.UPDATES == 800
    assert tuple(wide.PHYSICAL_WEIGHTS) == PHYSICAL_WEIGHTS
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.SEEDS == (41, 42, 43)
    assert process_init.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert str(process_init.PREVIOUS_DOSE_ARM_RUN).endswith("outputs/r7_s3_wide_interior_dose_20261009_attempt02")
    assert str(process_init.DEFAULT_OUT).endswith("outputs/r7_s3_process_init_20261014_attempt01")


def test_only_the_process_init_factor_changed():
    _check_constants()


@pytest.mark.parametrize("attr, wrong", [("DIM", 256), ("DEPTH", 8), ("STEPS", 8),
                                         ("REASONING_STEPS", 8), ("UPDATES", 800), ("NEW_PARAMS", 1),
                                         ("PER_SEED_SECONDS", 12600.), ("PLANNED_SECONDS", 36000.),
                                         ("HARD_CAP_SECONDS", 54000.),
                                         ("ESTIMATED_PEAK_BYTES", 12197 * 2**20),
                                         ("DEVICE_MODE", "single"),
                                         ("FORMAT", "r7-s3-capacity-width-protocol-v1")])
def test_a_corrupted_constant_fails_the_checks(monkeypatch, attr, wrong):
    """Counterproof: the checks are sensitive to each frozen constant."""
    _check_constants()
    monkeypatch.setattr(process_init, attr, wrong)
    with pytest.raises(AssertionError):
        _check_constants()


def test_capacity_spec_changes_only_process_init(monkeypatch):
    """Counterproof that the spec builder touches exactly one field."""
    base = _tiny_spec(192)
    monkeypatch.setattr(process_init.recipe._shared(), "archived_process_spec",
                        lambda: (dict(base), {}, {}))
    spec = process_init.capacity_spec()
    assert spec["process_init_conditioning"] is True
    assert spec["dim"] == process_init.DIM == 192
    assert spec["depth"] == process_init.DEPTH == 4
    assert {k: v for k, v in spec.items() if k != "process_init_conditioning"} \
        == {k: v for k, v in base.items() if k != "process_init_conditioning"}
    assert base["process_init_conditioning"] is False, "the archived spec must not be mutated in place"


def test_capacity_spec_refuses_a_spec_that_is_not_dim192(monkeypatch):
    monkeypatch.setattr(process_init.recipe._shared(), "archived_process_spec",
                        lambda: (_tiny_spec(128), {}, {}))
    with pytest.raises(RuntimeError):
        process_init.capacity_spec()


def test_capacity_spec_refuses_an_already_conditioned_spec(monkeypatch):
    monkeypatch.setattr(process_init.recipe._shared(), "archived_process_spec",
                        lambda: (_tiny_spec(192, process_init_conditioning=True), {}, {}))
    with pytest.raises(RuntimeError):
        process_init.capacity_spec()


def test_protocol_constants_carry_process_init_but_keep_dim_depth_k4():
    body = process_init.protocol_constants()
    assert body["format"] == process_init.FORMAT
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
    assert "process_init_conditioning" in body["difference"]
    assert "dim 192" in body["difference"] and "K=4" in body["difference"]


def test_process_init_block_is_the_frozen_rule():
    block = process_init.process_init_block()
    assert block["bit_identical_off_path"] is True
    assert block["new_params"] == 37440
    assert set(block["new_keys"]) == set(process_init.NEW_KEYS)
    assert "zero" in block["positive_control"].lower()
    assert "matched_generic" in block["negative_control"]
    assert process_init.process_init_block() == block


def test_expand_parent_state_process_init_adds_exactly_the_four_keys():
    """Counterproof: only the four H-A keys are new and every shared key is untouched."""
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(8)).state_dict()
    target = _tiny_spec(8, process_init_conditioning=True)
    model, state, receipt = process_init.expand_parent_state_process_init(parent_state, target)
    assert set(state) == set(make_model("process", target).state_dict())
    assert receipt["new_keys"] == sorted(process_init.NEW_KEYS)
    assert receipt["parent_keys"] == len(parent_state)
    assert receipt["new_params"] == 88  # LayerNorm(8) 16 + Linear(8,8) 72
    assert receipt["proj_zeroing_is_identity"] is True
    assert receipt["bit_identical_off_path"] is True
    for key, value in parent_state.items():
        assert torch.equal(state[key], value)
    model.load_state_dict(state, strict=True)


def test_expand_parent_state_process_init_refuses_a_missing_key():
    """Counterproof: if the parent already carries a new key, only 3 are added -> abort."""
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(8)).state_dict()
    parent_state["process_init_proj.1.weight"] = torch.zeros(8, 8)
    with pytest.raises(RuntimeError, match="exactly"):
        process_init.expand_parent_state_process_init(
            parent_state, _tiny_spec(8, process_init_conditioning=True))


def test_expand_parent_state_process_init_refuses_an_extra_key():
    """Counterproof: a shared key absent from the parent makes 5 new keys -> abort."""
    from training.r7_experiment import make_model
    torch.manual_seed(7)
    parent_state = make_model("process", _tiny_spec(8)).state_dict()
    del parent_state["backbone.lead_time.net.0.weight"]
    with pytest.raises(RuntimeError, match="exactly"):
        process_init.expand_parent_state_process_init(
            parent_state, _tiny_spec(8, process_init_conditioning=True))


def test_switch_off_is_bit_identical_to_the_current_implementation():
    """The pathway is built last under isolated_stream(): the off path moves no existing weight."""
    from training.r7_experiment import make_model
    torch.manual_seed(1234)
    off = make_model("process", _tiny_spec(8)).state_dict()
    torch.manual_seed(1234)
    on = make_model("process", _tiny_spec(8, process_init_conditioning=True)).state_dict()
    assert set(on) - set(off) == set(process_init.NEW_KEYS)
    assert set(off) - set(on) == set()
    assert all(torch.equal(off[key], on[key]) for key in off)


def test_zeroing_the_projection_is_identically_zero():
    """The positive control: a zeroed Linear makes the injected term exactly zero."""
    from training.r7_experiment import make_model
    torch.manual_seed(0)
    model = make_model("process", _tiny_spec(8, process_init_conditioning=True))
    with torch.no_grad():
        model.process_init_proj[1].weight.zero_()
        model.process_init_proj[1].bias.zero_()
        pooled = model.process_init_proj(torch.randn(3, 5, 8).mean(dim=1))
    assert torch.count_nonzero(pooled) == 0


def test_positive_control_reports_bit_identity_and_zeroing():
    control = process_init._positive_control(_tiny_spec(8, process_init_conditioning=True))
    assert control["bit_identical_off_path"] is True
    assert control["proj_zeroing_is_identity"] is True
    assert control["new_keys"] == sorted(process_init.NEW_KEYS)
    assert control["zeroed_proj_output_abs_max"] == 0.0


def test_forward_with_a_zeroed_projection_matches_the_off_path():
    """The strongest identity: a zeroed projection reproduces the off-path forecast bit for bit."""
    from test_r7_m4_autoregressive_rollout import batch
    from training.r7_experiment import make_model
    torch.manual_seed(1234)
    off = make_model("process", _tiny_spec(8)).eval()
    torch.manual_seed(1234)
    on = make_model("process", _tiny_spec(8, process_init_conditioning=True)).eval()
    sample = batch(17)
    with torch.no_grad():
        reference = off(sample, reasoning_steps=4).forecast
        conditioned = on(sample, reasoning_steps=4).forecast
    assert not torch.equal(reference, conditioned), "switching the pathway on must change the forward"
    with torch.no_grad():
        on.process_init_proj[1].weight.zero_()
        on.process_init_proj[1].bias.zero_()
        zeroed = on(sample, reasoning_steps=4).forecast
    assert torch.equal(reference, zeroed)


def test_previous_arm_pins_read_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(process_init, "PREVIOUS_DOSE_ARM_RUN", _synthetic_run(tmp_path))
    pins = process_init.previous_arm_pins()
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
    monkeypatch.setattr(process_init, "PREVIOUS_DOSE_ARM_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        process_init.previous_arm_pins()


def test_previous_arm_pins_refuse_a_cohort_that_differs_from_the_pin(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    receipt_path = run / "seed42_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evaluations"]["72"]["n_evaluated"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(process_init, "PREVIOUS_DOSE_ARM_RUN", run)
    with pytest.raises(RuntimeError):
        process_init.previous_arm_pins()


def test_seed_binds_process_init_into_training_and_scoring(tmp_path, monkeypatch):
    """The single factor lives in the model spec the contract binds verbatim."""
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
    spec = _tiny_spec(8, process_init_conditioning=True)
    expansion = {"new_keys": sorted(process_init.NEW_KEYS), "proj_zeroing_is_identity": True,
                 "bit_identical_off_path": True}
    monkeypatch.setattr(process_init, "_parent_model",
                        lambda body, seed: (SimpleNamespace(), {"w": 1}, Path("/tmp/parent.pt"),
                                            "p" * 64, expansion))

    body = {"train_data_identity": "d" * 64, "source_sha256": "s" * 64, "protocol_sha256": "p" * 64,
            "training_code_sha256": "t" * 64,
            "supervision": {"runner_block": {"kind": "spatial_mask", "selected_cells": 4225}},
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "arms": {"candidate": {"model": {"spec": spec}}}}
    output = tmp_path / "run"
    output.mkdir()
    process_init._seed(output, body, 41, None)

    assert captured["training"]["steps"] == 4
    assert captured["training"]["updates"] == 2400
    assert captured["training"]["physical_weights"] == PHYSICAL_WEIGHTS
    assert captured["training"]["contract"]["model"]["process_init_conditioning"] is True
    assert [call["reasoning_steps"] for call in captured["scoring"]] == [4] * 5
    assert [call["lead_hours"] for call in captured["scoring"]] == [(lead,) for lead in wide.EVALUATION_LEADS]
    receipt = json.loads((output / "seed41_receipt.json").read_text(encoding="utf-8"))
    assert receipt["dim"] == 192 and receipt["updates"] == 2400 and receipt["reasoning_steps"] == 4
    assert receipt["process_init"] == expansion
    assert "H-A process_init_conditioning=True (4 new keys)" \
        in captured["training"]["contract"]["initialization"]["description"]


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
    monkeypatch.setattr(process_init.interior, "supervision_block", lambda: {"stub": 1})
    monkeypatch.setattr(process_init.devices, "assignment_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(process_init, "_expected_candidate", lambda: candidate)
    # ``execution_files()`` includes an outputs/ path that does not exist on a bare checkout,
    # so restrict the inventory to files that are actually present and stub the hasher: the
    # assertions under test are about the frozen fields, not about on-disk source hashing.
    real_files = [name for name in process_init.execution_files() if (ROOT / name).is_file()]
    monkeypatch.setattr(process_init, "execution_files", lambda: real_files)
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)


def _protocol_body(candidate):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = {
        "execution_files_sha256": {name: sha256_file(ROOT / name) for name in process_init.execution_files()},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": process_init.FORMAT,
        "recipe": {"updates": process_init.UPDATES, "steps": process_init.STEPS},
        "evaluation": {"reasoning_steps": process_init.REASONING_STEPS},
        "process_init": process_init.process_init_block(),
        "arms": {"candidate": candidate,
                 "narrow_control": {"record_id": wide.NARROW_CONTROL_RECORD, "updates": 800,
                                    "seeds": list(wide.SEEDS)},
                 "previous_dose_arm": {"record_id": process_init.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": process_init.UPDATES, "seeds": list(wide.SEEDS)}},
        "planned_seconds": process_init.PLANNED_SECONDS, "hard_cap_seconds": process_init.HARD_CAP_SECONDS,
        "per_seed_seconds": process_init.PER_SEED_SECONDS,
        "estimated_peak_bytes": process_init.ESTIMATED_PEAK_BYTES,
        "headroom_margin_bytes": process_init.HEADROOM_MARGIN_BYTES,
        "scientific_claim": False, "test_read": False,
        "test_manifest_never_read": str(wide.TEST_MANIFEST),
    }
    body["protocol_sha256"] = canonical_digest(body)
    return body


def _write_protocol(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_validate_protocol_accepts_the_frozen_candidate(tmp_path, monkeypatch):
    candidate = {"model": {"spec": _tiny_spec(192, process_init_conditioning=True)}}
    _patch_validate(monkeypatch, candidate)
    body = process_init.validate_protocol(_write_protocol(tmp_path, _protocol_body(candidate)))
    assert body["format"] == process_init.FORMAT
    assert body["process_init"] == process_init.process_init_block()


@pytest.mark.parametrize("spec", [_tiny_spec(128, process_init_conditioning=True),
                                  _tiny_spec(192, depth=8, process_init_conditioning=True),
                                  _tiny_spec(192, process_init_conditioning=False)])
def test_validate_protocol_refuses_a_wrong_dim_depth_or_switch(tmp_path, monkeypatch, spec):
    candidate = {"model": {"spec": spec}}
    _patch_validate(monkeypatch, candidate)
    _write_protocol(tmp_path, _protocol_body(candidate))
    with pytest.raises(RuntimeError, match="process_init_conditioning on"):
        process_init.validate_protocol(tmp_path)


def test_validate_protocol_refuses_a_changed_process_init_block(tmp_path, monkeypatch):
    from training.r7_experiment import canonical_digest
    candidate = {"model": {"spec": _tiny_spec(192, process_init_conditioning=True)}}
    _patch_validate(monkeypatch, candidate)
    body = _protocol_body(candidate)
    body["process_init"] = {**body["process_init"], "bit_identical_off_path": False}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    _write_protocol(tmp_path, body)
    with pytest.raises(RuntimeError, match="process-init"):
        process_init.validate_protocol(tmp_path)


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        process_init.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        process_init.main(["--out", str(out)])
    assert not out.exists()


def test_driver_exposes_the_round_driver_contract_names():
    """The campaign round wrapper refuses a driver missing these; keep them exposed."""
    for name in ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS"):
        assert hasattr(process_init, name), name
    assert tuple(process_init.SEEDS) == (41, 42, 43)
    assert tuple(process_init.SEEDS) == tuple(wide.SEEDS)


def test_round_spec_matches_the_driver_and_device_mode():
    """The H-A brief's round-spec must agree with the driver it names."""
    spec = round_spec(ROOT / "docs" / "goals" / "s3-process-init-conditioning.md")
    assert spec["arm"] == "s3-process-init-conditioning"
    assert spec["driver"] == "scripts/study_r7_s3_process_init.py"
    assert spec["control_record"] == process_init.PREVIOUS_DOSE_ARM_RECORD
    assert spec["seeds"] == list(process_init.SEEDS)
    assert spec["test_path"] == ["tests/test_r7_s3_process_init.py"]
    assert spec["device_mode"] == process_init.DEVICE_MODE == "dual"


def test_device_mode_is_dual_and_the_round_spec_agrees():
    """The driver and the round-spec must not disagree about how many cards the round uses."""
    from training import r7_device_selection as devices
    assert process_init.DEVICE_MODE == devices.MODE_DUAL
