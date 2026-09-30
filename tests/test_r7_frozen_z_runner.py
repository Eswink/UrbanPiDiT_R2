"""CPU-only frozen-Z integration checks on synthetic, temporary fixtures.

These are engineering checks, not forecast-skill evidence. No real test split,
archived checkpoint, network request or CUDA execution is used.
"""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import default_collate

from data.synthetic_atmos import SyntheticAtmosDataset
from training import r7_evaluate as evaluator
from training import r7_frozen_z_intervention as control
from training import r7_scheduled_runner as runner
from training.r7_experiment import (
    canonical_digest, dataset_identity, load_checkpoint, make_model, rng_state,
    seed_everything,
)
from training.r7_local_runner import update_group


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(kind="process", channels=1):
    config = dict(in_channels=channels, history_steps=2, dim=8, patch_size=2,
                  depth=1, heads=2, window_size=2, dropout=0.0)
    if kind == "generic":
        config.update(latent_tokens=2, default_reasoning_steps=2)
    if kind == "process":
        config.update(anchored_processes=1, free_processes=1, default_reasoning_steps=2,
                      local_solver_state=True, positional_process_readout=True,
                      use_forecast_feedback=True)
    return config


def _options(tmp_path, kind="process"):
    return dict(kind=kind, model_config=_config(kind), data_identity="synthetic-frozen-z-only",
                output_dir=tmp_path / "run", total_updates=1, batch_size=1, steps=2,
                seed=7, lr=2e-4, clip=1.0, process_weight=0.0, device_name="cpu",
                warmup_updates=1, validation_every=0)


def _spec(config, hw=(4, 6), seed=7):
    patch = config["patch_size"]
    token_hw = tuple((size + patch - 1) // patch for size in hw)
    return control.make_spec(seed=seed, token_hw=token_hw, dim=config["dim"], patch_size=patch)


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            _assert_exact(first, second)
    else:
        assert left == right


def _restore(saved):
    contract = saved["contract"]
    model = make_model(contract["kind"], contract["model"])
    control.install_frozen_z(model, contract["intervention"])
    model.load_state_dict(saved["model"], strict=True)
    control.validate_frozen_z(model, contract["intervention"])
    return model.eval()


def _rewrite(saved, path):
    with path.open("xb") as handle:
        torch.save(saved, handle)
    return path


@pytest.mark.parametrize("kind", ["native", "generic", "process"])
def test_ordinary_defaults_keep_exact_contract_rng_optimizer_and_update(tmp_path, monkeypatch, kind):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path / "default", kind)
    seed_everything(options["seed"])
    reference = make_model(kind, options["model_config"]).train()
    assert all(parameter.requires_grad for parameter in reference.parameters())
    optimizer = torch.optim.AdamW(reference.parameters(), lr=options["lr"], weight_decay=1e-4)
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(options["seed"]))
    batch = default_collate([ds[int(order[0])]])
    reference_loss = update_group(reference, optimizer, [batch], kind=kind,
                                  device=torch.device("cpu"), steps=2, bf16=False,
                                  process_weight=0.0, clip=1.0)
    reference_rng = rng_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary arm must not install/validate Z or check a deadline")

    monkeypatch.setattr(control, "install_frozen_z", forbidden)
    monkeypatch.setattr(control, "validate_frozen_z", forbidden)
    monkeypatch.setattr(runner, "_check_deadline", forbidden)
    first, report = runner.run_scheduled_updates(ds, **options)
    second_options = dict(options, output_dir=tmp_path / "explicit-none" / "run")
    second, second_report = runner.run_scheduled_updates(
        ds, intervention=None, deadline=None, **second_options)
    saved, again = load_checkpoint(first), load_checkpoint(second)
    expected = dict(kind=kind, model=options["model_config"], data_identity=options["data_identity"],
                    batch_size=1, steps=2, seed=7, lr=2e-4, process_weight=0.0, clip=1.0,
                    bf16=False, device_type="cpu", torch_version=str(torch.__version__),
                    dataset_length=2, optimization="native" if kind == "native" else "streamed-truncated",
                    schedule="linear-warmup-then-cosine", warmup_updates=1, minimum_lr_ratio=0.1,
                    validation_every=0, early_stopping_patience=None, minimum_improvement=0.001,
                    validation_lead_hours=[6], step_hours=6, validation_split=None)
    assert report["contract"] == second_report["contract"] == saved["contract"] == expected
    assert report["signature"] == saved["signature"] == canonical_digest(expected)
    assert "intervention" not in saved["contract"] and "deadline" not in saved["contract"]
    assert report["scientific_claim"] is False
    assert report["losses"] == second_report["losses"]
    assert report["losses"][0]["loss"] == reference_loss
    _assert_exact(saved["model"], reference.state_dict())
    _assert_exact(saved["optimizer"], optimizer.state_dict())
    _assert_exact(saved["rng"], reference_rng)
    _assert_exact(saved, again)


def test_install_follows_shared_transfer_and_only_trainable_params_enter_optimizer(tmp_path, monkeypatch):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path)
    seed_everything(91)
    anchor = make_model("process", options["model_config"])
    shared = {name: value.detach().clone() for name, value in anchor.state_dict().items()}
    spec = _spec(options["model_config"])
    observed = {}
    real_install, real_validate = control.install_frozen_z, control.validate_frozen_z
    real_optimizer = torch.optim.AdamW

    def install(model, frozen_spec):
        _assert_exact(model.state_dict(), shared)
        observed["model"] = model
        before = {name: id(parameter) for name, parameter in model.named_parameters()}
        state = rng_state()
        handle = real_install(model, frozen_spec)
        assert before == {name: id(parameter) for name, parameter in model.named_parameters()}
        _assert_exact(rng_state(), state)
        observed["buffer"] = model._frozen_z_value.detach().clone()
        return handle

    def optimizer(parameters, **kwargs):
        parameters = list(parameters)
        model = observed["model"]
        assert parameters and all(parameter.requires_grad for parameter in parameters)
        assert [id(p) for p in parameters] == [id(p) for p in model.parameters() if p.requires_grad]
        assert not model.solver_init.requires_grad
        assert all(not p.requires_grad for p in model.solver_cell.parameters())
        observed["optimizer_count"] = len(parameters)
        return real_optimizer(parameters, **kwargs)

    def validate(model, frozen_spec):
        for name, value in model.state_dict().items():
            if name == "solver_init" or name.startswith("solver_cell."):
                _assert_exact(value, shared[name])
        _assert_exact(model._frozen_z_value, observed["buffer"])
        observed["validations"] = observed.get("validations", 0) + 1
        return real_validate(model, frozen_spec)

    monkeypatch.setattr(control, "install_frozen_z", install)
    monkeypatch.setattr(control, "validate_frozen_z", validate)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    checkpoint, report = runner.run_scheduled_updates(
        ds, **dict(options, total_updates=2), shared_initial_state=shared, intervention=spec)
    saved = load_checkpoint(checkpoint)
    assert saved["contract"]["intervention"] == spec
    assert report["signature"] == canonical_digest(saved["contract"])
    assert report["shared_initial_state"]["applied_count"] == len(shared)
    assert observed["validations"] >= 2  # both sides of checkpoint publication
    assert len(saved["optimizer"]["param_groups"][0]["params"]) == observed["optimizer_count"]
    _assert_exact(saved["model"]["_frozen_z_value"], observed["buffer"])
    for name, value in shared.items():
        if name == "solver_init" or name.startswith("solver_cell."):
            _assert_exact(saved["model"][name], value)
    assert all(math.isfinite(entry["loss"]) for entry in report["losses"])
    assert report["updates_this_run"] == 2
    assert any(not torch.equal(saved["model"][name], value) for name, value in shared.items()
               if name.startswith("proposal_head."))
    assert any(not torch.equal(saved["model"][name], value) for name, value in shared.items()
               if name.startswith("solver_gate."))


@pytest.mark.parametrize("change", ["format", "token-tuple", "shape-tuple", "tensor-digest"])
def test_runner_rejects_unsupported_specs_instead_of_normalizing_them(tmp_path, change):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path)
    spec = _spec(options["model_config"])
    if change == "format":
        spec["format"] = "unsupported-frozen-z"
    elif change == "token-tuple":
        spec["token_hw"] = tuple(spec["token_hw"])
    elif change == "shape-tuple":
        spec["shape"] = tuple(spec["shape"])
    else:
        spec["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen Z spec"):
        runner.run_scheduled_updates(ds, **options, intervention=spec)
    assert not list(options["output_dir"].glob("*.pt"))
    assert not (options["output_dir"] / "training_report.json").exists()


def test_strict_reload_restores_buffer_and_real_intervention_output(tmp_path):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path)
    spec = _spec(options["model_config"])
    checkpoint, report = runner.run_scheduled_updates(ds, **options, intervention=spec)
    saved = load_checkpoint(checkpoint)
    restored = _restore(saved)
    second = _restore(saved)
    wrong = make_model("process", options["model_config"]).eval()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        wrong.load_state_dict(saved["model"], strict=True)
    wrong.load_state_dict({name: value for name, value in saved["model"].items()
                          if name != "_frozen_z_value"}, strict=True)
    batch = {"coarse_history": default_collate([ds[0]])["coarse_history"]}
    with torch.no_grad():
        prediction = restored(batch, reasoning_steps=2)
        again = second(batch, reasoning_steps=2)
        ordinary = wrong(batch, reasoning_steps=2)
    _assert_exact(restored._frozen_z_value, saved["model"]["_frozen_z_value"])
    _assert_exact(prediction.forecast, again.forecast)
    _assert_exact(prediction.solver_state, restored._frozen_z_value)
    assert not torch.equal(prediction.forecast, ordinary.forecast)
    assert not torch.equal(prediction.solver_state, ordinary.solver_state)
    assert report["scientific_claim"] is False


def test_frozen_resume_reinstalls_before_strict_load_and_validates(tmp_path):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path / "first")
    spec = _spec(options["model_config"])
    first, _ = runner.run_scheduled_updates(ds, **options, intervention=spec)
    second, report = runner.run_scheduled_updates(
        ds, **dict(options, output_dir=tmp_path / "resumed", total_updates=2),
        intervention=spec, resume=first)
    before, after = load_checkpoint(first), load_checkpoint(second)
    assert report["updates_this_run"] == 1 and after["updates"] == 2
    assert after["signature"] == before["signature"]
    _assert_exact(before["model"]["_frozen_z_value"], after["model"]["_frozen_z_value"])
    _restore(after)
    broken = copy.deepcopy(before)
    broken["model"]["_frozen_z_value"].add_(1)
    bad = _rewrite(broken, tmp_path / "tampered.pt")
    with pytest.raises(ValueError):
        runner.run_scheduled_updates(
            ds, **dict(options, output_dir=tmp_path / "bad-resume", total_updates=2),
            intervention=spec, resume=bad)
    assert not (tmp_path / "bad-resume" / "training_report.json").exists()


def test_post_publication_frozen_validation_cannot_report_success(tmp_path, monkeypatch):
    ds = SyntheticAtmosDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path)
    real_make, real_save = runner.make_model, runner.save_exclusive
    models = []

    def make(*args):
        model = real_make(*args)
        models.append(model)
        return model

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        models[0]._frozen_z_value.add_(1)

    monkeypatch.setattr(runner, "make_model", make)
    monkeypatch.setattr(runner, "save_exclusive", save)
    with pytest.raises(ValueError):
        runner.run_scheduled_updates(ds, **options, intervention=_spec(options["model_config"]))
    assert list(options["output_dir"].glob("update_*.pt"))  # a real partial artifact, not a PASS
    assert not (options["output_dir"] / "training_report.json").exists()


@pytest.mark.parametrize("stage", ["expired", "before-update", "before-validation", "before-publish"])
def test_deadline_refuses_updates_validation_or_publication_without_fake_success(tmp_path, monkeypatch, stage):
    clock = {"now": 0.0, "steps": 0}
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    class AdvancingDataset(SyntheticAtmosDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if stage == "before-update":
                clock["now"] = 2.0
            return sample

    ds = AdvancingDataset(length=2, hw=(4, 6), channels=1)
    options = _options(tmp_path)
    real_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        clock["steps"] += 1
        result = real_step(optimizer, *args, **kwargs)
        if stage in ("before-validation", "before-publish"):
            clock["now"] = 2.0
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not run after deadline")

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(runner, "score_validation", forbidden)
    if stage == "before-validation":
        options.update(validation_every=1, validation_dataset=object())
    with pytest.raises(RuntimeError, match="deadline"):
        runner.run_scheduled_updates(ds, **options, deadline=-1.0 if stage == "expired" else 1.0)
    assert clock["steps"] == (0 if stage in ("expired", "before-update") else 1)
    assert not list(options["output_dir"].glob("*.pt"))
    assert not (options["output_dir"] / "training_report.json").exists()


@pytest.fixture
def frozen_store_run(tmp_path):
    # This existing fixture publisher makes train/val/test under tmp_path only.
    # Its source label is explicitly synthetic rather than the ERA5 default.
    from test_r7_storage_safety import build

    paths = build(tmp_path / "synthetic-data",
                  source_label="synthetic frozen-Z engineering fixture, not observations")
    identity, ds = dataset_identity(paths["train"])
    options = dict(_options(tmp_path), data_identity=identity)
    spec = _spec(options["model_config"], hw=(3, 4))
    checkpoint, report = runner.run_scheduled_updates(ds, **options, intervention=spec)
    return paths, checkpoint, report


def test_evaluator_reconstructs_spec_strictly_and_provenance_is_conditional(tmp_path, monkeypatch, frozen_store_run):
    from data.r7_evaluation import ZarrRolloutDataset
    from model.r7_rollout import rollout_model_input

    paths, checkpoint, report = frozen_store_run
    saved = load_checkpoint(checkpoint)
    spec = report["contract"]["intervention"]
    real_make, observed = evaluator.make_model, []

    def make(*args):
        model = real_make(*args)
        observed.append(model)
        return model

    def forbidden(*args):
        raise AssertionError("no deadline helper call when deadline is None")

    monkeypatch.setattr(evaluator, "make_model", make)
    monkeypatch.setattr(evaluator, "_check_deadline", forbidden)
    provenance = evaluator.evaluate_local(paths["val"], output_dir=tmp_path / "eval",
                                           checkpoint=checkpoint, lead_hours=(6,), max_samples=1)
    assert provenance["intervention"] == spec and provenance["scientific_claim"] is False
    assert provenance["training_identity"] == report["data_identity"]
    assert provenance["split"] == "val" and provenance["n_evaluated"] == 1
    assert "synthetic" in provenance["source_declaration"]
    assert json.loads((tmp_path / "eval" / "provenance.json").read_text(encoding="utf-8")) == provenance
    assert control.validate_frozen_z(observed[0], spec) == canonical_digest(spec)
    _assert_exact(observed[0]._frozen_z_value, saved["model"]["_frozen_z_value"])
    ds = ZarrRolloutDataset(tmp_path / "synthetic-data" / "output", split="val", lead_hours=(6,))
    batch = rollout_model_input(ds[0], lead_hours=6.0, device=torch.device("cpu"))
    with torch.no_grad():
        _assert_exact(observed[0](batch, reasoning_steps=2).forecast,
                      _restore(saved)(batch, reasoning_steps=2).forecast)
    identity, training = dataset_identity(paths["train"])
    ordinary, _ = runner.run_scheduled_updates(
        training, **dict(_options(tmp_path / "ordinary"), data_identity=identity))
    plain = evaluator.evaluate_local(paths["val"], output_dir=tmp_path / "ordinary-eval",
                                      checkpoint=ordinary, lead_hours=(6,), max_samples=1)
    assert "intervention" not in plain and plain["scientific_claim"] is False
    assert not hasattr(observed[1], "_frozen_z_value")


@pytest.mark.parametrize("change, message, error", [
    ("buffer", None, ValueError),
    ("missing-buffer", "missing.*buffer", RuntimeError),
    ("spec-format", None, ValueError),
    ("spec-digest", None, ValueError),
    ("model-code", "model implementation differs", ValueError),
    ("data-identity", "training data/normalization identity mismatch", ValueError),
    ("contract-signature", "contract digest mismatch", ValueError),
])
def test_evaluator_rejects_tampering_without_bypassing_existing_identities(
        tmp_path, frozen_store_run, change, message, error):
    paths, checkpoint, _ = frozen_store_run
    saved = load_checkpoint(checkpoint)
    if change == "buffer":
        saved["model"]["_frozen_z_value"].add_(1)
    elif change == "missing-buffer":
        saved["model"].pop("_frozen_z_value")
    elif change == "spec-format":
        saved["contract"]["intervention"]["format"] = "unsupported-frozen-z"
    elif change == "spec-digest":
        saved["contract"]["intervention"]["tensor_sha256"] = "0" * 64
    elif change == "model-code":
        saved["model_code_sha256"] = "0" * 64
    else:
        saved["contract"]["data_identity"] = "wrong-synthetic-identity"
    if change != "contract-signature":
        saved["signature"] = canonical_digest(saved["contract"])
    altered = _rewrite(saved, tmp_path / "altered.pt")
    with pytest.raises(error, match=message):
        evaluator.evaluate_local(paths["val"], output_dir=tmp_path / "rejected",
                                 checkpoint=altered, lead_hours=(6,), max_samples=1)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("stage", ["setup", "climatology", "sample"])
def test_evaluator_deadline_checks_before_setup_climatology_and_samples(
        tmp_path, monkeypatch, frozen_store_run, stage):
    from data.r7_evaluation import ZarrRolloutDataset

    paths, checkpoint, _ = frozen_store_run
    clock = {"now": 2.0 if stage == "setup" else 0.0, "climatology_calls": 0}
    monkeypatch.setattr(evaluator, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    real_make, real_clim = evaluator.make_model, evaluator.fit_training_climatology

    def make(*args):
        model = real_make(*args)
        if stage == "climatology":
            clock["now"] = 2.0
        return model

    def climate(*args):
        clock["climatology_calls"] += 1
        result = real_clim(*args)
        clock["now"] = 2.0
        return result

    def forbidden(*args):
        raise AssertionError("no held-out sample should be read after the deadline")

    monkeypatch.setattr(evaluator, "make_model", make)
    monkeypatch.setattr(evaluator, "fit_training_climatology", climate)
    monkeypatch.setattr(ZarrRolloutDataset, "__getitem__", forbidden)
    with pytest.raises(RuntimeError, match="deadline"):
        evaluator.evaluate_local(paths["val"], output_dir=tmp_path / "expired-eval",
                                 checkpoint=checkpoint, lead_hours=(6,), max_samples=1, deadline=1.0)
    assert clock["climatology_calls"] == (1 if stage == "sample" else 0)
    assert not (tmp_path / "expired-eval" / "provenance.json").exists()
    assert not (tmp_path / "expired-eval" / "rmse.csv").exists()
