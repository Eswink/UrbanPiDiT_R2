"""CI-safe checks for the S3 sharpness-aware-minimization training factor.

The frozen factor is the training-time update rule only: ``sam_rho`` goes from ``None``
(the registered single ``result.loss.backward()``) to the predeclared ``0.05`` (one global
rescaling of the registered gradient into an ascent step, a second forward/backward of the
same registered loss at ``w + e``, a bit-for-bit restore, then the registered clip and
optimizer step). These tests run on synthetic tensors and fixtures only: nothing reads
``outputs/`` or ``test.jsonl``. Every frozen claim has a counterproof, including "the off
path is bitwise the registered backward", "``rho = 0`` reproduces the registered path",
"the assigned gradient is the gradient at the perturbed point, not the first pass's
gradient accumulated onto it", "the parameters are restored before the optimizer step",
"the rescaling is one global scalar", and "the training path accepts the new config on
CPU".
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load("study_r7_s3_sharpness_aware_minimization")
parent = driver.parent
import training.r7_long_rollout_runner as runner
import training.r7_sam as sam
from training.r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model
from training.r7_long_rollout import training_long_rollout

WEIGHTS = driver.REGISTERED_PHYSICAL_WEIGHTS
SAM_RHO = driver.CANDIDATE_SAM_RHO


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _tiny_spec():
    return dict(architecture="window", in_channels=17, out_channels=17, history_steps=2,
                dim=32, depth=2, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=2, detach_between_steps=False,
                anchored_processes=4, free_processes=1, use_forecast_feedback=True,
                positional_process_readout=True, local_solver_state=True,
                solver_state_recurrence=True, solver_gate_proposal=True, draft_query_feedback=True)


def _probe_batch(channels=17, height=4, width=5):
    generator = torch.Generator().manual_seed(11)
    return {"coarse_history": torch.randn(1, 2, channels, height, width, generator=generator),
            "latitude": torch.linspace(50., 44., height),
            "longitude": torch.linspace(90., 98., width),
            "lead_time_hours": torch.tensor([6.]),
            "init_utc_hour": torch.tensor([18.]),
            "init_day_of_year": torch.tensor([366.]),
            "init_calendar_year": torch.tensor([2016.]),
            "history_offsets_hours": torch.tensor([[-6., 0.]])}


def _tiny_rollout(seed=1234):
    torch.manual_seed(seed)
    model = make_model("process", _tiny_spec()).train()
    batch = _probe_batch()
    batch["physical_targets"] = torch.randn(
        1, 12, 17, 4, 5, generator=torch.Generator().manual_seed(5))
    return model, batch


def _build(model, batch):
    return training_long_rollout(model, batch, 2, physical_weights=WEIGHTS)


def _contract(rho=None, surgery="off"):
    body = {"steps": 2, "clip": 1e9,
            "autoregression": {"physical_weights": list(WEIGHTS), "draft_supervision": "deep"}}
    if rho is not None:
        body["autoregression"]["sam_rho"] = rho
    if surgery != "off":
        body["autoregression"]["gradient_surgery"] = surgery
    return body


class _NoOpOptimizer:
    """Keep the assigned gradients visible; lr is irrelevant to these checks."""

    def __init__(self):
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


class _RecordingOptimizer:
    """Records what ``optimizer.step()`` would have seen, then does nothing."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.param_groups = [{"lr": 0.0}]
        self.snapshot = None
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        self.steps += 1
        self.snapshot = [parameter.detach().clone() for parameter in self.parameters]


def _grads(model):
    """Every parameter's gradient, ``None`` where the backward left none."""
    return [None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()]


def _same(grads_a, grads_b):
    if len(grads_a) != len(grads_b):
        return False
    for one, other in zip(grads_a, grads_b):
        if (one is None) != (other is None):
            return False
        if one is not None and not torch.equal(one, other):
            return False
    return True


# --------------------------------------------------------------------------- the rule itself


def test_the_frozen_constants_are_the_predeclared_ones():
    assert SAM_RHO == 0.05 and driver.CONTROL_SAM_RHO is None and driver.SAM_EPS == 1e-12
    assert driver.UPDATES == 2400 and driver.STEPS == driver.REASONING_STEPS == 4
    assert (driver.DIM, driver.DEPTH) == (192, 4) and driver.SEEDS == (41, 42, 43)
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert (driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == (43200., 86400.)
    assert driver.PER_SEED_SECONDS == 21600.
    assert sam.DEFAULT_SAM_RHO == 0.05 and sam.SAM_EPS == 1e-12


def test_the_sam_block_is_frozen():
    block = driver.sam_block()
    assert block["candidate_rho"] == 0.05 and block["control_rho"] is None
    assert block["eps"] == 1e-12 and block["forwards_per_update"] == 2
    assert "global norm" in block["rule"] and "bit for bit" in block["rule"]
    assert block["sweep"].startswith("none")
    assert "2010.01412" in block["reference"]


def test_changed_sam_constants_are_refused(monkeypatch):
    for name, value in (("CANDIDATE_SAM_RHO", 0.1), ("CONTROL_SAM_RHO", 0.0),
                        ("SAM_EPS", 1e-8)):
        monkeypatch.setattr(driver, name, value)
        with pytest.raises(RuntimeError):
            driver.sam_block()
        monkeypatch.undo()


def test_the_radius_vocabulary_is_closed():
    assert sam.require_sam_rho(None) is None
    assert sam.require_sam_rho(0.05) == 0.05 and sam.require_sam_rho(0) == 0.0
    for bad in (True, False, "0.05", -0.05, float("nan"), float("inf"), [0.05]):
        with pytest.raises(ValueError):
            sam.require_sam_rho(bad)


@pytest.mark.parametrize("override", [
    {"physical_weights": (1.0, 0.5, 0.5)},
    {"physical_weights": (0.5,) * 12},
    {"draft_supervision": "final_only"},
    {"gradient_lambda": 0.5},
    {"ema_decay": 0.999},
    {"gradient_surgery": "pcgrad"},
])
def test_sam_refuses_every_closed_axis(override):
    axes = dict(physical_weights=WEIGHTS, draft_supervision="deep", gradient_lambda=0.0,
                ema_decay=None, gradient_surgery="off")
    axes.update(override)
    with pytest.raises(ValueError):
        sam.require_sam_axes(**axes)


def test_the_update_rule_control_proves_the_rule_on_synthetic_tensors():
    control = parent.update_rule_control()
    assert control["synthetic"] is True and control["rho"] == 0.05
    for name, value in control["checks"].items():
        assert value, name


def test_summarize_updates_refuses_a_row_missing_a_field():
    rows = [{key: 0.0 for key in sam.UPDATE_KEYS}]
    assert sam.summarize_updates(rows)["updates"] == 1
    broken = dict(rows[0])
    broken.pop("sam_grad_norm")
    with pytest.raises(RuntimeError, match="missing"):
        sam.summarize_updates([broken])


# --------------------------------------------------------------------------- runner integration


def test_off_path_is_bitwise_the_registered_backward(monkeypatch):
    model, batch = _tiny_rollout()
    reference = copy.deepcopy(model)
    expected = _build(reference, batch)
    expected.loss.backward()
    expected_grads = _grads(reference)
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    metrics = runner._update(model, _NoOpOptimizer(), batch, _contract(), torch.device("cpu"), None)
    assert _same(_grads(model), expected_grads)
    assert set(metrics) == {"loss", "l6", "l12", "per_step_losses", "gradient_norm"}
    assert not set(sam.UPDATE_KEYS) & set(metrics)


def test_rho_zero_reproduces_the_registered_path(monkeypatch):
    """The SAM branch at rho=0 is the registered update, bit for bit."""
    model, batch = _tiny_rollout()
    reference = copy.deepcopy(model)
    expected = _build(reference, batch)
    expected.loss.backward()
    expected_grads = _grads(reference)
    original = [parameter.detach().clone() for parameter in model.parameters()]
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    metrics = runner._update(model, _NoOpOptimizer(), batch, _contract(rho=0.0),
                             torch.device("cpu"), None)
    assert _same(_grads(model), expected_grads)
    assert all(torch.equal(parameter.detach(), base)
               for parameter, base in zip(model.parameters(), original))
    assert metrics["sam_perturbation_norm"] == 0.0
    assert metrics["sam_restore_residual"] == 0.0


def test_the_update_gradient_is_the_gradient_at_the_perturbed_point(monkeypatch):
    model, batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    metrics = runner._update(model, _NoOpOptimizer(), batch, _contract(rho=SAM_RHO),
                             torch.device("cpu"), None)
    sam_grads = _grads(model)
    # Independent reference: the same (restored) parameters, the registered gradient
    # rescaled by one global scalar, then a fresh forward/backward at w + e.
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    registered = _build(reference, batch)
    registered.loss.backward()
    first_grads = _grads(reference)
    scale = SAM_RHO / (sam.global_norm(first_grads) + sam.SAM_EPS)
    with torch.no_grad():
        for parameter, gradient in zip(reference.parameters(), first_grads):
            if gradient is not None:
                parameter.add_(gradient * scale)
    reference.zero_grad(set_to_none=True)
    perturbed = _build(reference, batch)
    perturbed.loss.backward()
    assert _same(sam_grads, _grads(reference))
    # Counter-evidence: the first pass's gradient was not accumulated onto the second.
    accumulated = [None if gradient is None else gradient * scale + gradient
                   for gradient in first_grads]
    assert not _same(sam_grads, accumulated)
    assert not _same(sam_grads, first_grads)
    assert metrics["sam_grad_norm"] == pytest.approx(sam.global_norm(first_grads), abs=1e-12)
    assert metrics["sam_perturbation_norm"] == pytest.approx(SAM_RHO, abs=1e-9)
    # The ratio is 1 up to the fp32 rounding of ``g * scale`` (~6e-9 relative here).
    assert metrics["sam_perturbation_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["sam_loss_increase"] > 0.0


def test_the_parameters_are_restored_before_the_optimizer_step(monkeypatch):
    model, batch = _tiny_rollout()
    original = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = _RecordingOptimizer(model.parameters())
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    metrics = runner._update(model, optimizer, batch, _contract(rho=SAM_RHO),
                             torch.device("cpu"), None)
    assert optimizer.steps == 1 and optimizer.snapshot is not None
    assert all(torch.equal(seen, base) for seen, base in zip(optimizer.snapshot, original))
    assert metrics["sam_restore_residual"] == 0.0
    # Counter-evidence: the assigned gradient is not the registered gradient, so the
    # restore did not silently discard the SAM direction.
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    registered = _build(reference, batch)
    registered.loss.backward()
    assert not _same(_grads(model), _grads(reference))


def test_a_reversed_sign_would_be_a_different_update(monkeypatch):
    """Counter-evidence: the ascent step is +rho*g/||g||, not the descent direction."""
    model, batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    runner._update(model, _NoOpOptimizer(), batch, _contract(rho=SAM_RHO), torch.device("cpu"), None)
    sam_grads = _grads(model)
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    registered = _build(reference, batch)
    registered.loss.backward()
    first_grads = _grads(reference)
    scale = SAM_RHO / (sam.global_norm(first_grads) + sam.SAM_EPS)
    with torch.no_grad():
        for parameter, gradient in zip(reference.parameters(), first_grads):
            if gradient is not None:
                parameter.sub_(gradient * scale)
    reference.zero_grad(set_to_none=True)
    descent = _build(reference, batch)
    descent.loss.backward()
    assert not _same(sam_grads, _grads(reference))


def _sam_update(monkeypatch, clip, seed=1234):
    """One SAM ``_update`` on a fresh, bitwise-identical tiny model."""
    model, batch = _tiny_rollout(seed)
    contract = _contract(rho=SAM_RHO)
    contract["clip"] = clip
    with monkeypatch.context() as patch:
        patch.setattr(runner, "training_long_rollout",
                      lambda *args, **kwargs: _build(model, batch))
        metrics = runner._update(model, _NoOpOptimizer(), batch, contract,
                                 torch.device("cpu"), None)
    return model, metrics


def test_the_registered_clip_still_acts_on_the_sam_gradient(monkeypatch):
    """The registered clip runs after the restore, on the SAM gradient."""
    loose_model, loose = _sam_update(monkeypatch, 1e9)
    tight_model, tight = _sam_update(monkeypatch, 1e-6)
    assert tight["sam_update_grad_norm"] == pytest.approx(loose["sam_update_grad_norm"], rel=1e-12)
    # clip_grad_norm_ returns the pre-clip norm, so the two calls report the same one;
    # the counter-evidence is the gradient actually stored afterwards.
    assert tight["gradient_norm"] == pytest.approx(loose["gradient_norm"], rel=1e-12)
    assert sam.global_norm(_grads(loose_model)) == pytest.approx(loose["gradient_norm"], rel=1e-6)
    assert sam.global_norm(_grads(tight_model)) == pytest.approx(1e-6, rel=1e-6)


def test_sam_cannot_be_combined_with_the_gradient_surgery_in_the_runner(monkeypatch):
    model, batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    with pytest.raises(ValueError, match="gradient-surgery"):
        runner._update(model, _NoOpOptimizer(), batch, _contract(rho=SAM_RHO, surgery="pcgrad"),
                       torch.device("cpu"), None)


def test_an_undeclared_radius_leaves_the_registered_contract_untouched(monkeypatch):
    model, batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    off = runner._update(model, _NoOpOptimizer(), batch, _contract(), torch.device("cpu"), None)
    assert not set(sam.UPDATE_KEYS) & set(off)
    candidate_model, candidate_batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout",
                        lambda *args, **kwargs: _build(candidate_model, candidate_batch))
    on = runner._update(candidate_model, _NoOpOptimizer(), candidate_batch,
                        _contract(rho=SAM_RHO), torch.device("cpu"), None)
    assert set(sam.UPDATE_KEYS) <= set(on)
    assert set(on) - set(off) == set(sam.UPDATE_KEYS)


# --------------------------------------------------------------------------- CI-safe training smoke


def _long_options(tmp_path, *, rho=SAM_RHO, updates=2, declared=None, surgery="off"):
    import pandas as pd
    from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    paths, _ = fixture_manifest(tmp_path / "fixture",
                                times=pd.date_range("2016-01-01", periods=40, freq="6h"))
    identity, _ = dataset_identity(paths["train"])
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=12)
    spec = tiny_spec("process")
    torch.manual_seed(23)
    model = make_model("process", spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": "process", "model": spec, "data_identity": identity,
                "source_sha256": canonical_digest("synthetic-source"),
                "protocol_sha256": canonical_digest("synthetic-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"physical_steps": 12, "physical_weights": list(WEIGHTS),
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"],
                                   "draft_supervision": "deep"}}
    if declared is not None:
        contract["autoregression"]["sam_rho"] = declared
    if surgery != "off":
        contract["autoregression"]["gradient_surgery"] = surgery
    options = dict(model=model, contract=contract, parent_weights=weights, steps=2,
                   updates=updates, seed=7, warmup=1, checkpoint_every=1,
                   physical_weights=WEIGHTS, draft_supervision="deep", sam_rho=rho,
                   gradient_surgery=surgery)
    return paths, options


def test_cpu_smoke_accepts_the_sam_training_path(tmp_path):
    paths, options = _long_options(tmp_path, declared=SAM_RHO)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert report["sam_rho"] == SAM_RHO and report["sam_forwards_per_update"] == 2
    assert report["contract"]["autoregression"]["sam_rho"] == SAM_RHO
    assert "gradient_surgery" not in report
    assert "gradient_surgery" not in report["contract"]["autoregression"]
    assert "decoder_base_mode" not in report["contract"]["model"]
    assert not any(key.startswith("ema_") for key in report)
    assert report["updates_this_run"] == 2
    for row in report["losses"]:
        assert len(row["per_step_losses"]) == 12
        assert set(sam.UPDATE_KEYS) <= set(row)
        assert row["sam_restore_residual"] == 0.0
    summary = sam.summarize_updates(report["losses"])
    assert summary["updates"] == 2 and summary["perturbation_norm_max"] > 0.0
    saved = load_checkpoint(checkpoint)
    assert saved["contract"]["autoregression"]["sam_rho"] == SAM_RHO


def test_off_path_training_report_is_unchanged(tmp_path):
    paths, options = _long_options(tmp_path, rho=None, updates=1)
    _, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert "sam_rho" not in report and "sam_eps" not in report
    assert "sam_rho" not in report["contract"]["autoregression"]
    assert not set(sam.UPDATE_KEYS) & set(report["losses"][0])


def test_sam_refuses_a_declared_radius_mismatch(tmp_path):
    paths, options = _long_options(tmp_path, declared=0.1)
    with pytest.raises(ValueError, match="SAM radius"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()
    paths, options = _long_options(tmp_path / "b", rho=None, updates=1, declared=SAM_RHO)
    with pytest.raises(ValueError, match="must not declare"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused-b", **options)
    assert not (tmp_path / "refused-b" / "training_report.json").exists()


@pytest.mark.parametrize("override", [
    {"draft_supervision": "final_only"},
    {"gradient_lambda": 0.5},
    {"ema_decay": 0.999},
])
def test_sam_training_refuses_a_closed_axis(tmp_path, override):
    paths, options = _long_options(tmp_path, declared=SAM_RHO)
    options.update(override)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_sam_training_refuses_the_gradient_surgery_combination(tmp_path):
    paths, options = _long_options(tmp_path, declared=SAM_RHO, surgery="pcgrad")
    with pytest.raises(ValueError, match="gradient-surgery"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_sam_training_refuses_a_changed_physical_contour(tmp_path):
    paths, options = _long_options(tmp_path, declared=SAM_RHO)
    options["physical_weights"] = (1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)


# --------------------------------------------------------- the frozen arm contract


def _contract_body():
    """Exactly the protocol keys ``_arm_contract`` reads, with synthetic values."""
    return {"train_data_identity": canonical_digest("synthetic-train"),
            "source_sha256": canonical_digest("synthetic-source"),
            "protocol_sha256": canonical_digest("synthetic-protocol"),
            "training_code_sha256": canonical_digest("synthetic-training-code"),
            "train_windows": {"window_sha256": canonical_digest("synthetic-window"),
                              "excluded_sample_ids": []},
            "supervision": {"runner_block": {"mask": "interior_32"}}}


def test_every_parent_attribute_the_driver_reads_is_re_exported():
    """Anti-drift guard for the defect that killed the first GPU attempt.

    ``_arm_contract`` reads names off the parent module, but it only runs inside a
    frozen, GPU-started round; a name the parent forgot to re-export therefore costs a
    whole freeze/prepare/archive cycle before it surfaces. This compares the driver's
    static attribute references against the module, so the gap shows up in CI instead.
    """
    import ast
    source = (ROOT / "scripts" / "study_r7_s3_sharpness_aware_minimization.py").read_text(
        encoding="utf-8")
    referenced = {node.attr for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id == "parent"}
    assert "ANCHOR_TABLE_RUN" in referenced, "the driver no longer records the anchor table run"
    missing = sorted(name for name in referenced if not hasattr(parent, name))
    assert not missing, f"the driver reads parent.{missing}, which the parent module lacks"


def test_the_arm_contract_initialization_block_resolves():
    body = _contract_body()
    spec = {"architecture": "window", "dim": 192, "depth": 4}
    expansion = {"anchor_table_sha256": canonical_digest("synthetic-anchor-table")}
    contract = driver._arm_contract(body, spec, Path("/synthetic/parent.pt"),
                                    canonical_digest("synthetic-parent"), expansion)
    init = contract["initialization"]
    assert contract["model"] is spec
    assert init["parent_run"] == str(parent.MIGRATED_PARENT_RUN)
    assert init["anchor_table_run"] == str(parent.ANCHOR_TABLE_RUN)
    assert init["anchor_table_sha256"] == expansion["anchor_table_sha256"]
    assert init["parent_migration_receipt_sha256"] == parent.MIGRATION_RECEIPT_SHA256
    assert init["parent_endpoint_updates"] == 1600
    assert contract["autoregression"]["sam_rho"] == SAM_RHO
    assert contract["autoregression"]["draft_supervision"] == driver.DRAFT_SUPERVISION
    assert contract["training_code_sha256"] == body["training_code_sha256"]
