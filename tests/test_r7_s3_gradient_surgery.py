"""CI-safe checks for the S3 short/long gradient-surgery (PCGrad) training factor.

The frozen factor is the training-time backward only: ``gradient_surgery`` goes from
``off`` (registered ``loss.backward()``) to ``pcgrad`` (two group gradients, global
conflict test, projection of group A onto group B's normal plane, then the sum). These
tests run on synthetic tensors and fixtures only: nothing reads ``outputs/`` or
``test.jsonl``. Every frozen claim has a counterproof, including "the group split is
exactly steps 0/1/3 vs 7/11 with the registered weights", "the projection is exact on
dyadic operands and only fires on a negative global dot", "the off path is bitwise the
registered backward and leaves the contract/report unchanged", and "the training path
accepts the new config on CPU".
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


driver = _load("study_r7_s3_gradient_surgery")
parent = driver.parent
import training.r7_gradient_surgery as surgery
import training.r7_long_rollout_runner as runner
from training.r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model
from training.r7_long_rollout import training_long_rollout

WEIGHTS = surgery.REGISTERED_PHYSICAL_WEIGHTS


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------- exact arithmetic


class _Pair(torch.nn.Module):
    """Two scalar parameters; per-step losses are linear, so gradients are dyadic."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(2.0))


def _linear_steps(model, coefficients):
    return torch.stack([model.a * a + model.b * b for a, b in coefficients])


#: g_A = (1,0), g_B = (-1,1): dot = -1, ||g_B||^2 = 2, coefficient = -0.5.
CONFLICT = [(1., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.),
            (-2., 2.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]
#: g_A = (1,0), g_B = (1,1): dot = +1, no projection.
ALIGNED = [(1., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.),
           (2., 2.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]
#: Global dot -3 while the per-parameter dots are -4 and +1.
MIXED = [(1., 1.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.),
         (-8., 2.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]


def test_the_group_split_is_the_registered_contour():
    assert surgery.GROUP_A_STEPS == (0, 1, 3) and surgery.GROUP_B_STEPS == (7, 11)
    assert surgery.GROUP_A_WEIGHTS == (1.0, 0.5, 0.5)
    assert surgery.GROUP_B_WEIGHTS == (0.5, 0.5)
    assert [index for index, weight in enumerate(WEIGHTS) if weight > 0] == [0, 1, 3, 7, 11]
    assert [index for index, weight in enumerate(WEIGHTS) if weight == 0] == [2, 4, 5, 6, 8, 9, 10]
    assert driver.gradient_surgery_block()["groups"] == {
        "A": {"steps": [0, 1, 3], "leads_hours": [6, 12, 24], "weights": [1.0, 0.5, 0.5]},
        "B": {"steps": [7, 11], "leads_hours": [48, 72], "weights": [0.5, 0.5]}}


def test_group_losses_select_the_weighted_steps_exactly():
    steps = torch.tensor([2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 2048., 4096.],
                         dtype=torch.float64)
    group_a, group_b = surgery.group_losses(steps)
    assert float(group_a) == 12.0          # 1*2 + .5*4 + .5*16
    assert float(group_b) == 2176.0        # .5*256 + .5*4096
    # Counter-evidence: a zero-weight step cannot move either group, a weighted step does.
    changed_zero = steps.clone()
    changed_zero[2] = 1e6
    a2, b2 = surgery.group_losses(changed_zero)
    assert torch.equal(group_a, a2) and torch.equal(group_b, b2)
    changed_weighted = steps.clone()
    changed_weighted[0] += 2.0
    a3, _ = surgery.group_losses(changed_weighted)
    assert float(a3 - group_a) == 2.0


def test_a_changed_weight_contour_is_refused():
    with pytest.raises(ValueError):
        surgery.group_losses(torch.zeros(12), (1.0,) * 12)
    with pytest.raises(ValueError):
        surgery.group_losses(torch.zeros(3), WEIGHTS)


def test_conflict_projects_group_a_with_exact_dyadic_arithmetic():
    model = _Pair()
    result = surgery.pcgrad_backward(model, _linear_steps(model, CONFLICT),
                                     registered_loss=torch.tensor(2.0), clip=1.0)
    assert result["conflict"] is True and result["projection_count"] == 1
    assert result["projected_group"] == "A"
    assert result["negative_parameter_dots"] == 1
    assert result["group_a_grad_norm"] == 1.0
    assert result["group_b_grad_norm"] == pytest.approx(2 ** 0.5, abs=0, rel=0)
    assert result["group_global_cosine"] == pytest.approx(-0.5 ** 0.5, abs=1e-15)
    assert result["group_sum_residual"] == 0.0
    assert torch.equal(model.a.grad, torch.tensor(-0.5))
    assert torch.equal(model.b.grad, torch.tensor(1.5))
    assert result["pre_clip_total_norm"] == pytest.approx(2.5 ** 0.5, abs=1e-15)


def test_no_conflict_leaves_the_groups_unchanged():
    model = _Pair()
    result = surgery.pcgrad_backward(model, _linear_steps(model, ALIGNED), clip=1.0)
    assert result["conflict"] is False and result["projection_count"] == 0
    assert result["projected_group"] is None
    assert result["negative_parameter_dots"] == 0
    assert torch.equal(model.a.grad, torch.tensor(2.0))
    assert torch.equal(model.b.grad, torch.tensor(1.0))


def test_the_conflict_test_is_global_not_per_parameter():
    """Local dots are -4 and +1; only the global dot decides the projection."""
    model = _Pair()
    result = surgery.pcgrad_backward(model, _linear_steps(model, MIXED), clip=1.0)
    assert result["negative_parameter_dots"] == 1
    assert result["conflict"] is True and result["projection_count"] == 1
    coefficient = -3.0 / 17.0
    assert model.a.grad == pytest.approx((1.0 - coefficient * -4.0) + -4.0, abs=1e-6)
    assert model.b.grad == pytest.approx((1.0 - coefficient * 1.0) + 1.0, abs=1e-6)


def test_an_unknown_mode_is_refused():
    assert surgery.require_gradient_surgery("off") == "off"
    assert surgery.require_gradient_surgery("pcgrad") == "pcgrad"
    for bad in ("PCGrad", "", None, True, "random_order"):
        with pytest.raises(ValueError):
            surgery.require_gradient_surgery(bad)


@pytest.mark.parametrize("override", [
    {"physical_weights": (1.0, 0.5, 0.5)},
    {"physical_weights": (0.5,) * 12},
    {"draft_supervision": "final_only"},
    {"gradient_lambda": 0.5},
    {"ema_decay": 0.999},
])
def test_pcgrad_refuses_every_closed_axis(override):
    axes = dict(physical_weights=WEIGHTS, draft_supervision="deep", gradient_lambda=0.0,
                ema_decay=None)
    axes.update(override)
    with pytest.raises(ValueError):
        surgery.require_pcgrad_axes(**axes)


def test_summarize_updates_aggregates_and_refuses_missing_fields():
    model = _Pair()
    first = surgery.pcgrad_backward(model, _linear_steps(model, CONFLICT), clip=1.0)
    second_model = _Pair()
    second = surgery.pcgrad_backward(second_model, _linear_steps(second_model, ALIGNED), clip=1.0)
    summary = surgery.summarize_updates([first, second])
    assert summary["updates"] == 2 and summary["conflict_updates"] == 1
    assert summary["conflict_ratio"] == 0.5 and summary["projection_count"] == 1
    assert summary["cosine_min"] == pytest.approx(first["group_global_cosine"], abs=1e-15)
    assert summary["cosine_max"] == pytest.approx(second["group_global_cosine"], abs=1e-15)
    assert summary["group_sum_residual_max"] == 0.0
    broken = dict(first)
    broken.pop("conflict")
    with pytest.raises(RuntimeError, match="missing"):
        surgery.summarize_updates([broken])


# --------------------------------------------------------------------------- runner integration


class _NoOpOptimizer:
    """Keep the assigned gradients visible; lr is irrelevant to this check."""

    def __init__(self):
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def _tiny_rollout():
    spec = parent._tiny_spec()
    torch.manual_seed(parent.EQUIVALENCE_SEED)
    model = make_model("process", spec).train()
    batch = parent._probe_batch()
    batch["physical_targets"] = torch.randn(
        1, 12, 17, 4, 5, generator=torch.Generator().manual_seed(5))
    return model, batch


def _build(model, batch):
    return training_long_rollout(model, batch, 2, physical_weights=WEIGHTS)


def _contract(mode="off"):
    body = {"steps": 2, "clip": 1e9,
            "autoregression": {"physical_weights": list(WEIGHTS), "draft_supervision": "deep"}}
    if mode != "off":
        body["autoregression"]["gradient_surgery"] = mode
    return body


def test_off_path_is_bitwise_the_registered_backward(monkeypatch):
    model, batch = _tiny_rollout()
    reference = copy.deepcopy(model)
    expected = _build(reference, batch)
    expected.loss.backward()
    expected_grads = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                      for name, parameter in reference.named_parameters()}
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    metrics = runner._update(model, _NoOpOptimizer(), batch, _contract(), torch.device("cpu"), None)
    for name, parameter in model.named_parameters():
        assert (parameter.grad is None) == (expected_grads[name] is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, expected_grads[name]), name
    assert set(metrics) == {"loss", "l6", "l12", "per_step_losses", "gradient_norm"}


def test_off_path_metrics_have_no_surgery_fields_but_pcgrad_does(monkeypatch):
    model, batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout", lambda *args, **kwargs: _build(model, batch))
    off = runner._update(model, _NoOpOptimizer(), batch, _contract(), torch.device("cpu"), None)
    assert not set(surgery.UPDATE_KEYS) & set(off)
    candidate_model, candidate_batch = _tiny_rollout()
    monkeypatch.setattr(runner, "training_long_rollout",
                        lambda *args, **kwargs: _build(candidate_model, candidate_batch))
    pcgrad = runner._update(candidate_model, _NoOpOptimizer(), candidate_batch,
                            _contract("pcgrad"), torch.device("cpu"), None)
    assert set(surgery.UPDATE_KEYS) <= set(pcgrad)
    assert pcgrad["projection_count"] in (0, 1) and isinstance(pcgrad["conflict"], bool)


def test_pcgrad_matches_an_independent_two_group_recombination():
    model, batch = _tiny_rollout()
    result = _build(model, batch)
    diagnostics = surgery.pcgrad_backward(model, result.per_step_losses,
                                          registered_loss=result.loss, clip=1.0)
    reference, reference_batch = _tiny_rollout()
    reference_result = _build(reference, reference_batch)
    group_a, group_b = surgery.group_losses(reference_result.per_step_losses)
    parameters = [parameter for parameter in reference.parameters() if parameter.requires_grad]
    grads_a = torch.autograd.grad(group_a, parameters, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(group_b, parameters, allow_unused=True)

    def _flat(gradient, parameter):
        return (torch.zeros(parameter.numel(), dtype=torch.float64)
                if gradient is None else gradient.detach().reshape(-1).double())

    flat_a = [_flat(gradient, parameter) for gradient, parameter in zip(grads_a, parameters)]
    flat_b = [_flat(gradient, parameter) for gradient, parameter in zip(grads_b, parameters)]
    dot = sum((a * b).sum() for a, b in zip(flat_a, flat_b))
    norm_b_sq = sum((b * b).sum() for b in flat_b)
    assert diagnostics["conflict"] == bool(float(dot) < 0)
    for parameter, a, b, raw_a, raw_b in zip(model.parameters(), flat_a, flat_b, grads_a, grads_b):
        if raw_a is None and raw_b is None:
            assert parameter.grad is None
            continue
        projected = a - (dot / norm_b_sq) * b if float(dot) < 0 else a
        expected = (projected + b).reshape(parameter.shape).to(parameter.dtype)
        assert torch.equal(parameter.grad, expected)


# --------------------------------------------------------------------------- CI-safe training smoke


def _long_options(tmp_path, *, mode="pcgrad", updates=2):
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
    if mode != "off":
        contract["autoregression"]["gradient_surgery"] = mode
    options = dict(model=model, contract=contract, parent_weights=weights, steps=2,
                   updates=updates, seed=7, warmup=1, checkpoint_every=1,
                   physical_weights=WEIGHTS, gradient_surgery=mode, draft_supervision="deep")
    return paths, options


def test_cpu_smoke_accepts_the_pcgrad_training_path(tmp_path):
    paths, options = _long_options(tmp_path)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert report["contract"]["autoregression"]["gradient_surgery"] == "pcgrad"
    assert report["gradient_surgery"] == "pcgrad"
    assert "decoder_base_mode" not in report["contract"]["model"]
    assert not any(key.startswith("ema_") for key in report)
    assert report["updates_this_run"] == 2
    for row in report["losses"]:
        assert len(row["per_step_losses"]) == 12
        assert set(surgery.UPDATE_KEYS) <= set(row)
        assert row["projection_count"] in (0, 1)
    summary = surgery.summarize_updates(report["losses"])
    assert summary["updates"] == 2
    saved = load_checkpoint(checkpoint)
    assert saved["contract"]["autoregression"]["gradient_surgery"] == "pcgrad"


def test_off_path_training_report_is_unchanged(tmp_path):
    paths, options = _long_options(tmp_path, mode="off", updates=1)
    _, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert "gradient_surgery" not in report
    assert "gradient_surgery" not in report["contract"]["autoregression"]
    assert not set(surgery.UPDATE_KEYS) & set(report["losses"][0])


@pytest.mark.parametrize("override", [
    {"draft_supervision": "final_only"},
    {"gradient_lambda": 0.5},
    {"ema_decay": 0.999},
])
def test_pcgrad_training_refuses_a_closed_axis(tmp_path, override):
    paths, options = _long_options(tmp_path)
    options.update(override)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_off_path_refuses_a_declared_surgery_field(tmp_path):
    paths, options = _long_options(tmp_path, mode="off", updates=1)
    options["contract"]["autoregression"]["gradient_surgery"] = "pcgrad"
    with pytest.raises(ValueError, match="gradient surgery"):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)


def test_pcgrad_refuses_a_changed_physical_contour(tmp_path):
    paths, options = _long_options(tmp_path)
    options["physical_weights"] = (1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
