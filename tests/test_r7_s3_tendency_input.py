"""CI-safe checks for the S3 explicit-tendency-input model factor.

The frozen factor is the encoder input representation only: ``tendency_input`` goes from
``False`` (``[X_{t-1}, X_t]``) to ``True`` (``[X_{t-1}, X_t, D_t]`` with
``D_t = X_t - X_{t-1}``) through a separate, zero-initialized, bias-free 17->dim patch.
These tests run on synthetic tensors and fixtures only: nothing reads ``outputs/`` or
``test.jsonl``. Every frozen claim has a counterproof, including "the two arms share
every pre-existing tensor under the same seed", "at initialization the forward is
bitwise identical", "the branch input is exactly X_t - X_{t-1}", "the reversed sign is a
different function", and "the training path accepts the new spec on CPU".
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


driver = _load("study_r7_s3_tendency_input")
parent = driver.parent
import training.r7_long_rollout_runner as runner
from model.coarse_encoder import CoarseEncoder
from model.layers.patch_grid import pad_patch_grid
from training.r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model

WEIGHTS = driver.REGISTERED_PHYSICAL_WEIGHTS
TENDENCY_KEY = parent.TENDENCY_KEY


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _build(spec, seed=1234):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return make_model("process", spec)


def _tiny_spec(**extra):
    spec = parent._tiny_spec()
    spec.update(extra)
    return spec


# --------------------------------------------------------------------------- the one added tensor


def test_switching_on_adds_exactly_one_zero_tensor():
    control = _build(_tiny_spec())
    candidate = _build(_tiny_spec(tendency_input=True))
    control_state, candidate_state = control.state_dict(), candidate.state_dict()
    assert sorted(set(candidate_state) - set(control_state)) == [TENDENCY_KEY]
    assert not (set(control_state) - set(candidate_state))
    for key in control_state:
        assert torch.equal(control_state[key], candidate_state[key]), key
    patch = candidate.backbone.encoder.tendency_patch
    assert tuple(patch.weight.shape) == (32, 17, 2, 2)
    assert patch.bias is None
    assert bool((patch.weight == 0).all())


def test_the_parameter_delta_at_the_frozen_shape_is_13056():
    encoder = CoarseEncoder(17, 2, dim=192, patch_size=2, depth=0, heads=2, window_size=2,
                            tendency_input=True)
    assert tuple(encoder.tendency_patch.weight.shape) == (192, 17, 2, 2)
    assert encoder.tendency_patch.weight.numel() == 13056
    assert encoder.tendency_patch.bias is None
    assert encoder.tendency_patch.stride == (2, 2)
    assert encoder.tendency_patch.kernel_size == (2, 2)


def test_the_initial_forward_is_bitwise_the_control():
    control = _build(_tiny_spec()).eval()
    candidate = _build(_tiny_spec(tendency_input=True)).eval()
    batch = parent._probe_batch()
    with torch.no_grad():
        a, b = control(batch), candidate(batch)
    assert torch.equal(a.forecast, b.forecast)
    assert torch.equal(a.initial_forecast, b.initial_forecast)
    assert torch.equal(a.draft_forecasts, b.draft_forecasts)
    assert torch.equal(a.final_correction, b.final_correction)


def test_a_nonzero_patch_changes_the_forecast():
    """Counter-evidence: the branch is actually read, not silently dropped."""
    control = _build(_tiny_spec()).eval()
    candidate = _build(_tiny_spec(tendency_input=True)).eval()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        candidate.backbone.encoder.tendency_patch.weight.copy_(
            torch.randn_like(candidate.backbone.encoder.tendency_patch.weight, generator=generator))
        batch = parent._probe_batch()
        assert not torch.equal(control(batch).forecast, candidate(batch).forecast)


# --------------------------------------------------------------------------- exact branch input


def _encoder_capture(history, weight, *, pad=True):
    encoder = CoarseEncoder(17, 2, dim=8, patch_size=2, depth=1, heads=2, window_size=2,
                            pad_to_patch=pad, tendency_input=True)
    with torch.no_grad():
        encoder.tendency_patch.weight.copy_(weight)
    captured = {}
    handle = encoder.tendency_patch.register_forward_hook(
        lambda module, inputs, output: captured.update(
            input=inputs[0].detach().clone(), output=output.detach().clone()))
    try:
        with torch.no_grad():
            tokens = encoder(history)[0]
    finally:
        handle.remove()
    with torch.no_grad():
        flat = history.reshape(1, -1, *history.shape[-2:])
        if pad:
            flat = pad_patch_grid(flat, 2)
        z = encoder.patch(flat) + captured["output"]
        height, width = z.shape[-2:]
        manual = z.flatten(2).transpose(1, 2)
        for block in encoder.blocks:
            manual = block(manual, (height, width))
        manual = encoder.norm(manual)
    return encoder, captured, tokens, manual


def test_the_branch_input_is_exactly_x_t_minus_x_prev():
    generator = torch.Generator().manual_seed(11)
    history = torch.randn(1, 2, 17, 4, 6, generator=generator)
    weight = torch.randn(8, 17, 2, 2, generator=torch.Generator().manual_seed(3))
    _, captured, tokens, manual = _encoder_capture(history, weight, pad=False)
    assert torch.equal(captured["input"], history[:, -1] - history[:, -2])
    assert torch.equal(tokens, manual)
    with torch.no_grad():
        reversed_output = torch.nn.functional.conv2d(
            history[:, -2] - history[:, -1], weight, stride=2)
    assert not torch.equal(captured["output"], reversed_output)


def test_padding_is_applied_to_the_difference_too():
    generator = torch.Generator().manual_seed(13)
    history = torch.randn(1, 2, 17, 5, 7, generator=generator)
    weight = torch.randn(8, 17, 2, 2, generator=torch.Generator().manual_seed(4))
    _, captured, _, _ = _encoder_capture(history, weight, pad=True)
    expected = pad_patch_grid(history[:, -1] - history[:, -2], 2)
    assert torch.equal(captured["input"], expected)
    # Counter-evidence: the unpadded difference has a different shape.
    assert captured["input"].shape[-2:] != (history.shape[-2], history.shape[-1])


@pytest.mark.parametrize("kwargs", [
    {"history_steps": 1}, {"tendency_input": 1}, {"tendency_input": "yes"}])
def test_impossible_tendency_configurations_are_refused(kwargs):
    base = dict(in_channels=17, history_steps=2, dim=8, patch_size=2, depth=1, heads=2,
                window_size=2, tendency_input=True)
    base.update(kwargs)
    with pytest.raises(ValueError):
        CoarseEncoder(**base)


def test_the_process_constructor_refuses_a_non_boolean_switch():
    with pytest.raises(ValueError, match="tendency_input"):
        _build(_tiny_spec(tendency_input=1))


def test_switch_off_keeps_the_pre_change_state_and_forward():
    """The off path is the previous implementation: no key and no forward branch."""
    model = _build(_tiny_spec(tendency_input=False))
    assert TENDENCY_KEY not in model.state_dict()
    assert model.backbone.encoder.tendency_input is False
    batch = parent._probe_batch()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output.forecast).all()


# --------------------------------------------------------------------------- CI-safe training smoke


def _long_options(tmp_path, *, spec_switch=True, model_switch=True, updates=2):
    import pandas as pd
    from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    paths, _ = fixture_manifest(tmp_path / "fixture",
                                times=pd.date_range("2016-01-01", periods=40, freq="6h"))
    identity, _ = dataset_identity(paths["train"])
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=12)
    spec = tiny_spec("process")
    if spec_switch:
        spec["tendency_input"] = True
    torch.manual_seed(23)
    model_spec = {key: value for key, value in spec.items() if key != "tendency_input"}
    if model_switch:
        model_spec["tendency_input"] = True
    model = make_model("process", model_spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": "process", "model": spec, "data_identity": identity,
                "source_sha256": canonical_digest("synthetic-source"),
                "protocol_sha256": canonical_digest("synthetic-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"physical_steps": 12, "physical_weights": list(WEIGHTS),
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"],
                                   "draft_supervision": "deep"}}
    options = dict(model=model, contract=contract, parent_weights=weights, steps=2,
                   updates=updates, seed=7, warmup=1, checkpoint_every=1,
                   physical_weights=WEIGHTS, draft_supervision="deep")
    return paths, options


def test_cpu_smoke_accepts_the_tendency_input_training_path(tmp_path):
    paths, options = _long_options(tmp_path)
    checkpoint, report = runner.fine_tune_long_rollout(paths["train"], tmp_path / "run", **options)
    assert report["contract"]["model"]["tendency_input"] is True
    assert "gradient_surgery" not in report
    assert "gradient_surgery" not in report["contract"]["autoregression"]
    assert "decoder_base_mode" not in report["contract"]["model"]
    assert report["updates_this_run"] == 2
    for row in report["losses"]:
        assert len(row["per_step_losses"]) == 12
    saved = load_checkpoint(checkpoint)
    assert saved["contract"]["model"]["tendency_input"] is True
    patch = saved["model"][TENDENCY_KEY]
    assert tuple(patch.shape) == (8, 17, 2, 2)
    assert int(patch.numel()) == 8 * 17 * 2 * 2


def test_the_training_path_refuses_a_model_contract_mismatch(tmp_path):
    paths, options = _long_options(tmp_path / "case-a", spec_switch=False, model_switch=True)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused-a", **options)
    paths, options = _long_options(tmp_path / "case-b", spec_switch=True, model_switch=False)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused-b", **options)
    assert not (tmp_path / "refused-a" / "training_report.json").exists()
    assert not (tmp_path / "refused-b" / "training_report.json").exists()
    assert not (tmp_path / "refused-a" / "training_report.json").exists()
    assert not (tmp_path / "refused-b" / "training_report.json").exists()


def test_a_closed_axis_is_still_refused_with_the_tendency_input(tmp_path):
    paths, options = _long_options(tmp_path)
    options["physical_weights"] = (1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
    assert not (tmp_path / "refused" / "training_report.json").exists()


def test_a_tampered_contract_spec_is_refused(tmp_path):
    paths, options = _long_options(tmp_path)
    options["contract"] = copy.deepcopy(options["contract"])
    options["contract"]["model"]["tendency_input"] = False
    with pytest.raises(ValueError):
        runner.fine_tune_long_rollout(paths["train"], tmp_path / "refused", **options)
