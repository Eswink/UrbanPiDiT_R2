"""Real FP32/CPU active-subset regression; synthetic engineering inputs only.

No training, weather payloads or archived checkpoints. A strict in-memory state
round-trip binds the fixed K references to the adaptive forecaster's weights.
The spies always delegate to the real step, anchored proposal and decoder.
"""
from contextlib import contextmanager
import hashlib
import json

import pytest
import torch

from model.local_solver_state_r7 import anchored_proposal
from model.process_forecast_r7 import ProcessForecastCoReasoner
import model.process_step_r7 as step_module
from model.r7_halting import AdaptiveProcessForecaster, ForecastGainController


FULL_FLAGS = dict(
    use_forecast_feedback=True, spatial_solver_feedback=False,
    spacetime_inputs=True, spacetime_field_mode="fields",
    positional_process_readout=True, pooled_readout_query=False,
    source_position_markers=True, source_role_markers=True,
    draft_query_feedback=True, known_context_inputs=True,
    local_solver_state=True, solver_state_recurrence=True, solver_gate_proposal=True,
)
RTOL, ATOL = 1e-5, 1e-6
ACTIVE_INDICES = [[0, 1, 2], [0, 2], [0, 2], [0, 2]]


class MiddleSampleStops(ForecastGainController):
    """Sample 1 stops at Kmin=1; original samples 0 and 2 run through Kmax=4."""

    def __init__(self):
        super().__init__(processes=2, channels=3, hidden=8)
        self.calls = []

    def forward(self, features):
        self.calls.append(features.detach().clone())
        gains = features.new_ones(features.shape[0])
        if len(self.calls) == 1:
            assert features.shape[0] == 3
            gains[1] = -1.
        return gains, torch.full_like(gains, 20.)


def _model():
    return ProcessForecastCoReasoner(
        in_channels=3, out_channels=3, history_steps=2, dim=16, depth=1,
        heads=2, window_size=2, patch_size=2, dropout=0.,
        anchored_processes=2, free_processes=2, default_reasoning_steps=4,
        **FULL_FLAGS,
    ).cpu().float().eval()


@pytest.fixture
def case():
    torch.manual_seed(713)
    model = _model()
    for name, value in FULL_FLAGS.items():
        assert getattr(model, name) == value, name
    assert model.backbone.known_context_inputs
    assert model.process_reader.draft_query_feedback
    generator = torch.Generator(device="cpu").manual_seed(713)
    history = torch.randn(3, 2, 3, 5, 7, generator=generator, dtype=torch.float32)
    history += torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1, 1) * 2
    data = {
        "coarse_history": history,
        "lead_time_hours": torch.tensor([6., 12., 18.]),
        "latitude": torch.linspace(-30., 30., 5),
        "longitude": torch.linspace(-179., 179., 7),
        "init_calendar_year": torch.tensor([2024, 2023, 2024]),
        "init_day_of_year": torch.tensor([60., 365., 180.]),
        "init_utc_hour": torch.tensor([0., 18., 6.]),
        "history_offsets_hours": torch.tensor([[-6., 0.]]).expand(3, -1).clone(),
    }
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    fixed = _model()
    fixed.load_state_dict(checkpoint, strict=True)
    assert set(fixed.state_dict()) == set(checkpoint)
    assert all(torch.equal(value, fixed.state_dict()[name]) for name, value in checkpoint.items())
    with torch.no_grad():
        once, full = fixed(data, reasoning_steps=1), fixed(data, reasoning_steps=4)
    adapter = AdaptiveProcessForecaster(model)
    adapter.controller = MiddleSampleStops()
    adapter.eval()
    assert adapter.controller.optimizer_updates.item() == 0
    assert all(value.device.type == "cpu" for value in data.values())
    assert once.forecast.dtype == full.forecast.dtype == torch.float32
    return adapter, data, checkpoint, once, full


def _return_unsliced_anchor(selected_anchor, full_anchor):
    """Fault: restore the original full-B anchor, without changing the real solver."""
    return full_anchor


@contextmanager
def _observe(adapter, monkeypatch, *, inject_unsliced=False):
    seen = {"base": None, "inputs": None, "steps": [], "proposals": [], "decoded": [],
            "completed": []}
    real_step = adapter.reasoning_step
    real_proposal = step_module.anchored_proposal
    assert real_proposal is anchored_proposal

    def backbone_hook(module, args, output):
        seen["base"], seen["inputs"] = output, args[0]

    def step_spy(process, context, draft, token_hw, **kwargs):
        if inject_unsliced:
            kwargs["anchor"] = _return_unsliced_anchor(kwargs["anchor"], seen["base"].base_state)
        call = dict(process=process, context=context, draft=draft, token_hw=token_hw, **kwargs)
        seen["steps"].append(call)
        result = real_step(process, context, draft, token_hw, **kwargs)
        call["result"] = result
        return result

    def proposal_spy(decoder, solver_state, token_hw, output_hw, anchor):
        seen["proposals"].append({"anchor": anchor, "solver": solver_state,
                                  "token_hw": token_hw, "output_hw": output_hw})
        result = real_proposal(decoder, solver_state, token_hw, output_hw, anchor)
        seen["completed"].append(len(seen["proposals"]) - 1)
        return result

    hooks = [adapter.forecaster.backbone.register_forward_hook(backbone_hook),
             adapter.forecaster.proposal_head.register_forward_pre_hook(
                 lambda module, args: seen["decoded"].append(args))]
    with monkeypatch.context() as patch:
        patch.setattr(adapter, "reasoning_step", step_spy)
        patch.setattr(step_module, "anchored_proposal", proposal_spy)
        try:
            yield seen
        finally:
            for hook in hooks:
                hook.remove()


def _checkpoint_sha256(checkpoint):
    digest = hashlib.sha256()
    for name, value in sorted(checkpoint.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_checkpoint_unchanged(adapter, checkpoint):
    state = adapter.forecaster.state_dict()
    assert set(state) == set(checkpoint)
    for name, value in checkpoint.items():
        assert torch.equal(value, state[name]), name
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert adapter.controller.optimizer_updates.item() == 0


def _assert_anchor_trace(seen, data, selected_trace):
    base = seen["base"]
    assert base.token_hw == (3, 4)
    assert torch.equal(base.base_state, data["coarse_history"][:, -1])
    for name in data:
        assert seen["inputs"][name] is data[name], name
    assert len(seen["steps"]) == len(seen["proposals"]) == len(seen["decoded"]) == 4
    assert seen["completed"] == [0, 1, 2, 3]
    for step, (call, proposal, decoded, selected) in enumerate(zip(
            seen["steps"], seen["proposals"], seen["decoded"], selected_trace)):
        anchor = proposal["anchor"]
        assert anchor is call["anchor"] and decoded[3] is anchor
        assert decoded[0] is proposal["solver"] is call["result"].solver_state
        assert anchor.shape == call["draft"].shape == (len(selected), 3, 5, 7)
        assert call["process"].shape == (len(selected), 4, 16)
        assert call["solver_state"].shape == proposal["solver"].shape == (len(selected), 12, 16)
        assert call["step_index"] == step
        assert proposal["token_hw"] == call["token_hw"] == (3, 4)
        assert proposal["output_hw"] == (5, 7)
        assert torch.equal(anchor, base.base_state[selected]), (step, selected)
        assert torch.equal(call["context"], base.context_tokens[selected])
        assert anchor.dtype == proposal["solver"].dtype == torch.float32
        assert anchor.device.type == proposal["solver"].device.type == "cpu"


def _assert_mixed_active_anchor(case, seen):
    """The same end-to-end behavioral predicate runs with and without the fault."""
    adapter, data, checkpoint, once, full = case
    result = adapter(data, max_steps=4, min_steps=1, allow_untrained=True)
    assert result.reasoning_steps_per_sample.tolist() == [4, 1, 4]
    assert result.active_masks.tolist() == [[True] * 4, [True, False, False, False], [True] * 4]
    assert result.decision_masks.tolist() == [[True, True, True, False],
                                            [True, False, False, False],
                                            [True, True, True, False]]
    assert [features.shape[0] for features in adapter.controller.calls] == [3, 2, 2]
    selected_trace = [mask.nonzero(as_tuple=False).flatten().tolist()
                      for mask in result.active_masks.unbind(1)]
    assert selected_trace == ACTIVE_INDICES
    _assert_anchor_trace(seen, data, selected_trace)
    errors = []
    for sample, reference in enumerate((full, once, full)):
        torch.testing.assert_close(result.forecast[sample], reference.forecast[sample],
                                   rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(result.process_predictions[sample],
                                   reference.process_predictions[sample, -1], rtol=RTOL, atol=ATOL)
        errors.append((result.forecast[sample] - reference.forecast[sample]).abs().max().item())
    assert result.forecast.dtype == torch.float32 and torch.isfinite(result.forecast).all()
    assert not result.forecast.requires_grad
    _assert_checkpoint_unchanged(adapter, checkpoint)
    return {"selected_trace": selected_trace, "steps": [4, 1, 4],
            "anchor_shapes": [list(call["anchor"].shape) for call in seen["proposals"]],
            "max_abs_forecast_error_per_sample": errors, "rtol": RTOL, "atol": ATOL,
            "same_checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "engineering_fixture_sha256": _checkpoint_sha256(data), "full_flags": FULL_FLAGS,
            "real_anchored_proposal_calls": len(seen["proposals"])}


def test_noncontiguous_active_anchor_matches_same_checkpoint_fixed_depth(case, monkeypatch, record_property):
    with _observe(case[0], monkeypatch) as seen:
        proof = _assert_mixed_active_anchor(case, seen)
    record_property("active_anchor_proof", json.dumps(proof, sort_keys=True))


def test_force_full_depth_preserves_same_checkpoint_fixed_forecast(case, monkeypatch, record_property):
    adapter, data, checkpoint, _, full = case
    with _observe(adapter, monkeypatch) as seen:
        result = adapter(data, max_steps=4, min_steps=1, force_full_depth=True)
        assert result.reasoning_steps_per_sample.tolist() == [4, 4, 4]
        assert result.active_masks.all() and not result.decision_masks.any()
        assert not adapter.controller.calls
        torch.testing.assert_close(result.forecast, full.forecast, rtol=0, atol=0)
        torch.testing.assert_close(result.process_predictions, full.process_predictions[:, -1], rtol=0, atol=0)
        _assert_anchor_trace(seen, data, [[0, 1, 2]] * 4)
    _assert_checkpoint_unchanged(adapter, checkpoint)
    record_property("force_full_proof", json.dumps({"steps": [4, 4, 4], "bit_equal_to_fixed_k4": True,
                                                    "real_anchored_proposal_calls": 4}, sort_keys=True))


def test_unsliced_anchor_fault_fails_same_real_model_predicate(case, monkeypatch, record_property):
    adapter, data, checkpoint, _, _ = case
    with _observe(adapter, monkeypatch, inject_unsliced=True) as seen:
        with pytest.raises(ValueError, match="the anchor must be"):
            _assert_mixed_active_anchor(case, seen)
        assert len(seen["steps"]) == len(seen["proposals"]) == 2
        assert seen["completed"] == [0] and len(seen["decoded"]) == 1
        assert seen["proposals"][-1]["anchor"] is seen["base"].base_state
        assert seen["proposals"][-1]["anchor"].shape == (3, 3, 5, 7)
        assert seen["proposals"][-1]["solver"].shape == (2, 12, 16)
        assert [call["step_index"] for call in seen["steps"]] == [0, 1]
        assert torch.equal(seen["steps"][-1]["context"], seen["base"].context_tokens[[0, 2]])
        assert [features.shape[0] for features in adapter.controller.calls] == [3]
    _assert_checkpoint_unchanged(adapter, checkpoint)
    record_property("fault_proof", json.dumps({"anchor_batch": 3, "solver_batch": 2,
        "active_indices_at_failure": [0, 2], "real_anchored_proposal_calls": 2,
        "real_decodes_completed": 1, "same_predicate_rejected": True}, sort_keys=True))
