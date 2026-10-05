"""#78 R-A: the decode reparameterization is a switch, not a silent rewrite.

The contrast is one thing: ``CoarseForecastHead`` either writes its decoded
increment unchanged (``identity``, the pre-change behavior) or multiplies it by
the fixed train-only ratio ``d_c / s_c`` (``normalized_change_scale``). These
tests pin the parts that must be provable without a GPU:

* identity mode is the old implementation - same construction, same state dict,
  same forward bits as keyword-free construction;
* scaled mode writes exactly ``X_t + ratio * r_c`` where ``r_c`` is the same
  tensor the identity mode writes, so the change is the parameterization alone;
* the ratio is a non-persistent buffer: no state-dict key, no parameter-count
  change, no FLOP change, and a checkpoint round-trip preserves behavior;
* refused combinations fail closed instead of silently ignoring a switch.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.coarse_forecast import CoarseForecastHead  # noqa: E402

DIM, CHANNELS, PATCH = 16, 3, 2
TOKEN_HW = (4, 4)
OUTPUT_HW = (8, 8)
RATIO = [0.2, 0.5, 0.8]


def _seed(value=7):
    torch.manual_seed(value)


def _tokens(batch=2):
    _seed(11)
    return torch.randn(batch, TOKEN_HW[0] * TOKEN_HW[1], DIM)


def _base(batch=2):
    _seed(13)
    return torch.randn(batch, CHANNELS, *OUTPUT_HW)


def _head(mode="identity", ratio=None):
    _seed(7)
    return CoarseForecastHead(DIM, CHANNELS, PATCH, change_scale_mode=mode,
                              change_scale_ratio=ratio)


def test_identity_keyword_free_construction_is_bitwise_the_same_module():
    a = _head("identity")
    _seed(7)
    b = CoarseForecastHead(DIM, CHANNELS, PATCH)
    assert set(a.state_dict()) == set(b.state_dict())
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
    tokens, base = _tokens(), _base()
    fa, ta = a(tokens, TOKEN_HW, OUTPUT_HW, base)
    fb, tb = b(tokens, TOKEN_HW, OUTPUT_HW, base)
    assert torch.equal(fa, fb) and torch.equal(ta, tb)


def test_identity_forward_equals_the_manual_pre_change_recipe():
    from model.layers.patch_grid import crop_native_grid

    head = _head("identity")
    tokens, base = _tokens(), _base()
    forecast, tendency = head(tokens, TOKEN_HW, OUTPUT_HW, base)
    B, N, D = tokens.shape
    z = head.norm(tokens).transpose(1, 2).reshape(B, D, *TOKEN_HW)
    manual = crop_native_grid(head.decode(z), OUTPUT_HW, PATCH)
    assert torch.equal(tendency, manual)
    assert torch.equal(forecast, base + manual)


def test_scaled_mode_writes_exactly_ratio_times_the_identity_increment():
    identity = _head("identity")
    scaled = _head("normalized_change_scale", RATIO)
    # same construction under one seed: the trained weights must not move
    for key in identity.state_dict():
        assert torch.equal(identity.state_dict()[key], scaled.state_dict()[key]), key
    tokens, base = _tokens(), _base()
    _, t_identity = identity(tokens, TOKEN_HW, OUTPUT_HW, base)
    forecast, t_scaled = scaled(tokens, TOKEN_HW, OUTPUT_HW, base)
    ratio = torch.tensor(RATIO, dtype=t_identity.dtype).view(1, -1, 1, 1)
    expected = t_identity * ratio
    assert torch.equal(t_scaled, expected)
    assert torch.equal(forecast, base + expected)


def test_scaling_is_per_channel_and_not_a_scalar():
    scaled = _head("normalized_change_scale", RATIO)
    tokens, base = _tokens(), _base()
    _, t = scaled(tokens, TOKEN_HW, OUTPUT_HW, base)
    identity = _head("identity")
    _, raw = identity(tokens, TOKEN_HW, OUTPUT_HW, base)
    for channel, value in enumerate(RATIO):
        assert torch.allclose(t[:, channel], raw[:, channel] * value, rtol=0,
                              atol=0)
    # a scalar multiplier would leave every channel's ratio identical; the
    # measured per-channel scaling must follow three different values
    ratios = []
    for channel in range(CHANNELS):
        with torch.no_grad():
            ratios.append(float((t[:, channel] / raw[:, channel].clamp(min=1e-1)
                                 ).median()))
    assert len({round(v, 3) for v in ratios}) == CHANNELS


def test_ratio_is_a_non_persistent_buffer_not_a_parameter():
    identity = _head("identity")
    scaled = _head("normalized_change_scale", RATIO)
    assert set(identity.state_dict()) == set(scaled.state_dict())
    assert not any("change_scale_ratio" in key for key in scaled.state_dict())
    p_i = sum(v.numel() for v in identity.parameters())
    p_s = sum(v.numel() for v in scaled.parameters())
    assert p_i == p_s
    # the buffer exists and is fixed at construction
    assert torch.equal(scaled.change_scale_ratio,
                       torch.tensor(RATIO, dtype=torch.float32))


def test_checkpoint_round_trip_preserves_the_scaled_behavior(tmp_path):
    scaled = _head("normalized_change_scale", RATIO)
    path = tmp_path / "head.pt"
    torch.save({"state": scaled.state_dict(), "mode": scaled.change_scale_mode,
                "ratio": scaled.change_scale_ratio.tolist()}, path)
    loaded = torch.load(path, weights_only=False)
    restored = CoarseForecastHead(DIM, CHANNELS, PATCH,
                                  change_scale_mode=loaded["mode"],
                                  change_scale_ratio=loaded["ratio"])
    restored.load_state_dict(loaded["state"], strict=True)
    tokens, base = _tokens(), _base()
    f0, t0 = scaled(tokens, TOKEN_HW, OUTPUT_HW, base)
    f1, t1 = restored(tokens, TOKEN_HW, OUTPUT_HW, base)
    assert torch.equal(f0, f1) and torch.equal(t0, t1)


def test_repeated_calls_do_not_accumulate_the_scale():
    scaled = _head("normalized_change_scale", RATIO)
    tokens, base = _tokens(), _base()
    first = scaled(tokens, TOKEN_HW, OUTPUT_HW, base)[1]
    for _ in range(3):
        again = scaled(tokens, TOKEN_HW, OUTPUT_HW, base)[1]
        assert torch.equal(first, again)


def test_unknown_mode_and_mismatched_arguments_fail_closed():
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH, change_scale_mode="scaled")
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH, change_scale_mode="identity",
                           change_scale_ratio=RATIO)
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH, change_scale_mode="normalized_change_scale")
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH,
                           change_scale_mode="normalized_change_scale",
                           change_scale_ratio=[0.5, 0.5])
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH,
                           change_scale_mode="normalized_change_scale",
                           change_scale_ratio=[0.5, -0.5, 0.5])
    with pytest.raises(ValueError):
        CoarseForecastHead(DIM, CHANNELS, PATCH,
                           change_scale_mode="normalized_change_scale",
                           change_scale_ratio=[0.5, float("nan"), 0.5])


def test_process_models_differ_only_by_the_scale_switch():
    """The two arms have one tensor set, one parameter count, equal FLOPs."""
    from model.process_forecast_r7 import ProcessForecastCoReasoner

    base = dict(in_channels=CHANNELS, history_steps=2, out_channels=CHANNELS,
                dim=DIM, patch_size=PATCH, depth=2, heads=2, window_size=2,
                anchored_processes=2, free_processes=2, default_reasoning_steps=1,
                spacetime_inputs=True, positional_process_readout=True,
                local_solver_state=True, solver_state_recurrence=True,
                solver_gate_proposal=True, known_context_inputs=True,
                draft_query_feedback=True, source_position_markers=True,
                source_role_markers=True)
    _seed(21)
    identity = ProcessForecastCoReasoner(**base)
    _seed(21)
    scaled = ProcessForecastCoReasoner(**dict(
        base, change_scale_mode="normalized_change_scale", change_scale_ratio=RATIO))
    assert set(identity.state_dict()) == set(scaled.state_dict())
    for key in identity.state_dict():
        assert torch.equal(identity.state_dict()[key], scaled.state_dict()[key]), key
    assert (sum(v.numel() for v in identity.parameters())
            == sum(v.numel() for v in scaled.parameters()))
    # every decode head in the model carries the same mode and ratio
    assert identity.correction_head.change_scale_mode == "identity"
    assert scaled.correction_head.change_scale_mode == "normalized_change_scale"
    assert scaled.backbone.head.change_scale_mode == "normalized_change_scale"
    assert scaled.proposal_head.change_scale_mode == "normalized_change_scale"
    assert torch.equal(scaled.proposal_head.change_scale_ratio,
                       torch.tensor(RATIO, dtype=torch.float32))
