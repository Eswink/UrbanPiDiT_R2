"""#72 M2-B: one step implementation, three callers, and the leak whitelist.

``docs/R7_MAIN_MODEL_V2_DESIGN.md`` section 8 records the defect this file
guards: the same recurrence was written out three times - in the fixed forward, in
the streamed backward and in the adaptive wrapper - so a change to one of them
could silently leave the other two behind. The first test here is structural: the
tell-tale of the process recurrence (``_process_prediction``,
``process_conditioning``) must appear in none of the three call sites, because the
recurrence now exists only in ``model/process_step_r7.py``. The rest are numerical:
the three paths must produce the same forecast, for both switches off and on.

The 3x3 neighbourhood in the local update is a *spatial* one, so the step needs the
patch grid and not a flattened token order; with ``window_size`` in the config the
numbers here would move if the two disagreed, which is why the tests run the real
configuration shape rather than a single-token grid.
"""
from __future__ import annotations

import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from model.recursive_weather_r7 import GenericRecursiveCell, recurrent_key
from training.r7_experiment import canonical_digest, load_checkpoint, save_exclusive
from training.r7_streaming import backward_streamed_truncated

REPO_FILES = ("model/r7_halting.py", "training/r7_streaming.py",
              "model/process_forecast_r7.py")

SMALL = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": 16,
         "depth": 1, "heads": 2, "window_size": 4, "patch_size": 2, "dropout": 0.0,
         "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 3,
         "use_forecast_feedback": True, "spacetime_inputs": True,
         "positional_process_readout": True}
STEPS = 3
HW = (6, 4)


def batch(hw=HW, size=2) -> dict:
    generator = torch.Generator().manual_seed(31)
    rows, columns = hw
    return {
        "coarse_history": torch.randn(size, 2, 3, rows, columns, generator=generator),
        "atmos_target": torch.randn(size, 3, rows, columns, generator=generator),
        "process_targets": torch.randn(size, 2, generator=generator),
        "lead_time_hours": torch.full((size,), 6.0),
        "latitude": torch.linspace(-30.0, 30.0, rows),
        "longitude": torch.linspace(0.0, 40.0, columns),
        "init_utc_hour": torch.full((size,), 3.0),
        "init_day_of_year": torch.full((size,), 40.0),
    }


def build(**switches) -> ProcessForecastCoReasoner:
    torch.manual_seed(23)
    return ProcessForecastCoReasoner(**SMALL, **switches)


def test_the_recurrence_lives_in_one_function():
    """Structural anti-drift: no call site may carry its own copy of the step.

    ``model/process_forecast_r7.py`` still *defines* the model's own pieces
    (``_process_prediction``, ``process_conditioning``) - that is where they
    belong. What it may not do any more is run them itself: the adapters and the
    model both have to reach the step through ``process_reasoning_step``.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("model/r7_halting.py", "training/r7_streaming.py"):
        source = (root / name).read_text(encoding="utf-8")
        stripped = "\n".join(line for line in source.splitlines()
                             if not line.lstrip().startswith("#"))
        for marker in ("_process_prediction", "process_conditioning"):
            assert marker not in stripped, (
                f"{name} contains {marker!r}, so it carries its own copy of the process "
                "recurrence; the step must exist only in model/process_step_r7.py")
    for name in REPO_FILES:
        assert "process_reasoning_step" in (root / name).read_text(encoding="utf-8"), name


def _three_paths(**switches):
    """The same model through the fixed, streamed and adaptive paths."""
    model = build(**switches)
    data = batch()
    model.train()
    with torch.no_grad():
        fixed = model(forecast_inputs(data), reasoning_steps=STEPS).forecast
    streamed = backward_streamed_truncated(
        model, data, reasoning_steps=STEPS, process_weight=0.0, loss_scale=1.0)
    model.eval()
    adapter = AdaptiveProcessForecaster(model).eval()
    with torch.no_grad():
        adaptive = adapter(data, max_steps=STEPS, min_steps=STEPS, force_full_depth=True)
    return fixed, streamed.final_forecast, adaptive.forecast


def test_the_three_paths_agree_with_the_new_switches_off():
    fixed, streamed, adaptive = _three_paths()
    assert torch.equal(streamed, fixed), "the streamed path left the fixed path behind"
    assert torch.equal(adaptive, fixed), "the adaptive path left the fixed path behind"


def test_the_three_paths_agree_with_the_new_switches_on():
    fixed, streamed, adaptive = _three_paths(source_role_markers=True, local_solver_state=True)
    assert torch.equal(streamed, fixed), "the streamed path left the fixed path behind"
    assert torch.equal(adaptive, fixed), "the adaptive path left the fixed path behind"


def test_the_adaptive_active_subset_reproduces_full_depth_when_nothing_stops():
    """An active-subset run that never stops must equal the fixed path exactly."""
    model = build(local_solver_state=True)
    adapter = AdaptiveProcessForecaster(model).eval()
    data = batch(size=3)
    with torch.no_grad():
        fixed = model(forecast_inputs(data), reasoning_steps=STEPS).forecast
        adaptive = adapter(data, max_steps=STEPS, min_steps=STEPS, force_full_depth=True)
    assert torch.equal(adaptive.forecast, fixed)
    assert (adaptive.reasoning_steps_per_sample == STEPS).all()
    assert adaptive.active_masks.all()


def test_the_role_markers_only_change_what_the_cell_reads():
    """The design contract's own criterion, with the counterproof next to it.

    Reversing the draft tokens permutes the cell's key set, and attention reads the
    keys as a set, so without markers the cell's answer is unchanged up to the
    floating-point order of the softmax sum. With markers the two halves are
    distinguishable, and the same permutation moves the answer.
    """
    torch.manual_seed(41)
    cell = GenericRecursiveCell(16, 2, mlp_ratio=3.0, dropout=0.0).eval()
    context = torch.randn(2, 24, 16)
    draft = torch.randn(2, 24, 16)
    reversed_draft = torch.flip(draft, dims=[1])
    query = torch.randn(2, 4, 16)
    with torch.no_grad():
        plain = cell(query, recurrent_key(context, draft))
        permuted = cell(query, recurrent_key(context, reversed_draft))
        marked = cell(query, recurrent_key(context, draft, role_context=torch.randn(1, 1, 16),
                                           role_draft=torch.randn(1, 1, 16)))
        marked_permuted = cell(query, recurrent_key(
            context, reversed_draft, role_context=torch.randn(1, 1, 16),
            role_draft=torch.randn(1, 1, 16)))
    assert torch.allclose(plain, permuted, atol=1e-5), (
        "without markers the cell should not notice the key order at all")
    assert not torch.allclose(marked, marked_permuted, atol=1e-5), (
        "the role vectors did not reach the cell's key")


def test_the_role_vectors_reach_the_forward_path():
    """A declared role that the forward never adds is a parameter the run cannot shape."""
    model = build(source_role_markers=True).eval()
    data = batch()
    with torch.no_grad():
        marked = model(forecast_inputs(data)).forecast
        keep = model.role_context.data.clone(), model.role_draft.data.clone()
        model.role_context.data.zero_()
        model.role_draft.data.zero_()
        zeroed = model(forecast_inputs(data)).forecast
        model.role_context.data, model.role_draft.data = keep
    assert not torch.equal(marked, zeroed), "the role vectors never reached the cell"


def test_poisoned_future_fields_leave_the_forward_untouched():
    model = build(local_solver_state=True).eval()
    clean = batch()
    poisoned = dict(clean, atmos_target=torch.full_like(clean["atmos_target"], 1e4),
                    future_diagnostic_targets=torch.full((2, 5), -7.0),
                    process_targets=torch.full_like(clean["process_targets"], 3e3),
                    atmos_baseline=torch.full((2, 3, *HW), 9e2))
    with torch.no_grad():
        first = model(forecast_inputs(clean), reasoning_steps=STEPS).forecast
        second = model(forecast_inputs(poisoned), reasoning_steps=STEPS).forecast
    assert torch.equal(first, second), "a future field reached the forward pass"


def test_poisoned_future_fields_leave_the_halting_selection_untouched():
    model = build().eval()
    adapter = AdaptiveProcessForecaster(model, gain_threshold=0.0).eval()
    clean = batch(size=4)
    poisoned = dict(clean, atmos_target=torch.full_like(clean["atmos_target"], -1e4))
    with torch.no_grad():
        first = adapter(clean, max_steps=4, min_steps=1, allow_untrained=True)
        second = adapter(poisoned, max_steps=4, min_steps=1, allow_untrained=True)
    assert torch.equal(first.active_masks, second.active_masks)
    assert torch.equal(first.decision_masks, second.decision_masks)
    assert torch.equal(first.reasoning_steps_per_sample, second.reasoning_steps_per_sample)
    assert torch.equal(first.predicted_gains, second.predicted_gains)
    assert torch.equal(first.continue_probabilities, second.continue_probabilities)
    assert torch.equal(first.forecast, second.forecast)


def test_a_checkpoint_round_trip_carries_every_rw_b_tensor(tmp_path):
    model = build(source_role_markers=True, local_solver_state=True)
    data = batch()
    contract = {"kind": "process", "model": {"dim": SMALL["dim"],
                                             "source_role_markers": True,
                                             "local_solver_state": True},
                "note": "round-trip fixture"}
    payload = {"format": "r7-local-v1", "contract": contract,
               "signature": canonical_digest(contract), "model": model.state_dict(),
               "updates": 1}
    path = tmp_path / "update_0000001.pt"
    save_exclusive(path, payload)
    restored = load_checkpoint(path)
    rebuilt = build(source_role_markers=True, local_solver_state=True)
    rebuilt.load_state_dict(restored["model"], strict=True)
    assert set(restored["model"]) == set(rebuilt.state_dict())
    assert any(name.startswith("solver_cell") for name in restored["model"])
    assert {"role_context", "role_draft"} <= set(restored["model"])
    model.eval(), rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt(forecast_inputs(data)).forecast,
                           model(forecast_inputs(data)).forecast)


def test_bf16_streamed_backward_finishes_with_finite_gradients():
    model = build(local_solver_state=True).train()
    result = backward_streamed_truncated(
        model, batch(), reasoning_steps=STEPS, process_weight=0.5, loss_scale=1.0,
        amp_dtype=torch.bfloat16)
    assert torch.isfinite(result.total)
    assert torch.isfinite(result.draft_errors).all()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "the streamed backward left no gradient behind"
    assert all(torch.isfinite(g).all() for g in grads)
