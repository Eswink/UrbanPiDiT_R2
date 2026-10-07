"""Synthetic CPU counterproofs for the local solver's feedback requirement."""
import hashlib

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_step_r7 import ProcessStepInput, process_reasoning_step


@pytest.fixture(params=[(anchor, recurrence, proposal) for anchor in (False, True)
                        for recurrence in (False, True) for proposal in (False, True)])
def solver_case(request):
    anchor, recurrence, proposal = request.param
    table = torch.arange(32, dtype=torch.float32).reshape(4, 2, 2, 2) / 8
    spec = dict(format='r7-train-climatology-anchor-v1', kind='train-only-month-hour-grid-mean-v1',
                channels=['toy_0', 'toy_1'], units=['toy_units', 'toy_units'],
                source_sha256='1' * 64, train_data_identity='2' * 64,
                climatology_mean_identity_sha256='3' * 64,
                normalization_mean=[0.0, 0.0], normalization_std=[1.0, 1.0],
                latitude=[50.0, 40.0], longitude=[100.0, 110.0],
                bucket_keys=[[1, hour] for hour in (0, 6, 12, 18)], bucket_counts=[2] * 4,
                training_years=[2017], n_selected_steps=8, selection='declared_train_years',
                table_sha256=hashlib.sha256(table.numpy().tobytes()).hexdigest(),
                table_shape=list(table.shape))
    torch.manual_seed(41)
    model = ProcessForecastCoReasoner(
        in_channels=2, out_channels=2, history_steps=2, dim=8, depth=1, heads=2,
        patch_size=2, window_size=2, dropout=0.0, anchored_processes=4, free_processes=1,
        positional_process_readout=True, local_solver_state=True,
        solver_state_recurrence=recurrence, solver_gate_proposal=proposal,
        draft_query_feedback=False, anomaly_feedback=False,
        climatology_anchor_spec=spec if anchor else None)
    if anchor:
        model.backbone.climatology_anchor.install(table)
    batch = dict(coarse_history=torch.randn(1, 2, 2, 2, 2),
                 latitude=torch.tensor(spec['latitude']), longitude=torch.tensor(spec['longitude']),
                 init_calendar_year=torch.tensor([2017]), init_day_of_year=torch.tensor([1]),
                 init_utc_hour=torch.tensor([0.0]), lead_time_hours=torch.tensor([6.0]))
    return model.eval(), batch


@pytest.mark.parametrize('steps', [0, 1])
def test_fixed_local_solver_rejects_feedback_false_even_at_zero_steps(solver_case, steps):
    model, batch = solver_case
    with torch.no_grad():
        ordinary = model(batch, reasoning_steps=steps)
        explicit = model(batch, reasoning_steps=steps, use_forecast_feedback=True)
        assert torch.isfinite(ordinary.forecast).all()
        assert torch.equal(ordinary.forecast, explicit.forecast)
        with pytest.raises(ValueError, match='local_solver_state requires use_forecast_feedback=True'):
            model(batch, reasoning_steps=steps, use_forecast_feedback=False)


def test_shared_local_solver_rejects_feedback_false_for_every_solver_branch(solver_case):
    model, batch = solver_case
    with torch.no_grad():
        base = model.backbone(batch)
        process = model.initial_process_state(model.process_queries.expand(1, -1, -1), batch,
                                              anchor=base.base_state, draft=base.forecast)
        tensors = ProcessStepInput(process, base.context_tokens, base.forecast, base.climatology_anchor)
        ordinary = process_reasoning_step(model, tensors, base.token_hw, anchor=base.base_state,
                                          use_forecast_feedback=True)
        assert torch.isfinite(ordinary.draft).all()
        with pytest.raises(ValueError, match='local_solver_state requires use_forecast_feedback=True'):
            process_reasoning_step(model, tensors, base.token_hw, anchor=base.base_state,
                                   use_forecast_feedback=False)
