import pytest
import torch
from training.r7_gain_oracle import retrospective_gain_analysis


def test_delayed_gain_is_missed_by_myopic_oracle():
    report=retrospective_gain_analysis(torch.tensor([[1.,1.1,.4],[1.,.8,.7]]),step_cost=.01)
    assert report['deployable'] is False and report['target_labels_used'] is True
    case=report['cases'][0]
    assert case['greedy_depth']==1 and case['optimal_retrospective_depth']==3
    assert case['objective_regret']==pytest.approx(.58,abs=1e-7)
    assert report['missed_delayed_benefit_count']==1
    assert report['cases'][1]['objective_regret']==0.


def test_ties_zero_cost_single_step_and_cost_tradeoff():
    for e in [torch.ones(2,1),torch.ones(2,4)]:
        report=retrospective_gain_analysis(e)
        assert all(r['greedy_depth']==r['optimal_retrospective_depth']==1 for r in report['cases'])
    errors=torch.tensor([[1.,.8,.7]])
    cheap=retrospective_gain_analysis(errors,step_cost=0.)['cases'][0]
    expensive=retrospective_gain_analysis(errors,step_cost=.5)['cases'][0]
    assert cheap['optimal_retrospective_depth']==3 and expensive['optimal_retrospective_depth']==1


@pytest.mark.parametrize('cost',[-1.,float('nan'),float('inf'),True])
def test_bad_cost(cost):
    with pytest.raises(ValueError):
        retrospective_gain_analysis(torch.ones(1,3),step_cost=cost)


@pytest.mark.parametrize('errors',[torch.empty(0,3),torch.ones(3),torch.ones(1,3,dtype=torch.long),
    torch.tensor([[float('nan')]]),torch.tensor([[-1.]]),torch.tensor([[float('inf')]])])
def test_invalid_errors(errors):
    with pytest.raises(ValueError):
        retrospective_gain_analysis(errors)


def test_no_graph_or_input_mutation_and_nonnegative_regret():
    generator=torch.Generator().manual_seed(5)
    errors=torch.rand(64,8,generator=generator,requires_grad=True)
    original=errors.detach().clone()
    result=retrospective_gain_analysis(errors,step_cost=.02)
    torch.testing.assert_close(errors,original)
    assert all(r['objective_regret']>=0 for r in result['cases'])
    assert errors.grad is None
