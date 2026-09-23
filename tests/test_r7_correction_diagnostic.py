import pytest
import torch
from torch import nn
from types import SimpleNamespace
from training.r7_correction_diagnostic import correction_terms, collect_correction_terms


def field(x):
    return torch.full((1, 1, 3, 4), float(x), dtype=torch.float64)


@pytest.mark.parametrize("after,wrong,over,alpha", [(0,False,False,1), (-2,False,True,1/3),
                                                   (2,True,False,0), (1,False,False,0)])
def test_analytic_corrections(after, wrong, over, alpha):
    r = correction_terms(field(1),field(after),field(0))
    assert r["wrong_direction"].item() == wrong
    assert r["overshoot"].item() == over
    assert r["retrospective_damping"].item() == pytest.approx(alpha)
    assert r["algebra_residual"].abs().max() < 1e-12
    assert r["retrospective_damped_mse"] <= r["mse_before"] + 1e-12
    assert r["retrospective_damped_mse"] <= r["mse_after"] + 1e-12


def test_weighted_identity_and_alignment():
    g = torch.Generator().manual_seed(3)
    a,b,y = [torch.randn(2,3,4,5,generator=g,dtype=torch.float64) for _ in range(3)]
    lat = torch.tensor([0.,20.,40.,60.])
    r = correction_terms(a,b,y,lat,b-a)
    w = torch.cos(torch.deg2rad(lat.double()))
    expected = ((a-y).square()*w[None,None,:,None]).sum((-2,-1))/(w.sum()*5)
    torch.testing.assert_close(r["mse_before"],expected)
    assert r["algebra_residual"].abs().max() < 1e-12
    torch.testing.assert_close(r["previous_update_cosine"],torch.ones(2,3,dtype=torch.float64))
    rev = correction_terms(a.flip(-2),b.flip(-2),y.flip(-2),lat.flip(0))
    torch.testing.assert_close(r["mse_after"],rev["mse_after"])


def test_zero_vectors_defined_masks():
    r = correction_terms(field(0),field(0),field(0),previous_update=field(0))
    assert not r["error_cosine_defined"].any() and not r["previous_cosine_defined"].any()
    assert r["zero_update"].all()
    assert r["previous_update_cosine"].item() == 0


@pytest.mark.parametrize("bad", [torch.zeros(1,2), torch.zeros(1,1,3,4,dtype=torch.int64),
                                  torch.full((1,1,3,4),float("nan"))])
def test_malformed_fields(bad):
    with pytest.raises(ValueError): correction_terms(bad,field(1),field(0))


@pytest.mark.parametrize("lat", [[91,0,0], [90,90,90], [1,2], [float("nan"),0,0]])
def test_invalid_latitude(lat):
    with pytest.raises(ValueError): correction_terms(field(0),field(1),field(0),lat)


def test_overflow_and_previous_mismatch():
    with pytest.raises(ValueError,match="overflow"):
        correction_terms(field(1e200),field(0),field(0))
    with pytest.raises(ValueError,match="previous"):
        correction_terms(field(0),field(1),field(0),previous_update=torch.zeros(3,4))


class Spy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(.1))
    def forward(self,batch,reasoning_steps):
        assert set(batch) == {"coarse_history","lead_time_hours"}
        first = batch["coarse_history"][:,-1]
        return SimpleNamespace(draft_forecasts=torch.stack(
            [first+k*self.bias for k in range(reasoning_steps+1)],1))


def test_labels_never_passed_and_model_unchanged():
    m = Spy().eval()
    batch = dict(coarse_history=torch.ones(2,2,1,3,4),atmos_target=torch.zeros(2,1,3,4),
                 process_targets=torch.zeros(2,8),atmos_baseline=torch.zeros(2,1,3,4),
                 lead_time_hours=torch.tensor([6.,6.]))
    before = m.bias.detach().clone()
    r = collect_correction_terms(m,batch,max_steps=3)
    assert len(r)==3 and m.bias.grad is None
    torch.testing.assert_close(m.bias,before)
    assert r[1]["previous_update_cosine"].mean() == pytest.approx(1)
    m.train()
    with pytest.raises(ValueError,match="eval"):collect_correction_terms(m,batch)
    m.eval()
    for k in (0,True,9):
        with pytest.raises(ValueError):collect_correction_terms(m,batch,max_steps=k)
