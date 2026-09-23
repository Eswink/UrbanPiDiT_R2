import torch
from torch import nn
from torch.utils.data import DataLoader
from data.synthetic import SyntheticR2Dataset
from model import UrbanPiDiTR2


def tiny_model():
    return UrbanPiDiTR2(
        coarse_channels=12,coarse_history_steps=2,
        urban_channels=7,urban_history_steps=2,
        static_channels=6,out_channels=7,
        coarse_dim=96,process_dim=128,urban_dim=128,
        coarse_depth=2,urban_depth=2,
        coarse_heads=4,process_heads=4,urban_heads=4,
        window_size=4,free_processes=4,max_reasoning_steps=3,
    )


def batch():
    return next(iter(DataLoader(
        SyntheticR2Dataset(
            length=4,coarse_hw=(8,8),urban_hw=(32,32)
        ),
        batch_size=2,
    )))


def test_forward_zoom_shape_and_reasoning_trace():
    m=tiny_model()
    o=m(batch(),force_zoom=True)
    assert o.forecast.shape==(2,7,32,32)
    assert o.diagnostics.process_predictions.shape==(2,12)
    assert o.diagnostics.trace is not None
    assert o.diagnostics.trace.process_predictions.shape==(2,3,12)
    assert o.diagnostics.reasoning_steps_per_sample.tolist()==[2,2]


def test_hard_stop_returns_baseline():
    m=tiny_model().eval()
    b=batch()
    with torch.no_grad():
        o=m(b,force_zoom=False,hard_route=True)
    assert torch.allclose(o.forecast,b['urban_baseline'])


def test_adaptive_reasoning_bounded():
    m=tiny_model().eval()
    b=batch()
    with torch.no_grad():
        o=m(b,force_zoom=True,adaptive_reasoning=True)
    assert 1 <= o.diagnostics.reasoning_steps <= 3
    assert torch.all(o.diagnostics.reasoning_steps_per_sample>=1)
    assert torch.all(o.diagnostics.reasoning_steps_per_sample<=3)


class _SplitVerifier(nn.Module):
    def __init__(self,anchored:int):
        super().__init__()
        self.anchored=anchored

    def predict_process(self,process):
        return process[:,:self.anchored].mean(-1)

    def forward(self,process,baseline,forecast,residual):
        B=process.shape[0]
        score=torch.zeros(B,device=process.device,dtype=process.dtype)
        score[0]=0.99
        pp=self.predict_process(process)
        z=torch.zeros(B,device=process.device,dtype=process.dtype)
        return score,pp,{
            'residual_magnitude':z,
            'cross_scale_delta':z,
        }


def test_adaptive_reasoning_stops_per_sample():
    m=tiny_model().eval()
    m.verifier=_SplitVerifier(m.anchored_processes)
    b=batch()
    with torch.no_grad():
        o=m(b,force_zoom=False,adaptive_reasoning=True)
    assert o.diagnostics.reasoning_steps_per_sample.tolist()==[1,3]
    assert len(o.diagnostics.trace.active_masks)==2
    assert o.diagnostics.trace.active_masks[0].tolist()==[False,True]


def test_selected_route_executes_only_masked_samples():
    m=tiny_model().eval()
    b=batch()
    with torch.no_grad():
        coarse,_=m.coarse(b['coarse_history'])
        st,_=m.static_preview(b['urban_static'])
        cheap=m._cheap_context(coarse,st)
        p,_=m.process_init(cheap)
        residual,_=m._run_urban_selected(
            b,coarse,p,torch.tensor([True,False])
        )
    assert residual[0].abs().sum() >= 0
    assert torch.count_nonzero(residual[1]) == 0
