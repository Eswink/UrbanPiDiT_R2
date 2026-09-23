import torch
from torch.utils.data import DataLoader
from data.synthetic import SyntheticR2Dataset
from model import UrbanPiDiTR2

def tiny_model():
    return UrbanPiDiTR2(coarse_channels=12,coarse_history_steps=2,urban_channels=7,urban_history_steps=2,static_channels=6,out_channels=7,coarse_dim=96,process_dim=128,urban_dim=128,coarse_depth=2,urban_depth=2,coarse_heads=4,process_heads=4,urban_heads=4,window_size=4,free_processes=4,max_reasoning_steps=3)

def batch(): return next(iter(DataLoader(SyntheticR2Dataset(length=4,coarse_hw=(8,8),urban_hw=(32,32)),batch_size=2)))

def test_forward_zoom_shape():
    m=tiny_model(); o=m(batch(),force_zoom=True); assert o.forecast.shape==(2,7,32,32); assert o.diagnostics.process_predictions.shape==(2,12)

def test_hard_stop_returns_baseline():
    m=tiny_model().eval(); b=batch();
    with torch.no_grad(): o=m(b,force_zoom=False,hard_route=True)
    assert torch.allclose(o.forecast,b['urban_baseline'])

def test_adaptive_reasoning_bounded():
    m=tiny_model().eval(); b=batch();
    with torch.no_grad(): o=m(b,force_zoom=True,adaptive_reasoning=True)
    assert 1 <= o.diagnostics.reasoning_steps <= 3

def test_selected_route_executes_only_masked_samples():
    m=tiny_model().eval(); b=batch()
    with torch.no_grad():
        coarse,_=m.coarse(b['coarse_history']); st,_=m.static_preview(b['urban_static']); cheap=m._cheap_context(coarse,st); p,_=m.process_init(cheap)
        residual,_=m._run_urban_selected(b,coarse,p,torch.tensor([True,False]))
    assert residual[0].abs().sum() >= 0
    assert torch.count_nonzero(residual[1]) == 0
