from __future__ import annotations
import copy
from unittest.mock import patch
import pytest
import torch
from model.recursive_weather_r7 import solver_conditioning
from model.r7_halting import AdaptiveProcessForecaster
from training.r7_streaming import backward_streamed_truncated,_recursive_step
from test_r7_streaming import model_and_batch,retained_loss,assert_gradients_equal
from test_r7_adaptive_inference import ScriptedController,TargetPoison
from training.r7_halting import controller_calibration_loss


@pytest.mark.parametrize("kind",["generic","process"])
def test_default_equation_and_no_new_parameters(kind):
    default,b=model_and_batch(kind)
    false,_=model_and_batch(kind,spatial_solver_feedback=False)
    true,_=model_and_batch(kind,spatial_solver_feedback=True)
    assert list(default.state_dict())==list(true.state_dict())
    assert sum(p.numel() for p in false.parameters())==sum(p.numel() for p in true.parameters())
    for name,value in default.state_dict().items():
        torch.testing.assert_close(value,false.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(value,true.state_dict()[name],rtol=0,atol=0)
    default.eval()
    module="model.process_forecast_r7" if kind=="process" else "model.recursive_weather_r7"
    with torch.no_grad():
        actual=default(b,reasoning_steps=3)
        with patch(module+".solver_conditioning",
                   lambda context,summary,draft_tokens=None,**kw:context+summary[:,None,:]):
            explicit_old=default(b,reasoning_steps=3)
    torch.testing.assert_close(actual.draft_forecasts,explicit_old.draft_forecasts,rtol=0,atol=0)


@pytest.mark.parametrize("kind",["generic","process"])
@pytest.mark.parametrize("k",[0,1,3])
@pytest.mark.parametrize("checkpointing",[False,True])
def test_streamed_gradient_equivalence_with_spatial_feedback(kind,k,checkpointing):
    ref,b=model_and_batch(kind,spatial_solver_feedback=True,activation_checkpointing=checkpointing)
    # Odd native grid also exercises aligned padded draft/context patches.
    b["coarse_history"]=torch.randn(2,2,4,9,11)
    b["atmos_target"]=torch.randn(2,4,9,11)
    b["latitude"]=torch.linspace(40,38,9).expand(2,-1)
    streamed=copy.deepcopy(ref)
    expected,loss=retained_loss(ref,b,k);loss.backward()
    log=backward_streamed_truncated(streamed,b,reasoning_steps=k)
    torch.testing.assert_close(expected.forecast,log.final_forecast)
    torch.testing.assert_close(loss.detach(),log.total,rtol=1e-5,atol=1e-6)
    assert_gradients_equal(ref,streamed)


@pytest.mark.parametrize("kind",["generic","process"])
@pytest.mark.parametrize("enabled",[False,True])
def test_fixed_latent_local_draft_information_reaches_solver(kind,enabled):
    model,b=model_and_batch(kind,spatial_solver_feedback=enabled)
    model.eval()
    with torch.no_grad():
        base=model.backbone(b)
        query=model.process_queries if kind=="process" else model.latent
        state=query.expand(2,-1,-1)
        # Deliberately freeze latent to separate direct spatial from pooled feedback.
        name="_reason" if kind=="process" else "_cell"
        contexts=[]
        hook=model.correction_head.register_forward_pre_hook(lambda m,a:contexts.append(a[0].clone()))
        changed=base.forecast.clone()
        changed[:,0,:2,:2]+=3.
        with patch.object(model,name,lambda state,context:state):
            _recursive_step(model,state,base.context_tokens,base.forecast,base.token_hw)
            _recursive_step(model,state,base.context_tokens,changed,base.token_hw)
        hook.remove()
    delta=contexts[1]-contexts[0]
    if enabled:
        assert delta[:,0].abs().max()>1e-5
        torch.testing.assert_close(delta[:,1:],torch.zeros_like(delta[:,1:]),rtol=0,atol=0)
    else:
        torch.testing.assert_close(delta,torch.zeros_like(delta),rtol=0,atol=0)


def test_disable_forecast_feedback_disables_both_paths():
    model,b=model_and_batch(spatial_solver_feedback=True,use_forecast_feedback=False)
    streamed=copy.deepcopy(model)
    _,loss=retained_loss(model,b,3);loss.backward()
    backward_streamed_truncated(streamed,b,reasoning_steps=3)
    assert_gradients_equal(model,streamed)
    assert all(p.grad is None for p in model.draft_encoder.parameters())
    model.eval()
    with torch.no_grad():
        a=model(b,reasoning_steps=3)
        model.spatial_solver_feedback=False
        c=model(b,reasoning_steps=3)
    torch.testing.assert_close(a.draft_forecasts,c.draft_forecasts,rtol=0,atol=0)


def test_spatial_adaptive_full_and_active_subsets_target_free():
    model,b=model_and_batch(spatial_solver_feedback=True)
    model.eval()
    adapter=AdaptiveProcessForecaster(model).eval()
    with torch.no_grad():
        fixed=model(b,reasoning_steps=3)
        once=model(b,reasoning_steps=1)
    full=adapter(b,max_steps=3,force_full_depth=True)
    torch.testing.assert_close(full.forecast,fixed.forecast)
    adapter.controller=ScriptedController("mixed").eval()
    sizes=[]
    h=model.correction_head.register_forward_pre_hook(lambda m,a:sizes.append(a[0].shape[0]))
    result=adapter(TargetPoison(b),max_steps=3,allow_untrained=True)
    h.remove()
    assert sizes==[2,1,1] and result.reasoning_steps_per_sample.tolist()==[1,3]
    torch.testing.assert_close(result.forecast[0],once.forecast[0])
    torch.testing.assert_close(result.forecast[1],fixed.forecast[1])


def test_spatial_bf16_calibration_and_streaming_finite():
    model,b=model_and_batch(spatial_solver_feedback=True)
    log=backward_streamed_truncated(model,b,reasoning_steps=3,amp_dtype=torch.bfloat16)
    assert torch.isfinite(log.total)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    adapter=AdaptiveProcessForecaster(model)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        loss=controller_calibration_loss(adapter,b,max_steps=3)
    loss.total.backward()
    assert all(p.grad is None for p in model.parameters())
    adapter.eval()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        out=adapter(b,max_steps=3,force_full_depth=True)
    assert torch.isfinite(out.forecast).all()


@pytest.mark.parametrize("bad",[1,0,"false",None])
def test_flag_strict_boolean(bad):
    for kind in ("generic","process"):
        with pytest.raises(ValueError,match="boolean"):
            model_and_batch(kind,spatial_solver_feedback=bad)


def test_solver_shape_guards_and_identity():
    c,s,d=torch.randn(2,3,4),torch.randn(2,4),torch.randn(2,3,4)
    torch.testing.assert_close(solver_conditioning(c,s,d),c+s[:,None,:],rtol=0,atol=0)
    with pytest.raises(ValueError):solver_conditioning(c,s[:1],d)
    with pytest.raises(ValueError):solver_conditioning(c,s,None,spatial_feedback=True)
    with pytest.raises(ValueError):solver_conditioning(c,s,d[:1],spatial_feedback=True)
