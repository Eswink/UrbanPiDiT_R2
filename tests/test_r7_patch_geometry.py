import pytest
import torch
from torch import nn
from model.coarse_encoder import CoarseEncoder
from model.coarse_forecast import CoarseForecastHead
from model.weather_forecaster_r7 import NativeAtmosForecaster
from model.recursive_weather_r7 import DraftTokenEncoder,GenericRecursiveWeatherForecaster
from model.process_forecast_r7 import ProcessForecastCoReasoner


@pytest.mark.parametrize('hw',[(1,1),(1,7),(9,15),(8,12)])
@pytest.mark.parametrize('kind',['native','generic','process'])
def test_small_odd_and_even_native_outputs(hw,kind):
    cls={'native':NativeAtmosForecaster,'generic':GenericRecursiveWeatherForecaster,'process':ProcessForecastCoReasoner}[kind]
    m=cls(in_channels=2,dim=16,depth=1,heads=4,window_size=2)
    x=torch.randn(1,2,2,*hw,requires_grad=True)
    out=m({'coarse_history':x})
    assert out.forecast.shape==(1,2,*hw)
    expected=((hw[0]+1)//2,(hw[1]+1)//2)
    assert out.token_hw==expected
    out.forecast.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_trailing_corner_enters_patch_features():
    enc=CoarseEncoder(2,2,dim=8,depth=0,heads=2,pad_to_patch=True)
    a=torch.randn(1,2,2,9,15)
    b=a.clone()
    b[:,:,:,-1,-1]+=7
    ta,ha=enc(a)
    tb,hb=enc(b)
    assert ha==hb==(5,8)
    assert not torch.allclose(ta[:,-1],tb[:,-1])
    draft=DraftTokenEncoder(2,8,2)
    assert not torch.allclose(draft(a[:,-1])[0][:,-1],draft(b[:,-1])[0][:,-1])


def test_legacy_encoder_default_is_unchanged():
    enc=CoarseEncoder(2,2,dim=8,depth=0,heads=2)
    assert enc(torch.randn(1,2,2,9,15))[1]==(4,7)


def test_decoder_crops_instead_of_resizing():
    head=CoarseForecastHead(8,2,patch_size=2)
    tokens=torch.randn(1,40,8)
    full=head.decode(head.norm(tokens).transpose(1,2).reshape(1,8,5,8))
    base=torch.zeros(1,2,9,15)
    _,actual=head(tokens,(5,8),(9,15),base)
    torch.testing.assert_close(actual,full[...,:9,:15],rtol=0,atol=0)
    with pytest.raises(ValueError,match='geometry'):
        head(tokens,(5,8),(11,15),torch.zeros(1,2,11,15))


def test_divisible_encoder_keeps_identical_state_dict_behavior():
    old=CoarseEncoder(2,2,dim=8,depth=2,heads=2,window_size=2)
    new=CoarseEncoder(2,2,dim=8,depth=2,heads=2,window_size=2,pad_to_patch=True)
    new.load_state_dict(old.state_dict())
    x=torch.randn(1,2,2,8,12)
    torch.testing.assert_close(old(x)[0],new(x)[0],rtol=0,atol=0)
