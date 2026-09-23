import torch
from model.layers.window_attention import WindowAttentionBlock, _shift_attention_mask


def test_window_attention_nondivisible_hw():
    b=WindowAttentionBlock(64,heads=4,window_size=4)
    x=torch.randn(2,35,64)
    y=b(x,(5,7))
    assert y.shape==x.shape
    assert torch.isfinite(y).all()


def test_shift_mask_blocks_finite_wraparound():
    mask=_shift_attention_mask(
        h=8,w=8,hp=8,wp=8,window=4,shift=2,batch=1,
        device=torch.device("cpu"),dtype=torch.float32,periodic_width=False,
    )
    assert mask.shape==(4,1,16,16)
    last=mask[-1,0]
    assert torch.isneginf(last).any()
    assert torch.isfinite(last).any()


def test_periodic_width_keeps_longitude_wrap_but_not_latitude_wrap():
    finite=_shift_attention_mask(
        h=8,w=8,hp=8,wp=8,window=4,shift=2,batch=1,
        device=torch.device("cpu"),dtype=torch.float32,periodic_width=False,
    )
    periodic=_shift_attention_mask(
        h=8,w=8,hp=8,wp=8,window=4,shift=2,batch=1,
        device=torch.device("cpu"),dtype=torch.float32,periodic_width=True,
    )
    assert torch.isneginf(periodic).sum() < torch.isneginf(finite).sum()
    assert torch.isneginf(periodic).any()


def test_shifted_nondivisible_forward_is_finite():
    b=WindowAttentionBlock(32,heads=4,window_size=4,shift=True)
    x=torch.randn(2,35,32)
    y=b(x,(5,7))
    assert y.shape==x.shape
    assert torch.isfinite(y).all()
