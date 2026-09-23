import torch
from model.layers.window_attention import WindowAttentionBlock

def test_window_attention_nondivisible_hw():
    b=WindowAttentionBlock(64,heads=4,window_size=4); x=torch.randn(2,35,64); y=b(x,(5,7)); assert y.shape==x.shape
