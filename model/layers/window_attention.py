from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .sdpa import FeedForward


def _pad_hw(x, window):
    B,H,W,D=x.shape
    ph=(window-H%window)%window
    pw=(window-W%window)%window
    if ph or pw:
        x=F.pad(x,(0,0,0,pw,0,ph))
    return x,(H,W)


def _partition(x,w):
    B,H,W,D=x.shape
    return x.view(B,H//w,w,W//w,w,D).permute(0,1,3,2,4,5).reshape(-1,w*w,D)


def _reverse(wins,w,B,H,W,D):
    return wins.view(B,H//w,W//w,w,w,D).permute(0,1,3,2,4,5).reshape(B,H,W,D)


def _shift_attention_mask(
    *,
    h:int,
    w:int,
    hp:int,
    wp:int,
    window:int,
    shift:int,
    batch:int,
    device:torch.device,
    dtype:torch.dtype,
    periodic_width:bool=False,
):
    """Build an additive mask for shifted finite-domain windows.

    The feature map is padded before the cyclic shift. For a finite domain, tokens
    that crossed the top/bottom or left/right edge because of torch.roll must not
    attend to tokens from the opposite edge. Width can be explicitly periodic
    for global-longitude style grids. Padded keys are always masked.
    """
    valid=torch.zeros((1,hp,wp,1),device=device,dtype=torch.bool)
    valid[:,:h,:w]=True
    if shift:
        valid=torch.roll(valid,shifts=(-shift,-shift),dims=(1,2))

    yy=torch.arange(hp,device=device)
    xx=torch.arange(wp,device=device)
    wrap_y=((yy+shift)>=hp).long() if shift else torch.zeros_like(yy)
    if shift and not periodic_width:
        wrap_x=((xx+shift)>=wp).long()
    else:
        wrap_x=torch.zeros_like(xx)
    region=(wrap_y[:,None]*2+wrap_x[None,:]).view(1,hp,wp,1)

    region_w=_partition(region,window).squeeze(-1)
    valid_w=_partition(valid,window).squeeze(-1)

    same_region=region_w[:,:,None].eq(region_w[:,None,:])
    key_valid=valid_w[:,None,:]
    allowed=same_region & key_valid

    mask=torch.zeros(allowed.shape,device=device,dtype=dtype)
    mask=mask.masked_fill(~allowed,float("-inf"))
    mask=mask[:,None,:,:]
    if batch>1:
        mask=mask.repeat(batch,1,1,1)
    return mask


class WindowAttentionBlock(nn.Module):
    """Local window attention with explicit finite-domain shift semantics.

    Complexity remains O(HW*window^2). By default both spatial axes are finite.
    periodic_width=True may be used when the width axis represents a truly
    periodic longitude domain.
    """
    def __init__(
        self,
        dim:int,
        heads:int=8,
        window_size:int=8,
        mlp_ratio:float=4.0,
        dropout:float=0.0,
        shift:bool=False,
        periodic_width:bool=False,
    ):
        super().__init__()
        assert dim%heads==0
        self.dim=dim
        self.heads=heads
        self.ws=window_size
        self.shift=shift
        self.periodic_width=bool(periodic_width)
        self.hd=dim//heads
        self.dropout=float(dropout)
        self.n1=nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,dim*3)
        self.proj=nn.Linear(dim,dim)
        self.n2=nn.LayerNorm(dim)
        self.ff=FeedForward(dim,mlp_ratio,dropout)

    def forward(self,tokens:torch.Tensor, hw:tuple[int,int]):
        B,N,D=tokens.shape
        H,W=hw
        if N!=H*W:
            raise ValueError(f'token 数 {N} 与 hw={hw} 不一致')

        # Pad before shifting so padding and finite-boundary semantics can share
        # one attention mask. This matches the intended shifted-window geometry.
        x=self.n1(tokens).view(B,H,W,D)
        x,(oh,ow)=_pad_hw(x,self.ws)
        Hp,Wp=x.shape[1:3]
        shift=self.ws//2 if self.shift else 0
        if shift:
            x=torch.roll(x,shifts=(-shift,-shift),dims=(1,2))

        wins=_partition(x,self.ws)
        qkv=self.qkv(wins).view(
            wins.shape[0],wins.shape[1],3,self.heads,self.hd
        ).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]

        attn_mask=None
        if shift or Hp!=H or Wp!=W:
            attn_mask=_shift_attention_mask(
                h=H,w=W,hp=Hp,wp=Wp,window=self.ws,shift=shift,batch=B,
                device=q.device,dtype=q.dtype,periodic_width=self.periodic_width,
            )

        y=F.scaled_dot_product_attention(
            q,k,v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y=self.proj(y.transpose(1,2).reshape(wins.shape[0],wins.shape[1],D))
        x=_reverse(y,self.ws,B,Hp,Wp,D)
        if shift:
            x=torch.roll(x,shifts=(shift,shift),dims=(1,2))
        x=x[:,:oh,:ow]

        out=tokens+x.reshape(B,N,D)
        return out+self.ff(self.n2(out))
