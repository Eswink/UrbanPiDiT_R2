from __future__ import annotations

import torch

from models.morpho_process_proxy import PROCESS_PROXY_NAMES
from models.process_conditioned_adaln import ProcessConditionedAdaLN


def _proxy_fields(batch: int = 2, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
    return {name: torch.rand(batch, 1, height, width) for name in PROCESS_PROXY_NAMES}


def test_process_adaln_zero_alpha_matches_base_modulation() -> None:
    batch, nodes, dim = 2, 16, 32
    adaln = ProcessConditionedAdaLN(dim=dim, depth=3, init_alpha=0.0)
    x = torch.randn(batch, nodes, dim)
    shift = torch.randn(batch, dim)
    scale = torch.randn(batch, dim)

    out = adaln(x, shift, scale, proxy_fields=_proxy_fields(batch), block_idx=1)
    expected = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    assert torch.allclose(out, expected, atol=1e-6)


def test_process_adaln_alpha_gets_gradient() -> None:
    batch, nodes, dim = 2, 16, 32
    adaln = ProcessConditionedAdaLN(dim=dim, depth=2, init_alpha=0.0)
    x = torch.randn(batch, nodes, dim)
    shift = torch.zeros(batch, dim)
    scale = torch.zeros(batch, dim)

    out = adaln(x, shift, scale, proxy_fields=_proxy_fields(batch), block_idx=0)
    loss = out.square().mean()
    loss.backward()

    assert adaln.alpha.grad is not None
    assert adaln.alpha.grad[0].abs() > 0


def test_process_adaln_per_block_alpha_is_independent() -> None:
    adaln = ProcessConditionedAdaLN(dim=16, depth=4, init_alpha=0.0)

    assert adaln.alpha.shape == (4,)
    with torch.no_grad():
        adaln.alpha[2] = 0.5

    assert float(adaln.alpha[2].detach()) != float(adaln.alpha[1].detach())