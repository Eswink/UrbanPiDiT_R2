from __future__ import annotations

import torch

from models.urban_morphology_control import UrbanMorphologyControlBranch


def _features(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 8, 8, 8),
        torch.randn(batch, 16, 4, 4),
        torch.randn(batch, 32, 2, 2),
    ]


def test_control_branch_outputs_per_block_token_residuals() -> None:
    branch = UrbanMorphologyControlBranch(feature_channels=(8, 16, 32), dim=24, depth=3)

    residuals = branch(_features(), torch.randn(2, 24))

    assert len(residuals) == 3
    assert [tuple(x.shape) for x in residuals] == [(2, 64, 24), (2, 64, 24), (2, 64, 24)]


def test_control_branch_zero_init_returns_zero_residuals() -> None:
    branch = UrbanMorphologyControlBranch(feature_channels=(8, 16, 32), dim=24, depth=3)

    residuals = branch(_features(), torch.randn(2, 24))

    for residual in residuals:
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_control_branch_disabled_returns_empty_list() -> None:
    branch = UrbanMorphologyControlBranch(feature_channels=(8, 16, 32), dim=24, depth=3, enabled=False)

    assert branch(_features(), torch.randn(2, 24)) == []