"""CI-safe checks for the wide interior-supervision (dilution control) driver.

The round separates "the frozen box lacks long-lead information" from "full-grid
supervision diluted the signal" by keeping the wide 129x129 input and moving the
loss back onto the frozen central 65x65 box. The fidelity-critical parts are that
the mask is exactly ``interior_32`` (4225 cells = the narrow arm's own grid), that
the masked objective is the restricted weighted mean rather than the full-grid
mean, that ``mask=None`` still reproduces the registered objective, and that the
driver refuses an existing output or a missing store.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interior = _load("study_r7_s3_wide_interior_supervision")
wide = interior.wide


def test_recipe_is_the_registered_one():
    assert interior.FORMAT == "r7-s3-wide-interior-supervision-protocol-v1"
    assert wide.SEEDS == (41, 42, 43)
    assert (wide.UPDATES, wide.LR, wide.WARMUP) == (800, 2e-5, 10)
    assert wide.PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.EVALUATION_LEADS == (6, 12, 24, 48, 72)
    assert interior.SUPERVISION_KIND == "interior_32"


def test_supervision_mask_is_exactly_the_frozen_central_box():
    mask = interior.supervision_mask()
    assert mask.shape == (129, 129)
    assert int(mask.sum()) == 65 * 65 == 4225
    # counterproof: the mask is not the full grid and not the outer ring
    assert int(mask.sum()) != 129 * 129
    assert bool(mask[0].any()) is False and bool(mask[-1].any()) is False
    assert bool(mask[32, 32]) is True and bool(mask[31, 31]) is False


def test_supervision_block_matches_what_the_runner_will_record():
    from training.r7_long_rollout_runner import _supervision_block

    block = interior.supervision_block()
    assert block["kind"] == "interior_32"
    assert block["runner_block"] == _supervision_block(interior.supervision_mask())
    assert block["runner_block"]["kind"] == "spatial_mask"
    assert block["runner_block"]["selected_cells"] == 4225
    assert block["runner_block"]["shape"] == [129, 129]
    assert _supervision_block(None) == {"kind": "full_grid",
                                        "description": "every cell of the target grid is supervised"}


def test_full_grid_supervision_block_refuses_a_degenerate_mask():
    from training.r7_long_rollout_runner import _supervision_block

    with pytest.raises(ValueError):
        _supervision_block([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        _supervision_block([1, 1, 1])
    with pytest.raises(ValueError):
        _supervision_block([[1, -1], [1, 1]])


def test_masked_objective_is_the_restricted_weighted_mean():
    import torch
    from training.r7_losses import latitude_weighted_mse

    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1., 0.], [0., 0.]]]])
    latitude = torch.tensor([0., 60.])
    only_first = torch.zeros(2, 2, dtype=torch.bool)
    only_first[0, 0] = True
    # Full grid: cos(60)=0.5 is renormalized to 0.5/0.75 = 2/3, so the single
    # nonzero cell contributes (4/3)/4.
    assert torch.allclose(latitude_weighted_mse(prediction, target, latitude),
                          torch.tensor(1.0 / 3.0))
    # Masked to that one cell: the error there is 1, so the restricted mean is 1.
    assert torch.allclose(latitude_weighted_mse(prediction, target, latitude, mask=only_first),
                          torch.tensor(1.0))
    # An all-ones mask must reproduce the full-grid objective exactly.
    everything = torch.ones(2, 2, dtype=torch.bool)
    assert torch.allclose(latitude_weighted_mse(prediction, target, latitude, mask=everything),
                          latitude_weighted_mse(prediction, target, latitude))
    # No latitude: the masked mean over the single cell is still 1, not 0.25.
    assert torch.allclose(latitude_weighted_mse(prediction, target, None, mask=only_first),
                          torch.tensor(1.0))


def test_deep_supervision_masked_matches_its_own_draft_losses():
    import torch
    from training.r7_recursive_losses import deep_supervised_forecast_mse
    from training.r7_losses import latitude_weighted_mse

    torch.manual_seed(0)
    drafts = torch.randn(2, 3, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    latitude = torch.tensor([10., 20., 30., 40.])
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1:3, 1:3] = True
    weights = torch.linspace(1.0, 2.0, 3)
    weights = weights / weights.sum()
    expected = sum(weight * latitude_weighted_mse(drafts[:, i], target, latitude, mask=mask)
                   for i, weight in enumerate(weights))
    assert torch.allclose(deep_supervised_forecast_mse(drafts, target, latitude, final_weight=2.,
                                                       mask=mask), expected)


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        interior.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    # With no built store the driver must fail on the missing store, not run; the
    # store path is monkeypatched so this holds whether or not it is built here.
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        interior.main(["--out", str(out)])
    assert not out.exists()
