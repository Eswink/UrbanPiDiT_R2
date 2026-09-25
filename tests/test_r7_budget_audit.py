"""Offline tests for the parameter/FLOP budget audit (#5).

The acceptance criterion is that the generic baseline shares a comparable
parameter/FLOP budget with the process-aware model. These tests cover the parity
arithmetic and the counting contract without needing a GPU; the module runs on
CPU, so they also exercise real counting on small synthetic shapes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "training" / "r7_budget_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("r7_budget_audit_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_arithmetic_is_exact_and_symmetric():
    module = _module()
    same = module._parity(1000, 1000, 0.05)
    assert same["relative_difference"] == 0.0
    assert same["within_tolerance"] is True

    over = module._parity(1000, 1060, 0.05)
    assert over["relative_difference"] == pytest.approx(0.06)
    assert over["within_tolerance"] is False

    under = module._parity(1000, 940, 0.05)
    assert under["relative_difference"] == pytest.approx(-0.06)
    assert under["within_tolerance"] is False

    # Exactly at the tolerance is inside, and direction must not change that.
    edge_up = module._parity(1000, 1050, 0.05)
    edge_dn = module._parity(1000, 950, 0.05)
    assert edge_up["within_tolerance"] and edge_dn["within_tolerance"]


def test_parity_rejects_a_nonpositive_baseline():
    module = _module()
    for baseline in (0, -1):
        with pytest.raises(ValueError):
            module._parity(baseline, 10, 0.05)


def test_parity_tolerance_is_declared_before_measurement():
    """The tolerance must be a module constant, not chosen after seeing results."""
    module = _module()
    assert isinstance(module.PARITY_TOLERANCE, float)
    assert 0 < module.PARITY_TOLERANCE <= 0.25
    source = MODULE.read_text(encoding="utf-8")
    # The constant must be defined above its first use in audit().
    assert source.index("PARITY_TOLERANCE = ") < source.index("PARITY_TOLERANCE)")


def test_count_parameters_is_the_sum_over_named_parameters():
    torch = pytest.importorskip("torch")
    module = _module()
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.LayerNorm(3), torch.nn.Linear(3, 2))
    linear_a = 4 * 3 + 3      # weight + bias
    layer_norm = 3 + 3        # elementwise affine weight + bias
    linear_b = 3 * 2 + 2
    expected = linear_a + layer_norm + linear_b
    assert module.count_parameters(model) == expected
    # Must agree with the framework's own accounting, not just my arithmetic.
    assert module.count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_forward_flops_grow_with_reasoning_depth():
    """The K axis must be the thing that costs; otherwise the audit is measuring nothing."""
    import torch

    module = _module()
    torch.manual_seed(0)
    batch = {
        "coarse_history": torch.randn(2, 2, 17, 12, 12),
        "atmos_target": torch.randn(2, 17, 12, 12),
    }
    config = {"in_channels": 17, "out_channels": 17, "history_steps": 2, "dim": 32,
              "depth": 2, "heads": 4, "window_size": 8, "patch_size": 2,
              "dropout": 0.0, "default_reasoning_steps": 1, "latent_tokens": 16}
    from training.r7_experiment import make_model

    model = make_model("generic", config).eval()
    counts = {}
    for steps in (1, 2, 4):
        with torch.enable_grad():
            counts[steps] = module.count_forward_flops(model, batch, steps)
    assert counts[1] > 0
    assert counts[1] < counts[2] < counts[4], counts


def test_generic_and_process_share_a_comparable_budget_on_real_shapes():
    """The #5 acceptance criterion, checked directly on the published store.

    Skips without the optional real subset, like the other real-data tests:
    a clean checkout has no such file and no synthetic stand-in is allowed.
    """
    import torch

    module = _module()
    if not module.REAL_MANIFEST.is_file():
        pytest.skip("optional published regional ERA5 store is not present")

    from training.r7_experiment import make_model

    device = torch.device("cpu")
    batch = module.build_batch(device, batch_size=2)
    channels = int(batch["coarse_history"].shape[2])
    anchored = int(batch["process_targets"].shape[1])

    counts, flops = {}, {}
    for variant in module.VARIANTS:
        config = module.model_config_for(variant, channels, anchored, dim=64, depth=2)
        torch.manual_seed(0)
        model = make_model(variant, config).eval()
        counts[variant] = module.count_parameters(model)
        for steps in (1, 2):
            with torch.enable_grad():
                flops[(variant, steps)] = module.count_forward_flops(model, batch, steps)
        del model

    param_parity = module._parity(counts["generic"], counts["process"], module.PARITY_TOLERANCE)
    assert param_parity["within_tolerance"], counts
    for steps in (1, 2):
        entry = module._parity(flops[("generic", steps)], flops[("process", steps)],
                               module.PARITY_TOLERANCE)
        assert entry["within_tolerance"], (steps, flops)


def test_report_is_exclusive_and_self_describing(tmp_path):
    module = _module()
    report = {
        "format": "r7-recursive-budget-audit-v1", "scientific_claim": False,
        "acceptance_met": True, "rows": [],
        "parity": {"params": module._parity(10, 10, 0.05), "flops": {}},
    }
    path = tmp_path / "nested" / "report.json"
    module.write_report(path, report)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        module.write_report(path, report)


def test_module_declares_its_counting_convention_and_limits():
    """A FLOP number is meaningless without its convention, so it must be stated."""
    source = MODULE.read_text(encoding="utf-8")
    assert "FlopCounterMode" in source
    assert "forward" in source.lower()
    assert "backward is not counted" in source
    for flag in ("scientific_claim", "limitations", "not evidence of forecast skill"):
        assert flag in source, flag
