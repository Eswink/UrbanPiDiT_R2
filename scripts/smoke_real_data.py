from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.multiscale_dataset import ManifestNPZDataset
from data.schema import validate_sample
from model import UrbanPiDiTR2
from training.losses import R2Loss


def _batch_first_sample(config: dict) -> tuple[dict[str, torch.Tensor], str]:
    manifest = ROOT / config["data"]["train_manifest"]
    dataset = ManifestNPZDataset(manifest)
    sample = dataset[0]
    validate_sample(sample, batched=False)
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    validate_sample(batch, batched=True)
    return batch, str(sample["sample_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/r2_v6_real_smoke.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    batch, sample_id = _batch_first_sample(cfg)

    model = UrbanPiDiTR2(**cfg["model"])
    model.train()
    out = model(batch, force_zoom=True, adaptive_reasoning=False, hard_route=False)
    losses = R2Loss(**cfg["loss"])(batch, out)
    losses.total.backward()
    grad_ok = any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and bool((p.grad.abs().sum() > 0).item())
        for p in model.parameters()
    )
    assert torch.isfinite(out.forecast).all(), "forecast 含非有限值"
    assert torch.isfinite(losses.total), "loss 非有限"
    assert grad_ok, "未检测到有限非零梯度"

    # 独立验证 hard routing 的 STOP 与 ZOOM 两条真实 batch 路径。
    model.eval()
    original_threshold = model.zoom_threshold
    with torch.no_grad():
        model.zoom_threshold = 1.1
        stop_out = model(batch, force_zoom=None, adaptive_reasoning=False, hard_route=True)
        assert not bool(stop_out.diagnostics.route_mask.any().item())
        assert torch.allclose(stop_out.residual, torch.zeros_like(stop_out.residual))
        assert torch.allclose(stop_out.forecast, batch["urban_baseline"])

        model.zoom_threshold = -0.1
        zoom_out = model(batch, force_zoom=None, adaptive_reasoning=False, hard_route=True)
        assert bool(zoom_out.diagnostics.route_mask.all().item())
        assert torch.isfinite(zoom_out.forecast).all()

        model.zoom_threshold = original_threshold
        adaptive_out = model(batch, force_zoom=None, adaptive_reasoning=True, hard_route=True)
        assert torch.isfinite(adaptive_out.forecast).all()
        assert 1 <= int(adaptive_out.diagnostics.reasoning_steps) <= model.max_reasoning_steps

    print("REAL_DATA_SMOKE_PASS")
    print("sample_id:", sample_id)
    print("coarse_history:", tuple(batch["coarse_history"].shape))
    print("urban_history:", tuple(batch["urban_history"].shape))
    print("urban_static:", tuple(batch["urban_static"].shape))
    print("forecast:", tuple(out.forecast.shape))
    print("loss:", float(losses.total.detach()))
    print("zoom_probability:", out.diagnostics.zoom_probability.detach().cpu().tolist())
    print("verifier_score:", out.diagnostics.verifier_score.detach().cpu().tolist())
    print("gradient_ok:", grad_ok)
    print("hard_stop_path_ok:", True)
    print("hard_zoom_path_ok:", True)
    print("adaptive_route_mask:", adaptive_out.diagnostics.route_mask.detach().cpu().tolist())
    print("adaptive_reasoning_steps:", adaptive_out.diagnostics.reasoning_steps)


if __name__ == "__main__":
    main()
