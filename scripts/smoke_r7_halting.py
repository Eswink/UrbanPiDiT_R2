"""Tiny CPU engineering smoke. No real weather skill or speedup is certified."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster
from training.r7_halting import calibrate_controller_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--updates', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 2 or args.updates < 1:
        parser.error('steps >= 2 and updates >= 1 are required')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    core = ProcessForecastCoReasoner(in_channels=4, out_channels=4, dim=32,
        depth=1, heads=4, window_size=4, anchored_processes=2, free_processes=2)
    adapter = AdaptiveProcessForecaster(core)
    optimizer = torch.optim.AdamW(adapter.controller.parameters(), lr=1e-3)
    history = torch.randn(2, 2, 4, 8, 12)
    batch = {'coarse_history': history, 'atmos_target': history[:, -1] + 0.02,
             'latitude': torch.linspace(45., 43.25, 8)}
    before = {key: value.clone() for key, value in core.state_dict().items()}
    for _ in range(args.updates):
        loss = calibrate_controller_step(adapter, optimizer, batch, max_steps=args.steps)
    unchanged = all(torch.equal(value, before[key]) for key, value in core.state_dict().items())
    adapter.eval()
    result = adapter({'coarse_history': history}, max_steps=args.steps)
    if not unchanged or not torch.isfinite(result.forecast).all():
        raise RuntimeError('halting smoke failed')
    print(json.dumps({
        'source': 'synthetic-engineering-smoke', 'scientific_validation': False,
        'device': 'cpu', 'controller_updates': adapter.controller.optimizer_updates.item(),
        'controller_parameters': sum(p.numel() for p in adapter.controller.parameters()),
        'forecaster_unchanged': unchanged, 'calibration_loss': loss.total.item(),
        'max_reasoning_steps': args.steps,
        'reasoning_steps_per_sample': result.reasoning_steps_per_sample.tolist(),
    }, indent=2))


if __name__ == '__main__':
    main()
