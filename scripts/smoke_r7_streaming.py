"""Small synthetic update smoke. Does not download data or claim weather skill."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.process_forecast_r7 import ProcessForecastCoReasoner
from training.r7_streaming import train_streamed_update
from training.r7_memory import SavedTensorMeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    if args.steps < 0 or args.updates < 1:
        ap.error("steps must be nonnegative and updates positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = ProcessForecastCoReasoner(in_channels=4, dim=32, depth=1, heads=4,
        window_size=4, anchored_processes=2, free_processes=2).to(args.device).train()
    batch = {"coarse_history": torch.randn(2, 2, 4, 8, 12, device=args.device),
             "atmos_target": torch.randn(2, 4, 8, 12, device=args.device),
             "process_targets": torch.randn(2, 2, device=args.device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    results = []
    for _ in range(args.updates):
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with SavedTensorMeter() as meter:
            log = train_streamed_update(model, optimizer, [batch], reasoning_steps=args.steps,
                amp_dtype=torch.bfloat16 if args.bf16 else None)[0]
        row = {"loss": float(log.total), "logical_saved_tensor_peak_bytes": meter.peak_live_bytes}
        if args.device == "cuda":
            torch.cuda.synchronize()
            row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        results.append(row)
    print(json.dumps({"synthetic_only": True, "device": args.device, "steps": args.steps,
                      "optimizer_updates": args.updates, "results": results}))


if __name__ == "__main__":
    main()
