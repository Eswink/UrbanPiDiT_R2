"""Measure what RW-B costs, at the configuration the experiment will use.

``docs/R7_MAIN_MODEL_V2_DESIGN.md`` section 7 sets an engineering target before
anything is frozen: the parameter increment must be measured, and it must stay
under a quarter of the main model - "unmeasured light-weight" is explicitly
forbidden, and so is claiming constant VRAM without measuring it. This script is
the measurement, on CPU, with no training and no GPU hours.

The FLOP convention is the one the previous rounds declared and used, not a new
one: ``FlopCounterMode`` over a single forward pass under ``torch.enable_grad()``,
with backward measured separately rather than assumed to be twice the forward. The
attention matmuls are counted because there is no parameter hook anywhere in the
path (SDPA is parameterless, so a hook would silently undercount it).

The input shape is the audited one - 17 channels on the 65x65 native grid, two
history steps, ``dim=192`` - because a ratio measured on a toy grid says nothing
about the model that would be trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# The script is reachable both as ``scripts/measure_r7_rw_b_cost.py`` and through a
# bare filename from the repository root, so the package root is put on the path
# explicitly rather than relying on the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDITED = {"in_channels": 17, "out_channels": 17, "history_steps": 2, "dim": 192,
           "depth": 4, "heads": 4, "window_size": 4, "patch_size": 2, "dropout": 0.0,
           "anchored_processes": 8, "free_processes": 8, "default_reasoning_steps": 3,
           "use_forecast_feedback": True}
BASE = dict(AUDITED, spacetime_inputs=True, positional_process_readout=True)
ARMS = (
    ("rw_a", {}),
    ("rw_a_roles", {"source_role_markers": True}),
    ("rw_a_rw_b", {"local_solver_state": True}),
    ("rw_a_roles_rw_b", {"source_role_markers": True, "local_solver_state": True}),
)
FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks. Backward is measured "
    "separately, never assumed to be 2x forward."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audited_batch(rows: int, columns: int, size: int) -> dict:
    import torch
    generator = torch.Generator().manual_seed(20240928)
    return {
        "coarse_history": torch.randn(size, 2, 17, rows, columns, generator=generator),
        "lead_time_hours": torch.full((size,), 6.0),
        "latitude": torch.linspace(-30.0, 30.0, rows),
        "longitude": torch.linspace(0.0, 25.0, columns),
        "init_utc_hour": torch.full((size,), 3.0),
        "init_day_of_year": torch.full((size,), 40.0),
    }


def measure(model, batch, *, reasoning_steps: int) -> dict:
    """Forward and forward+backward FLOPs under the declared convention."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            model(batch, reasoning_steps=reasoning_steps)
        forward = int(counter.get_total_flops())
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            output = model(batch, reasoning_steps=reasoning_steps)
            output.forecast.square().mean().backward()
        both = int(counter.get_total_flops())
    model.zero_grad()
    return {"forward_flops": forward, "forward_backward_flops": both,
            "backward_flops": both - forward}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=65)
    parser.add_argument("--columns", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reasoning-steps", type=int, default=3)
    args = parser.parse_args()

    import torch
    from training.r7_experiment import model_code_digest
    from model.process_forecast_r7 import ProcessForecastCoReasoner

    torch.set_num_threads(8)
    torch.manual_seed(17)
    batch = audited_batch(args.rows, args.columns, args.batch_size)
    started = time.monotonic()
    records = []
    for name, switches in ARMS:
        torch.manual_seed(17)
        model = ProcessForecastCoReasoner(**BASE, **switches).eval()
        parameters = sum(p.numel() for p in model.parameters())
        rows = measure(model, batch, reasoning_steps=args.reasoning_steps)
        records.append({"arm": name, "switches": switches, "parameters": parameters,
                        **rows})
    baseline = records[0]
    for record in records:
        record["parameter_increment_vs_rw_a"] = record["parameters"] - baseline["parameters"]
        record["parameter_ratio_vs_rw_a"] = record["parameters"] / baseline["parameters"]
        record["forward_flops_ratio_vs_rw_a"] = (
            record["forward_flops"] / baseline["forward_flops"])
        record["forward_backward_flops_ratio_vs_rw_a"] = (
            record["forward_backward_flops"] / baseline["forward_backward_flops"])

    rw_b = next(r for r in records if r["arm"] == "rw_a_rw_b")
    result = {
        "format": "r7-rw-b-cost-measurement-v1",
        "scientific_claim": False,
        "training_executed": False,
        "gpu_used": False,
        "device": "cpu",
        "model_code_digest": model_code_digest(),
        "driver_sha256": sha256_file(Path(__file__)),
        "grid": [args.rows, args.columns],
        "batch_size": args.batch_size,
        "reasoning_steps": args.reasoning_steps,
        "flop_convention": FLOP_CONVENTION,
        "arms": records,
        "parameter_target_ratio": 0.25,
        "parameter_target_met": (
            rw_b["parameter_increment_vs_rw_a"] <= 0.25 * baseline["parameters"]),
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "CPU FLOPs are a count of the declared ops, not a wall-clock or GPU claim",
            "the working-memory cost of holding Z across steps is not measured here; "
            "no constant-VRAM statement is made anywhere in this round",
            "measured at the audited architecture, not at every configuration",
        ],
    }
    target = Path(args.output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({
        "complete": True,
        "parameter_target_met": result["parameter_target_met"],
        "arms": {r["arm"]: {"parameters": r["parameters"],
                            "delta": r["parameter_increment_vs_rw_a"],
                            "forward_ratio": round(r["forward_flops_ratio_vs_rw_a"], 4)}
                 for r in records},
        "elapsed_seconds": round(result["elapsed_seconds"], 1),
        "output": str(target),
    }))


if __name__ == "__main__":
    main()
