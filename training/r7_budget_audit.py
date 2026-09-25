"""Parameter and FLOP budget audit for the R7 recursive baselines.

Issue #5's acceptance criterion is that the generic recursive baseline "must
share comparable parameter/FLOP budget with process-aware R7.3". Parameter counts
were already easy to produce, but **no FLOP accounting existed anywhere in the
repository**, so the FLOP half of that criterion had never been checked. This
module measures both and reports the parity verdict explicitly, against real
published ERA5 windows rather than synthetic shapes.

Counting convention (stated because a FLOP number without its convention is not
comparable to anything):

- `torch.utils.flop_counter.FlopCounterMode` over a forward pass, which counts
  linear/conv/matmul and the attention matmuls dispatched through the aten ops it
  knows. Elementwise and normalization ops are not counted.
- Backward is not counted. It is roughly 2x forward for these graphs, but that
  ratio is architecture-dependent, so only forward is reported and compared.
- Counting requires gradients enabled: `FlopCounterMode` registers hooks that
  fail under `torch.no_grad()` on these models.

Results carry `scientific_claim: false`. A budget match says nothing about skill.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("generic", "process")
DEFAULT_K = (1, 2, 4, 6, 8)
REAL_MANIFEST = ROOT / "outputs" / "r7_regional_real" / "manifests" / "val.jsonl"
# A budget is "comparable" if the process-aware model stays within this fraction
# of the generic baseline. Declared here, before measurement, so it cannot be
# relaxed to fit the numbers afterwards.
PARITY_TOLERANCE = 0.05


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_forward_flops(model, batch, reasoning_steps: int) -> int:
    """Forward FLOPs for one prediction at this reasoning depth.

    Gradients must stay enabled for the counter's hooks; the caller therefore
    must not wrap this in `torch.no_grad()`.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from model.r7_halting import forecast_inputs

    with FlopCounterMode(display=False) as counter:
        model(forecast_inputs(batch), reasoning_steps=reasoning_steps)
    return int(counter.get_total_flops())


def build_batch(device, *, batch_size: int, manifest: Path = REAL_MANIFEST):
    """Real physical-unit windows from the published R7 store."""
    import torch

    from data.r7_zarr_dataset import ZarrAtmosWindowDataset

    dataset = ZarrAtmosWindowDataset(manifest)
    if batch_size > len(dataset):
        raise ValueError(
            f"manifest has {len(dataset)} windows, fewer than the requested {batch_size}")
    history, target, processes = [], [], []
    latitude = None
    for index in range(batch_size):
        sample = dataset[index]
        history.append(sample["coarse_history"])
        target.append(sample["atmos_target"])
        latitude = sample["latitude"]
        if "process_targets" in sample:
            processes.append(sample["process_targets"])
    batch = {
        "coarse_history": torch.stack(history).to(device),
        "atmos_target": torch.stack(target).to(device),
        "latitude": latitude.to(device),
    }
    if processes:
        batch["process_targets"] = torch.stack(processes).to(device)
    return batch


def model_config_for(variant: str, channels: int, anchored: int, *, dim=128, depth=4):
    """The established comparison settings, widened to the data's channel count."""
    config = {
        "in_channels": channels, "out_channels": channels, "history_steps": 2,
        "dim": dim, "depth": depth, "heads": 4, "window_size": 8,
        "patch_size": 2, "dropout": 0.0, "default_reasoning_steps": 1,
    }
    if variant == "process":
        config.update(anchored_processes=anchored, free_processes=8)
    else:
        config["latent_tokens"] = 16
    return config


def audit(args):
    """Measure both variants across K and return the full report."""
    import torch

    from training.r7_experiment import canonical_digest, make_model

    device = torch.device(args.device)
    batch = build_batch(device, batch_size=args.batch)
    channels = int(batch["coarse_history"].shape[2])
    anchored = int(batch["process_targets"].shape[1]) if "process_targets" in batch else 8
    real_manifest = args.manifest

    rows = []
    for variant in VARIANTS:
        config = model_config_for(variant, channels, anchored, dim=args.dim, depth=args.depth)
        # Deterministic init so a re-run compares the same parameter count.
        torch.manual_seed(args.seed)
        model = make_model(variant, config).to(device).eval()
        params = count_parameters(model)
        row = {"variant": variant, "params": params, "model_config": config, "flops": {}}
        for steps in args.steps:
            # Counting needs autograd enabled; see the module docstring.
            with torch.enable_grad():
                row["flops"][str(steps)] = count_forward_flops(model, batch, steps)
        del model
        rows.append(row)

    by_variant = {row["variant"]: row for row in rows}
    generic, process = by_variant["generic"], by_variant["process"]
    parity = {
        "params": _parity(generic["params"], process["params"], PARITY_TOLERANCE),
        "flops": {
            str(steps): _parity(generic["flops"][str(steps)], process["flops"][str(steps)],
                                PARITY_TOLERANCE)
            for steps in args.steps
        },
    }
    all_ok = parity["params"]["within_tolerance"] and all(
        entry["within_tolerance"] for entry in parity["flops"].values())
    report = {
        "format": "r7-recursive-budget-audit-v1",
        "scientific_claim": False,
        "complete": True,
        "data": str(real_manifest.relative_to(ROOT)) if real_manifest.is_relative_to(ROOT)
                else str(real_manifest),
        "data_is_real": True,
        "batch_size": args.batch,
        "channels": channels,
        "anchored_processes": anchored,
        "k_values": list(args.steps),
        "device": args.device,
        "seed": args.seed,
        "flop_convention": (
            "FlopCounterMode over one forward pass; elementwise/normalization not "
            "counted; backward not counted; gradients must be enabled"
        ),
        "parity_tolerance": PARITY_TOLERANCE,
        "rows": rows,
        "parity": parity,
        "acceptance_met": bool(all_ok),
        "limitations": [
            "forward-only FLOPs; backward is not counted and is not assumed equal",
            "one window batch from the validation split; not a training-budget measurement",
            "a budget match is not evidence of forecast skill",
            "parameter counts are architecture-only and exclude optimizer state",
        ],
    }
    report["report_digest"] = canonical_digest(
        {k: v for k, v in report.items() if k != "report_digest"})
    return report


def _parity(baseline: int, candidate: int, tolerance: float) -> dict:
    if baseline <= 0:
        raise ValueError("baseline budget must be positive")
    relative = (candidate - baseline) / baseline
    return {
        "baseline": int(baseline), "candidate": int(candidate),
        "relative_difference": relative,
        "within_tolerance": abs(relative) <= tolerance,
        "tolerance": tolerance,
    }


def write_report(path, report):
    """Exclusive create; the parent directory is created but never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description="Audit parameter/FLOP parity between the generic and process R7 models.")
    parser.add_argument("--out", required=True, help="new JSON report path")
    parser.add_argument("--manifest", default=str(REAL_MANIFEST))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.manifest = Path(args.manifest)
    report = audit(args)
    write_report(args.out, report)
    print(json.dumps({
        "params": {row["variant"]: row["params"] for row in report["rows"]},
        "params_relative_difference": report["parity"]["params"]["relative_difference"],
        "flops_relative_difference": {
            k: round(v["relative_difference"], 6)
            for k, v in report["parity"]["flops"].items()},
        "acceptance_met": report["acceptance_met"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
