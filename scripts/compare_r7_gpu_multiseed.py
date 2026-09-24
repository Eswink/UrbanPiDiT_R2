"""Multi-seed, paired comparison of the R7 training tricks on a real GPU.

Why this exists (#20): the existing bring-up sweep measures each training
configuration once, at one seed, so it cannot separate a trick's real effect
from run-to-run variation — and that variation is documented at up to ~40 % on
this machine. A single-seed delta of a few percent is therefore not evidence.

This module replicates each configuration across seeds and compares the trick
against its baseline **within the same seed**, so machine drift largely cancels
and the reported seed-level dispersion shows what the data can and cannot
support. Memory is measured one cell per process because `max_memory_reserved`
otherwise carries the allocator high-water mark between cells — the correction
recorded in docs/R7_GPU_BRINGUP.md.

Process isolation is the shell driver's job, not this module's: it never spawns
a subprocess. Three explicit phases make the order auditable — `--plan` freezes
the grid and protocol, one `--cell` invocation measures one cell, and
`--aggregate` reads the cells back and writes the summary.

Results carry `scientific_claim: false`. Three seeds is a noise check, not a
statistical test; the output says so rather than implying significance.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SEEDS = (41, 42, 43)
MODEL_KINDS = ("generic", "process")
TRAINING_MODES = ("full_bptt", "streamed_truncated")
DEFAULT_K = (1, 4, 8)
REAL_MANIFEST = ROOT / "outputs" / "r7_regional_real" / "manifests" / "val.jsonl"
METRICS = ("peak_allocated_bytes", "peak_reserved_bytes", "step_seconds")
# Fields that identify one measured cell. A paired comparison fixes every one of
# them except the factor under test.
CELL_KEY_FIELDS = ("kind", "k", "training_mode", "activation_checkpointing", "seed")


def build_cells(seeds, kinds, modes, checkout, steps):
    """Explicit (kind, K, mode, checkpointing, seed) grid, in a stable order."""
    return [
        {"kind": kind, "k": steps_k, "training_mode": mode,
         "activation_checkpointing": bool(ckpt), "seed": int(seed)}
        for kind, steps_k, mode, ckpt, seed in product(kinds, steps, modes, checkout, seeds)
    ]


def cell_tag(cell):
    return "{}_K{}_{}_ck{}_s{}".format(
        cell["kind"], cell["k"], cell["training_mode"],
        int(cell["activation_checkpointing"]), cell["seed"])


def _model_config(args, cell, channels, anchored=None):
    config = {
        "in_channels": channels, "history_steps": 2, "out_channels": channels,
        "dim": args.dim, "patch_size": 2, "depth": args.depth, "heads": args.heads,
        "window_size": args.window, "dropout": 0.0, "default_reasoning_steps": 1,
        "activation_checkpointing": cell["activation_checkpointing"],
    }
    if cell["kind"] == "process":
        # The process target width is a property of the dataset, not a free
        # parameter: the real store publishes eight diagnostics, so the model's
        # anchored width must adopt it rather than a synthetic default.
        config["anchored_processes"] = int(args.anchored if anchored is None else anchored)
        config["free_processes"] = args.free
        # This comparison isolates the training trick, not the process
        # semantics or forecast feedback, which have their own evaluation.
        config["use_forecast_feedback"] = False
    else:
        config["latent_tokens"] = args.latent
    return config


def _real_batch(dataset, count, device):
    """Stack real physical-unit windows through the production R7 reader."""
    import torch

    history, target, processes = [], [], []
    latitude = None
    for index in range(count):
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


def _synthetic_batch(cell, model_config, device, batch_size, grid, seed):
    """Synthetic shape fixture, built exactly as the bring-up sweep builds it."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    height, width = grid
    channels = model_config["in_channels"]
    history = torch.randn(batch_size, model_config.get("history_steps", 2), channels,
                          height, width, generator=generator)
    target = torch.randn(batch_size, channels, height, width, generator=generator)
    batch = {"coarse_history": history.to(device), "atmos_target": target.to(device)}
    if cell["kind"] == "process":
        process = torch.randn(batch_size, model_config["anchored_processes"],
                              generator=generator)
        batch["process_targets"] = process.to(device)
    return batch


def _step_loss(kind, model, batch, steps, bf16, device):
    """One forward loss; the caller owns backward/step so the timing is honest."""
    import torch

    from model.r7_halting import forecast_inputs
    from training.r7_process_forecast_losses import process_forecast_coreasoning_loss
    from training.r7_recursive_losses import deep_supervised_forecast_mse

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(bf16)):
        out = model(forecast_inputs(batch), reasoning_steps=steps)
        if kind == "process":
            return process_forecast_coreasoning_loss(
                batch, out, forecast_weight=1.0, process_weight=0.1, final_weight=2.0).total
        return deep_supervised_forecast_mse(
            out.draft_forecasts, batch["atmos_target"], batch.get("latitude"), final_weight=2.0)


def run_cell(cell, args):
    """Measure one cell in THIS process and return its row."""
    import torch

    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from training.r7_streaming import backward_streamed_truncated

    device = torch.device(args.device)
    torch.manual_seed(cell["seed"])
    anchored = None
    if args.data == "real":
        from data.r7_zarr_dataset import ZarrAtmosWindowDataset

        dataset = ZarrAtmosWindowDataset(REAL_MANIFEST)
        sample = dataset[0]
        channels = int(sample["coarse_history"].shape[1])
        if "process_targets" in sample:
            anchored = int(sample["process_targets"].shape[0])
        if args.batch > len(dataset):
            raise ValueError("real split has fewer windows than the requested batch")
    else:
        dataset = None
        channels = args.channels
    model_config = _model_config(args, cell, channels, anchored=anchored)
    builder = (GenericRecursiveWeatherForecaster if cell["kind"] == "generic"
               else ProcessForecastCoReasoner)
    model = builder(**model_config).to(device).train()
    row = dict(cell)
    row.update({
        "params": sum(p.numel() for p in model.parameters()),
        "channels": channels, "batch_size": args.batch, "data": args.data,
        "dim": args.dim, "depth": args.depth, "bf16": bool(args.bf16),
    })
    batch = (_real_batch(dataset, args.batch, device) if dataset is not None
             else _synthetic_batch(cell, model_config, device, args.batch,
                                   tuple(args.grid), cell["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    amp = torch.bfloat16 if args.bf16 else None

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if cell["training_mode"] == "full_bptt":
            loss = _step_loss(cell["kind"], model, batch, cell["k"], args.bf16, device)
            loss.backward()
        else:
            loss = backward_streamed_truncated(
                model, batch, reasoning_steps=cell["k"], forecast_weight=1.0,
                process_weight=0.1, final_weight=2.0, amp_dtype=amp).total
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), seconds

    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        losses, times = [], []
        for _ in range(args.measure):
            loss, seconds = one_step()
            losses.append(loss)
            times.append(seconds)
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        row["step_seconds"] = sum(times) / len(times)
        row["loss"] = losses[-1]
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        row["status"] = "oom"
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        row["error"] = str(error).splitlines()[0][:200]
    except (RuntimeError, ValueError, TypeError) as error:
        row["status"] = "error"
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        row["error"] = "{}: {}".format(type(error).__name__, error)[:200]
    finally:
        del model, optimizer, batch
    return row


def paired_deltas(rows, *, factor, baseline_value):
    """Per-seed deltas of one trick arm against its baseline.

    Pairs are formed on the full cell identity *except* the factor under test,
    so the other trick's state is held fixed. Omitting it would place both
    checkpointing states of one arm in the same group and compare the wrong
    things.

    Pairing inside one seed is what makes this a noise check: seed-to-seed
    variation widens the spread of the deltas instead of being mistaken for the
    trick's effect. A metric missing from either arm is reported as None, and an
    incomplete pairing raises rather than silently comparing unpaired numbers.

    `baseline_value` is compared by equality, not truthiness: `training_mode`
    is categorical and every one of its values is truthy.
    """
    key_fields = tuple(f for f in CELL_KEY_FIELDS if f != factor)
    arm, reference = {}, {}
    for row in rows:
        key = tuple(row[f] for f in key_fields)
        store = reference if row[factor] == baseline_value else arm
        if key in store:
            raise ValueError("duplicate row for group {} arm {}".format(key, factor))
        store[key] = row
    if set(arm) != set(reference):
        raise ValueError("incomplete pairs for {}: {}".format(
            factor, sorted(set(arm) ^ set(reference))))
    if not arm:
        raise ValueError("no rows matched factor {}".format(factor))
    result = {"factor": factor, "baseline": baseline_value, "groups": len(arm),
              "paired_on": list(key_fields), "metrics": {}}
    for metric in METRICS:
        pairs = [(arm[key].get(metric), reference[key].get(metric)) for key in sorted(arm)]
        if any(left is None or right is None for left, right in pairs):
            result["metrics"][metric] = None
            continue
        deltas = [float(left) - float(right) for left, right in pairs]
        result["metrics"][metric] = {
            "n": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "min_delta": min(deltas), "max_delta": max(deltas),
            "all_same_sign": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
            "arm_mean": statistics.fmean(
                float(arm[key][metric]) for key in sorted(arm)),
            "baseline_mean": statistics.fmean(
                float(reference[key][metric]) for key in sorted(reference)),
        }
    return result


def summarize(rows):
    """Seed-paired summary of both trick factors, per model kind and K."""
    summaries = []
    for factor, baseline_value, label in (
        ("training_mode", "full_bptt", "streamed_truncated vs full_bptt"),
        ("activation_checkpointing", False, "checkpointing on vs off"),
    ):
        for kind in sorted({row["kind"] for row in rows}):
            for steps_k in sorted({row["k"] for row in rows}):
                subset = [r for r in rows if r["kind"] == kind and r["k"] == steps_k]
                try:
                    block = paired_deltas(
                        subset, factor=factor, baseline_value=baseline_value)
                except ValueError:
                    continue
                # Identify the block in the published report: without these a
                # reader cannot tell which model/K a delta belongs to.
                block["comparison"] = label
                block["kind"] = kind
                block["k"] = steps_k
                summaries.append(block)
    return summaries


def _write_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _gpu_name(device):
    """Record the device name when one is actually queryable.

    `torch.cuda.get_device_name` raises when no GPU is visible, which is the
    normal state in CI. The plan must still be freezeable there (the tests do
    exactly that), so an unavailable device is recorded as null rather than
    turning protocol freezing into a CUDA requirement.
    """
    import torch

    if "cuda" not in device or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(torch.device(device))
    except (RuntimeError, AssertionError):
        return None


def protocol_payload(args, cells):
    """The frozen protocol. Written before any cell is measured."""
    import torch

    return {
        "format": "r7-gpu-multiseed-trick-comparison-v1",
        "frozen_before_any_step": True,
        "purpose": (
            "replicate each training trick across seeds and compare it against its "
            "baseline within the same seed, so a delta smaller than seed-to-seed "
            "dispersion is visible rather than reported as an effect"
        ),
        "seeds": list(args.seeds), "model_matrix": list(args.kinds),
        "training_modes": list(args.modes), "k_values": list(args.steps),
        "checkpointing": [bool(c) for c in args.ckpt],
        "data": args.data,
        "real_manifest": (str(REAL_MANIFEST.relative_to(ROOT))
                          if args.data == "real" else None),
        "batch_size": args.batch, "dim": args.dim, "depth": args.depth,
        "grid": None if args.data == "real" else list(args.grid),
        "dtype": "bf16-autocast/fp32-master" if args.bf16 else "fp32",
        "warmup_steps": args.warmup, "measure_steps": args.measure,
        "timing_rule": "torch.cuda.synchronize() before each phase boundary",
        "memory_rule": "reset_peak_memory_stats() before the measured window",
        "reserved_isolation": "one process per cell via the shell driver",
        "case_count": len(cells), "device": args.device,
        "gpu_name": _gpu_name(args.device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "scientific_claim": False,
        "limitations": [
            "three seeds only: a dispersion check, not a significance test",
            "fixed short measured window; no convergence or forecast-skill claim",
            "absolute step times vary between runs on this machine",
        ],
    }


def write_plan(args):
    """Freeze the protocol and the cell grid into a run directory.

    The directory must either not exist or be empty: outputs themselves are
    created exclusively by `_write_exclusive`, so an existing plan is never
    silently overwritten while an empty pre-created directory is accepted.
    """
    from training.r7_experiment import canonical_digest

    folder = Path(args.out).resolve()
    if folder.exists():
        if not folder.is_dir() or any(folder.iterdir()):
            raise FileExistsError(f"run directory is not empty: {folder}")
    else:
        folder.mkdir(parents=True)
    for name in ("protocol.json", "cells.jsonl"):
        if (folder / name).exists():
            raise FileExistsError(f"refusing to overwrite existing {name}")
    cells = build_cells(args.seeds, args.kinds, args.modes, args.ckpt, args.steps)
    protocol = protocol_payload(args, cells)
    _write_exclusive(folder / "protocol.json", protocol)
    digest = canonical_digest(protocol)
    lines = [json.dumps({"protocol_digest": digest})]
    lines += [json.dumps(cell) for cell in cells]
    with (folder / "cells.jsonl").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"protocol_digest": digest, "cells": len(cells)}, ensure_ascii=False))
    return cells


def measure_cell(args):
    cell = json.loads(args.cell)
    folder = Path(args.cell_out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row = run_cell(cell, args)
    row["elapsed_seconds"] = time.perf_counter() - started
    _write_exclusive(folder / "cell.json", row)
    print(json.dumps({key: row.get(key) for key in (
        "kind", "k", "training_mode", "activation_checkpointing", "seed", "status",
        "peak_allocated_bytes", "peak_reserved_bytes", "step_seconds")}, ensure_ascii=False))


def aggregate(args):
    from training.r7_experiment import canonical_digest

    folder = Path(args.out).resolve()
    with (folder / "protocol.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    planned = [json.loads(line) for line in
               (folder / "cells.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not planned or "protocol_digest" not in planned[0]:
        raise ValueError("cells.jsonl must start with its protocol digest header")
    expected = planned[1:]
    rows = []
    missing = []
    for cell in expected:
        path = folder / "cells" / cell_tag(cell) / "cell.json"
        if not path.is_file():
            missing.append(cell_tag(cell))
            continue
        with path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    failed = [cell_tag(r) for r in rows if r.get("status") != "ok"]
    complete = not missing and not failed
    report = {
        "format": "r7-gpu-multiseed-trick-comparison-v1",
        "protocol_digest": canonical_digest(protocol),
        "protocol": protocol,
        "scientific_claim": False,
        "complete": complete,
        "measured_cells": len(rows), "planned_cells": len(expected),
        "missing_cells": missing, "failed_cells": failed,
        "elapsed_seconds": sum(float(r.get("elapsed_seconds") or 0.0) for r in rows),
        "rows": rows,
        "summary": summarize(rows) if complete else [],
        "limitations": protocol["limitations"] + [
            "summary is omitted unless every planned cell completed without failure",
            "a delta smaller than sd_delta is not established by this run",
        ],
    }
    _write_exclusive(folder / "multiseed.json", report)
    print(json.dumps({"protocol_digest": report["protocol_digest"],
                      "measured": len(rows), "planned": len(expected),
                      "complete": complete, "comparisons": len(report["summary"]),
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed paired comparison of R7 training tricks on one GPU.")
    parser.add_argument("--plan", action="store_true",
                        help="freeze the protocol and cell grid, then exit")
    parser.add_argument("--aggregate", action="store_true",
                        help="read measured cells under --out and write the summary")
    parser.add_argument("--out", help="run directory (plan/aggregate)")
    parser.add_argument("--cell", help="JSON of one cell to measure in this process")
    parser.add_argument("--cell-out", help="fresh directory for that one cell")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", choices=("real", "synthetic"), default="real",
                        help="real uses the published regional ERA5 R7 store")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--kinds", nargs="+", choices=list(MODEL_KINDS), default=list(MODEL_KINDS))
    parser.add_argument("--modes", nargs="+", choices=list(TRAINING_MODES),
                        default=list(TRAINING_MODES))
    parser.add_argument("--ckpt", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grid", type=int, nargs=2, default=[12, 12])
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--anchored", type=int, default=4)
    parser.add_argument("--free", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not _cuda_available():
        parser.error("CUDA requested but unavailable; no silent CPU fallback")

    if args.cell:
        if not args.cell_out:
            parser.error("--cell requires --cell-out")
        measure_cell(args)
        return
    if not args.out:
        parser.error("--out is required for --plan/--aggregate")
    if args.plan == args.aggregate:
        parser.error("choose exactly one of --plan or --aggregate")
    write_plan(args) if args.plan else aggregate(args)


def _cuda_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
