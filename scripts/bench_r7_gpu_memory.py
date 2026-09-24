"""Bounded single-GPU K / training-mode / checkpointing memory-latency sweep.

This is an engineering bring-up instrument, not a weather-skill benchmark. It
measures PyTorch peak allocated/reserved bytes and wall-clock phase times for
the two R7 recursive models under full BPTT and streamed truncated training.

Inputs are synthetic shapes. Nothing here downloads data or claims forecast
skill; every result carries scientific_claim=false and explicit limitations.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import forecast_inputs
from training.r7_recursive_losses import deep_supervised_forecast_mse
from training.r7_process_forecast_losses import process_forecast_coreasoning_loss
from training.r7_streaming import backward_streamed_truncated
from training.r7_experiment import canonical_digest

MODEL_KINDS = ("generic", "process")
TRAINING_MODES = ("full_bptt", "streamed_truncated")


def build_batch(batch_size, channels, grid, history_steps, anchored, device, seed=7):
    """Synthetic shape fixture. Normalized fields, finite, deterministic."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    height, width = grid
    history = torch.randn(
        batch_size, history_steps, channels, height, width, generator=generator)
    target = torch.randn(batch_size, channels, height, width, generator=generator)
    batch = {"coarse_history": history.to(device), "atmos_target": target.to(device)}
    if anchored:
        process = torch.randn(batch_size, anchored, generator=generator)
        batch["process_targets"] = process.to(device)
    return batch


def build_model(kind, model_config, device):
    options = dict(model_config)
    if kind == "generic":
        model = GenericRecursiveWeatherForecaster(**options)
    else:
        model = ProcessForecastCoReasoner(**options)
    return model.to(device).train()


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def _autocast(device, bf16):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)


def full_bptt_loss(model, batch, steps):
    """The project's own canonical deep-supervision objective (no new variant)."""
    if isinstance(model, ProcessForecastCoReasoner):
        out = model(forecast_inputs(batch), reasoning_steps=steps)
        return process_forecast_coreasoning_loss(
            batch, out, forecast_weight=1.0, process_weight=0.1, final_weight=2.0).total
    out = model(forecast_inputs(batch), reasoning_steps=steps)
    return deep_supervised_forecast_mse(
        out.draft_forecasts, batch["atmos_target"], batch.get("latitude"), final_weight=2.0)


def measure_full_bptt(model, optimizer, batch, steps, device, bf16, clip=1.0):
    """Forward / backward / optimizer-step split with an explicit sync at each edge."""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with _autocast(device, bf16):
        loss = full_bptt_loss(model, batch, steps)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    mark = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - mark
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_step_seconds": optimizer_seconds,
    }


def measure_streamed(model, batch, steps, device, bf16, clip=1.0):
    """Streamed truncated path interleaves forward and backward by construction.

    A clean forward/backward split would be an artefact of instrumentation, so
    those two fields stay null and only the whole-step time is reported.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = backward_streamed_truncated(
        model, batch, reasoning_steps=steps, forecast_weight=1.0, process_weight=0.1,
        final_weight=2.0, amp_dtype=torch.bfloat16 if bf16 else None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(result.total.detach()),
        "forward_seconds": None,
        "backward_seconds": None,
        "optimizer_step_seconds": None,
        "step_seconds": seconds,
    }


def measure_phase(model, optimizer, batch, case, device):
    """Measure one optimizer step; caller owns peak-memory reset and sync."""
    started = time.perf_counter()
    if case["training_mode"] == "full_bptt":
        pieces = measure_full_bptt(model, optimizer, batch, case["k"], device, case["bf16"])
    else:
        pieces = measure_streamed(model, batch, case["k"], device, case["bf16"])
    pieces.setdefault("step_seconds", time.perf_counter() - started)
    return pieces


def run_case(case, device, warmup_steps, measure_steps):
    """Run one sweep cell; on failure keep the peak reached before the failure."""
    torch.manual_seed(case["seed"])
    model_config = dict(case["model_config"])
    model = build_model(case["kind"], model_config, device)
    batch = build_batch(case["batch_size"], model_config["in_channels"], case["grid"],
        model_config.get("history_steps", 2),
        model_config.get("anchored_processes") if case["kind"] == "process" else None,
        device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    row = {
        "kind": case["kind"], "k": case["k"], "training_mode": case["training_mode"],
        "activation_checkpointing": case["activation_checkpointing"],
        "batch_size": case["batch_size"], "grid": list(case["grid"]),
        "in_channels": model_config["in_channels"], "dim": model_config["dim"],
        "depth": model_config["depth"], "patch_size": model_config["patch_size"],
        "dtype": "bf16-autocast/fp32-master" if case["bf16"] else "fp32",
        "params": parameter_count(model),
        "synthetic_shape_only": True,
    }
    try:
        for _ in range(warmup_steps):
            measure_phase(model, optimizer, batch, case, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured = [measure_phase(model, optimizer, batch, case, device)
                    for _ in range(measure_steps)]
        torch.cuda.synchronize(device)
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        row["status"] = "ok"
        row["loss"] = measured[-1]["loss"]
        for key in ("step_seconds", "forward_seconds", "backward_seconds",
                    "optimizer_step_seconds"):
            values = [m[key] for m in measured if m.get(key) is not None]
            row[key] = sum(values) / len(values) if values else None
    except torch.cuda.OutOfMemoryError as error:
        row["status"] = "oom"
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        row["error"] = str(error).splitlines()[0][:200]
    except (RuntimeError, ValueError, TypeError) as error:
        row["status"] = "error"
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        row["error"] = f"{type(error).__name__}: {error}"[:200]
    finally:
        del model, optimizer, batch
    return row


def build_cases(args):
    """Cross product of model x K x mode x checkpointing."""
    steps = args.steps or [1, 2, 4, 8]
    base = {"in_channels": args.channels, "history_steps": 2, "out_channels": args.channels,
            "dim": args.dim, "patch_size": 2, "depth": args.depth, "heads": args.heads,
            "window_size": args.window, "dropout": 0.0, "default_reasoning_steps": 1}
    cases = []
    for kind in MODEL_KINDS:
        for k in steps:
            for mode in TRAINING_MODES:
                for checkpointing in (False, True):
                    model_config = dict(base, activation_checkpointing=checkpointing)
                    if kind == "process":
                        # Process State replaces the generic latent bank with
                        # anchored + free process tokens; latent_tokens is not
                        # a constructor argument of that model.
                        model_config["anchored_processes"] = args.anchored
                        model_config["free_processes"] = args.free
                    else:
                        model_config["latent_tokens"] = args.latent
                    cases.append({"kind": kind, "k": k, "training_mode": mode,
                        "activation_checkpointing": checkpointing,
                        "batch_size": args.batch, "grid": tuple(args.grid),
                        "model_config": model_config, "bf16": args.bf16,
                        "seed": args.seed})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grid", type=int, nargs=2, default=[12, 12])
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--anchored", type=int, default=4)
    parser.add_argument("--free", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no silent CPU fallback")

    folder = Path(args.out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cases = build_cases(args)
    device = torch.device(args.device)

    protocol = {
        "purpose": "single-GPU recursive memory/latency bring-up sweep",
        "frozen_before_any_step": True,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "dtype": "bf16-autocast/fp32-master" if args.bf16 else "fp32",
        "warmup_steps": args.warmup, "measure_steps": args.measure,
        "timing_rule": "torch.cuda.synchronize() before each phase boundary",
        "memory_rule": "torch.cuda.reset_peak_memory_stats() before the measured window",
        "model_matrix": list(MODEL_KINDS), "k_values": args.steps or [1, 2, 4, 8],
        "training_modes": list(TRAINING_MODES), "checkpointing": [False, True],
        "case_count": len(cases),
        "data": "synthetic shape fixture only; not multivariate real ERA5",
        "scientific_claim": False,
        "limitations": [
            "synthetic normalized fields; not multivariate real ERA5",
            "not a higher-resolution truth and not a forecast-skill measurement",
            "single synthetic shape fixture reused across cases",
        ],
    }
    (folder / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    protocol_digest = canonical_digest(protocol)

    started = time.perf_counter()
    rows, stopped_early = [], False
    for case in cases:
        if time.perf_counter() - started > args.max_seconds:
            stopped_early = True
            break
        row = run_case(case, device, args.warmup, args.measure)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in
            ("kind", "k", "training_mode", "activation_checkpointing", "status",
             "peak_allocated_bytes", "peak_reserved_bytes", "step_seconds", "error")},
            ensure_ascii=False), flush=True)

    report = {
        "format": "r7-gpu-single-device-sweep-v1",
        "protocol_digest": protocol_digest,
        "protocol": protocol,
        "scientific_claim": False,
        "stopped_early_on_deadline": stopped_early,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "limitations": protocol["limitations"] + [
            "peak memory covers the measured window of this bounded run only",
            "no multi-seed comparison and no convergence claim",
        ],
    }
    (folder / "sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    failures = [r for r in rows if r["status"] != "ok"]
    print(json.dumps({"protocol_digest": protocol_digest, "cases": len(rows),
                      "ok": len(rows) - len(failures), "failed": len(failures),
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
