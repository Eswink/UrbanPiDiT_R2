"""Dual-GPU DDP bring-up smoke: participation, loss identity, sampler dedup, resume.

Run under `torchrun --nproc_per_node=2`. The DDP path is verified against a
single-GPU reference built from the identical synthetic samples, so a passing
run is evidence about distributed correctness only - not forecast skill, and not
a single-model memory improvement (the two cards hold two model replicas).

Data is the deterministic synthetic shape fixture; nothing here downloads or
claims weather truth. Every artifact carries scientific_claim=false.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, default_collate

from data.synthetic_atmos import SyntheticAtmosDataset
from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
from model.r7_halting import forecast_inputs
from training.r7_recursive_losses import deep_supervised_forecast_mse
from training.r7_experiment import canonical_digest, model_code_digest, save_exclusive

MODEL_CONFIG = {"in_channels": 11, "history_steps": 2, "out_channels": 11, "dim": 128,
    "patch_size": 2, "depth": 4, "heads": 4, "window_size": 4, "dropout": 0.0,
    "activation_checkpointing": False, "latent_tokens": 16,
    "default_reasoning_steps": 4, "detach_between_steps": False}
DATASET_CONFIG = {"length": 32, "hw": (12, 12), "history_steps": 2, "channels": 11,
    "lead_time_hours": 6.0, "grid_spacing_deg": 0.25}
VAL_LENGTH = 12


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(seed, device):
    seed_everything(seed)
    model = GenericRecursiveWeatherForecaster(**MODEL_CONFIG).to(device)
    return model.train()


def batch_loss(model, batch, steps, bf16):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(forecast_inputs(batch), reasoning_steps=steps)
    return deep_supervised_forecast_mse(
        out.draft_forecasts, batch["atmos_target"], batch.get("latitude"), final_weight=2.0)


def collate_indices(dataset, indices, device):
    batch = default_collate([dataset[i] for i in indices])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def local_indices(rank, world_size, length, shuffle, seed, epoch=0):
    """The exact index list this rank will see, from the real DistributedSampler."""
    sampler = DistributedSampler(SyntheticAtmosDataset(**dict(DATASET_CONFIG, length=length)),
        num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
    sampler.set_epoch(epoch)
    return list(sampler)


def check_sampler_partition(world_size, tag, shuffle=True, length=None):
    """The union of per-rank indices must be the dataset exactly once."""
    size = DATASET_CONFIG["length"] if length is None else length
    pooled = []
    for rank in range(world_size):
        pooled.extend(local_indices(rank, world_size, size, shuffle, 42))
    expected = list(range(size))
    return {"tag": tag, "count": len(pooled), "dataset_length": size,
        "no_duplicates": len(pooled) == len(set(pooled)),
        "covers_dataset": sorted(pooled) == expected,
        "per_rank_counts": [len(local_indices(r, world_size, size, shuffle, 42))
                            for r in range(world_size)]}


def run_ddp_steps(model, optimizer, dataset, config, device, rank, steps, bf16,
                  accumulation, start_update=0):
    """Train optimizer updates until `steps`; resumes mid-schedule when asked."""
    world_size = dist.get_world_size()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=config.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=config.per_gpu_batch, sampler=sampler,
        collate_fn=default_collate)
    losses, updates, epoch = [], start_update, 0
    # A resumed run must continue the same schedule, so the samples the saved run
    # already consumed are skipped by position, not re-drawn.
    skip_groups = start_update
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while updates < steps:
        sampler.set_epoch(epoch)
        pending = []
        for raw in loader:
            pending.append(raw)
            if len(pending) < accumulation:
                continue
            if skip_groups > 0:
                skip_groups -= 1
                pending = []
                continue
            value, consumed = _accumulate_step(model, optimizer, pending, device,
                config, bf16, len(pending) == accumulation)
            losses.append({"update": updates + 1, "loss": value, "epoch": epoch,
                           "indices": consumed})
            pending = []
            updates += 1
            if updates >= steps:
                break
        if updates >= steps or not pending:
            epoch += 1
            continue
        if skip_groups > 0:
            skip_groups -= 1
            epoch += 1
            continue
        value, consumed = _accumulate_step(model, optimizer, pending, device, config,
            bf16, True)
        losses.append({"update": updates + 1, "loss": value, "epoch": epoch,
                       "indices": consumed, "partial_accumulation": len(pending)})
        updates += 1
        epoch += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return losses, time.perf_counter() - started


def _accumulate_step(model, optimizer, batches, device, config, bf16, sync):
    """One optimizer update over the pending microbatches, sample-weighted.

    Returns the sample-weighted loss and the dataset indices consumed, so the
    single-GPU reference can replay the identical global batch.
    """
    context = _null() if (sync or dist.get_world_size() == 1) else model.no_sync()
    with context:
        optimizer.zero_grad(set_to_none=True)
        count = sum(b["coarse_history"].shape[0] for b in batches)
        total, consumed = 0.0, []
        for raw in batches:
            consumed.extend(sample_index(raw))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in raw.items()}
            loss = batch_loss(model, batch, config.reasoning_steps, bf16)
            (loss * (batch["coarse_history"].shape[0] / count)).backward()
            total += float(loss.detach()) * (batch["coarse_history"].shape[0] / count)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return total, consumed


def _null():
    """No-op stand-in for the DDP no_sync() context on the synchronizing step."""
    return contextlib.nullcontext()


def ddp_main(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    out = Path(args.out).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticAtmosDataset(**DATASET_CONFIG)
    val_dataset = SyntheticAtmosDataset(**dict(DATASET_CONFIG, length=VAL_LENGTH, seed=999))
    sampler_audit = check_sampler_partition(world_size, "train", shuffle=True)
    val_audit = check_sampler_partition(world_size, "val", shuffle=False, length=VAL_LENGTH)

    model = build_model(args.seed, device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4, weight_decay=1e-4)
    resume_info = None
    start_update = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("format") != "r7-ddp-smoke-v1":
            raise SystemExit("resume requires an r7-ddp-smoke-v1 checkpoint")
        if saved["world_size"] != world_size:
            raise SystemExit("checkpoint world_size differs from this run")
        if saved["model_code_sha256"] != model_code_digest():
            raise SystemExit("checkpoint model implementation differs; replay with its recorded code")
        # Saved from the unwrapped module, so it is loaded there too: the DDP
        # wrapper would otherwise expect a `module.` prefix on every key.
        model.load_state_dict({k: v.to(device) for k, v in saved["weights"].items()},
                              strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_update = int(saved["updates"])
        if start_update >= args.steps:
            raise SystemExit("resume endpoint must exceed the saved updates")
        resume_info = {"resumed_from": str(Path(args.resume).name),
                       "start_update": start_update, "target_update": args.steps}
        dist.barrier()
    losses, seconds = run_ddp_steps(ddp_model, optimizer, dataset, args, device, rank,
        args.steps, args.bf16, args.accumulation, start_update=start_update)

    # Validation pass: every val sample exactly once across ranks, statistics pooled.
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.per_gpu_batch, sampler=val_sampler,
        collate_fn=default_collate)
    seen, val_losses = [], []
    ddp_model.eval()
    with torch.no_grad():
        for raw in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in raw.items()}
            val_losses.append(float(batch_loss(ddp_model, batch, args.steps, args.bf16).detach()))
            seen.extend(list(batch["sample_id"]))
    gathered = [None] * world_size
    dist.all_gather_object(gathered, seen)

    payload = {"rank": rank, "world_size": world_size, "local_rank": local_rank,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "seen_val_ids": seen, "losses": losses,
        "seconds": seconds, "val_losses": val_losses,
        "resume_info": resume_info,
        "params": sum(p.numel() for p in model.parameters()),
        "sampler_audit": sampler_audit, "val_sampler_audit": val_audit,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

    if rank == 0:
        flat_val = [i for chunk in (gathered or [[seen]]) for i in chunk]
        payload["pooled_val_ids"] = flat_val
        payload["val_ids_unique"] = len(flat_val) == len(set(flat_val))
        payload["val_covers_dataset"] = sorted(flat_val) == sorted(
            [f"synthetic_atmos_{i:05d}" for i in range(VAL_LENGTH)])
        weights = to_cpu(model.state_dict())
        optimizer_state = to_cpu(optimizer.state_dict())
        checkpoint = {"format": "r7-ddp-smoke-v1", "model_code_sha256": model_code_digest(),
            "signature": canonical_digest({"model": MODEL_CONFIG, "seed": args.seed,
                "steps": args.steps, "per_gpu_batch": args.per_gpu_batch,
                "accumulation": args.accumulation, "world_size": world_size}),
            "updates": args.steps, "weights": weights, "optimizer": optimizer_state,
            "world_size": world_size, "per_gpu_batch": args.per_gpu_batch,
            "accumulation": args.accumulation, "seed": args.seed}
        save_exclusive(out / f"ddp_update_{args.steps:07d}.pt", checkpoint)
        payload["checkpoint_written"] = True
    else:
        payload["checkpoint_written"] = False
    write_json(out / f"ddp_rank{rank}.json", payload)
    dist.barrier()
    dist.destroy_process_group()


def to_cpu(value):
    """Recursively detach a state dict so it can be torch.save'd portably."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def sample_index(batch):
    """Recover integer indices from the dataset's own sample_id field."""
    ids = batch["sample_id"]
    if isinstance(ids, str):
        ids = [ids]
    return [int(value.rsplit("_", 1)[1]) for value in ids]


def reference_main(args):
    """Single-GPU reference on the exact global batch the DDP run consumed.

    Reads the index list each rank recorded, unions them per update, and replays
    that global batch on one device. Identity is per-sample, so this verifies the
    distributed run used the same data as a single-GPU run - not merely that the
    two loss curves happen to be close.
    """
    device = torch.device("cuda:0")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rank_files = [out / f"ddp_rank{r}.json" for r in range(args.world_size)]
    missing = [str(p) for p in rank_files if not p.exists()]
    if missing:
        raise SystemExit(f"reference replay needs the DDP rank records first: {missing}")
    ranks = [json.loads(p.read_text(encoding="utf-8")) for p in rank_files]
    dataset = SyntheticAtmosDataset(**DATASET_CONFIG)
    model = build_model(args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    losses = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for update in range(len(ranks[0]["losses"])):
        chunk = []
        for entries in ranks:
            chunk.extend(entries["losses"][update]["indices"])
        batch = collate_indices(dataset, chunk, device)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, args.reasoning_steps, args.bf16)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"update": update + 1, "loss": float(loss.detach()),
                       "indices": chunk, "global_batch_size": len(chunk)})
    torch.cuda.synchronize(device)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"weights": weights}, out / "reference_weights.pt")
    write_json(out / "reference.json", {"device": str(device), "losses": losses,
        "seconds": time.perf_counter() - started,
        "replayed_from": [str(p.name) for p in rank_files],
        "weights_hash": {k: hashlib_of(v) for k, v in model.state_dict().items()},
        "params": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "global_batch": args.global_batch})


def hashlib_of(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def compare_main(args):
    out = Path(args.out).resolve()
    checkpoint = torch.load(out / f"ddp_update_{args.steps:07d}.pt",
        map_location="cpu", weights_only=True)
    reference = json.loads((out / "reference.json").read_text(encoding="utf-8"))
    ref_weights = torch.load(out / "reference_weights.pt", map_location="cpu",
                             weights_only=True)["weights"]
    ranks = [json.loads((out / f"ddp_rank{r}.json").read_text(encoding="utf-8"))
             for r in range(args.world_size)]
    ddp_losses = [r["losses"] for r in ranks]
    # DDP's per-rank loss is the rank's own sample mean; the global-batch loss is
    # their average because every rank holds the same number of samples.
    mean_loss = [sum(r["losses"][i]["loss"] for r in ranks) / len(ranks)
                 for i in range(len(ranks[0]["losses"]))]
    ref_loss = [entry["loss"] for entry in reference["losses"]]
    deltas = [abs(a - b) for a, b in zip(mean_loss, ref_loss)]
    weight_diff = {k: float((checkpoint["weights"][k].float()
        - ref_weights[k].float()).abs().max()) for k in list(checkpoint["weights"])}
    reported_global = args.per_gpu_batch * args.world_size * args.accumulation
    batch_sizes = [entry["global_batch_size"] for entry in reference["losses"]]
    every_strip_complete = all(size == reported_global for size in batch_sizes)
    report = {
        "ddp_per_rank_losses": ddp_losses,
        "ddp_mean_loss": mean_loss,
        "single_gpu_global_batch_loss": ref_loss,
        "loss_abs_delta": deltas, "loss_max_abs_delta": max(deltas),
        "loss_within_tolerance": max(deltas) < 1e-3,
        "weight_max_abs_delta": max(weight_diff.values()),
        "weight_keys_compared": len(weight_diff),
        "global_batch": reported_global,
        "global_batch_reported_by_ddp": args.per_gpu_batch * args.world_size * args.accumulation,
        "global_batch_formula":
            f"{args.per_gpu_batch} per_gpu x {args.world_size} world_size x "
            f"{args.accumulation} accumulation = {reported_global}",
        "reference_strip_count": len(batch_sizes),
        "global_batch_sizes": batch_sizes,
        "every_strip_complete": every_strip_complete,
        "checkpoint_file_count": len(list(out.glob("ddp_update_*.pt"))),
        "checkpoint_written_by_ranks": [r["checkpoint_written"] for r in ranks],
        "both_ranks_participated": len({r["device"] for r in ranks}) == args.world_size,
        "rank_devices": sorted(r["device"] for r in ranks),
        "rank_gpu_names": sorted(r["gpu_name"] for r in ranks),
        "rank_peak_allocated_bytes": [r["peak_allocated_bytes"] for r in ranks],
        "strict_param_count": sum(1 for r in ranks
                                  if sum(1 for _ in r["losses"]) == args.steps) == args.world_size,
        "train_indices_unique": all(r["sampler_audit"]["no_duplicates"] for r in ranks),
        "train_indices_cover_dataset": all(r["sampler_audit"]["covers_dataset"] for r in ranks),
        "train_pooled_count": ranks[0]["sampler_audit"]["count"],
        "train_dataset_length": ranks[0]["sampler_audit"]["dataset_length"],
        "val_ids_unique": all(r["val_sampler_audit"]["no_duplicates"] for r in ranks),
        "val_covers_dataset": ranks[0].get("val_covers_dataset"),
        "val_pooled_unique": ranks[0].get("val_ids_unique"),
        "val_id_count": len(ranks[0].get("pooled_val_ids", [])),
        "val_expected_count": VAL_LENGTH,
        "scientific_claim": False,
        "limitations": [
            "synthetic shape fixture; not multivariate real ERA5 and not weather truth",
            "periodic, not a convergence or forecast-skill measurement",
            "DDP gives throughput, not single-model memory headroom",
        ],
    }
    write_json(out / "comparison.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ddp", "reference", "compare"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--per-gpu-batch", type=int, default=2, dest="per_gpu_batch")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=2, dest="world_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-steps", type=int, default=4, dest="reasoning_steps")
    parser.add_argument("--global-batch", type=int, default=None, dest="global_batch")
    parser.add_argument("--resume")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.bf16 and not torch.cuda.is_available():
        parser.error("bf16 on CUDA requested but CUDA is unavailable")
    if args.mode == "ddp":
        args.global_batch = args.per_gpu_batch * args.world_size * args.accumulation
        ddp_main(args)
    elif args.mode == "reference":
        args.global_batch = args.global_batch or args.per_gpu_batch * args.world_size
        reference_main(args)
    else:
        compare_main(args)


if __name__ == "__main__":
    main()
