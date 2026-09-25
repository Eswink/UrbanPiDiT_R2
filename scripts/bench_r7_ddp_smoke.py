"""Dual-GPU DDP bring-up smoke: participation, loss identity, sampler dedup, resume.

Run under `torchrun --nproc_per_node=2`. The DDP path is verified against a
single-GPU reference built from the identical synthetic samples, so a passing
run is evidence about distributed correctness only - not forecast skill, and not
a single-model memory improvement (the two cards hold two model replicas).

#61 fixes and scope, all labeled in the artifacts:
- The validation depth comes **only** from `--reasoning-steps`; a forward hook
  on the recursion cell records the depth actually executed per validation
  batch and per training update, so `--steps 10 --reasoning-steps 4` verifies
  K=4, never K=10.
- Resume rejects every run-contract change (seed, per-GPU batch, accumulation,
  reasoning depth, training mode, dataset signature, world size, model code)
  and checkpoints missing contract fields.
- Training modes are labeled separately: `full_bptt`, `retained_truncated`
  (detach between reasoning steps; per-step graphs retained) and
  `streamed_truncated`. Streamed backward is **refused** here rather than
  silently falling back to full BPTT; it is not verified under DDP. These
  labels are distinct semantics and are never claimed equivalent.
- Sampler padding is explicit: `DistributedSampler(drop_last=False)` repeats
  indices to even out ranks, so lengths that do not divide the world size
  report their padded duplicates instead of claiming "no duplicates".

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
CHECKPOINT_FORMAT = "r7-ddp-smoke-v2"
TRAINING_MODES = ("full_bptt", "retained_truncated", "streamed_truncated")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")


def model_config_for_mode(mode):
    """Per-mode model semantics; the three labels are distinct, never aliases.

    `retained_truncated` detaches between reasoning steps, so each step keeps
    its own (retained) graph and gradients are truncated between steps - which
    is NOT equivalent to `full_bptt`. `streamed_truncated` shares the truncate-
    between-steps gradient contract but streams the backward; it is refused in
    this script until a dedicated DDP verification exists.
    """
    config = dict(MODEL_CONFIG)
    if mode == "full_bptt":
        return config
    if mode == "retained_truncated":
        config["detach_between_steps"] = True
        return config
    raise ValueError(f"training mode {mode!r} has no model mapping in this script")


def dataset_signature(length, val_length):
    """Digest of the exact synthetic dataset identity for the resume contract."""
    return canonical_digest({"dataset": dict(DATASET_CONFIG, length=int(length)),
                             "val_length": int(val_length)})


def refuse_unverified_combinations(mode, training_mode):
    """Fail closed instead of silently substituting an unverified code path."""
    if training_mode == "streamed_truncated":
        raise SystemExit(
            "streamed_truncated is not verified under this DDP smoke (multiple "
            "backward passes, encoder boundary gradients and unused params are "
            "unaudited); refusing instead of silently falling back to full BPTT. "
            "Single-GPU streamed training lives in training/r7_streaming.py.")
    if mode in ("ddp", "reference", "compare") and training_mode not in TRAINING_MODES:
        raise SystemExit(f"unknown training mode: {training_mode!r}")
    return True


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(seed, device, training_mode="full_bptt"):
    seed_everything(seed)
    model = GenericRecursiveWeatherForecaster(
        **model_config_for_mode(training_mode)).to(device)
    return model.train()


def batch_loss(model, batch, steps, bf16):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(forecast_inputs(batch), reasoning_steps=steps)
    return deep_supervised_forecast_mse(
        out.draft_forecasts, batch["atmos_target"], batch.get("latitude"), final_weight=2.0)


def attach_reasoning_counter(module):
    """Count recursion-cell forward calls; returns (handle, call-count list).

    The cell fires exactly once per reasoning step per forward, so the list
    length is the depth actually executed - observed, not assumed.
    """
    counts = []
    handle = module.cell.register_forward_hook(
        lambda mod, inputs, output: counts.append(1))
    return handle, counts


def run_validation(model, val_loader, device, *, reasoning_steps, bf16):
    """Evaluate with K taken only from `reasoning_steps`, observed via the hook.

    Returns the validation losses, the val sample ids this rank saw, and the
    per-batch observed depth. Raises if the observed depth differs from the
    requested one, so a silently misrouted `--steps` value cannot pass.
    """
    base = model.module if hasattr(model, "module") else model
    handle, counts = attach_reasoning_counter(base)
    seen, val_losses, observed = [], [], []
    was_training = base.training
    base.eval()
    try:
        with torch.no_grad():
            for raw in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in raw.items()}
                before = len(counts)
                loss = batch_loss(model, batch, reasoning_steps, bf16)
                val_losses.append(float(loss.detach()))
                seen.extend(list(batch["sample_id"]))
                observed.append(len(counts) - before)
    finally:
        handle.remove()
        if was_training:
            base.train()
    if any(count != reasoning_steps for count in observed):
        raise SystemExit(
            f"observed validation reasoning depth {observed} differs from the "
            f"requested {reasoning_steps}; the smoke refuses to report K it did "
            "not observe")
    return {"val_losses": val_losses, "seen_val_ids": seen,
            "observed_reasoning_steps": observed,
            "requested_reasoning_steps": reasoning_steps}


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
    """What the DistributedSampler actually hands each rank, padding included.

    With drop_last=False a length that does not divide the world size is padded
    by repeating indices; the audit reports those duplicates explicitly instead
    of claiming every sample appears exactly once.
    """
    size = DATASET_CONFIG["length"] if length is None else length
    pooled = []
    for rank in range(world_size):
        pooled.extend(local_indices(rank, world_size, size, shuffle, 42))
    counts = {}
    for index in pooled:
        counts[index] = counts.get(index, 0) + 1
    padded = sorted(index for index, n in counts.items() if n > 1)
    return {"tag": tag, "count": len(pooled), "dataset_length": size,
        "covers_dataset": sorted(counts) == list(range(size)),
        "no_duplicates": len(pooled) == len(set(pooled)),
        "padding_strategy": "DistributedSampler(drop_last=False) repeats indices "
                            "to even out per-rank counts",
        "padding_samples": len(pooled) - size,
        "padded_indices": padded,
        "per_rank_counts": [len(local_indices(r, world_size, size, shuffle, 42))
                            for r in range(world_size)]}


def no_sync_for(position, last, sync, distributed):
    """DDP accumulation rule: defer the all-reduce on all but the last microbatch
    of a complete group; an incomplete group must synchronize on its final
    microbatch because no further microbatch is coming."""
    return bool(distributed and sync and position < last)


def run_ddp_steps(model, optimizer, dataset, config, device, rank, steps, bf16,
                  accumulation, start_update=0):
    """Train optimizer updates until `steps`; resumes mid-schedule when asked.

    Every loss entry records the recursion depth actually executed per
    microbatch (observed through the forward hook), so training K is verified
    rather than assumed.
    """
    world_size = dist.get_world_size()
    base = model.module if hasattr(model, "module") else model
    handle, counts = attach_reasoning_counter(base)
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
            value, consumed, depth_used = _accumulate_step(
                model, optimizer, pending, device, config, bf16,
                len(pending) == accumulation, counts)
            losses.append({"update": updates + 1, "loss": value, "epoch": epoch,
                           "indices": consumed,
                           "reasoning_calls": depth_used // len(pending)})
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
        value, consumed, depth_used = _accumulate_step(
            model, optimizer, pending, device, config, bf16, True, counts)
        losses.append({"update": updates + 1, "loss": value, "epoch": epoch,
                       "indices": consumed, "reasoning_calls": depth_used // len(pending),
                       "partial_accumulation": len(pending)})
        updates += 1
        epoch += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    handle.remove()
    return losses, time.perf_counter() - started


def _accumulate_step(model, optimizer, batches, device, config, bf16, sync, counts=None):
    """One optimizer update over the pending microbatches, sample-weighted.

    Gradient semantics: the first n-1 microbatches of a complete accumulation
    group run under no_sync() (forward included); the final backward
    synchronizes the accumulated gradient once. An incomplete group
    synchronizes on its only/final microbatch. Returns the sample-weighted
    loss, the dataset indices consumed, and the recursion depth actually
    executed across these microbatches.
    """
    distributed = dist.is_available() and dist.is_initialized() \
        and dist.get_world_size() > 1
    optimizer.zero_grad(set_to_none=True)
    count = sum(b["coarse_history"].shape[0] for b in batches)
    total, consumed = 0.0, []
    before = len(counts) if counts is not None else None
    last = len(batches) - 1
    for position, raw in enumerate(batches):
        consumed.extend(sample_index(raw))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in raw.items()}
        context = model.no_sync() if no_sync_for(position, last, sync, distributed) \
            else contextlib.nullcontext()
        with context:
            loss = batch_loss(model, batch, config.reasoning_steps, bf16)
            (loss * (batch["coarse_history"].shape[0] / count)).backward()
        total += float(loss.detach()) * (batch["coarse_history"].shape[0] / count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    depth_used = (len(counts) - before) if counts is not None else config.reasoning_steps
    return total, consumed, depth_used


def validate_resume_contract(saved, *, world_size, model_code_sha256, seed,
                             per_gpu_batch, accumulation, reasoning_steps,
                             training_mode, dataset_signature_value, target_updates):
    """Fail closed on any run-contract change between checkpoint and this run.

    Returns the saved update count. Every field that changes the training or
    evaluation contract is checked explicitly, and checkpoints that predate a
    field are refused rather than assumed compatible.
    """
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise SystemExit(
            f"resume requires a {CHECKPOINT_FORMAT} checkpoint; got "
            f"{saved.get('format')!r}. Older formats cannot prove their "
            "contract and are refused.")
    expectations = {
        "world_size": world_size,
        "model_code_sha256": model_code_sha256,
        "seed": seed,
        "per_gpu_batch": per_gpu_batch,
        "accumulation": accumulation,
        "reasoning_steps": reasoning_steps,
        "training_mode": training_mode,
        "dataset_signature": dataset_signature_value,
    }
    for field, expected in expectations.items():
        if field not in saved:
            raise SystemExit(
                f"checkpoint lacks {field!r}; the run contract cannot be verified")
        if saved[field] != expected:
            raise SystemExit(
                f"resume contract change rejected: {field} is "
                f"{saved[field]!r} in the checkpoint but {expected!r} in this run")
    recorded = saved.get("signature")
    recomputed = canonical_digest({
        "model": saved.get("model_config"), "seed": seed, "steps": saved.get("target_steps"),
        "per_gpu_batch": per_gpu_batch, "accumulation": accumulation,
        "world_size": world_size, "reasoning_steps": reasoning_steps,
        "training_mode": training_mode, "dataset_signature": dataset_signature_value})
    if recorded != recomputed:
        raise SystemExit(
            "checkpoint signature does not match its recorded contract fields; "
            "refusing a self-inconsistent checkpoint")
    start_update = int(saved["updates"])
    if start_update >= target_updates:
        raise SystemExit("resume endpoint must exceed the saved updates")
    return start_update


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

    dataset = SyntheticAtmosDataset(**dict(DATASET_CONFIG, length=args.length))
    val_dataset = SyntheticAtmosDataset(**dict(DATASET_CONFIG, length=args.val_length,
                                               seed=999))
    sampler_audit = check_sampler_partition(world_size, "train", shuffle=True,
                                            length=args.length)
    val_audit = check_sampler_partition(world_size, "val", shuffle=False,
                                        length=args.val_length)

    model = build_model(args.seed, device, args.training_mode)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4, weight_decay=1e-4)
    signature_value = dataset_signature(args.length, args.val_length)
    resume_info = None
    start_update = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_update = validate_resume_contract(
            saved, world_size=world_size, model_code_sha256=model_code_digest(),
            seed=args.seed, per_gpu_batch=args.per_gpu_batch,
            accumulation=args.accumulation, reasoning_steps=args.reasoning_steps,
            training_mode=args.training_mode, dataset_signature_value=signature_value,
            target_updates=args.steps)
        # Saved from the unwrapped module, so it is loaded there too: the DDP
        # wrapper would otherwise expect a `module.` prefix on every key.
        model.load_state_dict({k: v.to(device) for k, v in saved["weights"].items()},
                              strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        resume_info = {"resumed_from": str(Path(args.resume).name),
                       "start_update": start_update, "target_update": args.steps}
        dist.barrier()
    losses, seconds = run_ddp_steps(ddp_model, optimizer, dataset, args, device, rank,
        args.steps, args.bf16, args.accumulation, start_update=start_update)

    # Validation pass: every val sample exactly once across ranks, statistics
    # pooled, K taken from --reasoning-steps and observed via the forward hook.
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.per_gpu_batch, sampler=val_sampler,
        collate_fn=default_collate)
    validation = run_validation(ddp_model, val_loader, device,
                                reasoning_steps=args.reasoning_steps, bf16=args.bf16)
    seen = validation["seen_val_ids"]
    val_losses = validation["val_losses"]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, seen)

    payload = {"rank": rank, "world_size": world_size, "local_rank": local_rank,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "model_code_sha256": model_code_digest(),
        "seen_val_ids": seen, "losses": losses,
        "seconds": seconds, "val_losses": val_losses,
        "training_mode": args.training_mode,
        "reasoning_steps": args.reasoning_steps,
        "requested_eval_reasoning_steps": validation["requested_reasoning_steps"],
        "observed_eval_reasoning_steps": validation["observed_reasoning_steps"],
        "training_reasoning_calls": [entry["reasoning_calls"] for entry in losses],
        "dataset_signature": signature_value,
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
            [f"synthetic_atmos_{i:05d}" for i in range(args.val_length)])
        weights = to_cpu(model.state_dict())
        optimizer_state = to_cpu(optimizer.state_dict())
        checkpoint = {"format": CHECKPOINT_FORMAT,
            "model_code_sha256": model_code_digest(),
            "model_config": model_config_for_mode(args.training_mode),
            "signature": canonical_digest({
                "model": model_config_for_mode(args.training_mode), "seed": args.seed,
                "steps": args.steps, "per_gpu_batch": args.per_gpu_batch,
                "accumulation": args.accumulation, "world_size": world_size,
                "reasoning_steps": args.reasoning_steps,
                "training_mode": args.training_mode,
                "dataset_signature": signature_value}),
            "target_steps": args.steps, "updates": args.steps,
            "weights": weights, "optimizer": optimizer_state,
            "world_size": world_size, "per_gpu_batch": args.per_gpu_batch,
            "accumulation": args.accumulation, "seed": args.seed,
            "reasoning_steps": args.reasoning_steps,
            "training_mode": args.training_mode,
            "dataset_signature": signature_value}
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
    dataset = SyntheticAtmosDataset(**dict(DATASET_CONFIG, length=args.length))
    model = build_model(args.seed, device, args.training_mode)
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
        "training_mode": args.training_mode,
        "reasoning_steps": args.reasoning_steps,
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
    partial_groups = [entry["partial_accumulation"] for r in ranks
                      for entry in r["losses"] if "partial_accumulation" in entry]
    observed_eval = [count for r in ranks
                     for count in r["observed_eval_reasoning_steps"]]
    observed_training_depth = [entry["reasoning_calls"] for r in ranks
                               for entry in r["losses"]]
    training_modes = sorted({r["training_mode"] for r in ranks})
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
        "partial_accumulation_group_sizes": partial_groups,
        "training_mode": training_modes[0] if len(training_modes) == 1 else training_modes,
        "model_code_sha256": ranks[0]["model_code_sha256"],
        "model_code_sha256_consistent":
            len({r["model_code_sha256"] for r in ranks}) == 1,
        "training_mode_labels_are_distinct": TRAINING_MODES,
        "reasoning_steps_requested": args.reasoning_steps,
        "optimizer_updates_requested": args.steps,
        "eval_and_train_depth_knobs_were_exercised_independently":
            args.steps != args.reasoning_steps,
        "observed_eval_reasoning_steps": observed_eval,
        "eval_depth_is_reasoning_steps":
            bool(observed_eval) and all(count == args.reasoning_steps for count in observed_eval),
        "observed_training_reasoning_calls": sorted(set(observed_training_depth)),
        "training_depth_is_reasoning_steps": bool(observed_training_depth) and all(
            count == args.reasoning_steps for count in observed_training_depth),
        "dataset_signature": ranks[0]["dataset_signature"],
        "dataset_signature_consistent": len({r["dataset_signature"] for r in ranks}) == 1,
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
        "train_padding_samples": ranks[0]["sampler_audit"]["padding_samples"],
        "train_padded_indices": ranks[0]["sampler_audit"]["padded_indices"],
        "train_pooled_count": ranks[0]["sampler_audit"]["count"],
        "train_dataset_length": ranks[0]["sampler_audit"]["dataset_length"],
        "val_ids_unique": all(r["val_sampler_audit"]["no_duplicates"] for r in ranks),
        "val_covers_dataset": ranks[0].get("val_covers_dataset"),
        "val_pooled_unique": ranks[0].get("val_ids_unique"),
        "val_id_count": len(ranks[0].get("pooled_val_ids", [])),
        "val_expected_count": args.val_length,
        "scientific_claim": False,
        "limitations": [
            "synthetic shape fixture; not multivariate real ERA5 and not weather truth",
            "periodic, not a convergence or forecast-skill measurement",
            "DDP gives throughput, not single-model memory headroom",
            "full_bptt, retained_truncated and streamed_truncated are distinct "
            "semantics; this run verified only the labeled mode, and streamed "
            "backward is refused here rather than substituted",
            "Process-arm and real-manifest DDP support are not claimed by this smoke",
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
    parser.add_argument("--training-mode", default="full_bptt", dest="training_mode",
                        choices=list(TRAINING_MODES))
    parser.add_argument("--length", type=int, default=DATASET_CONFIG["length"],
                        help="train dataset length; 31/33 exercise sampler padding")
    parser.add_argument("--val-length", type=int, default=VAL_LENGTH, dest="val_length")
    parser.add_argument("--global-batch", type=int, default=None, dest="global_batch")
    parser.add_argument("--resume")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    refuse_unverified_combinations(args.mode, args.training_mode)
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
