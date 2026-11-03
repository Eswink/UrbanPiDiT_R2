"""One batch size, one process: measure real peak reserved memory and step time.

Run through ``scripts/r7_s3_batch_ceiling_probe.py``, which re-executes this file
once per batch size so a failed allocation cannot poison the next measurement.

Why a separate process per batch: ``torch.cuda.max_memory_reserved`` is a
high-water mark for the whole process, and the caching allocator keeps blocks it
has already reserved. Measuring batch 2 after batch 8 in the same process would
report batch 8's footprint for batch 2. The only honest per-batch number comes
from a fresh interpreter with a fresh context.

This file is a measurement instrument, not an experiment: it writes
``scientific_claim: false``, reads only the val-side store frames it needs, never
touches the test split, and discards its scratch training directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The registered spec of the round this probe sizes (K=4 / depth=4 / dim=192),
# the same one the R1 matched round and R2/R4 used.
SPEC = {
    "architecture": "window", "in_channels": 17, "history_steps": 2, "dim": 192,
    "patch_size": 2, "depth": 4, "heads": 4, "window_size": 8,
    "default_reasoning_steps": 4, "detach_between_steps": False,
    "use_forecast_feedback": True, "anchored_processes": 8, "free_processes": 8,
    "spacetime_inputs": True, "spacetime_field_mode": "fields",
}
PHYSICAL_WEIGHTS = [1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
MANIFEST = ("outputs/r7_s3_fy_multiyear_20261023_attempt01/store/"
            "manifests_treatment/train.jsonl")
PROTOCOL = "outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/protocol.json"
ARM = "treatment"


def _times_and_memory(batch_size, updates, steps):
    """Real forward/backward/step on real windows from the real store."""
    import torch
    from torch.utils.data import default_collate
    from training.r7_experiment import make_model
    from training.r7_long_rollout import training_long_rollout
    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset

    manifest = ROOT / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"manifest missing: {manifest}")
    # The 11 year-end exclusions and the window digest are the ones the round froze;
    # the dataset asserts them exactly, so they are read from the frozen protocol
    # rather than re-derived here.
    frozen = json.loads((ROOT / PROTOCOL).read_text(encoding="utf-8"))
    declared = frozen["arms"][ARM]["train_windows"]
    dataset = ZarrLongRolloutDataset(
        manifest, physical_steps=len(PHYSICAL_WEIGHTS),
        expected_exclusions=[item["sample_id"] for item in declared["exclusions"]],
        expected_window_sha256=declared["window_sha256"])
    if len(dataset) < batch_size:
        raise SystemExit(f"dataset has {len(dataset)} windows, need {batch_size}")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_reserved(device)

    model = make_model("process", SPEC).to(device).train()
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    # The first batch is built before the timing loop so the measurement times the
    # training step, not the manifest discovery that precedes it.
    order = list(range(batch_size))
    # Warm the store handles once: the per-update timing below must include data
    # construction, because that is where the 1.1-1.3 s of the real update goes.
    warm = default_collate([dataset[index] for index in order])
    del warm

    durations = []
    losses = []
    for _ in range(updates):
        started = time.perf_counter()
        # Data construction is inside the timed region on purpose: a probe that
        # times only the GPU step reports about half the real cost.
        batch = default_collate([dataset[index] for index in order])
        moved = {name: value.to(device) if torch.is_tensor(value) else value
                 for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        result = training_long_rollout(
            model, moved, reasoning_steps=steps, physical_weights=PHYSICAL_WEIGHTS)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(float(result.loss.detach()))

    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "batch_size": batch_size, "updates": updates, "steps": steps,
        "parameters": params,
        "peak_reserved_bytes": int(peak_reserved),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_mib": round(peak_reserved / 2**20, 1),
        "baseline_reserved_bytes": int(baseline),
        "cuda_free_bytes_after": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "seconds_per_update": [round(value, 4) for value in durations],
        "seconds_per_update_mean": round(sum(durations) / len(durations), 4),
        "losses": [round(value, 6) for value in losses],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    # Pin the card before the first CUDA call; a bare "cuda" then means this card.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)

    record = {"batch_size": args.batch, "scientific_claim": False, "test_read": False,
              "device": args.device, "ok": False}
    try:
        record.update(_times_and_memory(args.batch, args.updates, args.steps))
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the whole point is to record, never raise
        import torch
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["out_of_memory"] = isinstance(exc, torch.cuda.OutOfMemoryError)
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
                      ("batch_size", "ok", "out_of_memory", "peak_reserved_mib",
                       "seconds_per_update_mean", "error")}, ensure_ascii=False))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
