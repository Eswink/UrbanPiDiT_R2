"""Re-infer the registered checkpoints and **keep the fields**, then run balance diagnostics.

Why this script exists: every earlier evaluation in this repo collapsed the spatial axes
before writing anything. ``training/r7_boundary_metrics.py:99`` sums over ``(-1, -2)`` and
``training/r7_rollout_metrics.py:75`` says it outright -- ``# Only compact [L,C] sufficient
statistics survive update()``. A survey on 2026-11-14 found
``find outputs -path "*/evaluation/*" -type f ! -name "*.csv" ! -name "*.json"`` returns **0**:
no forecast field has ever been written to disk, so every spatial diagnostic would otherwise
have to re-infer from scratch, forever.

This script writes the fields once, as ``float32`` ``[lead, channel, H, W]`` per case, plus the
matching targets and the grid. The diagnostics in `training/r7_balance_diagnostics.py` are
then pure functions of those arrays and can be re-run at zero GPU cost.

It does **not** modify `training/r7_evaluate.py` (316 lines) or
`training/r7_long_rollout_runner.py` (597/600 lines, R-051); it reuses
`EvaluationPreparation`, `ZarrRolloutDataset`, `rollout_model_input` and
`autoregressive_rollout` exactly as `evaluate_local` does, and stops before the accumulators
that would throw the fields away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: One dump per case is a separate file, so an interrupted run resumes rather than restarting.
CASE_TEMPLATE = "case_{index:05d}.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_state(checkpoint: Path, spec: dict):
    """Build the model under the current revision and load the checkpoint strictly.

    ``load_checkpoint`` returns the saved tensors under ``"model"``, not ``"state"``
    (``training/r7_experiment.py:124-134``); reading the wrong key would raise ``KeyError``
    only after the whole evaluation had been set up, so the key is named here and asserted.
    """
    import torch
    from training.r7_experiment import load_checkpoint, make_model

    saved = load_checkpoint(checkpoint)
    if "model" not in saved:
        raise KeyError("checkpoint carries no 'model' state dict; keys are %s"
                       % sorted(saved.keys()))
    model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


def _dump_one(model, dataset, index, *, lead_hours, step_hours, device, out_dir: Path):
    """Run one case's rollout and write ``forecast``/``target`` in **physical units**.

    ``ZarrRolloutDataset.__getitem__`` applies ``(frames - mean) / std`` at read time
    (``data/r7_evaluation.py:113-114``), so both the forecast and ``rollout_targets`` come
    back in the **normalised** space the model was trained in. The balance diagnostics are
    dimensional -- ``delta_HB`` compares a geopotential thickness in ``m**2 s**-2`` against
    ``R*T`` -- so writing the normalised arrays would produce deviations of order -50 and
    thicknesses of order 1, which reads as "the model catastrophically violates hydrostatic
    balance" when it only means "the units are wrong".

    Measured 2026-11-14: without the de-normalisation the 72 h forecast's ``delta_HB`` median
    is -51.12 with the target's standard deviation at 7088; with it, both fields are O(0.01).
    Both ``mean`` and ``std`` are saved alongside, so a reader can always undo the transform
    and verify which space a file is in.
    """
    import numpy as np
    import torch
    # Both live in `model.r7_rollout`, not `training`: `training/r7_evaluate.py:14` imports them
    # the same way. Getting this wrong fails only at the first rollout.
    from model.r7_rollout import autoregressive_rollout, rollout_model_input

    sample = dataset[index]
    initial = rollout_model_input(sample, lead_hours=float(sample["lead_time_hours"]),
                                  device=device)
    trajectory = autoregressive_rollout(model, initial, lead_hours=list(lead_hours),
                                        step_hours=step_hours,
                                        history_interval_hours=step_hours,
                                        inference_kwargs={})
    # ``forecasts`` is [B, L, C, H, W]; this round is deterministic and single-member.
    forecast = trajectory.forecasts[0].detach().to(torch.float32).cpu().numpy()
    target = sample["rollout_targets"].detach().to(torch.float32).cpu().numpy()
    latitude = sample["latitude"].detach().to(torch.float64).cpu().numpy()
    longitude = sample["longitude"].detach().to(torch.float64).cpu().numpy()

    mean = np.asarray(dataset.mean, dtype=np.float64).reshape(-1, 1, 1)
    std = np.asarray(dataset.std, dtype=np.float64).reshape(-1, 1, 1)
    forecast = (forecast.astype(np.float64) * std + mean).astype(np.float32)
    target = (target.astype(np.float64) * std + mean).astype(np.float32)

    path = out_dir / CASE_TEMPLATE.format(index=index)
    np.savez_compressed(
        path,
        forecast=forecast,
        target=target,
        latitude=latitude,
        longitude=longitude,
        lead_hours=np.asarray(lead_hours, dtype=np.int64),
        channels=np.asarray(dataset.names),
        units=np.asarray(dataset.units),
        mean=mean.reshape(-1),
        std=std.reshape(-1),
        units_are_physical=np.asarray(True),
        init_time=np.asarray(str(sample["init_time"])),
    )
    return path, int(forecast.shape[0])


def dump_arm(checkpoint: Path, spec: dict, manifest: Path, out_dir: Path, *,
             lead_hours=(6, 12, 24, 48, 72), step_hours: int = 6, max_cases: int = 10**9,
             device_name: str = "cuda", deadline: float | None = None,
             resume: bool = True) -> dict:
    """Dump every case's fields for one checkpoint. Returns a receipt-shaped report."""
    import torch
    from training.r7_evaluate import EvaluationPreparation
    from training.r7_experiment import select_device

    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    prepared = EvaluationPreparation(manifest, step_hours=step_hours)
    dataset = prepared.dataset(lead_hours)

    model, saved = _load_checkpoint_state(checkpoint, spec)
    model.to(device)

    written, skipped = [], 0
    total = min(int(max_cases), len(dataset))
    for index in range(total):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"deadline reached after {index} of {total} cases")
        path = out_dir / CASE_TEMPLATE.format(index=index)
        if resume and path.is_file():
            skipped += 1
            continue
        path, _ = _dump_one(model, dataset, index, lead_hours=lead_hours,
                            step_hours=step_hours, device=device, out_dir=out_dir)
        written.append(path.name)

    files = sorted(out_dir.glob("case_*.npz"))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_updates": int(saved.get("updates", -1)),
        "model_code_sha256": saved.get("model_code_sha256"),
        "manifest": str(manifest),
        "lead_hours": list(lead_hours),
        "step_hours": step_hours,
        "cases_written": len(written),
        "cases_reused": skipped,
        "cases_total": total,
        "fields_dir": str(out_dir),
        "fields_bytes": int(sum(p.stat().st_size for p in files)),
        "fields_digest": _digest_of_listing(files),
        "device": str(device),
    }


def _digest_of_listing(files) -> str:
    """One digest over (name, sha256) pairs, so a resumed dump is comparable to a fresh one."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path,
                       help="JSON file holding the model spec as {\"kind\": \"process\", "
                            "\"spec\": {...}}")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cases", type=int, default=10**9)
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    options = parser.parse_args(argv)

    body = json.loads(options.spec.read_text(encoding="utf-8"))
    deadline = (time.monotonic() + options.deadline_seconds
                if options.deadline_seconds is not None else None)
    report = dump_arm(options.checkpoint, body["spec"], options.manifest, options.out,
                      max_cases=options.max_cases, device_name=options.device,
                      deadline=deadline, resume=not options.no_resume)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
