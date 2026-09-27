"""Resume consistency on the M2 store: a resumed run must reproduce the original.

The campaign requires resume consistency to be covered on the segment actually
used, not only on synthetic fixtures. This runs one arm to a small update count,
resumes it to a larger endpoint, and compares against a single uninterrupted run
of that same endpoint on the **M2 store**, then checks the artifact identities a
resumed run could silently break: model weights, optimizer state, RNG state, the
cursor/epoch, the update count and the recorded code digest.

Checkpoints are read through the repository's own ``load_checkpoint``, which
loads with ``weights_only=True`` and fails closed when the recorded
``model_code_sha256`` or contract signature does not match. Nothing here
deserializes a checkpoint with arbitrary-object loading.

Small update counts keep this inside the project's bounded-run discipline; the
point is the restore contract, not a training result.

    python scripts/study_r7_69_resume_check.py --out outputs/r7_m2_resume_check
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM = "generic"
ARM_CONFIG = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
              "window_size": 4, "patch_size": 2, "dropout": 0.0,
              "latent_tokens": 16, "default_reasoning_steps": 3}
HALF_UPDATES = 6
FULL_UPDATES = 12
BATCH_SIZE = 2
SEED = 41
# A fixed RNG seed for the resume comparison only: this is a determinism check on
# one arm, not a reported experiment, so no scientific seed is being selected.
DETERMINISM_SEED = 20260927
FIELDS_COMPARED = ("model", "optimizer", "rng", "cursor", "epoch", "updates",
                   "model_code_sha256", "contract")


def _max_abs_delta(left, right, path=""):
    """Largest elementwise difference over two nested tensor/container structures.

    Comparison is numeric rather than bitwise because this project already
    records that same-machine GPU training is not bitwise reproducible: two
    identically seeded fresh runs diverge by ~3e-08 after a few updates. A
    bitwise check would therefore report a resume defect that is really just the
    platform, so the control run below establishes the noise floor and this
    function measures against it.
    """
    worst = 0.0
    mismatch = None
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf"), f"{path}: key sets differ"
        for key in left:
            delta, problem = _max_abs_delta(left[key], right[key], f"{path}.{key}")
            if problem is None and delta > worst:
                worst = delta
            mismatch = mismatch or problem
        return worst, mismatch
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float("inf"), f"{path}: length differs"
        for index, (a, b) in enumerate(zip(left, right)):
            delta, problem = _max_abs_delta(a, b, f"{path}[{index}]")
            if problem is None and delta > worst:
                worst = delta
            mismatch = mismatch or problem
        return worst, mismatch
    if torch.is_tensor(left):
        if left.shape != right.shape:
            return float("inf"), f"{path}: shapes differ"
        if left.numel() == 0:
            return 0.0, None
        caster = torch.float64 if left.is_floating_point() else torch.int64
        delta = float((left.to(caster) - right.to(caster)).abs().max().item())
        return delta, None
    if left == right:
        return 0.0, None
    return float("inf"), f"{path}: {left!r} != {right!r}"


def main():
    parser = argparse.ArgumentParser(description="M2 resume consistency check (#69).")
    parser.add_argument("--manifests", default="outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_experiment import (dataset_identity, load_checkpoint, seed_everything)
    from training.r7_scheduled_runner import run_scheduled_updates

    train_manifest = Path(args.manifests) / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    channels = int(dataset[0]["coarse_history"].shape[1])
    arm_config = {"in_channels": channels, "out_channels": channels,
                  "history_steps": 2, **ARM_CONFIG}
    # The runner refuses to reduce validation to the training split, so the real
    # M2 held-out block is supplied. The runner snapshots and restores the RNG
    # around scoring, so this cadence cannot perturb the training stream; it is
    # what publishes the intermediate checkpoint the resume leg starts from.
    validation_dataset = ZarrRolloutDataset(Path(args.manifests).parent / "cache.zarr",
                                            split="val", lead_hours=(6,),
                                            history_steps=2, step_hours=6)

    def run(total_updates, run_dir, resume=None, validation_every=0):
        # The sampler draws from the global RNG, so both legs start from the same
        # global state; the comparison is then about the checkpoint restore and
        # not about which windows happened to be drawn first.
        seed_everything(DETERMINISM_SEED)
        return run_scheduled_updates(
            dataset, kind=ARM, model_config=arm_config, data_identity=identity,
            output_dir=run_dir, total_updates=total_updates, batch_size=BATCH_SIZE,
            steps=3, seed=SEED, lr=2e-4, clip=1.0, process_weight=0.0,
            warmup_updates=2, minimum_lr_ratio=0.1,
            validation_every=validation_every,
            early_stopping_patience=None, device_name=args.device, resume=resume,
            validation_dataset=validation_dataset, validation_lead_hours=(6,))

    reference_dir = output / "reference"
    resumed_dir = output / "resumed"
    control_dir = output / "control"

    # The reference run declares the FULL endpoint and validates every
    # HALF_UPDATES, so it publishes an intermediate checkpoint at update
    # HALF_UPDATES. The LR schedule is a function of ``total_updates``, so a leg
    # trained with a smaller endpoint is NOT a prefix of a longer one - it would
    # run a different learning rate at every step and any difference would be an
    # artifact of the test, not of the restore. Resuming from this run's own
    # intermediate checkpoint keeps the schedule, the validation cadence, the
    # sample order and the contract signature identical, which is exactly what a
    # real resume does.
    run(FULL_UPDATES, reference_dir, validation_every=HALF_UPDATES)
    reference_checkpoint = reference_dir / f"update_{FULL_UPDATES:07d}.pt"
    intermediate = reference_dir / f"update_{HALF_UPDATES:07d}.pt"
    if not intermediate.is_file():
        raise FileNotFoundError(f"the reference run published no intermediate "
                                f"checkpoint at update {HALF_UPDATES}: {intermediate}")

    run(FULL_UPDATES, resumed_dir, resume=str(intermediate),
        validation_every=HALF_UPDATES)
    resumed_checkpoint = resumed_dir / f"update_{FULL_UPDATES:07d}.pt"
    # Control: a second, independent run of the same endpoint with the same seed.
    # Without it a mismatch between the resumed and reference runs cannot be
    # attributed to the restore: this platform may simply not be bitwise
    # reproducible, and the campaign records exactly that for same-machine GPU runs.
    run(FULL_UPDATES, control_dir, validation_every=HALF_UPDATES)
    control_checkpoint = control_dir / f"update_{FULL_UPDATES:07d}.pt"

    resumed = load_checkpoint(resumed_checkpoint)
    reference = load_checkpoint(reference_checkpoint)
    control = load_checkpoint(control_checkpoint)

    def compare(left, right):
        """Per-field numeric deltas between two loaded checkpoints."""
        problems, deltas = [], {}
        for field in FIELDS_COMPARED:
            if field not in left or field not in right:
                problems.append(f"{field}: absent from one checkpoint")
                continue
            delta, problem = _max_abs_delta(left[field], right[field], field)
            deltas[field] = delta
            if problem:
                problems.append(problem)
        return deltas, problems

    # The control pair (two identically seeded fresh runs) establishes the
    # platform's own noise floor; the resume pair is then judged against it.
    control_deltas, control_problems = compare(reference, control)
    resume_deltas, resume_problems = compare(resumed, reference)
    # A tolerance one order of magnitude above the observed platform noise, so a
    # genuine restore bug (which loses whole updates, not the last mantissa bit)
    # cannot hide behind it.
    noise_floor = max(control_deltas.values(), default=0.0)
    tolerance = max(noise_floor * 10.0, 1e-9)
    within_noise = {field: delta <= tolerance for field, delta in resume_deltas.items()}
    genuine = [problem for problem in resume_problems]
    for field, ok in within_noise.items():
        if not ok:
            genuine.append(f"{field}: delta {resume_deltas[field]:.3e} exceeds the "
                           f"{tolerance:.3e} tolerance derived from the control noise "
                           f"floor {noise_floor:.3e}")

    reproducible = noise_floor == 0.0 and not control_problems
    if reproducible:
        verdict = "bitwise consistent" if not genuine else "resume differs from a reference run"
    elif genuine:
        verdict = f"resume differs beyond the platform noise floor ({noise_floor:.3e})"
    else:
        verdict = ("consistent within the platform noise floor: two identically seeded "
                   f"fresh runs differ by up to {noise_floor:.3e}, and the resumed run "
                   "differs from its reference by no more than that")
    result = {
        "format": "r7-69-resume-check-v1",
        "scientific_claim": False,
        "store": str(Path(args.manifests).parent / "cache.zarr"),
        "data_identity": identity,
        "arm": ARM,
        "half_updates": HALF_UPDATES,
        "full_updates": FULL_UPDATES,
        "seed": SEED,
        "sampler_rng_seed": DETERMINISM_SEED,
        "device": args.device,
        "resumed_checkpoint": str(resumed_checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "control_checkpoint": str(control_checkpoint),
        "fields_compared": list(FIELDS_COMPARED),
        "checkpoint_loader": ("training.r7_experiment.load_checkpoint "
                              "(weights_only=True, fails closed on digest mismatch)"),
        "control_deltas": {k: float(v) for k, v in control_deltas.items()},
        "resume_deltas": {k: float(v) for k, v in resume_deltas.items()},
        "control_problems": control_problems,
        "resume_problems": genuine,
        "fresh_runs_bitwise_reproducible": reproducible,
        "platform_noise_floor": float(noise_floor),
        "tolerance": float(tolerance),
        "problems": genuine,
        "consistent": not genuine,
        "verdict": verdict,
        "note": ("both legs are seeded identically before their first update so the "
                 "comparison isolates the checkpoint restore. The control run (a "
                 "second identically seeded fresh run of the reference endpoint) "
                 "establishes the platform's noise floor, because same-machine GPU "
                 "training here is known not to be bitwise reproducible; the resume "
                 "pair is judged against that measured floor rather than against "
                 "bitwise equality"),
    }
    with (output / "resume_check.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    # The checkpoints are large and reproducible; only the verdict is kept.
    for directory in (resumed_dir, reference_dir, control_dir):
        shutil.rmtree(directory, ignore_errors=True)
    print(json.dumps({"verdict": verdict, "noise_floor": noise_floor,
                      "tolerance": tolerance,
                      "resume_deltas": {k: round(v, 12) for k, v in resume_deltas.items()}}))
    return 0 if result["consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())
