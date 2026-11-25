"""G1-B's frozen constants, arm pair, region scope and predeclared exits (decision 0119).

One factor is under test: whether the residual-branch gain depends on the visit index.
Two arms and one pinned reference carry it:

    A  ``visit_damped``    g_k = 1/sqrt(1+k), k the 0-based visit index
    B  ``uniform_damped``  g   = 1/sqrt(1+(K-1)/2), K = reasoning_steps = 4
    C  the registered incumbent, not retrained

Three things here are frozen *before* any reading and would change what counts as passing,
so they are stated as constants rather than left to the driver.

**The uniform arm is not the visit arm's mean gain.** At K=4 A's gains average 0.696114
while B multiplies by 0.632456, 9.14 percent less. ``1/sqrt`` is convex, so a constant
taken at the mean index always sits below the mean of the gains. A-B therefore mixes
"visit dependence" with "A damps less on average", and the protocol says so in
``attribution_boundary`` instead of letting a later reader infer a cleaner contrast.

**``interior_32`` is this instance's whole grid.** The G0 reading that admitted this
branch is a 65x65 instance, where ``boundary_masks(65, 65, (32,))`` leaves one cell. The
registered cross-check ``docs/R7_S3_WIDE_REGION_FULL.md`` proves the 129x129 instance's
``interior_32`` is 4,225 points = 65x65 and that its t2m climatology RMSE matches the
narrow instance to ~1e-10 per lead, so scoring the narrow instance's ``full`` region *is*
scoring that interior, not substituting a different one.

**The bridge is a within-round identity witness.** Decision 0118's exit 3 asked for A/B's
first-update loss to equal the archived parent's. B cannot satisfy it: its constant is
0.632456 already at k=0. And no archived first update from p32,132 exists, because the
archived ``loss_curve.first`` belongs to a run that started from p1600. The replacement
compares same-process tensors instead: the off path must reproduce the pre-switch
expression bit for bit, and A at k=0 must equal the off path because ``g_0`` is exactly
1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r7_s3_visit_gain_constants import (  # noqa: E402
    ARMS,
    ARM_NAMES,
    ATTRIBUTION_BOUNDARY,
    BATCH_SIZE,
    BOUNDARY_MARGINS,
    CHECKPOINT_EVERY,
    CLIP,
    CLOSED_AXES,
    DEVICE_MODE,
    ENVELOPE_MARGIN,
    ESTIMATED_PEAK_BYTES,
    EXITS,
    FORMAT,
    GPU_UUID,
    HARD_CAP_SECONDS,
    HEADROOM_MARGIN_BYTES,
    LEADS,
    LIMITATIONS,
    LR,
    MEASURED_SECONDS_PER_EVALUATION,
    PARENT_CHECKPOINT_SHA256,
    PARENT_MODEL_CODE_SHA,
    PARENT_RECORD,
    PARENT_RUN,
    PARENT_TEMPLATE,
    PARENT_UPDATES,
    PER_SEED_SECONDS,
    PHYSICAL_WEIGHTS,
    PLANNED_SECONDS,
    REASONING_STEPS,
    REFERENCE_ARM,
    REGION,
    REGION_NOTE,
    SEEDS,
    SOURCE_SHA256,
    STAGE,
    STEP_HOURS,
    STORE,
    TRAIN_MANIFEST,
    TREATED_SECONDS_PER_UPDATE,
    UPDATES,
    VAL_MANIFEST,
    VISIT_DAMPED, UNIFORM_DAMPED,
    WARMUP,
    WEIGHT_DECAY)


def visit_gains(arm: str, reasoning_steps: int = REASONING_STEPS) -> list[float]:
    """The frozen gain sequence for one arm, one entry per visit."""
    from model.recursive_weather_r7 import visit_gain_value

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
    return [visit_gain_value(arm, k, reasoning_steps) for k in range(int(reasoning_steps))]


def gain_table() -> dict:
    """Both arms' gain sequences plus the measured gap between uniform and visit means."""
    visit = visit_gains(VISIT_DAMPED)
    uniform = visit_gains(UNIFORM_DAMPED)
    visit_mean = sum(visit) / len(visit)
    return {
        VISIT_DAMPED: visit,
        UNIFORM_DAMPED: uniform,
        "visit_mean": visit_mean,
        "uniform_constant": uniform[0],
        "uniform_relative_shortfall": 1.0 - uniform[0] / visit_mean,
        # Pinned as arithmetic, not prose: if this ever reached equality the attribution
        # boundary above would be wrong and the ADR would need rewriting before training.
        "equal_mean_damping": math.isclose(uniform[0], visit_mean, rel_tol=1e-12),
    }


def arm_spec(arm: str, *, parent_spec: dict | None = None) -> dict:
    """The production spec for one arm, with the gain mode bound in.

    Read from a registered checkpoint rather than hand-written: the trunk has 28 spec keys and
    only some are obvious, and a hand-written spec would describe a model the registered weights
    are not.
    """
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    spec = dict(parent_spec or registered_parent_spec())
    spec.pop("visit_gain", None)
    spec["visit_gain"] = arm
    if int(spec.get("default_reasoning_steps", 0)) != REASONING_STEPS:
        raise RuntimeError(f"the parent spec reasons at "
                           f"{spec.get('default_reasoning_steps')} visits, not {REASONING_STEPS}")
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the parent spec is not the registered dim-192 depth-4 trunk")
    if int(spec.get("in_channels", 0)) != 17:
        raise RuntimeError("the parent spec is not the 17-channel instance")
    return spec


def registered_parent_spec() -> dict:
    """The spec of this round's registered p32,132 incumbent, seed 41."""
    import torch

    path = parent_checkpoint(SEEDS[0])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = dict(saved["contract"]["model"])
    spec.pop("visit_gain", None)
    if int(saved["updates"]) != PARENT_UPDATES:
        raise RuntimeError(f"the registered parent is at {saved['updates']} updates, "
                           f"not {PARENT_UPDATES}")
    return spec


def parent_checkpoint(seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    return PARENT_RUN / PARENT_TEMPLATE.format(seed=int(seed), updates=PARENT_UPDATES)


def parent_receipt(seed: int) -> dict:
    path = PARENT_RUN / f"seed{int(seed)}_incumbent_receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"the archived parent receipt is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parent_pin(seed: int) -> tuple[Path, str]:
    """The registered parent and its archived SHA256, refusing a pin that moved."""
    from training.r7_arm_harness import sha256_file

    path = parent_checkpoint(seed)
    if not path.is_file():
        raise FileNotFoundError(f"the registered parent is missing: {path}")
    receipt = parent_receipt(seed)
    observed = sha256_file(path)
    if receipt.get("checkpoint_sha256") != observed:
        raise RuntimeError(f"seed {seed} parent drifted from its archived receipt: "
                           f"{observed} != {receipt.get('checkpoint_sha256')}")
    pinned = PARENT_CHECKPOINT_SHA256.get(int(seed))
    if pinned is not None and pinned != observed:
        raise RuntimeError(f"seed {seed} parent drifted from its frozen pin")
    return path, observed


def envelope() -> dict:
    """One seed's requirement from the treated rate, against the frozen envelope."""
    per_arm = UPDATES * TREATED_SECONDS_PER_UPDATE + MEASURED_SECONDS_PER_EVALUATION
    need = len(ARMS) * per_arm
    return {
        "treated_seconds_per_update": TREATED_SECONDS_PER_UPDATE,
        "seconds_per_evaluation": MEASURED_SECONDS_PER_EVALUATION,
        "updates": UPDATES,
        "arms_per_seed": len(ARMS),
        "need_seconds": round(need, 1),
        "need_hours": round(need * len(SEEDS) / 3600.0, 3),
        "per_seed_seconds": PER_SEED_SECONDS,
        "margin": round(PER_SEED_SECONDS / need, 4),
        "declared_margin": ENVELOPE_MARGIN,
        "sufficient": PER_SEED_SECONDS >= ENVELOPE_MARGIN * need,
        "hard_cap_seconds": HARD_CAP_SECONDS,
        "waves": 2,
        "hard_cap_covers_waves": HARD_CAP_SECONDS >= 2 * PER_SEED_SECONDS,
        "rule": ("a truncated worker is not a pass, so the envelope is sized from the treated "
                 "rate with a declared multiple; the guard test compares the envelope against "
                 "this recomputation rather than against the constant it was built from"),
    }


def _write_exclusive(path: Path, body: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def train_windows() -> dict:
    """The 12-step long-rollout window digest the training contract must declare."""
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows

    found = preflight_long_rollout_windows(str(TRAIN_MANIFEST), physical_steps=12)
    return {"window_sha256": found["window_sha256"],
            "excluded_sample_ids": list(found["excluded_sample_ids"])}


def _frozen_constants() -> dict:
    from training.r7_experiment import model_code_digest

    return {
        "format": FORMAT, "stage": STAGE,
        "arms": [{"arm": name, "visit_gain": mode} for name, mode in ARMS],
        "reference_arm": REFERENCE_ARM,
        "seeds": list(SEEDS), "lead_hours": list(LEADS), "step_hours": STEP_HOURS,
        "updates": UPDATES, "batch_size": BATCH_SIZE,
        "steps": REASONING_STEPS, "lr": LR, "warmup_updates": WARMUP,
        "weight_decay": WEIGHT_DECAY, "clip": CLIP,
        "physical_weights": list(PHYSICAL_WEIGHTS),
        "checkpoint_every": CHECKPOINT_EVERY,
        "train_manifest": str(TRAIN_MANIFEST.relative_to(ROOT)),
        "val_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
        "store": str(STORE.relative_to(ROOT)),
        "region": REGION, "boundary_margins": list(BOUNDARY_MARGINS), "region_note": REGION_NOTE,
        "scientific_claim": False, "test_read": False, "r": 0,
        "visit_gains": gain_table(),
        "attribution_boundary": ATTRIBUTION_BOUNDARY,
        "closed_axes": CLOSED_AXES,
        "parent": {
            "run": str(PARENT_RUN.relative_to(ROOT)), "record_id": PARENT_RECORD,
            "updates": PARENT_UPDATES, "archived_model_code_sha256": PARENT_MODEL_CODE_SHA,
            "current_model_code_sha256": model_code_digest(),
        },
        "envelope": envelope(),
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                    "soft_rule": ("a soft overrun continues and is recorded; only the envelope "
                                  "truncates a worker, and a truncated worker is not a pass"),
                    "decision": "0119"},
        "exits": dict(EXITS),
        "limitations": LIMITATIONS,
    }


def arm_registry() -> list[dict]:
    """The six arm records: two gain modes x three seeds, each with its spec and parent pin."""
    records = []
    for seed in SEEDS:
        path, digest = parent_pin(seed)
        for name, mode in ARMS:
            records.append({
                "arm": name, "seed": int(seed), "visit_gain": mode,
                "model": {"spec": arm_spec(name)},
                "parent": {"checkpoint": str(path.relative_to(ROOT)), "sha256": digest,
                           "endpoint_updates": PARENT_UPDATES},
            })
    if len(records) != len(ARMS) * len(SEEDS):
        raise RuntimeError("the registry does not hold the predeclared six arms")
    return records


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol, including the six-arm registry."""
    from training.r7_device_selection import protocol_fields
    from training.r7_experiment import canonical_digest

    measured = envelope()
    if not measured["sufficient"]:
        raise RuntimeError(f"the per-seed envelope {PER_SEED_SECONDS} s is only "
                           f"{measured['margin']}x the {measured['need_seconds']} s the treated "
                           f"rate requires; declaring {ENVELOPE_MARGIN}x is the frozen contract")
    body = {**_frozen_constants(),
            "registry": arm_registry(),
            **protocol_fields(DEVICE_MODE, single_uuid=GPU_UUID),
            "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata and the arm registry prepared in a bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(Path(output) / "preparation_protocol.json", body)
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print the frozen constants and the gain table without writing")
    args = parser.parse_args(argv)
    if args.show:
        print(json.dumps({"gains": gain_table(), "envelope": envelope()}, indent=2,
                         ensure_ascii=False))
        return 0
    parser.error("nothing to do; pass --show")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
