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

FORMAT = "r7-s3-visit-gain-v1"
STAGE = "S3-VISIT-GAIN"

VISIT_DAMPED, UNIFORM_DAMPED = "visit_damped", "uniform_damped"
ARMS = ((VISIT_DAMPED, VISIT_DAMPED), (UNIFORM_DAMPED, UNIFORM_DAMPED))
ARM_NAMES = tuple(name for name, _ in ARMS)
REFERENCE_ARM = "incumbent"

SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
STEP_HOURS = 6

#: The round's visit count. ``uniform_damped``'s constant is a function of it, so it is frozen
#: here as well as in the spec, and the two are asserted equal before training.
REASONING_STEPS = 4

#: 2,400 updates. This is the dose the same cell was last priced at and the rung the G0
#: headroom reading is about; it is not the 32,132-update p4000 rung, which is a different
#: question and a different cost.
UPDATES = 2_400
BATCH_SIZE = 2
LR = 2e-5
WARMUP = 10
WEIGHT_DECAY = 1e-4
CLIP = 1.0
PHYSICAL_WEIGHTS = (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
CHECKPOINT_EVERY = 1_200

#: ``evaluate_local`` writes ``boundary_rmse.csv`` only when ``boundary_margins`` is given. On
#: this 65x65 instance ``(32,)`` would leave a single cell, so the margins are ``(1,)`` and the
#: region read is ``full``, which is the wide instance's ``interior_32`` (see the module docstring).
BOUNDARY_MARGINS = (1,)
REGION = "full"
REGION_NOTE = ("full(65x65) == interior_32(129x129): docs/R7_S3_WIDE_REGION_FULL.md records the "
               "wide instance's interior_32 as 4,225 points = 65x65 with t2m climatology RMSE "
               "matching the narrow v3-D2 instance to ~1e-10 per lead, so this region is that "
               "interior and not a substitute for it")

TRAIN_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/train.jsonl"
VAL_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl"
STORE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"

#: The registered in-protocol round whose ``incumbent`` arm is the parent this branch starts
#: from: three seeds at 32,132 updates, each with its own archived receipt and state digest.
PARENT_RUN = ROOT / "outputs/r7_s3_typed_evidence_in_protocol_20261120_attempt02"
PARENT_RECORD = "s3-typed-evidence-in-protocol"
PARENT_UPDATES = 32_132
PARENT_TEMPLATE = "seed{seed}/training/incumbent/update_{updates:07d}.pt"

#: The parent checkpoints' own SHA256, copied from each seed's archived
#: ``seed{N}_incumbent_receipt.json``. Pinning them here means a drifted parent is refused twice:
#: against the receipt and against this table.
PARENT_CHECKPOINT_SHA256 = {
    41: "8c154bd901f8d6a05f30476c2dc931ca253f99ff692d8ed9258f163e5e88e3e2",
    42: "9cb93456bc2ea244e69bdfe70b90219f5e61e853929886eed87360c514fd1831",
    43: "aacdeb59c9f101b11523f366ccc9fdcbfecfe36867bb9710136bd836ea482e19",
}

#: The archived model-code digest the parents were trained under (decision 0118's identity gate).
#: Adding ``visit_gain`` moves the live digest away from it, which is why a model-only migration
#: is required rather than a relaxed guard.
PARENT_MODEL_CODE_SHA = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"

#: Measured treated rate from the in-protocol round (generic 1.4043/1.4312/1.4271 s/update over
#: full runs, typed marginal 1.4565/1.4618/1.4618). The damped arms add two multiplies per visit
#: and are expected to sit at or slightly above these; 1.55 is above all six as a starting
#: envelope, and the per-seed probe re-derives it before any arm commits.
TREATED_SECONDS_PER_UPDATE = 1.55
#: Five-lead scoring per arm, measured 2,987.30-3,176.62 s at 32,132 updates. This round's arms
#: are the same size and the same leads, so the same bound applies.
MEASURED_SECONDS_PER_EVALUATION = 3_200.0
ENVELOPE_MARGIN = 2.0

#: One seed runs two arms and five leads each: 2 x (2,400 x 1.55 + 3,200) = 13,840 s, plus the
#: 400-update probe pair. 28,500 s is 2.06x that. The first draft used 27,600 s and the guard in
#: :func:`envelope` rejected it at 1.9942x -- the envelope is recomputed from the treated rate
#: rather than compared against the constant it was derived from, which is why the shortfall
#: surfaced here instead of after a truncated seed.
PER_SEED_SECONDS = 28_500.0
PLANNED_SECONDS = 118_800.0
HARD_CAP_SECONDS = 216_000.0

ESTIMATED_PEAK_BYTES = 9_728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
DEVICE_MODE = "dual"
#: The round's declared primary card. ``protocol_fields`` takes every visible card with this one
#: first, and the parent scheduler rotates the frozen list across the three seeds.
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"

#: The exclusions, verbatim, so a later reader does not have to reconstruct them.
CLOSED_AXES = {
    "capacity_0048": ("dim and depth stay 192/4, the state_dict key set is identical and the "
                      "parameter count is bit-identical across A, B and C; this is a signal-"
                      "propagation contrast, not a capacity one"),
    "mixing_0077": "global_mixing_mode is untouched and the receptive field is unchanged",
    "reasoning_depth": ("record:s3-reasoning-depth stays unresolved and is not re-run; K remains 4 "
                        "and is not swept"),
    "supervision": "the deep-supervision organization is untouched; both supervision branches are closed",
    "tier3": "no loop-index embedding, no per-loop LoRA, no spectral norm, no zero-init head",
    "physics_loss": "no physical penalty term is added; C1/C2 stay closed",
}

ATTRIBUTION_BOUNDARY = (
    "A-B isolates whether the gain depends on the visit index, but it is not a clean visit-"
    "alignment contrast: at K=4 A's gains average 0.696114 while B multiplies by 0.632456, so B "
    "damps 9.14 percent more on average (1/sqrt is convex, so a constant at the mean index sits "
    "below the mean of the gains). A-B must be read as 'at B's damping level, is visit dependence "
    "better'. B-C has a zero parameter difference and is descriptive only; B is a treated arm, not "
    "the parent, so B-C is not a causal reading of damping either.")

EXITS = {
    "one": ("72h full(65x65) t2m: A-B is negative on all three seeds. Sign is per seed; a single "
            "flip makes this false and closes the branch"),
    "two": ("6h and 12h: the seed mean of A-B is <= 0 at each of the two leads. A single seed may "
            "flip; the mean is what is judged"),
    "three_a": ("off-path identity: a cell built with visit_gain=None reproduces the pre-switch "
                "expression bit for bit, installs the archived 155 tensors exactly, and its "
                "state_dict key set and every tensor equal the archived parent's"),
    "three_b": ("k=0 bridge: visit_damped at visit 0 equals visit_gain=None on the same input bit "
                "for bit, because g_0 = 1/sqrt(1+0) is exactly 1.0. Holds for A only; B's "
                "constant is 0.632456 at every visit including the first"),
    "four": ("every seed's parameter key set and parent state_digest are bit-identical: the "
             "migration receipt proves the p32,132 state went in unchanged"),
    "reading_rule": ("an exit with no measurement is null, never false and never true. Three seeds "
                     "are consistency evidence, not a significance test. r = 0, so nothing here is "
                     "a confirmatory claim about the 2023 test split"),
    "closure": ("any exit false closes this branch: no added seed, no sweep over gain families, no "
                "change to K, no post-hoc threshold edit"),
}

LIMITATIONS = [
    "screening at 2,400 updates on one 65x65 instance (2000-2021 train / 2022 val, 17 channels); "
    "the 129x129 wide instance is a different store with different parents and is not covered",
    "A-B is not an equal-mean-damping contrast: B's constant is 9.14 percent below A's mean gain at "
    "K=4, so a negative A-B cannot be attributed to visit alignment alone",
    "the damping constants come from a post-LN DeepNorm paper (arXiv:2607.13491); this trunk is "
    "pre-LN, so importing the form is an extrapolation and none of that paper's numbers is an "
    "expected value here",
    "C is the registered incumbent at 32,132 updates, not retrained, so B-C compares a 2,400-update "
    "run against a longer one and is descriptive only",
    "the region read is full(65x65), which equals the wide instance's interior_32; that equivalence "
    "rests on a ~1e-10 climatology match recorded in docs/R7_S3_WIDE_REGION_FULL.md, not on a "
    "bitwise identity",
    "the 2023 test split stays sealed; r = 0; test_read = false",
    "three seeds are consistency evidence, not a significance test; no interval or multiplicity "
    "correction is computed",
    "GPU runs are co-resident, so latency and memory observations include neighbour load; "
    "reproducibility is config-level with no bitwise GPU promise",
    "the exits are predeclared screening gates on the +6h-centred objective family, not a skill "
    "claim: a branch that stays open still needs its own full-dose confirmation round",
]


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
