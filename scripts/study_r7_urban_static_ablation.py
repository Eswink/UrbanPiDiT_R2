"""S3 urban static covariates: three arms x three seeds from the pinned p1600 parent.

One factor, three levels, one code revision:

    control   -- ``static_conditioning=False``: no pathway is constructed at all
    treatment -- the published 8-layer static field + the published fixed projection
    placebo   -- the same 8-layer field with each covariate's cells spatially permuted
                 (fixed seed), same published projection

The arms share the store, both manifests, the parent checkpoint, the seed, the update count and
every recipe constant. They differ in ``backbone.static_conditioning_pathway.{field,projection}``
and nothing else, which is why the difference is a *state dict* difference and never a code one:
``model_code_digest()`` and ``load_checkpoint`` would reject a foreign ``model/**``, and the goal
document forbids editing it.

**How exit (f) must be read, stated before the numbers exist.** A per-cell permutation leaves each
covariate's raw multiset bit-identical but shrinks the per-token feature spread the pathway
actually reads (measured: 0.83x features, 0.69x codes) because ``static_features`` averages 2x2
patches. So the placebo differs from the treatment in spatial structure *and* in token amplitude.
(f) asks whether the coherent term beats a structureless one; if it passes, the honest claim is
"coherent full-amplitude static conditioning beats a permutation-destroyed version", not "the
effect is specifically due to spatial structure". If (a) passes and (f) fails, the round is
recorded negative - the effect would be equivalent to a position code - and the amplitude caveat
does not rescue it.

The 2023 test split is never read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_urban_static_ablation_protocol as proto
from scripts import r7_urban_static_placebo as placebo_mod
from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

ARMS = proto.ARMS
CONTROL, TREATMENT, PLACEBO = proto.CONTROL, proto.TREATMENT, proto.PLACEBO
SEEDS = proto.SEEDS
UPDATES, STEPS, BATCH_SIZE = proto.UPDATES, proto.STEPS, proto.BATCH_SIZE
LEADS = proto.LEADS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS
#: Exit (c)'s predeclared short-lead tolerance, and exit (f)'s attribution threshold in kelvin.
#: Both are frozen here, before any reading exists, and are never widened afterwards.
SHORT_LEAD_TOLERANCE = -0.005
ATTRIBUTION_THRESHOLD_K = 0.001


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


def _arm_states(seed):
    """The three state dicts for one seed, plus the parent record they all descend from."""
    static, manifest, fields, variables = proto._static_product()
    parent = proto._load_parent_state(seed)
    placebo, _permutations = placebo_mod.placebo_field(fields)
    states = {arm: proto._arm_state(parent, (fields, variables), placebo, arm=arm)
              for arm in ARMS}
    return states, parent, fields, variables, placebo


def _arm_contract(body, arm, parent):
    """The training contract for one arm: identical except for the declared initialization."""
    return {"kind": "process", "model": proto.arm_spec(arm),
            "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {
                "parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                "parent_checkpoint": str(parent["path"]),
                "parent_checkpoint_sha256": parent["sha256"],
                "parent_endpoint_updates": parent["updates"],
                "arm": arm,
                "description": (
                    "registered v3-BD p1600 endpoint under the current revision, plus the two "
                    "static buffers for the on arms; arm difference is buffer values only. "
                    f"{UPDATES} updates at batch {BATCH_SIZE}")},
            "autoregression": {
                "physical_steps": 12,
                "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                "window_sha256": body["train_windows"]["window_sha256"],
                "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}}


def _train_arm(output, body, arm, seed, deadline, parent, state):
    """One arm's training, with the parent state dict as the verified initialization."""
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_arm_harness import sha256_file
    model = proto._build(arm)
    folder = output / f"seed{seed}/training/{arm}"
    print(f"[seed {seed}] arm {arm}: {UPDATES} updates from the p1600 parent", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        curve.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, arm, parent), parent_weights=state,
        physical_weights=curve.PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES,
        lr=curve.LR, warmup=curve.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=BATCH_SIZE, checkpoint_every=curve.CHECKPOINT_EVERY,
        supervision_mask=None, loader=(4, 4))
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["train_data_identity"]):
        raise RuntimeError(f"seed {seed} arm {arm} did not reach the frozen endpoint")
    if report["supervision"]["kind"] != "full_grid":
        raise RuntimeError(f"seed {seed} arm {arm} supervised on "
                           f"{report['supervision']['kind']}, not the registered full grid")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained = saved["model"]
    # The arm's buffers must come out of training equal to what went in: they are buffers, not
    # parameters, so an optimizer step must not have touched them. Measured per arm.
    buffer_report = {}
    for key in proto.BUFFER_KEYS:
        present = key in trained
        buffer_report[key] = {
            "present": present,
            "unchanged_during_training": bool(present and torch.equal(trained[key], state[key])),
        }
    return {"seed": seed, "arm": arm, "updates": UPDATES, "batch_size": BATCH_SIZE,
            "reasoning_steps": STEPS, "training_dir": str(folder),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "training_report_sha256": sha256_file(folder / "training_report.json"),
            "parent_checkpoint_sha256": parent["sha256"],
            "parent_endpoint_updates": parent["updates"],
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": round(report["elapsed_seconds"] / UPDATES, 4),
            "static_conditioning": arm != CONTROL,
            "initial_static_buffers": buffer_report,
            "evaluations": {}}


def _score(output, body, arm, seed, receipt, deadline):
    """Five leads on val 2022 for one arm, sharing one evaluation preparation."""
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    prepared = EvaluationPreparation(curve.VAL_MANIFEST, deadline=deadline)
    tables = {}
    for lead in LEADS:
        destination = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {arm}: scoring lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(curve.VAL_MANIFEST), output_dir=destination, checkpoint=receipt["checkpoint"],
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=curve.REASONING_STEPS, boundary_margins=curve.BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        if provenance["split"] != "val" or "test_read" in provenance:
            raise RuntimeError(f"seed {seed} arm {arm} lead {lead} did not stay on val")
        table = destination / "boundary_rmse.csv"
        tables[str(lead)] = {"dir": str(destination),
                             "boundary_rmse_csv_sha256": sha256_file(table),
                             "n_evaluated": provenance["n_evaluated"],
                             "elapsed_seconds": provenance["elapsed_seconds"]}
    return tables


def _seed(output, body, seed, deadline):
    """Train and score all three arms for one seed, so every pairing is within-seed."""
    import torch
    states, parent, fields, variables, placebo = _arm_states(seed)
    receipts = {}
    for arm in ARMS:
        receipt = _train_arm(output, body, arm, seed, deadline, parent, states[arm])
        receipt["evaluations"] = _score(output, body, arm, seed, receipt, deadline)
        receipts[arm] = receipt
        print(f"[seed {seed}] arm {arm}: done", flush=True)
    payload = {"seed": seed, "arms": receipts,
               "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
               "static": {"static_identity": body["static"]["static_identity"],
                          "placebo_seed": placebo_mod.PLACEBO_SEED,
                          "variables": list(variables)}}
    wide.write_exclusive(output / f"seed{seed}_receipt.json", payload)
    return payload


def _region_rmse(table, region, variable):
    import csv
    with Path(table).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] == region and row["variable"] == variable:
                return float(row["rmse"])
    raise RuntimeError(f"{region}/{variable} missing from {table}")


def _row_count(table):
    import csv
    with Path(table).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def readings(output, body):
    """Score every arm at every lead and compute the predeclared exits from the tables."""
    from training.r7_arm_harness import sha256_file
    climatology = body.get("climatology_reference") or curve.climatology_reference()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"],
              "factor": "urban static covariate pathway, three levels",
              "static": body["static"], "preflight": body["preflight"],
              "family_argument": body["family_argument"], "seeds": {}}
    for seed in SEEDS:
        receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {}
        for arm in ARMS:
            arm_receipt = receipt["arms"][arm]
            entry[arm] = {"rmse": {}, "skill": {}, "rows": {}, "n_evaluated": {}}
            for lead in LEADS:
                record = arm_receipt["evaluations"][str(lead)]
                table = Path(record["dir"]) / "boundary_rmse.csv"
                if sha256_file(table) != record["boundary_rmse_csv_sha256"]:
                    raise RuntimeError(f"seed {seed} arm {arm} lead {lead} table drifted")
                rmse = _region_rmse(table, curve.REGION, "t2m")
                denominator = climatology["leads"][str(lead)]["t2m_rmse"]
                entry[arm]["rmse"][str(lead)] = rmse
                entry[arm]["skill"][str(lead)] = 1.0 - (rmse / denominator) ** 2
                entry[arm]["rows"][str(lead)] = _row_count(table)
                entry[arm]["n_evaluated"][str(lead)] = record["n_evaluated"]
            if sorted(int(key) for key in entry[arm]["rmse"]) != sorted(LEADS):
                raise RuntimeError(f"seed {seed} arm {arm} scored "
                                   f"{sorted(entry[arm]['rmse'])}, not {sorted(LEADS)}")
        # Every arm must have scored the same rows; unequal counts would mean different cases.
        counts = {arm: entry[arm]["rows"] for arm in ARMS}
        if len({json.dumps(value, sort_keys=True) for value in counts.values()}) != 1:
            raise RuntimeError(f"seed {seed}: the arms scored different row counts {counts}")
        entry["row_counts_equal_across_arms"] = True
        result["seeds"][str(seed)] = entry
    result["exit_conditions"] = evaluate_exit(result)
    return result


def _paired_delta(seeds, arm_a, arm_b, lead, *, on="skill"):
    """Per-seed paired difference ``arm_a - arm_b`` at one lead, keyed by seed string."""
    return {str(seed): float(entry[arm_a][on][str(lead)] - entry[arm_b][on][str(lead)])
            for seed, entry in seeds.items()}


def evaluate_exit(result):
    """The predeclared (a)-(f); every clause computed from readings, none hand-filled.

    (d) is deliberately **not** evaluated here. Decision 0109: a literal ``True`` is not a
    criterion. The control arm's bit-equivalence is carried by the
    ``static_conditioning=False -> static_conditioning_pathway=None`` early return plus
    ``tests/test_r7_urban_static_ablation.py``'s counter-proof test, which fails if that path is
    ever constructed.
    """
    seeds = result["seeds"]
    skill_72 = _paired_delta(seeds, TREATMENT, CONTROL, 72)
    skill_24 = _paired_delta(seeds, TREATMENT, CONTROL, 24)
    skill_48 = _paired_delta(seeds, TREATMENT, CONTROL, 48)
    skill_6 = _paired_delta(seeds, TREATMENT, CONTROL, 6)
    skill_12 = _paired_delta(seeds, TREATMENT, CONTROL, 12)
    # (f) is in KELVIN, not skill: placebo RMSE minus treatment RMSE, so positive means the
    # treatment is better (lower error). Using skill here would silently change the units the
    # threshold was predeclared in.
    rmse_72_placebo = {str(seed): float(entry[PLACEBO]["rmse"]["72"]) for seed, entry in seeds.items()}
    rmse_72_treatment = {str(seed): float(entry[TREATMENT]["rmse"]["72"]) for seed, entry in seeds.items()}
    attribution_k = {seed: rmse_72_placebo[seed] - rmse_72_treatment[seed] for seed in rmse_72_placebo}

    a = all(value >= 0.0 for value in skill_72.values())
    b = all(value >= 0.0 for value in list(skill_24.values()) + list(skill_48.values()))
    c = all(value >= SHORT_LEAD_TOLERANCE for value in list(skill_6.values()) + list(skill_12.values()))
    e = bool(result["preflight"].get("e_preflight_passed"))
    f = all(value >= ATTRIBUTION_THRESHOLD_K for value in attribution_k.values())

    if not (a and b and c):
        verdict = "permanently-close-the-urban-static-input-branch"
    elif not f:
        verdict = "negative-attribution-failure-effect-equivalent-to-a-position-code"
    elif not e:
        verdict = "preflight-failed-no-reading-is-admissible"
    else:
        verdict = "advance-to-independent-review"
    return {
        "a_72h_skill_nonnegative": a,
        "a_per_seed_delta_treatment_minus_control": skill_72,
        "b_mid_skill_nonnegative": b,
        "b_per_seed_delta_24h": skill_24,
        "b_per_seed_delta_48h": skill_48,
        "c_short_skill_within_tolerance": c,
        "c_per_seed_delta_6h": skill_6,
        "c_per_seed_delta_12h": skill_12,
        "c_tolerance": SHORT_LEAD_TOLERANCE,
        "d_control_bitwise_not_evaluated_here": (
            "carried by the static_conditioning=False -> pathway=None code path plus "
            "tests/test_r7_urban_static_ablation.py's counter-proof; decision 0109 forbids a "
            "literal verdict in the driver"),
        "e_preflight_passed": e,
        "f_attribution_treatment_beats_placebo": f,
        "f_per_seed_rmse_improvement_kelvin": attribution_k,
        "f_threshold_kelvin": ATTRIBUTION_THRESHOLD_K,
        "verdict": verdict,
        "reading_rule": ("(a) true with (f) false is recorded NEGATIVE: the static effect is "
                         "independent of spatial structure and equivalent to a position code. "
                         "(f) never replaces (a) and (a) alone is never a success."),
        "placebo_amplitude_caveat": result["preflight"]["placebo_contrast"]["confound_statement"],
    }


def worker(output, phase, seed, deadline):
    """One stage of this round, matching the wrapper's ``--worker`` contract."""
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        return proto.prepared_protocol(output)
    if phase == "archive":
        return wide._archive(output, proto.validate_protocol(output))
    body = proto.validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside the frozen round")
    from scripts import study_r7_s3_v3_rollout_ft as recipe
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    arguments = parser.parse_args(argv)
    if not arguments.worker:
        raise SystemExit("dual-card rounds are driven by `r7_round.py stage-seeds`; invoke this "
                         "driver with --worker <prepare|archive|seed|reading>")
    # The worker's return value is discarded: ``sys.exit(value)`` would print a protocol-sized
    # body into the stage log and set a nonzero status.
    worker(arguments.out, arguments.worker, arguments.seed, arguments.deadline)


if __name__ == "__main__":
    main()
