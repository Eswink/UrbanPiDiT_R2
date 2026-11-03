"""S3 R4: does keeping patch positions in the typed evidence change long-lead readings?

Decision 0078 admitted this round because the closed arm's null was a null under a weak
manipulation, and the closed page said so itself. Decision 0079 then measured that the
originally proposed fix -- raising a score multiplier -- cannot work as intended: the query
starts at exactly zero and AdamW cancels a uniform gradient rescale, so the multiplier moves
forward sensitivity and not learning amplitude. The round's question was narrowed accordingly.
Decision 0080 fixed the execution: both arms start from a freshly initialised zero query, and
the payload is rebuilt for the 65x65 store because the closed arm's payload carries another
store's statistics. Decision 0081 fixed the constant at 8000 by measurement on real val frames.

So: both arms train the same 2000-2021 treatment manifest at 16,071 updates and batch 2, use
the same migrated parent, the same published 65x65 sidecar and the same frozen
``position_scale``; the only factor is whether the four typed-evidence fields are aggregated by
a uniform mean over patch positions or by a softmax with one shared learned query.

The frozen protocol layer lives in ``scripts/r7_s3_position_retention_protocol.py`` so both
files stay under R-051's 600-line cap; this file is the training, scoring and reading runtime.

A separate driver, not a parameterisation of the closed one: the closed protocol must stay
readable as it was frozen, and an importing module would inherit its constants silently.
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

from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.r7_s3_fy_year_coverage_climatology import (
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_YEARS, D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS)
from scripts.r7_s3_position_retention_protocol import (
    BATCH_SIZE, BOUNDARY_MARGINS, FULL_GRID_POINTS, HARD_CAP_SECONDS, LR, PER_SEED_SECONDS,
    PHYSICAL_WEIGHTS, PLANNED_SECONDS, REASONING_STEPS, REGION, SEEDS, STEPS, STORE,
    TRAIN_MANIFEST, UPDATES, VAL_MANIFEST, WARMUP, _prepared_protocol, freeze, typed_payload,
    validate_protocol)
from scripts.r7_s3_position_retention_support import (
    ARMS, CANDIDATE, CLOSED_ROUND_BATCH_SIZE, CLOSED_ROUND_RECORD, CLOSED_ROUND_UPDATES,
    CONTROL, DECISION_RULE, ENTROPY_BAND, ENTROPY_ONE_HOT_FLOOR, MODES, POSITION_SCALE,
    QUERY_NORM_RANGE, SIDECAR_RUN, SIDECAR_TYPED_EVIDENCE_IDENTITY, arm_name)
from training import r7_device_selection as devices

def _expand(output, body, seed, mode):
    """Build one arm: migrated parent state plus a freshly initialised typed pathway.

    Decision 0080 route 2: the pathway is NOT migrated from the closed arm. Both arms of one
    seed are constructed under ``isolated_stream()`` from one seed, so they share every shared
    tensor bit for bit and the spatial arm's query starts at exactly zero -- that is what makes
    the two arms' starting point identical rather than merely similar.
    """
    import torch

    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from model.spacetime_conditioning_r7 import isolated_stream
    spec = body["arms"][arm_name(mode)]["model"]["spec"]
    path, expected = parent_pins.pinned_parent(seed)
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    if saved["updates"] != parent_pins.ENDPOINT:
        raise RuntimeError(f"seed {seed} parent is not the frozen endpoint")
    parent_state = saved["model"]
    with isolated_stream():
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    target_state = target.state_dict()
    missing = [key for key in target_state if key not in parent_state]
    expected_new = sorted(key for key in target_state if key.startswith("typed_evidence."))
    if sorted(missing) != expected_new:
        raise RuntimeError(f"seed {seed}: the typed pathway must add exactly {expected_new}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in target_state]
    if extra:
        raise RuntimeError(f"seed {seed}: the parent carries keys the target does not: {extra}")
    state = dict(parent_state)
    for key in missing:
        state[key] = target_state[key].detach().clone()
    target.load_state_dict(state, strict=True)
    query = state.get("typed_evidence.position_query")
    query_norm = 0.0 if query is None else float(query.detach().norm())
    if query is not None and query_norm != 0.0:
        raise RuntimeError(f"seed {seed}: the query did not start at exactly zero "
                           f"(|q|={query_norm})")
    pathway = {key: state[key] for key in sorted(expected_new)}
    expansion = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
                 "init_seed": int(seed),
                 "parent_state_sha256": _state_digest(parent_state),
                 "expanded_state_sha256": _state_digest(state),
                 "pathway_init_sha256": _state_digest(pathway),
                 "position_query_norm": query_norm,
                 "position_scale": POSITION_SCALE,
                 "new_params": sum(int(target_state[key].numel()) for key in missing)}
    return target, state, path, expected, expansion


def _zero_query_is_pooled(model):
    """The spatial arm's aggregation must reach the pooled mean when the query is zero.

    A zero query makes every score exactly zero for any ``position_scale``, so ``softmax`` is
    exactly ``1/P`` at every position -- that half is bitwise and is asserted as such. The
    aggregation itself multiplies each position by ``1/P`` and sums, while ``value.mean(dim=1)``
    sums and then divides by ``P``: the same real number, two different float32 programs. The
    reachable claim is therefore a bound measured here, not equality (decision 0082 records the
    correction; the closed arm's text demanded equality, which no implementation can satisfy).
    """
    import torch
    router = getattr(model, "typed_evidence", None)
    if router is None:
        raise RuntimeError("the built model carries no typed-evidence router")
    if router.mode == MODES[CONTROL]:
        return {"applicable": False, "reason": "the pooled arm has no query to zero"}
    eps = torch.finfo(torch.float32).eps
    with torch.no_grad():
        saved = router.position_query.detach().clone()
        router.position_query.zero_()
        scores = (torch.randn(2, 1089, router.dim, generator=torch.Generator().manual_seed(7))
                  @ router.position_query) * router.position_scale / (router.dim ** 0.5)
        uniform = bool(torch.equal(torch.softmax(scores, dim=1),
                                   torch.full_like(scores, 1.0 / scores.shape[1])))
        worst_difference, worst_ratio, bound_at_worst = 0.0, 0.0, 0.0
        for trial in range(8):
            value = torch.randn(2, 1089, router.dim,
                                generator=torch.Generator().manual_seed(700 + trial))
            zeroed = router._position_attention(value)
            pooled = value.mean(dim=1)
            difference = float((zeroed - pooled).abs().max())
            bound = 2 * value.shape[1] * eps * float(value.abs().max())
            if difference > worst_difference:
                worst_difference, bound_at_worst = difference, bound
            worst_ratio = max(worst_ratio, difference / bound)
        router.position_query.copy_(saved)
    if not uniform:
        raise RuntimeError("a zero query did not produce exactly uniform position weights")
    if worst_ratio > 1.0:
        raise RuntimeError("a zero query is not the pooled mean within float32 accumulation "
                           f"(max|d|={worst_difference:.3e} > bound {bound_at_worst:.3e})")
    return {"applicable": True, "zero_query_max_abs": worst_difference,
            "zero_query_bound": bound_at_worst, "zero_query_bound_ratio": worst_ratio,
            "uniform_weights_exact": uniform, "zero_query_is_pooled": True}


def _forward_and_entropy(model, payload):
    """Check the round's entropy criterion on a real val frame before any training runs.

    Reads one val frame, standardises it with this store's vectors, encodes the four typed
    fields and measures the softmax entropy the frozen constant produces at the query norm the
    arm will plausibly reach. A query of exactly zero gives entropy 1.0 for every scale, so the
    criterion is evaluated at the measured query-norm range, and the zero-start value is
    recorded alongside so the two are never confused.
    """
    import numpy as np
    import torch
    import zarr
    router = getattr(model, "typed_evidence", None)
    if router is None:
        return {"applicable": False}
    root = zarr.open_group(str(STORE), mode="r")
    record = json.loads(VAL_MANIFEST.read_text(encoding="utf-8").splitlines()[0])
    state = torch.tensor(np.asarray(root["state"][int(record["target_index"])],
                                    dtype=np.float32))
    mean = torch.tensor(payload["denorm_mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(payload["denorm_std"], dtype=torch.float32)[:, None, None]
    latitude = torch.tensor(np.asarray(root["latitude"][:], dtype=np.float32))[None]
    longitude = torch.tensor(np.asarray(root["longitude"][:], dtype=np.float32))[None]
    with torch.no_grad():
        fields = router.field_maps(((state - mean) / std)[None],
                                   {"latitude": latitude, "longitude": longitude})
        tokens = router.encode(fields)
        value = router.projections[0](tokens[:, 0])[0]
        positions = value.shape[0]

        def entropy(norm):
            generator = torch.Generator().manual_seed(41)
            query = torch.randn(value.shape[-1], generator=generator)
            query = query / query.norm() * norm
            scores = (value @ query) * router.position_scale / math.sqrt(value.shape[-1])
            weights = torch.softmax(scores, dim=0)
            return float(-(weights * torch.log(weights + 1e-30)).sum() / math.log(positions))

        result = {"applicable": router.mode != MODES[CONTROL],
                  "patch_positions": int(positions),
                  "position_scale": router.position_scale,
                  "entropy_at_zero_query": 1.0,
                  "entropy_by_norm": {str(norm): entropy(norm) for norm in QUERY_NORM_RANGE}}
    low, high = ENTROPY_BAND
    values = list(result["entropy_by_norm"].values())
    result["inside_band"] = all(low <= value <= high for value in values)
    result["one_hot"] = any(value < ENTROPY_ONE_HOT_FLOOR for value in values)
    return result


def _arm_contract(body, name, mode, parent, parent_sha, expansion):
    arm = body["arms"][name]
    return {"kind": "process", "model": arm["model"]["spec"],
            "data_identity": arm["train_data_identity"], "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "typed_evidence_sidecar": SIDECAR_RUN,
                               "typed_evidence_identity": SIDECAR_TYPED_EVIDENCE_IDENTITY,
                               "position_scale": POSITION_SCALE,
                               "description": ("registered v3-BD l6 endpoint, model-only "
                                               "migration under the current revision, then a "
                                               "freshly initialised typed-evidence pathway; "
                                               f"mode={mode}")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(PHYSICAL_WEIGHTS),
                               "window_sha256": arm["train_windows"]["window_sha256"],
                               "excluded_sample_ids": arm["train_windows"]["excluded_sample_ids"]}}


def _check_provenance(provenance, seed, name, lead):
    if provenance["split"] != "val" or "test_read" in provenance:
        raise RuntimeError(f"seed {seed} {name} lead {lead} did not stay on val")
    if (provenance["n_evaluated"] != VAL_COHORTS[str(lead)]
            or provenance["n_available_windows"] != VAL_COHORTS[str(lead)]
            or provenance["lead_hours"] != [lead]):
        raise RuntimeError(f"seed {seed} {name} lead {lead} cohort differs from the frozen one")
    regions = {region["region"]: region for region in
               (provenance["boundary_scoring"] or {}).get("regions", [])}
    if regions.get(REGION, {}).get("n_grid_points") != FULL_GRID_POINTS:
        raise RuntimeError(f"seed {seed} {name} lead {lead} full region is not 65x65")
    climatology = provenance["climatology"]
    if (climatology["training_years"] != CLIMATOLOGY_YEARS
            or len(climatology["bucket_counts"]) != CLIMATOLOGY_BUCKETS):
        raise RuntimeError(f"seed {seed} {name} lead {lead} climatology is not the 48-bucket fit")


def _train_arm(output, body, seed, name, mode, deadline):
    """Fine-tune one arm from the migrated parent and score its five leads."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha, expansion = _expand(output, body, seed, mode)
    control = _zero_query_is_pooled(model)
    entropy = _forward_and_entropy(model, typed_payload())
    contract = _arm_contract(body, name, mode, parent, parent_sha, expansion)
    folder = output / f"seed{seed}/training/{name}"
    print(f"[seed {seed}] arm {name}: training {UPDATES} updates at batch {BATCH_SIZE} "
          "from the migrated parent", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        TRAIN_MANIFEST, folder, model=model, contract=contract, parent_weights=state,
        physical_weights=PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline, batch_size=BATCH_SIZE)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["arms"][name]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    update1 = dict(report["losses"][0])
    receipt = {"seed": seed, "arm": name, "typed_evidence_mode": mode,
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha,
               "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES,
               "batch_size": BATCH_SIZE, "reasoning_steps": STEPS,
               "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"][name]["train_windows"]["window_sha256"],
               "expansion": expansion, "zero_query_control": control,
               "entropy_check": entropy,
               "update1": {"loss": update1["loss"], "gradient_norm": update1["gradient_norm"]},
               "evaluations": {}}
    print(f"[seed {seed}] arm {name}: trained in {report['elapsed_seconds']:.0f}s; "
          "building the shared evaluation inputs", flush=True)
    prepared = EvaluationPreparation(VAL_MANIFEST, deadline=deadline)
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {name}: evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        _check_provenance(provenance, seed, name, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
        print(f"[seed {seed}] arm {name}: lead {lead}h done in "
              f"{provenance['elapsed_seconds']:.0f}s over {provenance['n_evaluated']} cases",
              flush=True)
    return receipt


def _seed(output, body, seed, deadline):
    """Run one seed's two arms and assert the bridge identities before moving on."""
    import torch
    receipts = {}
    for name, mode in ARMS:
        print(f"[seed {seed}] arm {name}: start", flush=True)
        receipt = _train_arm(output, body, seed, name, mode, deadline)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        print(f"[seed {seed}] arm {name}: done", flush=True)
    verdict = bridge_verdict({str(seed): receipts})
    if not verdict["passed"]:
        raise RuntimeError(f"seed {seed}: the bridge identities did not hold: {verdict}")


def bridge_verdict(receipts):
    """Rule (e): the four identities, reported and explicitly never a scientific verdict."""
    seeds = {}
    for seed in SEEDS:
        control = receipts[str(seed)][CONTROL]
        candidate = receipts[str(seed)][CANDIDATE]
        a, b = control["expansion"], candidate["expansion"]
        entropy = candidate.get("entropy_check") or {}
        seeds[str(seed)] = {
            "control_query_norm": a.get("position_query_norm"),
            "candidate_query_norm": b.get("position_query_norm"),
            "zero_start_both_arms": (a.get("position_query_norm") == 0.0
                                     and b.get("position_query_norm") == 0.0),
            "control_pathway_init_sha256": a.get("pathway_init_sha256"),
            "candidate_pathway_init_sha256": b.get("pathway_init_sha256"),
            "pathway_init_equal": a.get("pathway_init_sha256") == b.get("pathway_init_sha256"),
            "control_position_scale": a.get("position_scale"),
            "candidate_position_scale": b.get("position_scale"),
            "position_scale_equal": (a.get("position_scale") == b.get("position_scale")
                                     == POSITION_SCALE),
            "zero_query_max_abs": (candidate.get("zero_query_control") or {}).get(
                "zero_query_max_abs"),
            "zero_query_bound": (candidate.get("zero_query_control") or {}).get(
                "zero_query_bound"),
            "uniform_weights_exact": bool(
                (candidate.get("zero_query_control") or {}).get("uniform_weights_exact")),
            "zero_query_is_pooled": bool(
                (candidate.get("zero_query_control") or {}).get("zero_query_is_pooled")),
            "entropy_inside_band": entropy.get("inside_band"),
            "entropy_one_hot": entropy.get("one_hot"),
            "entropy_by_norm": entropy.get("entropy_by_norm"),
        }
    for seed, entry in seeds.items():
        entry["passed"] = bool(
            entry["zero_start_both_arms"] and entry["pathway_init_equal"]
            and entry["position_scale_equal"] and entry["uniform_weights_exact"]
            and entry["zero_query_is_pooled"])
    return {"scientific": False, "record_id": CLOSED_ROUND_RECORD,
            "note": ("each of the four identities is an engineering fact about the two arms' "
                     "starting point, not a reading; the closed arm's bridge asserted a "
                     "different thing (migrated tensor SHA equals the archive) and is not "
                     "comparable"),
            "seeds": seeds, "passed": all(entry["passed"] for entry in seeds.values())}


def evaluate_exit(seeds, *, seeds_order=SEEDS):
    """The predeclared (a)-(d) conjunction; disagreement is never averaged away."""
    seeds = {str(key): value for key, value in seeds.items()}
    def skill(seed, lead):
        return seeds[str(seed)]["arms"][CANDIDATE]["full_t2m_skill"][str(lead)]
    def delta(seed, lead):
        return seeds[str(seed)]["delta_candidate_minus_control"][str(lead)]
    a = all(skill(seed, 72) > 0.0 for seed in seeds_order)
    b = all(delta(seed, 72) < 0.0 for seed in seeds_order)
    c = all(skill(seed, 48) > 0.0 for seed in seeds_order)
    means = {lead: sum(delta(seed, lead) for seed in seeds_order) / len(seeds_order)
             for lead in (6, 12, 24)}
    d = all(value <= 0.0 for value in means.values())
    passed = bool(a and b and c and d)
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d),
            "seed_mean_delta_6_12_24": {str(lead): value for lead, value in means.items()},
            "all": passed, "compute": {"updates": UPDATES, "batch_size": BATCH_SIZE},
            "verdict": "advance-to-independent-review" if passed
            else "close-position-retention-branch-at-matched-compute"}


def _readings(output, body):
    """The paired reading: five leads, both arms, three seeds, plus the exits and the bridge."""
    from training.r7_arm_harness import sha256_file
    receipts = {str(seed): {name: json.loads((output / f"seed{seed}_{name}_receipt.json")
                                             .read_text(encoding="utf-8"))
                            for name, _ in ARMS} for seed in SEEDS}
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": DECISION_RULE,
                "factor": body["factor"]["name"], "updates": UPDATES,
                "batch_size": BATCH_SIZE,
                "position_scale": POSITION_SCALE,
                "closed_round": {"record": CLOSED_ROUND_RECORD,
                                 "updates": CLOSED_ROUND_UPDATES,
                                 "batch_size": CLOSED_ROUND_BATCH_SIZE},
                "declared_difference": (
                    "both arms train on the same manifest at the same compute, share the same "
                    "migrated parent, install the same published 65x65 typed-evidence sidecar, "
                    "carry the same frozen position_scale and both start from an exactly zero "
                    "query; they differ only in the patch-position aggregation"),
                "climatology_reference_sha256": sha256_file(D2_REFERENCE), "seeds": {}}
    for seed in SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, _ in ARMS:
            arm = {"typed_evidence_mode": MODES[name], "full_t2m_rmse": {},
                   "full_t2m_skill": {}}
            for lead in EVALUATION_LEADS:
                table = Path(receipts[str(seed)][name]["evaluations"][str(lead)]["dir"]) \
                    / "boundary_rmse.csv"
                value = wide._region_variable_rmse(table, REGION, "t2m")
                denominator = body["climatology_reference"]["leads"][str(lead)]["t2m_rmse"]
                arm["full_t2m_rmse"][str(lead)] = value
                arm["full_t2m_skill"][str(lead)] = 1.0 - (value / denominator) ** 2
            entry["arms"][name] = arm
        entry["delta_candidate_minus_control"] = {str(lead): (
            entry["arms"][CANDIDATE]["full_t2m_rmse"][str(lead)]
            - entry["arms"][CONTROL]["full_t2m_rmse"][str(lead)]) for lead in EVALUATION_LEADS}
        readings["seeds"][str(seed)] = entry
    readings["exit_conditions"] = evaluate_exit(readings["seeds"])
    readings["bridge_e"] = bridge_verdict(receipts)
    readings["entropy"] = {str(seed): (receipts[str(seed)][CANDIDATE].get("entropy_check")
                                       or {}) for seed in SEEDS}
    return readings


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = _prepared_protocol(output)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]}", flush=True)
        return body
    if phase == "archive":
        body = dose.interior._archive(output, validate_protocol(output))
        print("[worker] archive done", flush=True)
        return body
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        print("[worker] reading done", flush=True)
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)
    print(f"[seed {seed}] done", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage-seeds`, not by main")
    return worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    main()
