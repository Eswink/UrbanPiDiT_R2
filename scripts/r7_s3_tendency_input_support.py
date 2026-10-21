"""Read-only control pinning and reading helpers for the tendency-input driver.

These functions were split out of ``scripts/study_r7_s3_tendency_input.py`` to keep that
driver inside R-051. They import the driver lazily and never run at import time, so CI
never needs ``outputs/`` (the tests monkeypatch every function that would).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scripts import r7_s3_decoder_base_anchor_support as pinned
from scripts import study_r7_s3_wide_single_factor as wide

ROOT = Path(__file__).resolve().parents[1]
all_variable_rmse = pinned.all_variable_rmse
table_readings = pinned.table_readings


def _driver():
    from scripts import study_r7_s3_tendency_input as driver
    return driver


def _climatology(body):
    return body["wide_climatology_reference"]["leads"]


def control_protocol():
    """Load the pinned H-D protocol and check it is the registered one."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    driver = _driver()
    record = driver.dose._index_record(driver.PAIRED_CONTROL_RECORD)
    if sha256_file(ROOT / record["evidence_path"]) != record["evidence_sha256"]:
        raise RuntimeError("the H-D result doc drifted from its index record")
    body = json.loads((driver.PAIRED_CONTROL_RUN / "protocol.json").read_text(encoding="utf-8"))
    if (body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items()
                                                      if k != "protocol_sha256"})
            or body["protocol_sha256"] != record["protocol_sha256"]):
        raise RuntimeError("the H-D protocol differs from its registered record")
    if body["scientific_claim"] is not False or body["test_read"] is not False:
        raise RuntimeError("the H-D control protocol is not report-only")
    if body["arms"]["control"]["anomaly_feedback"] is not False:
        raise RuntimeError("the H-D control arm is not the feedback-off arm")
    return record, body


def paired_control_pins(expected_identity=None, expected_source=None):
    """Pin the H-D control arm N: no tendency input, registered gradient, final weights."""
    from training.r7_arm_harness import sha256_file
    driver = _driver()
    record, protocol = control_protocol()
    if (expected_identity is not None and record["data_identity"] != expected_identity) or (
            expected_source is not None and record["source_sha256"] != expected_source):
        raise RuntimeError("the H-D control record data/source identity differs from this run")
    control_spec = protocol["arms"]["control"]["model"]["spec"]
    if "tendency_input" in control_spec:
        raise RuntimeError("the pinned H-D control already carries a tendency-input switch")
    if "decoder_base_mode" in control_spec:
        raise RuntimeError("the pinned H-D control spec is not the pre-decoder-switch spec")
    if control_spec != driver.control_spec():
        raise RuntimeError("the pinned H-D control spec is not exactly this round's control spec")
    if driver.arm_spec() != dict(control_spec, tendency_input=True):
        raise RuntimeError("the candidate spec is not the pinned control spec plus the one switch")
    control_training = protocol.get("training_code_sha256")
    if control_training != driver.CONTROL_TRAINING_CODE_SHA256:
        raise RuntimeError("the pinned H-D control did not run the predeclared training revision")
    if protocol.get("model_code_sha256") != driver.CONTROL_MODEL_CODE_SHA256:
        raise RuntimeError("the pinned H-D control did not run the predeclared model revision")
    readings_path = driver.PAIRED_CONTROL_RUN / "readings.json"
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    if (readings["scientific_claim"] is not False or readings["test_read"] is not False
            or readings["protocol_sha256"] != record["protocol_sha256"]):
        raise RuntimeError("the H-D control readings differ from the registered protocol")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "training_code_sha256": control_training,
            "model_code_sha256": driver.CONTROL_MODEL_CODE_SHA256,
            "updates": driver.UPDATES, "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
            "draft_supervision": driver.DRAFT_SUPERVISION, "tendency_input": False,
            "scored_checkpoint_kind": "final_optimizer_weights", "seeds": list(driver.SEEDS),
            "control_spec_canonical_digest": protocol["arms"]["control"]["model"].get(
                "spec_canonical_digest"),
            "run_dir": str(driver.PAIRED_CONTROL_RUN),
            "readings_sha256": sha256_file(readings_path), "seeds_readings": {}}
    for seed in driver.SEEDS:
        receipt = json.loads((driver.PAIRED_CONTROL_RUN / f"seed{seed}_control_receipt.json")
                             .read_text(encoding="utf-8"))
        report = json.loads((driver.PAIRED_CONTROL_RUN / f"seed{seed}/training/control/"
                             "training_report.json").read_text(encoding="utf-8"))
        prior = readings["seeds"][str(seed)]["arms"]["control"]
        if (receipt["arm"] != "control" or receipt["anomaly_feedback"] is not False
                or int(receipt["updates"]) != driver.UPDATES
                or int(receipt["reasoning_steps"]) != driver.REASONING_STEPS
                or receipt["parent_checkpoint_sha256"] != driver.anchor_arm.anchor_parent.MIGRATED_PARENT_SHA256[seed]
                or "ema_decay" in receipt or "ema_checkpoint" in receipt
                or Path(receipt["checkpoint"]).name != f"update_{driver.UPDATES:07d}.pt"):
            raise RuntimeError(f"the H-D seed {seed} control receipt is not the registered arm")
        declared = report["contract"]["autoregression"]
        if (report["contract"]["model"].get("decoder_base_mode", "persistence") != "persistence"
                or report["contract"]["model"].get("tendency_input", False) is not False
                or "gradient_surgery" in declared
                or "ema_decay" in report or "ema_checkpoint" in report
                or "ema_decay" in declared
                or report["objective"] != "deep_supervised_latitude_area_mse"
                or report["internal_deep_supervision"] is not True
                or int(report["contract"]["steps"]) != driver.REASONING_STEPS
                or declared["physical_weights"] != list(driver.REGISTERED_PHYSICAL_WEIGHTS)
                or declared.get("spatial_gradient_lambda") is not None):
            raise RuntimeError(f"the H-D seed {seed} control report is not the registered recipe")
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"],
                 "first_update_loss": float(report["losses"][0]["loss"]),
                 "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"], "leads": {}}
        for lead in wide.EVALUATION_LEADS:
            lead_record = receipt["evaluations"][str(lead)]
            folder = Path(lead_record["dir"])
            if sha256_file(folder / "boundary_rmse.csv") != lead_record["boundary_rmse_csv_sha256"]:
                raise RuntimeError(f"the H-D control table drifted for seed {seed} lead {lead}")
            if int(lead_record["n_evaluated"]) != wide.WIDE_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"the H-D control cohort differs for seed {seed} lead {lead}")
            entry["leads"][str(lead)] = {
                "dir": str(folder), "boundary_rmse_csv_sha256": lead_record["boundary_rmse_csv_sha256"],
                "n_evaluated": lead_record["n_evaluated"],
                "interior_t2m_rmse": prior["interior_t2m_rmse"][str(lead)],
                "interior_t2m_skill": prior["interior_t2m_skill"][str(lead)],
                "delta_minus_previous_dose": prior["delta_minus_previous_dose"][str(lead)]}
        pins["seeds_readings"][str(seed)] = entry
    return pins


def dose_readings(body, seed):
    """The pinned registered 2400-update interior-dose arm, verified against its pins."""
    return pinned.dose_readings(body, seed)


def control_readings(body, seed):
    """The pinned H-D control arm's RMSE/skill and full 17-variable table."""
    from training.r7_arm_harness import sha256_file
    pinned_arm = body["arms"]["paired_control_arm"]["seeds_readings"][str(seed)]
    climatology = _climatology(body)
    arm = {"interior_t2m_rmse": {}, "interior_t2m_skill": {},
           "delta_minus_previous_dose": {}, "all_variable_interior_rmse": {},
           "first_update_loss": pinned_arm["first_update_loss"],
           "tendency_input": body["arms"]["paired_control_arm"]["tendency_input"]}
    for lead in wide.EVALUATION_LEADS:
        entry = pinned_arm["leads"][str(lead)]
        folder = Path(entry["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != entry["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the pinned H-D control table drifted from its freeze pin")
        row = table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        if row["interior_t2m_rmse"] != entry["interior_t2m_rmse"]:
            raise RuntimeError("the recomputed H-D control RMSE differs from its readings.json")
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
        arm["delta_minus_previous_dose"][str(lead)] = entry["delta_minus_previous_dose"]
        arm["all_variable_interior_rmse"][str(lead)] = row["all_variable_interior_rmse"]
    return arm


def tendency_patch_summary(weight):
    """Descriptive statistics of the final tendency patch; never a pass threshold."""
    import torch
    weight = torch.as_tensor(weight)
    if tuple(weight.shape) != (192, 17, 2, 2):
        raise RuntimeError(f"the tendency patch shape {tuple(weight.shape)} is not 192x17x2x2")
    return {"numel": int(weight.numel()), "nonzero": int(torch.count_nonzero(weight)),
            "all_zero": int(torch.count_nonzero(weight)) == 0,
            "abs_max": float(weight.detach().abs().max()),
            "l2_norm": float(weight.detach().double().square().sum().sqrt())}


def candidate_readings(output, body, seed):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import load_checkpoint
    driver = _driver()
    receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    if (receipt["seed"] != seed
            or receipt["physical_weights"] != list(driver.REGISTERED_PHYSICAL_WEIGHTS)
            or receipt["draft_supervision"] != driver.DRAFT_SUPERVISION
            or receipt.get("tendency_input") is not True
            or "gradient_surgery" in receipt or "decoder_base_mode" in receipt
            or receipt.get("scored_checkpoint_kind") != "final_optimizer_weights"
            or not math.isfinite(float(receipt.get("first_update_loss", float("nan"))))
            or "ema_decay" in receipt or "ema_checkpoint" in receipt
            or Path(receipt.get("checkpoint", "")).name != f"update_{driver.UPDATES:07d}.pt"):
        raise RuntimeError("the candidate receipt is not the frozen tendency-input arm")
    saved = load_checkpoint(receipt["checkpoint"])
    measured = tendency_patch_summary(saved["model"][driver.parent.TENDENCY_KEY])
    stored = receipt.get("tendency_patch")
    if not isinstance(stored, dict):
        raise RuntimeError("the candidate receipt carries no tendency-patch summary")
    for key in ("numel", "nonzero", "all_zero"):
        if stored.get(key) != measured[key]:
            raise RuntimeError(f"the stored tendency-patch summary drifted: {key}")
    for key in ("abs_max", "l2_norm"):
        if not math.isclose(float(stored.get(key, float("nan"))), measured[key],
                            rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"the stored tendency-patch summary drifted: {key}")
    if saved["model"][driver.parent.TENDENCY_KEY].numel() != driver.TENDENCY_PARAMETERS:
        raise RuntimeError("the scored checkpoint's tendency patch is not the declared size")
    climatology = _climatology(body)
    arm = {"physical_weights": receipt["physical_weights"],
           "draft_supervision": receipt["draft_supervision"], "tendency_input": True,
           "scored_checkpoint_kind": receipt["scored_checkpoint_kind"],
           "first_update_loss": float(receipt["first_update_loss"]),
           "tendency_patch": measured, "expansion": receipt.get("expansion", {}),
           "interior_t2m_rmse": {}, "interior_t2m_skill": {}, "all_variable_interior_rmse": {}}
    for lead in wide.EVALUATION_LEADS:
        entry = receipt["evaluations"][str(lead)]
        folder = Path(entry["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != entry["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the candidate table drifted from its receipt")
        if entry["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
            raise RuntimeError("the candidate evaluation did not score the frozen checkpoint")
        row = table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
        arm["all_variable_interior_rmse"][str(lead)] = row["all_variable_interior_rmse"]
    return arm


def decision_flags(seeds):
    """The predeclared exit, computed from the reported readings only."""
    driver = _driver()

    def values(key, lead):
        return [seeds[str(seed)][key][str(lead)] for seed in driver.SEEDS]

    def deltas(lead):
        return values("delta_candidate_minus_control", lead)

    a = all(value > 0 for value in values("candidate_skill", 72))
    b = all(value < 0 for value in deltas(72))
    c = all(value > 0 for value in values("candidate_skill", 48))
    d = all(sum(deltas(lead)) / len(driver.SEEDS) <= 0 for lead in (6, 12, 24))
    bridge = [value for seed in driver.SEEDS
              for value in seeds[str(seed)]["control_minus_previous_dose"].values()]
    if len(bridge) != len(driver.SEEDS) * len(wide.EVALUATION_LEADS):
        raise RuntimeError("the bridge must cover every seed x lead")
    e = bool(bridge) and all(driver.BRIDGE_MIN <= value <= driver.BRIDGE_MAX for value in bridge)
    scientific = a and b and c and d
    return {"a_72h_skill_all_positive": a, "b_72h_delta_negative_all_seeds": b,
            "c_48h_skill_all_positive": c, "d_short_lead_seed_mean_not_worse": d,
            "e_bridge_consistent": e, "scientific_exit_met": scientific,
            "predeclared_exit_met": scientific and e, "identity_unresolved": not e}


def null_inert(seeds):
    """Null/inert exactly when the final patch is all zero or every paired delta is 0.0.

    No new threshold: the two branches are exact equalities at recorded precision, and a
    small nonzero effect still follows the (a)-(d) exit instead of being relabelled null.
    """
    driver = _driver()
    patches = [seeds[str(seed)]["candidate"]["tendency_patch"] for seed in driver.SEEDS]
    by_zero_weight = all(int(patch["nonzero"]) == 0 for patch in patches)
    deltas = [seeds[str(seed)]["delta_candidate_minus_control"][str(lead)]
              for seed in driver.SEEDS for lead in wide.EVALUATION_LEADS]
    by_zero_deltas = bool(deltas) and all(value == 0.0 for value in deltas)
    return {"all_seed_tendency_patch_zero": by_zero_weight,
            "all_paired_deltas_exactly_zero": by_zero_deltas,
            "nonzero_counts": [int(patch["nonzero"]) for patch in patches],
            "null_or_inert": bool(by_zero_weight or by_zero_deltas),
            "note": ("null/inert only on an exact all-zero final patch or exactly zero paired "
                     "deltas at recorded precision; a small nonzero effect follows (a)-(d) "
                     "and is never relabelled null")}


def readings(output, body):
    driver = _driver()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
              "updates": driver.UPDATES, "dim": driver.DIM,
              "reasoning_steps": driver.REASONING_STEPS,
              "factor": "model input representation: explicit 6 h tendency input off -> on",
              "tendency_input": body["tendency_input"],
              "declared_difference": body["difference"],
              "code_version_comparison": body.get("code_version_comparison"),
              "control_note": ("the control is the pinned previous-code "
                               "s3-anchor-isolation-result arm without a tendency input; it is "
                               "not retrained"),
              "seeds": {}}
    for seed in driver.SEEDS:
        candidate = candidate_readings(output, body, seed)
        control = control_readings(body, seed)
        registered = dose_readings(body, seed)
        result["seeds"][str(seed)] = {
            "seed": seed, "candidate": candidate, "paired_control": control,
            "registered_dose": registered, "candidate_skill": candidate["interior_t2m_skill"],
            "delta_candidate_minus_control": {
                str(lead): candidate["interior_t2m_rmse"][str(lead)]
                - control["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS},
            "delta_candidate_minus_registered_dose": {
                str(lead): candidate["interior_t2m_rmse"][str(lead)]
                - registered["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS},
            "control_minus_previous_dose": {
                str(lead): control["interior_t2m_rmse"][str(lead)]
                - registered["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS},
            "update0_diagnostic": {
                "candidate_loss": candidate["first_update_loss"],
                "control_loss": control["first_update_loss"],
                "candidate_minus_control": (candidate["first_update_loss"]
                                            - control["first_update_loss"]),
                "note": ("diagnostic only; the same parent and first synthetic-order batch, and "
                         "the tendency patch starts at exactly zero, never a gate")}}
    result["decision"] = decision_flags(result["seeds"])
    result["tendency_outcome"] = null_inert(result["seeds"])
    return result
