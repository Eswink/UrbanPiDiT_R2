"""Read-only control pinning and reading helpers for the decoder-base-anchor driver.

These functions were split out of ``scripts/study_r7_s3_decoder_base_anchor.py`` to keep
that driver inside R-051. They import the driver lazily and never run at import time, so
the driver can import this module without a cycle and CI never needs ``outputs/``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scripts import study_r7_s3_anchor_isolation as anchor_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    from scripts import study_r7_s3_decoder_base_anchor as driver
    return driver


def control_protocol():
    """Load the pinned H-D protocol and check it is the registered one."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    driver = _driver()
    record = dose._index_record(driver.PAIRED_CONTROL_RECORD)
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
    """Pin the H-D control arm N and verify it is the persistence-base arm."""
    from training.r7_arm_harness import sha256_file
    driver = _driver()
    record, protocol = control_protocol()
    if (expected_identity is not None and record["data_identity"] != expected_identity) or (
            expected_source is not None and record["source_sha256"] != expected_source):
        raise RuntimeError("the H-D control record data/source identity differs from this run")
    control_spec = protocol["arms"]["control"]["model"]["spec"]
    if control_spec.get("decoder_base_mode", driver.CONTROL_DECODER_BASE_MODE) \
            != driver.CONTROL_DECODER_BASE_MODE:
        raise RuntimeError("the pinned H-D control is not the persistence-base arm")
    if driver._without_mode(control_spec) != driver._without_mode(
            driver.arm_spec(driver.CONTROL_DECODER_BASE_MODE)):
        raise RuntimeError("the pinned control spec differs from this round's candidate in more "
                           "than the decoder base mode")
    if driver.arm_spec() != dict(control_spec, decoder_base_mode=driver.CANDIDATE_DECODER_BASE_MODE):
        raise RuntimeError("the candidate spec is not the pinned control spec plus the one switch")
    readings_path = driver.PAIRED_CONTROL_RUN / "readings.json"
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    if (readings["scientific_claim"] is not False or readings["test_read"] is not False
            or readings["protocol_sha256"] != record["protocol_sha256"]):
        raise RuntimeError("the H-D control readings differ from the registered protocol")
    control_training = protocol.get("training_code_sha256")
    if not isinstance(control_training, str) or len(control_training) != 64:
        raise RuntimeError("the H-D protocol carries no training-code digest")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "training_code_sha256": control_training,
            "updates": driver.UPDATES, "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
            "draft_supervision": driver.DRAFT_SUPERVISION,
            "decoder_base_mode": driver.CONTROL_DECODER_BASE_MODE,
            "control_spec_canonical_digest": protocol["arms"]["control"]["model"].get(
                "spec_canonical_digest"),
            "seeds": list(driver.SEEDS), "run_dir": str(driver.PAIRED_CONTROL_RUN),
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
                or receipt["parent_checkpoint_sha256"] != anchor_arm.anchor_parent.MIGRATED_PARENT_SHA256[seed]
                or "ema_decay" in receipt or "ema_checkpoint" in receipt
                or Path(receipt["checkpoint"]).name != f"update_{driver.UPDATES:07d}.pt"):
            raise RuntimeError(f"the H-D seed {seed} control receipt is not the persistence arm")
        recorded = report["contract"]["model"].get("decoder_base_mode",
                                                  driver.CONTROL_DECODER_BASE_MODE)
        if (recorded != driver.CONTROL_DECODER_BASE_MODE
                or report["objective"] != "deep_supervised_latitude_area_mse"
                or report["internal_deep_supervision"] is not True
                or int(report["contract"]["steps"]) != driver.REASONING_STEPS
                or report["contract"]["autoregression"]["physical_weights"]
                != list(driver.REGISTERED_PHYSICAL_WEIGHTS)
                or report["contract"]["autoregression"].get("spatial_gradient_lambda") is not None
                or "ema_decay" in report or "ema_checkpoint" in report):
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


def all_variable_rmse(path):
    """Every variable's interior_32 RMSE, for the required 17-variable report."""
    import csv
    values = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] != "interior_32":
                continue
            name = row["variable"]
            if name in values:
                raise RuntimeError(f"duplicate interior_32 row for {name}")
            value = float(row["rmse"])
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite interior_32 RMSE for {name}")
            values[name] = value
    if len(values) != 17:
        raise RuntimeError(f"expected 17 interior_32 variables, found {len(values)}")
    return values


def table_readings(folder, climatology):
    """RMSE, skill and the full 17-variable table for one pinned/new evaluation folder."""
    folder = Path(folder)
    rmse = wide._region_variable_rmse(folder / "boundary_rmse.csv", "interior_32", "t2m")
    return {"interior_t2m_rmse": rmse,
            "interior_t2m_skill": 1.0 - (rmse / climatology) ** 2,
            "all_variable_interior_rmse": all_variable_rmse(folder / "boundary_rmse.csv")}


def dose_readings(body, seed):
    from training.r7_arm_harness import sha256_file
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"interior_t2m_rmse": {}, "interior_t2m_skill": {}, "all_variable_interior_rmse": {}}
    for lead in wide.EVALUATION_LEADS:
        previous = body["arms"]["previous_dose_arm"]["seeds_evaluations"][str(seed)]["leads"][str(lead)]
        folder = Path(previous["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != previous["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the registered 2400-update dose arm table drifted from its pin")
        row = table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
        arm["all_variable_interior_rmse"][str(lead)] = row["all_variable_interior_rmse"]
    return arm


def control_readings(body, seed):
    from training.r7_arm_harness import sha256_file
    pinned = body["arms"]["paired_control_arm"]["seeds_readings"][str(seed)]
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"interior_t2m_rmse": {}, "interior_t2m_skill": {},
           "delta_minus_previous_dose": {}, "all_variable_interior_rmse": {},
           "first_update_loss": pinned["first_update_loss"],
           "decoder_base_mode": body["arms"]["paired_control_arm"]["decoder_base_mode"]}
    for lead in wide.EVALUATION_LEADS:
        entry, folder = pinned["leads"][str(lead)], Path(pinned["leads"][str(lead)]["dir"])
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


def candidate_readings(output, body, seed):
    from training.r7_arm_harness import sha256_file
    driver = _driver()
    receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    if (receipt["seed"] != seed
            or receipt["physical_weights"] != list(driver.REGISTERED_PHYSICAL_WEIGHTS)
            or receipt["draft_supervision"] != driver.DRAFT_SUPERVISION
            or receipt.get("decoder_base_mode") != driver.CANDIDATE_DECODER_BASE_MODE
            or receipt.get("scored_checkpoint_kind") != "final_optimizer_weights"
            or not math.isfinite(float(receipt.get("first_update_loss", float("nan"))))
            or "ema_decay" in receipt or "ema_checkpoint" in receipt
            or Path(receipt.get("checkpoint", "")).name != f"update_{driver.UPDATES:07d}.pt"):
        raise RuntimeError("the candidate receipt is not the frozen zero-base arm")
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"physical_weights": receipt["physical_weights"],
           "draft_supervision": receipt["draft_supervision"],
           "decoder_base_mode": receipt["decoder_base_mode"],
           "scored_checkpoint_kind": receipt["scored_checkpoint_kind"],
           "first_update_loss": float(receipt["first_update_loss"]),
           "interior_t2m_rmse": {}, "interior_t2m_skill": {}, "all_variable_interior_rmse": {}}
    for lead in wide.EVALUATION_LEADS:
        entry = receipt["evaluations"][str(lead)]; folder = Path(entry["dir"])
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


def readings(output, body):
    driver = _driver()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
              "updates": driver.UPDATES, "dim": driver.DIM,
              "reasoning_steps": driver.REASONING_STEPS,
              "factor": "decoder base semantics: persistence -> zero",
              "decoder_base": body["decoder_base"],
              "draft_supervision_organization": body["draft_supervision_organization"],
              "declared_difference": body["difference"],
              "code_version_comparison": body.get("code_version_comparison"),
              "budget_profile_note": body.get("budget_profile_note"),
              "control_note": ("the control is the pinned previous-code "
                               "s3-anchor-isolation-result arm; the new-code persistence path is "
                               "only asserted bitwise-equivalent and the control is not retrained"),
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
                "note": "diagnostic only; the same parent and the same first synthetic-order "
                        "batch, never a gate"}}
    result["decision"] = decision_flags(result["seeds"])
    return result
