"""Long-lead physical-weight rebalance at a fixed 2400 updates.

The registered dose arm (``docs/R7_S3_WIDE_INTERIOR_DOSE.md``, index record
``s3-wide-interior-dose``) pushed the 48 h interior_32 t2m skill positive for all
three seeds but left 72 h negative for all three. Update count is therefore not
the only remaining lever, and the predeclared anti-spin rule closes the dose
branch rather than piling on more updates. This driver keeps the wide 129x129
input, the interior_32 supervision mask, the 2400-update budget, the registered
recipe, the migrated v3-BD 1600 parent, the seeds, the evaluation region and the
climatology denominator fixed and changes exactly one factor: the physical
weights on the 48 h and 72 h steps, 0.5 -> 1.0, so the long leads stop being
down-weighted relative to 12 h and 24 h. The pinned reference is the registered
2400 arm; neither it nor the narrow arm is retrained. The 2023 test split is
never read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_interior_supervision as interior
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training.r7_study_process import bounded_process, worker_environment

FORMAT = "r7-s3-wide-long-lead-weight-protocol-v1"
UPDATES = 2400
# Registered weights down-weight every non-initial lead equally at 0.5; this arm
# lifts the two longest leads to the initial step's weight of 1.0 and nothing else.
PHYSICAL_WEIGHTS = (1., .5, 0., .5, 0., 0., 0., 1., 0., 0., 0., 1.)
REGISTERED_WEIGHTS = (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 12600., 36000., 54000.
DEFAULT_OUT = ROOT / "outputs/r7_s3_wide_long_lead_weight_20261009_attempt01"
PREVIOUS_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
EVIDENCE_INDEX = ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"
PREVIOUS_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*dose.ENTRY_FILES, "scripts/study_r7_s3_wide_long_lead_weight.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the physical weights on the two longest leads. The registered dose arm reached 48 h "
    "interior_32 t2m skill positive for all three seeds but 72 h stayed negative for all three, and the "
    "predeclared anti-spin rule closes the dose branch instead of adding updates. The registered 12-step "
    "weights are (1,.5,0,.5,0,0,0,.5,0,0,0,.5): 48 h and 72 h carry the same 0.5 weight as 12 h and 24 h. "
    "This arm raises only those two to 1.0, holding the 2400-update budget and every other factor fixed.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report the per-seed paired delta of this arm's "
    "interior_32 t2m RMSE minus (a) the registered 2400-update interior arm's and (b) the registered "
    "narrow 800-update arm's interior_32/full t2m RMSE, plus this arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. Both references are pinned, not "
    "retrained here.")
HYPOTHESIS_TEXT = (
    "If the 72 h residual is dominated by the long leads being trained with the same 0.5 weight as the "
    "middle leads, then lifting 48 h and 72 h to 1.0 at a fixed 2400 updates should raise 72 h skill "
    "toward the 48 h level. If 72 h stays negative for any seed, the weight lever is exhausted for this "
    "branch and the whole wide-input branch closes.")
DIFFERENCE_TEXT = (
    "One factor only: physical weights on the 48 h and 72 h steps, 0.5 -> 1.0. Same wide 129x129 input, "
    "same interior_32 supervision mask (mask_sha256 4a2a576b...), same 2400 updates, same registered "
    "long-rollout recipe, same migrated v3-BD 1600 parent, same LR 2e-5, warmup 10, FP32, K4, batch 1 "
    "and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "both references are pinned registered arms (narrow s3-rollout-dose, 2400-update interior "
    "s3-wide-interior-dose), not retrained inside this protocol",
    "raising the long-lead weights also rescales the total loss magnitude, so the effect is not a pure "
    "reallocation; the learning rate is held at the registered value",
    "the parent was trained on the narrow instance under full-grid supervision; the optimization "
    "landscape difference is not separated",
    "restricting supervision to interior_32 leaves the outer ring unconstrained during training",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(dose.execution_files()) | set(ENTRY_FILES))


def protocol_constants():
    body = dose.protocol_constants()
    body["format"] = FORMAT
    body["recipe"]["physical_weights"] = list(PHYSICAL_WEIGHTS)
    body["design_decision"] = DESIGN_DECISION_TEXT
    body["decision_rule"] = DECISION_RULE_TEXT
    body["hypothesis"] = HYPOTHESIS_TEXT
    body["difference"] = DIFFERENCE_TEXT
    body["limitations"] = list(LIMITATIONS)
    return body


def _index_record(record_id):
    for line in EVIDENCE_INDEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("record_id") == record_id:
                return record
    raise FileNotFoundError(f"evidence index record missing: {record_id}")


def previous_arm_pins():
    """Pin the registered 2400-update interior arm this round is reweighted against."""
    from training.r7_arm_harness import sha256_file
    record = _index_record(PREVIOUS_ARM_RECORD)
    doc = ROOT / record["evidence_path"]
    if sha256_file(doc) != record["evidence_sha256"]:
        raise RuntimeError("the registered 2400-update arm doc drifted from its index record")
    run = PREVIOUS_ARM_RUN
    if not run.is_dir():
        raise FileNotFoundError(f"the registered 2400-update arm output is missing: {run}")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "updates": UPDATES, "seeds": list(wide.SEEDS), "run_dir": str(run), "seeds_evaluations": {}}
    for seed in wide.SEEDS:
        receipt = json.loads((run / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"], "leads": {}}
        for lead in wide.EVALUATION_LEADS:
            record_lead = receipt["evaluations"][str(lead)]
            if record_lead["n_evaluated"] != wide.WIDE_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"registered 2400 arm seed {seed} lead {lead} cohort differs")
            entry["leads"][str(lead)] = {
                "boundary_rmse_csv_sha256": record_lead["boundary_rmse_csv_sha256"],
                "dir": record_lead["dir"]}
        pins["seeds_evaluations"][str(seed)] = entry
    return pins


def _identity():
    from training.r7_arm_harness import sha256_file
    body = wide._identity()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    body["supervision"] = interior.supervision_block()
    body["arms"]["candidate"]["updates"] = UPDATES
    body["arms"]["candidate"]["physical_weights"] = list(PHYSICAL_WEIGHTS)
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**protocol_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": 2**31, "headroom_margin_bytes": 2**31,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "preparation_protocol.json", body)
    return body


def _prepared_protocol(output):
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest({k: v for k, v in preliminary.items()
                                                           if k != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def _expected_candidate():
    from training.r7_experiment import canonical_digest
    spec, _, _ = recipe._shared().archived_process_spec()
    return {"mode": "long_rollout", "updates": UPDATES, "lr": wide.LR, "warmup": wide.WARMUP,
            "physical_weights": list(PHYSICAL_WEIGHTS), "steps": wide.STEPS,
            "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE, "clip": wide.CLIP,
            "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
            "selection": "frozen endpoint; no validation selection",
            "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}


def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items()
                                                    if k != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    for split, manifest in (("train", wide.TRAIN_MANIFEST), ("val", wide.VAL_MANIFEST)):
        if (body[split + "_manifest"] != str(manifest)
                or dataset_identity(manifest)[0] != body[split + "_data_identity"]):
            raise RuntimeError("dataset identity changed")
    if (model_code_digest() != body["model_code_sha256"]
            or training_code_digest() != body["training_code_sha256"]):
        raise RuntimeError("model/training source identity changed")
    if body["source_sha256"] != wide.source_sha256():
        raise RuntimeError("the wide store source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(wide.TRAIN_MANIFEST, physical_steps=12):
        raise RuntimeError("train-window metadata changed")
    if body["supervision"] != interior.supervision_block() or body["format"] != FORMAT:
        raise RuntimeError("the frozen supervision factor changed")
    if body["arms"]["candidate"] != _expected_candidate() or body["gpu_uuid"] != wide.GPU_UUID:
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    if list(body["recipe"]["physical_weights"]) != list(PHYSICAL_WEIGHTS):
        raise RuntimeError("the frozen long-lead weights changed")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    narrow = body["arms"]["narrow_control"]
    previous = body["arms"]["previous_dose_arm"]
    if (narrow["record_id"] != wide.NARROW_CONTROL_RECORD or narrow["updates"] != 800
            or narrow["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned narrow control differs from the registered record")
    if (previous["record_id"] != PREVIOUS_ARM_RECORD or previous["updates"] != UPDATES
            or previous["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned 2400-update arm differs from the registered record")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body


def _seed(output, body, seed, deadline):
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    mask = interior.supervision_mask().to("cuda:0")
    model, state, parent, parent_sha = wide._parent_model(body, seed)
    spec = body["arms"]["candidate"]["model"]["spec"]
    contract = {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
                "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
                "training_code_sha256": body["training_code_sha256"],
                "initialization": {"parent_run": str(wide.dose.MIGRATED_PARENT_RUN),
                                   "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                                   "parent_endpoint_updates": 1600,
                                   "description": ("registered v3-BD l6 endpoint, model-only migration; "
                                                   "region-agnostic spec reused on the wide store")},
                "autoregression": {"physical_steps": 12, "physical_weights": list(PHYSICAL_WEIGHTS),
                    "window_sha256": body["train_windows"]["window_sha256"],
                    "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                    "supervision": body["supervision"]["runner_block"]}}
    folder = output / f"seed{seed}/training/candidate"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model, contract=contract,
        parent_weights=state, physical_weights=PHYSICAL_WEIGHTS, updates=UPDATES, lr=wide.LR,
        warmup=wide.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        supervision_mask=mask)
    if (report["updates_this_run"] != UPDATES or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]
            or list(report["contract"]["autoregression"]["physical_weights"]) != list(PHYSICAL_WEIGHTS)):
        raise RuntimeError(f"seed {seed} did not reach the frozen endpoint under the frozen weights")
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES,
               "physical_weights": list(PHYSICAL_WEIGHTS), "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h"
        provenance = evaluate_local(wide.VAL_MANIFEST, output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=wide.REASONING_STEPS, boundary_margins=wide.BOUNDARY_MARGINS,
            deadline=deadline)
        wide._check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    wide.write_exclusive(output / f"seed{seed}_receipt.json", receipt)


def _pinned_rmse(body, arm, seed, lead, filename, key):
    from training.r7_arm_harness import sha256_file
    entry = body["arms"][arm]["seeds_evaluations"][str(seed)]["leads"][str(lead)]
    folder = Path(entry["dir"])
    if sha256_file(folder / filename) != entry[key]:
        raise RuntimeError(f"the registered {arm} arm {filename} drifted from its freeze pin")
    return folder


def _readings(output, body):
    climatology = body["wide_climatology_reference"]
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
                "updates": UPDATES, "physical_weights": list(PHYSICAL_WEIGHTS), "seeds": {}}
    for seed in wide.SEEDS:
        receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"seed": seed, "interior_t2m_rmse": {}, "narrow_full_t2m_rmse": {},
                 "delta_minus_narrow": {}, "previous_dose_t2m_rmse": {},
                 "delta_minus_previous_dose": {}, "interior_t2m_skill": {}}
        for lead in wide.EVALUATION_LEADS:
            wide_dir = Path(receipt["evaluations"][str(lead)]["dir"])
            wide_rmse = wide._region_variable_rmse(wide_dir / "boundary_rmse.csv", "interior_32", "t2m")
            narrow_dir = _pinned_rmse(body, "narrow_control", seed, lead, "rmse.csv", "rmse_csv_sha256")
            previous_dir = _pinned_rmse(body, "previous_dose_arm", seed, lead, "boundary_rmse.csv",
                                        "boundary_rmse_csv_sha256")
            narrow_rmse = recipe._shared().read_rmse_table(narrow_dir / "rmse.csv")["t2m"]
            previous_rmse = wide._region_variable_rmse(previous_dir / "boundary_rmse.csv",
                                                       "interior_32", "t2m")
            climatology_rmse = climatology["leads"][str(lead)]["interior_32_t2m_rmse"]
            entry["interior_t2m_rmse"][str(lead)] = wide_rmse
            entry["narrow_full_t2m_rmse"][str(lead)] = narrow_rmse
            entry["delta_minus_narrow"][str(lead)] = wide_rmse - narrow_rmse
            entry["previous_dose_t2m_rmse"][str(lead)] = previous_rmse
            entry["delta_minus_previous_dose"][str(lead)] = wide_rmse - previous_rmse
            entry["interior_t2m_skill"][str(lead)] = 1.0 - (wide_rmse / climatology_rmse) ** 2
        readings["seeds"][str(seed)] = entry
    return readings


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return interior._archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(wide.GPU_UUID)
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        return
    if seed not in wide.SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)


def run_attempt(output, *, code_archive, code_commit, code_sha256):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = fresh_output(output)
    if not Path(wide.STORE).is_dir():
        raise FileNotFoundError(f"the wide store is not built yet: {wide.STORE}")
    if (not code_archive or not isinstance(code_commit, str) or len(code_commit) != 40
            or not isinstance(code_sha256, str) or len(code_sha256) != 64):
        raise ValueError("prepared exact code archive/commit/SHA required before execution")
    started = time.perf_counter()
    deadline = started + HARD_CAP_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    processes, receipts = [], {}
    try:
        freeze(output, started=started, deadline=deadline, code_archive=Path(code_archive).absolute(),
               code_commit=code_commit, code_sha256=code_sha256)
        phases = [("prepare", None), ("archive", None)] + [("seed", s) for s in wide.SEEDS] \
            + [("reading", None)]
        peak, body = 0, None
        for phase, seed in phases:
            gate = recipe._shared().gpu_gate(wide.GPU_UUID, estimated_peak_bytes=2**31,
                                            owned_reserved_peak_bytes=peak, margin_bytes=2**31)
            tag = phase if seed is None else f"seed{seed}"
            wide.write_exclusive(output / f"{tag}_spawn_gate.json", gate)
            limit = min(deadline, time.perf_counter() + PER_SEED_SECONDS) if phase == "seed" else deadline
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                       "--out", str(output), "--deadline", str(limit)]
            if seed is not None:
                command += ["--seed", str(seed)]
            process = bounded_process(command, cwd=ROOT, log_path=output / f"{tag}.log", deadline=limit,
                                      env=worker_environment(wide.GPU_UUID))
            processes.append({"phase": phase, "seed": seed, **process})
            wide.write_exclusive(output / f"{tag}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(f"{tag} failed: {process['status']}")
            if phase == "prepare":
                body = json.loads((output / "prepared_protocol.json").read_text(encoding="utf-8"))
                wide.write_exclusive(output / "protocol.json", body)
            elif phase == "seed":
                receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
                receipts[str(seed)] = receipt
                peak = max(peak, receipt["owned_cuda_reserved_peak_bytes"])
            print(json.dumps({"phase_done": phase, "seed": seed,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        elapsed = time.perf_counter() - started
        if elapsed >= deadline:
            raise RuntimeError("round hard deadline passed before publication")
        result = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
                  "supervision": body["supervision"], "updates": UPDATES,
                  "physical_weights": list(PHYSICAL_WEIGHTS),
                  "train_data_identity": body["train_data_identity"],
                  "val_data_identity": body["val_data_identity"], "elapsed_seconds_total": elapsed,
                  "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
                  "hard_overrun_seconds": max(0., elapsed - HARD_CAP_SECONDS), "processes": processes,
                  "seeds": receipts, "limitations": body["limitations"]}
        result.update(json.loads((output / "readings.json").read_text(encoding="utf-8")))
        wide.write_exclusive(output / "result.json", result)
        wide.write_exclusive(output / "attempt.json", {"status": "complete", "scientific_claim": False,
            "test_read": False, "protocol_sha256": body["protocol_sha256"], "elapsed_seconds_total": elapsed,
            "decision_rule": body["decision_rule"]})
        return result
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        wide.write_exclusive(output / "failure.json", {"status": "failed", "scientific_claim": False,
            "test_read": False, "elapsed_seconds_total": elapsed,
            "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
            "hard_overrun_seconds": max(0., elapsed - HARD_CAP_SECONDS), "processes": processes,
            "seeds_completed": list(receipts), "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "limitations": ["Failed attempt preserved and charged in full; no automatic retry or in-place resume"]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--code-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.out, args.worker, args.seed, args.deadline)
    return run_attempt(args.out or DEFAULT_OUT, code_archive=args.code_archive,
                       code_commit=args.code_commit, code_sha256=args.code_sha256)


if __name__ == "__main__":
    main()
