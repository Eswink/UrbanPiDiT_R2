"""Interior-only supervision on the wide 129x129 store: the dilution control.

The registered wide single-factor arm (`docs/R7_S3_WIDE_SINGLE_FACTOR.md`) supervised
the full 129x129 grid and read neutral-to-worse on the frozen central 65x65 box at
every lead, so information gain was confounded with supervision dilution (3.94x the
narrow arm's supervision area). This driver keeps the wide 129x129 *input* and moves
the *supervision* back onto the frozen box: the loss is restricted to
``boundary_masks(129,129,(32,))['interior_32']``, which is exactly the region the
registered narrow 65x65 arm supervises. The recipe, parent state, seeds, evaluation
region, climatology denominator and cohort pins are the registered ones, reused from
``scripts/study_r7_s3_wide_single_factor.py``. A gain over the pinned narrow arm then
supports "the frozen box lacks long-lead information" without the dilution confound;
a null/negative reading leaves that hypothesis unsupported under a stronger control.
The 2023 test split is never read.
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

from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training.r7_study_process import bounded_process, worker_environment

FORMAT = "r7-s3-wide-interior-supervision-protocol-v1"
SUPERVISION_KIND = "interior_32"
DEFAULT_OUT = ROOT / "outputs/r7_s3_wide_interior_supervision_20261009_attempt01"
ENTRY_FILES = (*wide.ENTRY_FILES, "scripts/study_r7_s3_wide_interior_supervision.py")

DESIGN_DECISION_TEXT = (
    "Single factor: supervision domain, with the input region held at the wide 129x129. The registered "
    "800-update long-rollout recipe is applied to the wide store exactly as in the previous round, but "
    "the loss is restricted to boundary_masks(129,129,(32,))['interior_32'] - the frozen central 65x65 "
    "box, which is exactly the region the registered narrow 65x65 arm supervises. The previous round's "
    "full-grid arm confounded information gain with a 3.94x larger supervision area; this arm removes "
    "that confound, so the only difference from the pinned narrow arm is the input extent.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of this arm's "
    "interior_32 t2m RMSE minus the registered narrow arm's (index record s3-rollout-dose, 800 updates, "
    "seeds 41/42/43) full-region t2m RMSE at the five leads {6,12,24,48,72} h, and (b) this arm's "
    "interior_32 t2m skill 1-(rmse_wide/rmse_climatology)**2 against the wide train-only (2017-2021) "
    "climatology. The narrow arm is a pinned reference, not retrained here.")
HYPOTHESIS_TEXT = (
    "If the frozen 16-degree box lacked the information long leads need, then a wide 129x129 input whose "
    "supervision is confined to that same box - the same supervision domain as the narrow arm - should "
    "still beat the narrow arm, because the extra context is available to the model but not required to "
    "be predicted. The previous round's full-grid arm could not test this, since its supervision area "
    "was 3.94x larger.")
DIFFERENCE_TEXT = (
    "Two changes against the registered narrow arm: the input region (65x65 -> 129x129) and nothing "
    "else - the supervision region is the frozen central 65x65 box, which equals the narrow arm's own "
    "65x65 grid. Against the previous wide round the only change is the supervision region (full "
    "129x129 -> interior_32). Same recipe, same migrated v3-BD 1600 parent, same LR 2e-5, warmup 10, "
    "12-step physical weights, FP32, K4, batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "the narrow control is a pinned registered reference (docs/R7_S3_ROLLOUT_DOSE.md / index record "
    "s3-rollout-dose), not retrained inside this protocol",
    "the parent state was trained on the narrow 65x65 instance under full-grid supervision and is reused "
    "here because the model spec is region-agnostic; recorded as a protocol note",
    "restricting supervision to interior_32 also removes the outer ring from the loss, so the outer ring "
    "is unconstrained during training; only interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(wide.execution_files()) | set(ENTRY_FILES))


def supervision_block():
    """The frozen interior_32 mask, described and digested exactly as the runner will."""
    from training.r7_long_rollout_runner import _supervision_block
    return {"kind": SUPERVISION_KIND, "runner_block": _supervision_block(supervision_mask())}


def supervision_mask():
    from training.r7_boundary_metrics import boundary_masks
    masks = {name: mask for name, _, mask in boundary_masks(129, 129, wide.BOUNDARY_MARGINS)}
    mask = masks["interior_32"]
    if int(mask.sum()) != 4225 or mask.dtype is not __import__("torch").bool:
        raise RuntimeError("the frozen interior_32 mask is not the 4225-point central box")
    return mask


def protocol_constants():
    """The round-2 frozen fields with the supervision factor swapped in."""
    body = wide.protocol_constants()
    body["format"] = FORMAT
    body["design_decision"] = DESIGN_DECISION_TEXT
    body["decision_rule"] = DECISION_RULE_TEXT
    body["hypothesis"] = HYPOTHESIS_TEXT
    body["difference"] = DIFFERENCE_TEXT
    body["limitations"] = list(LIMITATIONS)
    body["supervision"] = supervision_block()
    return body


def _identity():
    from training.r7_arm_harness import sha256_file
    body = wide._identity()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    body["supervision"] = supervision_block()
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**protocol_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": 2**31, "headroom_margin_bytes": 2**31,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": wide.PLANNED_SECONDS, "hard_cap_seconds": wide.HARD_CAP_SECONDS,
            "per_seed_seconds": wide.PER_SEED_SECONDS,
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
    if body["supervision"] != supervision_block() or body["format"] != FORMAT:
        raise RuntimeError("the frozen supervision factor changed")
    if body["arms"]["candidate"] != _expected_candidate() or body["gpu_uuid"] != wide.GPU_UUID:
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    if (body["planned_seconds"] != wide.PLANNED_SECONDS
            or body["hard_cap_seconds"] != wide.HARD_CAP_SECONDS
            or body["per_seed_seconds"] != wide.PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    narrow = body["arms"]["narrow_control"]
    if (narrow["record_id"] != wide.NARROW_CONTROL_RECORD or narrow["updates"] != wide.UPDATES
            or narrow["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned narrow control differs from the registered record")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body


def _expected_candidate():
    from training.r7_experiment import canonical_digest
    spec, _, _ = recipe._shared().archived_process_spec()
    return {"mode": "long_rollout", "updates": wide.UPDATES, "lr": wide.LR, "warmup": wide.WARMUP,
            "physical_weights": list(wide.PHYSICAL_WEIGHTS), "steps": wide.STEPS,
            "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE, "clip": wide.CLIP,
            "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
            "selection": "frozen endpoint; no validation selection",
            "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}


def _seed(output, body, seed, deadline):
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    mask = supervision_mask().to("cuda:0")
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
                "autoregression": {"physical_steps": 12, "physical_weights": list(wide.PHYSICAL_WEIGHTS),
                    "window_sha256": body["train_windows"]["window_sha256"],
                    "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                    "supervision": body["supervision"]["runner_block"]}}
    folder = output / f"seed{seed}/training/candidate"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model, contract=contract,
        parent_weights=state, physical_weights=wide.PHYSICAL_WEIGHTS, updates=wide.UPDATES, lr=wide.LR,
        warmup=wide.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        supervision_mask=mask)
    if (report["updates_this_run"] != wide.UPDATES
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]):
        raise RuntimeError(f"seed {seed} did not reach the frozen endpoint under the frozen supervision")
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "evaluations": {}}
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


def _archive(output, body):
    import hashlib
    from io import BytesIO
    import zipfile
    from training.r7_arm_harness import sha256_file
    archive = Path(body["code_archive_input"])
    if any(p.is_symlink() for p in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs from freeze")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError("archive omits or changes execution source")
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    wide.write_exclusive(output / "archive_receipt.json", {"scientific_claim": False,
        "code_commit": body["code_commit"], "code_zip_sha256": sha256_file(output / "code.zip")})


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return _archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(wide.GPU_UUID)
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", wide._readings(output, body))
        return
    if seed not in wide.SEEDS or deadline - time.perf_counter() > wide.PER_SEED_SECONDS:
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
    deadline = started + wide.HARD_CAP_SECONDS
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
            limit = min(deadline, time.perf_counter() + wide.PER_SEED_SECONDS) if phase == "seed" else deadline
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
                  "supervision": body["supervision"], "train_data_identity": body["train_data_identity"],
                  "val_data_identity": body["val_data_identity"], "elapsed_seconds_total": elapsed,
                  "soft_overrun_seconds": max(0., elapsed - wide.PLANNED_SECONDS),
                  "hard_overrun_seconds": max(0., elapsed - wide.HARD_CAP_SECONDS), "processes": processes,
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
            "soft_overrun_seconds": max(0., elapsed - wide.PLANNED_SECONDS),
            "hard_overrun_seconds": max(0., elapsed - wide.HARD_CAP_SECONDS), "processes": processes,
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
