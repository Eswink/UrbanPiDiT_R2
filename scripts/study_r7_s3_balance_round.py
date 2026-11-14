"""The balance-diagnostics round driver: one worker per seed, five arms each, no training.

This round is evaluation only. It re-infers fifteen already-registered checkpoints over the 2022
validation set, writes every forecast field to disk, and then computes four physical-balance
diagnostics from those fields. Nothing is trained, so there is no optimizer, no dose argument and
no capacity confound -- the only thing under test is whether the diagnostic separates the arms.

Layout written under the attempt directory::

    seed41/stage_a/off/case_00000.npz ...
    seed41/stage_a/on/case_00000.npz ...
    seed41/urban_static/control/case_00000.npz ...
    diagnostics.json          the per-arm tables and the (a)-(e) clauses
    seed41_receipt.json       per-arm field digests, device, seconds

**Why the family directory exists.** Stage A's ``on``/``off`` and the urban-static
``control``/``treatment``/``placebo`` come from different training recipes, so a delta across two
families would mix two independent runs. The driver never compares across families; the verdict
module enforces the same rule.

**Worker shape.** ``r7_round.py stage-seeds`` spawns one worker per seed with
``CUDA_VISIBLE_DEVICES`` already pinned to the card it cleared. The worker therefore loads one
model at a time and never sees another card.
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

from scripts import r7_s3_balance_protocol as proto
from training import r7_device_selection as devices

SEEDS = proto.SEEDS
LEADS = proto.LEADS
STEP_HOURS = proto.STEP_HOURS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


def _arms_for_seed(body: dict, seed: int) -> list[dict]:
    """This seed's five arms, in family order, straight from the frozen registry."""
    arms = [arm for arm in body["arms"] if int(arm["seed"]) == seed]
    if len(arms) != proto.expected_arm_count() // len(SEEDS):
        raise RuntimeError(f"seed {seed} has {len(arms)} arms in the frozen registry, not "
                           f"{proto.expected_arm_count() // len(SEEDS)}")
    return arms


def _dump_seed(output: Path, body: dict, seed: int, deadline: float) -> dict:
    """Dump every arm of one seed. Returns the receipt body; writes each arm's own report."""
    from scripts import r7_balance_field_dump as dump
    from training.r7_arm_harness import sha256_file

    manifest = Path(body["val_manifest"])
    receipts = {}
    for arm in _arms_for_seed(body, seed):
        if time.monotonic() > deadline:
            raise TimeoutError(f"deadline reached before arm {arm['family']}/{arm['arm']} of "
                               f"seed {seed}")
        arm_dir = output / f"seed{seed}" / arm["family"] / arm["arm"]
        started = time.monotonic()
        report = dump.dump_arm(
            Path(arm["checkpoint"]), arm["model_spec"], manifest, arm_dir,
            lead_hours=tuple(LEADS), step_hours=STEP_HOURS, device_name="cuda",
            deadline=deadline, resume=True)
        elapsed = time.monotonic() - started
        if int(report["cases_total"]) != int(body["n_cases_expected"]):
            raise RuntimeError(f"{arm['family']}/{arm['arm']}/seed{seed}: dumped "
                               f"{report['cases_total']} cases, the protocol froze "
                               f"{body['n_cases_expected']}")
        if report["model_code_sha256"] != body["model_code_sha256"]:
            raise RuntimeError(f"{arm['family']}/{arm['arm']}/seed{seed}: the checkpoint's code "
                               "identity is not the frozen one")
        if sha256_file(Path(arm["checkpoint"])) != arm["checkpoint_sha256"]:
            raise RuntimeError(f"{arm['family']}/{arm['arm']}/seed{seed}: the checkpoint moved "
                               "since freeze")
        receipts[f'{arm["family"]}/{arm["arm"]}'] = {
            **report, "elapsed_seconds": elapsed,
            "seconds_per_case": elapsed / max(1, int(report["cases_total"])),
            "declared_checkpoint_sha256": arm["checkpoint_sha256"],
            "checkpoint_sha256_matches_freeze": True}
        print(f"[seed {seed}] {arm['family']}/{arm['arm']}: {report['cases_written']} written, "
              f"{report['cases_reused']} reused, {elapsed:.0f}s", flush=True)
    return {"seed": seed, "arms": receipts}


def _arm_directories(output: Path, body: dict) -> dict:
    return {f'{arm["family"]}/{arm["arm"]}/seed{arm["seed"]}': {
        "family": arm["family"], "arm": arm["arm"], "seed": int(arm["seed"]),
        "dir": str(output / f'seed{arm["seed"]}' / arm["family"] / arm["arm"])}
        for arm in body["arms"]}


def readings(output: Path, body: dict) -> dict:
    """Compute every arm's tables and the five clauses from the dumped fields."""
    from scripts import r7_balance_exits as exits
    from scripts import r7_balance_verdict as verdict
    from training.r7_arm_harness import sha256_file

    directories = _arm_directories(output, body)
    tables = verdict.table_of_arms(directories)
    # ``seeds`` is what lets a clause tell "measured on all three seeds" from "not yet measured".
    # Without it a partial dump would return a verdict the frozen design never authorised.
    families = {name: {"arms": list(spec["arms"]), "factor": spec["factor"],
                       "seeds": [int(seed) for seed in body["seeds"]]}
                for name, spec in body["families"].items()}
    judgement = exits.evaluate(tables, families, body["lead_hours"],
                               body["preflight"].get("e_preflight_passed") is True)
    return {
        "format": "r7-s3-balance-diagnostics-v1",
        "protocol_sha256": body["protocol_sha256"],
        "val_manifest": body["val_manifest"],
        "val_manifest_sha256": sha256_file(Path(body["val_manifest"])),
        "n_cases": body["n_cases_expected"],
        "n_arms": len(tables),
        "leads": list(body["lead_hours"]),
        "families": families,
        "family_argument": body["family_argument"],
        "arms": tables,
        "exits": judgement,
        "preflight": body["preflight"],
        "checks": {
            "delta_hb_offset_baseline": body["preflight"]["clauses"]
            ["hydrostatic_offset_is_a_baseline"]["baseline_median"],
            "geostrophic_coverage": body["preflight"]["clauses"]
            ["geostrophic_coverage_is_measured"]["levels"],
            "level_pair_note": ("850-500 uses virtual temperature; 500-250 substitutes t500 for "
                                "the absent t250 and is reported separately, never merged"),
        },
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "limitations": body["limitations"],
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
        return _archive(output, proto.validate_protocol(output))
    body = proto.validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside the frozen round")
    from training.r7_s3_v3_screen import pin_declared_gpu
    pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        verdict.write(output / "diagnostics.json", readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _write_exclusive(output / f"seed{seed}_receipt.json", _dump_seed(output, body, seed, deadline))


def _archive(output: Path, body: dict) -> dict:
    """Verify the code archive's comment and execution sources match the freeze."""
    import hashlib
    from io import BytesIO
    import zipfile
    from training.r7_arm_harness import sha256_file

    archive = Path(body["code_archive_input"])
    if any(path.is_symlink() for path in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs from freeze")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError(f"archive omits or changes execution source: {name}")
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    receipt = {"scientific_claim": False, "code_commit": body["code_commit"],
               "code_zip_sha256": sha256_file(output / "code.zip")}
    _write_exclusive(output / "archive_receipt.json", receipt)
    return receipt


def _write_exclusive(path: Path, body: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    arguments = parser.parse_args(argv)
    if not arguments.worker:
        raise SystemExit("this round is driven by `r7_round.py stage-seeds`; invoke the driver "
                         "with --worker <prepare|archive|seed|reading>")
    # The worker's return value is discarded: ``sys.exit(value)`` would print a protocol-sized
    # body into the stage log and set a nonzero status.
    worker(arguments.out, arguments.worker, arguments.seed, arguments.deadline)


if __name__ == "__main__":
    main()
