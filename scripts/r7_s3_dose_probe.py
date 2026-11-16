"""Stage B: the zero-training probe over checkpoints that already exist.

Two measurements, neither of which trains anything:

**B-1 dose curve.** The p1600 run retained 26 checkpoints at 5,000-update intervals. Only its
endpoint was ever scored, so the *shape* of skill against update count inside one trajectory is
unknown. A within-run curve is strictly better evidence than the four registered ladder rungs
because those are four independent runs with four cosine schedules, and the ladder itself spans two
different ``training_code_sha256`` values.

**B-2 inference depth.** ``evaluate_local`` takes a bare ``reasoning_steps`` with no controller --
the controller-only guards apply to ``policy_selection``, ``validation_thresholds`` and
``force_full_depth``. The registered model was trained at K=4 and the closed #65 C3 pilot saw the
best depth move with lead, so the question "is K=4 the right depth at a converged budget" has never
been asked of a 128,528-update checkpoint.

This module refuses to be a training entry point: it has no ``freeze`` and no ``seed`` worker, and
``REFUSED_WORKERS`` names every stage name that would imply one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "r7-s3-dose-depth-probe-v1"

P1600_RUN = ROOT / "outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01"
P4000_RUN = ROOT / "outputs/r7_s3_throughput_physics_b_p4000_20261105_attempt02"
VAL_MANIFEST = (ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
                / "store/manifests_treatment/val.jsonl")

#: B-1 reads thirteen checkpoints, one per 10,000 updates, plus the endpoint. Reading all 26 would
#: double the cost without changing the shape of the curve.
DOSE_UPDATES = (5_000, 15_000, 25_000, 35_000, 45_000, 55_000, 65_000, 75_000, 85_000, 95_000,
                105_000, 115_000, 125_000, 128_528)
DOSE_LEADS = (6, 24, 72)
#: B-2 sweeps the inference depth at the converged endpoint. K=4 is the registered training depth
#: and therefore the baseline every delta is stated against.
DEPTHS = (2, 4, 6, 8)
DEPTH_LEADS = (6, 12, 24, 48, 72)

#: The last five points decide "is the run saturated": with 14 points, five is enough to see whether
#: the tail is still falling rather than one noisy pair.
TAIL_POINTS = 5
#: A depth counts only when the paired gain clears this on at least three leads, the same floor the
#: round's (a) clause uses.
MINIMUM_LEAD_GAIN_K = 0.005
MINIMUM_CONCORDANT_LEADS = 3

#: Stage names that would turn this probe into a training driver. Refusing them is what keeps
#: "zero GPU-h training" a checkable property rather than a promise.
REFUSED_WORKERS = ("train", "seed", "seeding", "prepare", "archive", "reading", "fine-tune")


def checkpoints(run: Path, updates) -> list[Path]:
    """Resolve the named checkpoints, failing loudly on any that is absent."""
    resolved = []
    for update in updates:
        path = run / "seed41/training" / f"update_{update:07d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"the probe names update {update}, which is not on disk: {path}")
        resolved.append(path)
    return resolved


def checkpoint_identity(paths) -> dict:
    """Every checkpoint must load under one code identity, or the curve is not one trajectory."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import load_checkpoint, model_code_digest

    digests = {}
    for path in paths:
        saved = load_checkpoint(path)
        digests[path.name] = {"updates": int(saved["updates"]),
                              "model_code_sha256": saved["model_code_sha256"],
                              "checkpoint_sha256": sha256_file(path)}
    identities = {record["model_code_sha256"] for record in digests.values()}
    return {
        "passed": bool(len(identities) == 1 and identities == {model_code_digest()}),
        "current_model_code_sha256": model_code_digest(),
        "distinct_identities": sorted(identities),
        "checkpoints": digests,
        "rule": ("a dose curve is only one trajectory if every checkpoint loads under one code "
                 "identity; the registered four-rung ladder spans two and cannot be spliced"),
    }


def build_preparation(*, deadline: float):
    """The manifest/store/climatology preparation, built **once** for the whole probe.

    ``EvaluationPreparation`` validates all 1,458 held-out manifest records against the store and
    lazily fits the train-only climatology. Neither depends on the checkpoint, so rebuilding it per
    checkpoint would redo that work eighteen times (fourteen dose points plus four depth arms) for
    no change in what is computed. It is built here and handed to every ``score`` call.
    """
    from training.r7_evaluate import EvaluationPreparation

    return EvaluationPreparation(VAL_MANIFEST, deadline=deadline)


def score(checkpoint: Path, leads, *, device: str, deadline: float, prepared, root: Path,
          reasoning_steps=None) -> dict:
    """Score one checkpoint at the given leads, returning ``{lead: {variable: rmse}}``.

    ``root`` is the operator's ``--out`` directory. Scoring must write under it and **not** under
    the checkpoint's own run: those run directories belong to registered prior rounds and are
    archived evidence (R-002's read-only discipline), and ``evaluate_local`` creates its output
    directory with ``exist_ok=False``, so writing there both mutates an archived run and makes every
    second attempt fail with ``FileExistsError``.
    """
    import csv

    from scripts import r7_s3_typed_budget_protocol as proto
    from training.r7_evaluate import evaluate_local

    # ``reasoning_steps`` distinguishes the B-2 depth arms of one checkpoint; the dose points use
    # the trained depth and keep the plain stem, so the two measurements never share a directory.
    arm = checkpoint.stem if reasoning_steps is None else f"{checkpoint.stem}_k{reasoning_steps}"
    table = {}
    for lead in leads:
        destination = Path(root) / arm / f"lead_{lead:03d}h"
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name=device,
            reasoning_steps=reasoning_steps,
            # Without this the boundary CSV is silently not written and every reader below fails
            # after the GPU has been paid for; see the constant's note in the protocol module.
            boundary_margins=proto.BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        boundary = destination / "boundary_rmse.csv"
        values = {}
        with boundary.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["region"] == proto.REGION:
                    values[row["variable"]] = float(row["rmse"])
        table[str(lead)] = {"t2m": values["t2m"], "z500": values["z500"],
                            "n_evaluated": provenance["n_evaluated"]}
    return table


def dose_curve(table: dict, updates) -> dict:
    """Is the tail still falling? The two readings are reported, never merged."""
    series = []
    for update in updates:
        row = table.get(str(update)) or {}
        series.append({"updates": update, "t2m_72h": (row.get("72") or {}).get("t2m"),
                       "t2m_6h": (row.get("6") or {}).get("t2m")})
    measured = [row for row in series if row["t2m_72h"] is not None]
    if len(measured) < TAIL_POINTS:
        return {"value": None, "series": series,
                "reason": f"only {len(measured)} dose points are measured; the tail needs "
                          f"{TAIL_POINTS}"}
    tail = measured[-TAIL_POINTS:]
    steps = [row["t2m_72h"] for row in tail]
    strictly_falling = all(later < earlier for earlier, later in zip(steps, steps[1:]))
    return {
        "value": bool(strictly_falling),
        "tail_points": TAIL_POINTS,
        "tail_rmse_72h": steps,
        "tail_delta": steps[-1] - steps[0],
        "still_falling": bool(strictly_falling),
        "full_series": series,
        "reason": ("strict monotonic decrease over the last five measured dose points; a "
                   "non-monotone tail is reported as not saturated rather than smoothed"),
    }


def depth_sweep(table: dict, depths) -> dict:
    """Does any depth other than the trained K=4 improve three or more leads?"""
    baseline = table.get(str(4))
    if not baseline:
        return {"value": None, "reason": "the registered K=4 baseline was not measured"}
    per_depth = {}
    for depth in depths:
        row = table.get(str(depth))
        if not row:
            per_depth[str(depth)] = {"complete": False}
            continue
        deltas = {}
        for lead in DEPTH_LEADS:
            base = (baseline.get(str(lead)) or {}).get("t2m")
            value = (row.get(str(lead)) or {}).get("t2m")
            deltas[str(lead)] = None if base is None or value is None else value - base
        measured = {lead: delta for lead, delta in deltas.items() if delta is not None}
        clearing = [lead for lead, delta in measured.items()
                    if delta <= -MINIMUM_LEAD_GAIN_K]
        per_depth[str(depth)] = {
            "complete": len(measured) == len(DEPTH_LEADS),
            "delta_vs_k4": deltas,
            "n_clearing": len(clearing),
            "clearing_leads": clearing,
            "preferred": bool(len(clearing) >= MINIMUM_CONCORDANT_LEADS),
        }
    candidates = [depth for depth, row in per_depth.items()
                  if depth != "4" and row.get("preferred")]
    cheaper = [depth for depth in candidates if int(depth) < 4]
    return {
        "value": bool(candidates),
        "per_depth": per_depth,
        "candidates": sorted(candidates, key=int),
        "cheaper_than_trained": sorted(cheaper, key=int),
        "thresholds": {"minimum_lead_gain_k": MINIMUM_LEAD_GAIN_K,
                       "minimum_concordant_leads": MINIMUM_CONCORDANT_LEADS},
        "reason": ("a depth is a candidate only when it improves at least three leads by the "
                   "declared floor against the trained K=4; a cheaper candidate is reported "
                   "separately because a lower depth also lowers inference cost"),
    }


def run(output: Path, *, device: str, deadline: float, dose_only: bool = False,
        depth_only: bool = False) -> dict:
    """Score the dose points and the depth sweep; nothing here trains anything."""
    output = Path(output)
    started = time.monotonic()
    scoring_root = output.parent / f"{output.stem}_scoring"
    result = {"scientific_claim": False, "test_read": False, "r": 0, "gpu_hours": 0.0,
              "device": device, "format": FORMAT, "scoring_root": str(scoring_root)}
    # Built once: the manifest/store validation and the train-only climatology do not depend on
    # which checkpoint or depth is being scored.
    prepared = build_preparation(deadline=deadline)

    if not depth_only:
        paths = checkpoints(P1600_RUN, DOSE_UPDATES)
        result["dose_identity"] = checkpoint_identity(paths)
        table = {}
        for path in paths:
            point_started = time.monotonic()
            table[str(int(path.stem.split("_")[1]))] = score(
                path, DOSE_LEADS, device=device, deadline=deadline, prepared=prepared,
                root=scoring_root / "dose")
            print(f"[dose] {path.name}: {time.monotonic() - point_started:.0f}s", flush=True)
        result["dose_tables"] = table
        result["dose_curve"] = dose_curve(table, DOSE_UPDATES)

    if not dose_only:
        endpoint = P1600_RUN / "seed41/training/update_0128528.pt"
        if not endpoint.is_file():
            raise FileNotFoundError(f"the depth sweep needs the converged endpoint: {endpoint}")
        table = {}
        for depth in DEPTHS:
            depth_started = time.monotonic()
            table[str(depth)] = score(endpoint, DEPTH_LEADS, device=device, deadline=deadline,
                                      reasoning_steps=depth, prepared=prepared,
                                      root=scoring_root / "depth")
            print(f"[depth] K={depth}: {time.monotonic() - depth_started:.0f}s", flush=True)
        result["depth_tables"] = table
        result["depth_sweep"] = depth_sweep(table, DEPTHS)
    # One card, one process: wall clock on that card is the GPU-h this probe spent.
    result["gpu_hours"] = round((time.monotonic() - started) / 3600.0, 4)
    return result


def decision(record: dict) -> dict:
    """What the probe licenses about the main round's budget."""
    curve = record.get("dose_curve") or {}
    sweep = record.get("depth_sweep") or {}
    still_falling = curve.get("value")
    return {
        "dose_curve_status": ("saturated" if still_falling is False
                              else "unsaturated" if still_falling is True else "untested"),
        "depth_candidates": sweep.get("candidates"),
        "cheaper_depth_candidates": sweep.get("cheaper_than_trained"),
        "sixteen_epoch_is_warranted": bool(still_falling is True),
        "rule": ("16 epochs is only warranted when the within-run tail is still strictly falling; "
                 "an untested curve authorises nothing and is reported as such"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--dose-only", action="store_true")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--worker", default=None)
    arguments = parser.parse_args(argv)
    if arguments.worker in REFUSED_WORKERS:
        raise SystemExit(f"this probe trains nothing; {arguments.worker!r} is a training stage and "
                         f"is refused (allowed: none of {REFUSED_WORKERS})")
    if arguments.worker:
        raise SystemExit(f"unknown worker {arguments.worker!r}; this probe has no stages")
    from scripts.r7_m3_offline import deny_network
    deny_network()
    deadline = arguments.deadline
    if deadline is None:
        deadline = float("inf")
    record = {"scientific_claim": False, "test_read": False, "r": 0,
              "dose_updates": list(DOSE_UPDATES), "dose_leads": list(DOSE_LEADS),
              "depths": list(DEPTHS), "depth_leads": list(DEPTH_LEADS)}
    failed = False
    try:
        record.update(run(arguments.out, device=arguments.device, deadline=deadline,
                          dose_only=arguments.dose_only, depth_only=arguments.depth_only))
    except Exception as error:                                   # noqa: BLE001 - recorded, not hidden
        failed = True
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    record["decision"] = decision(record)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
                             + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
                      ("decision", "error", "gpu_hours") if key in record}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
