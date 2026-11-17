"""Assemble the (b) reading from the arms that actually exist, without touching a frozen source.

Why this exists
---------------
The round's own ``readings()`` hard-requires all nine arm receipts
(``scripts/study_r7_s3_typed_budget.py:313``: it does a plain ``json.loads`` of
``seed{N}_{arm}_receipt.json`` for every arm) and ``worker()`` refuses to write a reading whose
``complete`` flag is false. When one arm cannot finish -- here the typed arm, truncated on every
seed by the 35 h per-seed envelope -- the round therefore produces **no** ``readings.json`` at all,
and a clause that *is* measurable from the arms that did finish is lost with it.

Clause (b) is ``generic - incumbent``. It never reads the typed arm, so three seeds of
``incumbent`` and ``generic`` are sufficient for a real three-seed (b). This module computes exactly
that and nothing else:

* it reuses the frozen verdict module's own ``_clause_pair``/``clause_e`` machinery rather than
  reimplementing the thresholds, so the numbers are the frozen ones;
* it refuses to emit (a): ``typed`` is absent by construction here, so (a) can only be ``null``,
  and a tool that printed it would invite someone to read a null as a result;
* every clause it does not decide is reported as ``None`` with a reason, never as ``False``.

**This file lives under ``tools/`` on purpose.** ``tools/`` never enters a round's identity
surface (``tools/r7_explicit_inventory.py``: ``PACKAGES = ("data","model","training","scripts")``),
and a round's ``execution_files_sha256`` is a static list frozen at freeze time -- verified: the
frozen inventory is 84 entries and recomputing it today yields the same 84, with nothing added or
removed. So adding this tool cannot make ``validate_protocol`` reject the live attempt.

What it does NOT do
-------------------
It does not write ``readings.json`` (that name is the round's own stage receipt and is guarded), it
does not decide a scientific claim, and it does not upgrade (b) into a positive result. A true (b)
licenses one sentence -- "switching the physical-evidence pathway on is better than leaving it off"
-- and that sentence carries the +148,800 parameters the pathway adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_typed_budget_verdict as verdict          # noqa: E402
from scripts import study_r7_s3_typed_budget as driver          # noqa: E402

#: The only pair (b) needs. Kept explicit so nobody widens it by accident.
PARTIAL_PAIR = (verdict.GENERIC, verdict.INCUMBENT)


def arm_row(attempt: Path, seed: int, arm: str) -> dict | None:
    """One arm's RMSE table and per-lead evaluation directories, or ``None`` if absent.

    Reads the **per-arm** receipt, which exists whenever that one arm finished -- the seed-level
    ``seed{N}_receipt.json`` is only written after all three arms succeed, so it is the wrong
    granularity for a partially finished seed.
    """
    receipt_path = Path(attempt) / f"seed{seed}_{arm}_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluations = receipt.get("evaluations") or {}
    table = {}
    for lead, entry in evaluations.items():
        boundary = Path(entry["dir"]) / "boundary_rmse.csv"
        if not boundary.is_file():
            continue
        table[str(lead)] = {variable: driver._region_rmse(boundary, driver.proto.REGION, variable)
                            for variable in driver.CLAUSE_VARIABLES}
    if not table:
        return None
    return {"rmse": table, "parameters": receipt.get("parameters"),
            "typed_evidence_mode": receipt.get("typed_evidence_mode"),
            "seconds_per_update": receipt.get("seconds_per_update"),
            "evaluation_dirs": {str(lead): entry["dir"] for lead, entry in evaluations.items()}}


def assemble(attempt: Path, seeds, leads) -> dict:
    """Build the arm tables for the two arms (b) needs, plus the third arm left explicitly empty."""
    attempt = Path(attempt)
    arms = {name: {} for name in (verdict.INCUMBENT, verdict.GENERIC, verdict.TYPED)}
    present = {}
    for seed in seeds:
        for arm in PARTIAL_PAIR:
            row = arm_row(attempt, seed, arm)
            if row is not None:
                arms[arm][str(seed)] = row
        present[str(seed)] = {arm: str(seed) in arms[arm] for arm in PARTIAL_PAIR}
    return {
        "format": "r7-s3-typed-evidence-budget-partial-readings-v1",
        "attempt": str(attempt),
        "seeds": list(seeds), "lead_hours": list(leads),
        "arms": arms,
        "arm_presence": present,
        "scored_arm": list(PARTIAL_PAIR),
        "note": ("partial reading: only the pair (b) is decidable from these arms. The typed arm is "
                 "absent by construction, so (a) is null because it is untested -- not because the "
                 "type structure was measured and failed."),
    }


def evaluate_partial(body: dict) -> dict:
    """Compute (b) and (e) with the frozen module; leave (a), (c), (d) explicitly null.

    (b) is ``generic - incumbent`` and needs only those two arms. (e) is the z500@72h guard,
    which is also defined against each seed's own incumbent, so it is decidable here too -- but
    only over whichever seeds have both arms.
    """
    seeds = [int(seed) for seed in body["seeds"]]
    leads = [int(lead) for lead in body["lead_hours"]]
    arms = body["arms"]
    have_b = [seed for seed in seeds
              if str(seed) in arms[verdict.INCUMBENT] and str(seed) in arms[verdict.GENERIC]]
    # ``_clause_pair`` divides by ``len(measured)`` and raises ``ZeroDivisionError`` when the seed
    # list is empty -- a missing guard in the frozen module, found by running this tool on a round
    # where no generic arm existed yet. Do not let a partial reading crash here: an empty seed set
    # is "untested", which is exactly what the null discipline asks for.
    if not have_b:
        empty = {"value": None,
                 "reason": ("no seed has both an incumbent and a generic receipt, so (b) has no "
                            "contrast to compute; untested, not failed")}
        return {"value": None, "seeds_with_both_arms": [], "seeds_complete": False,
                "clauses": {"b_generic_minus_incumbent": dict(empty),
                            "e_z500_72h_guard": dict(empty),
                            "a_typed_minus_generic": {
                                "value": None,
                                "reason": "the typed arm produced no receipt on any seed"},
                            "c_claim_scope": {"value": None, "permitted_claim": None,
                                              "reason": "(c) is undefined while (a)/(b) are untested"},
                            "d_budget_direction": {"value": None,
                                                   "reason": "no 16-epoch reading exists"}}}
    # The interval needs provenance directories keyed by lead per arm per seed, the shape
    # ``evaluate`` builds for the full round.
    dirs = {name: {str(seed): (arms[name].get(str(seed)) or {}).get("evaluation_dirs") or {}
                   for seed in seeds}
            for name in (verdict.INCUMBENT, verdict.GENERIC, verdict.TYPED)}
    b_clause = verdict._clause_pair(arms, have_b, leads, "t2m",
                                    verdict.GENERIC, verdict.INCUMBENT, dirs=dirs)
    e_clause = verdict.clause_e(arms, have_b)
    seeds_complete = len(have_b) == len(seeds)
    return {
        "value": b_clause.get("value"),
        "seeds_with_both_arms": have_b,
        "seeds_complete": seeds_complete,
        "clauses": {
            "b_generic_minus_incumbent": b_clause,
            "e_z500_72h_guard": e_clause,
            "a_typed_minus_generic": {
                "value": None,
                "reason": ("the typed arm produced no receipt on any seed this round (it is "
                           "truncated by the per-seed envelope), so (a) is untested -- null, not "
                           "false and not true")},
            "c_claim_scope": {
                "value": None,
                "permitted_claim": None,
                "reason": "(c) is undefined while (a) is untested"},
            "d_budget_direction": {
                "value": None,
                "reason": "no 16-epoch reading exists in this round; (d) stays null"},
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        help="write the partial reading here; refused if it would be empty")
    args = parser.parse_args(argv)
    body = assemble(args.attempt, driver.SEEDS, driver.LEADS)
    result = evaluate_partial(body)
    payload = {"reading": {k: v for k, v in body.items() if k != "arms"},
               "arm_presence": body["arm_presence"],
               "judgement": result,
               "scientific_claim": False,
               "test_read": False}
    if args.out:
        if not result["seeds_with_both_arms"]:
            raise SystemExit("refusing to write: no seed has both arms, so there is nothing to read")
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
