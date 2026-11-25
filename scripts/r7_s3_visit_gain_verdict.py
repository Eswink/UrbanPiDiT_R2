"""G1-B's four predeclared exits, computed from receipts and CSVs (decision 0119 D5).

This module is the part of the round that reads numbers. It is deliberately separate from the
driver and has no GPU, no dataset and no model dependency: every function here takes the
already-written receipts and ``boundary_rmse.csv`` tables and returns a verdict, so a test can
feed it a synthetic arm pair and check that a wrong sign, a missing arm or a non-finite value
lands on the side of the predeclared rule rather than on the flattering side.

Three properties the rule has to keep, each of which a plausible implementation loses:

* **a missing arm is null, never false.** An arm that never reached its endpoint has no RMSE;
  recording ``false`` would claim a measured failure of the hypothesis.
* **exit 1 is a per-seed sign test.** A three-seed mean that happens to be negative while two
  seeds are positive does not satisfy it. This is deliberately stricter than exit 2.
* **exit 2 is a seed mean, not a seed sign test.** Short leads are allowed to flip on one seed;
  that is what makes it a different clause from exit 1 rather than a weaker copy.

The sign convention throughout is ``A - B``: negative means arm A (``visit_damped``) has the
lower RMSE, i.e. A is better. ``READING_RULE`` states this in the frozen protocol and
:func:`delta` is the only place the subtraction happens.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scripts import r7_s3_visit_gain_protocol as proto

FORMAT = "r7-s3-visit-gain-verdict-v1"


def read_region_variable(table: Path, region: str, variable: str) -> float:
    """One RMSE from a ``boundary_rmse.csv``, refusing an ambiguous or non-finite value."""
    rows = list(csv.DictReader(Path(table).read_text(encoding="utf-8").splitlines()))
    matches = [row for row in rows if row.get("region") == region
               and row.get("variable") == variable]
    if len(matches) != 1:
        raise ValueError(f"{table}: expected exactly one {region}/{variable} row, "
                         f"found {len(matches)}")
    value = float(matches[0]["rmse"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{table}: {region}/{variable} rmse {value!r} is not a positive finite")
    return value


def delta(a_rmse: float, b_rmse: float) -> float:
    """``A - B``. Negative means A is better; this is the only subtraction in the module."""
    return float(a_rmse) - float(b_rmse)


def arm_rmse(receipt: dict, lead: int, region: str = proto.REGION,
             variable: str = "t2m") -> float:
    """The t2m RMSE of one arm at one lead, from the receipt's own evaluation directory."""
    evaluation = (receipt.get("evaluations") or {}).get(str(int(lead)))
    if not evaluation:
        raise KeyError(f"receipt for arm {receipt.get('arm')!r} seed {receipt.get('seed')!r} has "
                       f"no evaluation at {lead} h")
    return read_region_variable(Path(evaluation["dir"]) / "boundary_rmse.csv", region, variable)


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def exit_one(deltas: dict) -> dict:
    """72 h: A - B negative on all three seeds.

    A zero delta is not negative, so a tie fails this exit rather than passing it.
    """
    values = {str(seed): deltas[str(seed)] for seed in proto.SEEDS if str(seed) in deltas}
    if len(values) != len(proto.SEEDS) or any(not math.isfinite(v) for v in values.values()):
        return {"value": None, "reason": "every seed's 72 h delta must be present and finite",
                "deltas": values, "signs": {}}
    signs = {seed: _sign(value) for seed, value in values.items()}
    passed = all(sign == -1 for sign in signs.values())
    return {"value": passed, "deltas": values, "signs": signs,
            "mean": sum(values.values()) / len(values),
            "rule": "negative on every seed; a single flip or a tie makes this false"}


def exit_two(deltas_by_lead: dict) -> dict:
    """6 h and 12 h: the seed mean of A - B is <= 0 at each lead."""
    leads = {}
    for lead in (6, 12):
        values = [deltas_by_lead[str(lead)][str(seed)] for seed in proto.SEEDS]
        if any(not math.isfinite(value) for value in values):
            return {"value": None, "reason": f"lead {lead} has a non-finite delta",
                    "leads": leads}
        mean = sum(values) / len(values)
        leads[str(lead)] = {"deltas": dict(zip((str(s) for s in proto.SEEDS), values)),
                            "mean": mean, "passed": mean <= 0.0}
    return {"value": all(entry["passed"] for entry in leads.values()), "leads": leads,
            "rule": "each of the two leads is judged by its own three-seed mean; one seed may flip"}


def exit_three(identity: dict) -> dict:
    """The within-round identity witness: off-path inertness plus A's k=0 bridge.

    Both halves must be true. ``identity`` carries the off-path and k=0 results as measured by
    the driver's own checks; this function only refuses partial or missing evidence.
    """
    off = identity.get("off_path")
    bridge = identity.get("visit_damped_k0")
    required_off = ("bitwise_equal_to_pre_switch_expression", "key_set_equal", "tensors_equal",
                    "parameter_count_equal", "no_new_buffers")
    missing = [name for name in required_off if not isinstance(off, dict) or name not in off]
    if missing or not isinstance(bridge, dict) or "bitwise_equal" not in bridge:
        return {"value": None, "reason": f"identity evidence incomplete: missing {missing}",
                "off_path": off, "visit_damped_k0": bridge}
    passed = all(bool(off[name]) for name in required_off) and bool(bridge["bitwise_equal"])
    return {"value": passed, "off_path": {name: bool(off[name]) for name in required_off},
            "visit_damped_k0": {"bitwise_equal": bool(bridge["bitwise_equal"]),
                                "gain_at_zero": bridge.get("gain_at_zero")},
            "rule": ("off path reproduces the pre-switch expression bit for bit with the same key "
                     "set and no new buffers, and visit_damped at visit 0 equals it bit for bit "
                     "because g_0 is exactly 1.0")}


def exit_four(migration: dict) -> dict:
    """Every seed's parameter key set and parent state digest are bit-identical."""
    seeds = (migration or {}).get("seeds") or {}
    entries = {}
    for seed in proto.SEEDS:
        entry = seeds.get(str(seed))
        if not isinstance(entry, dict):
            return {"value": None, "reason": f"seed {seed} has no migration evidence",
                    "seeds": entries}
        entries[str(seed)] = {
            "state_digest_equal": bool(entry.get("state_digest_equal")),
            "key_set_equal": bool(entry.get("key_set_equal")),
            "bitwise_identical": bool(entry.get("bitwise_identical")),
            "tensor_count": entry.get("tensor_count"),
            "source_matches_frozen_pin": bool(entry.get("source_matches_frozen_pin")),
        }
    passed = all(entry["bitwise_identical"] and entry["key_set_equal"]
                 and entry["state_digest_equal"] for entry in entries.values())
    return {"value": passed, "seeds": entries,
            "rule": "the p32,132 state went in unchanged: same key set, same tensor digest"}


def _collect_deltas(receipts: dict) -> dict:
    """``{lead: {seed: A - B}}`` from the per-seed arm receipts."""
    deltas = {}
    for lead in proto.LEADS:
        per_seed = {}
        for seed in proto.SEEDS:
            pair = receipts.get(str(seed)) or {}
            a, b = pair.get(proto.VISIT_DAMPED), pair.get(proto.UNIFORM_DAMPED)
            if a is None or b is None:
                per_seed[str(seed)] = float("nan")
                continue
            per_seed[str(seed)] = delta(arm_rmse(a, lead), arm_rmse(b, lead))
        deltas[str(lead)] = per_seed
    return deltas


def readings(receipts: dict, identity: dict, migration: dict, *, protocol_sha256: str) -> dict:
    """Assemble ``readings.json``: every exit as true/false/null, plus the descriptive B-C."""
    deltas = _collect_deltas(receipts)
    one = exit_one(deltas["72"])
    two = exit_two(deltas)
    three = exit_three(identity)
    four = exit_four(migration)
    body = {
        "format": FORMAT,
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "protocol_sha256": protocol_sha256,
        "decision": "0119",
        "region": proto.REGION,
        "region_note": proto.REGION_NOTE,
        "sign_convention": "A - B; negative means arm A (visit_damped) has the lower RMSE",
        "delta_by_lead": deltas,
        "exits": {"one": one, "two": two, "three": three, "four": four},
        "attribution_boundary": proto.ATTRIBUTION_BOUNDARY,
        "limitations": list(proto.LIMITATIONS),
    }
    values = [one["value"], two["value"], three["value"], four["value"]]
    body["branch_status"] = ("closed" if any(value is False for value in values)
                             else ("null" if any(value is None for value in values)
                                   else "open"))
    body["closure_rule"] = proto.EXITS["closure"]
    return body


def _load_receipts(output: Path) -> dict:
    receipts = {}
    for seed in proto.SEEDS:
        pair = {}
        for arm in proto.ARM_NAMES:
            path = Path(output) / f"seed{seed}_{arm}_receipt.json"
            if path.is_file():
                pair[arm] = json.loads(path.read_text(encoding="utf-8"))
        if pair:
            receipts[str(seed)] = pair
    return receipts


def write_readings(output: Path, body: dict) -> None:
    """Exclusive write: a reading is written once, never re-rendered over itself."""
    path = Path(output) / "readings.json"
    if path.exists():
        raise FileExistsError(f"readings already written: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_identity(output: Path) -> dict:
    """The identity witness the driver wrote before any optimizer step."""
    path = Path(output) / "identity_witness.json"
    if not path.is_file():
        raise FileNotFoundError(f"the identity witness is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build(output: Path, *, protocol_sha256: str, migration_path: Path) -> dict:
    """The reading stage's whole job: receipts + witness + migration -> ``readings.json``."""
    from scripts.r7_s3_visit_gain_parent import state_equality_evidence

    body = readings(_load_receipts(output), load_identity(output),
                    state_equality_evidence(migration_path), protocol_sha256=protocol_sha256)
    write_readings(output, body)
    return body
