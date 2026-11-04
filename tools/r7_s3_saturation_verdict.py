"""Stage B's saturation verdict: the frozen two-point first difference, computed by code.

The goal document §3 freezes one rule and nothing else:

    Δ = skill_72h(16 epoch) − skill_72h(4 epoch)
    Δ > 0.01                     -> unsaturated; a final run at 16-64 epoch follows
    Δ < 0.005 AND skill_72h(16) < skill_72h(4)  -> saturated; take the saturation point
    otherwise                    -> record only; no extrapolation, no extra points

One property of that rule is worth stating plainly, because it is easy to misread: **any
decline is saturated**. A negative Δ is always below the 0.005 threshold, and the second
clause then also holds, so the saturated branch reduces to "16 epoch scored below 4 epoch".
The "small delta" wording only bites on the rising side, where a rise under 0.005 is
record-only. This module applies the rule exactly as frozen; it does not repair that
asymmetry, because adjusting a pre-registered threshold after seeing the data is precisely
what this campaign forbids.

This module evaluates exactly that, and only that. It lives outside the per-point drivers
because no single point can compute a two-point difference -- the drivers deliberately record
``saturation_verdict: null`` and this tool consumes their ``readings.json`` files afterwards.

Three deliberate properties:

* **It consumes the registered readings, not re-derived numbers.** Every input is read from a
  point's own ``readings.json``, whose ``protocol_sha256`` is re-verified against the frozen
  protocol before the value is used. A hand-typed skill number cannot enter the verdict.
* **It refuses incomplete or mismatched input.** Missing point, wrong update count, wrong
  batch, non-finite skill, or a protocol digest that does not self-verify all raise; there is
  no "best effort" path, because a silently partial comparison is how a one-point reading gets
  mistaken for a saturation judgement.
* **It does not choose the next run.** ``unsaturated`` names the range the goal document
  already froze (16-64 epoch); picking a specific configuration is Stage F's menu, the user's
  choice, not this tool's.

CPU only, no GPU, no network, no scientific claim of its own: it applies a rule that was
frozen before any of the four points ran.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The two rungs the rule compares, verbatim from the goal document §3.
LOW_POINT, LOW_UPDATES = "p4000", 32132
HIGH_POINT, HIGH_UPDATES = "p1600", 128528
LEAD = "72"
VARIABLE = "t2m"
#: The two frozen thresholds. These are the goal document's numbers, not tunable here.
UNSATURATED_DELTA, SATURATED_DELTA = 0.01, 0.005


def _digest(body):
    from training.r7_experiment import canonical_digest
    return canonical_digest({key: value for key, value in body.items() if key != "protocol_sha256"})


def load_point(attempt_dir, point, updates):
    """Read one point's registered readings, verifying its protocol and its frozen shape.

    Raises rather than returning a partial result: a verdict computed from an unverified
    reading is exactly the failure this campaign exists to avoid.
    """
    folder = Path(attempt_dir)
    readings_path = folder / "readings.json"
    protocol_path = folder / "protocol.json"
    for path in (readings_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"{point}: missing {path}")
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_sha256") != _digest(protocol):
        raise RuntimeError(f"{point}: the frozen protocol does not self-verify")
    if readings.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"{point}: readings were produced under a different protocol")
    if readings.get("curve_point") != point or readings.get("updates") != updates:
        raise RuntimeError(f"{point}: readings report {readings.get('curve_point')}/"
                           f"{readings.get('updates')}, expected {point}/{updates}")
    if readings.get("batch_size") != 4:
        raise RuntimeError(f"{point}: batch is {readings.get('batch_size')}, the curve is at 4")
    if readings.get("test_read") is not False:
        raise RuntimeError(f"{point}: test_read must be False")
    if readings.get("scientific_claim") is not False:
        raise RuntimeError(f"{point}: this campaign records no scientific claim")
    leads = readings.get("leads") or {}
    try:
        skill = leads[LEAD]["skill"][VARIABLE]
        rmse = leads[LEAD]["rmse"][VARIABLE]
    except KeyError as exc:
        raise RuntimeError(f"{point}: no {VARIABLE} entry at lead {LEAD} h") from exc
    if not isinstance(skill, (int, float)) or not math.isfinite(skill):
        raise RuntimeError(f"{point}: {VARIABLE} {LEAD} h skill is not finite: {skill!r}")
    if not isinstance(rmse, (int, float)) or not math.isfinite(rmse):
        raise RuntimeError(f"{point}: {VARIABLE} {LEAD} h RMSE is not finite: {rmse!r}")
    if readings.get("saturation_verdict") is not None:
        raise RuntimeError(f"{point}: a single point must not carry a saturation verdict")
    return {"point": point, "updates": updates, "skill": float(skill), "rmse": float(rmse),
            "protocol_sha256": protocol["protocol_sha256"],
            "attempt_dir": str(folder),
            "eval_seconds": {"train": None, "scoring": None}}


def verdict(low, high):
    """Apply the frozen rule to the two readings and nothing more.

    The three outcomes are returned verbatim as the goal document words them, including
    ``saturated``, which contradicts this campaign's original premise and is therefore a
    valid result rather than a failure to be explained away.
    """
    delta = high["skill"] - low["skill"]
    high_below_low = high["skill"] < low["skill"]
    if delta > UNSATURATED_DELTA:
        outcome = "unsaturated"
        action = ("a final run at 16-64 epoch follows; choosing its exact configuration is "
                  "Stage F's menu and the user's selection, not this tool's")
    elif delta < SATURATED_DELTA and high_below_low:
        outcome = "saturated"
        action = ("take the saturation point as the final run; no larger point is added and "
                  "no extrapolation beyond it is claimed")
    else:
        outcome = "record-only"
        action = ("recorded as it fell: the difference is neither above the unsaturated "
                  "threshold nor a saturated decline, so there is no extrapolation and no "
                  "extra point")
    return {"delta": delta,
            "unsaturated_threshold": UNSATURATED_DELTA,
            "saturated_threshold": SATURATED_DELTA,
            "high_below_low": bool(high_below_low),
            "outcome": outcome, "action": action,
            "readings": {"low": low, "high": high}}


def evaluate(low_dir, high_dir):
    low = load_point(low_dir, LOW_POINT, LOW_UPDATES)
    high = load_point(high_dir, HIGH_POINT, HIGH_UPDATES)
    body = verdict(low, high)
    body.update({
        "format": "r7-s3-saturation-verdict-v1",
        "rule": (f"delta = skill_72h({HIGH_POINT}={HIGH_UPDATES} updates) - "
                 f"skill_72h({LOW_POINT}={LOW_UPDATES} updates); delta > {UNSATURATED_DELTA} "
                 f"-> unsaturated; delta < {SATURATED_DELTA} AND high < low -> saturated; "
                 "otherwise record only"),
        "lead_hours": int(LEAD), "variable": VARIABLE,
        "scientific_claim": False, "test_read": False,
        "single_seed_caveat": ("each point is ONE seed, so the delta is a difference of two "
                               "independent runs with no pairing structure and no interval; "
                               "the rule is applied as frozen, not as a significance test"),
        "limitations": [
            "n=1 per point: the delta carries unquantified run-to-run variance",
            "GPU training is not bitwise reproducible, so a re-run of either point could move the delta",
            "the rule was frozen before any point ran and is applied without adjustment",
            "this verdict is an engineering/calibration reading, not a generalisation claim",
        ]})
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", required=True, help="attempt dir of the 4-epoch point")
    parser.add_argument("--high", required=True, help="attempt dir of the 16-epoch point")
    parser.add_argument("--out", type=Path, help="write the verdict JSON here (default: stdout)")
    args = parser.parse_args(argv)
    body = evaluate(args.low, args.high)
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out is None:
        print(text)
    else:
        # Decision 0068 / R-001: never truncate an existing artifact.
        with Path(args.out).open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}: outcome={body['outcome']} delta={body['delta']:.6f}")


if __name__ == "__main__":
    sys.exit(main())
