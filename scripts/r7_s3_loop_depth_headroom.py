"""G0 loop-depth headroom verdict: the frozen (a)(b)(c) clauses of decision 0118.

Reads the JSON that ``diagnose_r7_gain.py`` writes (``training/r7_gain_oracle.py``,
format ``r7-gain-oracle-diagnostic-v1``) and applies the decision table from
``docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md``.

Three properties matter more than the arithmetic and each has a test:

* the thresholds are the ADR's literals, not arguments - there is no CLI knob that
  could widen them after seeing a reading;
* a missing seed, a non-finite MSE, ``max_steps < 8`` or a failed identity check
  makes the whole verdict ``unresolved``; a subset of seeds is never extrapolated;
* ``null`` is never read as permission. Clauses that cannot be evaluated are
  reported as ``None`` and the table falls through to ``unresolved``.

Output is written with ``open("x")`` so a second run has to use a new directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Frozen by decision 0118 Context/Decision 2. Changing any of these is a new ADR.
MIN_SEEDS_FOR_STATEMENT = 3
SATURATION_MAX_RELATIVE_GAIN = 0.01
PATHOLOGY_MIN_RELATIVE_RISE = 0.001
DEPTH_USABLE_MIN_RELATIVE_GAIN = 0.02
SATURATION_FROM_STEP = 3          # k in {3,4,5,6,7}: gaps m_k -> m_{k+1}
REQUIRED_MAX_STEPS = 8
EXPECTED_MODEL_CODE_SHA256 = (
    "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324")

VERDICT_DECISION = "admit-g1b"
VERDICT_SATURATED = "saturated-negative"
VERDICT_DEPTH_USABLE = "depth-usable-unresolved"
VERDICT_UNRESOLVED = "unresolved"

RECORD_FORMAT = "r7-gain-oracle-diagnostic-v1"
VERDICT_FORMAT = "r7-loop-depth-headroom-verdict-v1"


def _relative_gain(earlier: float, later: float) -> float:
    """``(earlier - later) / earlier``; positive means the later depth is better."""
    return (earlier - later) / earlier


def load_diagnostic(path: Path) -> dict:
    """Load one ``diagnose_r7_gain.py`` result and check its own self-declarations."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: diagnostic must be a JSON object")
    if payload.get("format") != RECORD_FORMAT:
        raise ValueError(f"{path}: format is {payload.get('format')!r}, expected {RECORD_FORMAT}")
    if payload.get("scientific_claim") is not False:
        raise ValueError(f"{path}: scientific_claim must be false")
    if payload.get("deployable") is not False:
        raise ValueError(f"{path}: a retrospective oracle is never deployable")
    return payload


def seed_clauses(mse_by_step: Sequence[float]) -> dict:
    """The three per-seed clause values, on one seed's ``mse_by_step``.

    Returns relative quantities only, so a caller can read *why* a seed landed
    where it did without re-deriving the arithmetic.
    """
    values = [float(v) for v in mse_by_step]
    if len(values) < REQUIRED_MAX_STEPS:
        raise ValueError(f"need at least {REQUIRED_MAX_STEPS} steps, got {len(values)}")
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError("mse_by_step must be finite and strictly positive")

    gaps = [_relative_gain(values[k - 1], values[k])
            for k in range(SATURATION_FROM_STEP, REQUIRED_MAX_STEPS)]
    # (a) no single step from k>=3 on gives >=1% relative gain, i.e. every gap is < 1%.
    saturated = all(gap < SATURATION_MAX_RELATIVE_GAIN for gap in gaps)
    # (b) k=3 -> k=4 relative *rise* above 0.1%, i.e. gain below -0.1%.
    pathology = _relative_gain(values[2], values[3]) < -PATHOLOGY_MIN_RELATIVE_RISE
    # (c) k=4 -> k=8 relative gain of at least 2%.
    depth_usable = _relative_gain(values[3], values[7]) >= DEPTH_USABLE_MIN_RELATIVE_GAIN
    return {
        "saturated": saturated,
        "pathology": pathology,
        "depth_usable": depth_usable,
        "gains_from_step3": gaps,
        "k3_to_k4_relative_change": _relative_gain(values[2], values[3]),
        "k4_to_k8_relative_gain": _relative_gain(values[3], values[7]),
        "best_step": min(range(1, len(values) + 1), key=lambda k: values[k - 1]),
    }


def _identity_failure(payload: Mapping[str, Any], expected_sha256: str) -> Optional[str]:
    """The cheapest reasons a reading may not be used at all."""
    if payload.get("forecaster_unchanged") is not True:
        return "forecaster_unchanged is not true"
    if int(payload.get("max_steps", 0)) < REQUIRED_MAX_STEPS:
        return f"max_steps is {payload.get('max_steps')}, need {REQUIRED_MAX_STEPS}"
    observed = payload.get("checkpoint_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return "checkpoint_sha256 is missing or malformed"
    return None


def evaluate(diagnostics: Mapping[int, Mapping[str, Any]], *,
             expected_model_code_sha256: str = EXPECTED_MODEL_CODE_SHA256,
             seed_model_code_sha256: Optional[Mapping[int, str]] = None) -> dict:
    """Apply the frozen decision table to one diagnostic per seed."""
    seeds = sorted(int(s) for s in diagnostics)
    per_seed: dict[str, dict] = {}
    blockers: list[str] = []

    if len(seeds) < MIN_SEEDS_FOR_STATEMENT:
        blockers.append(f"only {len(seeds)} seed(s) supplied, need {MIN_SEEDS_FOR_STATEMENT}")

    for seed in seeds:
        payload = diagnostics[seed]
        failure = _identity_failure(payload, expected_model_code_sha256)
        if failure is not None:
            blockers.append(f"seed {seed}: {failure}")
            continue
        if seed_model_code_sha256 is not None:
            observed = seed_model_code_sha256.get(seed)
            if observed != expected_model_code_sha256:
                blockers.append(f"seed {seed}: model_code_sha256 {observed!r} is not the "
                                f"frozen {expected_model_code_sha256}")
                continue
        try:
            per_seed[str(seed)] = seed_clauses(payload["mse_by_step"])
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(f"seed {seed}: {error}")

    # A subset is never extrapolated: any blocked seed voids the statement.
    clauses: dict[str, Optional[bool]] = {"a_saturated": None, "b_pathology": None,
                                          "c_depth_usable": None}
    if not blockers and len(per_seed) == len(seeds):
        three = len(seeds) == MIN_SEEDS_FOR_STATEMENT
        a_count = sum(1 for v in per_seed.values() if v["saturated"])
        b_count = sum(1 for v in per_seed.values() if v["pathology"])
        c_count = sum(1 for v in per_seed.values() if v["depth_usable"])
        clauses["a_saturated"] = three and a_count == len(seeds)
        clauses["b_pathology"] = b_count * 3 >= 2 * len(seeds)
        clauses["c_depth_usable"] = c_count * 3 >= 2 * len(seeds)
        clauses.update({"a_seed_count": a_count, "b_seed_count": b_count,
                        "c_seed_count": c_count, "n_seeds": len(seeds)})

    verdict, reason = _decision(clauses, blockers)
    return {"format": VERDICT_FORMAT, "verdict": verdict, "reason": reason,
            "clauses": clauses, "per_seed": per_seed, "blockers": blockers,
            "thresholds": {"saturation_max_relative_gain": SATURATION_MAX_RELATIVE_GAIN,
                           "pathology_min_relative_rise": PATHOLOGY_MIN_RELATIVE_RISE,
                           "depth_usable_min_relative_gain": DEPTH_USABLE_MIN_RELATIVE_GAIN,
                           "saturation_from_step": SATURATION_FROM_STEP,
                           "required_max_steps": REQUIRED_MAX_STEPS},
            "scientific_claim": False, "test_read": False,
            "objective_is_plus_6h": True}


def _decision(clauses: Mapping[str, Optional[bool]], blockers: Sequence[str]) -> tuple[str, str]:
    """The ADR's table, in its frozen priority order."""
    if blockers:
        return VERDICT_UNRESOLVED, "identity or coverage blocked: " + "; ".join(blockers)
    a, b, c = clauses["a_saturated"], clauses["b_pathology"], clauses["c_depth_usable"]
    if None in (a, b, c):
        return VERDICT_UNRESOLVED, "a clause could not be evaluated"
    if b:
        return VERDICT_DECISION, ("k3->k4 rises beyond 0.1% on at least two thirds of the "
                                  "seeds: the recurrence over-shoots, so the visit-gain "
                                  "hypothesis may be derived")
    if a:
        return VERDICT_SATURATED, ("no step from k>=3 buys 1% on any seed and k3->k4 does "
                                   "not rise: more visits to the same cell are useless; the "
                                   "residual-scaling branch is refused")
    if c:
        return VERDICT_DEPTH_USABLE, ("depth is usable but neither saturated nor pathological: "
                                      "only early-exit/policy work is admissible; do not reopen K")
    return VERDICT_UNRESOLVED, "the clause pattern matches no row of the frozen table"


def readings(write_dir: Path, diagnostics: Mapping[int, Mapping[str, Any]], **kwargs) -> dict:
    """Evaluate once and write ``readings.json`` exclusively."""
    verdict = evaluate(diagnostics, **kwargs)
    path = Path(write_dir) / "readings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return verdict
