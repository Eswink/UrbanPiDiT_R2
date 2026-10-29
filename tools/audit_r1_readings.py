"""Audit an R1 ``readings.json`` before anyone reads a verdict out of it.

The round's verdict comes from ``readings.json``'s ``exit_conditions`` and nowhere else:
no hand arithmetic, no mixing numbers across attempts. That rule is easy to state and easy
to violate by accident, because the file is a nested structure where a plausible-looking
verdict can be reconstructed from the wrong field -- ``full_t2m_skill`` is the candidate
arm's own skill while ``delta_treatment_minus_control`` is the paired difference, and they
are different quantities at different leads.

This tool refuses to say whether the round passed. It checks that the file is *readable as
evidence* before a human reads a verdict from it:

* all three seeds are present, with both arms and all five leads;
* the key set is exactly what the driver writes, so a truncated file cannot pass;
* each seed's two arms trained on different manifests (the round's whole point), and the
  per-seed pairing is present;
* ``exit_conditions`` is a bool conjunction over exactly (a)-(d), and its ``all`` agrees
  with the conjunction of its own parts -- a disagreement means the file was edited;
* ``bridge_e`` declares itself non-scientific, and its per-lead ``passed`` agrees with
  ``|difference| <= band`` recomputed from its own numbers.

Exit codes: 0 the file is readable as evidence, 1 an invariant is broken, 2 no file.

    python tools/audit_r1_readings.py --readings outputs/<round>/readings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEADS = ("6", "12", "24", "48", "72")
SEEDS = ("41", "42", "43")
ARMS = ("control", "treatment")
EXIT_KEYS = ("a", "b", "c", "d")


def _arm_faults(seed: str, arm: str, body: dict) -> list[str]:
    faults: list[str] = []
    if not isinstance(body.get("train_data_identity"), str) or not body["train_data_identity"]:
        faults.append(f"seed {seed}/{arm}: train_data_identity missing or not a string")
    for table in ("full_t2m_rmse", "full_t2m_skill"):
        values = body.get(table)
        if not isinstance(values, dict):
            faults.append(f"seed {seed}/{arm}: {table} is not an object")
            continue
        missing = [lead for lead in LEADS if lead not in values]
        if missing:
            faults.append(f"seed {seed}/{arm}: {table} lacks leads {missing}")
        for lead, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                faults.append(f"seed {seed}/{arm}: {table}[{lead}] is not a number")
    return faults


def audit(readings: dict) -> list[str]:
    """Every fault that makes the file unreadable as evidence; empty means it is readable."""
    faults: list[str] = []
    if readings.get("scientific_claim") is not False:
        faults.append("scientific_claim must be false")
    seeds = readings.get("seeds")
    if not isinstance(seeds, dict):
        return faults + ["seeds is not an object"]
    missing = [seed for seed in SEEDS if seed not in seeds]
    if missing:
        faults.append(f"seeds missing: {missing} -- a verdict must not be read yet")
    unexpected = sorted(set(seeds) - set(SEEDS))
    if unexpected:
        faults.append(f"seeds carries unexpected entries: {unexpected}")

    for seed in SEEDS:
        entry = seeds.get(seed)
        if not isinstance(entry, dict):
            if seed in seeds:
                faults.append(f"seed {seed}: entry is not an object")
            continue
        arms = entry.get("arms")
        if not isinstance(arms, dict):
            faults.append(f"seed {seed}: arms is not an object")
            continue
        for arm in ARMS:
            if not isinstance(arms.get(arm), dict):
                faults.append(f"seed {seed}: arm {arm!r} missing")
            else:
                faults.extend(_arm_faults(seed, arm, arms[arm]))
        delta = entry.get("delta_treatment_minus_control")
        if not isinstance(delta, dict) or any(lead not in delta for lead in LEADS):
            faults.append(f"seed {seed}: delta_treatment_minus_control lacks a lead")
        if isinstance(arms.get("control"), dict) and isinstance(arms.get("treatment"), dict):
            control = arms["control"].get("train_data_identity")
            treatment = arms["treatment"].get("train_data_identity")
            if isinstance(control, str) and control == treatment:
                faults.append(f"seed {seed}: both arms share train_data_identity {control[:12]} "
                              f"-- the round's factor is not present in this seed")

    conditions = readings.get("exit_conditions")
    if not isinstance(conditions, dict):
        faults.append("exit_conditions is not an object")
    else:
        for key in EXIT_KEYS:
            if not isinstance(conditions.get(key), bool):
                faults.append(f"exit_conditions[{key!r}] is not a bool")
        parts = [conditions.get(key) for key in EXIT_KEYS]
        if all(isinstance(value, bool) for value in parts):
            if conditions.get("all") != all(parts):
                faults.append(f"exit_conditions.all is {conditions.get('all')} but the "
                              f"conjunction of (a)-(d) is {all(parts)} -- the file was edited")
        means = conditions.get("seed_mean_delta_6_12_24")
        if isinstance(means, dict):
            for lead in ("6", "12", "24"):
                if not isinstance(means.get(lead), (int, float)) or isinstance(means.get(lead), bool):
                    faults.append(f"exit_conditions.seed_mean_delta_6_12_24[{lead!r}] not a number")

    bridge = readings.get("bridge_e")
    if not isinstance(bridge, dict):
        faults.append("bridge_e is not an object")
        return faults
    if bridge.get("scientific") is not False:
        faults.append("bridge_e.scientific must be false: it is an engineering check")
    leads = bridge.get("leads")
    if not isinstance(leads, dict):
        faults.append("bridge_e.leads is not an object")
        return faults
    for lead in LEADS:
        entry = leads.get(lead)
        if not isinstance(entry, dict):
            faults.append(f"bridge_e.leads[{lead!r}] missing")
            continue
        difference, band = entry.get("difference"), entry.get("band")
        if isinstance(difference, (int, float)) and isinstance(band, (int, float)):
            expected = abs(difference) <= band
            if entry.get("passed") is not expected:
                faults.append(f"bridge_e.leads[{lead!r}].passed is {entry.get('passed')} "
                              f"but |{difference:+.6f}| <= {band:.6f} is {expected}")
    return faults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readings", required=True, type=Path)
    args = parser.parse_args(argv)
    path = Path(args.readings)
    if not path.is_file():
        print(f"no readings file: {path}", file=sys.stderr)
        return 2
    try:
        readings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot parse {path}: {exc}", file=sys.stderr)
        return 2
    faults = audit(readings)
    if faults:
        print(f"UNREADABLE AS EVIDENCE ({len(faults)} fault(s)):")
        for fault in faults:
            print(f"  - {fault}")
        return 1
    conditions = readings["exit_conditions"]
    print("READABLE AS EVIDENCE: three seeds, both arms, five leads, exit_conditions and "
          "bridge_e internally consistent.")
    print(f"  exit_conditions: a={conditions['a']} b={conditions['b']} c={conditions['c']} "
          f"d={conditions['d']} all={conditions['all']}")
    print(f"  verdict (verbatim from the file): {conditions.get('verdict')}")
    print("  This tool does not decide the science; read the verdict from the file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
