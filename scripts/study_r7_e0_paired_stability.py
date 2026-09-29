"""E0-2: does the 0.25-0.29 K "module in place" effect survive a second run?

This is the second of the two diagnostics that the round-two and round-three
reports pre-registered and never executed (round-three section 13). The
pre-registered action is a read, not a new experiment: take round two's ``C - B``
and ``D - B`` per-seed deltas at t2m 48 h and 72 h from
``outputs/r7_71_72_round_two/paired_comparison.json``, put them beside round
three's ``E - A`` deltas from its own paired comparison, and decide whether the
0.25-0.29 K magnitude band reproduces across two independent runs.

The three pairs are only *approximate* controls for one another, and the
difference is not cosmetic:

- ``E - A`` adds the space-time conditioning module while feeding it zeros
  (round three's ``constant`` field mode), so it differs by that module and its
  +0.70% parameters;
- ``D - B`` adds the RW-A readout with the position-dependent query removed,
  +6.02% parameters;
- ``C - B`` adds the same readout with its query intact, the same +6.02%.

All three are "a module is present and carries no positional information"; none
of them is the same module, and only ``C - B`` has a positional read at all.

The script also re-checks the two numbers the round-three text quotes, because a
pre-registered question is only worth answering if its premise is right. The
archive says (``docs/R7_71_72_ROUND_THREE.md``, section 13) "t2m 48h -0.247 K,
72h -0.291 K, the three seeds share a sign". Both means are exactly reproduced
here; the sign claim is a separate question, and the same document's section 7
table marks 72 h as "all four pairs: no". Both statements cannot hold, so this
script reports each one separately instead of inheriting either.

Offline and read-only: it opens two archived JSON files and writes one. No
checkpoint, dataset, GPU or training is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# The two numbers the round-three text quotes, with the cells they belong to.
# They are archived statements to be checked, not thresholds to be met.
ARCHIVED_CLAIM = {
    "pair": "process_spacetime_constant - process_pooled",
    "variable": "t2m",
    "leads": {
        48: {"quoted_mean": -0.247, "quoted_same_sign": True, "doc_section": "13"},
        72: {"quoted_mean": -0.291, "quoted_same_sign": True,
             "doc_section": "13", "contradicted_by": "section 7 table marks 72h 'all four pairs: no'"},
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pair(spec: str) -> tuple[str, str, str]:
    """``LABEL=FOCUS:BASELINE`` -> ``(label, focus, baseline)``."""
    if "=" not in spec:
        raise ValueError(f"--pair expects LABEL=FOCUS:BASELINE, got {spec!r}")
    label, arms = spec.split("=", 1)
    if ":" not in arms:
        raise ValueError(f"--pair expects LABEL=FOCUS:BASELINE, got {spec!r}")
    focus, baseline = arms.split(":", 1)
    if not label or not focus or not baseline:
        raise ValueError(f"--pair expects LABEL=FOCUS:BASELINE, got {spec!r}")
    return label, focus, baseline


def cell_of(report: dict, pair: dict, variable: str, lead: int) -> dict:
    key = f"{lead}h|{variable}"
    if key not in pair["cells"]:
        raise KeyError(f"{key} is not reported in this paired comparison")
    cell = pair["cells"][key]
    deltas = {int(seed): float(value) for seed, value in cell["seed_deltas"].items()}
    if len(deltas) < 2:
        raise ValueError(f"{key} carries {len(deltas)} seed(s); a sign claim needs the set")
    signs = {0 if value == 0 else (1 if value > 0 else -1) for value in deltas.values()}
    return {
        "unit": cell["unit"],
        "comparator_outcome": cell["outcome"],
        "seed_deltas": {str(seed): value for seed, value in sorted(deltas.items())},
        "mean_delta": sum(deltas.values()) / len(deltas),
        "abs_mean_delta": abs(sum(deltas.values()) / len(deltas)),
        "n_seeds": len(deltas),
        "seeds_same_sign": len(signs) == 1,
        "same_sign_as_comparator": (len(signs) == 1) == (cell["outcome"] != "unresolved"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round-two", required=True, type=Path)
    parser.add_argument("--round-three", required=True, type=Path)
    parser.add_argument("--r2-pair", action="append", default=[], metavar="LABEL=FOCUS:BASELINE")
    parser.add_argument("--r3-pair", action="append", default=[], metavar="LABEL=FOCUS:BASELINE")
    parser.add_argument("--variable", default="t2m")
    parser.add_argument("--lead", action="append", type=int, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    leads = args.lead or [48, 72]
    rounds = {}
    for tag, path in (("round_two", args.round_two), ("round_three", args.round_three)):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("scientific_claim", False) is not False:
            raise ValueError(f"{path} does not declare scientific_claim false")
        rounds[tag] = {"path": str(path), "sha256": sha256_file(path), "report": report}

    declared = {"round_two": args.r2_pair, "round_three": args.r3_pair}
    table = {}
    for tag, specs in declared.items():
        if not specs:
            raise ValueError(f"no --{'r2' if tag == 'round_two' else 'r3'}-pair declared")
        report = rounds[tag]["report"]
        table[tag] = {}
        for spec in specs:
            label, focus, baseline = parse_pair(spec)
            key = f"{focus} - {baseline}"
            if key not in report["pairs"]:
                raise KeyError(f"{path_of(rounds, tag)} has no pair {key!r}")
            pair = report["pairs"][key]
            table[tag][label] = {
                "pair": key,
                "required": bool(pair.get("required", False)),
                "cells": {f"{lead}h": cell_of(report, pair, args.variable, lead) for lead in leads},
            }

    by_cell = {
        f"{lead}h": {
            tag: {
                label: {
                    "mean_delta": node["cells"][f"{lead}h"]["mean_delta"],
                    "seeds_same_sign": node["cells"][f"{lead}h"]["seeds_same_sign"],
                    "comparator_outcome": node["cells"][f"{lead}h"]["comparator_outcome"],
                }
                for label, node in labels.items()
            }
            for tag, labels in table.items()
        }
        for lead in leads
    }

    claim_check = None
    pair_key = ARCHIVED_CLAIM["pair"]
    if pair_key in rounds["round_three"]["report"]["pairs"]:
        pair = rounds["round_three"]["report"]["pairs"][pair_key]
        claim_check = {"pair": pair_key, "variable": ARCHIVED_CLAIM["variable"], "cells": {}}
        for lead, quoted in ARCHIVED_CLAIM["leads"].items():
            cell = cell_of(rounds["round_three"]["report"], pair, ARCHIVED_CLAIM["variable"], lead)
            mean_matches = abs(cell["mean_delta"] - quoted["quoted_mean"]) < 5e-4
            claim_check["cells"][f"{lead}h"] = {
                "quoted_mean": quoted["quoted_mean"], "recomputed_mean": cell["mean_delta"],
                "mean_reproduced": mean_matches,
                "quoted_same_sign": quoted["quoted_same_sign"],
                "recomputed_same_sign": cell["seeds_same_sign"],
                "sign_claim_reproduced": cell["seeds_same_sign"] is quoted["quoted_same_sign"],
                "seed_deltas": cell["seed_deltas"],
                "comparator_outcome": cell["comparator_outcome"],
                "doc_section": quoted["doc_section"],
                "contradicted_by": quoted.get("contradicted_by"),
            }

    result = {
        "format": "r7-e0-paired-stability-v1",
        "scientific_claim": False,
        "training_executed": False,
        "gpu_used": False,
        "offline_only": True,
        "variable": args.variable,
        "leads": leads,
        "inputs": {tag: {"path": node["path"], "sha256": node["sha256"]}
                   for tag, node in rounds.items()},
        "comparability": (
            "the three pairs are approximate controls, not the same manipulation: E-A adds the "
            "space-time module fed zeros (+0.70% parameters), D-B adds the RW-A readout with a "
            "pooled query (+6.02%), C-B adds the same readout position-dependently (+6.02%)"),
        "table": table,
        "by_cell": by_cell,
        "archived_claim_check": claim_check,
        "comparison_rule": (
            "sign agreement across seeds is consistency, not significance; magnitudes are compared "
            "as reproduced/not reproduced at the archived value's precision, and no significance "
            "threshold is introduced"),
        "limitations": [
            "two runs, one winter segment each, three seeds each; this is a stability observation "
            "across two protocols, never a pooled estimate",
            "the rounds have different protocol digests and different model_code_sha256, so their "
            "deltas are compared, not merged",
            "each delta is a difference of single-run validation RMSE; GPU non-determinism alone "
            "moves repeated A/B runs by ~1e-5 relative (round three section 9), far below these "
            "deltas",
            "only the t2m cells named in the pre-registered action are read here",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({
        "complete": True, "cells": len(leads),
        "by_cell": {cell: {tag: {label: {"mean": round(node["mean_delta"], 4),
                                        "same_sign": node["seeds_same_sign"]}
                                 for label, node in labels.items()}
                           for tag, labels in tags.items()}
                    for cell, tags in by_cell.items()},
        "output": str(args.out),
    }, indent=2))


def path_of(rounds: dict, tag: str) -> str:
    return rounds[tag]["path"]


if __name__ == "__main__":
    main()
