"""E0-1: replay the archived round-two checkpoints for correction geometry.

This is the first of the two diagnostics that the round-two and round-three
reports *pre-registered and never executed* (round-two section 11 and round-three
section 13). It answers the round-two question literally: on validation, do the
arms that add a process read to the solver (C and D) differ from the arm that only
adds the space-time inputs (B) in the **magnitude and sign of the correction the
solver proposes** - the reading "the read pulls the solver context off" predicts a
larger correction whose direction fails to oppose the current error.

Two archives are involved and they are not interchangeable:

- the checkpoints under ``outputs/r7_71_72_round_two`` were trained by the model
  code of commit ``d8aff68`` (``model_code_sha256 8d9262d1...``);
- the checkpoints under ``outputs/r7_71_72_round_three`` were trained by
  ``35c11c4`` (``f349adce...``), which is the model code at the branch head.

So ``--code-root`` selects which revision of ``model/`` and ``training/`` the
replay imports. Running a round-two checkpoint against a later revision would
measure that revision, not the run that produced the checkpoint.

Read-only by construction: the two underlying routines open their output with
``'x'``, refuse an existing path, score the validation split only, and re-check
that neither the model state digest nor the checkpoint bytes moved during the
run. Nothing here trains, steps an optimizer, or touches a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm(spec: str) -> tuple[str, int, Path]:
    """``NAME=SEED=PATH`` -> ``(name, seed, path)``; a typo must not be silent."""
    parts = spec.split("=")
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"--arm expects NAME=SEED=PATH, got {spec!r}")
    name, seed, path = parts[0], parts[1], Path(parts[2])
    if not seed.isdigit() or int(seed) < 1:
        raise ValueError(f"arm {name!r} has a non-positive-integer seed: {seed!r}")
    if not path.is_file():
        raise FileNotFoundError(f"arm {name!r} checkpoint does not exist: {path}")
    return name, int(seed), path


def parse_pair(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"--pair expects FOCUS:BASELINE, got {spec!r}")
    focus, baseline = spec.split(":", 1)
    if not focus or not baseline:
        raise ValueError(f"--pair expects FOCUS:BASELINE, got {spec!r}")
    return focus, baseline


def _scalar_mean(rows: list[dict], metric: str, variable: str) -> float | None:
    """``metric`` for one variable from the report's own per-variable summary rows."""
    values = [row[metric] for row in rows if row["variable"] == variable and metric in row]
    if not values:
        return None
    return sum(values) / len(values)


def summarise_correction(report: dict, variables: list[str]) -> dict:
    """Reduce one correction report to the quantities the question is about.

    ``update_energy`` is the squared correction magnitude and
    ``error_update_cosine`` is the cosine between the correction and the current
    error vector; their per-variable means are the "magnitude and sign" the
    round-two text asked for. Rates are means of the boolean flags, which the
    routine already stores as 0/1 per case and variable.
    """
    rows = [row for row in report["summary"] if int(row["round"]) == int(report["max_steps"])]
    per_variable = {}
    for variable in variables:
        per_variable[variable] = {
            key: _scalar_mean(rows, key, variable)
            for key in ("update_energy_mean", "error_update_cosine_mean",
                        "error_cosine_defined_mean", "wrong_direction_mean",
                        "overshoot_mean", "worsening_mean", "zero_update_mean",
                        "mse_before_mean", "mse_after_mean",
                        "previous_update_cosine_mean")
        }
    defined = [per_variable[v]["update_energy_mean"] for v in variables]
    defined = [value for value in defined if value is not None]
    cosines = [per_variable[v]["error_update_cosine_mean"] for v in variables]
    cosines = [value for value in cosines if value is not None]
    return {
        "last_round": int(report["max_steps"]),
        "per_variable": per_variable,
        "all_variable_mean_update_energy": (
            sum(defined) / len(defined) if defined else None),
        "all_variable_mean_error_update_cosine": (
            sum(cosines) / len(cosines) if cosines else None),
    }


def summarise_oracle(report: dict) -> dict:
    return {
        "mse_by_step": list(report["mse_by_step"]),
        "mean_objective_regret": float(report["mean_objective_regret"]),
        "missed_delayed_benefit_count": int(report["missed_delayed_benefit_count"]),
        "n_cases": int(report["n_cases"]),
    }


def ratio(focus: float | None, baseline: float | None) -> float | None:
    if focus is None or baseline is None or baseline == 0:
        return None
    return focus / baseline


def compare(summaries: dict, focus: str, baseline: str, variables: list[str]) -> dict:
    """Focus-minus-baseline for both metrics, per seed and pooled.

    Reported as a ratio of means as well as a difference, because the
    pre-registered sentence is about the correction getting *larger*, and the
    units of ``update_energy`` are normalized squared error - a difference in
    those units has no scale a reader can judge. No threshold is attached to
    either number here; the round's reading is sign agreement across seeds.
    """
    focus_arms = {arm: value for arm, value in summaries.items() if arm[0] == focus}
    baseline_arms = {arm: value for arm, value in summaries.items() if arm[0] == baseline}
    seeds = sorted({seed for _, seed in focus_arms} & {seed for _, seed in baseline_arms})
    if not seeds:
        raise ValueError(f"pair {focus}:{baseline} shares no seed")
    per_seed = {}
    for seed in seeds:
        f = focus_arms[(focus, seed)]["correction"]
        b = baseline_arms[(baseline, seed)]["correction"]
        per_seed[str(seed)] = {
            "focus_minus_baseline_update_energy": (
                f["all_variable_mean_update_energy"] - b["all_variable_mean_update_energy"]),
            "focus_over_baseline_update_energy": ratio(
                f["all_variable_mean_update_energy"], b["all_variable_mean_update_energy"]),
            "focus_minus_baseline_error_update_cosine": (
                f["all_variable_mean_error_update_cosine"]
                - b["all_variable_mean_error_update_cosine"]),
            "per_variable": {
                variable: {
                    "focus_minus_baseline_update_energy": (
                        f["per_variable"][variable]["update_energy_mean"]
                        - b["per_variable"][variable]["update_energy_mean"]),
                    "focus_minus_baseline_error_update_cosine": (
                        f["per_variable"][variable]["error_update_cosine_mean"]
                        - b["per_variable"][variable]["error_update_cosine_mean"]),
                    "focus_minus_baseline_wrong_direction_rate": (
                        f["per_variable"][variable]["wrong_direction_mean"]
                        - b["per_variable"][variable]["wrong_direction_mean"]),
                }
                for variable in variables
            },
        }
    def agree(key: str, inner: str | None = None) -> dict:
        values = []
        for seed in seeds:
            node = per_seed[str(seed)]
            values.append(node[inner][key] if inner else node[key])
        signs = {0 if value == 0 else (1 if value > 0 else -1) for value in values}
        return {"values": values, "same_sign": len(signs) == 1,
                "sign": signs.pop() if len(signs) == 1 else None}
    return {
        "focus": focus, "baseline": baseline, "seeds": seeds, "per_seed": per_seed,
        "update_energy_same_sign": agree("focus_minus_baseline_update_energy"),
        "error_update_cosine_same_sign": agree("focus_minus_baseline_error_update_cosine"),
    }


def import_code_root(code_root: Path, args) -> dict:
    """Import the replay revision and prove it is the one that was asked for.

    ``--code-root`` exists so a checkpoint is replayed by the revision that trained
    it. A shadowing import elsewhere on ``sys.path`` would silently measure a
    different revision, so the resolved module paths are checked rather than
    assumed, and the caller gets back the names it needs.
    """
    if not (code_root / "training" / "r7_correction_diagnostic.py").is_file():
        raise FileNotFoundError(f"no R7 code under --code-root {code_root}")
    sys.path.insert(0, str(code_root))
    import torch
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_calibration_runner import file_sha256
    from training.r7_correction_diagnostic import run_correction_diagnostic
    from training.r7_experiment import load_checkpoint, model_code_digest

    for name in ("data.r7_zarr_dataset", "training.r7_calibration_runner",
                 "training.r7_correction_diagnostic", "training.r7_experiment",
                 "model.process_forecast_r7"):
        origin = Path(sys.modules[name].__file__).resolve()
        if code_root not in origin.parents:
            raise RuntimeError(
                f"{name} resolved to {origin}, outside --code-root {code_root}; "
                "a shadowing import would replay the checkpoint with the wrong revision")
    torch.set_num_threads(int(args.threads))
    return {"dataset": ZarrAtmosWindowDataset, "sha": file_sha256,
            "correction": run_correction_diagnostic, "load": load_checkpoint,
            "digest": model_code_digest}


def measure_arms(arms, *, code, manifest, root: Path, variables, args):
    """Run both diagnostics for every declared arm and keep their provenance."""
    records, summaries = [], {}
    for name, seed, checkpoint in arms:
        digest_before = code["sha"](checkpoint)
        saved = code["load"](checkpoint)
        contract = saved["contract"]
        if contract.get("kind") != "process":
            raise ValueError(f"{name}/{seed} is not a process checkpoint")
        correction = code["correction"](
            manifest, checkpoint=checkpoint, output=root / f"correction_{name}_s{seed}.json",
            max_steps=args.max_steps_correction, max_samples=args.max_samples)
        summary = {"correction": summarise_correction(correction, variables), "oracle": None}
        if not args.skip_oracle:
            from training.r7_gain_oracle import run_oracle_diagnostic
            oracle = run_oracle_diagnostic(
                manifest, checkpoint=checkpoint, output=root / f"oracle_{name}_s{seed}.json",
                max_steps=args.max_steps_oracle, max_samples=args.max_samples,
                step_cost=0.0, device_name="cpu")
            summary["oracle"] = summarise_oracle(oracle)
        if code["sha"](checkpoint) != digest_before:
            raise RuntimeError(f"checkpoint {checkpoint} changed during the run")
        summaries[(name, seed)] = summary
        records.append({
            "arm": name, "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest_before,
            "training_model_code_sha256": saved.get("model_code_sha256"),
            "model_config": contract.get("model"),
            "data_identity": contract.get("data_identity"),
            "selected_update": saved.get("updates"),
        })
    return records, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True,
                        help="validation manifest; the routines reject any other split")
    parser.add_argument("--output", required=True,
                        help="new directory; every artifact is written below it")
    parser.add_argument("--arm", action="append", default=[], required=True,
                        metavar="NAME=SEED=PATH", help="repeat once per checkpoint")
    parser.add_argument("--pair", action="append", default=[],
                        metavar="FOCUS:BASELINE", help="declared comparison, repeatable")
    parser.add_argument("--variable", action="append", default=[],
                        help="variable to summarise; defaults to every channel")
    parser.add_argument("--max-steps-correction", type=int, default=3)
    parser.add_argument("--max-steps-oracle", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=22,
                        help="validation windows to score; the M2 val split has 22")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--code-root", default=None,
                        help="root to import model/training/data from; defaults to this file's repo")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="run only the correction-geometry diagnostic")
    parser.add_argument("--label", default="r7-e0-correction-replay-v1")
    args = parser.parse_args()

    code_root = Path(args.code_root).resolve() if args.code_root else Path(__file__).resolve().parents[1]
    code = import_code_root(code_root, args)
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dataset = code["dataset"](manifest)
    if {record["split"] for record in dataset.records} != {"val"}:
        raise ValueError("E0 scores the validation split only")
    variables = args.variable or list(dataset._store(dataset.records[0]).attrs["channels"])
    arms = [parse_arm(spec) for spec in args.arm]
    if len({(name, seed) for name, seed, _ in arms}) != len(arms):
        raise ValueError("each (arm, seed) may appear once")
    pairs = [parse_pair(spec) for spec in args.pair]
    root = Path(args.output)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    started = time.monotonic()
    records, summaries = measure_arms(arms, code=code, manifest=manifest, root=root,
                                      variables=variables, args=args)
    comparisons = [compare(summaries, focus, baseline, variables) for focus, baseline in pairs]
    result = {
        "format": args.label,
        "scientific_claim": False,
        "deployable": False,
        "training_executed": False,
        "gpu_used": False,
        "validation_only": True,
        "test_read": False,
        "code_root": str(code_root),
        "code_root_model_code_digest": code["digest"](),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": code["sha"](manifest),
        "max_steps_correction": int(args.max_steps_correction),
        "max_steps_oracle": int(args.max_steps_oracle),
        "max_samples": int(args.max_samples),
        "variables": variables,
        "arms": records,
        "summaries": {f"{name}/{seed}": value for (name, seed), value in summaries.items()},
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "comparison_rule": (
            "sign agreement across the declared seeds is consistency, not significance; "
            "no threshold, p-value or confidence interval is introduced here"),
        "limitations": [
            "one-step validation trajectory on a single 2016 winter segment; not independent "
            "weather events, not free-running skill",
            "the correction diagnostic re-runs the model forward, so these numbers are a fresh "
            "measurement on archived weights, not a replay of any previously reported number",
            "update_energy is squared error in training-normalized units; cross-variable means "
            "mix normalized units and are reported only as an index, never as Kelvin",
            "oracle depths come from the true errors of the run and are not an inference policy",
        ],
    }
    with (root / "e0_correction_replay.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({
        "complete": True, "arms": len(records), "pairs": len(comparisons),
        "elapsed_seconds": result["elapsed_seconds"], "output": str(root),
    }))


if __name__ == "__main__":
    main()
