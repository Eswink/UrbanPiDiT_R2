"""The balance round's frozen registry and protocol layer.

This round trains nothing: the fifteen checkpoints already exist, so the thing that must be frozen
is *which fifteen*, at *which digests*, under *which code identity*, scored on *which case set*.
``freeze`` writes that list; ``prepared_protocol`` adds the live identities and refuses to promote
if any checkpoint is unloadable; ``validate_protocol`` re-checks every digest before a single GPU
second is spent again.

Why the registry is derived rather than typed: a hand-written list of fifteen paths is exactly the
kind of thing that drifts silently. Each arm is located from the *receipt* its own round wrote,
and each checkpoint's model spec is read from the checkpoint's own ``contract`` rather than
reconstructed here, so no spec can be invented that the saved weights do not carry.

Exit (e)'s measurements live in `scripts/r7_s3_balance_preflight.py`; this module owns the
registry, the frozen constants and the re-validation. ``--preflight-only`` still runs the whole
gate, because the workflow calls this path and the arm-registry clause belongs to (e).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r7_s3_balance_preflight import (  # noqa: E402
    PROBE_FRAMES, STORE, VAL_MANIFEST, clause_coriolis, clause_finiteness,
    clause_geostrophic_coverage, clause_hydrostatic_offset, clause_units,
)

FORMAT = "r7-s3-balance-diagnostics-protocol-v1"
STAGE = "S3-BALANCE-DIAGNOSTICS"

#: The fifteen checkpoints, grouped by the round that produced them. **Arms are comparable only
#: inside their own family**: Stage A's on/off descends from one parent recipe and the urban
#: static control/treatment/placebo from another, so a cross-family difference mixes two
#: independent training runs and is reported as a level, never as a delta (decision 0114 §D5).
FAMILIES = {
    "stage_a": {
        "attempt": "outputs/r7_s3_spectral_geometry_a_20261112_attempt02",
        "arms": ("off", "on"),
        "factor": "spectral_geometry: beta_j AMSE-analogue normalisation on/off",
    },
    "urban_static": {
        "attempt": "outputs/r7_s3_urban_static_ablation_20261113_attempt01",
        "arms": ("control", "treatment", "placebo"),
        "factor": "static_conditioning: none / published 8-field / spatially permuted",
    },
}
SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
STEP_HOURS = 6
#: The identity the registered parents were trained under. Every checkpoint carries it and
#: ``load_checkpoint`` rejects a mismatch, so this is a precondition, not a preference.
PARENT_MODEL_CODE_SHA = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"

#: Measured 2026-11-14 on this store: one 12-step rollout plus a 2.1 MB npz write costs 1.09 s per
#: case (12 cases timed twice, 1.092 and 1.181 s/case). Diagnostics then cost 0.263 s/case on CPU.
MEASURED_SECONDS_PER_CASE = 1.09
MEASURED_CASES = 1447
_RERUN_SECONDS = 15 * MEASURED_CASES * MEASURED_SECONDS_PER_CASE
PLANNED_SECONDS = 14400.0                                  # the brief's soft budget: 4 h
HARD_CAP_SECONDS = 28800.0                                 # the brief's hard cap: 8 h
#: One worker holds one seed and dumps its five arms, so its own allowance is a third of the round.
PER_SEED_SECONDS = 15 * MEASURED_CASES * MEASURED_SECONDS_PER_CASE / 3 * 1.2

#: 8.9 GiB: measured on the 9 GiB-class arms of the same 65x65 dim-192 trunk. The dump holds no
#: optimizer state, so it is strictly smaller than training; the margin is the declared 512 MiB.
ESTIMATED_PEAK_BYTES = 9_569_304_576
HEADROOM_MARGIN_BYTES = 512 * 2**20

#: Device block. ``dual`` is the round-spec's ``device_mode``: three seeds spread over two cards,
#: one worker per seed, so the fifteen arms finish in about a third of a single-card wall time.
DEVICE_MODE = "dual"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"
#: This round's import closure. ``tools/r7_explicit_inventory`` walks it, so listing the entry
#: files is what makes ``execution_files_sha256`` cover the code that actually runs.
ENTRY_FILES = ("scripts/study_r7_s3_balance_round.py",
               "scripts/r7_s3_balance_protocol.py",
               "scripts/r7_s3_balance_preflight.py",
               "scripts/r7_balance_field_dump.py",
               "training/r7_balance_diagnostics.py")


def _attempt(relative: str) -> Path:
    return ROOT.joinpath(*relative.split("/"))


def _arm_checkpoint(family: str, arm: str, seed: int) -> Path:
    """The checkpoint one arm of one seed wrote, as its own round's receipt recorded it."""
    spec = FAMILIES[family]
    attempt = _attempt(spec["attempt"])
    receipt = attempt / f"seed{seed}_receipt.json"
    if not receipt.is_file():
        raise RuntimeError(f"the {family} round left no receipt at {receipt}; the arm registry "
                           "cannot be derived without it")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    record = (body.get("arms") or {}).get(arm)
    if not record:
        raise RuntimeError(f"{receipt} records no arm {arm!r}; keys are "
                           f"{sorted(body.get('arms') or {})}")
    path = Path(record["checkpoint"]).absolute()
    if not path.is_file():
        raise RuntimeError(f"{family}/{arm}/seed{seed}: the receipt names {path}, which is gone")
    return path


def registry() -> list[dict]:
    """The fifteen arms, each with the digest and code identity its checkpoint already carries."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import load_checkpoint

    arms = []
    for family, spec in FAMILIES.items():
        for arm in spec["arms"]:
            for seed in SEEDS:
                path = _arm_checkpoint(family, arm, seed)
                saved = load_checkpoint(path)
                if int(saved["updates"]) != 4000:
                    raise RuntimeError(f"{family}/{arm}/seed{seed}: {path} is at update "
                                       f"{saved['updates']}, not the frozen endpoint 4000")
                if saved["model_code_sha256"] != PARENT_MODEL_CODE_SHA:
                    raise RuntimeError(
                        f"{family}/{arm}/seed{seed}: checkpoint carries model_code_sha256 "
                        f"{saved['model_code_sha256']}, not the parent pin {PARENT_MODEL_CODE_SHA}")
                arms.append({
                    "family": family,
                    "arm": arm,
                    "seed": seed,
                    # The spec comes from the checkpoint's own contract. Reconstructing it here
                    # would let this module describe a model the saved tensors are not.
                    "model_spec": dict(saved["contract"]["model"]),
                    "updates": int(saved["updates"]),
                    "reasoning_steps": int(saved["contract"]["steps"]),
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256_file(path),
                    "checkpoint_model_code_sha256": saved["model_code_sha256"],
                })
    if len(arms) != expected_arm_count():
        raise RuntimeError(f"the registry has {len(arms)} arms, not the predeclared "
                           f"{expected_arm_count()}")
    return arms


def expected_arm_count() -> int:
    return sum(len(spec["arms"]) for spec in FAMILIES.values()) * len(SEEDS)


def clause_arm_registry() -> dict:
    """All fifteen checkpoints load strictly under one unchanged code identity.

    This is the round's hardest precondition and its cheapest check. ``load_checkpoint`` requires
    ``checkpoint['model_code_sha256'] == model_code_digest()``, so if anyone edited ``model/**``
    every arm becomes unloadable and the round cannot run at all -- the failure the wiring round
    already paid for. Checking it here means a broken registry costs seconds, not a GPU hour.
    """
    from training.r7_experiment import model_code_digest

    digest = model_code_digest()
    arms = registry()
    families: dict[str, list[str]] = {}
    for arm in arms:
        families.setdefault(arm["family"], []).append(f'{arm["arm"]}/seed{arm["seed"]}')
    return {
        "passed": bool(digest == PARENT_MODEL_CODE_SHA
                       and len(arms) == expected_arm_count()
                       and all(arm["checkpoint_model_code_sha256"] == digest for arm in arms)),
        "model_code_digest": digest,
        "parent_pin": PARENT_MODEL_CODE_SHA,
        "n_arms": len(arms),
        "expected_arms": expected_arm_count(),
        "families": {name: sorted(members) for name, members in families.items()},
        "rule": "every checkpoint must load under the current model code; any model/** edit "
                "makes all fifteen unloadable and the round impossible",
    }


def preflight() -> dict:
    """Exit (e): the five unit/coverage clauses plus the arm-registry precondition."""
    import zarr

    store = zarr.open_group(str(STORE), mode="r")
    clauses = {
        "units_are_geopotential_not_height": clause_units(store),
        "coriolis_magnitude": clause_coriolis(),
        "geostrophic_coverage_is_measured": clause_geostrophic_coverage(store),
        "hydrostatic_offset_is_a_baseline": clause_hydrostatic_offset(store),
        "fields_are_finite": clause_finiteness(store),
        "arm_registry_loads_strictly": clause_arm_registry(),
    }
    return {
        "format": "r7-s3-balance-preflight-v1",
        "store": str(STORE),
        "val_manifest": str(VAL_MANIFEST),
        "probe_frames": list(PROBE_FRAMES),
        "clauses": clauses,
        "e_preflight_passed": bool(all(clause["passed"] for clause in clauses.values())),
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "gpu_hours": 0.0,
        "limitations": [
            "this is a preflight, not a reading: it touches three train frames and no forecast",
            "the delta_HB offset is real (traced to the analysis) and is carried as a "
            "difference baseline, never as an absolute threshold",
        ],
    }


def _frozen_constants() -> dict:
    """Everything this round freezes that is not a live measurement."""
    return {
        "format": FORMAT,
        "stage": STAGE,
        "families": {name: {"arms": list(spec["arms"]), "factor": spec["factor"],
                            "attempt": spec["attempt"]}
                     for name, spec in FAMILIES.items()},
        "seeds": list(SEEDS),
        "lead_hours": list(LEADS),
        "step_hours": STEP_HOURS,
        "n_cases_expected": MEASURED_CASES,
        "val_manifest": str(VAL_MANIFEST),
        "store": str(STORE),
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "estimation": {
            "measured_seconds_per_case": MEASURED_SECONDS_PER_CASE,
            "measured_on": "2026-11-14, this store, 12 cases timed twice at 1.092 and 1.181 s",
            "n_arms": expected_arm_count(),
            "n_cases": MEASURED_CASES,
            "estimate_seconds": round(_RERUN_SECONDS, 1),
            "note": "an estimate from a 12-case measurement, not a measurement of the round",
        },
        "budgets": {
            "planned_seconds": PLANNED_SECONDS,
            "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "soft_rule": ("soft overrun continues and is recorded; only the hard cap truncates "
                          "a worker, and a truncated worker is not a pass"),
            "decision": "0030",
        },
        "exits": {
            "a": "three arms' delta_HB medians pairwise distinct on all three seeds",
            "b": "at least one (lead, variable) where the better-RMSE arm is worse balanced",
            "c": "per-arm three-seed range < the arm-to-arm gap",
            "d": "the analysis field is less unbalanced than the model field (positive control)",
            "e": "this preflight, every clause true",
            "reading_rule": (
                "(a) false means the diagnostic is dead and measured nothing; (b) false means it "
                "lives but says the same thing as RMSE. Only (a) and (b) both true licenses "
                "'physical balance is an independent evaluation axis'. A clause that had no "
                "contrast to compute is null (untested), never false and never true."),
        },
        "family_argument": (
            "arms are comparable only inside one family. stage_a's on/off descends from the "
            "p1600 parent under one recipe and differs only in whether beta_j normalisation "
            "acted during training; urban_static's control/treatment/placebo is a separate "
            "three-arm ablation from the same parent. A cross-family difference mixes two "
            "independent training runs, so cross-family rows report levels only and never a "
            "delta (decision 0114 D5)."),
        "limitations": [
            "500-250 hPa uses dry temperature: the store has no t250 or q250, so that pair must "
            "never be merged with 850-500 into a single 'mean hydrostatic deviation'",
            "delta_HB carries a +2.1..+2.9 percent systematic offset traced to the analysis "
            "itself (implied upper level 498.90 vs 500 hPa), so arm comparisons use differences",
            "the geostrophic statistic rests on 6.8-15.8 percent of the grid (Ro<0.1) and is not "
            "a domain-wide conclusion",
            "the shortest trustworthy wavelength is about 40 km, so the paper's sub-20 km claims "
            "are not reproducible at 0.25 degrees",
            "no gradient-wind balance: it needs cyclone centre identification and an N,M<=5 "
            "radial fit, and a 65x65 regional patch need not contain a cyclone. This is a "
            "limitation of the round, not a statement that the balance does not matter",
            "one model, one instance, one resolution, one region (27-43N / 107-123E)",
            "gpu re-inference is not bit-reproducible run-to-run (multi-thread reductions differ "
            "by an ulp); the round is config-reproducible and the dumped fields are the stable "
            "artefact, so every diagnostic re-run over them is exact",
        ],
    }


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol, including the fifteen-arm registry and its digests."""
    from training.r7_device_selection import protocol_fields
    from training.r7_experiment import canonical_digest

    body = {**_frozen_constants(),
            "arms": registry(),
            "gpu_uuid": GPU_UUID,
            "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **protocol_fields(DEVICE_MODE, single_uuid=GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata and the arm registry prepared in a bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(Path(output) / "preparation_protocol.json", body)
    return body


def prepared_protocol(output):
    """Add the live identities and the measured (e); a failing preflight writes no protocol."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    measurements = preflight()
    if not measurements["e_preflight_passed"]:
        raise RuntimeError(f"exit (e) failed before any GPU second was spent: "
                           f"{measurements['clauses']}")
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "val_data_identity": val_identity,
            "val_manifest_sha256": sha256_file(VAL_MANIFEST),
            "preflight": measurements,
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()}}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    """Re-check every frozen identity, the arm digests and the preflight."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("the model revision moved after freeze; the fifteen arms no longer "
                           "share one code identity")
    if model_code_digest() != PARENT_MODEL_CODE_SHA:
        raise RuntimeError("model_code_sha256 is not the parent pin any more")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("the training revision moved after freeze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    if body.get("test_read") is not False or body.get("scientific_claim") is not False:
        raise RuntimeError("this round is evaluation-only and claims nothing")
    measured = body["preflight"]
    if measured.get("e_preflight_passed") is not True:
        raise RuntimeError("the frozen preflight did not pass")
    arms = body["arms"]
    if len(arms) != expected_arm_count():
        raise RuntimeError(f"the frozen registry holds {len(arms)} arms, not "
                           f"{expected_arm_count()}")
    for arm in arms:
        if sha256_file(Path(arm["checkpoint"])) != arm["checkpoint_sha256"]:
            raise RuntimeError(f"arm {arm['family']}/{arm['arm']}/seed{arm['seed']} moved: "
                               f"{arm['checkpoint']} is not the frozen file")
    if sha256_file(VAL_MANIFEST) != body["val_manifest_sha256"]:
        raise RuntimeError("the validation manifest moved after freeze")
    return body


def execution_files():
    """Decision 0067: this round's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0], extra=ENTRY_FILES[1:])


def _write_exclusive(path: Path, body: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main(argv=None) -> int:
    """``--preflight-only`` is the cheap gate the workflow runs before freezing anything.

    It prints one JSON document and exits non-zero when exit (e) is false, so the caller reads a
    status rather than parsing prose. It never infers and never writes into the repository: the
    output path is the caller's.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.preflight_only:
        raise SystemExit("this module is the protocol layer; the round runs through "
                         "`r7_round.py stage`. Use --preflight-only for the (e) gate.")
    from scripts.r7_m3_offline import deny_network
    deny_network()
    measurements = preflight()
    payload = {"format": FORMAT, "stage": STAGE, "scientific_claim": False, "test_read": False,
               "gpu_hours": 0.0, **measurements}
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"e_preflight_passed": measurements["e_preflight_passed"],
                      "clauses": {name: clause["passed"]
                                  for name, clause in measurements["clauses"].items()},
                      "n_arms": measurements["clauses"]["arm_registry_loads_strictly"]["n_arms"],
                      "out": str(arguments.out) if arguments.out else None}, indent=2))
    return 0 if measurements["e_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
