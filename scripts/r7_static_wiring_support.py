"""Protocol blocks and exit readings for the static-conditioning wiring round (decision 0069).

Kept out of the driver so both stay inside R-051, and so the exit arithmetic can be tested
without building a model. Nothing here writes: the driver owns every file.

The four exits are engineering assertions, not a scientific read (decision 0069 clause 3):

- **W1** the static identity reaches ``protocol_sha256`` and nothing else: mutating one
  character of ``static_identity`` must change the protocol digest, while ``data_identity``
  - which describes the store and its manifests - must not move.
- **W2** the off path is the pre-existing computation. Its evidence is a pre/post-edit
  comparison produced by ``scripts/r7_static_wiring_reference.py``; this module only records
  the outcome the driver is handed, so the round cannot silently restate it.
- **W3** the on path runs: both arms score five leads and the two ``boundary_rmse.csv`` files
  have the same number of rows.
- **W4** the field the model conditions on is the published one: an independent,
  loop-based re-derivation from the ``.npz`` must equal the module's vectorised features.

A false exit ends the round as ``negative``; four true exits are ``engineering-positive`` and
still say nothing about whether the covariates help.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

#: The two arms differ in exactly this: the trunk switch. Same store, same manifests, same
#: seed, same updates. ``static_identity`` is the *identity* of the shared field, not a factor.
ARMS = ("control", "treatment")
FACTOR_NAME = "time-invariant urban static conditioning"
#: Held fixed across arms; recorded so a reader can see what the single factor rests on.
CONTROLLED = ("store", "train manifest", "val manifest", "seed", "updates", "batch size",
              "lr", "warmup", "clip", "physical weights", "model spec except static_conditioning",
              "initialization (seeded, no parent)", "evaluation sample cap")

LIMITATIONS = (
    "scientific_claim: false; this round proves a pathway exists and is auditable, not that "
    "urban static covariates improve forecasts",
    "one seed, two arms: an on/off difference here is not seed-consistency evidence and is "
    "not read as a mechanism result",
    "the arm diff is the switch only, but the static term is a fixed deterministic projection "
    "with parameter_delta = 0, so the path cannot learn which covariate matters; a null is "
    "therefore not evidence that the covariates are uninformative",
    "evaluation is capped well below the frozen val cohort to stay inside the round's wall "
    "clock; the t2m numbers are a pathway check, not a skill reading",
    "the round changes model/** bytes, so model_code_sha256 differs from 60de3b2d... and the "
    "registered migrated parents cannot be loaded by this code; R2-R5 need a fresh parent "
    "migration, which this round does not perform",
    "the static fields are derived statistics aggregated from native WorldCover/GHSL rasters, "
    "not observational truth, and LandScan is absent so there is no population cross-check",
)

DECISION_RULE = (
    "W1 static identity reaches protocol_sha256 only; W2 the off path reproduces the "
    "pre-change fingerprint exactly; W3 both arms score five leads with equal row counts; "
    "W4 the model's field equals an independent re-derivation of the published npz. "
    "Any false -> negative, and no on/off comparison is read. All true -> "
    "engineering-positive, with no scientific claim."
)


def static_block(manifest: dict, fields, variables, *, parameter_delta: int,
                 expansion: int, scales: dict, source_paths: dict) -> dict:
    """The frozen ``static`` section. Its digest feeds ``protocol_sha256``, nothing else."""
    components = manifest["identity_components"]
    return {
        "static_identity": components["static_identity"],
        "manifest_sha256_note": "static_identity already covers source + aggregation + output",
        "output_sha256": components["output_sha256"],
        "source_sha256": components["source_sha256"],
        "recheck_max_relative_delta": components["recheck_max_relative_delta"],
        "grid": {"points": manifest["grid"]["points"], "spacing_deg": manifest["grid"]["spacing_deg"],
                 "lat_order": manifest["grid"]["lat_order"], "lon_order": manifest["grid"]["lon_order"]},
        "variables": list(variables), "variable_count": len(variables),
        "expansion_per_variable": expansion, "feature_count": expansion * len(variables),
        "scales": dict(scales),
        "parameter_delta": int(parameter_delta),
        "parameter_form": ("no trained parameter: the field and a fixed seeded projection are "
                           "buffers, so on/off is an information difference, not capacity"),
        "time_invariance": ("the term is a function of position only; it is added at every lead "
                            "and every reasoning step and never reads a target or a clock"),
        "binding": ("static_identity is bound into protocol_sha256 only; data_identity keeps "
                    "describing the store and its manifests (decision 0069 clause 1)"),
        "paths": dict(source_paths),
        "derived_statistic_not_truth": manifest["derived_statistic_not_truth"],
    }


def exits(observations: dict) -> dict:
    """The predeclared W1-W4 conjunction, read only from measured observations."""
    w1, w2, w3, w4 = (observations[key] for key in ("w1", "w2", "w3", "w4"))
    passed = {"w1": bool(w1["passed"]), "w2": bool(w2["passed"]),
              "w3": bool(w3["passed"]), "w4": bool(w4["passed"])}
    return {**passed, "all": all(passed.values()),
            "verdict": "engineering-positive" if all(passed.values()) else "negative",
            "scientific_claim": False,
            "note": ("four true exits prove the pathway exists, binds its identity and does not "
                     "disturb the off path; they say nothing about whether the covariates help")}


def independent_patch_mean(stack, *, token_hw, patch_size: int, scales=None) -> list[list[float]]:
    """Loop-based patch average of the raw ``.npz`` arrays, written without torch.

    This is deliberately the slow, obvious implementation: it is the cross-check for W4, so it
    must not share code with the vectorised path it is checking. Padding replicates the edge
    the way ``pad_patch_grid`` does, which is the only convention the two are allowed to share
    (it is the encoder's, so both have to match it).

    ``scales`` is applied to each cell *before* averaging, in the same float64 precision the
    caller's comparison uses. Applying it afterwards would be mathematically equal and
    numerically different, and the point of this function is an exact comparison.
    """
    channels, rows, columns = len(stack), len(stack[0]), len(stack[0][0])
    token_rows, token_columns = token_hw
    pad_rows, pad_columns = token_rows * patch_size, token_columns * patch_size
    if pad_rows < rows or pad_columns < columns:
        raise ValueError("token grid is smaller than the native grid")
    factors = [1.0] * channels if scales is None else [float(value) for value in scales]
    if len(factors) != channels:
        raise ValueError(f"{len(factors)} scales for {channels} variables")
    out = []
    for channel in range(channels):
        grid = stack[channel]
        factor = factors[channel]
        for token_row in range(token_rows):
            for token_column in range(token_columns):
                total, count = 0.0, 0
                for row in range(token_row * patch_size, (token_row + 1) * patch_size):
                    for column in range(token_column * patch_size,
                                        (token_column + 1) * patch_size):
                        total += float(grid[min(row, rows - 1)][min(column, columns - 1)]) / factor
                        count += 1
                out.append(total / count)
    # Transposed to [N, K] the way the model stacks it: one row per token, one column per
    # variable, so element (i, k) is variable k averaged inside token i.
    return [[out[channel * token_rows * token_columns + index]
             for channel in range(channels)] for index in range(token_rows * token_columns)]


def rows_of(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


#: Two float64 sums of the same four values can disagree in the last bit when they accumulate
#: in a different order, which a sequential loop and a batched ``mean`` always do. This bound
#: is stated rather than hidden: the *bitwise* claim W4 makes is about the buffer against the
#: published ``.npz`` (exact, and asserted separately); the loop re-derivation is a second,
#: independent path and is judged at the float64 reduction-order bound.
REDUCTION_ORDER_TOLERANCE = 1e-12


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
