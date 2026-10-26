"""Frozen prose and the bridge table for the matched-compute year-coverage round.

Split out of ``scripts/study_r7_s3_fy_year_coverage_matched.py`` when that driver reached the
R-051 600-line hard cap, the same way the year-coverage round grew
``scripts/r7_s3_fy_year_coverage_climatology.py``. This is pure data plus one derived table:
the predeclared decision rule, the round's limitations, and the registered FY
annual-coverage candidate arm's per-seed t2m RMSE values the non-scientific bridge (e)
compares against. Nothing here reads a store, a manifest or a checkpoint.

The bridge values are the registered arm's own readings, copied from
``docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md`` (index record
``record:s3-fy-annual-coverage-result``, lead 72 h seed 41 = 3.873388 K). The bridge is an
identity/engineering check only: that arm's normalization and climatology denominator differ
from this round's, so it can never be a scientific verdict.
"""
from __future__ import annotations

LIMITATIONS = [
    "development screening on the 65x65 instance only (2000-2021 train / 2022 val, 17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed",
    "both arms share the 2000-2021 normalization, so the 2017-2021 control is not byte-comparable with the month round's treatment arm, which used its own 2017-2021 normalization",
    "this round re-opens a branch decision 0054 closed, under decision 0061's named user override; it is not a new regime, because store, grid, supervision region and data span are unchanged",
    "compute is matched at updates=16071, batch=2 (updates*batch=32142 >= the 32142-window "
    "treatment manifest), so each arm sees at least one epoch; the closed round's arms saw "
    "about 7% and one third of an epoch",
    "matching compute cannot bias the reading: it only lets the extra years be used; if exit (d) is still false the negative result is stronger, not weaker",
    "the climatology denominator is fit on 2000-2021, not the month round's 2017-2021 (72 h t2m 3.9269 vs 3.8971 K), so absolute skill is not comparable across rounds; both arms share it",
    "the external bridge to the month round's candidate arm differs in normalization and climatology denominator; it is an identity/engineering check, never a scientific verdict",
    "three seeds are consistency evidence, not a significance test",
    "GPU runs are co-resident; latency/memory observations include neighbor load",
    "validation split only; the test split stays sealed until a later preregistered read",
]
DECISION_RULE = (
    "Exit (a): treatment 72 h t2m skill > 0 on seeds 41/42/43. (b): treatment minus control 72 h t2m "
    "delta RMSE < 0 on all three seeds. (c): treatment 48 h t2m skill > 0 on all three seeds. "
    "(d): treatment minus control delta RMSE at 6/12/24 h, averaged over seeds, <= 0. All four true "
    "-> ask for independent review; any false -> permanently close the training-year coverage branch, "
    "this time with compute matched. The bridge (e) is reported but is not a scientific verdict.")
#: The closed round this one re-runs, and the registered arm the bridge compares against.
CLOSED_ROUND_UPDATES = 2400
BRIDGE_RECORD, BRIDGE_UPDATES = "s3-fy-annual-coverage-result", 2400
BRIDGE_VALUES = {
    "6": [2.274617729966317, 2.330257857646797, 2.2541542668765637],
    "12": [2.854628713743481, 2.917405657538362, 2.8308260057946293],
    "24": [2.6638667900712636, 2.7153914969395143, 2.6191724913125634],
    "48": [3.4549049828817884, 3.4505076729126953, 3.3418585745051472],
    "72": [3.8733884916192376, 3.8340689074964414, 3.7694503933113017],
}


def bridge_reference():
    """The registered arm's readings and the rule-(e) noise band, per lead."""
    body = {}
    for lead, values in BRIDGE_VALUES.items():
        mean, spread = sum(values) / len(values), max(values) - min(values)
        body[lead] = {"values": list(values), "mean": mean, "spread": spread,
                      "band": max(spread, 0.05 * mean)}
    return body
