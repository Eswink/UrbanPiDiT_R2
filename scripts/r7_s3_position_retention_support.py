"""R4's frozen text and the pinned values its bridge needs.

Decision 0078 admitted the position-retention reevaluation; 0079 narrowed its question and
proved that ``position_scale`` is a forward-sensitivity knob rather than a learning-amplitude
knob; 0080 fixed the execution design (route 2: both arms start from an exactly zero query,
and the payload must be rebuilt for the 65x65 store); 0081 fixed the constant at 8000 by
measurement on real val frames.

This module carries the text that must not drift between the protocol, the tests and the
readings. Everything here is data or declarations, never a scientific verdict.
"""
from __future__ import annotations

#: The two aggregation modes. ``pooled`` is the uniform mean over the P patch positions,
#: ``spatial`` is a softmax over P with one shared learned query. This is the only factor.
CONTROL, CANDIDATE = "control", "candidate"
MODES = {CONTROL: "typed_routing", CANDIDATE: "typed_spatial"}
ARMS = ((CONTROL, MODES[CONTROL]), (CANDIDATE, MODES[CANDIDATE]))

CLOSED_ROUND_RECORD = "s3-typed-spatial-result"
CLOSED_ROUND_UPDATES, CLOSED_ROUND_BATCH_SIZE = 2400, 1

#: Decision 0081: the frozen forward-sensitivity constant, identical in both arms.
POSITION_SCALE = 8000.0
#: The admissibility band measured before the freeze; entropy below the floor means one-hot.
ENTROPY_BAND = (0.90, 0.99)
ENTROPY_ONE_HOT_FLOOR = 0.5
#: The query magnitude range the constant was judged against: H-B's three trained values
#: (0.007393 / 0.012458 / 0.008398) and the Adam random-walk expectation 0.013576.
QUERY_NORM_RANGE = (0.0074, 0.0136)

#: The published 65x65 typed-evidence sidecar (decision 0080 section 2). Its identity is
#: pinned here so the protocol, the driver and the tests cannot drift from the reviewed bytes.
SIDECAR_RUN = "outputs/r7_s3_typed_evidence_fy65_20261103_attempt01"
SIDECAR_TYPED_EVIDENCE_IDENTITY = (
    "a61d22aaa45b376970ff50479517aa816af5a56e83e9dd4258f36b9df2d81a81")
SIDECAR_PREFLIGHT_IDENTITY = (
    "6e6e224a2714de3f8d01873e4461392a9cfc990be7fd949377c8a8723fc7c6e3")
#: Must equal the FY 65x65 treatment train manifest's dataset identity.
SIDECAR_DATA_IDENTITY = (
    "81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0")

#: The round the migrated parent came from and the revision it was migrated under.
PARENT_MODEL_CODE_SHA256 = (
    "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324")

DECISION_RULE = (
    "Exit (a): the candidate arm's 72 h full-grid t2m skill > 0 on seeds 41/42/43. "
    "(b): candidate minus control 72 h full-grid t2m delta RMSE < 0 on all three seeds. "
    "(c): the candidate arm's 48 h full-grid t2m skill > 0 on all three seeds. "
    "(d): candidate minus control delta RMSE at 6/12/24 h, averaged over seeds, <= 0. "
    "All four true -> ask for independent review; any false -> permanently close the "
    "position-retention branch (no scale sweep, no extra seeds, no other aggregation family, "
    "no table-granularity change, no rename-and-reopen). The bridge (e) is an identity check, "
    "reported but never a verdict.")

LIMITATIONS = [
    "development screening on the 65x65 multi-year instance only (2000-2021 train / 2022 val, "
    "17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed",
    "position_scale is an artificial, pre-declared constant; decision 0079 measured that AdamW "
    "cancels a uniform gradient rescale, so this constant changes forward sensitivity and NOT "
    "how large the query learns. This round therefore does NOT answer how large a query the "
    "optimiser would naturally produce under this objective",
    "both arms start from an exactly zero position query, so the round measures position "
    "weighting from the pooled arm's own starting point, not from the closed arm's trained "
    "endpoint; the closed arm's readings do not transfer here in either direction and no "
    "'better than H-B' claim is possible",
    "the typed-evidence payload must be the 65x65 sidecar: the closed arm's payload carries the "
    "129x129 store's de-normalisation vectors and field statistics, and applying it here "
    "mis-decodes physical fields by 14-31% on winds and flips the sign of v500",
    "the two arms are not parameter-matched: the spatial arm carries one extra [dim] query "
    "(+192 parameters, +0.13%); position_scale is a constant and contributes no parameters",
    "the entropy admissibility band constrains only the shape of the softmax, not whether "
    "position weighting is useful; a round whose entropy leaves the band may not judge the "
    "hypothesis, but a round inside it still proves nothing on its own",
    "three seeds are consistency evidence, not a significance test",
    "GPU runs are co-resident; latency and memory observations include neighbour load",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def arm_name(mode):
    """The arm that carries a given typed-evidence mode, rejecting anything else."""
    if not isinstance(mode, str):
        raise ValueError(f"typed_evidence_mode must be a string, got {type(mode).__name__}")
    for name, value in ARMS:
        if value == mode:
            return name
    raise ValueError(f"unknown typed_evidence_mode {mode!r}")


def arm_spec(base_spec, payload, mode):
    """The archived process spec with the typed-evidence pathway bound in.

    Both arms carry the same payload and the same frozen ``position_scale``; the only
    difference is the routing mode.
    """
    spec = dict(base_spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if (spec.get("climatology_anchor_spec") or spec.get("anomaly_feedback")
            or spec.get("anchor_output_prior") or spec.get("process_init_conditioning")
            or spec.get("typed_evidence_mode") or spec.get("global_mixing_mode")
            or spec.get("tendency_input") or spec.get("static_conditioning")):
        raise RuntimeError("the archived process spec already carries a closed-architecture switch")
    if int(spec.get("anchored_processes", 0)) < 4:
        raise RuntimeError("typed evidence writes the first four anchored slots; "
                           "anchored_processes must be >= 4")
    spec["typed_evidence_mode"] = mode
    spec["typed_evidence"] = {**payload, "position_scale": POSITION_SCALE}
    return spec


#: The arm-record fields the two arms are allowed to differ in. The routing mode appears twice
#: by construction -- once as a record field so the arm is self-describing, once inside the
#: bound ``model.spec`` where ``model/`` actually reads it -- and both copies must agree.
ARM_FACTOR_KEYS = ("model", "typed_evidence_mode")


def difference_keys(control, candidate):
    """Fields that differ between the two arms; must equal ``ARM_FACTOR_KEYS`` exactly."""
    return sorted(key for key in set(control) | set(candidate)
                  if control.get(key) != candidate.get(key))


def require_single_factor(control, candidate):
    """Raise unless the only arm differences are the routing mode and the spec it implies."""
    for name, arm in ((CONTROL, control), (CANDIDATE, candidate)):
        if arm.get("typed_evidence_mode") != MODES[name]:
            raise ValueError(f"the {name} arm's record does not carry its own routing mode")
        if arm["model"]["spec"].get("typed_evidence_mode") != arm["typed_evidence_mode"]:
            raise ValueError(f"the {name} arm's record and spec disagree on the routing mode")
    control_payload = dict(control["model"]["spec"].get("typed_evidence") or {})
    candidate_payload = dict(candidate["model"]["spec"].get("typed_evidence") or {})
    if control_payload != candidate_payload:
        raise ValueError("the arms do not install the same typed-evidence payload")
    if float(control_payload.get("position_scale", 0.0)) != POSITION_SCALE:
        raise ValueError("the frozen position_scale did not reach the payload")
    observed = tuple(difference_keys(control, candidate))
    if observed != ARM_FACTOR_KEYS:
        raise ValueError(f"the arms differ in {observed}, expected exactly {ARM_FACTOR_KEYS}")
    differing = sorted(key for key in set(control["model"]["spec"]) | set(candidate["model"]["spec"])
                       if control["model"]["spec"].get(key) != candidate["model"]["spec"].get(key))
    if differing != ["typed_evidence_mode"]:
        raise ValueError(f"the two specs differ in {differing}, "
                         "expected exactly ['typed_evidence_mode']")
