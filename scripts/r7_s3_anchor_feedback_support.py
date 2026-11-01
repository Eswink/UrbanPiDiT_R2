"""R2's frozen text and the pinned cross-round identity values.

Decision 0058 admitted the climatology-anchor anomaly-feedback reevaluation; plan 0026
schedules it as Track 1's R2 and records the user's batch decision (2). This module carries
the text that must not drift between the protocol, the tests and the readings, plus the one
number set the bridge (e) needs: the registered R1 treatment arm's first-update loss, which
is the only registered artefact that ran at *this* round's compute on *this* store and
manifest. Everything here is data or declarations, never a scientific verdict.
"""
from __future__ import annotations

from scripts import r7_s3_anchor_feedback_parent as parent_pins

CLOSED_ROUND_UPDATES, CLOSED_ROUND_BATCH_SIZE = 2400, 1
CLOSED_ROUND_RECORD = "s3-anchor-isolation-result"
#: The one-factor declaration itself, so the driver and ``require_single_factor`` cannot drift.
CONTROL, CANDIDATE = "control", "candidate"
ARMS = ((CONTROL, False), (CANDIDATE, True))

#: R1 (`record:s3-fy-year-coverage-matched-result`) ran at this round's compute on this store:
#: 16,071 updates at batch 2, seeds 41/42/43, treatment manifest, spec without an anchor. This
#: round's control arm installs the same table with both consumers off, and the anchor round
#: measured its forward as bit-identical to the unanchored model, so its first update should
#: reproduce R1's bit for bit. A mismatch is an identity/engineering finding, never a verdict.
BRIDGE_RECORD = "s3-fy-year-coverage-matched-result"
BRIDGE_UPDATES, BRIDGE_BATCH_SIZE = 16071, 2
BRIDGE_UPDATE1_LOSS = {
    41: 1.1786479949951172,
    42: 1.4581815004348755,
    43: 0.8535277843475342,
}
#: Recorded alongside, never asserted: gradient norms differed in their last bits between two
#: attempts of the same protocol in the FY round, so they are evidence, not an identity.
BRIDGE_UPDATE1_GRADIENT_NORM = {
    41: 23.188753128051758,
    42: 32.805213928222656,
    43: 21.18001937866211,
}

LIMITATIONS = [
    "development screening on the 65x65 multi-year instance only (2000-2021 train / 2022 val, "
    "17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed",
    "the output prior and the proposal prior are NOT separated from each other: the anchor's "
    "presence switches both, so this round holds both off and isolates only the feedback "
    "representation",
    "the anchor table is a 48-bucket (twelve months x four synoptic hours) mean of the store's "
    "train-year frames on this grid; the closed arm's table was 16 buckets over four months on "
    "a different grid, so the two tables are not comparable",
    "both arms train on the 2000-2021 treatment manifest, so this round's control is not the "
    "closed round's control and the closed arm's readings, including its paired delta, do not "
    "transfer here in either direction",
    "the bridge is an identity check at identical compute (this round's control arm against the "
    "registered R1 treatment arm at update 1); it cannot speak about trained behaviour, and R1's "
    "own bridge failed four of five leads precisely because it compared different compute",
    "the model revision moved after R1 froze (the urban static covariate switch, off by default); "
    "the re-run migration carries the same per-seed tensor state and a different module-tree "
    "semantics digest, which every migration in this repository records rather than fails on",
    "three seeds are consistency evidence, not a significance test",
    "GPU runs are co-resident; latency and memory observations include neighbour load",
    "validation split only; the test split stays sealed until a later preregistered read",
]

DECISION_RULE = (
    "Exit (a): the candidate arm's 72 h full-grid t2m skill > 0 on seeds 41/42/43. "
    "(b): candidate minus control 72 h full-grid t2m delta RMSE < 0 on all three seeds. "
    "(c): the candidate arm's 48 h full-grid t2m skill > 0 on all three seeds. "
    "(d): candidate minus control delta RMSE at 6/12/24 h, averaged over seeds, <= 0. "
    "All four true -> ask for independent review; any false -> permanently close the "
    "climatology-anchor anomaly-feedback branch at matched compute (no table-granularity sweep, "
    "no anchor reduction, no extra seeds, no update change, no combination with a closed axis, "
    "no rename-and-reopen). The bridge (e) is an identity check, reported but never a verdict.")


def arm_spec(base_spec, payload, feedback):
    """The archived process spec with the anchor and the one-factor boolean bound in."""
    spec = dict(base_spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if (spec.get("climatology_anchor_spec") or spec.get("anomaly_feedback")
            or spec.get("anchor_output_prior") or spec.get("process_init_conditioning")
            or spec.get("typed_evidence_mode") or spec.get("global_mixing_mode")
            or spec.get("tendency_input") or spec.get("static_conditioning")):
        raise RuntimeError("the archived process spec already carries a closed-architecture switch")
    spec["climatology_anchor_spec"] = payload["spec"]
    spec["anchor_output_prior"] = False
    spec["anomaly_feedback"] = bool(feedback)
    return spec


def anchor_block(name, feedback, payload):
    """The frozen mechanism rule and its declared controls, for one arm."""
    pins = payload["pins"]
    return {"arm": name, "anomaly_feedback": bool(feedback),
            "anchor_table": {key: pins[key] for key in (
                "run", "table_sha256", "spec_sha256", "table_file_sha256", "n_buckets",
                "n_selected_steps", "bucket_counts", "source_sha256", "train_data_identity")},
            "anchor_output_prior": False,
            "decoder_and_proposal_anchor": "registered persistence anchor (last history frame)",
            "feedback": ("draft - C_valid" if feedback else "draft"),
            "new_keys": ["backbone.climatology_anchor.table", "backbone.climatology_anchor.ready"],
            "new_params": 0,
            "parameter_matched_by_construction": True,
            "shared_initialization": ("both arms install the same published table bytes; the anchor "
                                      "content digest is identical across arms and seeds"),
            "identity_control": ("with the table zeroed, draft - 0 = draft, so the candidate collapses "
                                 "onto the control bit for bit; asserted per seed by the expansion"),
            "control_identity": ("with anomaly_feedback off the anchor is read by nothing, so the "
                                 "control arm is bit-identical to the unanchored registered model "
                                 "outside the two anchor buffers")}


def arm_name(feedback):
    """The arm that carries a given ``anomaly_feedback`` value, rejecting anything else."""
    if type(feedback) is not bool:
        raise ValueError(f"anomaly_feedback must be boolean, got {type(feedback).__name__}")
    for name, value in ARMS:
        if value == bool(feedback):
            return name
    raise ValueError(f"unknown anomaly_feedback value {feedback}")


def check_model_revision(observed, *, expected=parent_pins.MODEL_CODE_SHA256):
    """The round must run on the revision that produced the migrated parent."""
    if observed != expected:
        raise RuntimeError(f"model revision {observed} is not the pinned {expected}; "
                           "commit or clear the working-tree model changes first")


def difference_keys(control, candidate):
    """Fields that differ between the two arms; must equal ARM_FACTOR_KEYS exactly."""
    return sorted(key for key in set(control) | set(candidate)
                  if control.get(key) != candidate.get(key))


def require_single_factor(control, candidate):
    """Raise unless the only arm differences are the feedback boolean and the spec it implies."""
    observed = tuple(difference_keys(control, candidate))
    if observed != ("anomaly_feedback", "model"):
        raise ValueError(f"the arms differ in {observed}, expected exactly "
                         "('anomaly_feedback', 'model')")
    control_spec, candidate_spec = control["model"]["spec"], candidate["model"]["spec"]
    if control_spec.get("anomaly_feedback") is not False \
            or candidate_spec.get("anomaly_feedback") is not True:
        raise ValueError("an arm's spec does not carry its own anomaly_feedback value")
    if control_spec.get("anchor_output_prior") is not False \
            or candidate_spec.get("anchor_output_prior") is not False:
        raise ValueError("an arm does not hold the output and proposal priors off")
    differing = sorted(key for key in set(control_spec) | set(candidate_spec)
                       if control_spec.get(key) != candidate_spec.get(key))
    if differing != ["anomaly_feedback"]:
        raise ValueError(f"the two specs differ in {differing}, expected exactly ['anomaly_feedback']")
    if control_spec.get("climatology_anchor_spec") != candidate_spec.get("climatology_anchor_spec"):
        raise ValueError("the arms do not install the same anchor table spec")
