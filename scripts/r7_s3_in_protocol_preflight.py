"""Exit (f): the six clauses that must pass before a GPU second is spent.

Each clause is a counter-proof of a specific way this round could measure nothing:

1. ``off_path_is_the_previous_model`` -- an explicit ``None`` must not silently build the pathway.
2. ``slot_arrival_is_diagonal`` -- a physical field that reaches another field's slot would make the
   ``typed`` arm not typed.
3. ``future_labels_are_unread`` -- poisoning every target must be a no-op.
4. ``capacity_is_matched_where_it_must_be`` -- B and C parameter-identical, A strictly smaller.
5. ``forward_arithmetic_is_matched`` -- the same input through both routers costs the same FLOPs.
6. ``envelope_covers_the_treated_rate`` -- the per-seed envelope is a declared multiple of the
   quantity recomputed from the **treated** rate, and the hard cap covers both seed waves.

Clause 6 is the one the previous round could not have detected. It compares two independently
stated numbers (the frozen envelope and a requirement recomputed from a measured rate), where the
earlier guard compared the envelope against a copy of the envelope and was therefore green by
construction while the run was structurally doomed.

The mechanism assertions live here rather than in the protocol layer for two reasons: R-051 caps a
file at 600 lines, and these functions are all "prove the design is wired as declared" rather than
"which parent, which digest". The protocol layer re-exports them so callers see one surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The store's channel order, pinned. ``families`` derives the partition from it, so a store that
#: reordered its channels would move which variables are tested together.
EXPECTED_CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                     "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")


def _protocol():
    """Late import: the protocol owns the constants these clauses read, and it imports this module."""
    from scripts import r7_s3_in_protocol_protocol as proto
    return proto


def payload() -> tuple[dict, dict]:
    """The published 65x65 typed-evidence sidecar payload and its own metadata."""
    import json

    import numpy as np
    import zarr

    from data.preprocess.r7_typed_evidence_scale import load_typed_evidence_scale

    proto = _protocol()
    meta = load_typed_evidence_scale(proto.SIDECAR_RUN)
    root = zarr.open_group(str(meta["store"]), mode="r")
    channels = list(root.attrs["channels"])
    if channels != list(meta["channels"]):
        raise ValueError("the sidecar channel order does not match the store")
    declared = json.loads((proto.SIDECAR_RUN / "typed_evidence_scale.json").read_text(encoding="utf-8"))
    if declared.get("data_identity") != proto.SIDECAR_DATA_IDENTITY:
        raise RuntimeError("the typed-evidence sidecar is not the pinned 65x65 build")
    body = {"channels": channels,
            "denorm_mean": [float(v) for v in np.asarray(root["normalization_mean"][:])],
            "denorm_std": [float(v) for v in np.asarray(root["normalization_std"][:])],
            "field_scale": [float(v) for v in meta["runtime_scale"]],
            "field_mean": [float(v) for v in meta["runtime_mean"]],
            "field_std": [float(v) for v in meta["runtime_std"]],
            "field_active": [bool(v) for v in meta["active_mask"]]}
    return meta, body


def store_channels() -> list[str]:
    """The store's channel names, in the order the model and the tables use."""
    import zarr

    root = zarr.open_group(str(_protocol().STORE), mode="r")
    channels = [str(name) for name in root.attrs["channels"]]
    if tuple(channels) != EXPECTED_CHANNELS:
        raise RuntimeError(f"the store's channel order is {channels}, not the pinned "
                           f"{list(EXPECTED_CHANNELS)}")
    return channels


def families() -> dict:
    """The two families, derived from the store's channel order and the pathway's declaration.

    The primary family is the set the physical pathway reads, which the model declares rather than
    this module choosing it. That is the whole point: the partition cannot be re-drawn after a
    reading exists, because it is a property of the mechanism.
    """
    from model.r7_process_tensor_diagnostics import TYPED_EVIDENCE_CHANNELS

    proto = _protocol()
    channels = store_channels()
    primary = tuple(TYPED_EVIDENCE_CHANNELS)
    missing = [name for name in primary if name not in channels]
    if missing:
        raise RuntimeError(f"the pathway reads channels the store does not carry: {missing}")
    if tuple(primary) != proto.FAMILY_PRIMARY_CHANNELS:
        raise RuntimeError("the pathway's declared channels moved; the primary family is frozen")
    secondary = tuple(name for name in channels if name not in primary)
    if len(primary) + len(secondary) != len(channels):
        raise RuntimeError("the family partition does not cover the store's channels exactly once")
    return {
        proto.FAMILY_PRIMARY_LABEL: {
            "channels": list(primary), "grids": len(primary) * len(proto.LEADS),
            "rule": ("the channels typed_local_fields consumes, as the model declares them; fixed "
                     "before any reading exists")},
        proto.FAMILY_SECONDARY_LABEL: {
            "channels": list(secondary), "grids": len(secondary) * len(proto.LEADS),
            "rule": ("every other channel, tested as its own family so the primary family is not "
                     "diluted by 13 unrelated variables")},
    }


def envelope() -> dict:
    """One seed's requirement, recomputed from the treated rate and compared to the envelope.

    A worker runs **three** arms and scores five leads each, so the quantity is three arms, not one.
    The produced rates are an upper bound on six measurements: the three completed ``generic`` arms
    (1.4043 / 1.4312 / 1.4271 s/update over a full run) and the three truncated ``typed`` arms'
    last-4,000-update marginal rates (1.4565 / 1.4618 / 1.4618).
    """
    proto = _protocol()
    per_arm = proto.UPDATES * proto.TREATED_SECONDS_PER_UPDATE + proto.MEASURED_SECONDS_PER_EVALUATION
    need = len(proto.ARMS) * per_arm
    return {
        "treated_seconds_per_update": proto.TREATED_SECONDS_PER_UPDATE,
        "incumbent_seconds_per_update": proto.INCUMBENT_SECONDS_PER_UPDATE,
        "seconds_per_evaluation": proto.MEASURED_SECONDS_PER_EVALUATION,
        "updates": proto.UPDATES,
        "arms_per_seed": len(proto.ARMS),
        "need_seconds": round(need, 1),
        "need_hours": round(need / 3_600.0, 3),
        "per_seed_seconds": proto.PER_SEED_SECONDS,
        "margin": round(proto.PER_SEED_SECONDS / need, 4),
        "declared_margin": proto.ENVELOPE_MARGIN,
        "sufficient": bool(proto.PER_SEED_SECONDS >= proto.ENVELOPE_MARGIN * need),
        "hard_cap_seconds": proto.HARD_CAP_SECONDS,
        "waves": 2,
        "hard_cap_covers_waves": bool(proto.HARD_CAP_SECONDS >= 2 * proto.PER_SEED_SECONDS),
        "rule": ("a truncated worker is not a pass, so the envelope is sized from the treated rate "
                 "with a declared multiple; the guard test compares the envelope against this "
                 "requirement, never against a copy of the envelope"),
    }


def capacity() -> dict:
    """(f.4) B and C parameter-identical; A strictly smaller by exactly the pathway's tensors."""
    from training.r7_experiment import make_model

    proto = _protocol()
    meta, body = payload()
    counts = {}
    for name, mode in proto.ARMS:
        spec = proto.arm_spec(mode)
        if mode is not None:
            spec = dict(spec, typed_evidence=dict(body))
        model = make_model("process", spec)
        counts[name] = int(sum(parameter.numel() for parameter in model.parameters()))
    pathway = counts[proto.GENERIC] - counts[proto.INCUMBENT]
    generic_equals_typed = counts[proto.GENERIC] == counts[proto.TYPED]
    incumbent_is_smaller = counts[proto.INCUMBENT] < counts[proto.GENERIC]
    return {
        "passed": bool(generic_equals_typed and incumbent_is_smaller and pathway > 0),
        "parameters": counts,
        "pathway_parameters": pathway,
        "pathway_fraction_of_incumbent": pathway / counts[proto.INCUMBENT],
        "generic_equals_typed": bool(generic_equals_typed),
        "incumbent_strictly_smaller": bool(incumbent_is_smaller),
        "rule": ("B and C must be parameter-identical so B->C is attributable to the type "
                 "structure; A must be strictly smaller so the pathway's own capacity cost is a "
                 "reported number, never an unstated confound"),
        "sidecar_identity": proto.SIDECAR_DATA_IDENTITY,
        "sidecar_fields": list(meta["fields"]),
    }


def clause_mode_wiring() -> dict:
    """(f.1) Switching the pathway off must be the previous implementation exactly.

    Each model is built inside its own seeded fork: building two models in one fork would let the
    second consume the first one's random draws, and the mismatch would be the harness's, not the
    pathway's.
    """
    import torch

    from training.r7_experiment import make_model

    proto = _protocol()

    def build(spec):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            return make_model("process", spec)

    incumbent = build(proto.arm_spec(None))
    explicit_off = build({**proto.arm_spec(None), "typed_evidence_mode": None})
    left, right = incumbent.state_dict(), explicit_off.state_dict()
    identical = (sorted(left) == sorted(right)
                 and all(torch.equal(left[key], right[key]) for key in left))
    return {
        "passed": bool(identical and not any("typed_evidence" in key for key in left)),
        "n_tensors": len(left),
        "rule": ("the incumbent arm must be byte-for-byte the model every registered run trained; "
                 "an explicit None must not silently build the pathway"),
    }


def clause_slot_arrival() -> dict:
    """(f.2) Perturbing one physical field must move only its own anchored slot."""
    import torch

    from model.typed_evidence_r7 import TYPED_ROUTING, TypedEvidenceRouter

    proto = _protocol()
    _, payload_body = payload()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        router = TypedEvidenceRouter(192, 2, mode=TYPED_ROUTING, **payload_body)
    router.eval()
    base_fields = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        baseline = router.token_evidence(base_fields)
    arrival = []
    for index in range(4):
        perturbed = base_fields.clone()
        perturbed[0, index] += 1.0
        with torch.no_grad():
            moved = router.token_evidence(perturbed)
        arrival.append([float(value) for value in (moved - baseline).abs().amax(dim=(0, 2))])
    diagonal = all(arrival[type_][slot] > 0.0 for type_ in range(4) for slot in [type_])
    off_diagonal = all(arrival[type_][slot] == 0.0
                       for type_ in range(4) for slot in range(4) if slot != type_)
    return {
        "passed": bool(diagonal and off_diagonal),
        "arrival_matrix": arrival,
        "rule": ("each type's evidence must reach only the anchored slot that belongs to it; a "
                 "non-diagonal entry would mean the 'typed' arm is not evaluating types"),
    }


def clause_future_is_unread() -> dict:
    """(f.3) Poisoning the labels must not move the pathway's contribution."""
    import torch

    from model.typed_evidence_r7 import TYPED_ROUTING, TypedEvidenceRouter

    proto = _protocol()
    _, payload_body = payload()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        router = TypedEvidenceRouter(192, 2, mode=TYPED_ROUTING, **payload_body)
    router.eval()
    anchor = torch.randn(1, 17, 8, 8)
    draft = torch.randn(1, 17, 8, 8)
    coordinates = {"latitude": torch.linspace(43.0, 27.0, 8),
                   "longitude": torch.linspace(107.0, 123.0, 8)}
    with torch.no_grad():
        first = router.evidence(dict(coordinates), anchor=anchor, draft=draft)
        poisoned = dict(coordinates,
                        atmos_target=torch.full_like(anchor, float("nan")),
                        process_targets=torch.full((1, 4, 8), float("nan")))
        second = router.evidence(poisoned, anchor=anchor, draft=draft)
    return {
        "passed": bool(torch.equal(first, second)),
        "max_abs_difference": float((first - second).abs().max()),
        "rule": ("evidence reads only the anchor state and the model's own draft; a label must "
                 "never reach it, so poisoning every target must be a no-op"),
    }


def clause_capacity() -> dict:
    """(f.4) B and C parameter-identical; A strictly smaller by exactly the pathway's tensors."""
    return capacity()


def clause_forward_flops() -> dict:
    """(f.5) The fused and per-type paths must cost the same arithmetic on the same input."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from model.typed_evidence_r7 import GENERIC_FUSION, TYPED_ROUTING, TypedEvidenceRouter

    _, body = payload()
    fields = torch.randn(2, 4, 8, 8)
    counts = {}
    for label, mode in (("generic", GENERIC_FUSION), ("typed", TYPED_ROUTING)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            router = TypedEvidenceRouter(192, 2, mode=mode, **body)
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            router.token_evidence(fields)
        counts[label] = int(counter.get_total_flops())
    counts["difference"] = counts["generic"] - counts["typed"]
    return {
        "passed": bool(counts["typed"] > 0 and counts["generic"] == counts["typed"]),
        "forward_flops": counts,
        "rule": ("the fused and per-type paths must perform the same arithmetic on the same input; "
                 "the projection shapes are equal, so any difference would be a second factor"),
    }


def clause_envelope() -> dict:
    """(f.6) The envelope must cover the requirement recomputed from the treated rate."""
    measured = envelope()
    return {
        "passed": bool(measured["sufficient"] and measured["hard_cap_covers_waves"]),
        "envelope": measured,
        "rule": ("the previous round sized its envelope from the incumbent rate and its guard read "
                 "the same constant, so the guard could not fail while the run was structurally "
                 "doomed. This clause compares the envelope to a requirement recomputed from the "
                 "treated rate"),
    }


CLAUSES = (
    ("off_path_is_the_previous_model", clause_mode_wiring),
    ("slot_arrival_is_diagonal", clause_slot_arrival),
    ("future_labels_are_unread", clause_future_is_unread),
    ("capacity_is_matched_where_it_must_be", clause_capacity),
    ("forward_arithmetic_is_matched", clause_forward_flops),
    ("envelope_covers_the_treated_rate", clause_envelope),
)


def evaluate() -> dict:
    """Run every clause once, by name, so a future clause cannot be added and silently ignored."""
    clauses = {name: function() for name, function in CLAUSES}
    return {
        "clauses": clauses,
        "f_preflight_passed": bool(all(clause["passed"] for clause in clauses.values())),
    }
