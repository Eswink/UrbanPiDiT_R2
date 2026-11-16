"""Exit (f): the six clauses that must pass before a GPU second is spent.

Each clause is a counter-proof of a specific way this round could measure nothing:

1. ``off_path_is_the_previous_model`` -- an explicit ``None`` must not silently build the pathway.
   ``tests/test_r7_79_typed_evidence.py::test_off_path_is_bitwise_the_previous_model`` is the
   model-side pin; this clause re-checks it under the *frozen* spec rather than a fixture's.
2. ``slot_arrival_is_diagonal`` -- a type that reaches another type's slot would make the "typed"
   arm not typed. The 4x4 arrival matrix must be exactly diagonal.
3. ``future_labels_are_unread`` -- poisoning every target must be a no-op, because evidence is
   derived from the anchor and the model's own draft only.
4. ``capacity_is_matched_where_it_must_be`` -- B and C parameter-identical, A strictly smaller.
   This is what makes ``C - B`` attributable to the type structure.
5. ``forward_arithmetic_is_matched`` -- the same input through both routers must cost the same
   FLOPs, measured with ``torch.utils.flop_counter`` (the repo's own instrument).
6. ``budget_and_closed_dose_are_registered`` -- the 4-epoch dose and the closed 2,400-update dose
   are both named, so a citation cannot quietly present this as a re-run.

Why this is a separate module: the protocol layer was 650 lines, over R-051's 600 hard cap. Splitting
by concern rather than by size keeps each file one job, and the balance round already uses this
shape (``scripts/r7_s3_balance_preflight.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _protocol():
    """Late import: the protocol owns the constants these clauses read, and it imports this module."""
    from scripts import r7_s3_typed_budget_protocol as proto
    return proto


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
    _, payload_body = proto.payload()
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
    _, payload_body = proto.payload()
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
    return _protocol().capacity()


def clause_forward_flops() -> dict:
    """(f.5) The fused and per-type paths must cost the same arithmetic on the same input."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from model.typed_evidence_r7 import GENERIC_FUSION, TYPED_ROUTING, TypedEvidenceRouter

    proto = _protocol()
    _, body = proto.payload()
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


def clause_budget_registration() -> dict:
    """(f.6) The frozen dose must be the declared one, and the closed dose must be named."""
    proto = _protocol()
    return {
        "passed": bool(proto.UPDATES == 32_132 and proto.BATCH_SIZE == 4
                       and proto.SEEDS == (41, 42, 43)
                       and proto.CLOSED_ROUND_RECORD == "s3-typed-spatial-result"),
        "updates": proto.UPDATES,
        "batch_size": proto.BATCH_SIZE,
        "epochs": proto.UPDATES / 32_131,
        "seeds": list(proto.SEEDS),
        "closed_round": {"record_id": proto.CLOSED_ROUND_RECORD,
                         "closed_updates": proto.CLOSED_ROUND_UPDATES},
        "rule": ("the closed typed-spatial round priced this arm pair at 2,400 updates; this round "
                 "changes only the dose, which must be stated in every citation"),
    }


CLAUSES = (
    ("off_path_is_the_previous_model", clause_mode_wiring),
    ("slot_arrival_is_diagonal", clause_slot_arrival),
    ("future_labels_are_unread", clause_future_is_unread),
    ("capacity_is_matched_where_it_must_be", clause_capacity),
    ("forward_arithmetic_is_matched", clause_forward_flops),
    ("budget_and_closed_dose_are_registered", clause_budget_registration),
)


def evaluate() -> dict:
    """Run every clause once, by name, so a future clause cannot be added and silently ignored."""
    clauses = {name: function() for name, function in CLAUSES}
    return {
        "clauses": clauses,
        "f_preflight_passed": bool(all(clause["passed"] for clause in clauses.values())),
    }
