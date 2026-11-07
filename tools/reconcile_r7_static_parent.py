"""Reconcile the registered parent into an on-arm ``static_conditioning`` state dict (decision 0097).

Why this exists. Decision 0075/0083's parent migration loads a previously exported state dict with
``strict=True``, which is right for a code-identity move. Decision 0097 measured a different
situation: the ``static_conditioning=True`` arm carries **two extra persistent buffers**
(``backbone.static_conditioning_pathway.field`` and ``.projection``) that the ``False`` arm never
constructs, so ``off_parent -> on_arm`` fails ``strict=True`` with ``Missing key(s)`` and the
reverse direction fails with ``unexpected`` keys. The registered parent has no static keys at all
(155 tensors, zero static), so it cannot be loaded into any on arm without deciding what those two
buffers should be.

**A correction measured after decision 0097 was written.** 0097 said ``strict=False`` "silently
drops" the two buffers, which reads as corruption. Measured, it does not corrupt them:
``StaticConditioning`` builds ``field`` from the published product and ``projection`` from a
``manual_seed(0)`` generator, so a fresh arm already holds the right numbers and ``strict=False``
merely leaves them unverified (confirmed here: after a ``strict=False`` load of the bare parent,
both buffers still equal the published values, and three independent builds agree). The hazard is
therefore **unverified provenance, not wrong bits** -- and unverified is still unacceptable for an
input that carries the entire static signal, but the record should not overstate it.


What this tool decides, and what it refuses to decide. It rebuilds the two buffers from the
**published** static product through ``load_static_fields()`` -- the same call the model makes --
and asserts they are bit-identical to the published stack before installing them, so the values are
**checked** rather than merely inherited from a constructor. It never reads a value out of a
checkpoint, because no checkpoint carries them.

What it does not do. It does not train, does not evaluate, produces no reading, changes no
criterion, and touches no file outside the exclusive output directory. It is 0 GPU-hours by
construction: the only tensor work is a ``load_state_dict`` on CPU.

Usage::

    .venv/bin/python tools/reconcile_r7_static_parent.py \
        --checkpoint outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/\
seed41/parent_update_0001600.pt \
        --out outputs/r7_static_parent_reconcile_20261108_attempt01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The two buffers the on arm adds. Named here so a rename in model/ fails loudly instead of
#: leaving this tool silently reconciling nothing.
BUFFER_KEYS = ("backbone.static_conditioning_pathway.field",
               "backbone.static_conditioning_pathway.projection")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state) -> str:
    """Order-independent digest over every tensor's name, dtype, shape and bytes."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _published_buffers():
    """The two buffers as the model would build them, plus the identity they came from."""
    import torch
    from model import static_conditioning_r7 as static
    manifest = static.require_static_identity()
    stack, variables = static.load_static_fields()
    # Build through the real module so the projection is the one the model would use: its seed and
    # shape are defined there, and re-deriving them here would create a second definition.
    conditioner = static.StaticConditioning(stack, dim=192, patch_size=2, periodic_width=False,
                                            variables=variables)
    built = {"backbone.static_conditioning_pathway.field": conditioner.field,
             "backbone.static_conditioning_pathway.projection": conditioner.projection}
    provenance = {
        "static_identity": manifest["static_identity"],
        "derived_statistic_not_truth": manifest["derived_statistic_not_truth"],
        "variable_count": len(variables),
        "variables": list(variables),
        # The published stack and the module's own buffer must be the same numbers, or the
        # conditioner in the model is not conditioning on the published product.
        "field_equals_published_stack": bool(torch.equal(conditioner.field, stack)),
        "field_shape": list(conditioner.field.shape),
        "projection_shape": list(conditioner.projection.shape),
    }
    return built, provenance


def _load_parent(checkpoint: Path):
    """Read the checkpoint and assert it is an off-arm parent written by this exact revision."""
    import torch
    from training.r7_experiment import model_code_digest

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "r7-local-v1":
        raise ValueError(f"unexpected checkpoint format: {payload.get('format')!r}")
    parent_state = payload["model"]
    recorded_code = payload.get("model_code_sha256")
    current_code = model_code_digest()
    if recorded_code != current_code:
        # Not a failure of this tool: it says the parent came from another revision, which is
        # exactly what 0075/0083 handle. Reported, not repaired.
        raise RuntimeError(f"parent checkpoint was written under {recorded_code}, this tree is "
                           f"{current_code}; this tool reconciles buffers, not code revisions")
    spec = payload["contract"]["model"] if "model" in payload.get("contract", {}) else None
    if spec is None:
        raise ValueError("checkpoint carries no contract.model spec to rebuild the arm from")
    spec = dict(spec)
    if spec.get("static_conditioning"):
        raise RuntimeError("the parent already carries static_conditioning; nothing to reconcile")
    return payload, parent_state, recorded_code, current_code, spec


def _probe_arms(spec, parent_state, buffers, make_model):
    """Build both arms and run the three load probes; return what each proved.

    Each probe gets a **fresh** arm: ``load_state_dict`` mutates the module, so a probe reusing an
    already-populated arm reports "nothing dropped" and would hide the hazard it exists to show.
    """
    import torch

    def fresh_on_arm():
        with torch.random.fork_rng(devices=[]):
            return make_model("process", dict(spec, static_conditioning=True))

    with torch.random.fork_rng(devices=[]):
        off_arm = make_model("process", dict(spec))
    off_keys = set(off_arm.state_dict())
    added = sorted(set(fresh_on_arm().state_dict()) - off_keys)
    if added != sorted(BUFFER_KEYS):
        raise RuntimeError(f"the on arm adds {added}, not the two buffers this tool knows about")
    if sorted(set(parent_state)) != sorted(off_keys):
        missing = sorted(off_keys - set(parent_state))
        extra = sorted(set(parent_state) - off_keys)
        raise RuntimeError(f"the parent is not exactly the off-arm key set; missing={missing[:5]} "
                           f"extra={extra[:5]}")

    armed = {**parent_state, **{key: buffers[key].detach().clone() for key in BUFFER_KEYS}}
    try:
        fresh_on_arm().load_state_dict(armed, strict=True)
        strict_error = None
    except RuntimeError as error:
        strict_error = str(error)
    if strict_error is not None:
        raise RuntimeError(f"the reconciled state dict did not load strictly: {strict_error}")
    try:
        fresh_on_arm().load_state_dict(parent_state, strict=True)
        unwrapped_ok, unwrapped_error = True, None
    except RuntimeError as error:
        unwrapped_ok, unwrapped_error = False, str(error)
    if unwrapped_ok:
        raise RuntimeError("the bare parent loaded into the on arm, so the two-buffer gap this "
                           "tool exists to close is not present; re-read decision 0097 before "
                           "trusting this receipt")

    # A strict=False load is the hazard decision 0097 forbids; record what it would swallow.
    # Direction matters: an off parent into an on arm leaves the two buffers *missing* (the model
    # has keys the state dict lacks), so the first element carries the hazard.
    dropped, surprise = fresh_on_arm().load_state_dict(parent_state, strict=False)
    if sorted(dropped) != sorted(BUFFER_KEYS):
        raise RuntimeError(f"strict=False left {sorted(dropped)} missing, expected exactly the two "
                           f"buffers (unexpected={sorted(surprise)})")
    probe = fresh_on_arm()
    probe.load_state_dict(parent_state, strict=False)
    probe_state = probe.state_dict()
    still_published = {key: bool(torch.equal(probe_state[key], buffers[key])) for key in BUFFER_KEYS}
    return {"armed": armed, "added": added, "unwrapped_error": unwrapped_error,
            "dropped": sorted(dropped), "surprise": sorted(surprise),
            "still_published": still_published}


def _receipt(payload, parent_state, recorded_code, current_code, provenance, buffers, probes,
             checkpoint):
    """The evidence record: every number here was measured above, none is retyped."""
    reconciled = probes["armed"]
    return {
        "format": "r7-static-parent-reconcile-v1",
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "gpu_hours": 0.0,
        "purpose": ("decision 0097 D1.d: build the on-arm state dict the registered parent cannot "
                    "supply, with the two static buffers taken from the published product"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "parent_updates": payload.get("updates"),
        "parent_model_code_sha256": recorded_code,
        "current_model_code_sha256": current_code,
        "parent_state_digest": _state_digest(parent_state),
        "parent_tensor_count": len(parent_state),
        "parent_static_keys": sorted(key for key in parent_state if "static" in key),
        "static": provenance,
        "buffers": {key: {"shape": list(buffers[key].shape), "dtype": str(buffers[key].dtype),
                          "sha256": hashlib.sha256(
                              buffers[key].detach().cpu().contiguous().numpy().tobytes()
                          ).hexdigest()}
                    for key in BUFFER_KEYS},
        "on_arm_added_keys": probes["added"],
        "reconciled_state_digest": _state_digest(reconciled),
        "reconciled_tensor_count": len(reconciled),
        "strict_load_of_reconciled_passed": True,
        "bare_parent_into_on_arm_failed": True,
        "bare_parent_error": (probes["unwrapped_error"] or "").split("\n")[0][:400],
        "strict_false_would_leave_missing": probes["dropped"],
        "strict_false_unexpected": probes["surprise"],
        "strict_false_buffer_values": {
            "measured": "a strict=False load of the bare parent leaves both buffers equal to the "
                        "constructed published values, so the loss is verification rather than "
                        "corruption; recorded because decision 0097 originally said 'drops'",
            "field_still_equals_published":
                probes["still_published"]["backbone.static_conditioning_pathway.field"],
            "projection_still_equals_published":
                probes["still_published"]["backbone.static_conditioning_pathway.projection"],
        },
        "limitations": [
            "This shows the on arm can be initialized from the registered parent plus the published "
            "buffers. It says nothing about whether static conditioning helps.",
            "The two buffers are a fixed function of position and are not trained.",
            "No training, evaluation or reading is produced here; gpu_hours is zero by construction.",
        ],
    }


def reconcile(checkpoint: Path, out: Path) -> dict:
    import torch
    from training.r7_experiment import make_model

    checkpoint, out = Path(checkpoint).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}; every attempt gets its own directory")
    payload, parent_state, recorded_code, current_code, spec = _load_parent(checkpoint)
    buffers, provenance = _published_buffers()
    if not provenance["field_equals_published_stack"]:
        raise RuntimeError("the module's field buffer differs from the published stack; refusing "
                           "to install buffers that are not the published product")
    probes = _probe_arms(spec, parent_state, buffers, make_model)
    receipt = _receipt(payload, parent_state, recorded_code, current_code, provenance, buffers,
                       probes, checkpoint)

    out.mkdir(parents=True)
    state_path = out / "reconciled_state.pt"
    torch.save(probes["armed"], state_path)
    receipt["reconciled_state_sha256"] = _sha256_file(state_path)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = reconcile(args.checkpoint, args.out)
    print(json.dumps({key: receipt[key] for key in
                      ("reconciled_tensor_count", "reconciled_state_digest",
                       "strict_load_of_reconciled_passed", "bare_parent_into_on_arm_failed",
                       "strict_false_would_leave_missing", "gpu_hours")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
