"""The SAM round's parent expansion and its CI-safe update-rule control.

This round does not touch ``model/`` at all: the single factor is the training-time
update rule, and the candidate's model spec is the pinned H-D control spec **verbatim**
(no ``tendency_input``, no ``decoder_base_mode``). ``model_code_digest()`` therefore stays
at ``60de3b2d...``, and the migrated parent that the tendency-input revision already
produced can be reused as is: the committed ``scripts/migrate_r7_parent_state.py`` was
re-run under exactly this revision and its receipt is pinned here through
``scripts/r7_s3_tendency_input_parent.py``.

What changes is the *expansion*. The tendency round's target spec carried
``tendency_input=True``, so its expanded state also held
``backbone.encoder.tendency_patch.weight``; under this round's control spec the target
carries no such key and the expansion adds exactly the two published climatology-anchor
buffers. That makes the expanded state digest different from the tendency round's, so the
three per-seed digests are pinned here and re-derived, not inherited.

``update_rule_control`` is the raw counter-proof that the SAM factor is a real, read
update rule. It runs on a two-parameter quadratic module — synthetic tensors only, never
a weather instance and never a claim about trained behaviour — and checks that the
perturbation is exactly the globally rescaled registered gradient, that the assigned
gradient is the gradient at the perturbed point, that the parameters are restored bit for
bit, that ``rho = 0`` reproduces the registered gradient exactly, and that a per-parameter
normalization or a reversed sign would be a different function.
"""
from __future__ import annotations

from pathlib import Path

import torch

from scripts import r7_s3_tendency_input_parent as migrated

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_PARENT_RUN = migrated.MIGRATED_PARENT_RUN
#: The published train-only climatology table the control's anchor reads; re-exported
#: because the driver writes it into the arm contract's initialization block.
ANCHOR_TABLE_RUN = migrated.ANCHOR_TABLE_RUN
MIGRATION_RECEIPT_SHA256 = migrated.MIGRATION_RECEIPT_SHA256
CURRENT_MODEL_CODE_SHA256 = migrated.CURRENT_MODEL_CODE_SHA256
MIGRATED_PARENT_SHA256 = dict(migrated.MIGRATED_PARENT_SHA256)
MIGRATED_PARENT_STATE_DIGEST = dict(migrated.MIGRATED_PARENT_STATE_DIGEST)
ANCHOR_INIT_SHA256 = migrated.ANCHOR_INIT_SHA256
ANCHOR_KEYS = tuple(migrated.ANCHOR_KEYS)
TENDENCY_KEY = migrated.TENDENCY_KEY
ENDPOINT = migrated.ENDPOINT
#: Expanded state (parent + the two published anchor buffers) per seed, under the
#: control spec. Recomputed for this round; the tendency round's values included the
#: zero tendency patch and are deliberately not reused.
EXPANDED_STATE_SHA256 = {
    41: "00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612",
    42: "baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1",
    43: "4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7",
}
#: The declared, predeclared SAM radius; ``training/r7_sam.py`` owns the rule itself.
SAM_RHO = 0.05
SAM_EPS = 1e-12

pinned_parent = migrated.pinned_parent
anchor_payload = migrated.anchor_payload
install_parent_pins = migrated.install_parent_pins
migration_provenance = migrated.migration_provenance


def expand_parent_state(parent_state, spec, payload, *, seed):
    """Install the verified climatology table into the current-revision parent.

    Returns ``(model, state, receipt)``. The registered parent carries neither the
    climatology anchor nor any tendency patch, and this round's spec asks for no
    tendency input, so exactly the two anchor buffers are new; every parent tensor is
    installed bit for bit.
    """
    import torch

    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import make_model
    if spec.get("tendency_input", False) is not False:
        raise RuntimeError("the SAM round's candidate spec must not declare a tendency input")
    if TENDENCY_KEY in parent_state:
        raise RuntimeError("the parent already carries a tendency patch; this is not the control spec")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    incoming = target.state_dict()
    missing = [key for key in incoming if key not in parent_state]
    expected = set(ANCHOR_KEYS)
    if set(missing) != expected or len(missing) != len(expected):
        raise RuntimeError(f"the climatology anchor must add exactly {sorted(expected)}, "
                           f"got {sorted(missing)}")
    extra = [key for key in parent_state if key not in incoming]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    target.backbone.climatology_anchor.install(payload["table"])
    installed = target.state_dict()
    state = dict(parent_state)
    for key in missing:
        state[key] = installed[key].detach().clone()
    target.load_state_dict(state, strict=True)
    anchored = {key: state[key] for key in ANCHOR_KEYS}
    expanded_digest, anchor_digest = _state_digest(state), _state_digest(anchored)
    if expanded_digest != EXPANDED_STATE_SHA256[seed]:
        raise RuntimeError(f"seed {seed}: expanded state digest differs from the predeclared export")
    if anchor_digest != ANCHOR_INIT_SHA256:
        raise RuntimeError(f"seed {seed}: installed climatology bytes differ from the H-D control")
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "init_seed": int(seed), "tendency_input": False,
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": expanded_digest,
               "anchor_init_sha256": anchor_digest,
               "anchor_table_sha256": payload["pins"]["table_sha256"],
               "note": ("the model spec and the model revision are unchanged from the pinned "
                        "control; this round's factor is the training update rule only")}
    return target, state, receipt


class _ScalarResult:
    """The minimum shape ``sam_backward`` needs: an object carrying a ``loss`` tensor."""

    def __init__(self, loss):
        self.loss = loss


class _Quadratic(torch.nn.Module):
    """A two-parameter module with an exactly quadratic loss; never a weather model."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([3.0, -2.0, 0.5]))
        self.b = torch.nn.Parameter(torch.tensor([1.25]))

    def loss(self):
        return (self.a * self.a).sum() + 2.0 * (self.b * self.b).sum()

    def analytic_gradient(self):
        """The exact gradient of ``loss`` at the module's current parameters."""
        return [2.0 * self.a.detach(), 4.0 * self.b.detach()]


def _registered_gradients(model):
    """The registered single-backward gradient of the quadratic loss."""
    model.loss().backward()
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def _sam_pass(model, rho, captured):
    """One SAM update through the real rule; returns the pre-update parameters and metrics."""
    from training import r7_sam
    first = model.loss()
    first.backward()

    def recompute():
        captured["perturbed"] = [parameter.detach().clone() for parameter in model.parameters()]
        captured["loss"] = model.loss()
        return _ScalarResult(captured["loss"])

    original = [parameter.detach().clone() for parameter in model.parameters()]
    return original, r7_sam.sam_backward(model, rho=rho, first_loss=first, recompute=recompute)


def update_rule_control(rho=SAM_RHO):
    """Raw counter-proofs of the SAM update rule, on synthetic tensors only."""
    from training import r7_sam
    reference = _Quadratic()
    registered = _registered_gradients(reference)
    norm = r7_sam.global_norm(registered)
    scale = float(rho) / (norm + SAM_EPS)
    expected = [gradient * scale for gradient in registered]
    per_parameter = [gradient * (float(rho) / (r7_sam.global_norm([gradient]) + SAM_EPS))
                     for gradient in registered]
    captured = {}
    model = _Quadratic()
    original, diagnostics = _sam_pass(model, rho, captured)
    # ``perturbed`` is the stored copy, so ``w + e`` is compared to the same expression
    # rather than to the (non-associative) difference ``(w + e) - w``.
    applied = [torch.equal(perturbed, base + direction) for perturbed, base, direction
               in zip(captured["perturbed"], original, expected)]
    sam_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    perturbed_gradient = [2.0 * captured["perturbed"][0], 4.0 * captured["perturbed"][1]]
    reverse = _Quadratic()
    with torch.no_grad():
        for parameter, direction in zip(reverse.parameters(), expected):
            parameter.sub_(direction)
    reverse.loss().backward()
    reverse_gradients = [parameter.grad.detach().clone() for parameter in reverse.parameters()]
    zero_captured = {}
    zero_model = _Quadratic()
    zero_original, zero_diagnostics = _sam_pass(zero_model, 0.0, zero_captured)
    zero_gradients = [parameter.grad.detach().clone() for parameter in zero_model.parameters()]
    checks = {
        "perturbation_is_the_rescaled_registered_gradient": all(applied),
        "one_global_scalar_rescales_every_parameter": all(
            torch.equal(direction, gradient * scale)
            for direction, gradient in zip(expected, registered)),
        "perturbation_norm_matches_the_declared_rho": abs(
            r7_sam.global_norm(expected) - float(rho)) < 1e-9,
        "per_parameter_normalization_would_differ": not all(
            torch.equal(direction, value)
            for direction, value in zip(expected, per_parameter)),
        "reversed_sign_gives_a_different_update_gradient": any(
            not torch.equal(one, other)
            for one, other in zip(sam_gradients, reverse_gradients)),
        "update_gradient_is_the_gradient_at_the_perturbed_point": all(
            torch.equal(parameter.grad, analytic)
            for parameter, analytic in zip(model.parameters(), perturbed_gradient)),
        "update_gradient_differs_from_the_registered_gradient": any(
            not torch.equal(parameter.grad, gradient)
            for parameter, gradient in zip(model.parameters(), registered)),
        "parameters_restored_bit_for_bit": all(
            torch.equal(parameter.detach(), base)
            for parameter, base in zip(model.parameters(), original)),
        "restore_residual_is_exactly_zero": diagnostics["sam_restore_residual"] == 0.0,
        "perturbed_loss_is_higher_than_the_first_loss": float(
            diagnostics["sam_loss_increase"]) > 0.0,
        "zero_rho_reproduces_the_registered_gradient": all(
            torch.equal(gradient, reference_gradient)
            for gradient, reference_gradient in zip(zero_gradients, registered)),
        "zero_rho_leaves_the_parameters_untouched": all(
            torch.equal(parameter.detach(), base)
            for parameter, base in zip(zero_model.parameters(), zero_original)),
        "zero_rho_perturbation_norm_is_exactly_zero": (
            zero_diagnostics["sam_perturbation_norm"] == 0.0),
    }
    return {"synthetic": True, "rho": float(rho), "eps": SAM_EPS,
            "grad_norm": norm, "scale": scale, "checks": checks,
            "note": ("synthetic quadratic tensors only; the bitwise results describe the update "
                     "rule, not a trained-weather outcome")}
