"""The single dispatch point for Stage C's auxiliary objectives.

The runner has one hook, not two, because it sits at 593 of R-051's 600 lines: a second
``constraint_term``-shaped call would have pushed it over. Both axes therefore arrive here and
leave through one return value.

What the axes are, and why they are separate decisions:

* ``physical_constraints: "c1"`` -- decision 0092's untargeted bundle. **Not opened**: decision
  0094 measured that its divergence term charges the analysis field 10.2x more than the model,
  so its descent direction points away from truth. The axis is kept because the contract key and
  its refusal are part of the registered record; it cannot be reached by any live protocol.
* ``derivative_matching: "c2"`` -- decision 0095's derivative-domain target match. ``pred =
  target`` scores exactly zero, so its optimum set is the truth. **Closed**: decision 0109
  measured a three-seed regression (15/15 lead x seed pairings negative), so this axis is
  retained as record only and no live protocol may set it.
* ``spectral_geometry: "amse-analogue"`` -- decision 0110's limited-area spectral loss. Not a
  penalty but a change of the loss's *geometry*: the per-band amplitude/coherence coupling is
  broken so the optimum is not the damped ``sigma_X = rho`` field standard MSE prefers. It is
  the only axis that looks at the forecast field itself across bands rather than at one
  derived quantity.

All are refused together with SAM or gradient surgery: a single-factor claim is only honest if
the combination is refused rather than silently allowed. And they are refused *with each
other*, for the same reason -- an on/off difference that mixes two new terms cannot be
attributed.
"""
from __future__ import annotations

#: The contract keys this dispatcher owns, in the order their refusals are reported.
AXES = ("physical_constraints", "derivative_matching", "spectral_geometry")


def auxiliary_term(result, moved, contract):
    """The auxiliary objective for this update, or ``None`` when every axis is off.

    ``None`` is returned before constructing anything, so a run with no axis set keeps the
    registered arithmetic bit-for-bit. A caller that wants an axis off must not reach a
    "weight 0" path: that would still build a graph and still change the numbers.
    """
    autoregression = contract["autoregression"]
    enabled = [name for name in AXES if autoregression.get(name) is not None]
    if not enabled:
        return None
    if len(enabled) > 1:
        # Decision 0095: two new terms at once make the on/off difference unattributable.
        raise ValueError(f"the auxiliary axes {enabled} cannot be enabled together; "
                         "each round varies exactly one")
    if enabled[0] == "physical_constraints":
        from .r7_physical_constraints import constraint_term
        return constraint_term(result, moved, contract)
    if enabled[0] == "spectral_geometry":
        from .r7_spectral_geometry import spectral_geometry_term
        return spectral_geometry_term(result, moved, contract)
    from .r7_derivative_matching import matching_term
    return matching_term(result, moved, contract)
