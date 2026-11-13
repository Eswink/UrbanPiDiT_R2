"""Stage A's protocol-level constants and the dose decision frozen before training.

Split out of ``test_r7_s3_spectral_geometry_a.py`` to stay under R-021's 400-line target: the
parent file crossed it when the two CUDA counterproofs were added. Nothing was dropped or
weakened -- every test below moved verbatim, and the parent keeps the rest.

The dose tests are here because plan 0031 §6 risk 3 reserves the question "is the AMSE term a
replacement objective or an additive correction" to the A round and requires it be settled
*before* freezing. ``dose_decision`` is where that settlement lives, so its tests are a
protocol-level concern rather than a property of the loss arithmetic.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ protocol-level constants

def test_the_round_freezes_the_counts_decision_0112_fixed():
    """4,000 updates, three seeds, batch 4, two arms: the objective names them all."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    assert proto.UPDATES == 4000
    assert tuple(proto.SEEDS) == (41, 42, 43)
    assert proto.BATCH_SIZE == 4
    assert proto.STEPS == 4
    assert tuple(proto.LEADS) == (6, 12, 24, 48, 72)
    assert len(proto.PREFLIGHT_CASES) == 8


def test_the_pinned_beta_covers_every_channel_and_is_not_constant():
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    assert len(proto.BETA_PINNED) == 17
    values = list(proto.BETA_PINNED.values())
    assert max(values) > min(values)


def test_the_beta_pin_matches_the_step_c_artifact():
    """The frozen list must be the Step C reading, not a hand-typed approximation."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    artifact = Path("/tmp/stepc3.json")
    if not artifact.is_file():
        pytest.skip("the Step C artifact is a /tmp file; re-run scripts/r7_s3_gap_decay_and_beta.py")
    import json
    record = json.loads(artifact.read_text(encoding="utf-8"))
    channels = record["beta"]["channels"]
    for name, pinned in proto.BETA_PINNED.items():
        measured = channels[name]["beta"]
        assert abs(measured - pinned) / pinned < 1e-5, f"{name}: {measured} vs {pinned}"


def test_the_protocol_is_validation_only():
    """Every frozen body carries the two flags the campaign requires."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    body = proto.frozen_constants()
    assert body["scientific_claim"] is False
    assert body["test_read"] is False
    assert body["beta_normalisation"]["frozen"] is True
    assert body["beta_normalisation"]["recomputed_during_training"] is False
    assert "AMSE-analogue" in body["beta_normalisation"]["method_name"]


def test_the_off_arm_declares_no_beta():
    """The control arm's contract must not carry a weight it never uses."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    arms = proto.frozen_constants()["arms"]
    assert arms[proto.CONTROL]["spectral_geometry"] is None
    assert arms[proto.CANDIDATE]["spectral_geometry"] == "amse-analogue"


# --------------------------------------------------------- the dose decision, fixed in advance

def test_the_dose_decision_uses_no_divisor():
    """Equation (10) is a pure sum; a divisor would be a fitted constant."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    decision = proto.dose_decision([58.0, 95.0])
    assert decision["divisor"] == 1
    assert decision["decision"] == "frozen-before-training"


def test_the_dose_decision_reports_the_objective_dominating_regime():
    """At the measured 58-95 the MSE is a residual, so the on arm trains on the AMSE term."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    decision = proto.dose_decision([58.016, 95.217])
    assert decision["on_arm_is_the_paper_regime"] is True
    assert "objective-dominating" in decision["regime"]
    low, high = decision["residual_mse_share"]
    assert 0.0 < low < high < 0.05


def test_the_dose_decision_can_name_the_inert_regime():
    """The same function must be able to report the C1 failure mode, or it decides nothing."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    inert = proto.dose_decision([0.01, 0.5])
    assert inert["on_arm_is_the_paper_regime"] is False
    assert "inert" in inert["regime"]
    mixed = proto.dose_decision([0.5, 5.0])
    assert mixed["on_arm_is_the_paper_regime"] is False
    assert "mixed" in mixed["regime"]
