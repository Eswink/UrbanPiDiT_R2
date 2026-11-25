"""Adversarial tests for the G0 headroom verdict (decision 0118).

Every test here corresponds to a way the verdict could silently authorise the
visit-gain branch when the reading does not. The two most important are
``test_a_true_b_false_refuses_the_branch`` (the ADR's first table row) and
``test_partial_seed_coverage_is_not_extrapolated`` (a missing seed must void the
statement rather than shrink the denominator).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import r7_s3_loop_depth_headroom as headroom

SHA = headroom.EXPECTED_MODEL_CODE_SHA256


def diagnostic(mse, *, max_steps=8, sha="a" * 64, unchanged=True):
    """A minimal payload shaped like ``diagnose_r7_gain.py``'s output."""
    return {"format": headroom.RECORD_FORMAT, "scientific_claim": False,
            "deployable": False, "mse_by_step": list(mse), "max_steps": max_steps,
            "checkpoint_sha256": sha, "forecaster_unchanged": unchanged}


def curve(*, k1, k2, k3, k4, k5, k6, k7, k8):
    return [k1, k2, k3, k4, k5, k6, k7, k8]


# The measured seed-41 shape: best at k=3, then monotone deterioration to k=8.
SEED41 = curve(k1=0.10852574, k2=0.10778324, k3=0.10772991, k4=0.10787800,
               k5=0.10829440, k6=0.10906053, k7=0.10998598, k8=0.11115479)


def three(curve_values, **overrides):
    return {seed: diagnostic(curve_values, **overrides) for seed in (41, 42, 43)}


class TestPerSeedClauses:
    def test_measured_shape_saturates_and_is_pathological(self):
        clauses = headroom.seed_clauses(SEED41)
        assert clauses["saturated"] is True
        assert clauses["pathology"] is True
        assert clauses["depth_usable"] is False
        assert clauses["best_step"] == 3

    def test_monotone_improvement_is_not_saturated(self):
        values = [1.0 - 0.05 * k for k in range(8)]
        clauses = headroom.seed_clauses(values)
        assert clauses["saturated"] is False
        assert clauses["pathology"] is False
        assert clauses["depth_usable"] is True

    def test_saturation_boundary_is_exclusive(self):
        # A gap of exactly 1% is not "less than 1%", so this seed is not saturated.
        values = [1.0, 1.0, 1.0, 1.0, 0.99, 0.99, 0.99, 0.99]
        assert headroom.seed_clauses(values)["saturated"] is False
        # Just under 1% saturates.
        values = [1.0, 1.0, 1.0, 1.0, 0.9901, 0.9901, 0.9901, 0.9901]
        assert headroom.seed_clauses(values)["saturated"] is True

    def test_pathology_boundary_is_exclusive(self):
        # An exact +0.1% rise does not trip the clause; anything above does.
        exact = [1.0, 1.0, 0.1, 0.1 * 1.001, 0.1, 0.1, 0.1, 0.1]
        assert headroom.seed_clauses(exact)["pathology"] is False
        above = [1.0, 1.0, 0.1, 0.1 * 1.0011, 0.1, 0.1, 0.1, 0.1]
        assert headroom.seed_clauses(above)["pathology"] is True

    def test_depth_usable_boundary_is_inclusive(self):
        exact = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98]
        assert headroom.seed_clauses(exact)["depth_usable"] is True
        short = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9801]
        assert headroom.seed_clauses(short)["depth_usable"] is False

    def test_non_finite_and_non_positive_are_rejected(self):
        for bad in (float("nan"), float("inf"), 0.0, -1.0):
            values = list(SEED41)
            values[4] = bad
            with pytest.raises(ValueError):
                headroom.seed_clauses(values)

    def test_short_curve_is_rejected(self):
        with pytest.raises(ValueError):
            headroom.seed_clauses(SEED41[:7])


class TestDecisionTable:
    def test_b_true_admits_the_branch(self):
        result = headroom.evaluate(three(SEED41))
        assert result["verdict"] == headroom.VERDICT_DECISION
        assert result["clauses"]["b_pathology"] is True

    def test_a_true_b_false_refuses_the_branch(self):
        # Saturating curve with no k3->k4 rise: the ADR forbids G1-B outright.
        values = [1.0, 0.99, 0.985, 0.9849, 0.9848, 0.9847, 0.9846, 0.9845]
        result = headroom.evaluate(three(values))
        assert result["clauses"]["a_saturated"] is True
        assert result["clauses"]["b_pathology"] is False
        assert result["verdict"] == headroom.VERDICT_SATURATED
        assert result["verdict"] != headroom.VERDICT_DECISION

    def test_c_only_is_unresolved_and_never_admits(self):
        # Improving from k4 to k8 by >2% but not saturating and not pathological.
        values = [1.0, 0.99, 0.99, 0.99, 0.98, 0.975, 0.972, 0.9695]
        result = headroom.evaluate(three(values))
        assert result["clauses"]["c_depth_usable"] is True
        assert result["clauses"]["b_pathology"] is False
        assert result["verdict"] == headroom.VERDICT_DEPTH_USABLE

    def test_b_needs_two_of_three_seeds(self):
        healthy = [1.0 - 0.05 * k for k in range(8)]
        one_bad = {41: diagnostic(SEED41), 42: diagnostic(healthy), 43: diagnostic(healthy)}
        assert headroom.evaluate(one_bad)["clauses"]["b_pathology"] is False
        two_bad = {41: diagnostic(SEED41), 42: diagnostic(SEED41), 43: diagnostic(healthy)}
        assert headroom.evaluate(two_bad)["clauses"]["b_pathology"] is True

    def test_a_needs_all_three_seeds(self):
        healthy = [1.0 - 0.05 * k for k in range(8)]
        two = {41: diagnostic(SEED41), 42: diagnostic(SEED41), 43: diagnostic(healthy)}
        assert headroom.evaluate(two)["clauses"]["a_saturated"] is False

    def test_no_matching_row_is_unresolved(self):
        # Saturating and pathological on one seed only; the other two each buy 1.5% at
        # k5->k6 (so not saturated) but only 1.5% over k4->k8 (so depth not usable).
        mixed = [1.0, 1.0, 1.0, 1.0, 1.0, 0.985, 0.985, 0.985]
        payload = {41: diagnostic(SEED41), 42: diagnostic(mixed), 43: diagnostic(mixed)}
        result = headroom.evaluate(payload)
        assert result["clauses"]["a_saturated"] is False
        assert result["clauses"]["b_pathology"] is False
        assert result["clauses"]["c_depth_usable"] is False
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED

    def test_a_perfectly_flat_curve_counts_as_saturated(self):
        # No step from k>=3 buys 1%, which is exactly what clause (a) asks for.
        result = headroom.evaluate(three([1.0] * 8))
        assert result["clauses"]["a_saturated"] is True
        assert result["verdict"] == headroom.VERDICT_SATURATED


class TestFailClosed:
    def test_partial_seed_coverage_is_not_extrapolated(self):
        payload = {41: diagnostic(SEED41), 42: diagnostic(SEED41)}
        result = headroom.evaluate(payload)
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED
        assert result["clauses"]["a_saturated"] is None
        assert any("only 2 seed" in b for b in result["blockers"])

    def test_one_blocked_seed_voids_the_whole_statement(self):
        payload = three(SEED41)
        payload[42] = diagnostic(SEED41, unchanged=False)
        result = headroom.evaluate(payload)
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED
        assert result["clauses"]["b_pathology"] is None
        assert any("forecaster_unchanged" in b for b in result["blockers"])

    def test_short_max_steps_blocks_identity(self):
        payload = three(SEED41, max_steps=4)
        result = headroom.evaluate(payload, expected_model_code_sha256=SHA)
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED
        assert any("max_steps" in b for b in result["blockers"])

    def test_missing_mse_key_blocks(self):
        payload = three(SEED41)
        del payload[43]["mse_by_step"]
        result = headroom.evaluate(payload)
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED
        assert any("seed 43" in b for b in result["blockers"])

    def test_wrong_model_code_sha_blocks_when_supplied(self):
        payload = three(SEED41)
        result = headroom.evaluate(payload, seed_model_code_sha256={41: "b" * 64, 42: SHA, 43: SHA})
        assert result["verdict"] == headroom.VERDICT_UNRESOLVED
        assert any("model_code_sha256" in b for b in result["blockers"])

    def test_loader_rejects_a_foreign_format(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"format": "something-else", "scientific_claim": False,
                                    "deployable": False}), encoding="utf-8")
        with pytest.raises(ValueError, match="format is"):
            headroom.load_diagnostic(path)

    def test_loader_rejects_a_deployable_or_claiming_oracle(self, tmp_path: Path):
        for field, value in (("deployable", True), ("scientific_claim", True)):
            body = diagnostic(SEED41)
            body[field] = value
            path = tmp_path / f"bad_{field}.json"
            path.write_text(json.dumps(body), encoding="utf-8")
            with pytest.raises(ValueError):
                headroom.load_diagnostic(path)


class TestWrite:
    def test_readings_are_exclusive(self, tmp_path: Path):
        first = headroom.readings(tmp_path, three(SEED41))
        assert first["format"] == headroom.VERDICT_FORMAT
        with pytest.raises(FileExistsError):
            headroom.readings(tmp_path, three(SEED41))

    def test_written_json_round_trips_without_nan(self, tmp_path: Path):
        headroom.readings(tmp_path, three(SEED41))
        reloaded = json.loads((tmp_path / "readings.json").read_text(encoding="utf-8"))
        assert reloaded["thresholds"]["saturation_max_relative_gain"] == 0.01
        assert all(not (isinstance(v, float) and math.isnan(v))
                   for v in reloaded["thresholds"].values())

    def test_thresholds_are_not_arguments(self):
        # The verdict is a pure function of the readings; there is no threshold knob.
        import inspect
        signature = inspect.signature(headroom.evaluate)
        for banned in ("threshold", "gain", "rise", "tol"):
            assert all(banned not in name for name in signature.parameters), banned
