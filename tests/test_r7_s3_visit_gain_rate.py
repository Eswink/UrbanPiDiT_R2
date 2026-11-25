"""Counter-proofs for the probe's rate estimator (decision 0120).

The attempt01 probe reported 2.48-2.65 s/update and refused the round; the same arm on an idle
host runs at 1.10. The difference was a concurrent test suite starving the trainer's four-worker
loader. These tests pin the estimator that replaced the cumulative rate, so a later edit cannot
quietly turn it back into a number that folds an early disturbance into the answer forever.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import r7_s3_visit_gain_protocol as proto
from scripts.r7_s3_visit_gain_rate import marginal_rate, quiet_machine


def _progress(tmp_path, rows):
    import json as _json
    path = tmp_path / "progress.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _progress(tmp_path, rows):
    import json as _json
    path = tmp_path / "progress.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_marginal_rate_separates_a_slow_start_from_a_fast_tail(tmp_path):
    """The estimator must report the tail, not the average, or a disturbance is permanent.

    Decision 0120's whole point: a cumulative rate folds an early stall into the answer forever,
    while the marginal rate reports what the loader is sustaining now. The synthetic rows here
    have a deliberately slow first half and a fast second half, so the two numbers must differ
    and must differ in the expected direction.
    """
    from scripts.r7_s3_visit_gain_identity import marginal_rate

    rows = [{"update": u, "elapsed_seconds": (u * 5.0 if u <= 200 else 1000.0 + (u - 200) * 1.0)}
            for u in range(10, 401, 10)]
    measured = marginal_rate(_progress(tmp_path, rows))
    assert measured["cumulative_seconds_per_update"] > 2.0
    assert measured["marginal_seconds_per_update"] == pytest.approx(1.0, abs=0.01)
    assert measured["marginal_over_cumulative"] < 0.5


def test_marginal_rate_matches_cumulative_when_the_rate_is_flat(tmp_path):
    """The idle-machine case: a flat run must give the same answer either way."""
    from scripts.r7_s3_visit_gain_identity import marginal_rate

    rows = [{"update": u, "elapsed_seconds": u * 1.10} for u in range(10, 401, 10)]
    measured = marginal_rate(_progress(tmp_path, rows))
    assert measured["marginal_seconds_per_update"] == pytest.approx(1.10, abs=1e-6)
    assert measured["cumulative_seconds_per_update"] == pytest.approx(1.10, abs=1e-6)
    assert measured["marginal_over_cumulative"] == pytest.approx(1.0, abs=1e-6)


def test_marginal_rate_is_null_rather_than_zero_on_a_starved_log(tmp_path):
    """One row is not a rate. It must be null, never a fabricated number."""
    from scripts.r7_s3_visit_gain_identity import marginal_rate

    measured = marginal_rate(_progress(tmp_path, [{"update": 10, "elapsed_seconds": 5.0}]))
    assert measured["marginal_seconds_per_update"] is None
    assert measured["cumulative_seconds_per_update"] is None


def test_marginal_rate_ignores_rows_without_an_update_or_elapsed(tmp_path):
    from scripts.r7_s3_visit_gain_identity import marginal_rate

    rows = [{"update": u, "elapsed_seconds": u * 1.10} for u in range(10, 410, 10)]
    rows.insert(3, {"note": "no update here"})
    rows.append({"update": 500})                      # no elapsed_seconds
    measured = marginal_rate(_progress(tmp_path, rows))
    assert measured["last_update"] == 400
    assert measured["marginal_seconds_per_update"] == pytest.approx(1.10, abs=1e-6)


def test_quiet_machine_reports_a_load_per_cpu_or_nulls():
    """The snapshot must be recordable on any host, including one without loadavg."""
    from scripts.r7_s3_visit_gain_identity import quiet_machine

    snapshot = quiet_machine()
    assert set(snapshot) == {"loadavg_1_5_15", "cpu_count", "load_per_cpu"}
    if snapshot["loadavg_1_5_15"] is not None:
        assert len(snapshot["loadavg_1_5_15"]) == 3
        assert snapshot["load_per_cpu"] is not None


def test_the_frozen_envelope_is_sized_from_the_measured_rate():
    """Decision 0120 replaced a guess (1.55) with a measurement (1.10) plus 18 percent."""
    measured = proto.envelope()
    assert proto.TREATED_SECONDS_PER_UPDATE == 1.30
    assert measured["sufficient"] is True
    assert measured["margin"] >= proto.ENVELOPE_MARGIN
    # The requirement must be recomputed, not asserted against the constant it came from.
    need = proto.ARMS.__len__() * (proto.UPDATES * proto.TREATED_SECONDS_PER_UPDATE
                                   + proto.MEASURED_SECONDS_PER_EVALUATION)
    assert measured["need_seconds"] == pytest.approx(need, rel=1e-9)


def test_the_measured_rate_has_headroom_over_every_measurement():
    """1.30 must stay above the measured 1.075-1.110 range, or the constant is a guess again."""
    measured_values = (1.075, 1.098, 1.110, 1.109, 1.0796, 1.0921)
    assert proto.TREATED_SECONDS_PER_UPDATE > max(measured_values) * 1.1
