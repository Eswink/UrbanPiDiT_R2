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


# ------------------------------------------------ the driver contract the wrapper asserts

def test_every_round_module_imports_and_exposes_the_wrapper_contract():
    """A split that leaves a name undefined dies inside a worker, after freeze has been paid for.

    Three module splits in this round each produced a NameError that only surfaced when
    ``stage prepare`` ran: first execution_files, then parent_module, and earlier the driver's
    own freeze entry point. Each cost a re-freeze and a rebuilt archive. Importing every module
    here, and asserting the five names r7_round_stages.load_driver requires, catches that class
    at test time instead of at stage time.
    """
    import importlib

    importlib.import_module("scripts.r7_s3_visit_gain_constants")
    importlib.import_module("scripts.r7_s3_visit_gain_rate")
    identity = importlib.import_module("scripts.r7_s3_visit_gain_identity")
    driver = importlib.import_module("scripts.study_r7_s3_visit_gain")
    parent = importlib.import_module("scripts.r7_s3_visit_gain_parent")
    for name in ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS"):
        assert hasattr(driver, name), f"the wrapper requires driver.{name}"
    for name in ("identity_witness", "prepared_protocol", "validate_protocol",
                 "_identity", "execution_files", "partial_receipt"):
        assert hasattr(identity, name), f"the driver imports identity.{name}"
    assert hasattr(importlib.import_module("scripts.r7_s3_visit_gain_rate"), "probe"), \
        "the driver imports rate.probe"
    for name in ("migrate", "migrate_seed", "state_equality_evidence"):
        assert hasattr(parent, name), f"the round needs parent.{name}"


def test_the_execution_inventory_covers_every_round_module():
    """A module outside the inventory can change without moving the round's identity."""
    from scripts.r7_s3_visit_gain_identity import execution_files

    listed = set(execution_files())
    for name in ("scripts/r7_s3_visit_gain_constants.py", "scripts/r7_s3_visit_gain_identity.py",
                 "scripts/r7_s3_visit_gain_rate.py", "scripts/r7_s3_visit_gain_protocol.py",
                 "scripts/r7_s3_visit_gain_parent.py", "scripts/r7_s3_visit_gain_verdict.py",
                 "scripts/study_r7_s3_visit_gain.py"):
        assert name in listed, f"{name} is not in execution_files(), so its edits are invisible"


def _minimal_protocol(tmp_path):
    """A protocol body small enough to build offline but shaped like the frozen one."""
    from scripts import r7_s3_visit_gain_protocol as proto

    body = {"protocol_sha256": "0" * 64, "seeds": list(proto.SEEDS),
            "registry": [{"arm": arm, "seed": proto.SEEDS[0],
                          "parent": {"checkpoint": "x", "sha256": "0" * 64}}
                         for arm, _ in proto.ARMS],
            "execution_files_sha256": {"scripts/r7_s3_visit_gain_rate.py": "0" * 64},
            "train_manifest_sha256": "0" * 64, "val_manifest_sha256": "0" * 64}
    return body


def test_validate_protocol_resolves_every_name_it_uses(tmp_path, monkeypatch):
    """``validate_protocol`` read an undefined SEEDS and died only at the archive stage.

    It is reachable offline: a body whose first check fails tells us the function body itself
    runs, which is what the NameError class needs. The test asserts the *error it raises*,
    not a success, so no protocol has to be valid for this to run.
    """
    import json as _json

    from scripts import r7_s3_visit_gain_identity as identity

    body = _minimal_protocol(tmp_path)
    (tmp_path / "protocol.json").write_text(_json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        identity.validate_protocol(tmp_path)


def test_prepared_protocol_resolves_every_name_it_uses(tmp_path):
    """``prepared_protocol`` read an undefined execution_files and died at the prepare stage.

    It cannot be run to completion offline: it recomputes the dataset identities, which needs
    ``outputs/``. What this asserts is that the function body gets **past its own names** and
    fails on a real dependency instead -- a ``NameError`` would be the split defect returning.
    """
    import json as _json

    from training.r7_experiment import canonical_digest

    from scripts import r7_s3_visit_gain_identity as identity

    # A deliberately wrong digest: the body then runs its first real statement and stops at
    # the mismatch check, so the unresolved-name class is still exercised without paying for
    # the dataset identity that follows it.
    preliminary = _minimal_protocol(tmp_path)
    preliminary["protocol_sha256"] = canonical_digest(preliminary)
    (tmp_path / "preparation_protocol.json").write_text(_json.dumps(preliminary),
                                                        encoding="utf-8")
    try:
        identity.prepared_protocol(tmp_path)
    except NameError as error:                      # the defect this test exists for
        raise AssertionError(f"prepared_protocol has an unresolved name: {error}") from error
    except Exception:
        pass                                        # any real dependency failure is acceptable
