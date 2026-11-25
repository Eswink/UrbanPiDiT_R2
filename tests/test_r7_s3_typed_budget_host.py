"""Driver and worker-layer counter-proofs for the typed-evidence budget round.

Split from ``test_r7_s3_typed_budget.py`` to stay inside R-051's 600-line cap. These cover the
stages that run *after* the GPU budget is spent -- assembling the arm tables, evaluating the
clauses, writing ``readings.json``, and verifying the code archive -- so a defect in any of them
would otherwise surface only once Segment A had been paid for. The balance round lost its whole
GPU round exactly this way.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import require_archived_revision  # noqa: E402

LEADS = (6, 12, 24, 48, 72)
SEEDS = (41, 42, 43)

#: ``readings()`` reaches ``protocol.capacity()``, which reads the production spec from a
#: checkpoint written under the pre-visit-gain revision. The synthetic receipts below are the
#: point of these tests; the archived spec is only a lookup they pass through.
PRODUCTION_REFERENCE = (ROOT / "outputs/r7_s3_spectral_geometry_a_20261112_attempt02"
                        / "seed41/training/off/update_0004000.pt")

def _write_provenance(folder, *, mse_t2m, leads, channels=("t2m", "z500"), n=40):
    """Per-initialisation rows, which clause (a) needs for its block-bootstrap interval.

    Without them the interval cannot be computed and (a) is null -- so a fixture that only writes
    boundary_rmse.csv would (correctly) fail to clear the pair.
    """
    import datetime as _dt
    import json as _json

    rows = []
    start = _dt.datetime(2022, 1, 1, 6)
    for index in range(n):
        init = start + _dt.timedelta(hours=6 * index)
        rows.append({"init_time": init.isoformat(),
                     "valid_times": [(init + _dt.timedelta(hours=int(lead))).isoformat()
                                     for lead in leads],
                     "mse": [[float(mse_t2m) / n, 400.0] for _lead in leads],
                     "cumulative_reasoning_steps": 4})
    (folder / "provenance.json").write_text(_json.dumps({
        "channels": list(channels), "units": ["K", "m**2 s**-2"],
        "lead_hours": [int(lead) for lead in leads], "step_hours": 6, "split": "val",
        "evaluation_manifest_sha256": "0" * 64, "n_evaluated": len(rows),
        "initializations": rows}), encoding="utf-8")


def test_readings_assembles_full_arm_tables_and_evaluates_a_f_through_e(tmp_path):
    """The end-to-end reading computation, on synthetic receipts, before any GPU is spent.

    The balance round spent its whole GPU budget and then died 7.2 s into this stage. Calling
    ``readings()`` directly here exercises the real path -- receipt parsing, boundary-CSV lookup,
    clause evaluation -- rather than only the writer, so a missing key or a wrong argument shows up
    at zero cost.
    """
    import csv as _csv

    from scripts import study_r7_s3_typed_budget as driver
    require_archived_revision(PRODUCTION_REFERENCE, what="the typed-evidence reading tables")

    # One receipt per arm per seed, each with five leads of boundary CSVs. t2m is made to fall
    # monotonically from incumbent to generic to typed so (a) and (b) are both genuine.
    base = {"incumbent": 2.000, "generic": 1.990, "typed": 1.980}
    # z500 is deliberately made to move the WRONG way on the treated arms (+10 percent each), so
    # the (e) guard has something real to reject rather than a zero difference it always accepts.
    z500 = {"incumbent": 50000.0, "generic": 55000.0, "typed": 55000.0}
    for seed in SEEDS:
        for arm, _mode in driver.proto.ARMS:
            lead_dirs = {}
            for lead in LEADS:
                folder = tmp_path / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
                folder.mkdir(parents=True)
                value = base[arm] + lead / 1000.0
                with (folder / "boundary_rmse.csv").open("w", newline="", encoding="utf-8") as h:
                    writer = _csv.DictWriter(h, fieldnames=["lead_hours", "variable", "region",
                                                            "rmse", "unit"])
                    writer.writeheader()
                    writer.writerow({"lead_hours": lead, "variable": "t2m", "region": "full",
                                     "rmse": repr(value), "unit": "K"})
                    writer.writerow({"lead_hours": lead, "variable": "z500", "region": "full",
                                     "rmse": repr(z500[arm] + lead * 100.0),
                                     "unit": "m**2 s**-2"})
                _write_provenance(folder, mse_t2m=value ** 2, leads=LEADS)
                lead_dirs[str(lead)] = {"dir": str(folder), "n_evaluated": 1447,
                                        "n_available_windows": 1447, "elapsed_seconds": 1.0,
                                        "boundary_rmse_csv_sha256": "0" * 64,
                                        "provenance_sha256": "0" * 64}
            receipt = {"seed": seed, "arm": arm, "typed_evidence_mode": dict(driver.proto.ARMS)[arm],
                       "parameters": driver.proto.capacity()["parameters"][arm],
                       "seconds_per_update": 1.1536, "evaluations": lead_dirs}
            (tmp_path / f"seed{seed}_{arm}_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8")

    body = {"protocol_sha256": "0" * 64, "updates": 32_132, "batch_size": 4,
            "closed_dose": {"record_id": "s3-typed-spatial-result", "updates": 2400},
            "independent_arm_argument": "synthetic", "limitations": ["synthetic test fixture"],
            "preflight": {"clauses": {"capacity_is_matched_where_it_must_be": True,
                                      "forward_arithmetic_is_matched": True}},
            "train_data_identity": "0" * 64,
            "val_manifest": str(driver.proto.VAL_MANIFEST)}
    result = driver.readings(tmp_path, body)

    # The five keys the receipt checker demands must be present on the real object, not just in
    # the writer's argument.
    for key in ("scientific_claim", "test_read", "protocol_sha256", "exit_conditions", "seeds"):
        assert key in result, key
    assert result["complete"] is True and result["missing"] == []
    assert len(result["arms"]["typed"]) == len(SEEDS)
    values = result["exits"]["values"]
    # t2m falls typed < generic < incumbent on every lead, so both pairs must clear.
    assert values["a_typed_minus_generic"] is True, values
    assert values["b_generic_minus_incumbent"] is True, values
    # (d) has no 16-epoch reading, so it must be null, never false and never true.
    assert values["d_budget_direction"] is None, values
    # z500 is constructed to worsen with the arms, so the guard must fail here; the point is that
    # it is computed rather than skipped.
    assert values["e_z500_72h_guard"] is False


def test_the_reading_worker_branch_runs_and_writes_readings_json(tmp_path, monkeypatch):
    """Drive ``worker(phase="reading")`` itself, which has never been executed.

    The balance round's equivalent branch died on a ``NameError`` 7.2 s in -- after its entire GPU
    round was paid for -- because the writer was imported inside one function and used in another.
    Calling the branch with the protocol and GPU pin stubbed out proves the branch reaches its
    writer and leaves a well-formed ``readings.json`` behind, at zero cost.
    """
    import csv as _csv
    import time as _time

    from scripts import study_r7_s3_typed_budget as driver
    from scripts.r7_m3_offline import deny_network  # noqa: F401 - imported for its side-effect path
    require_archived_revision(PRODUCTION_REFERENCE, what="the reading worker branch")

    # Synthetic receipts, exactly the shape the reading stage reads back.
    base = {"incumbent": 2.000, "generic": 1.990, "typed": 1.980}
    for seed in SEEDS:
        for arm, _mode in driver.proto.ARMS:
            lead_dirs = {}
            for lead in LEADS:
                folder = tmp_path / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
                folder.mkdir(parents=True)
                with (folder / "boundary_rmse.csv").open("w", newline="", encoding="utf-8") as h:
                    writer = _csv.DictWriter(h, fieldnames=["lead_hours", "variable", "region",
                                                            "rmse", "unit"])
                    writer.writeheader()
                    writer.writerow({"lead_hours": lead, "variable": "t2m", "region": "full",
                                     "rmse": repr(base[arm] + lead / 1000.0), "unit": "K"})
                    writer.writerow({"lead_hours": lead, "variable": "z500", "region": "full",
                                     "rmse": repr(50000.0 + lead * 100.0),
                                     "unit": "m**2 s**-2"})
                _write_provenance(folder, mse_t2m=(base[arm] + lead / 1000.0) ** 2, leads=LEADS)
                lead_dirs[str(lead)] = {"dir": str(folder), "n_evaluated": 1447,
                                        "n_available_windows": 1447, "elapsed_seconds": 1.0,
                                        "boundary_rmse_csv_sha256": "0" * 64,
                                        "provenance_sha256": "0" * 64}
            receipt = {"seed": seed, "arm": arm,
                       "typed_evidence_mode": dict(driver.proto.ARMS)[arm],
                       "parameters": driver.proto.capacity()["parameters"][arm],
                       "seconds_per_update": 1.1536, "evaluations": lead_dirs}
            (tmp_path / f"seed{seed}_{arm}_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8")

    frozen = {"protocol_sha256": "0" * 64, "updates": 32_132, "batch_size": 4,
              "closed_dose": {"record_id": "s3-typed-spatial-result", "updates": 2400},
              "independent_arm_argument": "synthetic", "limitations": ["synthetic test fixture"],
              "preflight": {"clauses": {"capacity_is_matched_where_it_must_be": True,
                                        "forward_arithmetic_is_matched": True}},
              "train_data_identity": "0" * 64,
              "val_manifest": str(driver.proto.VAL_MANIFEST),
              "deadline_perf_counter": _time.perf_counter() + 3600.0}
    monkeypatch.setattr(driver.proto, "validate_protocol", lambda _out: frozen)
    monkeypatch.setattr("scripts.r7_m3_offline.deny_network", lambda: None)
    monkeypatch.setattr("training.r7_s3_v3_screen.pin_declared_gpu", lambda _uuid: None)
    monkeypatch.setattr(driver.devices, "assigned_uuid", lambda _body: driver.proto.GPU_UUID)

    driver.worker(str(tmp_path), "reading", None, _time.perf_counter() + 1800.0)

    written = tmp_path / "readings.json"
    assert written.is_file(), "the reading branch did not write readings.json"
    loaded = json.loads(written.read_text(encoding="utf-8"))
    for key in ("scientific_claim", "test_read", "protocol_sha256", "exit_conditions", "seeds"):
        assert key in loaded, key
    assert loaded["scientific_claim"] is False
    assert loaded["test_read"] is False
    assert loaded["seeds"] == list(SEEDS)
    assert loaded["exit_conditions"]["d_budget_direction"]["value"] is None


def test_the_archive_stage_accepts_this_rounds_real_zip_and_refuses_a_wrong_comment(tmp_path):
    """The archive stage sits between prepare and seed and had no coverage at all.

    ``run_stage`` promotes ``prepared_protocol.json`` to ``protocol.json`` before archive runs, and
    the stage verifies the zip's comment against the frozen commit and every execution source's
    digest inside the zip. A wrong comment or a stale source there would stop the round before any
    training, so it is worth proving on the real archive rather than assuming.
    """
    import hashlib
    import shutil
    import zipfile

    from scripts import r7_s3_typed_budget_protocol as proto
    from scripts import study_r7_s3_typed_budget as driver

    root = Path(__file__).resolve().parents[1]
    archive = root / "outputs/r7_s3_typed_evidence_budget_code.zip"
    if not archive.is_file():
        pytest.skip("the round's code archive has not been built yet")
    # The archive must be rebuilt from the commit that will actually be frozen: an archive built
    # from an earlier commit carries earlier sources, and the stage refuses it by design. Skip
    # rather than fail while the round is mid-flight, but say so loudly.
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    with zipfile.ZipFile(archive) as _z:
        built_from = _z.comment.decode()
    if built_from != head:
        pytest.skip(f"the archive is stale: built from {built_from[:12]}, HEAD is {head[:12]}; "
                    f"rebuild it at freeze time")
    with zipfile.ZipFile(archive) as zipped:
        commit = zipped.comment.decode()
    assert len(commit) == 40, commit

    # The real archive must satisfy the stage, using this round's own execution inventory.
    body = {"code_archive_input": str(archive),
            "code_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "code_commit": commit,
            "execution_files_sha256": {name: hashlib.sha256(
                (root / name).read_bytes()).hexdigest()
                for name in proto.execution_files()}}
    out = tmp_path / "attempt"
    out.mkdir()
    receipt = driver._archive(out, body)
    assert receipt["code_commit"] == commit
    assert (out / "code.zip").is_file() and (out / "code_commit.txt").is_file()
    assert (out / "archive_receipt.json").is_file()

    # A wrong commit comment must be refused, and must leave no receipt behind.
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(RuntimeError, match="archive commit differs"):
        driver._archive(bad_dir, dict(body, code_commit="0" * 40))
    assert not (bad_dir / "archive_receipt.json").exists()

    # A stale execution source must be refused too.
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale = dict(body, execution_files_sha256=dict(body["execution_files_sha256"]))
    first = sorted(stale["execution_files_sha256"])[0]
    stale["execution_files_sha256"][first] = "0" * 64
    with pytest.raises(RuntimeError, match="archive omits or changes execution source"):
        driver._archive(stale_dir, stale)


def test_the_reading_stage_refuses_to_write_an_incomplete_reading(tmp_path, monkeypatch):
    """``readings.json`` is the stage receipt, so a premature write is permanent.

    ``write`` refuses to overwrite and ``stage_plan`` skips on receipt presence, so writing with
    arms missing would freeze an incomplete table AND mark the stage done: every later attempt
    would skip reading and the missing arms would never be scored. The branch must raise instead.
    """
    import time as _time

    from scripts import study_r7_s3_typed_budget as driver

    frozen = {"protocol_sha256": "0" * 64, "updates": 32_132, "batch_size": 4,
              "closed_dose": {"record_id": "s3-typed-spatial-result", "updates": 2400},
              "independent_arm_argument": "synthetic", "limitations": ["synthetic test fixture"],
              "preflight": {"clauses": {"capacity_is_matched_where_it_must_be": True,
                                        "forward_arithmetic_is_matched": True}},
              "train_data_identity": "0" * 64,
              "val_manifest": str(driver.proto.VAL_MANIFEST),
              "deadline_perf_counter": _time.perf_counter() + 3600.0}
    monkeypatch.setattr(driver.proto, "validate_protocol", lambda _out: frozen)
    monkeypatch.setattr("scripts.r7_m3_offline.deny_network", lambda: None)
    monkeypatch.setattr("training.r7_s3_v3_screen.pin_declared_gpu", lambda _uuid: None)
    monkeypatch.setattr(driver.devices, "assigned_uuid", lambda _body: driver.proto.GPU_UUID)

    # No receipts at all: every arm is missing.
    with pytest.raises(RuntimeError, match="refusing to write readings.json"):
        driver.worker(str(tmp_path), "reading", None, _time.perf_counter() + 1800.0)
    assert not (tmp_path / "readings.json").exists(), (
        "a refused reading must leave no receipt behind, or the stage would be marked done")


def test_clause_a_requires_an_interval_that_excludes_zero_not_just_three_seeds():
    """The frozen (a) needs a block-bootstrap interval excluding zero, not only sign agreement.

    Three seeds are consistency evidence; the interval is the uncertainty statement over held-out
    cases. Before this was wired in, clause (a) could pass on a delta whose resampled interval
    straddled zero. This exercises the real helper on real per-initialisation rows already on disk.
    """
    from scripts import r7_typed_budget_verdict as verdict

    base = (ROOT / "outputs/r7_s3_segB_probe_20261116"
            / "segment_b_depth_scoring/depth")
    k4, k8 = base / "update_0128528_k4", base / "update_0128528_k8"
    if not (k4 / "lead_072h" / "provenance.json").is_file():
        pytest.skip("the Segment B depth provenance is not available")
    dirs = {name: {"41": {str(lead): str((k8 if name == "typed" else k4) / f"lead_{lead:03d}h")
                         for lead in LEADS}}
            for name in ("typed", "generic", "incumbent")}
    out = verdict.pair_interval(verdict.parse_arms({}) if hasattr(verdict, "parse_arms") else {},
                                41, 72, "t2m", "typed", "generic", dirs=dirs)
    assert out["value"] is not None, out
    assert len(out["interval"]) == 2
    assert out["excludes_zero"] is True
    assert out["n_initializations"] > 1000
    # The depth sweep reported K=8 worse by about +0.30 K at 72 h; the interval must bracket that.
    assert 0.15 < out["value"] < 0.5, out["value"]


def test_a_missing_interval_is_untested_not_failed():
    """A clause that cannot compute its interval must be null, never False.

    'Could not establish uncertainty' and 'the effect is zero' are different facts; conflating them
    is the da1ed00 class of defect.
    """
    from scripts import r7_typed_budget_verdict as verdict

    # typed must be LOWER than generic on every lead, so the pair clears the sign and floor tests
    # and the only thing standing between it and a value is the missing interval.
    level = {"typed": 1.9, "generic": 2.0, "incumbent": 2.1}
    arms = {name: {"41": {"rmse": {str(lead): {"t2m": level[name], "z500": 400.0}
                                  for lead in LEADS},
                            "evaluation_dirs": {}}}
            for name in ("typed", "generic", "incumbent")}
    out = verdict._clause_pair(arms, [41], LEADS, "t2m", "typed", "generic",
                               dirs={name: {"41": {}} for name in arms})
    assert out["value"] is None, out
    assert "untested rather than failed" in out["reason"]


def test_a_preflight_entry_dict_is_read_not_compared():
    """(f) must read each entry's own ``passed`` flag, not compare the dict against True.

    ``checks.capacity`` and ``checks.forward_arithmetic`` are dicts carrying ``passed``, so an
    ``entry is True`` test reports a green preflight as failed -- the same shape confusion that
    once published a card contradicting its own reading. Absent is ``None``, never ``False``.
    """
    from scripts import r7_s3_typed_budget_register as register

    assert register._passed({"passed": True}) is True
    assert register._passed({"passed": False}) is False
    assert register._passed(True) is True
    assert register._passed({"passed": None}) is None
    assert register._passed({}) is None, "an entry without a verdict is untested, not passed"
    assert register._passed(None) is None

    clauses = {"off_path_is_the_previous_model": {"passed": True},
               "slot_arrival_is_diagonal": {"passed": True},
               "future_labels_are_unread": {"passed": True},
               "capacity_is_matched_where_it_must_be": {"passed": True},
               "forward_arithmetic_is_matched": {"passed": True},
               "budget_and_closed_dose_are_registered": {"passed": True}}
    readings = {"preflight": {"f_preflight_passed": True, "clauses": clauses},
                "checks": {"capacity": {"passed": True}, "forward_arithmetic": {"passed": True}},
                "exit_conditions": {name: {"value": None} for name in
                                    ("a_typed_minus_generic", "b_generic_minus_incumbent",
                                     "c_claim_scope", "d_budget_direction", "e_z500_72h_guard")},
                "exits": {"values": {}}, "verdict": "untested-no-arm-contrast"}
    summary = register.clause_summary(readings)
    assert summary["f_preflight"]["passed"] is True
    assert summary["f_preflight"]["capacity"] is True
    assert summary["f_preflight"]["forward_arithmetic"] is True


def test_a_receipt_without_elapsed_seconds_is_refused_not_counted_as_zero():
    """GPU hours for the ledger cannot be silently under-counted by a field-less receipt."""
    import json
    from pathlib import Path

    from scripts import r7_s3_typed_budget_register as register

    folder = Path(__file__).resolve().parents[1] / "outputs" / "_register_refusal_probe"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "seed41_incumbent_receipt.json"
    receipt.write_text(json.dumps({"seed": 41, "arm": "incumbent", "evaluations": {}}),
                       encoding="utf-8")
    try:
        try:
            register.segment_a_hours(folder)
        except RuntimeError as error:
            assert "elapsed_seconds" in str(error)
        else:
            raise AssertionError("a receipt without elapsed_seconds must be refused")
    finally:
        receipt.unlink()
        folder.rmdir()


def test_the_partial_reading_refuses_to_grant_clause_a_and_never_turns_absence_into_false():
    """The partial reader must emit (a) as ``None`` -- absence is untested, never a negative.

    The round loses ``readings.json`` entirely when one arm cannot finish, so this tool is the only
    way to read (b). Its counter-proof is the one that matters: a tool built to salvage a measurable
    clause could easily promote the *unmeasurable* ones to ``False`` ("we measured it and it failed")
    or let ``_clause_pair`` raise on an empty seed set (it divides by ``len(measured)``).
    """
    from tools import r7_s3_typed_budget_partial_reading as partial

    # No seed has any arm: the frozen helper would raise ZeroDivisionError here.
    body = {"seeds": [41, 42, 43], "lead_hours": list(LEADS),
            "arms": {"incumbent": {}, "generic": {}, "typed": {}}}
    empty = partial.evaluate_partial(body)
    assert empty["value"] is None
    assert empty["seeds_with_both_arms"] == []
    assert empty["clauses"]["b_generic_minus_incumbent"]["value"] is None
    assert empty["clauses"]["a_typed_minus_generic"]["value"] is None
    assert "untested" in empty["clauses"]["b_generic_minus_incumbent"]["reason"]

    # Incumbent present but generic absent: still untested, still never False.
    real = partial.assemble(ROOT / "outputs/r7_s3_typed_evidence_budget_20261116_attempt01",
                            SEEDS, LEADS)
    judgement = partial.evaluate_partial(real)
    for name in ("a_typed_minus_generic", "b_generic_minus_incumbent"):
        assert judgement["clauses"][name]["value"] is not False, name
    # The tool exists to read (b); it must never report the type-structure pair as decided.
    assert partial.PARTIAL_PAIR == ("generic", "incumbent")
    assert "typed" not in partial.PARTIAL_PAIR


def test_the_partial_reading_tool_is_outside_the_rounds_frozen_identity():
    """It lives under ``tools/``, which never enters a round's ``execution_files_sha256``.

    Adding a file inside ``scripts/``/``training/``/``model/``/``data/`` would make
    ``validate_protocol`` reject the live attempt at the reading stage -- after all the GPU hours
    were spent. This asserts the actual inventory membership rather than trusting the docstring.
    """
    from scripts import r7_s3_typed_budget_protocol as proto

    inventory = set(proto.execution_files())
    assert "tools/r7_s3_typed_budget_partial_reading.py" not in inventory
    assert not any(name.startswith("tools/") for name in inventory)
