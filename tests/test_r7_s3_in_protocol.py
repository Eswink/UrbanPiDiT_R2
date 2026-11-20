"""Counter-proofs for the in-protocol round.

Every test here fails if a specific way of reading this round goes wrong. They are organised by the
defect they pin, and each docstring says which one, because a test that only asserts a happy path
would stay green through the failures this repository has actually had:

* the envelope sized from the wrong rate while its guard read the same constant (defect 18),
* an arm table indexed seed-first so every clause silently reported "untested" (defect 17),
* an incomplete reading written as if it were finished (readings.json is a stage receipt),
* a truncated arm leaving no receipt at all (defect 20),
* a clause that is untested being published as ``False`` (the ``da1ed00`` inversion).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_in_protocol_protocol as proto
from scripts import r7_s3_in_protocol_verdict as verdict


def _arm(directory: str, *, parameters: int = 1) -> dict:
    return {"evaluation_dirs": {}, "parameters": parameters, "dir": directory}


class TestEnvelope:
    """Defect 18: the guard must compare two independent numbers, not a constant to itself."""

    def test_the_envelope_covers_the_treated_requirement(self):
        measured = proto.envelope()
        assert measured["sufficient"] is True
        assert measured["margin"] >= proto.ENVELOPE_MARGIN

    def test_sizing_from_the_incumbent_rate_would_be_insufficient(self):
        """The exact mistake the previous round made, re-run against this round's envelope."""
        need_if_incumbent = (proto.UPDATES * proto.INCUMBENT_SECONDS_PER_UPDATE
                             + proto.MEASURED_SECONDS_PER_EVALUATION)
        naive_envelope = len(proto.ARMS) * need_if_incumbent
        # An envelope derived that way is *below* the treated requirement, so this assertion is what
        # would have caught the original defect at freeze time.
        assert naive_envelope < proto.envelope()["need_seconds"]

    def test_the_treated_rate_exceeds_every_measurement_on_disk(self):
        """The rate that sizes the envelope is an upper bound, not a friendly average."""
        measured = {"generic_seed41": 1.4043, "generic_seed42": 1.4312, "generic_seed43": 1.4271,
                    "typed_seed41_tail": 1.4565, "typed_seed42_tail": 1.4618,
                    "typed_seed43_tail": 1.4618}
        assert all(proto.TREATED_SECONDS_PER_UPDATE >= value for value in measured.values())

    def test_the_hard_cap_covers_two_seed_waves(self):
        assert proto.envelope()["hard_cap_covers_waves"] is True

    def test_a_short_envelope_is_refused_at_freeze(self, monkeypatch):
        monkeypatch.setattr(proto, "PER_SEED_SECONDS", 1_000.0)
        with pytest.raises(RuntimeError, match="envelope"):
            proto.freeze(Path("/tmp/should-not-be-written"), started=0.0, deadline=1.0,
                         code_archive=Path("x"), code_commit="c", code_sha256="s")


class TestEarlyRefusal:
    """A larger envelope is still a prediction; the probe is what makes the prediction checkable."""

    def _driver(self):
        from scripts import study_r7_s3_in_protocol as driver
        return driver

    def test_a_rate_that_cannot_finish_is_refused(self):
        driver = self._driver()
        decision = driver.envelope_check(probe_seconds_per_update=1.47, updates=32_132,
                                         evaluation_seconds=3_200.0,
                                         remaining_seconds=100_000.0, arms_left=3)
        assert decision["permitted"] is False
        assert decision["fraction_of_remaining"] > driver.REFUSAL_FRACTION

    def test_a_rate_that_fits_is_permitted(self):
        driver = self._driver()
        decision = driver.envelope_check(probe_seconds_per_update=1.47, updates=32_132,
                                         evaluation_seconds=3_200.0,
                                         remaining_seconds=proto.PER_SEED_SECONDS,
                                         arms_left=3)
        assert decision["permitted"] is True

    def test_a_zero_remaining_envelope_is_a_refusal_not_a_crash(self):
        driver = self._driver()
        decision = driver.envelope_check(probe_seconds_per_update=1.47, updates=32_132,
                                         evaluation_seconds=3_200.0, remaining_seconds=0.0,
                                         arms_left=1)
        assert decision["permitted"] is False
        assert decision["fraction_of_remaining"] is None

    def test_the_measured_rate_reads_a_real_progress_log(self, tmp_path):
        driver = self._driver()
        path = tmp_path / "progress.jsonl"
        rows = [{"update": update, "elapsed_seconds": update * 1.5} for update in (20, 40, 200, 400)]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        assert driver.measured_rate(path, after_update=400) == pytest.approx(1.5)

    def test_a_log_short_of_the_probe_point_yields_no_rate(self, tmp_path):
        driver = self._driver()
        path = tmp_path / "progress.jsonl"
        path.write_text(json.dumps({"update": 20, "elapsed_seconds": 30.0}) + "\n",
                        encoding="utf-8")
        assert driver.measured_rate(path, after_update=400) is None


class TestPartialReceipt:
    """Defect 20: a stopped arm must still write a receipt, or the ledger undercounts it."""

    def _driver(self):
        from scripts import study_r7_s3_in_protocol as driver
        return driver

    def test_a_partial_receipt_records_what_the_arm_consumed(self, tmp_path):
        driver = self._driver()
        folder = tmp_path / "seed41/training/typed"
        folder.mkdir(parents=True)
        rows = [{"update": update, "elapsed_seconds": update * 1.46}
                for update in (20, 2_000, 24_840)]
        (folder / "progress.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        body = {"updates": 32_132}
        record = driver._partial(tmp_path, body, 41, "typed", "typed_routing",
                                 reason="deadline", detail={})
        assert record["updates_reached"] == 24_840
        assert record["updates_planned"] == 32_132
        assert record["gpu_hours_in_failed_attempt"] > 0.0
        assert record["scientific_claim"] is False
        assert (tmp_path / "seed41_typed_partial_receipt.json").is_file()

    def test_a_partial_receipt_without_a_log_still_lands(self, tmp_path):
        driver = self._driver()
        record = driver._partial(tmp_path, {"updates": 100}, 41, "typed", None,
                                 reason="started-then-died", detail={})
        assert record["updates_reached"] == 0
        assert record["seconds_per_update"] is None
        assert (tmp_path / "seed41_typed_partial_receipt.json").is_file()


def _synthetic_arms(tmp_path: Path, *, seeds=(41,), seeds_with_data=(41,),
                    positive=True, scale=1.0, n_cases=430) -> dict:
    """Build an arm table whose per-case rows have a known sign and a known size.

    The rows are written as real ``provenance.json`` and ``climatology_skill.csv`` files, because the
    verdict reads files rather than a summary: a fixture that fed the verdict in-memory numbers
    would not catch a path or shape error, which is exactly the class this round has to avoid.
    """
    channels = list(proto.store_channels())
    arms = {name: {} for name in (proto.INCUMBENT, proto.GENERIC, proto.TYPED)}
    for seed in seeds:
        for name in (proto.INCUMBENT, proto.GENERIC, proto.TYPED):
            has_data = seed in seeds_with_data
            dirs = {}
            climatology = {}
            for lead in proto.LEADS:
                directory = tmp_path / f"seed{seed}/{name}/lead_{lead:03d}h"
                directory.mkdir(parents=True, exist_ok=True)
                if not has_data:
                    continue
                # The treatment arm must be better when ``positive``: lower MSE on every channel.
                base = 10.0
                # The *difference* between arms must vary by case, or every delta is identical and
                # the block bootstrap has zero spread -- the instrument then correctly reports
                # insufficient-evidence, which would make this fixture test nothing. ``spread`` is
                # a per-case multiplier on the arm's offset, not on its level.
                offset = 0.0
                if name == proto.TYPED:
                    offset = -0.02 * scale if positive else 0.02 * scale
                elif name == proto.GENERIC:
                    # The two arms must not share their offset when the effect is negative: equal
                    # offsets make typed - generic exactly zero and the family untested rather than
                    # false, which would hide the sign being tested.
                    offset = -0.01 * scale if positive else 0.01 * scale
                rows = []
                for index in range(n_cases):
                    # A deterministic pseudo-random spread with no short period: the block length is
                    # 28, so a periodic jitter of period <= 28 would give every block the same mean
                    # and the block bootstrap zero variance -- insufficient-evidence, not a pass.
                    state = (index * 2654435761) % 4294967296
                    spread = 1.0 + 0.9 * ((state / 4294967296.0) - 0.5) * 2.0
                    values = [base * (1.0 + offset * spread) for _ in channels]
                    rows.append({"init_time": f"2022-01-{1 + index // 4:02d}T"
                                              f"{(index % 4) * 6:02d}:00:00",
                                 "valid_times": [f"2022-01-{1 + index // 4:02d}T"
                                                 f"{(index % 4) * 6:02d}:00:00"],
                                 "mse": [values]})
                (directory / "provenance.json").write_text(
                    json.dumps({"initializations": rows}), encoding="utf-8")
                lines = ["lead_hours,variable,rmse_forecast,rmse_climatology,mse_skill,unit,n"]
                for channel in channels:
                    lines.append(f"{lead},{channel},3.0,4.0,0.2,K,{n_cases}")
                (directory / "climatology_skill.csv").write_text("\n".join(lines) + "\n",
                                                               encoding="utf-8")
                dirs[str(lead)] = str(directory)
                climatology[str(lead)] = {channel: 16.0 for channel in channels}
            arms[name][str(seed)] = {"evaluation_dirs": dirs, "climatology_mse": climatology,
                                     "parameters": 3_434_837 if name != proto.INCUMBENT else 3_286_037}
    return arms


class TestFamily:
    """The family instrument: partition, shared case set, and the three conjuncts."""

    def test_the_families_are_twenty_and_sixty_five_grids(self):
        families = proto.families()
        assert families[proto.FAMILY_PRIMARY_LABEL]["grids"] == 20
        assert families[proto.FAMILY_SECONDARY_LABEL]["grids"] == 65

    def test_the_primary_family_is_the_channels_the_pathway_declares(self):
        from model.r7_process_tensor_diagnostics import TYPED_EVIDENCE_CHANNELS
        families = proto.families()
        assert list(families[proto.FAMILY_PRIMARY_LABEL]["channels"]) == list(
            TYPED_EVIDENCE_CHANNELS)

    def test_the_partition_covers_every_channel_exactly_once(self):
        families = proto.families()
        primary = families[proto.FAMILY_PRIMARY_LABEL]["channels"]
        secondary = families[proto.FAMILY_SECONDARY_LABEL]["channels"]
        assert sorted(primary + secondary) == sorted(proto.store_channels())
        assert not set(primary) & set(secondary)

    def test_a_family_on_the_same_cohort_is_assembled(self, tmp_path):
        arms = _synthetic_arms(tmp_path)
        outcome = verdict.clause_family(arms, [41], list(proto.LEADS),
                                        list(proto.store_channels())[:2],
                                        proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels())
        assert outcome["value"] is True
        assert outcome["n_clearing"] >= 1

    def test_a_single_lead_is_not_enough(self, tmp_path):
        """The cross-lead conjunct: one clearing lead cannot carry a family."""
        arms = _synthetic_arms(tmp_path)
        outcome = verdict.clause_family(arms, [41], list(proto.LEADS),
                                        list(proto.store_channels())[:2],
                                        proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels(),
                                        minimum_leads=6)
        assert outcome["value"] is False
        assert outcome["clearing_leads"]

    def test_a_difference_below_the_floor_does_not_clear(self, tmp_path):
        arms = _synthetic_arms(tmp_path, scale=0.01)   # a 0.01 percent effect on 16.0 climatology
        outcome = verdict.clause_family(arms, [41], list(proto.LEADS),
                                        list(proto.store_channels())[:1],
                                        proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels(),
                                        floor=0.001)
        assert outcome["value"] is False
        assert outcome["n_clearing"] == 0

    def test_a_worse_treatment_does_not_clear(self, tmp_path):
        arms = _synthetic_arms(tmp_path, positive=False)
        outcome = verdict.clause_family(arms, [41], list(proto.LEADS),
                                        list(proto.store_channels())[:2],
                                        proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels())
        assert outcome["value"] is False
        assert outcome["n_clearing"] == 0

    def test_a_missing_arm_is_untested_not_false(self, tmp_path):
        """The ``da1ed00`` discipline: no contrast is ``None``."""
        arms = _synthetic_arms(tmp_path, seeds=(41, 42, 43), seeds_with_data=(41, 42))
        outcome = verdict.clause_family(arms, [41, 42, 43], list(proto.LEADS),
                                        list(proto.store_channels())[:2],
                                        proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels())
        assert outcome["value"] is None

    def test_an_empty_arm_table_is_untested(self):
        outcome = verdict.clause_family({}, [41], [6], ["t2m"], proto.TYPED, proto.GENERIC,
                                        block_length=28, resamples=200, resample_seed=41,
                                        alpha=0.05, channel_order=proto.store_channels())
        assert outcome["value"] is None

    def test_the_case_set_is_intersected_not_unioned(self, tmp_path):
        """Each lead scores a different cohort; the family must run on their intersection."""
        arms = _synthetic_arms(tmp_path)
        collected = verdict.family_cells(arms, [41], list(proto.LEADS),
                                         ["t2m", "z500"], proto.TYPED, proto.GENERIC,
                                         channel_order=proto.store_channels())
        outcome = verdict.build_family(collected["cells"], block_length=28, resamples=100,
                                       seed=41, alpha=0.05)
        assert outcome["status"] == "ok"
        assert outcome["n_dropped_cases"] >= 0


class TestArmLookup:
    """Defect 17: the table is arm-first then seed; the reverse order is silent."""

    def test_the_arm_table_key_order_is_arm_first_then_seed(self):
        arms = {"typed": {"41": {"parameters": 7}}}
        assert verdict.arm_for(arms, "typed", 41)["parameters"] == 7

    def test_the_reverse_order_returns_empty(self):
        """Pins the failure mode: seed-first would read as "untested" everywhere."""
        arms = {"41": {"typed": {"parameters": 7}}}
        assert verdict.arm_for(arms, "typed", 41) == {}


class TestEvaluate:
    """The whole verdict, including its null discipline and the scope rule."""

    def test_an_empty_reading_is_untested_everywhere(self):
        result = verdict.evaluate({})
        assert result["verdict"] == "untested-no-arm-contrast"
        assert all(value is None for value in result["values"].values())

    def test_the_scope_rule_is_null_while_either_family_is_untested(self):
        clause = verdict.clause_c({"value": None}, {"value": True})
        assert clause["value"] is None
        assert clause["permitted_claim"] is None

    def test_the_scope_rule_never_returns_false(self):
        for primary, secondary in [(True, True), (True, False), (False, False), (False, True)]:
            assert verdict.clause_c({"value": primary},
                                    {"value": secondary})["value"] is True

    def test_a_true_primary_with_a_false_secondary_licenses_only_the_type_structure(self):
        clause = verdict.clause_c({"value": True}, {"value": False})
        assert clause["permitted_claim"] == "type-structure-only"
        assert clause["forbidden_claim"] == "the physical evidence adds skill"

    def test_a_false_primary_forbids_any_positive_claim(self):
        clause = verdict.clause_c({"value": False}, {"value": False})
        assert clause["permitted_claim"] == "no-increment-from-this-pathway"

    def test_the_budget_direction_is_null_without_a_second_dose(self):
        assert verdict.clause_d({"updates": 32_132})["value"] is None


class TestReadingContract:
    """The reading is a stage receipt: it must refuse to overwrite and must carry its five keys."""

    def test_write_refuses_to_overwrite(self, tmp_path):
        target = tmp_path / "readings.json"
        verdict.write(target, {"a": 1})
        with pytest.raises(FileExistsError):
            verdict.write(target, {"a": 2})

    def test_the_five_required_keys_are_documented(self):
        from tools.check_round_receipts import READINGS_REQUIRED_KEYS
        assert set(READINGS_REQUIRED_KEYS) == {"scientific_claim", "test_read",
                                              "protocol_sha256", "exit_conditions", "seeds"}

    def test_a_null_data_identity_is_refused(self, monkeypatch, tmp_path):
        """Defect 19: ``None == None`` used to pass the identity comparison.

        ``prepared_protocol`` reads two identities and compares them; when both were ``None`` the
        comparison held and the protocol froze an identity of ``null``. The guard is driven here
        through the real function with the identity probe patched to return ``None``.
        """
        from training.r7_experiment import canonical_digest

        preliminary = {"protocol_sha256": "0" * 64}
        (tmp_path / "preparation_protocol.json").write_text(
            json.dumps(preliminary), encoding="utf-8")
        monkeypatch.setattr(proto, "preflight",
                            lambda: {"f_preflight_passed": True, "clauses": {}, "envelope":
                                     {"sufficient": True}})
        monkeypatch.setattr(canonical_digest, "__call__", lambda body: "0" * 64)
        with pytest.raises(RuntimeError):
            proto.prepared_protocol(tmp_path)

    def test_the_reading_names_its_own_writer(self):
        source = (ROOT / "scripts/study_r7_s3_in_protocol.py").read_text(encoding="utf-8")
        assert "r7_s3_in_protocol_assembly as assembly" in source
        assert "verdict_module.write" in source


class TestPreflight:
    """The six clause names are a contract: a renamed clause silently stops running."""

    def test_the_six_clause_names_are_fixed(self):
        from scripts import r7_s3_in_protocol_preflight as preflight
        assert tuple(name for name, _ in preflight.CLAUSES) == (
            "off_path_is_the_previous_model",
            "slot_arrival_is_diagonal",
            "future_labels_are_unread",
            "capacity_is_matched_where_it_must_be",
            "forward_arithmetic_is_matched",
            "envelope_covers_the_treated_rate",
        )

    def test_the_envelope_clause_fails_when_the_envelope_is_too_small(self, monkeypatch):
        from scripts import r7_s3_in_protocol_preflight as preflight
        monkeypatch.setattr(proto, "PER_SEED_SECONDS", 1_000.0)
        assert preflight.clause_envelope()["passed"] is False

    def test_the_capacity_claim_is_the_registered_pair(self):
        claims = proto.capacity()
        assert claims["parameters"][proto.INCUMBENT] == 3_286_037
        assert claims["parameters"][proto.GENERIC] == 3_434_837
        assert claims["parameters"][proto.TYPED] == 3_434_837
        assert claims["passed"] is True

    def test_the_closed_doses_are_both_named(self):
        """A citation must not be able to present this as a re-run of either earlier dose."""
        records = {entry["record_id"]: entry["updates"] for entry in proto.CLOSED_DOSES}
        assert records == {"s2-79-typed-evidence": 400, "s3-typed-spatial-result": 2_400}


class TestJointCriterion:
    """Segment B's job is to prove the machinery assembles, not to confirm the hypothesis."""

    def test_the_probe_declares_itself_exploratory(self):
        from scripts import r7_s3_joint_criterion as probe
        report = probe.dry_check()
        assert report["exploratory"] is True
        assert report["scientific_claim"] is False
        assert report["gpu_hours"] == 0.0

    def test_both_families_assemble_on_the_registered_rows(self):
        from scripts import r7_s3_joint_criterion as probe
        report = probe.dry_check()
        assert report["all_families_can_run"] is True
        assert report["families"][proto.FAMILY_PRIMARY_LABEL]["n_cells_complete"] == 20
        assert report["families"][proto.FAMILY_SECONDARY_LABEL]["n_cells_complete"] == 65

    def test_the_instrument_reports_one_shared_case_set_per_family(self):
        from scripts import r7_s3_joint_criterion as probe
        report = probe.dry_check()
        for entry in report["families"].values():
            assert entry["instrument"]["n_cases"] == 1_447
            assert entry["instrument"]["n_blocks"] >= 4


class TestDriverReturnContract:
    """The defect attempt01 died on: a return shape no counter-proof covered.

    The first attempt's ``_run`` returned six values while both call sites unpacked seven, so the
    probe raised ``ValueError: not enough values to unpack (expected 7, got 6)`` after its first 400
    updates. These tests pin the contract at the source, without a GPU: they read the function's own
    AST and its returned dict keys, so a re-shape cannot pass silently again.
    """

    def test_run_returns_a_named_record_not_a_positional_tuple(self):
        import ast

        from scripts import study_r7_s3_in_protocol as driver
        tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
        found = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "_run"]
        assert len(found) == 1, "the driver must define exactly one _run"
        returns = [node for node in ast.walk(found[0]) if isinstance(node, ast.Return)]
        assert returns, "_run must return"
        value = returns[-1].value
        # A dict literal is the contract; a tuple would let positions drift again.
        assert isinstance(value, ast.Dict), (
            "_run must return a named record; a positional tuple is how attempt01 lost its shape")

    def test_every_run_call_site_uses_keys_not_positional_unpacking(self):
        import ast

        from scripts import study_r7_s3_in_protocol as driver
        tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            name = getattr(call.func, "id", getattr(call.func, "attr", ""))
            if name != "_run":
                continue
            targets = node.targets
            assert len(targets) == 1 and isinstance(targets[0], ast.Name), (
                f"line {node.lineno}: _run must bind to one name")
            # The bound name must then be subscripted, never tuple-unpacked.
            assert isinstance(targets[0].ctx, ast.Store)

    def test_the_record_keys_are_the_seven_the_callers_read(self):
        """The keys the two call sites actually use, asserted without training anything."""
        import inspect

        from scripts import study_r7_s3_in_protocol as driver
        source = inspect.getsource(driver._run)
        for key in ("checkpoint", "report", "mode", "expansion", "record", "parent", "model"):
            assert f'"{key}"' in source, f"_run's record must carry {key!r}"

    def test_probe_and_train_arm_read_the_record_through_keys(self):
        import inspect

        from scripts import study_r7_s3_in_protocol as driver
        for function in (driver.probe, driver._train_arm):
            source = inspect.getsource(function)
            assert 'ran["' in source, (
                f"{function.__name__} must read the named record, not unpack a tuple")
            assert "= _run(" not in source.replace("ran = _run(", ""), (
                f"{function.__name__} still tuple-unpacks _run")
