"""Counter-proofs for the typed-evidence budget round.

The pathway's own wiring is already pinned by ``tests/test_r7_79_typed_evidence.py`` (off-path
bit-identity, per-type slot routing, capacity matching, label poisoning). What is *new* here is the
round's own machinery, and each test below is written to fail on a specific defect this repository
has already paid for once:

1. a clause that defaults to a value when its input is missing (the ``da1ed00`` incident);
2. a clause that passes on a single favourable sign;
3. the reading path that dies after the GPU has been spent (the balance round's 7.2 s ``NameError``);
4. a receipt contract that the reader and the writer disagree about;
5. a dose curve that reports "saturated" from a flat or noisy tail;
6. a depth sweep that mistakes a cheap-but-worse depth for a win;
7. an arm table whose key order is silently wrong, which turns every clause into "untested".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_typed_budget_verdict as verdict  # noqa: E402

LEADS = (6, 12, 24, 48, 72)
SEEDS = (41, 42, 43)


def _arm(t2m, z500_72):
    """A minimal arm table: t2m per lead plus the z500 value the (e) guard reads."""
    return {"rmse": {str(lead): {"t2m": t2m[lead], "z500": z500_72 if lead == 72 else 400.0}
                     for lead in LEADS}}


def _readings(incumbent, generic, typed):
    return {"seeds": list(SEEDS), "lead_hours": list(LEADS),
            "arms": {"incumbent": {str(seed): incumbent for seed in SEEDS},
                     "generic": {str(seed): generic for seed in SEEDS},
                     "typed": {str(seed): typed for seed in SEEDS}}}


BASE = _arm({6: 3.0, 12: 3.5, 24: 4.0, 48: 4.5, 72: 5.0}, 420.0)
GENERIC = _arm({6: 2.9, 12: 3.4, 24: 3.9, 48: 4.4, 72: 4.9}, 425.0)
TYPED = _arm({6: 2.8, 12: 3.3, 24: 3.8, 48: 4.3, 72: 4.8}, 425.0)


def test_missing_arms_are_untested_never_false_and_never_true():
    """A clause with no contrast must be ``None``.

    The ``da1ed00`` defect was a reader that defaulted every clause to ``false`` when its input was
    empty, which published a card contradicting ``readings.json``. An empty reading is *untested*,
    which is a third value.
    """
    empty = verdict.evaluate({})
    assert empty["values"], "the evaluator must still name every clause"
    assert all(value is None for value in empty["values"].values()), empty["values"]
    assert empty["verdict"] == "untested-no-arm-contrast"


def test_a_partial_seed_set_is_untested_not_a_result():
    """Two of three seeds is not a weaker pass; it is not a reading."""
    partial = {"seeds": list(SEEDS), "lead_hours": list(LEADS),
               "arms": {"incumbent": {"41": BASE}, "generic": {"41": GENERIC},
                        "typed": {"41": TYPED}}}
    outcome = verdict.evaluate(partial)
    assert outcome["values"]["a_typed_minus_generic"] is None
    assert outcome["values"]["b_generic_minus_incumbent"] is None
    assert "untested" in outcome["clauses"]["a_typed_minus_generic"]["reason"]


def test_one_favourable_lead_cannot_carry_the_clause():
    """One clearing lead out of five is not a result, even though it is a real improvement.

    The flat arm genuinely wins lead 6 by 0.101 K and ties the rest, so the lead is *counted*; the
    clause still fails because it needs two.
    """
    flat = _arm({6: 2.799, 12: 3.400, 24: 3.900, 48: 4.400, 72: 4.900}, 425.0)
    outcome = verdict.evaluate(_readings(BASE, GENERIC, flat))
    clause = outcome["clauses"]["a_typed_minus_generic"]
    assert clause["n_clearing"] == 1, clause["per_lead"]
    assert clause["clearing_leads"] == [6]
    assert clause["n_concordant"] == 1
    assert outcome["values"]["a_typed_minus_generic"] is False


def test_a_single_seed_sign_flip_breaks_concordance_at_that_lead():
    """The same lead must agree on all three seeds; one seed is reported, not averaged away."""
    flipped = _arm({6: 2.95, 12: 3.3, 24: 3.8, 48: 4.3, 72: 4.8}, 425.0)
    readings = _readings(BASE, GENERIC, TYPED)
    readings["arms"]["typed"]["43"] = flipped
    clause = verdict.evaluate(readings)["clauses"]["a_typed_minus_generic"]
    assert clause["per_lead"]["6"]["concordant"] is False
    assert 6 not in clause["clearing_leads"]
    assert clause["per_lead"]["12"]["concordant"] is True


def test_the_gain_floor_is_enforced():
    """A resolvable but negligible gain must not count as a lead."""
    tiny = _arm({6: 2.8995, 12: 3.3995, 24: 3.8995, 48: 4.3995, 72: 4.8995}, 425.0)
    clause = verdict.evaluate(_readings(BASE, GENERIC, tiny))["clauses"]["a_typed_minus_generic"]
    assert clause["n_concordant"] == 5
    assert clause["n_clearing"] == 0
    assert clause["value"] is False


def test_a_clean_sweep_passes_both_pairs_and_licenses_the_stronger_claim():
    outcome = verdict.evaluate(_readings(BASE, GENERIC, TYPED))
    assert outcome["values"]["a_typed_minus_generic"] is True
    assert outcome["values"]["b_generic_minus_incumbent"] is True
    assert outcome["verdict"] == "positive-typed-and-pathway"
    assert outcome["clauses"]["c_claim_scope"]["permitted_claim"] == "type-structure-and-pathway"


def test_a_lone_typed_win_licenses_only_the_type_structure():
    """(c) must forbid the physics claim when the pathway pair does not clear."""
    outcome = verdict.evaluate(_readings(BASE, BASE, TYPED))
    assert outcome["values"]["a_typed_minus_generic"] is True
    assert outcome["values"]["b_generic_minus_incumbent"] is False
    scope = outcome["clauses"]["c_claim_scope"]
    assert scope["permitted_claim"] == "type-structure-only"
    assert scope["forbidden_claim"] == "the physical evidence adds skill"


def test_a_failed_typed_pair_forbids_any_positive_claim():
    outcome = verdict.evaluate(_readings(BASE, GENERIC, GENERIC))
    assert outcome["values"]["a_typed_minus_generic"] is False
    scope = outcome["clauses"]["c_claim_scope"]
    assert scope["permitted_claim"] == "no-increment-from-this-pathway"
    assert outcome["verdict"] == "negative-no-type-structure-increment"


def test_the_z500_guard_fails_when_the_dynamics_move_the_wrong_way():
    """A t2m win with a 19 percent z500 blow-up must not read as a pass."""
    blown = _arm({6: 2.8, 12: 3.3, 24: 3.8, 48: 4.3, 72: 4.8}, 500.0)
    outcome = verdict.evaluate(_readings(BASE, GENERIC, blown))
    guard = outcome["clauses"]["e_z500_72h_guard"]
    assert guard["value"] is False
    assert guard["max_relative_change"] == pytest.approx(0.190476, abs=1e-6)


def test_the_z500_guard_compares_against_each_seeds_own_incumbent():
    """The guard is an arm-to-arm test; the 427.35 reference is informational only."""
    guard = verdict.evaluate(_readings(BASE, GENERIC, TYPED))["clauses"]["e_z500_72h_guard"]
    assert guard["value"] is True
    row = guard["per_arm"]["typed/seed41"]
    assert row["incumbent"] == pytest.approx(420.0)
    assert row["relative_change"] == pytest.approx((425.0 - 420.0) / 420.0)
    assert "against_reference" in row


def test_budget_direction_is_null_until_a_second_dose_exists():
    """(d) may never pass on one dose."""
    clause = verdict.evaluate(_readings(BASE, GENERIC, TYPED))["clauses"]["d_budget_direction"]
    assert clause["value"] is None
    assert "no 16-epoch reading" in clause["reason"]
    supplied = dict(_readings(BASE, GENERIC, TYPED),
                    budget_comparison={"direction_preserved": True, "reason": "supplied"})
    assert verdict.evaluate(supplied)["clauses"]["d_budget_direction"]["value"] is True


def test_the_arm_table_key_order_is_arm_first_then_seed():
    """The layout bug that turns every clause into a silent "untested"."""
    arms = {"typed": {"41": TYPED}, "generic": {"41": GENERIC}}
    assert verdict.arm_for(arms, "typed", 41) is TYPED
    assert verdict.arm_for(arms, "typed", 41)["rmse"]["6"]["t2m"] == pytest.approx(2.8)
    assert verdict.pair_delta(arms, 41, 6, "t2m", "typed", "generic") == pytest.approx(-0.1)


def test_write_refuses_to_overwrite(tmp_path):
    """A second reading is a new attempt directory, not a silent replacement."""
    target = tmp_path / "readings.json"
    verdict.write(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(FileExistsError):
        verdict.write(target, {"a": 2})


def test_the_protocol_exposes_the_names_the_wrapper_requires():
    """``tools/r7_round_stages.py:60`` refuses a driver missing any of these."""
    from scripts import study_r7_s3_typed_budget as driver
    for name in ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS"):
        assert hasattr(driver, name), name


def test_the_protocol_refuses_a_budget_other_than_the_frozen_one():
    """The 4-epoch dose is what makes the negative result citable; it must not drift."""
    from scripts import r7_s3_typed_budget_protocol as proto
    assert proto.UPDATES == 32_132
    assert proto.BATCH_SIZE == 4
    assert proto.SEEDS == (41, 42, 43)
    assert proto.CLOSED_ROUND_RECORD == "s3-typed-spatial-result"
    assert proto.CLOSED_ROUND_UPDATES == 2_400


def test_the_capacity_claims_hold_on_the_frozen_spec():
    """B and C must be parameter-identical and A strictly smaller.

    This is the clause the whole attribution rests on: if generic and typed differed in capacity,
    a B->C win could be a capacity effect and nothing else.
    """
    from scripts import r7_s3_typed_budget_protocol as proto
    measured = proto.capacity()
    counts = measured["parameters"]
    assert counts["generic"] == counts["typed"], counts
    assert counts["incumbent"] < counts["generic"], counts
    assert measured["passed"] is True
    assert measured["pathway_parameters"] == counts["generic"] - counts["incumbent"]


def test_the_frozen_spec_is_the_registered_production_trunk():
    """A hand-built spec would let the module describe a model the weights are not."""
    from scripts import r7_s3_typed_budget_protocol as proto
    spec = proto.arm_spec(None)
    assert spec["dim"] == 192 and spec["depth"] == 4
    assert spec["in_channels"] == 17 and spec["out_channels"] == 17
    assert "typed_evidence_mode" not in spec, "the incumbent must carry no pathway key"
    assert proto.arm_spec("typed_routing")["typed_evidence_mode"] == "typed_routing"
    with pytest.raises(ValueError):
        proto.arm_spec("typed_spatial")


def test_the_preflight_reports_every_clause_by_name():
    """(f) is a named set, so a future clause cannot be added and silently ignored."""
    from scripts import r7_s3_typed_budget_protocol as proto
    measured = proto.preflight()
    expected = {"off_path_is_the_previous_model", "slot_arrival_is_diagonal",
                "future_labels_are_unread", "capacity_is_matched_where_it_must_be",
                "forward_arithmetic_is_matched", "budget_and_closed_dose_are_registered"}
    assert set(measured["clauses"]) == expected
    assert measured["scientific_claim"] is False
    assert measured["test_read"] is False


def test_the_reading_path_writes_every_key_the_receipt_checker_demands(tmp_path):
    """The balance round died 7.2 s into this branch on a ``NameError``.

    ``tools/check_round_receipts.py:45-47`` requires five keys and an ``exit_conditions`` object.
    Building them here means the writing path is exercised without a GPU round.
    """
    from scripts import r7_typed_budget_verdict as writing
    body = {"protocol_sha256": "0" * 64, "updates": 32_132, "batch_size": 4,
            "seeds": list(SEEDS), "lead_hours": list(LEADS), "arms": {}, "complete": False,
            "scientific_claim": False, "test_read": False}
    outcome = writing.evaluate(body)
    body["exits"] = outcome
    body["exit_conditions"] = outcome["clauses"]
    target = tmp_path / "readings.json"
    writing.write(target, body)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    for key in ("scientific_claim", "test_read", "protocol_sha256", "exit_conditions", "seeds"):
        assert key in loaded, key
    assert isinstance(loaded["exit_conditions"], dict), "the receipt checker wants an object"
    assert loaded["scientific_claim"] is False
    assert loaded["test_read"] is False


def test_the_probe_refuses_every_training_stage_name():
    """The "zero training" property must be checkable, not promised."""
    from scripts import r7_s3_dose_probe as probe
    assert "train" in probe.REFUSED_WORKERS
    assert "seed" in probe.REFUSED_WORKERS
    for name in ("train", "seed", "prepare", "archive"):
        with pytest.raises(SystemExit):
            probe.main(["--out", "/tmp/never-written", "--worker", name])


def test_a_flat_dose_tail_is_not_saturated():
    """A flat tail must not be reported as "no more training will help"."""
    from scripts import r7_s3_dose_probe as probe
    updates = (5_000, 15_000, 25_000, 35_000, 45_000, 55_000, 65_000)
    table = {str(update): {"6": {"t2m": 2.0}, "72": {"t2m": 3.5}}
             for update in updates}
    curve = probe.dose_curve(table, updates)
    assert curve["value"] is False, "a constant tail is not strictly falling"
    falling = {str(update): {"6": {"t2m": 2.0}, "72": {"t2m": 4.0 - index * 0.1}}
               for index, update in enumerate(updates)}
    assert probe.dose_curve(falling, updates)["value"] is True


def test_a_dose_curve_with_too_few_points_is_untested():
    from scripts import r7_s3_dose_probe as probe
    updates = (5_000, 15_000)
    table = {str(update): {"6": {"t2m": 2.0}, "72": {"t2m": 3.5 - 0.1 * index}}
             for index, update in enumerate(updates)}
    assert probe.dose_curve(table, updates)["value"] is None


def test_a_cheaper_depth_is_reported_separately_and_a_worse_one_is_not_a_candidate():
    from scripts import r7_s3_dose_probe as probe
    baseline = {str(lead): {"t2m": 3.0} for lead in probe.DEPTH_LEADS}
    better_cheap = {str(lead): {"t2m": 2.9} for lead in probe.DEPTH_LEADS}
    worse_cheap = {str(lead): {"t2m": 3.1} for lead in probe.DEPTH_LEADS}
    sweep = probe.depth_sweep({"4": baseline, "2": better_cheap}, probe.DEPTHS)
    assert sweep["candidates"] == ["2"]
    assert sweep["cheaper_than_trained"] == ["2"]
    sweep = probe.depth_sweep({"4": baseline, "2": worse_cheap}, probe.DEPTHS)
    assert sweep["candidates"] == []
    assert sweep["cheaper_than_trained"] == []


def test_sixteen_epochs_is_only_warranted_when_the_tail_is_still_falling():
    from scripts import r7_s3_dose_probe as probe
    assert probe.decision({"dose_curve": {"value": True}})["sixteen_epoch_is_warranted"] is True
    assert probe.decision({"dose_curve": {"value": False}})["sixteen_epoch_is_warranted"] is False
    assert probe.decision({})["sixteen_epoch_is_warranted"] is False
    assert probe.decision({})["dose_curve_status"] == "untested"


def test_the_probe_identity_clause_rejects_a_foreign_checkpoint(tmp_path):
    """One trajectory or no curve: two code identities must not be spliced."""
    from scripts import r7_s3_dose_probe as probe
    identity = probe.checkpoint_identity(probe.checkpoints(probe.P1600_RUN, (5_000, 128_528)))
    assert identity["passed"] is True
    assert len(identity["distinct_identities"]) == 1
    with pytest.raises(FileNotFoundError):
        probe.checkpoints(tmp_path, (5_000,))


def test_the_per_seed_envelope_exceeds_what_one_seed_actually_does():
    """A seed worker runs **three** arms and scores five leads each before it exits.

    The frozen 16 h envelope was shorter than one arm's training alone, so every seed would have
    died on ``_check_deadline`` inside the update loop after the GPU time was already spent. The
    envelope must exceed the module's own per-arm estimate times the arm count.
    """
    from scripts import r7_s3_typed_budget_protocol as proto
    per_arm = (proto.UPDATES * proto.MEASURED_SECONDS_PER_UPDATE
               + proto.MEASURED_SECONDS_PER_EVALUATION)
    assert proto.PER_SEED_SECONDS > len(proto.ARMS) * per_arm, (
        f"PER_SEED_SECONDS={proto.PER_SEED_SECONDS} does not cover "
        f"{len(proto.ARMS)} arms x {per_arm:.0f} s")
    # The round's hard cap is the wall clock over the frozen cards, and three seeds on two cards
    # need two waves, so it must cover both waves rather than one seed.
    from training import r7_device_selection as devices
    cards = devices.protocol_fields(proto.DEVICE_MODE, single_uuid=proto.GPU_UUID)["gpu_uuids"]
    assert len(cards) == 2, cards
    waves = -(-len(proto.SEEDS) // len(cards))
    assert proto.HARD_CAP_SECONDS >= waves * proto.PER_SEED_SECONDS


def test_every_boundary_reader_is_backed_by_a_writer():
    """``evaluate_local`` writes ``boundary_rmse.csv`` only when ``boundary_margins`` is passed.

    Omitting it does not raise at evaluation time -- it omits the file, and the readers fail later,
    after the GPU has been paid for. The recorded probe failure was exactly this FileNotFoundError.
    """
    import inspect

    from scripts import r7_s3_dose_probe as probe
    from scripts import r7_s3_typed_budget_protocol as proto
    from scripts import study_r7_s3_typed_budget as driver
    from training.r7_evaluate import evaluate_local

    assert "boundary_margins" in inspect.signature(evaluate_local).parameters
    for module in (probe, driver):
        source = inspect.getsource(module)
        assert "boundary_margins" in source, module.__name__
        assert "boundary_rmse.csv" in source, module.__name__
    assert proto.REGION == "full"
    assert proto.BOUNDARY_MARGINS == (1,)


def test_the_probe_never_writes_into_a_registered_run(tmp_path):
    """Scoring must land under the operator's ``--out`` root, never in a prior round's directory.

    Two reasons, both real: those run directories are archived evidence and read-only, and
    ``evaluate_local`` creates its output with ``exist_ok=False``, so a retry into the same path
    fails with ``FileExistsError`` -- which is exactly what the first probe attempt left behind
    (``outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/seed41/probe/``).
    """
    import inspect

    from scripts import r7_s3_dose_probe as probe

    parameters = inspect.signature(probe.score).parameters
    assert "root" in parameters, "score must take the operator's output root"
    assert parameters["root"].default is inspect.Parameter.empty, "root must be required"
    source = inspect.getsource(probe.score)
    assert "checkpoint.parent.parent" not in source, (
        "scoring derives its output path from the checkpoint's own run, which mutates an "
        "archived run and collides on re-run")
    # The two measurements of one checkpoint must not share a directory: B-2 sweeps four depths.
    assert "_k" in source and "reasoning_steps is None" in source
    # And the driver must route both measurements under that root rather than the checkpoint's.
    run_source = inspect.getsource(probe.run)
    assert run_source.count("scoring_root") >= 3, run_source.count("scoring_root")


def test_the_probe_builds_its_preparation_once_not_per_checkpoint():
    """The climatology fit is 722 s and reads 117 GB of store; it must not run per checkpoint.

    ``EvaluationPreparation`` validates every held-out manifest record and lazily fits the train-only
    climatology. Neither depends on the checkpoint, but the probe scored fourteen dose points plus
    four depth arms, so a per-call build refit the climatology eighteen times. Measured on this
    store: 29.7 s to build, 722.0 s and 117 GB read for the fit.
    """
    import inspect

    from scripts import r7_s3_dose_probe as probe

    # ``score`` must take the preparation rather than build one.
    assert "prepared" in inspect.signature(probe.score).parameters
    score_source = inspect.getsource(probe.score)
    assert "EvaluationPreparation(" not in score_source, (
        "score builds its own preparation; the climatology would be refit per checkpoint")
    # And ``run`` must build exactly one and pass it to every measurement.
    run_source = inspect.getsource(probe.run)
    assert run_source.count("build_preparation") == 1, run_source.count("build_preparation")
    assert run_source.count("prepared=prepared") == 2, run_source.count("prepared=prepared")
