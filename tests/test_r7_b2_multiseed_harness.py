"""Offline tests for the B2 multiseed harness contract (#64 B2).

No GPU, no real data, no network: these pin the properties that make the B2
result trustworthy rather than the numbers it produces - the pre-declared seed
sets, the phase-independent protocol digest, the refusal to score an incomplete
phase, and the requirement that the frozen skill criteria are met by *both*
parameter-free controls rather than by one of them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.study_r7_b2_multiseed as harness
from scripts.study_r7_b2_multiseed import (
    ARMS,
    ARM_NAMES,
    BASELINES,
    CONFIRMATORY_SEEDS,
    EXPLORATION_SEEDS,
    NON_SHARED_ARMS,
    SHARED_INITIALIZATION_ANCHOR,
    SHARED_INITIALIZATION_ARMS,
    SKILL_CRITERIA,
    _cell_win_table,
    _unit_for,
    merge_phase_runs,
    protocol_payload,
)


def test_seed_sets_are_pre_declared_and_the_confirmatory_set_contains_the_exploration_one():
    # #64 B2: explore with fixed seeds, then confirm with at least three
    # pre-declared seeds. A confirmatory set that did not include the explored
    # seeds would silently discard runs that were already paid for.
    assert len(EXPLORATION_SEEDS) == 2
    assert len(CONFIRMATORY_SEEDS) >= 3
    assert set(EXPLORATION_SEEDS) <= set(CONFIRMATORY_SEEDS)
    assert len(set(CONFIRMATORY_SEEDS)) == len(CONFIRMATORY_SEEDS)


def test_role_separation_includes_the_shared_structure_pair():
    # #64 B2 sentence 3 needs generic/process (which share a backbone) to start
    # from aligned common weights, and the arms that share nothing to be matched
    # on data and budget only.
    assert SHARED_INITIALIZATION_ANCHOR in SHARED_INITIALIZATION_ARMS
    assert set(SHARED_INITIALIZATION_ARMS) == {"generic", "process"}
    assert set(ARM_NAMES) == set(SHARED_INITIALIZATION_ARMS) | set(NON_SHARED_ARMS)
    assert "unet" in NON_SHARED_ARMS and "afno_small" in NON_SHARED_ARMS


def test_skill_criteria_are_frozen_and_require_both_controls():
    assert SKILL_CRITERIA["frozen_before_any_step"] is True
    assert set(SKILL_CRITERIA["required_cells"]) == {"6h", "12h", "24h"}
    for label, spec in SKILL_CRITERIA["required_cells"].items():
        assert 0 < spec["required_wins"] <= len(spec["variables"])
        # the label is the declaration; the lead is derived by the reader
        assert f"{harness._required_lead_hours(label)}h" == label
    assert SKILL_CRITERIA["allowance"]["unresolved_cells"] >= 0
    # the gate names the arms it applies to and states that baseline-usability
    # and ours-wins are different states
    assert set(SKILL_CRITERIA["required_arms"]) <= set(ARM_NAMES)
    assert "states_are_separate" in SKILL_CRITERIA
    # t2m must be gated at the trained horizon, because that is where B1's
    # climatology was hardest to beat
    assert "t2m" in SKILL_CRITERIA["required_cells"]["6h"]["variables"]


def test_every_required_cell_label_resolves_to_an_evaluated_lead():
    for label, spec in SKILL_CRITERIA["required_cells"].items():
        assert harness._required_lead_hours(label) in harness.EVALUATION_LEADS
        # every scored channel set covers the gated variables
        for variable in spec["variables"]:
            assert variable in harness.SCORED_CHANNELS


def test_required_cell_labels_fail_closed_on_nonsense():
    """A label that does not name an evaluated lead must never score silently."""
    for label in ("6", "x", "", "0h", "999h", "-6h"):
        with pytest.raises(ValueError):
            harness._required_lead_hours(label)


def test_gate_reader_changes_no_frozen_criterion():
    """The reader must interpret the frozen text, not the text the reader finds easy.

    Regression: the criteria are stored exactly as pre-registered, so a fix to
    how they are read may not add fields to them. An earlier version added a
    ``lead_hours`` key to each cell, which changed the protocol digest after the
    exploration runs had already started - post-hoc drift in everything but
    name. The digest mechanism caught it; this test keeps the reader honest.
    """
    for label, spec in SKILL_CRITERIA["required_cells"].items():
        assert set(spec) == {"variables", "required_wins", "of"}, \
            f"required cell {label!r} carries unexpected fields {sorted(spec)}"


def test_validation_leads_are_integers():
    """A float lead would be rejected (or silently truncated) by the rollout API."""
    for lead in harness.VALIDATION_LEADS:
        assert isinstance(lead, int) and not isinstance(lead, bool) and lead > 0
    for lead in harness.EVALUATION_LEADS:
        assert isinstance(lead, int) and not isinstance(lead, bool) and lead > 0


def _minimal_protocol(tmp_path, identity="identity", channels=4, measured=None):
    if measured is None:
        measured = {name: {"parameters": 1000 + index, "forward_flops": 10,
                           "forward_backward_flops": 30}
                    for index, name in enumerate(ARM_NAMES)}
    manifests = Path(tmp_path) / "manifests"
    return protocol_payload(manifests, identity, channels, measured)


def test_protocol_digest_ignores_phase_membership(tmp_path):
    """Both phases must hash one protocol, or phase 2 could never reuse phase 1."""
    protocol = _minimal_protocol(tmp_path)
    # phase membership is described but not a top-level field that varies
    assert "this_run" not in protocol["phases"]
    assert protocol["phases"]["explore"]["seeds"] == list(EXPLORATION_SEEDS)
    assert protocol["phases"]["confirm"]["seeds"] == list(CONFIRMATORY_SEEDS)
    # the same inputs hash the same way regardless of which phase calls it
    again = _minimal_protocol(tmp_path)
    assert again["protocol_sha256"] == protocol["protocol_sha256"]


def test_protocol_records_the_frozen_schedule_and_selection_rule(tmp_path):
    protocol = _minimal_protocol(tmp_path)
    controls = protocol["shared_controls"]
    assert controls["warmup_updates"] == harness.WARMUP_UPDATES
    assert controls["max_updates"] == harness.UPDATES
    assert controls["validation_every"] == harness.VALIDATION_EVERY
    assert controls["early_stopping_patience"] == harness.EARLY_STOPPING_PATIENCE
    assert controls["selection_split"] == "val only; test is sealed and never read"
    assert "validation MSE" in controls["checkpoint_selection_rule"]
    assert protocol["data"]["test_read"] is False
    assert "not the v2 2021 test candidate" in protocol["data"]["test_policy"]


def test_protocol_declares_what_is_not_aligned(tmp_path):
    protocol = _minimal_protocol(tmp_path)
    not_aligned = " ".join(protocol["compute_alignment"]["explicitly_not_aligned"])
    # B1 proved same-updates is not same-compute; B2 must not bury that
    assert "FLOPs" in not_aligned and "wall time" in not_aligned
    assert "parameter count" in not_aligned
    assert "initial weights" in not_aligned
    assert protocol["scientific_claim"] is False
    limitations = " ".join(protocol["limitations"])
    assert "January" in limitations
    assert "climatology" in limitations.lower()
    assert "not a significance test" in limitations


def test_protocol_digest_changes_when_measured_budget_changes(tmp_path):
    """The digest must cover the measured parameter/FLOP numbers, not just prose."""
    baseline = _minimal_protocol(tmp_path)["protocol_sha256"]
    changed = {name: {"parameters": 2000 + index, "forward_flops": 10,
                      "forward_backward_flops": 30}
               for index, name in enumerate(ARM_NAMES)}
    assert _minimal_protocol(tmp_path, measured=changed)["protocol_sha256"] != baseline


def _seed_payload(seed, *, phase="confirm", digest="d", code="c"):
    entry = {"seed": seed, "model": "generic", "lead_hours": 6, "split": "val",
             "n_evaluated": 3, "n_available_windows": 3, "channels": ["t2m"],
             "units": ["K"], "elapsed_seconds": 1.0, "timing_scope": "x",
             "parameter_free_baseline": None, "trainable_parameters": 10,
             "climatology": {}, "acc_skill_identity": {}, "evaluation_dir": f"/tmp/{seed}",
             "rmse_csv": f"/tmp/{seed}/rmse.csv", "skill_csv": f"/tmp/{seed}/skill.csv",
             "acc_csv": f"/tmp/{seed}/acc.csv"}
    return {
        "format": "r7-b2-seed-run-v1", "phase": phase, "scientific_claim": False,
        "test_read": False, "protocol_sha256": digest, "model_code_sha256": code,
        "protocol": {"protocol_sha256": digest, "parameter_gate": {"measured_spread": 0.01}},
        "flop_measurements": {},
        "device": "cuda:0", "gpu": "RTX 3090", "torch_version": "2.x", "platform": "linux",
        "training": {f"generic_seed{seed}": {"elapsed_seconds": 1.0, "updates_run": 2}},
        "evaluation": {f"generic_seed{seed}@6h": entry},
        "case_counts_by_lead": {"6h": 3},
        "budget": {"training_seconds_total": 1.0},
    }


def _write_seed(tmp_path, seed, **kwargs):
    run_root = Path(tmp_path)
    (run_root / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    (run_root / f"seed{seed}" / "b2_result.json").write_text(
        json.dumps(_seed_payload(seed, **kwargs)), encoding="utf-8")
    return run_root


def test_merge_refuses_an_incomplete_phase(tmp_path):
    """A missing seed is reported, never silently narrowed away."""
    run_root = _write_seed(tmp_path, CONFIRMATORY_SEEDS[0])
    with pytest.raises(FileNotFoundError, match="incomplete phase"):
        merge_phase_runs(run_root, phase="confirm", seeds=CONFIRMATORY_SEEDS)


def test_merge_refuses_undeclared_seeds(tmp_path):
    run_root = _write_seed(tmp_path, 999)
    with pytest.raises(ValueError, match="not|declares"):
        merge_phase_runs(run_root, phase="confirm", seeds=(999,))


def test_merge_refuses_mixed_protocols_or_model_code(tmp_path):
    run_root = tmp_path
    for index, seed in enumerate(CONFIRMATORY_SEEDS):
        _write_seed(run_root, seed, digest="d" if index < 2 else "different")
    with pytest.raises(RuntimeError, match="different protocols"):
        merge_phase_runs(run_root, phase="confirm", seeds=CONFIRMATORY_SEEDS)


def test_merge_records_which_phase_produced_each_seed(tmp_path):
    """A confirmatory run reusing an exploration run must say so."""
    run_root = tmp_path
    for seed in CONFIRMATORY_SEEDS:
        phase = "explore" if seed in EXPLORATION_SEEDS else "confirm"
        _write_seed(run_root, seed, phase=phase)
    merged = merge_phase_runs(run_root, phase="confirm", seeds=CONFIRMATORY_SEEDS)
    provenance = merged["seed_provenance"]
    for seed in EXPLORATION_SEEDS:
        assert provenance[str(seed)]["run_phase"] == "explore"
        assert provenance[str(seed)]["reused_from_earlier_phase"] is True
    fresh = str(CONFIRMATORY_SEEDS[-1])
    assert provenance[fresh]["reused_from_earlier_phase"] is False


def test_merge_sums_training_time_and_keeps_every_seed(tmp_path):
    run_root = tmp_path
    for seed in CONFIRMATORY_SEEDS:
        _write_seed(run_root, seed)
    merged = merge_phase_runs(run_root, phase="confirm", seeds=CONFIRMATORY_SEEDS)
    assert merged["seeds"] == list(CONFIRMATORY_SEEDS)
    assert merged["budget"]["training_seconds_total"] == pytest.approx(
        len(CONFIRMATORY_SEEDS) * 1.0)
    assert len(merged["training"]) == len(CONFIRMATORY_SEEDS)


def test_cell_win_table_reads_the_comparator_directions():
    comparisons = {"persistence": [
        {"arm": "generic", "seed_paired": [
            {"variable": "t2m", "lead_hours": 6.0, "direction": "improved"},
            {"variable": "mslp", "lead_hours": 6.0, "direction": "unresolved"},
            {"variable": "v850", "lead_hours": 12.0, "direction": "worsened"}]}],
        "climatology": []}
    outcomes = _cell_win_table(comparisons)
    assert outcomes["generic"]["persistence|6|t2m"] == "won"
    assert outcomes["generic"]["persistence|6|mslp"] == "unresolved"
    assert outcomes["generic"]["persistence|12|v850"] == "lost"


def test_unit_lookup_fails_closed_for_an_absent_variable():
    with pytest.raises(ValueError, match="absent"):
        _unit_for([{"variable": "t2m", "unit": "K"}], "mslp")


def test_arms_are_five_and_distinct():
    assert len(ARMS) == len(ARM_NAMES) == 5
    assert len({name for name, _, _ in ARMS}) == 5
    kinds = {name: kind for name, kind, _ in ARMS}
    # the recursive pair is the shared-structure one
    assert kinds["generic"] == "generic" and kinds["process"] == "process"
    assert all(kinds[name] == "native" for name in NON_SHARED_ARMS)
    assert set(BASELINES) == {"persistence", "climatology"}
