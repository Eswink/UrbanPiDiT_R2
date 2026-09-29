"""#72 M2-B (RW-B) bounded round: protocol freeze, arm pairing and the reading rule.

Two kinds of test here. The first kind is structural: the four arms of the round
must come from one seeded initialization, the reference arm must load into every
other arm through the trainer's own transfer rule, and each successive switch has
to actually add a tensor -- otherwise a comparison between two arms would be a
comparison between two identical models. The second kind is the reading rule: the
registered primary turns comparator cells into a verdict, and every rule in that
reading needs a counterproof, because a reading function that returns "unresolved"
for every input would also pass a test that only checked it returns a string.

The arm checks construct models but run no forward pass on real data, so they stay
on CPU. Everything that reads the M2 store is exercised by the round itself and by
``tests/test_r7_arm_harness.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from training.r7_arm_harness import (measure_arms, merge_seed_results, pair_cells,
                                     verify_arm_pairing, write_study_tables)
from training.r7_solver_probes import write_depth_probe_table


def _load_driver():
    """Import the round driver by path; it is a script, not an installed module."""
    path = ROOT / "scripts" / "study_r7_72_rw_b.py"
    spec = importlib.util.spec_from_file_location("study_r7_72_rw_b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_driver()


def _comparator_cell(lead, variable, deltas, direction, unit="K"):
    """One ``seed_paired`` row as the fixed comparator emits it."""
    seeds = sorted(deltas)
    signs = {0 if deltas[seed] == 0 else (1 if deltas[seed] > 0 else -1) for seed in seeds}
    return {"lead_hours": lead, "variable": variable, "unit": unit, "direction": direction,
            "sign_consistent": len(signs) == 1,
            "seed_deltas": [{"seed": seed, "delta": deltas[seed]} for seed in seeds]}


def _pair_cell(lead, variable, deltas, direction, unit="K"):
    """The same row after ``pair_cells``: what ``primary_reading`` consumes."""
    return {"lead_hours": lead, "variable": variable, "unit": unit, "direction": direction,
            "sign_consistent": len({0 if value == 0 else (1 if value > 0 else -1)
                                    for value in deltas.values()}) == 1,
            "seed_deltas": {str(seed): value for seed, value in deltas.items()}}


def _pair_payload(cells):
    """The shape ``pair_cells`` produces, which ``primary_reading`` consumes."""
    directions = [entry["direction"] for entry in cells.values()]
    totals = {name: directions.count(name)
              for name in ("improved", "worsened", "unresolved")}
    return {"cells": cells, "totals": totals}


def _primary_cells(deltas_by_lead, *, pair_cells_map=None, variables=("t2m",)):
    cells = {}
    for lead, deltas in deltas_by_lead.items():
        labels = list(deltas)
        directions = {name: ("improved" if deltas[name] < 0 else "worsened")
                      for name in labels}
        outcome = directions[labels[0]]
        cells[f"{lead}h|t2m"] = _pair_cell(lead, "t2m", deltas, outcome)
    return cells


# ---------------------------------------------------------------------------
# Protocol freeze
# ---------------------------------------------------------------------------

def test_protocol_payload_freezes_the_declared_primary_and_registers_every_arm():
    measured = {name: {"parameters": 100 + index, "forward_flops": 1000 + index,
                       "forward_backward_flops": 3000 + index}
                for index, name in enumerate(study.ARM_NAMES)}
    payload = study.protocol_payload(Path("outputs/r7_m2_segment/store/manifests"),
                                     "deadbeef", 17, measured)
    assert payload["frozen_before_any_step"] is True
    assert payload["scientific_claim"] is False
    assert payload["limitations"] and all(isinstance(item, str) and item
                                          for item in payload["limitations"])
    assert payload["data"]["test_read"] is False
    assert payload["seeds"] == list(study.SEEDS)
    assert [entry["name"] for entry in payload["arms"]] == list(study.ARM_NAMES)
    for entry in payload["arms"]:
        assert entry["parameters"] == measured[entry["name"]]["parameters"]
        assert entry["forward_flops"] == measured[entry["name"]]["forward_flops"]
    registration = payload["primary_registration"]
    assert registration["registered_before_first_optimizer_step"] is True
    assert registration["variable"] == study.PRIMARY_VARIABLE
    assert tuple(registration["leads_hours"]) == study.PRIMARY_LEADS
    assert [(entry["focus"], entry["baseline"])
            for entry in registration["pairs"]] == [study.PRIMARY_PAIR]
    assert registration["decision_text"] == study.PRIMARY_DECISION_TEXT


def test_protocol_digest_survives_a_json_round_trip(tmp_path):
    """The pin is a file the run writes; a digest that only survives in memory is not one."""
    from training.r7_experiment import canonical_digest

    measured = {name: {"parameters": 1, "forward_flops": 2, "forward_backward_flops": 3}
                for name in study.ARM_NAMES}
    payload = study.protocol_payload(tmp_path, "identity", 17, measured)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_digest({key: value for key, value in frozen.items()
                             if key != "protocol_sha256"}) == payload["protocol_sha256"]
    assert study.verified_registration(path)["protocol_sha256"] == payload["protocol_sha256"]


@pytest.mark.parametrize("field,value", [
    ("variable", "z500"),
    ("leads_hours", [6, 12]),
    ("decision_text", "   "),
])
def test_registration_guard_rejects_a_mutated_primary(tmp_path, field, value):
    """Counterproof: the guard is not a no-op that accepts whatever is on disk."""
    measured = {name: {"parameters": 1, "forward_flops": 2, "forward_backward_flops": 3}
                for name in study.ARM_NAMES}
    payload = study.protocol_payload(tmp_path, "identity", 17, measured)
    payload["primary_registration"][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RuntimeError):
        study.verified_registration(path)


def test_registration_guard_rejects_a_protocol_whose_body_no_longer_matches_the_digest(
        tmp_path):
    measured = {name: {"parameters": 1, "forward_flops": 2, "forward_backward_flops": 3}
                for name in study.ARM_NAMES}
    payload = study.protocol_payload(tmp_path, "identity", 17, measured)
    payload["seeds"] = [41]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RuntimeError):
        study.verified_registration(path)


def test_the_test_manifest_is_refused_before_the_reader_sees_it():
    study.refused_test_manifest(Path("outputs/r7_m2_segment/store/manifests/val.jsonl"))
    with pytest.raises(ValueError):
        study.refused_test_manifest(Path("outputs/r7_m2_segment/store/manifests/test.jsonl"))


# ---------------------------------------------------------------------------
# Arm pairing, measured
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pairing():
    return verify_arm_pairing(study.ARMS, baseline=study.BASELINE_ARM, channels=17, seed=41)


def test_every_arm_shares_one_bitwise_identical_initialization(pairing):
    assert pairing["all_shared_pairs_identical"] is True
    for key, entry in pairing["pairwise_shared_tensors"].items():
        assert entry["shared_tensors"] > 0, key
        assert entry["shared_tensors_identical"] is True, key
    assert pairing["anchor"] == study.BASELINE_ARM


def test_the_reference_arm_loads_into_every_other_arm_with_nothing_left_over(pairing):
    assert set(pairing["anchor_transfer"]) == set(study.ARM_NAMES) - {study.BASELINE_ARM}
    for name, entry in pairing["anchor_transfer"].items():
        assert entry["ignored_count"] == 0, name
        assert entry["post_load_all_applied_bitwise_equal"] is True, name
        assert entry["applied_count"] == pairing["tensor_counts"][study.BASELINE_ARM]


def test_each_switch_adds_its_own_tensors_over_the_previous_arm(pairing):
    """RW-A adds the reader, RW-B the solver, the role arm the two role vectors."""
    chain = ((study.RW_A_ARM, study.BASELINE_ARM),
             (study.RW_B_ARM, study.RW_A_ARM),
             (study.RW_B_ROLES_ARM, study.RW_B_ARM))
    counts = [pairing["tensor_counts"][name] for name, _ in chain]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    for focus, baseline in chain:
        entry = pairing["pairwise_shared_tensors"]["|".join(sorted((focus, baseline)))]
        assert entry["added_in_first"] or entry["added_in_second"], focus
    roles = pairing["pairwise_shared_tensors"]["|".join(sorted((study.RW_B_ROLES_ARM,
                                                                study.RW_B_ARM)))]
    added = set(roles["added_in_first"]) | set(roles["added_in_second"])
    assert added == {"role_context", "role_draft"}, sorted(added)


def test_the_tensor_counts_are_what_makes_the_switch_guard_non_vacuous(pairing):
    """The guard reads tensor counts, so an arm compared with itself would be rejected.

    ``run_seed`` refuses a pair whose tensor sets are equal and whose counts do not
    grow. This checks the two facts that guard depends on, and that the pairing
    table itself never contains a self-pair that would make the guard fire on a
    healthy round.
    """
    keys = list(pairing["pairwise_shared_tensors"])
    assert all(key.split("|")[0] != key.split("|")[1] for key in keys)
    for focus, baseline in ((study.RW_A_ARM, study.BASELINE_ARM),
                            (study.RW_B_ARM, study.RW_A_ARM),
                            (study.RW_B_ROLES_ARM, study.RW_B_ARM)):
        assert pairing["tensor_counts"][focus] > pairing["tensor_counts"][baseline]


def test_the_role_arm_adds_exactly_two_role_vectors_and_no_other_parameter():
    """RW-B + roles differs from RW-B by 2*D parameters, the two role vectors."""
    from training.r7_experiment import make_model, seed_everything
    from training.r7_arm_harness import arm_config

    config = dict(study.BASE_ARM_CONFIG, dim=32, depth=1, heads=2)
    with_roles = dict(config, source_role_markers=True)
    seed_everything(41)
    plain = make_model("process", arm_config("process", 17, config))
    seed_everything(41)
    roles = make_model("process", arm_config("process", 17, with_roles))
    added = sum(parameter.numel() for parameter in roles.parameters()) \
        - sum(parameter.numel() for parameter in plain.parameters())
    assert added == 2 * 32


def test_measured_arm_costs_increase_strictly_with_the_added_modules():
    """The cost table is not decoration: RW-A and RW-B really do buy more compute."""
    import torch
    from torch.utils.data import default_collate

    from data.r7_zarr_dataset import ZarrAtmosWindowDataset

    manifests = ROOT / "outputs" / "r7_m2_segment" / "store" / "manifests"
    if not (manifests / "train.jsonl").is_file():
        pytest.skip("the M2 segment is not present locally")
    dataset = ZarrAtmosWindowDataset(manifests / "train.jsonl")
    probe = default_collate([dataset[0], dataset[1]])
    measured = measure_arms(study.ARMS, channels=17, probe_batch=probe,
                            reasoning_steps=1, seed=41)
    parameters = [measured[name]["parameters"] for name in study.ARM_NAMES]
    flops = [measured[name]["forward_flops"] for name in study.ARM_NAMES]
    assert parameters == sorted(parameters) and len(set(parameters)) == len(parameters)
    assert flops == sorted(flops) and flops[0] < flops[-1]
    assert measured[study.RW_B_ROLES_ARM]["forward_flops"] == \
        measured[study.RW_B_ARM]["forward_flops"], "role markers are additive constants"
    del probe


# ---------------------------------------------------------------------------
# The reading rule
# ---------------------------------------------------------------------------

def test_a_sign_consistent_negative_delta_reads_supported():
    cells = _primary_cells({6: {41: -0.1, 42: -0.2}, 12: {41: -0.3, 42: -0.1}})
    reading = study.primary_reading({f"{study.RW_B_ARM} - {study.RW_A_ARM}":
                                     _pair_payload(cells)})
    primary = reading["primary"]
    assert primary["verdict_counts"] == {"supported": 2}
    assert primary["per_lead"]["6"]["outcome"] == "supported"
    assert primary["per_lead"]["6"]["delta_seed_mean"] == pytest.approx(-0.15)
    assert "supported" in primary["headline"]


def test_a_sign_consistent_positive_delta_reads_worsened():
    cells = _primary_cells({6: {41: 0.4, 42: 0.2}})
    reading = study.primary_reading({f"{study.RW_B_ARM} - {study.RW_A_ARM}":
                                     _pair_payload(cells)})
    assert reading["primary"]["verdict_counts"] == {"worsened": 1}
    assert reading["primary"]["per_lead"]["6"]["outcome"] == "worsened"


def test_a_lead_whose_seeds_disagree_is_unresolved_and_is_not_counted():
    """The whole point of the comparator's sign rule: no averaging a disagreement away."""
    cells = _primary_cells({6: {41: -0.1, 42: 0.2}, 12: {41: -0.2, 42: -0.3}})
    reading = study.primary_reading({f"{study.RW_B_ARM} - {study.RW_A_ARM}":
                                     _pair_payload(cells)})
    primary = reading["primary"]
    assert primary["per_lead"]["6"]["outcome"] == "unresolved"
    assert primary["per_lead"]["6"]["delta_seed_mean"] is None
    assert primary["verdict_counts"] == {"supported": 1}


def test_flipping_one_seed_delta_flips_the_verdict():
    """Counterproof: the verdict is a function of the numbers, not a constant."""
    supported = _primary_cells({6: {41: -0.1, 42: -0.2}})
    worsened = _primary_cells({6: {41: -0.1, 42: 0.2}})
    pair_key = f"{study.RW_B_ARM} - {study.RW_A_ARM}"
    first = study.primary_reading({pair_key: _pair_payload(supported)})
    second = study.primary_reading({pair_key: _pair_payload(worsened)})
    assert first["primary"]["per_lead"]["6"]["outcome"] == "supported"
    assert second["primary"]["per_lead"]["6"]["outcome"] == "unresolved"
    both_down = study.primary_reading(
        {pair_key: _pair_payload(_primary_cells({6: {41: 0.1, 42: 0.2}}))})
    assert both_down["primary"]["per_lead"]["6"]["outcome"] == "worsened"


def test_no_sign_consistent_lead_is_reported_as_such_rather_than_as_a_win():
    cells = {f"{lead}h|t2m": _pair_cell(lead, "t2m", {41: -0.1, 42: 0.2}, "unresolved")
             for lead in study.PRIMARY_LEADS}
    reading = study.primary_reading({f"{study.RW_B_ARM} - {study.RW_A_ARM}":
                                     _pair_payload(cells)})
    assert reading["primary"]["verdict_counts"] == {}
    assert reading["primary"]["headline"] == "no sign-consistent primary cell"


def test_disagreeing_leads_are_reported_as_a_disagreement_not_a_summary():
    cells = _primary_cells({6: {41: -0.1, 42: -0.2}, 12: {41: 0.1, 42: 0.2}})
    reading = study.primary_reading({f"{study.RW_B_ARM} - {study.RW_A_ARM}":
                                     _pair_payload(cells)})
    assert reading["primary"]["verdict_counts"] == {"supported": 1, "worsened": 1}
    assert "disagree" in reading["primary"]["headline"]


def test_the_role_marker_cell_is_read_the_same_way_and_is_not_merged_into_the_primary():
    cells = _primary_cells({6: {41: -0.1, 42: -0.2}})
    reading = study.primary_reading({
        f"{study.RW_B_ARM} - {study.RW_A_ARM}": _pair_payload(cells),
        f"{study.RW_B_ROLES_ARM} - {study.RW_B_ARM}": _pair_payload(
            _primary_cells({6: {41: 0.5, 42: 0.7}}))})
    assert reading["primary"]["verdict_counts"] == {"supported": 1}
    assert reading["role_marker_cell"]["reported_exactly_the_same_way"]["verdict_counts"] \
        == {"worsened": 1}
    assert reading["primary"]["pair"] != reading["role_marker_cell"]["pair"]


def test_pair_cells_counts_only_sign_consistent_cells_as_wins_or_losses():
    block = {"arm": "focus", "baseline": "base", "depth": 0, "case_identity": None,
             "variables_improved": 1, "variables_worsened": 0, "variables_compared": 3,
             "variables_unresolved": 2, "variables_sign_consistent": 1,
             "established_improved": [], "established_worsened": [],
             "beats_baseline_everywhere": False,
             "seed_paired": [_comparator_cell(6, "t2m", {41: -1.0, 42: -2.0}, "improved"),
                             _comparator_cell(6, "z500", {41: -1.0, 42: 2.0}, "unresolved"),
                             _comparator_cell(12, "t2m", {41: 1.0, 42: 3.0}, "worsened")]}
    pairs = pair_cells({("focus", "base"): block}, leads=(6, 12))
    entry = pairs["focus - base"]
    assert entry["totals"] == {"improved": 1, "worsened": 1, "unresolved": 1}
    assert entry["per_lead"]["6"] == {"improved": 1, "worsened": 0, "unresolved": 1}
    assert entry["per_lead"]["12"] == {"improved": 0, "worsened": 1, "unresolved": 0}
    assert entry["cells"]["6h|z500"]["sign_consistent"] is False
    assert entry["cells"]["6h|t2m"]["seed_deltas"] == {"41": -1.0, "42": -2.0}


# ---------------------------------------------------------------------------
# Merge and tables fail closed
# ---------------------------------------------------------------------------

def _seed_result(seed, *, protocol="p", code="c", arms=None):
    arms = list(study.ARM_NAMES) if arms is None else arms
    return {"protocol_sha256": protocol, "model_code_sha256": code, "protocol": {},
            "flop_measurements": {},
            "device": "cpu", "gpu": None, "torch_version": "test",
            "arm_pairing": {"all_shared_pairs_identical": True},
            "training": {name: {"protocol_sha256": protocol,
                                "selected_update": 1, "elapsed_seconds": 1.0,
                                "seconds_per_update": 1.0, "early_stopped": False,
                                "updates_run": 1, "peak_allocated_bytes": 1,
                                "peak_reserved_bytes": 2, "shared_initial_state":
                                    {"applied_count": 0, "ignored_count": 0}}
                         for name in arms},
            "evaluation": {"x": {"arm": "x", "seed": seed, "lead_hours": 6,
                                 "elapsed_seconds": 1.0, "evaluation_dir": "d",
                                 "rmse_csv": "r", "skill_csv": "s"}},
            "case_counts_by_lead": {"6": 1}, "probes": {}, "budget":
                {"training_seconds_total": 1.0}}


def _write_seed(output_dir, result, seed):
    target = output_dir / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "seed_result.json").write_text(json.dumps(result), encoding="utf-8")


def test_merge_refuses_an_incomplete_round(tmp_path):
    _write_seed(tmp_path, _seed_result(41), 41)
    with pytest.raises(FileNotFoundError):
        merge_seed_results(tmp_path, seeds=(41, 42), arms=study.ARM_NAMES, fmt="f")


def test_merge_refuses_two_protocols_in_one_round(tmp_path):
    _write_seed(tmp_path, _seed_result(41, protocol="p1"), 41)
    _write_seed(tmp_path, _seed_result(42, protocol="p2"), 42)
    with pytest.raises(RuntimeError, match="protocol"):
        merge_seed_results(tmp_path, seeds=(41, 42), arms=study.ARM_NAMES, fmt="f")


def test_merge_refuses_two_model_code_digests_in_one_round(tmp_path):
    _write_seed(tmp_path, _seed_result(41, code="c1"), 41)
    _write_seed(tmp_path, _seed_result(42, code="c2"), 42)
    with pytest.raises(RuntimeError, match="model code"):
        merge_seed_results(tmp_path, seeds=(41, 42), arms=study.ARM_NAMES, fmt="f")


def test_merge_refuses_a_round_missing_an_arm(tmp_path):
    _write_seed(tmp_path, _seed_result(41, arms=study.ARM_NAMES[:-1]), 41)
    _write_seed(tmp_path, _seed_result(42, arms=study.ARM_NAMES[:-1]), 42)
    with pytest.raises(RuntimeError, match="arm runs"):
        merge_seed_results(tmp_path, seeds=(41, 42), arms=study.ARM_NAMES, fmt="f")


def test_merge_carries_every_run_digest_when_the_round_is_complete(tmp_path):
    _write_seed(tmp_path, _seed_result(41), 41)
    _write_seed(tmp_path, _seed_result(42), 42)
    merged = merge_seed_results(tmp_path, seeds=(41, 42), arms=study.ARM_NAMES, fmt="f")
    assert merged["test_read"] is False and merged["scientific_claim"] is False
    assert len(merged["run_protocol_sha256"]) == 2 * len(study.ARM_NAMES)
    assert set(merged["run_protocol_sha256"].values()) == {"p"}


def test_the_correction_probe_reads_the_rollout_reader_field_names():
    """Pin the field the correction probe maps, and the depth it maps it from.

    The first attempt at this round died here: ``collect_correction_terms`` wants
    ``atmos_target``, the rollout reader calls the same thing ``rollout_targets``
    and stacks one entry per declared lead. Both halves of that mapping are
    checked against the reader itself, because a rename on either side has to
    fail loudly rather than at the end of a training round.
    """
    from data.r7_evaluation import ZarrRolloutDataset

    store = ROOT / "outputs" / "r7_m2_segment" / "store" / "cache.zarr"
    if not store.exists():
        pytest.skip("the M2 segment is not present locally")
    dataset = ZarrRolloutDataset(store, split="val", lead_hours=(6,), history_steps=2,
                                 step_hours=6)
    sample = dataset[0]
    assert "rollout_targets" in sample
    assert "atmos_target" not in sample, "the probe maps this name itself"
    assert "latitude" in sample
    assert sample["rollout_targets"].shape[0] == 1, "one target stack per declared lead"


def test_tables_are_written_once_and_a_second_finalize_cannot_overwrite_them(tmp_path):
    merged = {"protocol": {"arms": [{"name": study.BASELINE_ARM, "parameters": 1,
                                     "forward_flops": 2, "forward_backward_flops": 3,
                                     "switches": {"local_solver_state": False}}]},
              "training": {"41": {study.BASELINE_ARM: {
                  "protocol_sha256": "p", "selected_update": 1, "updates_run": 1,
                  "early_stopped": False, "seconds_per_update": 1.0,
                  "elapsed_seconds": 2.0, "peak_allocated_bytes": 1,
                  "peak_reserved_bytes": 2,
                  "shared_initial_state": {"applied_count": 0, "ignored_count": 0}}}},
              "evaluation": {}, "probes": {"41": {"depth": {}}}}
    write_study_tables(merged, tmp_path)
    for name in ("arm_table.csv", "training_table.csv", "memory_table.csv",
                 "rmse_table.csv", "case_table.csv"):
        assert (tmp_path / name).is_file(), name
    write_depth_probe_table(merged, tmp_path)
    assert (tmp_path / "depth_probe_table.csv").is_file()
    with pytest.raises(FileExistsError):
        write_study_tables(merged, tmp_path)
    with pytest.raises(FileExistsError):
        write_depth_probe_table(merged, tmp_path)
