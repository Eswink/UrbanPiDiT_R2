"""Offline tests for the B3 capacity scale-up contract (#64 B3).

No GPU, no real data: these pin the invariants that make B3 a *controlled*
capacity ablation - the 15-20M hard band, the declared seed set, the
phase-independent protocol digest, the refusal to report an incomplete stage,
and the cross-scale comparison's refusal to compare against an absent B2 table
without saying so.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.study_r7_b3_scale as b3
from scripts.study_r7_b3_scale import (
    ARMS,
    ARM_NAMES,
    B2_SCALE_PARAMETERS,
    BASELINES,
    MAXIMUM_PARAMETERS,
    MINIMUM_PARAMETERS,
    SCALE_SEEDS,
    SHARED_INITIALIZATION_ANCHOR,
    SHARED_INITIALIZATION_ARMS,
    _cross_scale_comparison,
    protocol_payload,
    write_comparison,
)


def test_scale_target_is_15_to_20m_and_never_30m():
    assert MINIMUM_PARAMETERS == 15_000_000
    assert MAXIMUM_PARAMETERS == 20_000_000
    assert b3.NOMINAL_PARAMETERS == 18_000_000
    # 30M is explicitly not the target; the ceiling must stay below it
    assert MAXIMUM_PARAMETERS < 25_000_000


def test_measured_arm_sizes_sit_inside_the_declared_band():
    """The declared configs must measure inside 15-20M before any training runs."""
    from training.r7_experiment import make_model

    for name, kind, config in ARMS:
        resolved = {"in_channels": 17, "out_channels": 17, "history_steps": 2, **config}
        measured = sum(p.numel() for p in make_model(kind, resolved).parameters())
        assert MINIMUM_PARAMETERS <= measured <= MAXIMUM_PARAMETERS, (
            f"{name} measured {measured:,}, outside the 15-20M requirement")


def test_every_arm_has_a_b2_2p8m_counterpart_to_compare_against():
    assert set(ARM_NAMES) <= set(B2_SCALE_PARAMETERS)
    for name in ARM_NAMES:
        assert B2_SCALE_PARAMETERS[name] > 0


def test_seed_set_is_pre_declared_and_uses_two_cards_without_ddp():
    assert len(SCALE_SEEDS) == 2
    assert len(set(SCALE_SEEDS)) == 2
    # DDP was not used: the cards are not NVLink-coupled, so independent
    # seeds beat one DDP job; the protocol must say so rather than imply it
    protocol = protocol_payload(Path("/tmp/manifests"), "identity", 17,
                               {name: {"parameters": 18_000_000 + index,
                                       "forward_flops": 1,
                                       "b2_parameters_at_2p8M": B2_SCALE_PARAMETERS[name],
                                       "scale_factor_vs_b2": 6.5}
                                for index, name in enumerate(ARM_NAMES)}, 0.01)
    strategy = protocol["hardware_strategy"]
    assert strategy["ddp"] is False
    assert strategy["cards"] == 2
    assert "NVLink" in strategy["ddp_reason"]
    # the placement rule is declared, not left to a default
    assert "one independent seed per card" in strategy["placement"]


def test_shared_initialization_pair_matches_b2():
    # #64 B2 sentence 3 asks the shared-structure pair be carried into later
    # experiments; B3 keeps that pairing rather than re-picking it.
    assert set(SHARED_INITIALIZATION_ARMS) == {"generic", "process"}
    assert SHARED_INITIALIZATION_ANCHOR == "generic"


def _protocol(tmp_path):
    measured = {name: {"parameters": 18_000_000 + index, "forward_flops": 10,
                       "b2_parameters_at_2p8M": B2_SCALE_PARAMETERS[name],
                       "scale_factor_vs_b2": 6.5}
                for index, name in enumerate(ARM_NAMES)}
    return protocol_payload(tmp_path / "manifests", "identity", 17, measured, 0.01)


def test_protocol_declares_capacity_as_the_only_change(tmp_path):
    protocol = _protocol(tmp_path)
    assert "capacity" in protocol["declared_change"].lower()
    assert protocol["scientific_claim"] is False
    assert protocol["data"]["test_read"] is False
    held = protocol["held_fixed_from_b2"]
    assert held["case_sets"].startswith("identical")
    limitations = " ".join(protocol["limitations"])
    assert "January" in limitations
    assert "significance test" in limitations
    assert "FLOPs" in limitations  # capacity changes compute; never assume equal


def test_protocol_digest_is_stable_for_fixed_inputs(tmp_path):
    assert _protocol(tmp_path)["protocol_sha256"] == _protocol(tmp_path)["protocol_sha256"]


def test_protocol_records_the_scale_factor_and_b2_baseline(tmp_path):
    protocol = _protocol(tmp_path)
    for entry in protocol["arms"]:
        assert entry["b2_parameters_at_2p8M"] > 0
        assert entry["scale_factor_vs_b2"] > 1
        assert entry["parameters"] / entry["b2_parameters_at_2p8M"] == pytest.approx(
            entry["scale_factor_vs_b2"])


def _seed_record(seed, *, rmse=1.0, protocol="p", code="c", tmp_path=None):
    entry = {"seed": seed, "model": "generic", "lead_hours": 6, "split": "val",
             "n_evaluated": 3, "n_available_windows": 3, "channels": ["t2m"],
             "units": ["K"], "elapsed_seconds": 1.0,
             "parameter_free_baseline": None, "trainable_parameters": 10,
             "evaluation_dir": str(tmp_path), "rmse_csv": str(tmp_path / f"rmse{seed}.csv"),
             "skill_csv": str(tmp_path / f"skill{seed}.csv")}
    return {
        "format": "r7-b3-scale-seed-v1", "seed": seed, "scientific_claim": False,
        "test_read": False, "protocol_sha256": protocol, "model_code_sha256": code,
        "device": "cuda:0", "gpu": "RTX 3090", "torch_version": "2.x", "platform": "linux",
        "protocol": {"protocol_sha256": protocol,
                     "arms": [{"name": name, "kind": "native", "parameters": 18_000_000,
                               "forward_flops": 1, "b2_parameters_at_2p8M": 2_800_000,
                               "scale_factor_vs_b2": 6.5} for name in ARM_NAMES]},
        "training": {name: {"updates_run": 800, "selected_update": 800, "samples_seen": 1600,
                            "elapsed_seconds": 1.0, "seconds_per_update": 0.001,
                            "early_stopped": False, "peak_reserved_bytes": 1,
                            "epoch_mean_loss": {"0": 1.0}, "epoch_update_counts": {"0": 1},
                            "checkpoint": "c", "checkpoint_sha256": "s"}
                     for name in ARM_NAMES},
        "evaluation": {f"{name}@{lead}h": dict(entry, model=name, lead_hours=lead)
                       for name in list(ARM_NAMES) + list(BASELINES)
                       for lead in b3.EVALUATION_LEADS},
        "case_counts_by_lead": {f"{lead}h": 3 for lead in b3.EVALUATION_LEADS},
    }


def _write_protocol(tmp_path, digest="p"):
    (Path(tmp_path) / "protocol.json").write_text(
        json.dumps({"protocol_sha256": digest,
                    "arms": [{"name": name, "kind": "native", "parameters": 18_000_000,
                              "forward_flops": 1, "b2_parameters_at_2p8M": 2_800_000,
                              "scale_factor_vs_b2": 6.5} for name in ARM_NAMES]}),
        encoding="utf-8")


def _write_seed_records(tmp_path, *, protocol="p", code="c"):
    _write_protocol(tmp_path, digest=protocol)
    for seed in SCALE_SEEDS:
        path = Path(tmp_path) / f"scale_seed{seed}.json"
        path.write_text(json.dumps(_seed_record(seed, protocol=protocol, code=code,
                                                tmp_path=tmp_path)), encoding="utf-8")
        # the comparison reads real RMSE tables, so provide them
        with (Path(tmp_path) / f"rmse{seed}.csv").open("w", encoding="utf-8",
                                                        newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["lead_hours", "variable", "unit", "rmse", "n_initializations"])
            for lead in b3.EVALUATION_LEADS:
                writer.writerow([lead, "t2m", "K", 1.0, 3])
        with (Path(tmp_path) / f"skill{seed}.csv").open("w", encoding="utf-8",
                                                         newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["lead_hours", "variable", "rmse_forecast", "rmse_climatology",
                             "mse_skill", "unit", "n_initializations"])
            for lead in b3.EVALUATION_LEADS:
                writer.writerow([lead, "t2m", 1.0, 1.5, 0.5, "K", 3])
    return Path(tmp_path)


def test_write_comparison_refuses_a_missing_declared_seed(tmp_path):
    _write_protocol(tmp_path)
    for seed in SCALE_SEEDS[:-1]:
        (Path(tmp_path) / f"scale_seed{seed}.json").write_text(
            json.dumps(_seed_record(seed, tmp_path=tmp_path)), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        write_comparison(tmp_path)


def test_write_comparison_finds_records_in_per_seed_directories(tmp_path):
    """One process per card writes into its own directory; both layouts must work."""
    _write_protocol(tmp_path)
    target = Path(tmp_path) / f"seed{SCALE_SEEDS[0]}"
    target.mkdir(parents=True)
    (target / f"scale_seed{SCALE_SEEDS[0]}.json").write_text(
        json.dumps(_seed_record(SCALE_SEEDS[0], tmp_path=tmp_path)), encoding="utf-8")
    # the second seed is still absent, so the failure must name what it looked for
    with pytest.raises(FileNotFoundError) as error:
        write_comparison(tmp_path)
    message = str(error.value)
    assert f"seed{SCALE_SEEDS[1]}" in message
    assert str(SCALE_SEEDS[1]) in message


def test_cross_scale_comparison_reports_an_absent_b2_table_honestly(tmp_path):
    """An unavailable B2 table must be declared, not silently treated as no-op."""
    _write_seed_records(tmp_path)
    records = {seed: json.loads((Path(tmp_path) / f"scale_seed{seed}.json").read_text(
        encoding="utf-8")) for seed in SCALE_SEEDS}
    payload = _cross_scale_comparison(records, Path(tmp_path),
                                     b2_rmse=Path(tmp_path) / "missing_b2.csv")
    assert payload["b2_table_available"] is False
    assert all(row["b2_rmse_2p8M"] is None for row in payload["rows"])
    assert all(row["n_b2_seeds"] == 0 for row in payload["rows"])
    assert "not a significance test" in payload["note"]


def test_cross_scale_comparison_joins_the_b2_table_by_variable_and_lead(tmp_path):
    """The B2 column must be that table's own value, paired on (model, var, lead)."""
    _write_seed_records(tmp_path)
    b2_path = Path(tmp_path) / "b2_rmse_table.csv"
    with b2_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", "variable", "unit", "lead_hours", "rmse",
                         "rmse_climatology", "mse_skill", "n_initializations"])
        for seed in (41, 42, 43):
            for lead in b3.EVALUATION_LEADS:
                writer.writerow([seed, "generic", "t2m", "K", lead, 2.0, 1.5, 0.4, 26])
    records = {seed: json.loads((Path(tmp_path) / f"scale_seed{seed}.json").read_text(
        encoding="utf-8")) for seed in SCALE_SEEDS}
    payload = _cross_scale_comparison(records, Path(tmp_path), b2_rmse=b2_path)
    assert payload["b2_table_available"] is True
    row = next(row for row in payload["rows"]
               if row["arm"] == "generic" and row["variable"] == "t2m"
               and row["lead_hours"] == 6)
    assert row["b2_rmse_2p8M"] == pytest.approx(2.0)
    assert row["n_b2_seeds"] == 3
    # the B3 value here is 1.0, so it improves on the B2 2.8M value
    assert row["improved_vs_b2_2p8M"] is True


def test_write_comparison_emits_the_four_required_tables(tmp_path):
    _write_seed_records(tmp_path)
    result = write_comparison(tmp_path, b2_rmse=Path(tmp_path) / "missing.csv")
    for key in ("parameters", "flops", "wall_time", "cases", "rmse"):
        assert (Path(tmp_path) / result["tables"][key]).is_file(), key
    parameter_rows = list(csv.DictReader(
        (Path(tmp_path) / result["tables"]["parameters"]).open(encoding="utf-8")))
    for row in parameter_rows:
        if row["family"] == "parameter-free":
            assert row["inside_15_20M"] == "not_applicable"
        else:
            assert row["inside_15_20M"] == "True"
