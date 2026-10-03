"""Synthetic tmp_path-only coverage and metric counterexamples; no GPU evidence."""
from copy import deepcopy
from datetime import datetime, timedelta
import math
import csv
import json

import pytest

import training.r7_v2_results as results_module
from training.r7_v2_results import finalize, validate_full_set, verify_execution, verify_metric_artifacts
from training.r7_v2_protocol import (
    B_ARMS, C_ARMS, LEADS, LIMITATIONS, build_protocol, child_contract, digest, job_key,
    planned_jobs, sha256_file, write_json,
)
from training.r7_v2_tables import candidate_selection

from training.r7_v2_tables import (
    REGIONS, aggregate_metrics, paired_comparisons, pooled_statistics, validate_metrics,
)

CHANNELS = ["t2m"] + [f"variable_{index}" for index in range(1, 17)]
UNITS = ["K"] + ["m/s"] * 8 + ["Pa"] * 8


def cases_for(lead=6, count=2):
    start = datetime(2016, 1, 1)
    return [[(start + timedelta(hours=index * 6)).isoformat(),
             [(start + timedelta(hours=index * 6 + lead)).isoformat()]] for index in range(count)]


def metric_fixture(*, lead=6, scale=1., count=2):
    cases = cases_for(lead, count)
    case_rows, regional = [], []
    for region in REGIONS:
        for variable, unit in zip(CHANNELS, UNITS):
            group = []
            for index, (init, valid) in enumerate(cases):
                error = scale * (index + 1)
                row = {"init_time": init, "valid_times": valid, "lead_hours": lead,
                       "region": region, "variable": variable, "unit": unit,
                       "mse": error ** 2, "climatology_mse": 4., "acc_dot": 2 * (2 + error),
                       "acc_forecast_energy": (2 + error) ** 2, "acc_target_energy": 4.}
                group.append(row)
            values = pooled_statistics(group)
            regional.append({"region": region, "variable": variable, "unit": unit,
                             "lead_hours": lead, **values})
            case_rows.extend(group)
    return regional, case_rows, cases


def normalized_rows(*, seed=41, arm="baseline", scale=1., lead=6, kernel=4):
    regions, rows, cases = metric_fixture(lead=lead, scale=scale)
    verified = validate_metrics(regions, rows, cases=cases, channels=CHANNELS, units=UNITS, lead=lead)
    return [{**row, "seed": seed, "arm": arm, "K": kernel} for row in verified]


def test_same_case_exact_physical_rmse_skill_pooled_acc_and_bad_cases():
    regions, rows, cases = metric_fixture(scale=2.)
    result = validate_metrics(regions, rows, cases=cases, channels=CHANNELS, units=UNITS, lead=6)
    first = result[0]
    assert len(result) == 51
    assert first["rmse"] == math.sqrt(10.)
    assert first["mse_skill"] == -1.5
    assert first["pooled_acc"] == pytest.approx(10 / math.sqrt(104))
    assert first["pooled_acc"] > 0 and first["mse_skill"] < 0
    assert first["bad_case_counts"]["worse_than_climatology"] == 1
    assert first["n_initializations"] == 2
    assert {row["unit"] for row in result} == {"K", "m/s", "Pa"}


@pytest.mark.parametrize("change", ["case", "duplicate", "missing-case", "unit", "lead", "missing-region",
                                    "missing-variable", "bad-rmse", "bad-skill", "nan", "negative", "bad-acc"])
def test_metric_identity_cohort_and_calculation_guards_have_counterexamples(change):
    regions, rows, cases = metric_fixture()
    if change == "case":
        rows[0]["init_time"], rows[0]["valid_times"] = cases_for(count=3)[2]
    elif change == "duplicate": rows.append(deepcopy(rows[0]))
    elif change == "missing-case": rows.pop()
    elif change == "unit": rows[0]["unit"] = "C"
    elif change == "lead": rows[0]["lead_hours"] = 12
    elif change == "missing-region": regions = [row for row in regions if row["region"] != "edge_2"]
    elif change == "missing-variable": regions.pop()
    elif change == "bad-rmse": regions[0]["rmse"] += .1
    elif change == "bad-skill": regions[0]["mse_skill"] += .1
    elif change == "nan": rows[0]["mse"] = float("nan")
    elif change == "negative": rows[0]["mse"] = -1
    elif change == "bad-acc": rows[0]["acc_dot"] = 100
    with pytest.raises(ValueError):
        validate_metrics(regions, rows, cases=cases, channels=CHANNELS, units=UNITS, lead=6)


def test_undefined_bad_variable_is_retained_and_not_imputed():
    regions, rows, cases = metric_fixture()
    for row in rows:
        if row["variable"] == CHANNELS[-1]:
            row.update(climatology_mse=0., acc_dot=0., acc_forecast_energy=row["mse"], acc_target_energy=0.)
    for row in regions:
        if row["variable"] == CHANNELS[-1]:
            same = [case for case in rows if case["variable"] == row["variable"] and case["region"] == row["region"]]
            row.update(pooled_statistics(same))
    verified = validate_metrics(regions, rows, cases=cases, channels=CHANNELS, units=UNITS, lead=6)
    undefined = [row for row in verified if row["variable"] == CHANNELS[-1]]
    assert len(verified) == 51 and len(undefined) == 3
    assert all(row["mse_skill"] is None and row["pooled_acc"] is None for row in undefined)
    assert all(row["bad_case_counts"]["zero_anomaly_energy"] == 2 for row in undefined)
    regions[-1]["pooled_acc"] = 0.
    with pytest.raises(ValueError, match="undefined"):
        validate_metrics(regions, rows, cases=cases, channels=CHANNELS, units=UNITS, lead=6)


def test_official_comparator_pairs_by_seed_not_position_and_keeps_mixed():
    rows = []
    for seed, scale in [(41, .5), (42, 2.), (43, .7)]:
        rows.extend(normalized_rows(seed=seed))
        rows.extend(normalized_rows(seed=seed, arm="candidate", scale=scale))
    table = aggregate_metrics(rows, seeds=[41, 42, 43], arms=["baseline", "candidate"], leads=[6], kernels=[4])
    reversed_table = aggregate_metrics(list(reversed(rows)), seeds=[41, 42, 43], arms=["baseline", "candidate"],
                                       leads=[6], kernels=[4])
    assert table == reversed_table
    pairs = paired_comparisons(table, pairs=[("candidate", "baseline")], kernels=[4])
    pair = pairs["candidate - baseline"]
    assert pair["totals"] == {"improved": 0, "worsened": 0, "unresolved": 51}
    assert set(pair["cells"]["k4|full|6h|t2m"]["seed_deltas"]) == {"41", "42", "43"}
    candidate = next(row for row in table if row["arm"] == "candidate" and row["variable"] == "t2m")
    assert candidate["rmse_seed_mean"] != candidate["rmse_pooled_cases"]
    assert "pvalue" not in pair


@pytest.mark.parametrize("scales,outcome", [([.5, .6, .7], "improved"), ([2., 2.1, 2.2], "worsened"),
                                            ([1., .6, .7], "unresolved")])
def test_all_seed_strict_sign_has_no_zero_tolerance_loophole(scales, outcome):
    rows = []
    for seed, scale in zip([41, 42, 43], scales):
        rows.extend(normalized_rows(seed=seed))
        rows.extend(normalized_rows(seed=seed, arm="candidate", scale=scale))
    table = aggregate_metrics(rows, seeds=[41, 42, 43], arms=["baseline", "candidate"], leads=[6], kernels=[4])
    pair = paired_comparisons(table, pairs=[("candidate", "baseline")], kernels=[4])["candidate - baseline"]
    assert pair["totals"][outcome] == 51
    assert sum(pair["totals"].values()) == 51


@pytest.mark.parametrize("change", ["missing-seed3", "wrong-seed", "duplicate", "unit", "case", "missing-K"])
def test_full_seed_and_kernel_sets_cannot_use_silent_intersection(change):
    rows = [row for seed in [41, 42, 43] for kernel in [1, 2, 4]
            for row in normalized_rows(seed=seed, kernel=kernel)]
    if change == "missing-seed3": rows = [row for row in rows if row["seed"] != 43]
    elif change == "wrong-seed": rows[0]["seed"] = 44
    elif change == "duplicate": rows.append(deepcopy(rows[0]))
    elif change == "unit": rows[0]["unit"] = "C"
    elif change == "case": rows[0]["cases"] = cases_for(count=3)[:2][::-1]
    elif change == "missing-K": rows = [row for row in rows if row["K"] != 2]
    with pytest.raises(ValueError):
        aggregate_metrics(rows, seeds=[41, 42, 43], arms=["baseline"], leads=[6], kernels=[1, 2, 4])


def actual_artifacts(*, lead=6, kernel=4, count=2, scale=1., std=2.):
    regional, canonical, cases = metric_fixture(lead=lead, count=count, scale=scale)
    def actual(row, pooled=False):
        converted = {key: value for key, value in row.items() if key not in ("cases", "bad_case_counts", "pooled_acc")}
        converted.update(acc=row["pooled_acc"] if pooled else pooled_statistics([row])["pooled_acc"],
                         mse_climatology=row["climatology_mse"],
                         acc_statistic_units="normalized_anomaly_squared", margin_cells=0 if row["region"] == "full" else 2,
                         n_grid_points=25 if row["region"] == "full" else 1 if row["region"] == "interior" else 24,
                         full_area_fraction=1. if row["region"] == "full" else .04 if row["region"] == "interior" else .96)
        converted.update({key: row[key] / std ** 2 * (count if pooled else 1)
                          for key in ["acc_dot", "acc_forecast_energy", "acc_target_energy"]})
        converted["skill_status"] = "defined" if row["mse_skill"] is not None else "undefined_zero_climatology_energy"
        converted["bad_reasons"] = [key for key, bad in [
            ("negative_mse_skill", converted["mse_skill"] is not None and converted["mse_skill"] < 0),
            ("negative_acc", converted["acc"] is not None and converted["acc"] < 0),
            ("undefined_mse_skill", converted["mse_skill"] is None),
            ("undefined_acc", converted["acc"] is None)] if bad]
        return converted
    initializations, flat = [], []
    for index, (init, valid) in enumerate(cases):
        group = []
        for row in canonical:
            if row["init_time"] != init: continue
            one = {**row, **pooled_statistics([row])}
            converted = actual(one)
            converted.pop("init_time"); converted.pop("valid_times")
            group.append(converted)
        initializations.append({"sample_id": f"case{index}", "init_time": init, "valid_times": valid,
                                "cumulative_reasoning_steps": [kernel * lead // 6], "region_metrics": group})
        flat.extend({"sample_id": f"case{index}", "init_time": init, "valid_time": valid[0],
                     "valid_times": valid, "reasoning_steps": kernel, **row} for row in group)
    regions = [actual(row, pooled=True) for row in regional]
    data = {"channels": CHANNELS, "units": UNITS, "normalization_std": [std] * 17,
            "data_identity": "fixture-data", "evaluation_cases": {str(lead): {"cases": cases, "n_available": count}}}
    protocol = {"data": data, "sources": {"val_manifest_sha256": "fixture-val"}}
    job = {"phase": "evaluate", "seed": 41, "arm": B_ARMS[0], "lead": lead, "reasoning_steps": kernel}
    provenance = {"scientific_claim": False, "limitations": ["synthetic fixture only"], "test_read": False,
                  "split": "val", "lead_hours": [lead], "step_hours": 6, "channels": CHANNELS, "units": UNITS,
                  "n_evaluated": count, "n_available_windows": count, "reasoning_steps": kernel,
                  "checkpoint_training_steps": 4, "independently_trained_k1": False,
                  "inference_options": {"reasoning_steps": kernel}, "training_identity": "fixture-data",
                  "evaluation_manifest_sha256": "fixture-val", "initializations": initializations,
                  "region_metrics": regions, "bad_variable_metrics": [row for row in regions if row["bad_reasons"]],
                  "bad_case_metrics": [row for row in flat if row["bad_reasons"]]}
    return protocol, job, provenance, regions, flat


def csv_fixture(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
                         for row in rows)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_actual_helper_adapter_converts_normalized_sums_and_matches_all_artifacts(tmp_path):
    protocol, job, provenance, regions, flat = actual_artifacts(scale=2.)
    csv_regions = csv_fixture(tmp_path / "regions.csv", regions)
    csv_cases = csv_fixture(tmp_path / "cases.csv", flat)
    verified = verify_metric_artifacts(provenance, csv_regions, csv_cases, protocol, job)
    assert len(verified) == 51
    assert verified[0]["rmse"] == math.sqrt(10.)
    assert verified[0]["mse_skill"] == -1.5
    assert verified[0]["acc_target_energy"] == 4.
    assert verified[0]["seed"] == 41 and verified[0]["K"] == 4
    assert verified[0]["bad_case_counts"]["worse_than_climatology"] == 1
    assert job_key(job) == "evaluate_seed41_continue_l6_lead006h_k4"
    assert set(C_ARMS) == {"old_ours", "process", "matched_generic"}


@pytest.mark.parametrize("change", ["std", "scale", "stat-unit", "case-order", "kernel", "missing-embedded",
                                    "embedded-csv", "missing-bad", "top-csv", "case-horizon"])
def test_actual_helper_adapter_fails_on_identity_or_statistics_drift(change):
    protocol, job, provenance, regions, flat = actual_artifacts(scale=3.)
    # Keep redundant dictionaries independent so edits do not accidentally alter both sides.
    provenance, regions, flat = deepcopy(provenance), deepcopy(regions), deepcopy(flat)
    if change == "std": protocol["data"]["normalization_std"][0] = 1.
    elif change == "scale": flat[0]["acc_dot"] += .1
    elif change == "stat-unit": flat[0]["acc_statistic_units"] = "physical"
    elif change == "case-order": flat[0]["valid_times"] = ["2016-01-02T00:00:00"]
    elif change == "kernel": provenance["initializations"][0]["cumulative_reasoning_steps"] = 1
    elif change == "missing-embedded": provenance["initializations"][0]["region_metrics"].pop()
    elif change == "embedded-csv": provenance["initializations"][0]["region_metrics"][0]["rmse"] += .1
    elif change == "missing-bad": provenance["bad_case_metrics"].pop()
    elif change == "top-csv": regions[0]["mse_skill"] += .1
    elif change == "case-horizon": provenance["initializations"][0]["valid_times"][0] = "2016-01-02T00:00:00"
    with pytest.raises(ValueError): verify_metric_artifacts(provenance, regions, flat, protocol, job)


def test_physical_identity_negative_acc_forces_negative_skill_and_wrong_scaling_fails():
    row = {"mse": 16., "climatology_mse": 4., "acc_dot": -4.,
           "acc_forecast_energy": 4., "acc_target_energy": 4.}
    calculated = pooled_statistics([row])
    assert calculated["pooled_acc"] == -1.
    assert calculated["mse_skill"] == -3.
    assert calculated["bad_case_counts"]["negative_acc_dot"] == 1
    row["mse"] /= 4.
    with pytest.raises(ValueError, match="identity"): pooled_statistics([row])


def full_protocol(tmp_path, stage="B"):
    output = tmp_path / ("r7_74_autoregressive_20261003_attempt01" if stage == "B" else "r7_v2_comparison_20261003_attempt01")
    output.mkdir()
    std = [2.] * 17
    data = {"data_identity": digest("fixture-data"), "channels": CHANNELS, "units": UNITS,
            "normalization_mean": [0.] * 17, "normalization_std": std, "test_read": False,
            "evaluation_cases": {str(lead): {"cases": cases_for(lead, count), "n_available": count}
                                 for lead, count in zip(LEADS, [22, 21, 19, 15, 11])}}
    sources = {"source_sha256": digest("fixture-source"), "val_manifest_sha256": "fixture-val"}
    sidecar = {"identity": digest("fixture-sidecar")}
    windows = {"excluded_sample_ids": [], "window_sha256": digest("fixture-windows")}
    arms, seeds = (B_ARMS, [41, 42]) if stage == "B" else (C_ARMS, [41, 42, 43])
    model_spec = {"detach_between_steps": False, "in_channels": 17, "out_channels": 17}
    from training.r7_v2_frontier import ADAPTIVE_GATE_SCHEMA
    configuration = None if stage == "B" else {
        "mode": "l6", "model_specs": {arm: {"kind": "generic" if arm == "matched_generic" else "process", "model": model_spec}
                                        for arm in arms},
        "initialization": {"anchor": {"kind": "process", "model": model_spec}, "mapping": {arm: {"w": "w"} for arm in arms}},
        "primary": {"variable": "t2m", "lead_hours": [6, 12], "unit": "K"}, "tolerances": {"K": 0.},
        "case_unit_selection": "same exact full per-lead cohort", "adaptive_gate": deepcopy(ADAPTIVE_GATE_SCHEMA),
        "reference": "fixture-independent-C", "selection_evidence": "no posthoc choice"}
    pairing, measurements = {}, {}
    for seed in seeds:
        pairing[str(seed)], measurements[str(seed)] = {}, {}
        for arm in arms:
            mode = "two_step" if stage == "B" and arm == "rollout_l6_l12" else "l6"
            init = {"seed": seed, "same_anchor": "fixture"}
            pairing[str(seed)][arm] = {"full_initial_state_sha256": digest([seed, "initial"]),
                                      "initialization_report_sha256": digest(init), "model_spec": model_spec,
                                      "anchor_state_sha256": digest([seed, "anchor"]), "seed": seed}
            measurements[str(seed)][arm] = {"parameters": 100, "trainable_parameters": 100,
                "forward_flops": 100 if mode == "l6" else 200, "forward_backward_flops": 220 if mode == "l6" else 440,
                "actual_model_forward_calls": 1 if mode == "l6" else 2, "physical_steps": 1 if mode == "l6" else 2,
                "objective": mode, "reasoning_steps": 4, "gradient_norm": 1., "gradient_tensors": 2,
                "nonzero_gradient_tensors": 2, "forward_scope": "actual CPU objective", "backward_scope": "actual loss.backward"}
    profile = {"scientific_claim": False, "limitations": ["fixture only"], "test_read": False, "device": "cpu",
               "no_optimizer_step": True, "full_internal_k_bptt": True, "full_physical_step_bptt": True,
               "pairing": pairing, "measurements": measurements}
    parents = {str(seed): {"original_output": str(tmp_path / f"parent{seed}"), "model_spec": model_spec}
               for seed in seeds} if stage == "B" else {}
    (output / "code.zip").write_bytes(b"synthetic unit-test archive, never deployment code")
    files = {"training/r7_v2_results.py": sha256_file(results_module.__file__)}
    (output / "code_commit.txt").write_text("fixture-read-only-head\n", encoding="utf-8")
    (output / "code_status.txt").write_text("?? fixture-only-source.py\n", encoding="utf-8")
    code = {"model_code_sha256": digest("fixture-model"), "source_tree_sha256": digest(files),
            "code_zip_sha256": sha256_file(output / "code.zip"), "files": files,
            "base_commit": "fixture-read-only-head", "working_tree_modified": True,
            "code_commit_sha256": sha256_file(output / "code_commit.txt"),
            "code_status_sha256": sha256_file(output / "code_status.txt")}
    protocol = build_protocol(stage=stage, output=output, manifests=tmp_path / "manifests", data=data,
        sources=sources, sidecar=sidecar, windows=windows, parents=parents, profile=profile, code=code,
        gpu_uuid="GPU-fixture", round_started_perf_counter=0., boot_id="fixture-boot", configuration=configuration)
    for name, value in {"protocol.json": protocol, "cpu_profile.json": profile,
                        "environment.json": {"fixture": True}, "prepare_started.json": {"fixture": True},
                        "prepare_attempt.json": {"status": "prepared-not-run"}, "run_started.json": {"fixture": True}}.items():
        write_json(output / name, value, output=output)
    return output, protocol


def training_receipt(output, protocol, job):
    arm, seed = job["arm"], job["seed"]
    folder = output / f"seed{seed}" / "training" / arm
    folder.mkdir(parents=True)
    updates = protocol["arm_configs"][arm]["updates"]
    checkpoint = folder / f"update_{updates:07d}.pt"
    checkpoint.write_bytes(f"fixture-checkpoint-{arm}-{seed}".encode())
    initialization = {"seed": seed, "same_anchor": "fixture"}
    pairing = protocol["cpu_profile"]["pairing"][str(seed)][arm]
    contract = child_contract(protocol, job, pairing["model_spec"])
    contract.update(initialization=initialization, parent_provenance=initialization if protocol["stage"] == "B" else None)
    controls, mode = protocol["shared_controls"], protocol["arm_configs"][arm]["mode"]
    contract.update(seed=seed, steps=4, total_updates=updates, mode=mode, lr=controls["lr"],
                    warmup_updates=controls["warmup"], weight_decay=controls["weight_decay"],
                    batch_size=controls["batch_size"], clip=controls["clip"], bf16=controls["bf16"],
                    checkpoint_every=controls["checkpoint_every"], output_dir=str(folder), device_type="cuda")
    contract["autoregression"].update(objective="deep_supervised_latitude_area_mse", loss_space="normalized", internal_deep_supervision=True)
    report = {"scientific_claim": False, "limitations": ["fixture only"], "test_read": False, "contract": contract,
              "signature": digest(contract), "updates_this_run": updates, "total_updates": updates, "selected_update": updates,
              "resumed_from_updates": 0, "selection_split": None, "selected_checkpoint": str(checkpoint),
              "windows": protocol["windows"], "physical_step_detach": False, "internal_k_detach": False,
              "parent_optimizer_imported": False, "elapsed_seconds": 1.,
              "losses": [{"update": index, "loss": 1.5 if mode == "two_step" else 1., "l6": 1.,
                          "l12": 1. if mode == "two_step" else None, "gradient_norm": 1.} for index in range(1, updates + 1)]}
    report_path = folder / "training_report.json"
    write_json(report_path, report, output=output)
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "training_report": str(report_path), "training_report_sha256": sha256_file(report_path),
            "contract": contract, "report": report, "signature": digest(contract), "updates_run": updates,
            "selected_update": updates, "initial_state_sha256": pairing["full_initial_state_sha256"],
            "initialization": initialization, "parent_optimizer_imported": False, "resume": False}


def complete_attempt(tmp_path, stage="B"):
    output, protocol = full_protocol(tmp_path, stage)
    (output / "workers").mkdir()
    training, execution_rows, headrooms = {}, [], []
    for index, job in enumerate(protocol["jobs"]):
        seed, arm, key = job["seed"], job["arm"], job_key(job)
        if job["phase"] == "train":
            details = training_receipt(output, protocol, job)
            training[(seed, arm)] = details
        else:
            lead, kernel = job["lead"], job["reasoning_steps"]
            count = len(protocol["data"]["evaluation_cases"][str(lead)]["cases"])
            scale = {"continue_l6": 2., "rollout_l6_l12": 1., "equal_compute_l6": 1.5,
                     "old_ours": 2., "process": 1., "matched_generic": 1.5}[arm]
            _, _, provenance, regions, flat = actual_artifacts(lead=lead, kernel=kernel, count=count, scale=scale)
            train = training[(seed, arm)]
            provenance.update(training_identity=protocol["data"]["data_identity"], checkpoint=train["checkpoint"],
                checkpoint_sha256=train["checkpoint_sha256"], checkpoint_contract_sha256=train["signature"],
                model_code_sha256=protocol["code"]["model_code_sha256"], model_kind=train["contract"]["kind"],
                model_config=train["contract"]["model"], process_scale_sidecar_identity=None,
                loop_elapsed_seconds=.5, whole_elapsed_seconds=.8,
                isolated_forward={"gpu_latency_measured": True, "reasoning_steps": kernel, "repetitions": 10,
                                  "seconds_per_batch": [.01 * kernel] * 10, "actual_reasoning_steps_per_sample": [[kernel]] * 10,
                                  "median_seconds_per_batch": .01 * kernel, "mean_seconds_per_batch": .01 * kernel,
                                  "scope": "resident-batch model K forward plus synchronization; excludes IO/transfers/metrics/validation"})
            directory = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h/k{kernel}"
            directory.mkdir(parents=True)
            csv_fixture(directory / "region_metrics.csv", regions)
            csv_fixture(directory / "per_case_metrics.csv", flat)
            from test_r7_v2_baseline_results import baseline_fixture
            baseline_regions, baseline_cases = baseline_fixture(provenance)
            csv_fixture(directory / "baseline_region_metrics.csv", baseline_regions)
            csv_fixture(directory / "baseline_per_case_metrics.csv", baseline_cases)
            write_json(directory / "provenance.json", provenance, output=output)
            details = {"evaluation_dir": str(directory), "region_metrics_csv": str(directory / "region_metrics.csv"),
                       "per_case_metrics_csv": str(directory / "per_case_metrics.csv"), "provenance": str(directory / "provenance.json"),
                       "baseline_region_metrics_csv": str(directory / "baseline_region_metrics.csv"),
                       "baseline_per_case_metrics_csv": str(directory / "baseline_per_case_metrics.csv"),
                       "artifact_sha256": {name: sha256_file(directory / name) for name in
                                          ["region_metrics.csv", "per_case_metrics.csv", "provenance.json",
                                           "baseline_region_metrics.csv", "baseline_per_case_metrics.csv"]},
                       "channels": CHANNELS, "units": UNITS, "n_evaluated": count, "n_available_windows": count,
                       "split": "val", "lead_hours": lead, "reasoning_steps": kernel,
                       "checkpoint": train["checkpoint"], "checkpoint_sha256": train["checkpoint_sha256"],
                       "evaluation_provenance": provenance}
        entry = {"status": "success", "job": job, "scientific_claim": False, "limitations": ["fixture only"], "test_read": False,
                 "protocol_sha256": protocol["protocol_sha256"], "model_code_sha256": protocol["code"]["model_code_sha256"],
                 "source_tree_sha256": protocol["code"]["source_tree_sha256"], "code_zip_sha256": protocol["code"]["code_zip_sha256"],
                 "data_identity": protocol["data"]["data_identity"], "source_sha256": protocol["sources"]["source_sha256"],
                 "sidecar_identity": protocol["sidecar"]["identity"], "windows_sha256": digest(protocol["windows"]),
                 "windows": protocol["windows"], "sources": protocol["sources"], "sidecar": protocol["sidecar"],
                 "arm_config": protocol["arm_configs"][arm], "shared_controls": protocol["shared_controls"],
                 "cpu_profile": protocol["cpu_profile"]["measurements"][str(seed)][arm],
                 "parent_provenance": protocol["parents"].get(str(seed)),
                 "baseline": {"allocated_bytes": 0, "reserved_bytes": 0},
                 "pre_init_baseline": {"allocated_bytes": 0, "reserved_bytes": 0},
                 "initialized_baseline": {"allocated_bytes": 0, "reserved_bytes": 0},
                 "peak_allocated_bytes": 64, "peak_reserved_bytes": 128, "elapsed_seconds": 1., **details}
        path = output / "workers" / f"{key}.json"
        write_json(path, entry, output=output)
        observed = 0. if index == 0 else 128 / 2 ** 20
        snapshot = {"uuid": protocol["gpu"]["uuid"], "read_only": True, "free_mib": 10000,
                    "required_free_mib": 4096, "observed_owned_peak_mib": observed, "neighbors": [{"pid": 999, "used_mib": "40"}]}
        spawn, reap = 2. + index * 3., 3. + index * 3.
        write_json(output / "workers" / f"{key}.timing.json", {"job": job, "status": "success", "returncode": 0,
                   "protocol_sha256": protocol["protocol_sha256"], "headroom": snapshot, "test_read": False,
                   "scientific_claim": False, "limitations": ["fixture only"], "cleanup": "already-exited",
                   "spawned_perf_counter": spawn, "last_owned_reap_perf_counter": reap, "ended_perf_counter": reap}, output=output)
        (output / "workers" / f"{key}.log").write_text("fixture log, no GPU run")
        execution_rows.append({"job": job, "result": str(path), "elapsed_seconds": 1., "peak_reserved_bytes": 128})
        headrooms.append({"job": job, "snapshot": snapshot})
    first, last, ended = 2., 3. + (len(protocol["jobs"]) - 1) * 3., 10. + len(protocol["jobs"]) * 3.
    execution = {"status": "results-complete", "finalized": False, "scientific_claim": False, "limitations": ["fixture only"],
                 "test_read": False, "protocol_sha256": protocol["protocol_sha256"], "stage": stage,
                 "jobs_planned": protocol["jobs"], "jobs_completed": protocol["jobs"], "jobs_results": execution_rows,
                 "headroom_checks": headrooms, "partial": False, "budget_limited": False, "owned_unreaped": False,
                 "continuous_clock": True, "failed_job_key": None, "failure_reason": None,
                 "billing_scope": protocol["gpu"]["billing"], "started_perf_counter": 0., "ended_perf_counter": ended,
                 "monotonic_boot_id": protocol["monotonic_boot_id"], "first_gpu_spawn_started_perf_counter": first,
                 "last_owned_gpu_reap_perf_counter": last, "gpu_phase_elapsed_seconds": last - first,
                 "gpu_hours_charged": (last - first) / 3600, "whole_elapsed_seconds": ended}
    return output, protocol, execution


@pytest.mark.parametrize("stage,ntrain,neval,ncells", [("B", 6, 30, 255), ("C", 9, 135, 765)])
def test_finalize_complete_real_schema_inventory_stage_not_final_cost_seal(tmp_path, stage, ntrain, neval, ncells):
    output, protocol, execution = complete_attempt(tmp_path, stage)
    before = digest(execution)
    outcome = finalize(output, protocol, execution)
    assert outcome["status"] == "stage-sealed" and outcome["finalized"] is False
    assert outcome["scientific_claim"] is False and outcome["coverage_complete"] is True
    assert outcome["whole_cost_status"] == outcome["gpu_cost_status"] == "pending-driver-final-seal"
    assert digest(execution) == before
    assert not (output / "attempt.json").exists()
    result = json.loads((output / "stage_result.json").read_text())
    assert len(result["training"]) == ntrain and len(result["evaluation"]) == neval
    assert len(result["metrics"]) == neval * 51 and len(result["aggregate"]) == 3 * ncells
    assert all(sum(counts.values()) == ncells for counts in outcome["pair_totals"].values())
    if stage == "B": assert outcome["candidate_selection"]["selected_mode"] == "two_step"
    else:
        assert outcome["adaptive_gate"]["evaluated"] is True and outcome["adaptive_gate"]["gate_met"] is False
        assert outcome["adaptive_gate"]["status"] == "not-started"
        assert outcome["adaptive_gate"]["oracle_deployable"] is False
        assert len(outcome["adaptive_gate"]["seed_lead_evidence"]) == 6
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["files_digest"] == digest(manifest["files_sha256"])
    assert not set(manifest["excluded_recursive_or_future"]) & set(manifest["files_sha256"])
    assert manifest["files_sha256"]["code_commit.txt"] == protocol["code"]["code_commit_sha256"]
    assert manifest["files_sha256"]["code_status.txt"] == protocol["code"]["code_status_sha256"]
    assert all(sha256_file(output / name) == pin for name, pin in manifest["files_sha256"].items())
    assert len(list(csv.DictReader((output / "allocator_table.csv").open()))) == ntrain + neval
    with pytest.raises(FileExistsError): finalize(output, protocol, execution)


@pytest.fixture(scope="module")
def immutable_b_receipts(tmp_path_factory):
    output, protocol, execution = complete_attempt(tmp_path_factory.mktemp("complete-b-receipts"))
    original_files = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    return output, protocol, execution, original_files


@pytest.fixture
def full_b(immutable_b_receipts):
    output, protocol, execution, original_files = immutable_b_receipts
    yield output, deepcopy(protocol), deepcopy(execution)
    for path in output.rglob("*"):
        if path.is_file() and path not in original_files:
            path.unlink()
    for path, content in original_files.items():
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


@pytest.mark.parametrize("change", ["skipped", "failed", "partial", "missing-job", "seed", "unit", "source", "protocol",
                                    "artifact-tamper", "checkpoint-tamper", "training-report-tamper", "windows",
                                    "code-commit-tamper", "code-status-tamper", "code-identity-missing"])
def test_complete_set_receipt_and_hash_guards_never_accept_partial_or_mismatch(full_b, change):
    output, protocol, execution = full_b
    job = protocol["jobs"][-1]
    path = output / "workers" / f"{job_key(job)}.json"
    entry = json.loads(path.read_text())
    if change in ("skipped", "failed", "partial"): entry["status"] = change
    elif change == "missing-job": path.rename(path.with_suffix(".missing"))
    elif change == "seed": entry["job"]["seed"] = 41
    elif change == "unit": entry["units"][0] = "C"
    elif change == "source": entry["source_sha256"] = digest("wrong-source")
    elif change == "protocol": entry["protocol_sha256"] = digest("wrong-protocol")
    elif change == "artifact-tamper": (output / f"seed{job['seed']}/evaluation/{job['arm']}/lead_072h/k4/region_metrics.csv").write_text("tampered")
    elif change == "checkpoint-tamper": (output / "seed41/training/continue_l6/update_0000200.pt").write_bytes(b"tampered")
    elif change == "training-report-tamper": (output / "seed41/training/continue_l6/training_report.json").write_text("{}")
    elif change == "windows": entry["windows"]["excluded_sample_ids"] = ["wrong"]
    elif change == "code-commit-tamper": (output / "code_commit.txt").write_text("wrong-head", encoding="utf-8")
    elif change == "code-status-tamper": (output / "code_status.txt").write_text("", encoding="utf-8")
    elif change == "code-identity-missing": (output / "code_commit.txt").rename(output / "code_commit.missing")
    if change not in ("missing-job", "artifact-tamper", "checkpoint-tamper", "training-report-tamper"):
        path.write_text(json.dumps(entry))
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize(output, protocol, execution)
    assert not (output / "stage_result.json").exists()
    assert not (output / "artifact_manifest.json").exists()


@pytest.mark.parametrize("change", ["partial", "wrong-seed", "summed-cost", "unreaped", "headroom", "missing-timing"])
def test_execution_needs_all_jobs_continuous_cost_and_every_spawn_guard(full_b, change):
    output, protocol, original = full_b
    execution = deepcopy(original)
    if change == "partial": execution["jobs_completed"] = execution["jobs_completed"][:-1]
    elif change == "wrong-seed": execution["jobs_results"][0]["job"]["seed"] = 43
    elif change == "summed-cost": execution["gpu_phase_elapsed_seconds"] = len(protocol["jobs"])
    elif change == "unreaped": execution["owned_unreaped"] = True
    elif change == "headroom": execution["headroom_checks"][0]["snapshot"]["free_mib"] = 4095
    elif change == "missing-timing":
        path = output / "workers" / f"{job_key(protocol['jobs'][-1])}.timing.json"
        path.rename(path.with_suffix(".missing"))
    with pytest.raises((ValueError, FileNotFoundError)): verify_execution(output, protocol, execution)


def test_candidate_negative_or_mixed_does_not_stop_independent_c_or_pick_other_endpoints():
    from training.r7_v2_protocol import B_REPORTING
    pairs = {}
    for baseline in ("continue_l6", "equal_compute_l6"):
        pairs[f"rollout_l6_l12 - {baseline}"] = {"cells": {f"k4|full|{lead}h|t2m": {
            "unit": "K", "seed_deltas": {"41": -.1, "42": -.2}} for lead in (6, 12)}}
    protocol = {"reporting": B_REPORTING}
    assert candidate_selection(protocol, pairs)["status"] == "supported"
    pairs["rollout_l6_l12 - continue_l6"]["cells"]["k4|full|12h|t2m"]["seed_deltas"]["42"] = 0.
    selection = candidate_selection(protocol, pairs)
    assert selection["status"] == "negative_or_mixed" and selection["selected_mode"] == "l6"
    assert selection["independent_C_continues"] is True
    assert selection["hypothesis_action"] == "terminate rollout hypothesis only"
    assert candidate_selection({"reporting": {}}, pairs)["status"] == "refused"


@pytest.mark.parametrize("work", [4, [4.], [True], [4, 4], [], [8]])
def test_actual_cumulative_k_single_lead_integer_list_has_counterexamples(work):
    protocol, job, provenance, regions, flat = actual_artifacts()
    provenance["initializations"][0]["cumulative_reasoning_steps"] = work
    with pytest.raises(ValueError, match="kernel count"):
        verify_metric_artifacts(provenance, regions, flat, protocol, job)


def test_real_evaluate_helper_schema_is_accepted_on_tmp_synthetic_store(tmp_path, monkeypatch):
    from test_r7_v2_evaluation import fixture as make_fixture, _run
    fixture = make_fixture.__wrapped__(tmp_path, monkeypatch)
    report = _run(fixture, lead=12, k=2, name="adapter-actual-helper")
    cases = [[item["init_time"], item["valid_times"]] for item in report["initializations"]]
    protocol = {"data": {"channels": report["channels"], "units": report["units"],
                          "normalization_std": fixture["root"]["normalization_std"][:].tolist(),
                          "data_identity": report["training_identity"],
                          "evaluation_cases": {"12": {"cases": cases, "n_available": report["n_available_windows"]}}},
                "sources": {"val_manifest_sha256": report["evaluation_manifest_sha256"]}}
    job = {"phase": "evaluate", "seed": 41, "arm": B_ARMS[0], "lead": 12, "reasoning_steps": 2}
    directory = tmp_path / "adapter-actual-helper"
    with (directory / "region_metrics.csv").open() as stream: regional = list(csv.DictReader(stream))
    with (directory / "per_case_metrics.csv").open() as stream: per_case = list(csv.DictReader(stream))
    verified = verify_metric_artifacts(report, regional, per_case, protocol, job)
    assert len(verified) == 51 and all(row["n_initializations"] == 21 for row in verified)
    assert all(item["cumulative_reasoning_steps"] == [4] for item in report["initializations"])
    assert any(row["bad_case_counts"]["worse_than_climatology"] for row in verified)


@pytest.mark.parametrize("marker_kind", ["regular", "broken_symlink"])
def test_authoritative_publication_failure_refuses_execution_and_finalization(full_b, marker_kind):
    output, protocol, execution = full_b
    marker = output / "publication_failure.json"
    if marker_kind == "regular":
        marker.write_text("not trusted JSON; presence is authoritative", encoding="utf-8")
    else:
        marker.symlink_to(output / "missing_failure_payload.json")
    before = marker.read_bytes() if marker.is_file() else str(marker.readlink())
    for consume in (verify_execution, finalize):
        with pytest.raises(ValueError, match="publication failure"):
            consume(output, protocol, execution)
    assert not (output / "stage_result.json").exists()
    assert (marker.read_bytes() if marker.is_file() else str(marker.readlink())) == before
    if marker.is_symlink():
        marker.unlink()
