"""CPU-only synthetic tmp_path counterproofs, not weather or scientific evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
from data.r7_autoregressive_dataset import preflight_training_windows
from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
from model.r7_halting import DECLARED_MODEL_INPUTS
from training import r7_gap_diagnostic as gap
from training.r7_autoregressive_runner import _module_semantics
from training.r7_experiment import canonical_digest, dataset_identity, make_model, model_code_digest, save_exclusive


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    original = zarr.open_group
    def opened(path, *args, **kwargs):
        root = original(path, *args, **kwargs)
        return MetadataRoot(root) if kwargs.get("mode") == "r" and Path(path).name == "store.zarr" else root
    monkeypatch.setattr(zarr, "open_group", opened)
    yield
    torch.set_num_threads(previous)


class MetadataRoot:
    """Cache only static synthetic metadata, never state; keep actual Zarr fields.

    Every open refreshes snapshots so fixture mutations remain visible. This
    avoids thousands of repeated tiny disk metadata reads in counterproofs.
    """
    def __init__(self, root):
        self.root, self.attrs = root, root.attrs
        self.arrays = {name: np.asarray(root[name][:]) for name in
            ("time_ns", "latitude", "longitude", "normalization_mean", "normalization_std")}
        self.state = root["state"]

    def __contains__(self, name):
        return name in self.root

    def __getitem__(self, name):
        if name == "state":
            return self.state
        return self.arrays[name] if name in self.arrays else self.root[name]


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _manifest_fixture(tmp_path):
    times = pd.DatetimeIndex([stamp for year in range(2017, 2023) for month in gap.MONTHS
        for stamp in pd.date_range(f"{year}-{month:02d}-01", periods=17, freq="6h")])
    paths, root = fixture_manifest(tmp_path, times=times)
    root.attrs.update(split_years={"train": gap.TRAIN_YEARS, "val": [2022], "test": [2023]},
                      normalization_years=gap.TRAIN_YEARS, scientific_claim=False,
                      units=[f"fixture_unit_{i}" for i in range(17)])
    root["normalization_mean"][:] = np.arange(17, dtype=np.float32) / 10
    root["normalization_std"][:] = np.arange(1, 18, dtype=np.float32)
    source = tmp_path / "synthetic-source.bin"
    source.write_bytes(b"opaque synthetic engineering identity; never real observations")
    root.attrs["source"] = str(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    preflight = {"schema_version": 1, "source_path": str(source), "scientific_claim": False,
                 "fingerprint": {"scope": "full-local-file", "sha256": source_hash, "bytes": source.stat().st_size}}
    (paths["train"].parent / "source_preflight.json").write_text(json.dumps(preflight))
    original = [row for row in _records(paths["train"])
                if row["history_indices"][-1] % 17 in (1, 2, 3, 4, 15)]
    for split in ("train", "val"):
        rows = [dict(row, split=split, sample_id=f"{split}-{i:04d}") for i, row in enumerate(original)
                if (pd.Timestamp(row["init_time"]).year < 2022) == (split == "train")]
        paths[split] = paths["train"].parent / f"{split}.jsonl"
        _write_records(paths[split], rows)
    return paths, root, source, source_hash


def _fixture(tmp_path):
    paths, root, source, source_hash = _manifest_fixture(tmp_path)
    for split in ("train", "val"):
        _write_records(paths[split], [row for row in _records(paths[split])
                                     if row["history_indices"][-1] % 17 in (1, 15)])
    plan = gap.metadata_case_plan(paths["train"], paths["val"])
    train_identity = dataset_identity(paths["train"])[0]
    val_identity = dataset_identity(paths["val"])[0]
    spec = dict(tiny_spec(), default_reasoning_steps=4, known_context_inputs=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = make_model("process", spec)
    semantics = canonical_digest(_module_semantics(model))
    checkpoint_paths, hashes = {}, {}
    for role, updates, mode in (("parent", 1600, "l6"), ("candidate", 200, "long_rollout")):
        windows = preflight_training_windows(paths["train"]) if role == "parent" else plan["train_windows"]
        ar = {"mode": mode, "physical_steps": 1 if role == "parent" else 12,
              "step_hours": 6, "fixed_transition_lead_hours": 6,
              "window_sha256": windows["window_sha256"], "windows": windows,
              "excluded_sample_ids": windows["excluded_sample_ids"],
              "detach_physical_steps": False, "detach_reasoning_steps": False}
        if role == "candidate":
            ar["physical_weights"] = [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5]
        contract = {"kind": "process", "model": spec, "seed": 41, "steps": 4, "mode": mode,
                    "total_updates": updates, "bf16": False, "data_identity": train_identity,
                    "source_sha256": source_hash, "model_code_sha256": model_code_digest(),
                    "model_semantics_sha256": semantics, "autoregression": ar, "lr": 2e-5, "warmup_updates": 10,
                    "protocol_sha256": canonical_digest(f"synthetic {role} training protocol"),
                    "scientific_claim": False, "test_read": False, "limitations": ["Synthetic fixture only"],
                    "initialization": {} if role == "parent" else {"parent_checkpoint_sha256": hashes["parent"]}}
        checkpoint_paths[role] = tmp_path / f"{role}.pt"
        save_exclusive(checkpoint_paths[role], {"format": "r7-local-v1", "contract": contract,
            "signature": canonical_digest(contract), "updates": updates,
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "optimizer": {"must_never_be_imported": "invalid optimizer payload"}})
        hashes[role] = hashlib.sha256(checkpoint_paths[role].read_bytes()).hexdigest()
    protocol = {"scientific_claim": False, "test_read": False,
        "limitations": ["Synthetic CPU fixture, no scientific claim"],
        "expected_contracts": {"parent": {"autoregression": {"physical_steps": 1}},
                               "candidate": {"initialization": {"parent_checkpoint_sha256": hashes["parent"]}}},
        "expected_climatology": deepcopy(plan["climatology_metadata"])}
    protocol["protocol_sha256"] = canonical_digest(protocol)
    options = {"parent_checkpoint": checkpoint_paths["parent"], "candidate_checkpoint": checkpoint_paths["candidate"],
        "parent_sha256": hashes["parent"], "candidate_sha256": hashes["candidate"],
        "expected_case_plan": plan, "expected_train_identity": train_identity,
        "expected_val_identity": val_identity, "expected_source_sha256": source_hash,
        "protocol_sha256": protocol["protocol_sha256"], "protocol": protocol}
    return paths, root, source, options


class FieldSpy:
    def __init__(self, array, reads, forbidden):
        self.array, self.reads, self.forbidden = array, reads, forbidden

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, index):
        self.reads.append(index)
        if self.forbidden:
            raise AssertionError("weather field read before qualification")
        return self.array[index]


class RootSpy:
    def __init__(self, root, reads, forbidden):
        self.root, self.attrs, self.reads, self.forbidden = root, root.attrs, reads, forbidden

    def __contains__(self, name):
        return name in self.root

    def __getitem__(self, name):
        value = self.root[name]
        return FieldSpy(value, self.reads, self.forbidden) if name in ("state", "process_diagnostics_raw") else value


def _spy(monkeypatch, root, *, forbidden=True):
    original, reads = zarr.open_group, []
    def opened(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return RootSpy(value, reads, forbidden) if Path(path).name == "store.zarr" and kwargs.get("mode") == "r" else value
    monkeypatch.setattr(zarr, "open_group", opened)
    return reads


def _run(paths, options):
    return gap.run_gap_diagnostic(paths["train"], paths["val"], **options)


def test_metadata_no_fields_lower_median_exact_pins_digest_and_counts(tmp_path, monkeypatch):
    paths, root, _, _ = _manifest_fixture(tmp_path)
    reads = _spy(monkeypatch, root)
    plan = gap.metadata_case_plan(paths["train"], paths["val"])
    assert reads == [] and plan == gap.metadata_case_plan(paths["train"], paths["val"])
    assert plan["n_cases"] == 24 and [c["split"] for c in plan["cases"]].count("train") == 20
    assert all(plan[key] is False for key in ("scientific_claim", "test_read", "state_fields_read"))
    assert plan["selection_sha256"] == canonical_digest({k: v for k, v in plan.items() if k != "selection_sha256"})
    raw = np.asarray(root["time_ns"][:])
    for case in plan["cases"]:
        assert case["group_complete_windows"] == 4 and case["selection_rank"] == 1
        assert pd.Timestamp(case["init_time"]).hour == 12
        assert case["group_label"] == gap.LABELS[case["split"]]
        record = _records(paths[case["split"]])[case["manifest_index"]]
        assert case["record_sha256"] == canonical_digest(record) and case["sample_id"] == record["sample_id"]
        assert case["history_indices"] == record["history_indices"] and case["history_times"] == record["history_times"]
        init = pd.Timestamp(case["init_time"]).value
        assert [int(raw[i]) - init for i in case["history_indices"]] == [-6 * gap.HOUR_NS, 0]
        assert [int(raw[i]) - init for i in case["target_indices"]] == [6 * step * gap.HOUR_NS for step in range(1, 13)]
        assert case["valid_times"] == [case["target_times"][i] for i in gap.TARGET_POSITIONS]
        assert case["history_offsets_hours"] == [-6., 0.] and case["init_calendar_year"] == case["year"]
    assert plan["climatology_metadata"]["n_selected_steps"] == 340
    assert len(plan["climatology_metadata"]["bucket_counts"]) == 16
    rows = _records(paths["val"])
    _write_records(paths["val"], [dict(row, sample_id=row["sample_id"] + "-renamed") for row in rows])
    assert gap.metadata_case_plan(paths["train"], paths["val"])["selection_sha256"] != plan["selection_sha256"]
    assert reads == []


@pytest.mark.parametrize("fault", ["marker", "store_status", "wrong_years", "wrong_month", "split", "time",
    "cadence", "history", "lead", "units", "duplicate_id", "duplicate_init", "reordered", "missing_group", "other_store"])
def test_bad_metadata_fails_without_state_reads(tmp_path, monkeypatch, fault):
    paths, root, _, _ = _manifest_fixture(tmp_path)
    rows = _records(paths["train"])
    if fault == "marker":
        (paths["train"].parent / "BUILD_COMPLETE.json").write_text('{"schema_version":1,"build_complete":false}')
    elif fault == "store_status":
        root.attrs["build_complete"] = False
    elif fault == "wrong_years":
        root.attrs["split_years"] = {"train": [2017], "val": [2022], "test": [2023]}
        root.attrs["normalization_years"] = [2017]
    elif fault == "wrong_month":
        raw = np.asarray(root["time_ns"][:])
        raw[:17] += 31 * 24 * gap.HOUR_NS
        root["time_ns"][:] = raw
        for row in rows:
            if row["history_indices"][0] < 17:
                row["history_times"] = [pd.Timestamp(int(raw[i])).isoformat() for i in row["history_indices"]]
                row["init_time"] = row["history_times"][-1]
                row["target_time"] = pd.Timestamp(int(raw[row["target_index"]])).isoformat()
    elif fault == "split":
        rows[0]["split"] = "val"
    elif fault == "time":
        rows[0]["target_time"] = rows[1]["target_time"]
    elif fault == "cadence":
        rows[1]["history_indices"][0] = 0
        rows[1]["history_times"][0] = rows[0]["history_times"][0]
    elif fault == "history":
        rows[0]["history_indices"] = rows[0]["history_indices"][-1:]
        rows[0]["history_times"] = rows[0]["history_times"][-1:]
    elif fault == "lead":
        rows[0]["lead_time_hours"] = 12
    elif fault == "units":
        root.attrs["units"] = ["unknown"] * 17
    elif fault == "duplicate_id":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif fault == "duplicate_init":
        rows[1] = dict(rows[0], sample_id="distinct-id-same-init")
    elif fault == "reordered":
        rows[:2] = list(reversed(rows[:2]))
    elif fault == "missing_group":
        rows = [row for row in rows if pd.Timestamp(row["init_time"]).month != 4]
    else:
        rows[1]["store_path"] = "../another-store.zarr"
    _write_records(paths["train"], rows)
    reads = _spy(monkeypatch, root)
    with pytest.raises(ValueError):
        gap.metadata_case_plan(paths["train"], paths["val"])
    assert reads == []


@pytest.mark.parametrize("split", ["train", "val"])
def test_missing_unscored_intermediate_cannot_use_terminal_or_adjacent_index(tmp_path, monkeypatch, split):
    paths, root, _, _ = _manifest_fixture(tmp_path)
    raw = np.asarray(root["time_ns"][:])
    bad = 7 if split == "train" else 5 * 4 * 17 + 7
    terminal = int(raw[bad + 7])
    raw[bad] += gap.HOUR_NS
    root["time_ns"][:] = raw
    rows = [row for row in _records(paths[split]) if bad not in row["history_indices"] + [row["target_index"]]]
    _write_records(paths[split], rows)
    reads = _spy(monkeypatch, root)
    with pytest.raises(ValueError, match="no complete"):
        gap.metadata_case_plan(paths["train"], paths["val"])
    assert terminal in set(raw.tolist()) and reads == []


def test_cross_split_target_is_never_bridged(tmp_path, monkeypatch):
    paths, root, _, _ = _manifest_fixture(tmp_path)
    root.attrs.update(split_mode="time_ranges", split_time_ranges={
        "train": [["2017-01-01", "2017-01-02T18:00:00"], ["2017-01-03", "2022-01-01"]],
        "val": [["2022-01-01", "2023-01-01"]], "test": [["2023-01-01", "2024-01-01"]]})
    rows = [row for row in _records(paths["train"])
            if all(not (pd.Timestamp("2017-01-02T18") <= pd.Timestamp(t) < pd.Timestamp("2017-01-03"))
                   for t in row["history_times"] + [row["target_time"]])]
    _write_records(paths["train"], rows)
    reads = _spy(monkeypatch, root)
    with pytest.raises(ValueError, match="no complete"):
        gap.metadata_case_plan(paths["train"], paths["val"])
    assert reads == []


@pytest.mark.parametrize("names", [("test.jsonl", "val.jsonl"), ("train.jsonl", "test.jsonl"), ("renamed.jsonl", "val.jsonl")])
def test_sealed_or_alternate_manifest_rejected_before_any_read(tmp_path, monkeypatch, names):
    monkeypatch.setattr(gap.ZarrAtmosWindowDataset, "__init__", lambda *_: pytest.fail("sealed reader invoked"))
    with pytest.raises(ValueError, match="explicit"):
        gap.metadata_case_plan(tmp_path / names[0], tmp_path / names[1])
    (tmp_path / "val.jsonl").symlink_to(tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="nonsymlink"):
        gap.metadata_case_plan(tmp_path / "train.jsonl", tmp_path / "val.jsonl")


@pytest.mark.parametrize("fault", ["case_order", "case_duplicate", "case_time", "case_hash", "case_count", "train_identity",
    "val_identity", "source_hash", "source_bytes", "source_root", "source_preflight", "checkpoint_hash", "model_code",
    "signature", "seed", "endpoint", "mode", "data", "source", "semantics", "bf16", "state_fp16", "state_nan",
    "state_shape", "state_missing", "different_spec", "window", "weights", "lr", "protocol", "climate_counts"])
def test_identity_contract_or_frozen_cases_refused_before_fields(tmp_path, monkeypatch, fault):
    paths, root, source, options = _fixture(tmp_path)
    path = options["candidate_checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    contract, altered = saved["contract"], False
    if fault.startswith("case_"):
        plan = options["expected_case_plan"]
        if fault == "case_order":
            plan["cases"][:2] = list(reversed(plan["cases"][:2]))
        elif fault == "case_duplicate":
            plan["cases"][1] = deepcopy(plan["cases"][0])
        elif fault == "case_time":
            plan["cases"][0]["target_times"][2] = "2017-01-09T00:00:00"
        elif fault == "case_hash":
            plan["cases"][0]["record_sha256"] = "0" * 64
        else:
            plan["cases"].pop()
        plan["selection_sha256"] = canonical_digest({k: v for k, v in plan.items() if k != "selection_sha256"})
    elif fault in ("train_identity", "val_identity", "source_hash", "checkpoint_hash"):
        key = {"train_identity": "expected_train_identity", "val_identity": "expected_val_identity",
               "source_hash": "expected_source_sha256", "checkpoint_hash": "candidate_sha256"}[fault]
        options[key] = "0" * 64
    elif fault == "source_bytes":
        source.write_bytes(source.read_bytes() + b"tampered")
    elif fault == "source_root":
        other = tmp_path / "different-source.bin"
        other.write_bytes(source.read_bytes())
        root.attrs["source"] = str(other)
        options["expected_case_plan"] = gap.metadata_case_plan(paths["train"], paths["val"])
        options["expected_train_identity"] = dataset_identity(paths["train"])[0]
        options["expected_val_identity"] = dataset_identity(paths["val"])[0]
    elif fault == "source_preflight":
        preflight = paths["train"].parent / "source_preflight.json"
        report = json.loads(preflight.read_text())
        report["fingerprint"]["scope"] = "partial-source-not-allowed"
        preflight.write_text(json.dumps(report))
    elif fault in ("protocol", "climate_counts"):
        if fault == "protocol":
            options["protocol"]["scientific_claim"] = True
        else:
            options["protocol"]["expected_climatology"]["n_selected_steps"] = 339
            options["protocol"].pop("protocol_sha256")
            options["protocol_sha256"] = canonical_digest(options["protocol"])
            options["protocol"]["protocol_sha256"] = options["protocol_sha256"]
    else:
        altered = True
        if fault == "model_code":
            saved["model_code_sha256"] = "0" * 64
        elif fault == "signature":
            saved["signature"] = "0" * 64
        elif fault == "seed":
            contract["seed"] = 42
        elif fault == "endpoint":
            saved["updates"] = 199
        elif fault == "mode":
            contract["mode"] = "two_step"
        elif fault in ("data", "source", "semantics"):
            contract[{"data": "data_identity", "source": "source_sha256", "semantics": "model_semantics_sha256"}[fault]] = "0" * 64
        elif fault == "bf16":
            contract["bf16"] = True
        elif fault == "different_spec":
            contract["model"]["default_lead_hours"] = 12.
            model = make_model("process", contract["model"])
            contract["model_semantics_sha256"] = canonical_digest(_module_semantics(model))
        elif fault == "window":
            contract["autoregression"]["window_sha256"] = "0" * 64
        elif fault == "weights":
            contract["autoregression"]["physical_weights"][-1] = 0.
        elif fault == "lr":
            contract["lr"] = 1e-4
        else:
            key = next(k for k, v in saved["model"].items() if v.is_floating_point() and v.numel() > 1)
            if fault == "state_fp16":
                saved["model"][key] = saved["model"][key].half()
            elif fault == "state_nan":
                saved["model"][key].reshape(-1)[0] = float("nan")
            elif fault == "state_shape":
                saved["model"][key] = saved["model"][key].reshape(-1)[:1]
            else:
                saved["model"].pop(key)
        if fault != "signature":
            saved["signature"] = canonical_digest(contract)
    if altered:
        torch.save(saved, path)
        options["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    reads = _spy(monkeypatch, root)
    with pytest.raises((ValueError, RuntimeError)):
        _run(paths, options)
    assert reads == []


def test_declared_climatology_mean_digest_and_default_counts_fail_closed(tmp_path, monkeypatch):
    paths, root, _, options = _fixture(tmp_path)
    original_open = zarr.open_group
    reads = _spy(monkeypatch, root)
    without_protocol = {k: v for k, v in options.items() if k != "protocol"}
    with pytest.raises(ValueError, match="climatology metadata"):
        _run(paths, without_protocol)
    assert reads == []
    monkeypatch.setattr(zarr, "open_group", original_open)
    options["protocol"]["expected_climatology"]["mean_identity_sha256"] = "0" * 64
    options["protocol"].pop("protocol_sha256")
    options["protocol_sha256"] = canonical_digest(options["protocol"])
    options["protocol"]["protocol_sha256"] = options["protocol_sha256"]
    seen = _spy(monkeypatch, root, forbidden=False)
    monkeypatch.setattr(gap, "predict_case", lambda *_a, **_k: pytest.fail("forecast before mean identity"))
    with pytest.raises(ValueError, match="mean_identity_sha256"):
        _run(paths, options)
    assert len(seen) == 340


def test_full_tiny_process_same24_cases_once_train_only_climate_no_optimizer_or_test(tmp_path, monkeypatch):
    paths, root, _, options = _fixture(tmp_path)
    (paths["train"].parent / "test.jsonl").write_text("SEALED INVALID TEST MANIFEST MUST NOT BE READ")
    reads = _spy(monkeypatch, root, forbidden=False)
    fitting, restored = [], []
    original_fit, original_make = gap.fit_training_climatology, gap.make_model
    def fit(store):
        fitting.append(store)
        return original_fit(store)
    def make(kind, spec):
        model = original_make(kind, spec)
        restored.append(model)
        return model
    monkeypatch.setattr(gap, "fit_training_climatology", fit)
    monkeypatch.setattr(gap, "make_model", make)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *_a, **_k: pytest.fail("optimizer created"))
    monkeypatch.setattr(torch, "autocast", lambda *_a, **_k: pytest.fail("autocast entered"))
    original_open = zarr.open_group
    result = _run(paths, options)
    assert zarr.open_group is original_open and len(fitting) == 1 and len(restored) == 2
    assert len(reads) == 340 + 20 * 14 + 4 * 7
    assert result["n_evaluated"] == len(result["cases"]) == 24 and result["lead_hours"] == list(gap.LEADS)
    assert result["scientific_claim"] is result["test_read"] is result["optimizer_imported"] is False
    assert result["optimizer_updates"] == 0 and result["diagnostic_training_mode"] is False and result["limitations"]
    assert "verdict" not in result and "decision" not in result and "test" not in result["aggregates"]["split"]
    assert result["source_sha256"] == options["expected_source_sha256"]
    assert result["protocol_sha256"] == options["protocol_sha256"]
    assert result["selection_sha256"] == options["expected_case_plan"]["selection_sha256"]
    assert result["source_identity"]["scope"] == "full-local-file" and result["bf16"] is False
    assert result["climatology"]["training_years"] == gap.TRAIN_YEARS
    assert result["climatology"]["n_selected_steps"] == 340 and len(result["climatology"]["means"]) == 16
    assert result["climatology"]["mean_identity_sha256"] == canonical_digest(result["climatology"]["means"])
    assert result["climatology"]["bucket_counts"] == options["expected_case_plan"]["climatology_metadata"]["bucket_counts"]
    times = pd.DatetimeIndex(np.asarray(root["time_ns"][:]).astype("datetime64[ns]"))
    for (month, hour), selected in {(m, h): [i for i, t in enumerate(times)
            if t.year in gap.TRAIN_YEARS and (t.month, t.hour) == (m, h)]
            for m in gap.MONTHS for h in (0, 6, 12, 18)}.items():
        mean = np.mean([np.asarray(root["state"][i], dtype=np.float64) for i in selected], axis=0)
        digest = result["climatology"]["means"][f"{month:02d}-{hour:02d}"]["sha256"]
        online = None
        for count, index in enumerate(selected, start=1):
            field = np.asarray(root["state"][index], dtype=np.float64)
            online = field.copy() if count == 1 else online + (field - online) / count
        np.testing.assert_allclose(online, mean, rtol=1e-13, atol=1e-13)
        assert hashlib.sha256(online.tobytes()).hexdigest() == digest
    for role, model in zip(("parent", "candidate"), restored):
        assert all(not m.training for m in model.modules()) and not model._forward_pre_hooks
        assert all(p.dtype == torch.float32 and p.grad is None for p in model.parameters())
        assert result["checkpoints"][role]["sha256"] == options[role + "_sha256"]
        assert result["checkpoints"][role]["optimizer_imported"] is False
        saved = torch.load(options[role + "_checkpoint"], map_location="cpu", weights_only=True)
        assert all(torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items())
    for actual, frozen in zip(result["cases"], options["expected_case_plan"]["cases"]):
        assert all(actual[k] == v for k, v in frozen.items())
        for key in ("parent", "candidate", "climatology"):
            assert np.asarray(actual["metrics"][key]["mse"]).shape == (5, 17)
        assert actual["metrics"]["parent"] == actual["metrics"]["candidate"]
    assert result["aggregates"]["split"]["train"]["n_cases"] == 20
    assert result["aggregates"]["split"]["val"]["n_cases"] == 4
    assert set(result["aggregates"]["train_year"]) == {str(y) for y in gap.TRAIN_YEARS}
    assert all(row["n_cases"] == 5 for row in result["aggregates"]["train_month"].values())
    assert np.asarray(result["train_val_gap"]["parent"]["val_minus_train_mse"]).shape == (5, 17)
    assert result["owned_cuda_allocated_peak_bytes"] == result["owned_cuda_reserved_peak_bytes"] == 0
    assert json.loads(json.dumps(result, allow_nan=False))["n_evaluated"] == 24
    assert {p.name for p in tmp_path.iterdir()} == {"store.zarr", "manifests", "synthetic-source.bin", "parent.pt", "candidate.pt"}


def test_target_poisoning_actual_graph_rollout_calendar_offsets_and_whitelist(tmp_path):
    paths, _, _, options = _fixture(tmp_path)
    plan = options["expected_case_plan"]
    ds = ZarrLongRolloutDataset(paths["train"], expected_exclusions=plan["train_windows"]["excluded_sample_ids"],
                               expected_window_sha256=plan["train_windows"]["window_sha256"])
    sample = ds[0]
    spec = dict(tiny_spec(), default_reasoning_steps=4, known_context_inputs=True)
    model = make_model("process", spec).eval()
    inputs, forecasts = [], []
    def observe(_model, args, output):
        inputs.append({k: v.detach().clone() for k, v in args[0].items()})
        forecasts.append(output.forecast.detach().clone())
    hook = model.register_forward_hook(observe)
    clean = gap.predict_case(model, sample)
    poison = deepcopy(sample)
    for key in ("physical_targets", "atmos_target", "future_target"):
        poison[key].fill_(float("nan"))
    poison.update(climatology=torch.full_like(sample["physical_targets"], 12345.), process_targets=torch.tensor([9999.]),
                  rollout_targets=torch.full_like(sample["physical_targets"], -9999.), init_year=torch.tensor(1234.))
    poisoned = gap.predict_case(model, poison)
    hook.remove()
    torch.testing.assert_close(clean, poisoned, rtol=0, atol=0)
    assert len(inputs) == 24 and not model._forward_pre_hooks
    for step, batch in enumerate(inputs[:12]):
        stamp = pd.Timestamp(sample["init_time"]) + pd.Timedelta(hours=6 * step)
        assert set(batch) == set(DECLARED_MODEL_INPUTS) and batch["lead_time_hours"].tolist() == [6.]
        assert batch["init_calendar_year"].item() == stamp.year and batch["init_day_of_year"].item() == stamp.dayofyear
        assert batch["init_utc_hour"].item() == stamp.hour
        assert batch["history_offsets_hours"].tolist() == [[-6., 0.]]
        if step:
            assert torch.equal(batch["coarse_history"][:, -1], forecasts[step - 1])
    assert clean.dtype == torch.float32 and clean.shape == (1, 5, 17, 3, 4)


def test_analytic_physical_std_latitude_weight_all_variables_and_equal_case_aggregation():
    std = np.arange(1, 18, dtype=np.float32)
    prediction = torch.ones(1, 5, 17, 2, 2)
    prediction[..., 1, :] = 3
    mse = gap.physical_mse(prediction, torch.zeros_like(prediction), torch.tensor([0., 60.]),
        training_std=std, variables=[f"v{i}" for i in range(17)], units=[f"u{i}" for i in range(17)])
    expected = np.broadcast_to(std.astype(np.float64) ** 2 * (1 + .5 * 9) / 1.5, (5, 17))
    np.testing.assert_allclose(mse.numpy(), expected, rtol=1e-14)
    cases = []
    for i, factor in enumerate((1., 9.)):
        cases.append({"split": "train", "year": 2017, "month": 1, "sample_id": str(i),
                      **gap.metric_summary({"parent": expected * factor, "candidate": expected * factor / 2,
                                            "climatology": expected * 4})})
    pooled = gap.aggregate_cases(cases)["split"]["train"]
    np.testing.assert_allclose(pooled["metrics"]["parent"]["rmse"], np.sqrt(expected * 5))
    assert np.asarray(pooled["metrics"]["parent"]["rmse"]).shape == (5, 17)
    assert not np.allclose(pooled["metrics"]["parent"]["rmse"], np.sqrt(expected) * 2)
    assert pooled["metrics"]["parent"]["mse_skill"][0][0] == pytest.approx(-.25)


def test_zero_denominators_are_explicit_null_without_epsilon_and_bad_mse_refused():
    zero, ones = np.zeros((5, 17)), np.ones((5, 17))
    result = gap.metric_summary({"parent": zero, "candidate": ones, "climatology": zero})
    assert result["metrics"]["parent"]["mse_skill"][0] == [None] * 17
    assert result["metrics"]["candidate"]["mse_skill_status"][0] == ["undefined_zero_climatology_mse"] * 17
    assert result["pair_delta"]["relative_mse_change"][0] == [None] * 17
    assert result["pair_delta"]["relative_mse_change_status"][0] == ["undefined_zero_parent_mse"] * 17
    for bad in (np.full((5, 17), np.nan), np.full((5, 17), -1), np.ones((5, 16))):
        with pytest.raises(ValueError, match="per-variable MSE"):
            gap.metric_summary({"parent": bad, "candidate": ones, "climatology": ones})
    assert json.loads(json.dumps(result, allow_nan=False))["pair_delta"]["relative_mse_change"][0][0] is None


def test_deadline_before_open_before_state_and_guard_restoration(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="deadline"):
        gap.run_gap_diagnostic(tmp_path / "train.jsonl", tmp_path / "val.jsonl", parent_checkpoint="unused",
            candidate_checkpoint="unused", parent_sha256="unused", candidate_sha256="unused", expected_case_plan={},
            expected_train_identity="unused", expected_val_identity="unused", expected_source_sha256="unused",
            protocol_sha256="unused", deadline=time.perf_counter() - 1)
    paths, _, _, options = _fixture(tmp_path)
    plan = options["expected_case_plan"]
    ds, _ = gap._case_datasets(plan, paths["train"], paths["val"])
    original, cached = zarr.open_group, ds.reader._stores
    with pytest.raises(RuntimeError, match="deadline"):
        with gap._guard_reads(Path(plan["store"]), ds.reader, time.perf_counter() - 1):
            ds[0]
    assert zarr.open_group is original and ds.reader._stores is cached
    monkeypatch.setattr(gap, "fit_training_climatology", lambda *_: (_ for _ in ()).throw(RuntimeError("injected fit failure")))
    with pytest.raises(RuntimeError, match="injected"):
        _run(paths, options)
    assert zarr.open_group is original


def test_forward_transition_deadline_restores_hooks_and_rejects_train_or_autocast(monkeypatch):
    model = make_model("process", dict(tiny_spec(), default_reasoning_steps=4, known_context_inputs=True)).eval()
    sample = {"coarse_history": torch.ones(2, 17, 3, 4), "latitude": torch.tensor([40., 39.75, 39.5]),
              "longitude": torch.tensor([115., 115.25, 115.5, 115.75]), "init_calendar_year": torch.tensor(2017.),
              "init_day_of_year": torch.tensor(1.), "init_utc_hour": torch.tensor(6.),
              "history_offsets_hours": torch.tensor([-6., 0.])}
    clock, calls = [0.], []
    monkeypatch.setattr(gap.time, "perf_counter", lambda: clock[0])
    def expire(_model, _args, _output):
        calls.append(1)
        clock[0] = 11.
    hook = model.register_forward_hook(expire)
    with pytest.raises(RuntimeError, match="deadline"):
        gap.predict_case(model, sample, deadline=10.)
    hook.remove()
    assert len(calls) == 1 and not model._forward_pre_hooks
    with pytest.raises(ValueError, match="eval FP32"):
        gap.predict_case(model.train(), sample)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="eval FP32"):
            gap.predict_case(model.eval(), sample)
