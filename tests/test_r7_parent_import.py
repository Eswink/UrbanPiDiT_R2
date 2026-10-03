"""Tmp-only synthetic identity fixtures: never open real data or old outputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from data.preprocess.r7_process_scale_sidecar import (
    BUILD_COMPLETE_BYTES, PUBLICATION_CONTRACT, fit_process_scale,
)
from training import r7_experiment as experiment
from training import r7_parent_import as parent_import
from training.r7_arm_harness import sha256_file
from training.r7_m3_identity import source_identity
from training.r7_m3_protocol import expected_training_contract

MODEL_SPEC = {"in_channels": 17, "out_channels": 17, "history_steps": 2,
              "architecture": "window", "dim": 8, "depth": 1, "heads": 2,
              "patch_size": 2, "window_size": 2, "dropout": 0.0,
              "anchored_processes": 8, "free_processes": 8,
              "use_forecast_feedback": True, "spacetime_inputs": True,
              "positional_process_readout": True, "default_reasoning_steps": 4}
ARCHIVED_PROCESS = b'''raise RuntimeError("ARCHIVE MUST NOT EXECUTE")
class ProcessForecastCoReasoner:
    def __init__(self, in_channels, out_channels=None, history_steps=2,
                 dim=8, depth=1, heads=2, patch_size=2, window_size=2,
                 dropout=0.0, anchored_processes=8, free_processes=8,
                 use_forecast_feedback=True, spacetime_inputs=True,
                 positional_process_readout=True, default_reasoning_steps=4,
                 local_solver_state=False, detach_between_steps=False):
        raise RuntimeError("ARCHIVED CONSTRUCTOR MUST NOT EXECUTE")
'''


class MetadataArray:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        raise AssertionError("parent import must never read weather arrays")


class MetadataRoot(dict):
    def __init__(self, source, channels):
        super().__init__(state=MetadataArray((3, 17, 8, 8)),
                         normalization_mean=np.arange(17, dtype=np.float32),
                         normalization_std=np.arange(17, dtype=np.float32) + 1)
        self.attrs = {"source": str(source), "channels": channels}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _protocol(bundle):
    body = {key: value for key, value in bundle.protocol.items() if key != "protocol_sha256"}
    bundle.protocol["protocol_sha256"] = experiment.canonical_digest(body)
    bundle.pins["protocol_sha256"] = bundle.protocol["protocol_sha256"]
    _write_json(bundle.paths["original_protocol"], bundle.protocol)


def _save(bundle, *, resign=False, accept_signature=False):
    if resign:
        bundle.saved["signature"] = experiment.canonical_digest(bundle.saved["contract"])
    torch.save(bundle.saved, bundle.paths["checkpoint"])
    bundle.parents[41]["checkpoint_sha256"] = sha256_file(bundle.paths["checkpoint"])
    if accept_signature:
        bundle.parents[41]["signature"] = bundle.saved["signature"]


def _members(bundle, values, *, update_map=True, update_model=True):
    with zipfile.ZipFile(bundle.paths["codezip"], "w") as archive:
        for name, value in values.items():
            archive.writestr(name, value)
    code = bundle.protocol["code"]
    code["code_zip_sha256"] = sha256_file(bundle.paths["codezip"])
    bundle.pins["code_zip_sha256"] = code["code_zip_sha256"]
    if update_map:
        code["files"] = {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}
        code["source_tree_sha256"] = experiment.canonical_digest(code["files"])
    if update_model:
        digest = hashlib.sha256()
        for name, value in sorted(values.items()):
            if name.startswith("model/"):
                digest.update(name[len("model/"):].encode() + b"\0")
                digest.update(value)
        code["model_code_sha256"] = digest.hexdigest()
        bundle.pins["model_code_sha256"] = code["model_code_sha256"]
    _protocol(bundle)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "inputs" / "manifests"
    scale_dir = tmp_path / "scale"
    manifest_dir.mkdir(parents=True)
    scale_dir.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"opaque synthetic unit-test bytes, not real meteorological data")
    train = manifest_dir / "train.jsonl"
    records = [{"split": "train", "sample_id": "fixture-0", "store_path": "../../cache.zarr",
                "history_indices": [0, 1], "target_index": 2, "lead_time_hours": 6}]
    train.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    (manifest_dir / "val.jsonl").write_text("synthetic val metadata hash only\n", encoding="utf-8")
    (manifest_dir / "BUILD_COMPLETE.json").write_bytes(BUILD_COMPLETE_BYTES)
    (scale_dir / "BUILD_COMPLETE.json").write_bytes(BUILD_COMPLETE_BYTES)
    preflight = {"schema_version": 1, "source_path": str(source),
                 "fingerprint": {"scope": "full-local-file", "sha256": sha256_file(source),
                                 "bytes": source.stat().st_size}}
    _write_json(manifest_dir / "source_preflight.json", preflight)
    # source_identity's established receipt placement is entirely under tmp_path.
    _write_json(tmp_path / "source_receipt.json", {
        "local_artifact": {"sha256": sha256_file(source)}, "synthetic_fallback": False})
    channels = [f"fixture_channel_{index}" for index in range(17)]
    root = MetadataRoot(source, channels)
    class Reader:
        def __init__(self):
            self.manifest = train
            self.records = records
        def __len__(self):
            return len(self.records)
        def _store(self, record):
            return root
    reader = Reader()
    identity = experiment.canonical_digest({"synthetic_fixture": True, "channels": channels,
                                             "manifest_sha256": sha256_file(train)})
    monkeypatch.setattr(experiment, "dataset_identity", lambda path: (identity, reader))
    sources = source_identity(manifest_dir, root)
    metadata = fit_process_scale(np.arange(24, dtype=np.float64).reshape(3, 8))
    metadata.update(store=str(tmp_path / "cache.zarr"), train_manifest=str(train),
                    data_identity=identity, train_manifest_sha256=sha256_file(train),
                    train_diagnostics_sha256=hashlib.sha256(b"fixture diagnostics").hexdigest(),
                    train_diagnostics_dtype="<f8", train_sample_ids=["fixture-0"],
                    train_frame_indices=[0, 1, 2], train_time_ns=[0, 6 * 3600 * 10**9, 12 * 3600 * 10**9],
                    fit_frame_selection="train-manifest-history-target-union",
                    source_identity={"path": str(source), **preflight["fingerprint"],
                                     "source_preflight_sha256": sources["preflight_report_sha256"]},
                    publication_contract=dict(PUBLICATION_CONTRACT))
    preflight_payload = {"schema": "r7-process-scale-preflight-v1", "schema_version": 1,
                         "mode": "read-only-preflight", "scientific_claim": False, "metadata": deepcopy(metadata)}
    metadata["preflight_identity"] = experiment.canonical_digest(preflight_payload)
    metadata["sidecar_identity"] = experiment.canonical_digest(metadata)
    sidecar = scale_dir / "scale_metadata.json"
    _write_json(sidecar, metadata)
    sidecar_pins = {"path": str(sidecar), "identity": metadata["sidecar_identity"],
                    "metadata_sha256": sha256_file(sidecar),
                    "publication_marker_sha256": sha256_file(scale_dir / "BUILD_COMPLETE.json"),
                    "preflight_identity": metadata["preflight_identity"]}
    protocol = {"format": "r7-73-process-supervision-protocol-v1", "scientific_claim": False,
                "frozen_before_any_step": True, "test_read": False, "manifests": str(manifest_dir),
                "code": {}, "sidecar": sidecar_pins, "sources": sources,
                "data": {"data_identity": identity, "channels": channels,
                         "normalization_mean": root["normalization_mean"].tolist(),
                         "normalization_std": root["normalization_std"].tolist(),
                         "store": metadata["store"], "train_windows": 1, "shape": list(root["state"].shape)},
                "arms": [{"name": "aux_off", "kind": "process", "model_config": deepcopy(MODEL_SPEC),
                          "supervision_weights": {key: 0.0 for key in parent_import.DIAGNOSTIC_WEIGHTS}}],
                "shared_controls": {"updates": 400}}
    paths = {"checkpoint": tmp_path / "parent.pt", "original_protocol": tmp_path / "protocol.json",
             "codezip": tmp_path / "code.zip", "sidecar": sidecar, "trainmanifest": train}
    bundle = SimpleNamespace(paths=paths, protocol=protocol, source=source, root=root, reader=reader,
                             metadata=metadata, pins={"data_identity": identity,
                             "source_sha256": sha256_file(source), "sidecar_identity": metadata["sidecar_identity"]},
                             parents={41: {}}, saved={})
    _members(bundle, {"model/__init__.py": b'raise RuntimeError("DO NOT IMPORT ZIP")\n',
                      "model/process_forecast_r7.py": ARCHIVED_PROCESS,
                      "training/never_execute.py": b'raise RuntimeError("DO NOT RUN ZIP")\n'})
    contract = {"kind": "process", "model": deepcopy(MODEL_SPEC), "data_identity": identity,
                "steps": 4, "seed": 41, "process_weight": 0.0, "dataset_length": 1,
                "process_supervision": expected_training_contract(protocol, "aux_off")}
    model = experiment.make_model("process", MODEL_SPEC)
    bundle.saved = {"format": "r7-local-v1", "contract": contract,
                    "signature": experiment.canonical_digest(contract),
                    "model_code_sha256": bundle.pins["model_code_sha256"],
                    "model": {key: value.clone() for key, value in model.state_dict().items()},
                    "updates": 400, "epoch": 8, "cursor": 1,
                    "optimizer": {"state": {"sentinel": "NEVER RESTORE"}}, "rng": {"sentinel": "NEVER RESTORE"}}
    _save(bundle, accept_signature=True)
    monkeypatch.setattr(parent_import, "M3_PINS", bundle.pins)
    monkeypatch.setattr(parent_import, "M3_PARENTS", bundle.parents)
    return bundle


def test_explicit_import_copies_every_key_and_keeps_parent_identity(artifacts, monkeypatch):
    before = {path: sha256_file(path) for path in artifacts.paths.values()}
    previous_contract = deepcopy(artifacts.saved["contract"])
    def forbidden(*args, **kwargs):
        raise AssertionError("old loader/RNG/optimizer/archive execution must not be used")
    monkeypatch.setattr(experiment, "load_checkpoint", forbidden)
    monkeypatch.setattr(experiment, "restore_rng", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    monkeypatch.setattr(torch, "load", _weights_only_load(torch.load))
    model, report = parent_import.import_parent(**artifacts.paths)
    assert {path: sha256_file(path) for path in artifacts.paths.values()} == before
    assert report["parent"]["contract"] == previous_contract
    assert report["parent"]["signature"] == experiment.canonical_digest(previous_contract)
    assert report["parent"]["model_code_sha256"] == report["archive"]["model_code_sha256"]
    assert report["initialization_contract"]["model_code_sha256"] == experiment.model_code_digest()
    assert report["parent"]["model_code_sha256"] != report["initialization_contract"]["model_code_sha256"]
    assert report["unused_source_keys"] == report["uninitialized_target_keys"] == []
    assert report["optimizer_reset"] and not report["rng_restored"] and not report["resume"]
    assert report["child_counters"] == {"updates": 0, "epoch": 0, "cursor": 0}
    assert report["scientific_claim"] is False and report["limitations"]
    assert report["parent_state_sha256"] == report["target_state_sha256"]
    assert len(report["mapping"]) == len(model.state_dict())
    assert report["parent"]["contract"]["process_supervision"]["input_diagnostic_weight"] == 0.0
    assert report["declared_behavior_changes"] == {}
    assert report["special_diagnostics"]["inactive_scale_channels"] == []
    assert "inactive" in report["special_diagnostics"]["process_to_context"]
    for entry in report["mapping"]:
        key = entry["source_key"]
        assert key == entry["target_key"]
        assert entry["old_sha256"] == entry["new_sha256"]
        assert torch.equal(model.state_dict()[key], artifacts.saved["model"][key])
        assert model.state_dict()[key].data_ptr() != artifacts.saved["model"][key].data_ptr()
    assert report["report_sha256"] == experiment.canonical_digest({
        key: value for key, value in report.items() if key != "report_sha256"})


def _weights_only_load(original):
    def load(*args, **kwargs):
        assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"
        return original(*args, **kwargs)
    return load


def test_import_prediction_matches_same_weights_without_calendar(artifacts):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = experiment.make_model("process", MODEL_SPEC).eval()
        reference.load_state_dict(artifacts.saved["model"], strict=True)
        imported, _ = parent_import.import_parent(**artifacts.paths)
        imported.eval()
        batch = {"coarse_history": torch.arange(2 * 17 * 8 * 8).reshape(1, 2, 17, 8, 8).float() / 1000,
                 "lead_time_hours": torch.tensor([6.0]), "latitude": torch.linspace(40, 47, 8),
                 "longitude": torch.linspace(0, 7, 8), "init_utc_hour": torch.tensor([12.0]),
                 "init_day_of_year": torch.tensor([11.0])}
        with torch.inference_mode():
            first, second = reference(batch, reasoning_steps=4), imported(batch, reasoning_steps=4)
        assert "init_calendar_year" not in batch
        assert torch.equal(first.forecast, second.forecast)
        assert torch.equal(first.process_state, second.process_state)
        assert torch.equal(first.draft_forecasts, second.draft_forecasts)
        assert torch.isfinite(second.forecast).all()
    finally:
        torch.set_num_threads(threads)


def test_only_detach_gradient_behavior_may_change(artifacts):
    spec = dict(MODEL_SPEC, detach_between_steps=True)
    model, report = parent_import.import_parent(**artifacts.paths, model_spec=spec)
    assert model.detach_between_steps is True
    assert report["declared_behavior_changes"] == {"detach_between_steps": {"parent": False, "target": True}}
    assert "resume" in report["gradient_semantics"]
    assert "detach_between_steps" not in report["parent"]["contract"]["model"]
    assert report["parent_state_sha256"] == report["target_state_sha256"]


@pytest.mark.parametrize("change", [{"dim": 16}, {"default_reasoning_steps": 3},
                                   {"local_solver_state": True}, {"use_forecast_feedback": False},
                                   {"detach_between_steps": 1}, {"unknown_switch": True}])
def test_changed_target_architecture_is_rejected(artifacts, change):
    with pytest.raises(ValueError, match="architecture|boolean|arguments"):
        parent_import.import_parent(**artifacts.paths, model_spec=dict(MODEL_SPEC, **change))


def test_generic_requires_explicit_owner_mapping(artifacts):
    with pytest.raises(ValueError, match="Generic requires owner mapping"):
        parent_import.import_parent(**artifacts.paths, target_kind="generic")


@pytest.mark.parametrize("kind", ["missing", "extra", "shape", "dtype", "nonfinite"])
def test_state_rejects_unaccounted_or_invalid_tensors(artifacts, kind):
    state = artifacts.saved["model"]
    key = next(key for key, value in state.items() if value.is_floating_point() and value.numel() > 1)
    if kind == "missing":
        del state[key]
    elif kind == "extra":
        state["unused_parameter"] = torch.zeros(1)
    elif kind == "shape":
        state[key] = state[key].flatten()[:1]
    elif kind == "dtype":
        state[key] = state[key].double()
    else:
        state[key].reshape(-1)[0] = float("nan")
    _save(artifacts)
    with pytest.raises(ValueError, match="keys|shape/dtype|nonfinite"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("kind", ["signature", "contract", "saved_digest", "contract_digest", "format"])
def test_old_signature_and_digest_are_verified_without_rewriting(artifacts, kind):
    if kind == "signature":
        artifacts.saved["signature"] = "0" * 64
    elif kind == "contract":
        artifacts.saved["contract"]["steps"] = 3
    elif kind == "saved_digest":
        artifacts.saved["model_code_sha256"] = experiment.model_code_digest()
    elif kind == "contract_digest":
        artifacts.saved["contract"]["model_code_sha256"] = experiment.model_code_digest()
        artifacts.saved["signature"] = experiment.canonical_digest(artifacts.saved["contract"])
    else:
        artifacts.saved["format"] = "not-r7-local-v1"
    _save(artifacts)
    snapshot = deepcopy(artifacts.saved["contract"])
    with pytest.raises(ValueError, match="signature|model digest|format"):
        parent_import.import_parent(**artifacts.paths)
    assert artifacts.saved["contract"] == snapshot


@pytest.mark.parametrize("change", ["steps", "local_solver_state", "aux_weight", "fixed_inverse", "updates"])
def test_resigned_contract_still_requires_accepted_m3_controls(artifacts, change):
    contract = artifacts.saved["contract"]
    if change == "steps":
        contract["steps"] = 3
    elif change == "local_solver_state":
        contract["model"]["local_solver_state"] = True
    elif change == "aux_weight":
        contract["process_supervision"]["input_diagnostic_weight"] = 0.1
    elif change == "fixed_inverse":
        contract["process_supervision"]["fixed_train_context"]["normalization_mean"][0] += 1
    else:
        artifacts.saved["updates"] = 399
    _save(artifacts, resign=True, accept_signature=True)
    with pytest.raises(ValueError, match="training contract"):
        parent_import.import_parent(**artifacts.paths)


def test_parent_checkpoint_byte_pin_cannot_be_replaced(artifacts):
    with artifacts.paths["checkpoint"].open("ab") as handle:
        handle.write(b"changed bytes outside tensor payload")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("change", ["bad_signature", "rehashed_unaccepted"])
def test_protocol_digest_and_trust_root_are_strict(artifacts, change):
    artifacts.protocol["test_read"] = True
    if change == "rehashed_unaccepted":
        body = {key: value for key, value in artifacts.protocol.items() if key != "protocol_sha256"}
        artifacts.protocol["protocol_sha256"] = experiment.canonical_digest(body)
    _write_json(artifacts.paths["original_protocol"], artifacts.protocol)
    with pytest.raises(ValueError, match="protocol canonical digest"):
        parent_import.import_parent(**artifacts.paths)


def test_code_zip_byte_pin_is_strict(artifacts):
    with artifacts.paths["codezip"].open("ab") as handle:
        handle.write(b"tampered zip trailer")
    with pytest.raises(ValueError, match="code zip digest"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("change", ["missing", "extra", "source_bytes", "old_model_digest"])
def test_archive_hash_map_and_model_digest_are_recomputed(artifacts, change):
    with zipfile.ZipFile(artifacts.paths["codezip"]) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    if change == "missing":
        del values["training/never_execute.py"]
    elif change == "extra":
        values["model/extra_parameter.py"] = b"EXTRA = 1\n"
    else:
        values["model/__init__.py"] += b"# different model bytes\n"
    _members(artifacts, values, update_map=change == "old_model_digest", update_model=False)
    with pytest.raises(ValueError, match="member hash map|model source bytes digest"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("name", ["../escape.py", "/model/absolute.py", "model/../escape.py",
                                 "model\\escape.py", "model//extra.py", "model/legacy_v531/a.py",
                                 "outputs/dont_execute.py", "C:/model/a.py"])
def test_archive_unsafe_or_unrelated_member_paths_are_rejected(artifacts, name):
    with zipfile.ZipFile(artifacts.paths["codezip"]) as archive:
        values = {member: archive.read(member) for member in archive.namelist()}
    values[name] = b"raise RuntimeError('never execute')\n"
    _members(artifacts, values, update_map=False, update_model=False)
    with pytest.raises(ValueError, match="archive member"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("kind", ["duplicate", "symlink", "oversize"])
def test_archive_member_metadata_is_fail_closed(artifacts, kind):
    with zipfile.ZipFile(artifacts.paths["codezip"], "a") as archive:
        member = zipfile.ZipInfo("model/additional.py")
        if kind == "symlink":
            member.create_system = 3
            member.external_attr = 0o120777 << 16
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("model/__init__.py", b"duplicated")
        else:
            archive.writestr(member, b"x" * (parent_import.MAX_MEMBER_BYTES + 1) if kind == "oversize" else b"target")
    artifacts.protocol["code"]["code_zip_sha256"] = sha256_file(artifacts.paths["codezip"])
    artifacts.pins["code_zip_sha256"] = artifacts.protocol["code"]["code_zip_sha256"]
    _protocol(artifacts)
    with pytest.raises(ValueError, match="duplicate, symlink, encrypted or oversized"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("field", list(("checkpoint", "original_protocol", "codezip", "sidecar", "trainmanifest")))
def test_missing_artifact_is_rejected(artifacts, field):
    artifacts.paths[field].unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("kind", ["symlink", "test", "parent_traversal"])
def test_input_paths_reject_symlinks_test_and_traversal(artifacts, kind):
    if kind == "symlink":
        alias = artifacts.paths["checkpoint"].parent / "alias.pt"
        alias.symlink_to(artifacts.paths["checkpoint"])
        artifacts.paths["checkpoint"] = alias
    elif kind == "test":
        artifacts.paths["trainmanifest"] = artifacts.paths["trainmanifest"].with_name("test.jsonl")
    else:
        artifacts.paths["checkpoint"] = artifacts.paths["checkpoint"].parent / "unused" / ".." / "parent.pt"
    with pytest.raises(ValueError, match="symlink|sealed test|unsafe path"):
        parent_import.import_parent(**artifacts.paths)


@pytest.mark.parametrize("kind", ["sidecar", "sidecar_marker", "source", "source_preflight", "receipt",
                                 "train_marker", "train_manifest", "fixed_inverse", "actual_data"])
def test_sidecar_source_completion_and_actual_inverse_are_required(artifacts, monkeypatch, kind):
    if kind == "sidecar":
        metadata = deepcopy(artifacts.metadata)
        metadata["dimensionless_mean"][0] += 1
        _write_json(artifacts.paths["sidecar"], metadata)
    elif kind == "sidecar_marker":
        (artifacts.paths["sidecar"].parent / "BUILD_COMPLETE.json").write_bytes(b"{}")
    elif kind == "source":
        artifacts.source.write_bytes(b"tampered source bytes")
    elif kind == "source_preflight":
        preflight = artifacts.paths["trainmanifest"].parent / "source_preflight.json"
        _write_json(preflight, {"schema_version": 1, "source_path": str(artifacts.source),
                              "fingerprint": {"scope": "full-local-file", "sha256": "0" * 64, "bytes": 1}})
    elif kind == "receipt":
        _write_json(artifacts.paths["trainmanifest"].parent.parent.parent / "source_receipt.json",
                    {"local_artifact": {"sha256": "0" * 64}, "synthetic_fallback": False})
    elif kind == "train_marker":
        (artifacts.paths["trainmanifest"].parent / "BUILD_COMPLETE.json").write_bytes(b"{}")
    elif kind == "train_manifest":
        artifacts.paths["trainmanifest"].write_text("tampered manifest\n", encoding="utf-8")
    elif kind == "fixed_inverse":
        artifacts.root["normalization_mean"][0] += 1
    else:
        monkeypatch.setattr(experiment, "dataset_identity", lambda path: ("0" * 64, artifacts.reader))
    with pytest.raises(ValueError, match="identity|source|BUILD_COMPLETE|inverse|complete|marker"):
        parent_import.import_parent(**artifacts.paths)


def test_unaccepted_checkpoint_is_rejected_before_deserialization(artifacts, monkeypatch):
    artifacts.paths["checkpoint"].write_bytes(b"unaccepted payload")
    def forbidden(*args, **kwargs):
        raise AssertionError("hash acceptance must precede deserialization")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="before deserialization"):
        parent_import.import_parent(**artifacts.paths)


def test_signature_verifier_uses_only_the_old_digest(artifacts, monkeypatch):
    snapshot = deepcopy(artifacts.saved["contract"])
    def forbidden(*args, **kwargs):
        raise AssertionError("signature verification must not consult the current code")
    monkeypatch.setattr(experiment, "model_code_digest", forbidden)
    actual = parent_import.verify_parent_checkpoint_signature(
        artifacts.saved, archived_model_sha256=artifacts.pins["model_code_sha256"])
    assert actual == artifacts.saved["signature"]
    assert artifacts.saved["contract"] == snapshot


def test_tensor_hash_binds_shape_dtype_and_handles_bfloat16():
    first = torch.arange(4, dtype=torch.float32)
    assert parent_import.tensor_sha256(first) != parent_import.tensor_sha256(first.reshape(2, 2))
    assert parent_import.tensor_sha256(first) != parent_import.tensor_sha256(first.double())
    assert parent_import.tensor_sha256(first.bfloat16()) == parent_import.tensor_sha256(first.bfloat16().clone())
