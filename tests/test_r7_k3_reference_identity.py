"""Byte-only guard tests: deliberately malformed archives/paths/protocols must fail."""
from __future__ import annotations

import ast
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import stat
import sys
import types
import zipfile

from test_r7_k3_reference_support import archive_fixture, archive_source, no_actual_runtime
import pytest
from r7_k3_reference_identity import (
    ARCHIVE_PROTOCOL_FILE, ARCHIVE_PROTOCOL_SHA, ARCHIVE_SHA, ARCHIVE_TREE_SHA, MODEL_MAP_SHA, MODEL_SHA, MODULES,
    assert_origins, extract_archive, isolate_archive, read_protocol, runner_identity, safe_members, store_snapshot, verify_digest, verify_extraction,
)
from r7_k3_reference_support import (
    GPU_UUID, LIMITATIONS, clock_identity, digest, job_key, local_path, output_path, planned_jobs, read_json, sha256_file, validate_clock, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fresh_worker_namespace():
    """Simulate a fresh worker only inside these tests; restore collected active modules."""
    packages = ("data", "model", "training", "scripts")
    original = {name: module for name, module in tuple(sys.modules.items())
                if name.split(".")[0] in packages}
    with pytest.MonkeyPatch.context() as scoped:
        for name in original:
            scoped.delitem(sys.modules, name)
        yield scoped
    assert all(sys.modules.get(name) is module for name, module in original.items())


def test_archive_original80_map_bytepins_and_model28_digest_only(archive_source):
    source = archive_source
    files = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    assert len(files) == 80 and digest(files) == ARCHIVE_TREE_SHA
    model_map = {n: h for n, h in files.items() if n.startswith("model/")}
    assert len(model_map) == 28 and digest(model_map) == MODEL_MAP_SHA
    verify_extraction(source, files)
    assert sha256_file(source / "training/r7_experiment.py") == "67104eeffdadd41e499f5909b19278d114f83577510f423cbc4721cffaa5b025"
    assert sha256_file(source / "model/r7_rollout.py") == "91880ac0572a0249457a88482e17f65c855d7905acb7deefb06026ba8a1ecae9"
    assert all("legacy" not in n for n in files)


@pytest.mark.parametrize("bad", ["../escape.py", "/absolute.py", "a/../../b.py", "a\\b.py", "a//b.py", "./b.py", "model/legacy_v6/a.py"])
def test_archive_traversal_noncanonical_legacy_rejected_before_write(tmp_path, bad):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "x") as z:
        z.writestr("safe.py", b"safe")
        z.writestr(bad, b"bad")
    files = {"safe.py": hashlib.sha256(b"safe").hexdigest(), bad: hashlib.sha256(b"bad").hexdigest()}
    output = tmp_path / "not_created"
    with pytest.raises(ValueError): extract_archive(archive, files, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["duplicate", "symlink", "directory", "wrong_hash", "extra", "missing"])
def test_archive_unsafe_map_and_filetype_rejected_before_write(tmp_path, kind):
    path = tmp_path / "bad.zip"
    files = {"safe.py": hashlib.sha256(b"safe").hexdigest()}
    with zipfile.ZipFile(path, "x") as z:
        if kind == "symlink":
            info = zipfile.ZipInfo("safe.py")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, b"safe")
        elif kind == "directory": z.writestr("safe.py/", b"")
        elif kind != "missing": z.writestr("safe.py", b"safe")
        if kind == "duplicate": z.writestr("safe.py", b"safe")
        if kind == "extra": z.writestr("extra.py", b"extra")
    if kind == "wrong_hash": files["safe.py"] = "0" * 64
    output = tmp_path / "not_created"
    with pytest.raises(ValueError): extract_archive(path, files, output)
    assert not output.exists()


def test_archive_exclusive_full_extraction_tamper_extra_and_symlink_refused(tmp_path, archive_source):
    source = archive_source
    files = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "x") as z:
        for name in files: z.writestr(name, (source / name).read_bytes())
    output = tmp_path / "extracted"
    extract_archive(path, files, output)
    verify_extraction(output, files)
    with pytest.raises(ValueError): extract_archive(path, files, output)
    (output / "importer.py").write_text("extra")
    with pytest.raises(ValueError, match="exact map"): verify_extraction(output, files)
    (output / "importer.py").unlink()
    (output / "model/r7_rollout.py").write_text("changed")
    with pytest.raises(ValueError, match="pin mismatch"): verify_extraction(output, files)


def test_local_path_and_output_refuse_protected_ancestors_and_existing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    good = repo / "outputs/fresh"
    assert output_path(good, repo, fresh=True) == good
    for bad in (repo, repo / "data/processed/new", repo / "model/new", repo / "tools/new",
                repo / "outputs/r7_m2_segment/new", repo / "outputs/r7_72_rw_b_subtraction/new",
                repo / "outputs/r7_73_process_supervision/new"):
        with pytest.raises(ValueError): output_path(bad, repo, fresh=True)
    good.mkdir(parents=True)
    with pytest.raises(FileExistsError): output_path(good, repo, fresh=True)
    link = tmp_path / "linked"
    link.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError): local_path(link / "child")
    with pytest.raises(ValueError): local_path(repo / "test.jsonl")
    with pytest.raises(ValueError): local_path("https://example.invalid/data")


def test_store_guard_is_opaque_and_never_reads_any_state_or_process_chunk(tmp_path, monkeypatch):
    store = tmp_path / "cache.zarr"
    (store / "state/c").mkdir(parents=True)
    (store / "normalization_std/c").mkdir(parents=True)
    (store / "process_diagnostics_raw/c").mkdir(parents=True)
    (store / "zarr.json").write_text("{\"synthetic\":true}")
    (store / "state/zarr.json").write_text("{\"shape\":[3,1,1,1]}")
    (store / "state/c/0").write_bytes(b"opaque synthetic state bytes")
    (store / "process_diagnostics_raw/c/0").write_bytes(b"opaque synthetic diagnostic bytes")
    (store / "normalization_std/c/0").write_bytes(b"opaque synthetic norm bytes")
    real_open = Path.open
    def guarded_open(path, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if path in (store / "state/c/0", store / "process_diagnostics_raw/c/0") and "r" in mode:
            raise AssertionError("weather chunks must never be read")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    snapshot = store_snapshot(store)
    assert len(snapshot["file_sha256"]) == 3 and snapshot["files"] == 5
    (store / "state/c/0").write_bytes(b"changed opaque state")
    assert store_snapshot(store)["inventory_sha256"] != snapshot["inventory_sha256"]
    (store / "normalization_std/c/0").write_bytes(b"changed opaque norm")
    assert store_snapshot(store)["file_sha256"] != snapshot["file_sha256"]


def synthetic_protocol(output, repo):
    p = {"format": "r7-original-k3-reference-v1", "output": str(output), "repo": str(repo), "jobs": planned_jobs(),
         "training_updates": 0, "reasoning_steps": 3, "step_hours": 6, "max_samples": 32, "boundary_margin": 2,
         "planned_seconds": 1800, "hard_cap_seconds": 3600, "cleanup_reserve_seconds": 10,
         "test_read": False, "scientific_claim": False, "limitations": LIMITATIONS,
         "code": runner_identity(ROOT / "tools"), "clock": clock_identity(0.),
         "gpu": {"uuid": GPU_UUID, "device": "cuda:0", "policy": "shared", "headroom_margin_mib": 2048, "estimated_peak_mib": 342},
         "archive": {"sha256": ARCHIVE_SHA, "protocol_sha256": ARCHIVE_PROTOCOL_SHA, "protocol_file_sha256": ARCHIVE_PROTOCOL_FILE,
                     "source_tree_sha256": ARCHIVE_TREE_SHA, "files": {}, "model_code_sha256": MODEL_SHA,
                     "model_map_sha256": MODEL_MAP_SHA, "path": str(repo / "outputs/r7_73_process_supervision/code.zip")}}
    _, identity = archive_fixture()
    p["archive"]["files"] = identity["files"]
    p["protocol_sha256"] = digest(p)
    return p


@pytest.mark.parametrize("key,value", [("reasoning_steps", 4), ("max_samples", 11), ("training_updates", 1), ("boundary_margin", 1),
                                         ("planned_seconds", 900), ("hard_cap_seconds", 1800), ("test_read", True), ("scientific_claim", True), ("jobs", [])])
def test_frozen_scope_rejects_even_self_consistent_changed_digest(tmp_path, key, value):
    p = synthetic_protocol(tmp_path, tmp_path.parent / "repo")
    p[key] = value
    p["protocol_sha256"] = digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    write_json(tmp_path / "protocol.json", p)
    with pytest.raises(ValueError): read_protocol(tmp_path / "protocol.json")


def test_protocol_digest_tamper_and_exclusive_publication_fail(tmp_path):
    p = synthetic_protocol(tmp_path, tmp_path.parent / "repo")
    path = tmp_path / "protocol.json"
    write_json(path, p)
    assert read_protocol(path) == p
    with pytest.raises(FileExistsError): write_json(path, p)
    p["protocol_sha256"] = "wrong"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): read_protocol(path)
    with pytest.raises(ValueError): verify_digest({"protocol_sha256": "wrong"}, "wrong")


def test_isolation_preserves_repo_nested_venv_sitepackages_and_removes_current_code(tmp_path, monkeypatch):
    root, companion, repo = (tmp_path / n for n in ("archive", "companion", "repo"))
    for path in (root, companion, repo): path.mkdir()
    environment = repo / ".venv"
    site = environment / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (repo / "model").mkdir()
    monkeypatch.setattr(sys, "prefix", str(environment))
    monkeypatch.setattr(sys, "path", [str(repo), str(repo / "tools"), str(site), str(root), str(companion)])
    monkeypatch.chdir(root)
    with fresh_worker_namespace():
        isolate_archive(root, companion, repo)
        assert sys.path[:2] == [str(root), str(companion)]
        assert str(site) in sys.path
        assert str(repo) not in sys.path and str(repo / "tools") not in sys.path


def test_isolation_rejects_preimport_and_origin_escape(tmp_path, monkeypatch):
    root, companion, repo = (tmp_path / n for n in ("archive", "companion", "repo"))
    for path in (root, companion, repo): path.mkdir()
    monkeypatch.chdir(root)
    with fresh_worker_namespace() as scoped:
        module = types.ModuleType("model")
        module.__file__ = str(repo / "model/__init__.py")
        scoped.setitem(sys.modules, "model", module)
        with pytest.raises(ValueError, match="before isolation"): isolate_archive(root, companion, repo)
        monkeypatch.setattr(sys, "path", [str(root), str(companion)])
        with pytest.raises(ValueError, match="escaped"): assert_origins(root, companion)


def test_cli_offline_bootstrap_before_companion_and_clock_before_all(tmp_path, monkeypatch):
    path = ROOT / "tools/r7_k3_reference.py"
    fake = types.ModuleType("r7_k3_reference_driver")
    calls = []
    def prepare(repo, output, started):
        assert started > 0
        with pytest.raises(RuntimeError): socket.create_connection(("synthetic.invalid", 1))
        calls.append(("prepare", started))
        return {"synthetic": True}
    def run_all(protocol, started):
        calls.append(("all", started))
    fake.prepare, fake.run_all = prepare, run_all
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys, "argv", [str(path), "--mode", "all", "--repo", str(tmp_path), "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc: runpy.run_path(str(path), run_name="__main__")
    assert exc.value.code == 0 and calls[0][1] == calls[1][1]


@pytest.mark.parametrize("arguments", [["--mode", "archive_worker"], ["--mode", "prepare"],
                                      ["--mode", "all", "--repo", "/tmp", "--output", "/tmp/out", "--seed", "41"],
                                      ["--mode", "archive_worker", "--protocol", "/tmp/protocol.json", "--seed", "41", "--lead", "6", "--deadline", "nan"]])
def test_cli_invalid_arguments_rejected_before_runtime_import(monkeypatch, arguments):
    path = ROOT / "tools/r7_k3_reference.py"
    fake = types.ModuleType("r7_k3_reference_driver")
    fake.prepare = lambda *a, **k: pytest.fail("invalid args reached runtime")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    with pytest.raises(SystemExit) as exc: runpy.run_path(str(path), run_name="__main__")
    assert exc.value.code == 2


def test_source_syntax_sizes_names_and_forbidden_runtime_operations():
    for name in MODULES:
        path = ROOT / "tools" / name
        tree = ast.parse(path.read_bytes())
        assert len(path.read_text().splitlines()) <= 600
        assert "-" not in path.stem
        assert all(n.end_lineno - n.lineno + 1 <= 200 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        text = path.read_text()
        assert "torch.load(" not in text and "weights_only=False" not in text
        assert "empty_cache(" not in text and "os.kill(" not in text and "killpg(" not in text
    assert planned_jobs() == [{"seed": s, "lead": l} for s in (41, 42) for l in (6, 12, 24, 48, 72)]
    with pytest.raises(ValueError): job_key({"seed": 41, "lead": 18})


@pytest.mark.parametrize("mutation", ["reset_anchor", "boot_id", "implementation", "monotonic", "adjustable"])
def test_frozen_monotonic_clock_identity_rejects_reset_crossboot_and_semantic_drift(mutation):
    identity = clock_identity(123.)
    started = 123.
    if mutation == "reset_anchor": started = 124.
    elif mutation == "boot_id": identity[mutation] = "00000000-0000-0000-0000-000000000000"
    elif mutation == "implementation": identity[mutation] = "fake_clock"
    elif mutation == "monotonic": identity[mutation] = False
    else: identity[mutation] = True
    with pytest.raises(ValueError): validate_clock(identity, started)
