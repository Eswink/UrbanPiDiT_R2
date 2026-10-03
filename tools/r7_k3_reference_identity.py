"""Byte/metadata-only pins, safe archive extraction and exclusive source isolation."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

from r7_k3_reference_support import (
    ARCHIVE_PROTOCOL_FILE, ARCHIVE_PROTOCOL_SHA, ARCHIVE_SHA, ARCHIVE_TREE_SHA,
    COUNTS, DATA_SHA, GPU_UUID, LEADS, LIMITATIONS, MODEL_MAP_SHA, MODEL_SHA,
    OLD_PROTOCOL_FILE, OLD_PROTOCOL_SHA, PARENTS, SOURCE_BASE, clock_identity, digest, local_path, output_path,
    planned_jobs, read_json, require, sha256_file, validate_clock, write_json,
)

MODULES = ("r7_k3_reference.py", "r7_k3_reference_support.py", "r7_k3_reference_identity.py",
           "r7_k3_reference_cases.py", "r7_k3_reference_evaluate.py",
           "r7_k3_reference_worker.py", "r7_k3_reference_driver.py")
MODEL_SPEC = dict(in_channels=17, out_channels=17, history_steps=2, architecture="window", dim=192,
                  depth=4, heads=4, window_size=4, patch_size=2, dropout=0.0, anchored_processes=8,
                  free_processes=8, use_forecast_feedback=True, spacetime_inputs=True,
                  default_reasoning_steps=3, positional_process_readout=True)


def verify_digest(protocol, expected):
    require(protocol.get("protocol_sha256") == expected == digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}),
            "original protocol digest changed")


def original_archive(repo, *, check=lambda: None):
    old = local_path(repo) / "outputs/r7_73_process_supervision"
    require(sha256_file(old / "protocol.json", check=check) == ARCHIVE_PROTOCOL_FILE, "original archive protocol bytes changed")
    protocol = read_json(old / "protocol.json")
    verify_digest(protocol, ARCHIVE_PROTOCOL_SHA)
    code = protocol["code"]
    require(code["code_zip_sha256"] == ARCHIVE_SHA and sha256_file(old / "code.zip", check=check) == ARCHIVE_SHA,
            "archived sourcebridge zip pin mismatch")
    require(len(code["files"]) == 80 and digest(code["files"]) == code["source_tree_sha256"] == ARCHIVE_TREE_SHA,
            "exact original 80-member map required")
    model_map = {n: h for n, h in code["files"].items() if n.startswith("model/")}
    require(len(model_map) == 28 and digest(model_map) == MODEL_MAP_SHA and code["model_code_sha256"] == MODEL_SHA,
            "genuine28-model sourcebridge map required")
    return protocol


def safe_members(archive, files):
    infos = archive.infolist()
    names = [i.filename for i in infos]
    require(len(names) == len(set(names)) and set(names) == set(files), "exact archive member map/uniqueness required")
    for info in infos:
        name = info.filename
        member = PurePosixPath(name)
        mode = info.external_attr >> 16
        require(name == member.as_posix() and not member.is_absolute() and ".." not in member.parts and "\\" not in name,
                "archive traversal or noncanonical member forbidden")
        require(not info.is_dir() and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                "archive directories/symlinks/special files forbidden")
        require(not any("legacy" in part for part in member.parts), "legacy code may not enter archive runtime")
    return infos


def extract_archive(archive_path, files, root, *, check=lambda: None):
    """Validate entire archive before first write; caller supplies a fresh canonical root."""
    root = local_path(root)
    require(not root.exists(), "archive extraction requires fresh exclusive root")
    with zipfile.ZipFile(local_path(archive_path)) as archive:
        infos = safe_members(archive, files)
        for info in infos:
            check()
            import hashlib
            require(hashlib.sha256(archive.read(info)).hexdigest() == files[info.filename], "archive member bytes differ")
        root.mkdir(parents=True, exist_ok=False)
        for info in infos:
            check()
            path = local_path(root / info.filename)
            require(path.is_relative_to(root), "archive member escaped exclusive root")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(archive.read(info))
    verify_extraction(root, files, check=check)


def verify_extraction(root, files, *, check=lambda: None):
    root = local_path(root)
    actual = set()
    for path in root.rglob("*"):
        local_path(path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(files), "extracted archive exact map required; no added importer/wrapper/partial repairs")
    for name, expected in files.items():
        require(sha256_file(root / name, check=check) == expected, "extracted source pin mismatch: " + name)
    import hashlib
    value = hashlib.sha256()
    for path in sorted((root / "model").rglob("*.py")):
        value.update(path.relative_to(root / "model").as_posix().encode() + b"\0")
        value.update(path.read_bytes())
    require(value.hexdigest() == MODEL_SHA, "extracted model digest differs from genuine K3")


def parent_metadata(repo, seed, *, check=lambda: None):
    directory = local_path(repo) / f"outputs/r7_72_rw_b_subtraction/seed{seed}"
    report_path = directory / "training/process_spacetime_rwa/training_report.json"
    checkpoint = report_path.parent / "update_0000400.pt"
    require(sha256_file(directory / "protocol.json", check=check) == OLD_PROTOCOL_FILE, "original72 protocol file changed")
    original = read_json(directory / "protocol.json")
    verify_digest(original, OLD_PROTOCOL_SHA)
    pins = PARENTS[seed]
    require(sha256_file(report_path, check=check) == pins["report_sha256"], "original training report changed")
    require(sha256_file(checkpoint, check=check) == pins["checkpoint_sha256"], "original selected400 checkpoint opaque pin changed")
    report = read_json(report_path)
    contract = report["contract"]
    require(digest(contract) == report["signature"] == pins["signature"], "original K3 training signature differs")
    require(contract["model"] == MODEL_SPEC and contract["kind"] == "process" and contract["steps"] == 3
            and contract["seed"] == seed and contract["data_identity"] == DATA_SHA
            and contract["process_weight"] == 0.0 and contract["bf16"] is False
            and contract["optimization"] == "streamed-truncated", "genuine400-update K3 training contract required")
    require(not any(k in contract for k in ("process_supervision", "intervention", "sidecar")), "old K3 has no new auxiliary/sidecar/intervention")
    require(report["total_updates"] == report["updates_this_run"] == report["selected_update"] == 400
            and report["early_stopped"] is False and report["selection_split"] == "val"
            and report["selected_checkpoint"] == checkpoint.relative_to(repo).as_posix(), "genuine selected400 report required")
    return {**pins, "checkpoint": str(checkpoint), "report": str(report_path), "contract": contract,
            "protocol": str(directory / "protocol.json"), "protocol_file_sha256": OLD_PROTOCOL_FILE,
            "protocol_sha256": OLD_PROTOCOL_SHA}


def store_snapshot(store, *, check=lambda: None):
    """Opaque metadata/norm/coordinate hashes plus all-file stat guard, never state decode/test reads."""
    store = local_path(store)
    hashes, inventory = {}, {}
    for path in sorted(store.rglob("*")):
        check()
        path = local_path(path)
        if not path.is_file():
            continue
        relative = path.relative_to(store).as_posix()
        st = path.stat()
        inventory[relative] = [st.st_size, st.st_mtime_ns, st.st_ino]
        parts = path.relative_to(store).parts
        if path.name in ("zarr.json", ".zattrs", ".zarray", ".zgroup") or parts[0] in (
                "latitude", "longitude", "time_ns", "normalization_mean", "normalization_std",
                "process_normalization_mean", "process_normalization_std"):
            hashes[relative] = sha256_file(path, check=check)
    require(hashes and inventory, "existing complete store snapshot required")
    return {"file_sha256": hashes, "inventory_sha256": digest(inventory), "files": len(inventory),
            "scope": "state/process chunks stat-only (including sealed chunks); no chunk data opened for this guard"}


def source_pins(repo, original, *, check=lambda: None):
    sources = original["sources"]
    manifests = local_path(repo) / "outputs/r7_m2_segment/store/manifests"
    pairs = [(sources["source_path"], sources["source_sha256"]),
             (sources["source_receipt_path"], sources["source_receipt_sha256"]),
             (manifests / "source_preflight.json", sources["preflight_report_sha256"]),
             (manifests / "BUILD_COMPLETE.json", sources["build_complete_sha256"]),
             (manifests / "train.jsonl", sources["train_manifest_sha256"]),
             (manifests / "val.jsonl", sources["val_manifest_sha256"])]
    pins = {str(local_path(path)): expected for path, expected in pairs}
    for path, expected in pins.items():
        require(sha256_file(path, check=check) == expected, "original real source/preflight/publication/manifest pin mismatch")
    require(read_json(manifests / "BUILD_COMPLETE.json") == {"schema_version": 1, "build_complete": True}, "BUILD_COMPLETE required")
    receipt, preflight = read_json(sources["source_receipt_path"]), read_json(manifests / "source_preflight.json")
    require(receipt["synthetic_fallback"] is False and receipt["local_artifact"]["sha256"] == sources["source_sha256"]
            and preflight["fingerprint"]["sha256"] == sources["source_sha256"]
            and local_path(sources["source_path"]).stat().st_size == sources["source_bytes"], "real receipt/preflight/source disagree")
    for split in ("train", "val"):
        records = [json.loads(s) for s in (manifests / f"{split}.jsonl").read_text().splitlines() if s.strip()]
        require(len(records) == original["data"][f"{split}_windows"] and all(r["split"] == split for r in records), "train/val manifest membership differs")
    return pins


def companion_zip_bytes(root, files, *, check=lambda: None):
    """Deterministic stored ZIP pinned before freeze; no clock/HEAD/compression-version identity."""
    import io
    import hashlib
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(files):
            check()
            content = local_path(Path(root) / name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == files[name], "companion source changed while freezing zip")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content)
    check()
    return buffer.getvalue()


def runner_identity(root, *, check=lambda: None):
    import hashlib
    root = local_path(root)
    files = {name: sha256_file(root / name, check=check) for name in MODULES}
    zip_hash = hashlib.sha256(companion_zip_bytes(root, files, check=check)).hexdigest()
    return {"root": str(root), "files": files, "source_sha256": digest(files), "companion_zip_sha256": zip_hash,
            "source_base_commit": SOURCE_BASE, "source_commit": None,
            "commit_note": "616b base plus exact companion bytes; no invented historical/new commit or current-HEAD runtime dependency"}


def verify_runner(code, *, check=lambda: None):
    actual = runner_identity(code["root"], check=check)
    require(actual == code, "frozen companion driver source changed")


def build_protocol(repo, output, *, started, check=lambda: None):
    repo, output = local_path(repo), local_path(output)
    original = original_archive(repo, check=check)
    cases = original["data"]["evaluation_cases"]
    require([cases[str(l)]["n_available"] for l in LEADS] == list(COUNTS)
            and all(len(cases[str(l)]["cases"]) == n for l, n in zip(LEADS, COUNTS)), "fixed22/21/19/15/11 cases required")
    data = original["data"]
    require(data["data_identity"] == DATA_SHA and data["shape"] == [240, 17, 65, 65], "existing genuine M2 data required")
    protocol = {"format": "r7-original-k3-reference-v1", "issue": "#74 original #72 checkpoint, all17 variables x5leads",
                "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False, "training_updates": 0,
                "output": str(output), "repo": str(repo), "planned_seconds": 1800, "hard_cap_seconds": 3600,
                "clock": clock_identity(started),
                "cleanup_reserve_seconds": 10, "jobs": planned_jobs(), "max_samples": 32,
                "reasoning_steps": 3, "step_hours": 6, "boundary_margin": 2,
                "archive": {"path": str(repo / "outputs/r7_73_process_supervision/code.zip"), "sha256": ARCHIVE_SHA,
                            "protocol_file_sha256": ARCHIVE_PROTOCOL_FILE, "protocol_sha256": ARCHIVE_PROTOCOL_SHA,
                            "source_tree_sha256": ARCHIVE_TREE_SHA, "files": original["code"]["files"],
                            "model_code_sha256": MODEL_SHA, "model_map_sha256": MODEL_MAP_SHA},
                "code": runner_identity(Path(__file__).parent, check=check), "data": data,
                "input_file_sha256": source_pins(repo, original, check=check),
                "store_snapshot": store_snapshot(data["store"], check=check),
                "parents": {str(seed): parent_metadata(repo, seed, check=check) for seed in PARENTS},
                "gpu": {"uuid": GPU_UUID, "device": "cuda:0", "policy": "shared", "estimated_peak_mib": 342,
                        "headroom_margin_mib": 2048, "gate": "free >= max(342, ceil(observed own reserved peak MiB)) +2048 before startup/every spawn",
                        "helper_source": "sourcebridge has no standalone scripts/r7_m3_gpu.py; stdlib read-only NVIDIA helpers preserve original M3-driver/worker contracts"},
                "calendar": "unchanged old rollout: accumulated lead, fixed init day/hour, init_year metadata only,365.25 feature denominator",
                "reproducibility": "source/data/protocol pinned; not asserted GPU bitwise or full original historical source replay"}
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def read_protocol(path):
    path = local_path(path)
    p = read_json(path)
    require(p.get("protocol_sha256") == digest({k: v for k, v in p.items() if k != "protocol_sha256"}), "reference protocol digest changed")
    validate_clock(p["clock"], p["clock"]["earliest_started_perf_counter"])
    require(p.get("format") == "r7-original-k3-reference-v1" and p.get("output") == str(path.parent)
            and p.get("jobs") == planned_jobs() and p.get("training_updates") == 0 and p.get("reasoning_steps") == 3
            and p.get("step_hours") == 6 and p.get("max_samples") == 32 and p.get("boundary_margin") == 2
            and p.get("planned_seconds") == 1800 and p.get("hard_cap_seconds") == 3600 and p.get("cleanup_reserve_seconds") == 10
            and p.get("test_read") is False and p.get("scientific_claim") is False and p.get("limitations") == LIMITATIONS,
            "frozen K3 reference scope/budget differs")
    require(p["gpu"]["uuid"] == GPU_UUID and p["gpu"]["device"] == "cuda:0" and p["gpu"]["policy"] == "shared"
            and p["gpu"]["headroom_margin_mib"] == 2048 and p["gpu"]["estimated_peak_mib"] == 342, "fixed shared GPU gate required")
    require(output_path(path.parent, p["repo"]) == path.parent, "fresh independent output outside protected/original paths required")
    require(local_path(p["code"]["root"]) == Path(__file__).resolve().parent or
            Path(__file__).resolve().parent == path.parent / "companion_code", "protocol source must bind driver or frozen companion root")
    require(p["code"]["source_base_commit"] == SOURCE_BASE and p["code"]["source_commit"] is None
            and set(p["code"]["files"]) == set(MODULES) and digest(p["code"]["files"]) == p["code"]["source_sha256"],
            "exact companion source map/base identity required; no fabricated commit")
    archive = p["archive"]
    require(archive["sha256"] == ARCHIVE_SHA and archive["protocol_sha256"] == ARCHIVE_PROTOCOL_SHA
            and archive["protocol_file_sha256"] == ARCHIVE_PROTOCOL_FILE and archive["source_tree_sha256"] == ARCHIVE_TREE_SHA
            and digest(archive["files"]) == ARCHIVE_TREE_SHA and archive["model_code_sha256"] == MODEL_SHA
            and archive["model_map_sha256"] == MODEL_MAP_SHA
            and archive["path"] == str(local_path(p["repo"]) / "outputs/r7_73_process_supervision/code.zip"),
            "fixed genuine K3 compatible archive pins required")
    return p


def verify_inputs(protocol, *, check=lambda: None):
    check()
    require(read_protocol(Path(protocol["output"]) / "protocol.json") == protocol, "frozen reference protocol changed during execution")
    original = original_archive(protocol["repo"], check=check)
    require(original["code"]["files"] == protocol["archive"]["files"] and original["data"] == protocol["data"], "archive/data qualification changed")
    for path, expected in protocol["input_file_sha256"].items():
        require(sha256_file(path, check=check) == expected, "frozen input bytes changed")
    require(source_pins(protocol["repo"], original, check=check) == protocol["input_file_sha256"], "frozen source map differs")
    require(store_snapshot(protocol["data"]["store"], check=check) == protocol["store_snapshot"], "store changed after freeze")
    for seed in PARENTS:
        require(parent_metadata(protocol["repo"], seed, check=check) == protocol["parents"][str(seed)], "original parent contract/pins changed")
    verify_runner(protocol["code"], check=check)


def isolate_archive(root, companion, repo):
    root, companion, repo = map(local_path, (root, companion, repo))
    require(Path.cwd().resolve() == root, "worker cwd must be exact archive root")
    packages = ("data", "model", "training", "scripts")
    require(not any(n.split(".")[0] in packages for n in sys.modules), "project packages imported before isolation")
    environment = Path(sys.prefix).resolve()
    entries = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        in_environment = environment != repo and path.is_relative_to(environment)
        if path in (root, companion) or (path.is_relative_to(repo) and not in_environment):
            continue
        if not in_environment and any((path / n).exists() for n in packages):
            continue
        entries.append(str(path))
    sys.path[:] = [str(root), str(companion), *entries]
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()


def assert_origins(root, companion):
    root, companion = map(local_path, (root, companion))
    require(Path.cwd().resolve() == root and Path(sys.path[0]).resolve() == root, "archive import isolation lost")
    for name, module in tuple(sys.modules.items()):
        package = name.split(".")[0]
        if package not in ("data", "model", "training", "scripts") and not package.startswith("r7_k3_reference"):
            continue
        paths = list(getattr(module, "__path__", ()))
        if getattr(module, "__file__", None):
            paths.append(module.__file__)
        expected = companion if package.startswith("r7_k3_reference") else root
        require(paths and all(local_path(p).is_relative_to(expected) for p in paths), "import origin escaped sourcebridge: " + name)
