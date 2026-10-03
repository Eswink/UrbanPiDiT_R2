"""Read-only accepted-input pins and a v2-specific static active source closure."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

from .r7_v2_protocol import (ROOT, digest, local_path, read_json, safe_output,
                             sha256_file, write_path)

ARCHIVE_ROOTS = (
    "training/r7_v2_protocol.py", "training/r7_v2_driver.py", "training/r7_v2_worker.py",
    "training/r7_v2_identity.py", "training/r7_v2_profile.py", "scripts/study_r7_v2_remaining.py",
    "training/r7_parent_import.py", "data/r7_autoregressive_dataset.py",
    "training/r7_autoregressive_rollout.py", "training/r7_autoregressive_runner.py",
    "training/r7_v2_evaluation.py", "training/r7_v2_results.py", "scripts/r7_m3_offline.py",
    "training/r7_m3_driver.py", "training/r7_m3_identity.py",
)
CONFIG_FILES = ("pyproject.toml", "requirements.txt", "requirements-r7-data.txt", "requirements-dev.txt")
PARENT_OUTPUT = ROOT / "outputs/r7_73_process_supervision"


def pin_inputs(manifests, sidecar, *, check=lambda: None):
    """Reuse accepted M3 identity methods, never its build/frozen protocol or clocks."""
    from data.r7_autoregressive_dataset import preflight_training_windows
    from .r7_m3_identity import dataset_pins, sidecar_pins, source_identity
    manifests, sidecar = local_path(manifests), local_path(sidecar)
    check()
    data, reader, root = dataset_pins(manifests)
    check()
    sources = source_identity(manifests, root)
    check()
    pins, metadata = sidecar_pins(sidecar, data, sources, reader, root)
    check()
    windows = preflight_training_windows(manifests / "train.jsonl")
    check()
    if (windows["test_read"] is not False or windows["state_fields_read"] is not False
            or windows["input_windows"] != data["train_windows"] or windows["usable_windows"] < 1
            or windows["usable_windows"] + len(windows["excluded_sample_ids"]) != windows["input_windows"]
            or windows["excluded_sample_ids"] != [item["sample_id"] for item in windows["exclusions"]]):
        raise ValueError("explicit exact metadata-only training window preflight required")
    return data, sources, pins, windows, reader, root, metadata


def pin_parents(manifests, sidecar, data, sources, *, check=lambda: None):
    from .r7_parent_import import import_parent
    from .r7_v2_profile import seed_cpu
    original_path = local_path(PARENT_OUTPUT / "protocol.json")
    original = read_json(original_path)
    body = {key: value for key, value in original.items() if key != "protocol_sha256"}
    if original.get("protocol_sha256") != digest(body):
        raise ValueError("original parent protocol digest mismatch")
    choices = [arm for arm in original["arms"] if arm["name"] == "aux_off"]
    if len(choices) != 1:
        raise ValueError("exact parent aux_off model specification required")
    spec = {**choices[0]["model_config"], "detach_between_steps": False}
    parents = {}
    for seed in (41, 42):
        check()
        seed_cpu(seed)
        checkpoint = local_path(PARENT_OUTPUT / f"seed{seed}/training/aux_off/update_0000400.pt")
        paths = {"checkpoint": str(checkpoint), "original_protocol": str(original_path),
                 "codezip": str(local_path(PARENT_OUTPUT / "code.zip")),
                 "sidecar": str(local_path(sidecar)), "trainmanifest": str(local_path(Path(manifests) / "train.jsonl"))}
        model, report = import_parent(checkpoint, **{key: value for key, value in paths.items() if key != "checkpoint"},
                                      target_kind="process", model_spec=spec)
        if (report["data"]["data_identity"] != data["data_identity"]
                or report["data"]["sources"] != sources or report["parent"]["updates"] != 400
                or report["parent"]["contract"]["seed"] != seed or report["optimizer_reset"] is not True
                or report["resume"] is not False):
            raise ValueError("accepted matching seed selected400 weight-only parent import required")
        del model
        parents[str(seed)] = {"original_output": str(local_path(PARENT_OUTPUT)), **paths,
                              "file_sha256": {key: sha256_file(path) for key, path in paths.items()},
                              "original_protocol_sha256": original["protocol_sha256"], "model_spec": dict(spec),
                              "target_kind": "process", "selected_update": 400,
                              "import_report": report, "import_report_sha256": digest(report)}
        check()
    return parents


def verify_parent_pins(protocol, *, check=lambda: None):
    for parent in protocol["parents"].values():
        for key, expected in parent["file_sha256"].items():
            check()
            if sha256_file(parent[key]) != expected:
                raise ValueError(f"read-only parent input changed after freeze: {key}")
        if (digest(parent["import_report"]) != parent["import_report_sha256"]
                or read_json(parent["original_protocol"])["protocol_sha256"] != parent["original_protocol_sha256"]):
            raise ValueError("frozen parent provenance changed")
    check()


def verify_input_pins(protocol, *, check=lambda: None):
    pinned = pin_inputs(protocol["manifests"], protocol["sidecar"]["path"], check=check)
    if pinned[:4] != (protocol["data"], protocol["sources"], protocol["sidecar"], protocol["windows"]):
        raise ValueError("frozen source/data/sidecar/exact windows changed")
    verify_parent_pins(protocol, check=check)
    return pinned


def _module_path(module):
    if not module or module.split(".")[0] not in ("data", "model", "training", "scripts"):
        return None
    path = ROOT.joinpath(*module.split("."))
    return next((candidate for candidate in (path.with_suffix(".py"), path / "__init__.py")
                 if candidate.is_file()), None)


def source_closure():
    """Static local imports plus explicit roots/model digest scope; no git or outputs."""
    pending = [ROOT / name for name in ARCHIVE_ROOTS]
    pending.extend(path for path in (ROOT / "model").rglob("*.py")
                   if not any("legacy" in part for part in path.relative_to(ROOT).parts))
    seen = set()
    while pending:
        path = local_path(pending.pop())
        if path in seen:
            continue
        relative = path.relative_to(ROOT)
        if (not path.is_file() or path.suffix != ".py"
                or relative.parts[0] not in ("data", "model", "training", "scripts")
                or any("legacy" in part for part in relative.parts)):
            raise ValueError(f"missing/unsafe active v2 source closure member: {relative}")
        seen.add(path)
        package = ".".join(relative.with_suffix("").parts[:-1])
        for parent in path.parents:
            if parent == ROOT:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                names = [module] + [module + "." + alias.name for alias in node.names]
            for name in names:
                candidate = _module_path(name)
                if candidate is not None:
                    pending.append(candidate)
    return sorted({path.relative_to(ROOT).as_posix() for path in seen}
                  | {name for name in CONFIG_FILES if (ROOT / name).is_file()})


def model_digest():
    value = hashlib.sha256()
    for path in sorted((ROOT / "model").rglob("*.py")):
        name = path.relative_to(ROOT / "model").as_posix()
        if "legacy" not in name:
            value.update(name.encode() + b"\0")
            value.update(local_path(path).read_bytes())
    return value.hexdigest()


def _safe_member(name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or str(relative) != name or "\\" in name or ":" in name
            or any(part in (".", "..", "outputs", "tests", "manifests") or "legacy" in part for part in relative.parts)
            or (name not in CONFIG_FILES and (relative.parts[0] not in ("data", "model", "training", "scripts")
                                              or relative.suffix != ".py"))):
        raise ValueError("unsafe/unrelated v2 code archive member")
    return relative


def archive_code(output, *, check=lambda: None):
    output = safe_output(output)
    files = {}
    with zipfile.ZipFile(write_path(output / "code.zip", output), "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_closure():
            check()
            _safe_member(name)
            content = local_path(ROOT / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    check()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True, timeout=10)
    for name, content in (("code_commit.txt", commit + "\n"), ("code_status.txt", status)):
        with write_path(output / name, output).open("x", encoding="utf-8") as stream:
            stream.write(content)
    check()
    return {"source_root": str(ROOT), "files": files, "source_tree_sha256": digest(files),
            "code_zip_sha256": sha256_file(output / "code.zip"), "model_code_sha256": model_digest(),
            "base_commit": commit, "working_tree_modified": bool(status.strip()),
            "code_commit_sha256": sha256_file(output / "code_commit.txt"),
            "code_status_sha256": sha256_file(output / "code_status.txt"),
            "archive_scope": "static v2 explicit roots/local import closure plus active model digest and configs; no manifests/tests/outputs/legacy",
            "git_inspection": "read-only HEAD and porcelain status; unrelated dirty files do not widen the source archive",
            "identity": "exact HEAD plus content addressed dirty source closure, not asserted clean"}


def verify_code(protocol, *, check=lambda: None):
    code, output = protocol["code"], safe_output(protocol["output"])
    for name, key in (("code_commit.txt", "code_commit_sha256"), ("code_status.txt", "code_status_sha256")):
        check()
        if sha256_file(output / name) != code[key]:
            raise ValueError("frozen Git identity receipt changed")
    if (local_path(output / "code_commit.txt").read_text(encoding="utf-8").strip() != code["base_commit"]
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip() != code["base_commit"]):
        raise ValueError("run HEAD differs from frozen exact Git identity")
    if bool(local_path(output / "code_status.txt").read_text(encoding="utf-8").strip()) != code["working_tree_modified"]:
        raise ValueError("frozen dirty status receipt differs")
    if (code["source_root"] != str(ROOT) or set(code["files"]) != set(source_closure())
            or digest(code["files"]) != code["source_tree_sha256"]
            or sha256_file(output / "code.zip") != code["code_zip_sha256"]):
        raise ValueError("frozen v2 archive/source closure changed")
    with zipfile.ZipFile(output / "code.zip") as archive:
        members = archive.infolist()
        if len(members) != len(code["files"]) or {item.filename for item in members} != set(code["files"]):
            raise ValueError("v2 archive exact member inventory mismatch")
        for item in members:
            check()
            _safe_member(item.filename)
            if item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG):
                raise ValueError("archive symlink/special member forbidden")
            expected = code["files"][item.filename]
            if (hashlib.sha256(archive.read(item)).hexdigest() != expected
                    or sha256_file(ROOT / item.filename) != expected):
                raise ValueError(f"active source changed after protocol freeze: {item.filename}")
    if model_digest() != code["model_code_sha256"]:
        raise ValueError("active model implementation digest changed")
    check()
