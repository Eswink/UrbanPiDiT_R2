"""M3 source/data/sidecar and bounded active-source archive pins, never test fields."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from .r7_arm_harness import sha256_file
from .r7_m3_protocol import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOTS = (
    "scripts/study_r7_73_process_supervision.py", "training/r7_m3_worker.py",
    "training/r7_m3_driver.py", "training/r7_scheduled_runner.py", "training/r7_evaluate.py",
    "training/r7_process_forecast_losses.py", "training/r7_process_supervision.py",
    "data/preprocess/r7_process_scale_sidecar.py",
)
CONFIG_FILES = ("pyproject.toml", "requirements.txt", "requirements-r7-data.txt",
                "requirements-dev.txt")


def local_path(path, *, name=None):
    requested = Path(path)
    if "://" in str(path) or requested.is_symlink() or requested.name == "test.jsonl":
        raise ValueError("local nonsymlink non-test paths only")
    result = requested.resolve()
    if name is not None and result.name != name:
        raise ValueError(f"expected {name}, not a sealed or alternate manifest")
    if result.name == "test.jsonl":
        raise ValueError("sealed test manifest forbidden")
    return result


def source_identity(manifests, root=None):
    """Source is hashed opaquely, never decoded; receipt/preflight/marker must agree."""
    manifests = local_path(manifests)
    preflight_path = manifests / "source_preflight.json"
    receipt_path = manifests.parent.parent / "source_receipt.json"
    marker_path = manifests / "BUILD_COMPLETE.json"
    preflight, receipt, marker = map(read_json, (preflight_path, receipt_path, marker_path))
    if marker != {"schema_version": 1, "build_complete": True}:
        raise ValueError("unchanged BUILD_COMPLETE publication marker required")
    fingerprint = preflight["fingerprint"]
    source = local_path(preflight["source_path"])
    source_hash = sha256_file(source)
    if (preflight.get("schema_version") != 1 or fingerprint.get("scope") != "full-local-file"
            or source_hash != fingerprint["sha256"]
            or source_hash != receipt["local_artifact"]["sha256"]
            or source.stat().st_size != fingerprint["bytes"]
            or receipt.get("synthetic_fallback") is not False):
        raise ValueError("source bytes, local preflight and real-source receipt disagree")
    if root is not None and local_path(root.attrs["source"]) != source:
        raise ValueError("actual store source differs from audited source")
    return {"source_path": str(source), "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "source_receipt_path": str(receipt_path), "source_receipt_sha256": sha256_file(receipt_path),
            "preflight_report_sha256": sha256_file(preflight_path),
            "build_complete_sha256": sha256_file(marker_path),
            "train_manifest_sha256": sha256_file(local_path(manifests / "train.jsonl", name="train.jsonl")),
            "val_manifest_sha256": sha256_file(local_path(manifests / "val.jsonl", name="val.jsonl")),
            "scope": "source bytes hashed only for identity; no sealed test fields decoded"}


def dataset_pins(manifests):
    from data.r7_store import normalization, validate_record
    from training.r7_experiment import dataset_identity
    manifests = local_path(manifests)
    readers, identities, stores = {}, {}, set()
    for split in ("train", "val"):
        manifest = local_path(manifests / f"{split}.jsonl", name=f"{split}.jsonl")
        identity, reader = dataset_identity(manifest)
        for record in reader.records:
            if record["split"] != split or len(record["history_indices"]) != 2 or record["lead_time_hours"] != 6:
                raise ValueError("M3 needs fixed two-history +6h train/val manifests")
            store = local_path(manifest.parent / record["store_path"])
            stores.add(str(store))
            validate_record(reader._store(record), record)
        readers[split], identities[split] = reader, identity
    if len(stores) != 1:
        raise ValueError("M3 train/val must refer to the exact same existing store")
    root = readers["train"]._store(readers["train"].records[0])
    mean, std = normalization(root, count=17)
    channels, units = list(root.attrs["channels"]), list(root.attrs["units"])
    if len(channels) != 17 or len(set(channels)) != 17 or len(units) != 17:
        raise ValueError("all 17 atmospheric channels/units are required")
    result = {"store": stores.pop(), "data_identity": identities["train"],
              "val_data_identity": identities["val"], "channels": channels, "units": units,
              "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
              "train_windows": len(readers["train"]), "val_windows": len(readers["val"]),
              "shape": list(root["state"].shape), "test_read": False,
              "evaluation_cases": expected_evaluation_cases(readers["val"])}
    return result, readers["train"], root


def expected_evaluation_cases(reader):
    """Metadata/time-only: exact per-lead cases, never held-out state arrays."""
    from data.r7_evaluation import ZarrRolloutDataset
    from .r7_m3_protocol import EVALUATION_MAX_SAMPLES, LEADS
    store = local_path(reader.manifest.parent / reader.records[0]["store_path"])
    allowed = {r["init_time"] for r in reader.records}
    cases = {}
    for lead in LEADS:
        dataset = ZarrRolloutDataset(store, split="val", lead_hours=(lead,),
                                     history_steps=2, step_hours=6)
        windows = [window for window in dataset.windows
                   if dataset.times[window[0][-1]].isoformat() in allowed]
        if not windows:
            raise ValueError("no exact validation cases for a frozen lead")
        cases[str(lead)] = {
            "n_available": len(windows),
            "cases": [[dataset.times[history[-1]].isoformat(),
                       [dataset.times[i].isoformat() for i in targets]]
                      for history, targets in windows[:EVALUATION_MAX_SAMPLES]],
        }
    return cases


def sidecar_pins(path, data, sources, reader, root):
    from data.preprocess.r7_process_scale_sidecar import load_process_scale_sidecar
    from data.r7_store import validate_record
    requested = local_path(path)
    meta = load_process_scale_sidecar(requested)
    metadata_path = requested / "scale_metadata.json" if requested.is_dir() else requested
    selected = sorted({i for r in reader.records for i in validate_record(root, r)})
    raw = np.stack([np.asarray(root["process_diagnostics_raw"][i]) for i in selected])
    actual = {
        "data_identity": data["data_identity"],
        "train_manifest_sha256": sources["train_manifest_sha256"],
        "store": data["store"], "train_manifest": str(reader.manifest.resolve()),
        "train_sample_ids": [r["sample_id"] for r in reader.records],
        "train_frame_indices": selected, "train_time_ns": [int(root["time_ns"][i]) for i in selected],
        "train_diagnostics_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        "train_diagnostics_dtype": raw.dtype.str,
    }
    if any(meta.get(key) != value for key, value in actual.items()):
        raise ValueError("sidecar must bind actual dataset/train-union/raw/norm identity")
    source = meta["source_identity"]
    if (source["path"] != sources["source_path"] or source["sha256"] != sources["source_sha256"]
            or source["bytes"] != sources["source_bytes"]
            or source["source_preflight_sha256"] != sources["preflight_report_sha256"]):
        raise ValueError("sidecar source/preflight pins disagree")
    return {"path": str(metadata_path), "identity": meta["sidecar_identity"],
            "metadata_sha256": sha256_file(metadata_path),
            "publication_marker_sha256": sha256_file(metadata_path.parent / "BUILD_COMPLETE.json"),
            "schema": meta["schema"], "fit_split": meta["fit_split"],
            "preflight_identity": meta["preflight_identity"]}, meta


def verify_input_pins(protocol):
    data, reader, root = dataset_pins(protocol["manifests"])
    sources = source_identity(protocol["manifests"], root)
    sidecar, meta = sidecar_pins(protocol["sidecar"]["path"], data, sources, reader, root)
    if data != protocol["data"] or sources != protocol["sources"] or sidecar != protocol["sidecar"]:
        raise ValueError("frozen source/data/sidecar identities changed")
    return data, reader, root, meta


def _module_path(module):
    if not module or module.split(".")[0] not in ("data", "model", "training", "scripts"):
        return None
    path = ROOT.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def archive_paths():
    """Static local import closure, not git ls-files or whole repository byte reads."""
    pending = [ROOT / name for name in ARCHIVE_ROOTS]
    # Existing checkpoint verification hashes all active model implementation files.
    pending.extend(path for path in (ROOT / "model").rglob("*.py")
                   if not any("legacy" in part for part in path.relative_to(ROOT).parts))
    seen = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT)
        if path in seen:
            continue
        if (path.is_symlink() or not path.is_file() or path.suffix != ".py"
                or any("legacy" in part for part in relative.parts)
                or relative.parts[0] not in ("data", "model", "training", "scripts")):
            raise ValueError(f"not an active related source file: {path}")
        seen.add(path)
        module = ".".join(relative.with_suffix("").parts)
        package = module.rsplit(".", 1)[0] if path.name != "__init__.py" else module.rsplit(".", 1)[0]
        for parent in path.parents:
            if parent == ROOT:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name("." * node.level + target, package)
                imports = [target] + [target + "." + alias.name for alias in node.names]
            for name in imports:
                candidate = _module_path(name)
                if candidate is not None:
                    pending.append(candidate)
    return sorted({path.relative_to(ROOT).as_posix() for path in seen}
                  | {name for name in CONFIG_FILES if (ROOT / name).is_file()})


def archive_code(output):
    paths = archive_paths()
    identities = {}
    with zipfile.ZipFile(Path(output) / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            path = ROOT / name
            content = path.read_bytes()
            identities[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    for name, content in (("code_commit.txt", commit + "\n"), ("code_status.txt", status)):
        with (Path(output) / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    from .r7_experiment import model_code_digest
    return {"base_commit": commit, "files": identities, "source_tree_sha256": digest(identities),
            "code_zip_sha256": sha256_file(Path(output) / "code.zip"),
            "model_code_sha256": model_code_digest(), "working_tree_modified": bool(status.strip()),
            "archive_scope": "active local import closure, model digest implementation, explicit config; no manifests/tests/outputs/legacy"}


def verify_code(protocol):
    from .r7_experiment import model_code_digest
    code = protocol["code"]
    if sha256_file(Path(protocol["output"]) / "code.zip") != code["code_zip_sha256"]:
        raise ValueError("code archive digest changed")
    for name, expected in code["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".py", ".txt", ".toml"):
            raise ValueError("invalid related source archive member")
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"active related source changed after protocol freeze: {name}")
    if model_code_digest() != code["model_code_sha256"]:
        raise ValueError("model implementation digest changed")
