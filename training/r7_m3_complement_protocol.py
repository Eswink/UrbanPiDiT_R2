"""Independent, zero-training M3 complement pins; original artifacts stay opaque/read-only."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FORMAT = "r7-m3-validation-complement-protocol-v1"
PLANNED_SECONDS = 1800.0
HARD_CAP_SECONDS = 3600.0
CLEANUP_RESERVE_SECONDS = 10.0
SEEDS = (41, 42)
ARM_NAMES = ("aux_off", "input_aux", "future_draft_aux")
LEADS = (6, 12, 24, 48, 72)
LIMITATIONS = [
    "scientific_claim:false; validation complement only, zero training updates and no significance/SOTA claim",
    "original failed attempt and all its costs remain unchanged; this is not a resume or repair",
    "archived evaluator and selected400 checkpoints are mandatory even if current model bytes happen to match",
    "sealed test manifests/fields are never decoded; original artifact pins are opaque file hashes",
    "two seeds and one winter segment do not establish convergence or generalization",
    "shared GPU neighbors may affect wall time; their metadata are read-only and no measurements are removed",
    "code/data/protocol pinned reproducibility, not asserted bitwise GPU reproducibility",
    "archived M3 climatology skill is already in audited physical units; no second scaling or old CSV rewrite",
    "1800 seconds is a soft plan; only the independent 3600-second whole-round cap can truncate for time",
]
DRIVER_SOURCES = (
    "training/r7_m3_complement_protocol.py", "training/r7_m3_complement_driver.py",
    "training/r7_m3_complement_results.py", "scripts/replay_r7_m3_validation.py",
    "scripts/r7_m3_complement_worker.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def local_path(path):
    requested = Path(path).absolute()
    if "://" in str(path) or any(part.is_symlink() for part in (requested, *requested.parents)):
        raise ValueError("local nonsymlink paths required")
    if requested.name == "test.jsonl":
        raise ValueError("sealed test manifest forbidden")
    return requested.resolve()


def sha256_file(path):
    path = local_path(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with local_path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def preserve_artifact_error(original, additional, *, label):
    if original is None:
        raise additional
    message = f"{label}: {type(additional).__name__}: {additional}"
    original.add_note(message)
    import sys
    try:
        print(message, file=sys.stderr, flush=True)
    except BaseException as exc:
        original.add_note(f"secondary diagnostic output failed: {type(exc).__name__}: {exc}")


def original_jobs():
    trains = [{"phase": "train", "seed": seed, "arm": arm, "lead": None}
              for seed in SEEDS for arm in ARM_NAMES]
    evaluations = [{"phase": "evaluate", "seed": seed, "arm": arm, "lead": lead}
                   for seed in SEEDS for arm in ARM_NAMES for lead in LEADS]
    return trains + evaluations


def successful_original_jobs():
    return original_jobs()[:13]


def planned_jobs():
    """Only the exact missing 23 validation jobs, never any training or extra case."""
    return original_jobs()[13:]


def job_key(job):
    tail = "" if job["lead"] is None else f"_lead{job['lead']:03d}h"
    return f"{job['phase']}_seed{job['seed']}_{job['arm']}{tail}"


def evaluation_dir(output, job):
    return Path(output) / f"seed{job['seed']}" / "evaluation" / job["arm"] / f"lead_{job['lead']:03d}h"


def original_file_names():
    """Static 109-file inventory, including the old failed worker's log/timing."""
    names = ["attempt.json", "code_commit.txt", "code_status.txt", "code.zip", "cpu_profile.json",
             "environment.json", "execution_attempt.json", "prepare_attempt.json", "protocol.json", "run_started.json"]
    for job in original_jobs()[:6]:
        folder = f"seed{job['seed']}/training/{job['arm']}"
        names.append(folder + "/training_report.json")
        names.extend(folder + f"/update_{update:07d}.pt" for update in (100, 200, 300, 400))
    for job in successful_original_jobs():
        names.extend("workers/" + job_key(job) + suffix for suffix in (".json", ".log", ".timing.json"))
        if job["phase"] == "evaluate":
            folder = evaluation_dir(Path("."), job).as_posix()
            names.extend(folder + "/" + name for name in ("rmse.csv", "acc.csv", "climatology_skill.csv", "provenance.json"))
    names.extend("workers/" + job_key(planned_jobs()[0]) + suffix for suffix in (".log", ".timing.json"))
    return sorted(names)


def pin_original_files(output, *, check=lambda: None):
    output = local_path(output)
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("original artifacts may not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    if actual != set(original_file_names()):
        raise ValueError("exact original 109-file inventory required; partial/alternate original refused")
    result = {}
    for name in original_file_names():
        check()
        result[name] = sha256_file(output / name)
    check()
    return result


def verify_original_files(protocol, *, check=lambda: None):
    if pin_original_files(protocol["original_output"], check=check) != protocol["original_file_sha256"]:
        raise ValueError("original artifact pins changed; original failed attempt is immutable")


def _safe_member(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or path.suffix not in (".py", ".txt", ".toml")
            or any("legacy" in part or part in ("tests", "outputs") for part in path.parts)):
        raise ValueError(f"invalid pinned archive member: {name}")
    return path


def verify_original_archive(original_output, original, *, check=lambda: None):
    """Validate the zip itself, exact member set, bytes, source tree and model digest."""
    code = original["code"]
    path = local_path(original_output) / "code.zip"
    if sha256_file(path) != code["code_zip_sha256"] or digest(code["files"]) != code["source_tree_sha256"]:
        raise ValueError("original code zip/source tree pin mismatch")
    if len(code["files"]) != 80:
        raise ValueError("exact original 80 pinned archive members required")
    model_digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(code["files"]) or {item.filename for item in members} != set(code["files"]):
            raise ValueError("archive has missing, duplicate or unpinned members")
        for item in sorted(members, key=lambda value: value.filename):
            check()
            name = item.filename
            member = _safe_member(name)
            mode = item.external_attr >> 16
            if item.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("archive symlink/directory/special member forbidden")
            content = archive.read(item)
            if hashlib.sha256(content).hexdigest() != code["files"][name]:
                raise ValueError(f"original archive member pin mismatch: {name}")
            if member.parts[0] == "model" and member.suffix == ".py":
                model_digest.update(PurePosixPath(*member.parts[1:]).as_posix().encode() + b"\0")
                model_digest.update(content)
    check()
    if model_digest.hexdigest() != code["model_code_sha256"]:
        raise ValueError("original archive model digest mismatch")
    return code


def extract_original_archive(protocol, original, *, check=lambda: None):
    verify_original_archive(protocol["original_output"], original, check=check)
    root = local_path(protocol["output"]) / "archived_code"
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(Path(protocol["original_output"]) / "code.zip") as archive:
        for name, expected in sorted(original["code"]["files"].items()):
            check()
            path = root.joinpath(*_safe_member(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(archive.read(name))
            if sha256_file(path) != expected:
                raise ValueError("extracted archived code pin mismatch")
    check()
    return root


def qualify_original(output):
    """CPU/static eligibility; archive-isolated load_checkpoint/input checks run later."""
    output = local_path(output)
    original = read_json(output / "protocol.json")
    body = {key: value for key, value in original.items() if key != "protocol_sha256"}
    if (original.get("protocol_sha256") != digest(body)
            or original.get("format") != "r7-73-process-supervision-protocol-v1"
            or original.get("output") != str(output) or original.get("jobs") != original_jobs()
            or original.get("scientific_claim") is not False or original.get("test_read") is not False
            or original["shared_controls"]["updates"] != 400 or original["shared_controls"]["reasoning_steps"] != 4
            or original["gpu"]["policy"] != "shared" or original["gpu"]["estimated_peak_mib"] != 342
            or original["gpu"]["headroom_margin_mib"] != 2048):
        raise ValueError("original frozen training protocol/controls mismatch")
    for name in ("attempt.json", "execution_attempt.json"):
        attempt = read_json(output / name)
        if attempt.get("status") != "failed" or attempt.get("jobs_completed") != successful_original_jobs():
            raise ValueError("original failed 6-training/7-validation terminal history required")
    for job in successful_original_jobs():
        entry = read_json(output / "workers" / (job_key(job) + ".json"))
        if (entry.get("job") != job or entry.get("status") != "success"
                or entry.get("protocol_sha256") != original["protocol_sha256"]
                or entry.get("model_code_sha256") != original["code"]["model_code_sha256"]
                or entry.get("data_identity") != original["data"]["data_identity"]
                or entry.get("scientific_claim") is not False or entry.get("test_read") is not False
                or not entry.get("limitations")):
            raise ValueError("original successful exact-job receipt identity mismatch")
        if job["phase"] == "train":
            report = read_json(output / f"seed{job['seed']}" / "training" / job["arm"] / "training_report.json")
            if (entry["updates_run"] != 400 or entry["selected_update"] != 400 or entry["early_stopped"]
                    or report["updates_this_run"] != 400 or report["selected_update"] != 400 or report["early_stopped"]
                    or report["selection_split"] != "val" or report["signature"] != digest(report["contract"])
                    or report["contract"]["process_supervision"] != entry["process_supervision"]):
                raise ValueError("all six exact 400-update selected400 validation checkpoints required")
    return original


def _source_closure():
    pending = [ROOT / name for name in DRIVER_SOURCES]
    seen = set()
    while pending:
        path = local_path(pending.pop())
        if path in seen:
            continue
        relative = path.relative_to(ROOT)
        if not path.is_file() or any("legacy" in part for part in relative.parts):
            raise ValueError(f"missing active complement source: {relative}")
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
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name("." * node.level + target, package)
                names = [target] + [target + "." + alias.name for alias in node.names]
            for name in names:
                if name.split(".")[0] not in ("training", "scripts", "model", "data"):
                    continue
                base = ROOT.joinpath(*name.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
                        break
    return sorted(path.relative_to(ROOT).as_posix() for path in seen)


def archive_driver_code(output):
    files = {}
    with zipfile.ZipFile(Path(output) / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in _source_closure():
            content = (ROOT / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    for name, content in (("code_commit.txt", commit + "\n"), ("code_status.txt", status)):
        with (Path(output) / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return {"source_root": str(ROOT), "base_commit": commit, "files": files,
            "source_tree_sha256": digest(files), "code_zip_sha256": sha256_file(Path(output) / "code.zip"),
            "code_commit_sha256": sha256_file(Path(output) / "code_commit.txt"),
            "code_status_sha256": sha256_file(Path(output) / "code_status.txt"),
            "working_tree_modified": bool(status.strip()),
            "archive_scope": "static complement driver/aggregator local import closure; no outputs/manifests/tests"}


def verify_driver_code(protocol, *, check=lambda: None):
    code, output = protocol["code"], Path(protocol["output"])
    if not set(DRIVER_SOURCES) <= set(code["files"]) or digest(code["files"]) != code["source_tree_sha256"]:
        raise ValueError("complete complement driver/aggregator source pins required")
    for name in ("code.zip", "code_commit.txt", "code_status.txt"):
        check()
        key = {"code.zip": "code_zip_sha256", "code_commit.txt": "code_commit_sha256", "code_status.txt": "code_status_sha256"}[name]
        if sha256_file(output / name) != code[key]:
            raise ValueError("complement code archive/commit/status changed")
    with zipfile.ZipFile(output / "code.zip") as archive:
        if len(archive.namelist()) != len(code["files"]) or set(archive.namelist()) != set(code["files"]):
            raise ValueError("complement code archive contains unpinned members")
        for name, expected in code["files"].items():
            check()
            _safe_member(name)
            if (sha256_file(Path(code["source_root"]) / name) != expected
                    or hashlib.sha256(archive.read(name)).hexdigest() != expected):
                raise ValueError(f"complement source changed after freeze: {name}")
    check()


def monotonic_boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def build_protocol(*, output, original_output, original, original_files, code,
                   round_started_perf_counter, boot_id):
    output, original_output = local_path(output), local_path(original_output)
    body = {"format": PROTOCOL_FORMAT, "output": str(output), "original_output": str(original_output),
            "original_protocol_sha256": original["protocol_sha256"],
            "original_code_zip_sha256": original["code"]["code_zip_sha256"],
            "original_file_sha256": original_files, "jobs": planned_jobs(),
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS, "training_updates": 0,
            "gpu": {"policy": "shared", "uuid": original["gpu"]["uuid"], "estimated_peak_mib": 342,
                    "headroom_margin_mib": 2048, "threads": 4,
                    "gate": "memory.free >= max(frozen342MiB, observed owned reserved peak) + 2048MiB before EVERY spawn",
                    "billing": "continuous monotonic interval before first CUDA spawn through last owned GPU reap; startup/gaps/failures/cleanup included",
                    "neighbor_policy": "read-only metadata; never signal non-owned PIDs; no retries/fallback"},
            "code": code, "scientific_claim": False, "limitations": LIMITATIONS,
            "test_read": False, "frozen_before_any_step": True,
            "round_started_perf_counter": round_started_perf_counter, "monotonic_boot_id": boot_id,
            "whole_clock_scope": "earliest prepare CLI entry through freeze, prepare/run interval, CPU verification/extraction/imports, aggregation and owned cleanup; same boot only",
            "whole_round_cost_reference": str(output / "attempt.json")}
    return validate_protocol({**body, "protocol_sha256": digest(body)})


def validate_protocol(protocol):
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != digest(body):
        raise ValueError("complement protocol digest mismatch")
    if (protocol.get("format") != PROTOCOL_FORMAT or protocol.get("scientific_claim") is not False
            or not protocol.get("limitations") or protocol.get("test_read") is not False
            or protocol.get("frozen_before_any_step") is not True or protocol.get("training_updates") != 0
            or protocol.get("jobs") != planned_jobs() or protocol.get("planned_seconds") != PLANNED_SECONDS
            or protocol.get("hard_cap_seconds") != HARD_CAP_SECONDS
            or protocol.get("cleanup_reserve_seconds") != CLEANUP_RESERVE_SECONDS):
        raise ValueError("frozen zero-training complement jobs/budget/flags changed")
    anchor, boot = protocol.get("round_started_perf_counter"), protocol.get("monotonic_boot_id")
    if (isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or not math.isfinite(anchor)
            or anchor < 0 or not isinstance(boot, str) or not boot):
        raise ValueError("frozen same-boot prepare clock anchor required")
    output, original = local_path(protocol["output"]), local_path(protocol["original_output"])
    if (str(output) != protocol["output"] or str(original) != protocol["original_output"]
            or output.is_relative_to(original) or original.is_relative_to(output)
            or protocol.get("whole_round_cost_reference") != str(output / "attempt.json")):
        raise ValueError("absolute disjoint original/complement output paths required")
    gpu = protocol["gpu"]
    if (gpu.get("policy") != "shared" or gpu.get("estimated_peak_mib") != 342
            or gpu.get("headroom_margin_mib") != 2048 or gpu.get("threads") != 4
            or not isinstance(gpu.get("uuid"), str) or not gpu["uuid"].startswith("GPU-") or "," in gpu["uuid"]):
        raise ValueError("frozen shared UUID/342+2048 headroom contract changed")
    pins = protocol["original_file_sha256"]
    if set(pins) != set(original_file_names()) or len(pins) != 109:
        raise ValueError("all original 109 static opaque file pins required")
    values = [*pins.values(), protocol["original_protocol_sha256"], protocol["original_code_zip_sha256"]]
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values):
        raise ValueError("SHA256 pins required")
    if protocol["original_code_zip_sha256"] != pins["code.zip"]:
        raise ValueError("original archive file pin differs")
    return protocol


def verify_protocol(path):
    protocol = validate_protocol(read_json(path))
    if local_path(path) != Path(protocol["output"]) / "protocol.json":
        raise ValueError("complement protocol path/output mismatch")
    return protocol
