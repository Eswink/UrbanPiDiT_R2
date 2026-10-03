"""Stdlib guards and frozen scope for the genuine #72 K3 reference sourcebridge."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import socket
import time

SOURCE_BASE = "616b029dce569b92bd08295737981512180a1ad3"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"
ARCHIVE_SHA = "18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95"
ARCHIVE_PROTOCOL_SHA = "404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d"
ARCHIVE_PROTOCOL_FILE = "ebf0070053e382d3d5cbd7bc5ac317c0aaf6ce027790a428aba690cf74c4cc33"
ARCHIVE_TREE_SHA = "e2a4e562a592e6b6f92c7d18a4d0fb236e95ebaf676464fccebe9acbe707f29b"
MODEL_SHA = "11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476"
MODEL_MAP_SHA = "d3e633e12fbc476db54e97298f7345a51ffabfc28465e7201ed1101d47bc8865"
OLD_PROTOCOL_SHA = "58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4"
OLD_PROTOCOL_FILE = "0bdab50ccfed2279f5c8fe2d188b6896d1a3af1272c0c518462dfc6003b2e134"
DATA_SHA = "ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07"
SEEDS = (41, 42)
LEADS = (6, 12, 24, 48, 72)
COUNTS = (22, 21, 19, 15, 11)
REGIONS = ("full", "interior_2", "edge_2")
FORECASTS = ("original_k3", "persistence", "climatology")
PARENTS = {
    41: {"checkpoint_sha256": "e7a9a33f9b690d07aaa6645b10e4b00e2806f6157bf1039592cf3cb347442dd2",
         "signature": "dcbf0a25066822e26d79b4c20f638a6e2c6c888c6dcedceb714e2509c2b2e4a3",
         "report_sha256": "acaee1d7eee4e0cae76f920f782bb83483e1c85b34c2e63ce559f50efe63e051"},
    42: {"checkpoint_sha256": "c2b97f1b20427d7b69d698de3e5ee5a21bdf1a47fede61598b0ae90a56a9db74",
         "signature": "dd1ffba5e098f6102fb522643106fe140b418da893786744a40d807896912e5e",
         "report_sha256": "53ab535c9d32726c941f2ebb13b67f43df224e7332d50601f67aab4460ba154c"},
}
LIMITATIONS = [
    "scientific_claim:false; engineering or reference measurements are not weather-skill acceptance",
    "genuine original #72 selected400 K3 parents; no training, K1/K2/K4 probes, or checkpoint substitution",
    "compatible archived sourcebridge: #72 has no recorded full code.zip/commit; not full historical orbitwise score replication",
    "unchanged old accumulated-lead/365.25 initialization-time calendar; unlike B/C Gregorian calendar; not calendar-matched or architecture-matched paired evidence",
    "existing M2 winter segment, all17 variables, validation only; sealed test is never selected or opened",
    "train-only month/hour climatology and normalization; persistence/climatology share every scored case",
    "equal-case pooled sufficient statistics, no mean case ACC/RMSE and no mixed-unit cross-variable average",
    "CPU engineering fixtures are synthetic and cannot stand in for actual weather or checkpoint qualification",
    "shared GPU neighbor timing is unadjusted; pinned code/data/protocol, GPU bitwise reproducibility not asserted",
]


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("K3 reference is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.socket.connect_ex = refused
    socket.create_connection = refused


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock_identity(started):
    finite_number(started, nonnegative=True)
    info = time.get_clock_info("perf_counter")
    require(info.monotonic and not info.adjustable, "nonadjustable monotonic whole-round clock required")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    require(len(boot) == 36 and all(c in "0123456789abcdef-" for c in boot), "Linux boot identity required; no cross-boot clock reuse")
    return {"earliest_started_perf_counter": started, "boot_id": boot, "implementation": info.implementation,
            "monotonic": True, "adjustable": False,
            "scope": "same-boot earliest CLI entry before imports/prepare; never reset at CUDA phase or reloaded run"}


def validate_clock(identity, started):
    require(identity == clock_identity(identity["earliest_started_perf_counter"]), "frozen monotonic boot identity changed")
    require(started == identity["earliest_started_perf_counter"], "whole earliest frozen clock may not be reset or replaced")


def check_deadline(deadline, *, clock=time.perf_counter):
    if clock() >= deadline:
        raise TimeoutError("K3 whole-round hard deadline exhausted; no retry")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_path(path):
    require("://" not in str(path), "local non-test paths required")
    path = Path(path).absolute()
    require(path.name != "test.jsonl", "local non-test paths required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink paths forbidden")
    require(path == path.resolve(), "canonical absolute paths required")
    return path


def sha256_file(path, *, check=lambda: None):
    value = hashlib.sha256()
    with local_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            check()
            value.update(chunk)
    check()
    return value.hexdigest()


def read_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with local_path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def output_path(path, repo, *, fresh=False):
    path, repo = local_path(path), local_path(repo)
    forbidden = [repo / p for p in ("data", "model", "training", "scripts", "tools", "tests", "legacy_v531_full", "legacy_v6")]
    forbidden += [repo / "outputs" / p for p in ("r7_72_rw_b_subtraction", "r7_73_process_supervision", "r7_m2_segment")]
    require(not any(path.is_relative_to(p) or p.is_relative_to(path) for p in forbidden), "output overlaps source/protected/original input")
    if path.is_relative_to(repo):
        require(path.is_relative_to(repo / "outputs"), "repository writes restricted to fresh outputs")
    if fresh and path.exists():
        raise FileExistsError("new exclusive output required; no retries/repair")
    return path


def preserve_error(original, extra, label):
    if original is None:
        raise extra
    original.add_note(f"{label}: {type(extra).__name__}: {extra}")


def planned_jobs():
    return [{"seed": seed, "lead": lead} for seed in SEEDS for lead in LEADS]


def job_key(job):
    require(job in planned_jobs(), "only exact two-seed/five-single-lead K3 jobs permitted")
    return f"seed{job['seed']}_lead{job['lead']:03d}h"


def finite_number(value, *, nonnegative=False):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value), "finite numeric statistic required")
    require(not nonnegative or value >= 0, "nonnegative statistic required")
    return float(value)
