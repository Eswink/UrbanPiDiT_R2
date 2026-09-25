"""Assemble a small, downloadable GPU evidence audit pack (#62).

Collects the GPU bring-up artifacts that live only in the local untracked
`outputs/` tree — per-cell sweep JSONs, protocols, logs and the multi-seed
study — into one write-once pack directory with a SHA256 manifest, the code
identity, the environment facts and the failure/skip records, so a reviewer
can re-derive the published tables without access to this machine.

Excluded by design: large binaries (NetCDF, checkpoints, zarr stores), any
file that looks like a credential, and everything not needed to re-derive
tables. Nothing is copied from the protected data trees and no source artifact
is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCES = (
    "outputs/gpu_sweep_isolated", "outputs/gpu_sweep_isolated_a",
    "outputs/gpu_sweep_stageb", "outputs/gpu_sweep_full",
    "outputs/gpu_sweep_retained",
    "outputs/r7_multiseed_real", "outputs/r7_multiseed_real_v3",
    "outputs/gpu_d3", "outputs/gpu_d4", "outputs/gpu_degrade",
    "outputs/gpu_oom_boundary", "outputs/gpu_oom_isolated",
    "outputs/gpu_oom_probe", "outputs/gpu_resv_check",
)
# Text formats small enough to inline; everything else is listed as excluded.
INCLUDED_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".sh"}
MAX_FILE_BYTES = 2 * 1024 * 1024
CREDENTIAL_MARKERS = (
    "BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "BEGIN EC PRIVATE KEY",
    "BEGIN PRIVATE KEY", "ghp_", "github_pat_", "AKIA",
)
BINARY_SUFFIXES = {".pt", ".nc", ".zarr", ".zips", ".zip", ".npy", ".npz", ".bin"}


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_credential(path):
    try:
        head = Path(path).read_text(encoding="utf-8", errors="ignore")[:65536]
    except OSError:
        return True
    return any(marker in head for marker in CREDENTIAL_MARKERS)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_pack(sources, out, *, model_code_digest=None):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"pack output must not exist: {out}")
    out.mkdir(parents=True)
    manifest_files, excluded, credentials_skipped, failures = [], [], [], []
    environments, protocols = [], []
    for source in sources:
        source = Path(source)
        if not source.is_dir():
            excluded.append({"path": str(source), "reason": "source dir absent"})
            continue
        target_root = out / source.name
        file_count = 0
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if path.suffix in BINARY_SUFFIXES:
                excluded.append({"path": str(path), "reason": "binary artifact"})
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                excluded.append({"path": str(path), "reason": "file above size cap"})
                continue
            if looks_like_credential(path):
                credentials_skipped.append(str(path))
                continue
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            digest = sha256_of(destination)
            manifest_files.append({"path": str(destination.relative_to(out)),
                                   "sha256": digest, "bytes": path.stat().st_size})
            file_count += 1
            if path.name == "sweep.json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                protocol = payload.get("protocol", {})
                if protocol and protocol not in protocols:
                    protocols.append(protocol)
                    environments.append({key: protocol.get(key) for key in
                                         ("gpu_name", "torch_version", "cuda_version",
                                          "dtype", "warmup_steps", "measure_steps")})
                for row in payload.get("rows", []):
                    if row.get("status") != "ok":
                        failures.append({"cell": f"{source.name}/{relative.parent.name}",
                                         "status": row.get("status")})
    environment = None
    for candidate in environments:
        if any(candidate.get(key) for key in ("gpu_name", "torch_version")):
            environment = candidate
            break
    manifest = {
        "format": "r7-gpu-audit-pack-v1",
        "scientific_claim": False,
        "git_commit": git_commit(),
        "model_code_sha256": model_code_digest,
        "environment": environment,
        "sources": [str(s) for s in sources],
        "files": manifest_files,
        "file_count": len(manifest_files) + 1,  # MANIFEST.json itself
        "excluded": excluded,
        "credentials_skipped": credentials_skipped,
        "failed_or_skipped_cells": failures,
        "protocols": protocols,
        "limitations": [
            "per-cell JSONs and logs only; NetCDF/checkpoint binaries stay out of "
            "the pack and are listed under excluded",
            "the pack supports table re-derivation, not model replay",
            "timing rows are short microbenchmarks (see protocol warmup/measure)",
        ],
    }
    manifest_path = out / "MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    # The manifest cannot contain its own hash: hash the final file, record it
    # in the sidecar only, and never rewrite the manifest afterwards.
    manifest_hash = sha256_of(manifest_path)
    (out / "MANIFEST.sha256").write_text(f"{manifest_hash}  MANIFEST.json\n",
                                         encoding="utf-8")
    manifest["manifest_sha256"] = manifest_hash
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build the downloadable GPU evidence audit pack (#62).")
    parser.add_argument("--out", required=True, help="pack directory (must not exist)")
    parser.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES),
                        help="artifact directories to include (default: the GPU set)")
    args = parser.parse_args()
    from training.r7_experiment import model_code_digest

    manifest = build_pack(args.sources, args.out, model_code_digest=model_code_digest())
    print(json.dumps({"out": args.out, "file_count": manifest["file_count"],
                      "excluded": len(manifest["excluded"]),
                      "failed_or_skipped_cells": len(manifest["failed_or_skipped_cells"]),
                      "credentials_skipped": len(manifest["credentials_skipped"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
