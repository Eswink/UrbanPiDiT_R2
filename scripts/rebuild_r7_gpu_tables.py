"""Re-derive the GPU memory/timing/multi-seed tables from the audit pack only (#62).

The pack (built by `build_r7_gpu_audit_pack.py`) is the single input: this
script never touches `outputs/` or any raw artifact, so a reviewer can run it
from the pack alone. It re-aggregates the per-cell sufficient statistics into
the memory and timing tables and the three-seed comparison, and validates that
every cell's recorded metadata (dtype, shape, mode, K, batch, model config)
is internally consistent and consistent across cells that a table row joins.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Metadata keys that must agree inside one cell and across cells joined into
# one table row; anything else makes the row "mixed" and it is flagged. Keys
# the historical artifacts never recorded (heads, window) are absent here on
# purpose: nothing can be validated against a field that does not exist.
CELL_CONTRACT_KEYS = ("batch_size", "grid", "in_channels", "dim", "depth",
                      "patch_size", "dtype")


def iter_sweep_rows(pack):
    """Yield (cell name, row, protocol) from every sweep.json inside the pack."""
    for path in sorted(Path(pack).rglob("sweep.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        cell = path.parent.name
        for row in payload.get("rows", []):
            yield f"{path.parent.parent.name}/{cell}", row, protocol


def aggregate_sweeps(pack):
    """Memory/timing means per (kind, K, mode, checkpointing, full contract).

    The metadata contract is part of the group key, so cells recorded with
    different dtype/shape/config are never averaged into one row; a changed
    contract simply forms its own row instead of silently joining another.
    """
    grouped = {}
    problems = []
    for cell, row, protocol in iter_sweep_rows(pack):
        if row.get("status") != "ok":
            problems.append({"cell": cell, "problem": f"status={row.get('status')!r}"})
            continue
        missing = [key for key in CELL_CONTRACT_KEYS if key not in row]
        if missing:
            problems.append({"cell": cell,
                             "problem": f"missing metadata {sorted(missing)}"})
            continue
        protocol_dtype = protocol.get("dtype")
        if protocol_dtype and row.get("dtype") != protocol_dtype:
            problems.append({"cell": cell,
                             "problem": f"dtype {row.get('dtype')!r} differs from "
                                        f"protocol {protocol_dtype!r}"})
        key = (row.get("kind"), row.get("k"), row.get("training_mode"),
               bool(row.get("activation_checkpointing")),
               tuple((col, json.dumps(row.get(col), sort_keys=True))
                     for col in CELL_CONTRACT_KEYS))
        grouped.setdefault(key, []).append((cell, row))
    tables = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        kind, k, mode, ckpt, contract_pairs = key
        entries = grouped[key]
        contract = {col: json.loads(value) for col, value in contract_pairs}
        peaks_alloc = [row["peak_allocated_bytes"] / 2**20 for _, row in entries
                       if isinstance(row.get("peak_allocated_bytes"), (int, float))]
        peaks_reserved = [row["peak_reserved_bytes"] / 2**20 for _, row in entries
                          if isinstance(row.get("peak_reserved_bytes"), (int, float))]
        steps = [row["step_seconds"] * 1000 for _, row in entries
                 if isinstance(row.get("step_seconds"), (int, float))]
        tables.append({
            "kind": kind, "k": k, "training_mode": mode,
            "activation_checkpointing": ckpt,
            "cells": len(entries), "cell_names": sorted(cell for cell, _ in entries),
            "peak_allocated_mib_mean": round(statistics.fmean(peaks_alloc), 1)
            if peaks_alloc else None,
            "peak_reserved_mib_mean": round(statistics.fmean(peaks_reserved), 1)
            if peaks_reserved else None,
            "step_ms_mean": round(statistics.fmean(steps), 1) if steps else None,
            "step_ms_values": [round(value, 1) for value in steps],
            "contract": contract,
        })
    return tables, problems


def aggregate_multiseed(pack):
    """Three-seed paired comparison from cells.jsonl + multiseed.json, if present."""
    index = None
    for path in sorted(Path(pack).rglob("cells.jsonl")):
        index = path
        break
    if index is None:
        return None, ["no cells.jsonl in pack; the three-seed table cannot be rebuilt"]
    entries = [json.loads(line) for line in
               index.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = {}
    for entry in entries:
        key = (entry.get("kind"), entry.get("k"), entry.get("training_mode"),
               bool(entry.get("activation_checkpointing")))
        grouped.setdefault(key, []).append(entry.get("seed"))
    rows = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        kind, k, mode, ckpt = key
        seeds = sorted(seed for seed in grouped[key] if seed is not None)
        rows.append({"kind": kind, "k": k, "training_mode": mode,
                     "activation_checkpointing": ckpt, "seeds": seeds,
                     "seed_count": len(seeds)})
    notes = []
    counts = {row["seed_count"] for row in rows}
    if counts and max(counts) < 3:
        notes.append("fewer than three seeds present; directional agreement is "
                     "descriptive stability, not an established result")
    return rows, notes


def rebuild(pack, out):
    """Write the regenerated tables and the metadata validation report."""
    pack = Path(pack)
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"table output must not exist: {out}")
    manifest_path = pack / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"not an audit pack: {manifest_path} is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables, problems = aggregate_sweeps(pack)
    multiseed, notes = aggregate_multiseed(pack)
    report = {
        "format": "r7-gpu-tables-rebuilt-v1",
        "scientific_claim": False,
        "source_pack": {"path": str(pack), "git_commit": manifest.get("git_commit"),
                        "model_code_sha256": manifest.get("model_code_sha256"),
                        "environment": manifest.get("environment")},
        "timing_scope": ("short microbenchmark: protocol warmup/measure counts "
                         "bound every timing row; no long-training throughput "
                         "may be extrapolated from them"),
        "semantics_note": ("full_bptt, retained_truncated and streamed_truncated "
                           "have distinct gradient semantics and are never "
                           "claimed equivalent; timing comparisons hold the "
                           "other switches fixed per row"),
        "memory_timing_tables": tables,
        "multiseed_cells": multiseed,
        "metadata_problems": problems,
        "metadata_consistent": not problems,
        "notes": notes,
    }
    out.mkdir(parents=True)
    lines = ["# Rebuilt GPU tables (audit pack)", "",
             f"Source pack commit: `{manifest.get('git_commit')}`; "
             f"environment: `{manifest.get('environment')}`.", ""]
    for table in tables:
        lines.append(
            f"| {table['kind']} | K={table['k']} | {table['training_mode']} | "
            f"ckpt={int(table['activation_checkpointing'])} | "
            f"{table['peak_allocated_mib_mean']} | {table['peak_reserved_mib_mean']} | "
            f"{table['step_ms_mean']} | {table['cells']} |")
    lines.insert(4, "| kind | K | mode | ckpt | peak alloc MiB | peak reserved MiB | "
                     "step ms | cells |")
    lines.insert(5, "| --- | --- | --- | --- | --- | --- | --- | --- |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "tables.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild GPU tables from an audit pack alone (#62).")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = rebuild(args.pack, args.out)
    print(json.dumps({"metadata_consistent": report["metadata_consistent"],
                      "metadata_problems": len(report["metadata_problems"]),
                      "table_rows": len(report["memory_timing_tables"]),
                      "multiseed_rows": (len(report["multiseed_cells"])
                                         if report["multiseed_cells"] else 0)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
