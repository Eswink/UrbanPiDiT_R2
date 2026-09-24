"""Offline tests for the multi-seed trick comparison (#20).

These tests never touch a GPU and never read the network. The comparison module
is imported by path because ``scripts/`` is not an installed package, matching
how the other bring-up scripts are checked.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_r7_gpu_multiseed.py"
DRIVER = ROOT / "scripts" / "sweep_r7_multiseed.sh"


def _module():
    spec = importlib.util.spec_from_file_location("r7_multiseed_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(kind, steps_k, mode, ckpt, seed, allocated, reserved, seconds):
    return {
        "kind": kind, "k": steps_k, "training_mode": mode,
        "activation_checkpointing": ckpt, "seed": seed, "status": "ok",
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "step_seconds": seconds,
    }


def test_cells_are_a_complete_seeded_grid():
    module = _module()
    cells = module.build_cells((41, 42, 43), ("generic",), ("full_bptt", "streamed_truncated"),
                              (0, 1), (1, 4))
    assert len(cells) == 3 * 2 * 2 * 2
    keys = {(c["kind"], c["k"], c["training_mode"], c["activation_checkpointing"], c["seed"])
            for c in cells}
    assert len(keys) == len(cells), "grid must not contain duplicates"
    assert {c["seed"] for c in cells} == {41, 42, 43}
    # Deterministic order, so a re-plan produces the same cell list.
    assert cells == module.build_cells((41, 42, 43), ("generic",),
                                       ("full_bptt", "streamed_truncated"), (0, 1), (1, 4))
    tags = [module.cell_tag(c) for c in cells]
    assert len(set(tags)) == len(tags), "cell tags must be unique filesystem names"


def test_cell_tag_encodes_every_distinguishing_field():
    module = _module()
    left = {"kind": "generic", "k": 4, "training_mode": "full_bptt",
            "activation_checkpointing": False, "seed": 41}
    right = dict(left, seed=42)
    third = dict(left, activation_checkpointing=True)
    assert module.cell_tag(left) != module.cell_tag(right)
    assert module.cell_tag(left) != module.cell_tag(third)
    assert module.cell_tag(left) == "generic_K4_full_bptt_ck0_s41"


def test_paired_deltas_subtract_within_a_seed():
    """The pairing is the point: a seed-level offset must cancel, not inflate."""
    module = _module()
    rows = []
    for seed, offset in ((41, 1000), (42, 5000), (43, 9000)):
        rows.append(_row("generic", 4, "full_bptt", 0, seed, offset + 200, offset + 250, 0.10))
        rows.append(_row("generic", 4, "streamed_truncated", 0, seed, offset + 100, offset + 150, 0.08))
    block = module.paired_deltas(rows, factor="training_mode", baseline_value="full_bptt")
    assert block["groups"] == 3
    for metric in ("peak_allocated_bytes", "peak_reserved_bytes", "step_seconds"):
        entry = block["metrics"][metric]
        assert entry["n"] == 3
        assert entry["sd_delta"] == 0.0, metric
        assert entry["all_same_sign"] is True, metric
    # Streamed is 100 bytes below full BPTT in every seed, so the per-seed delta
    # is exact; the arm means average the three seed offsets (mean 5000).
    assert block["metrics"]["peak_allocated_bytes"]["mean_delta"] == -100.0
    assert block["metrics"]["peak_allocated_bytes"]["arm_mean"] == pytest.approx(5100.0)
    assert block["metrics"]["peak_allocated_bytes"]["baseline_mean"] == pytest.approx(5200.0)
    # `step_seconds` deltas are also seed-independent here, and the constant
    # per-seed offset must not leak into them.
    assert block["metrics"]["step_seconds"]["mean_delta"] == pytest.approx(-0.02)


def test_paired_deltas_report_spread_when_seeds_disagree():
    """Opposite signs across seeds must not be reported as an established effect."""
    module = _module()
    rows = [
        _row("generic", 4, "full_bptt", 0, 41, 1000, 1000, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 41, 1100, 1100, 0.10),
        _row("generic", 4, "full_bptt", 0, 42, 1000, 1000, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 42, 900, 900, 0.10),
    ]
    block = module.paired_deltas(rows, factor="training_mode", baseline_value="full_bptt")
    entry = block["metrics"]["peak_allocated_bytes"]
    assert entry["all_same_sign"] is False
    assert entry["min_delta"] == -100.0 and entry["max_delta"] == 100.0
    assert entry["sd_delta"] > 0


def test_paired_deltas_require_complete_pairs():
    module = _module()
    rows = [
        _row("generic", 4, "full_bptt", 0, 41, 1000, 1000, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 41, 900, 900, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 42, 900, 900, 0.10),
    ]
    with pytest.raises(ValueError, match="incomplete pairs"):
        module.paired_deltas(rows, factor="training_mode", baseline_value="full_bptt")


def test_paired_deltas_reject_duplicate_rows():
    module = _module()
    rows = [
        _row("generic", 4, "full_bptt", 0, 41, 1000, 1000, 0.10),
        _row("generic", 4, "full_bptt", 0, 41, 1000, 1000, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 41, 900, 900, 0.10),
    ]
    with pytest.raises(ValueError, match="duplicate row"):
        module.paired_deltas(rows, factor="training_mode", baseline_value="full_bptt")


def test_paired_deltas_return_none_for_a_missing_metric():
    module = _module()
    rows = [
        _row("generic", 4, "full_bptt", 0, 41, 1000, 1000, 0.10),
        _row("generic", 4, "streamed_truncated", 0, 41, 1000, 1000, 0.10),
    ]
    rows[1]["step_seconds"] = None
    block = module.paired_deltas(rows, factor="training_mode", baseline_value="full_bptt")
    assert block["metrics"]["step_seconds"] is None
    assert block["metrics"]["peak_allocated_bytes"] is not None


def test_summarize_covers_both_factors_per_kind_and_k():
    module = _module()
    rows = []
    for kind in ("generic", "process"):
        for steps_k in (1, 4):
            for ckpt in (False, True):
                for mode in ("full_bptt", "streamed_truncated"):
                    for seed in (41, 42):
                        base = 1000 if mode == "full_bptt" else 900
                        rows.append(_row(kind, steps_k, mode, ckpt, seed,
                                         base + (50 if ckpt else 0), base, 0.1))
    summary = module.summarize(rows)
    combos = {(b["factor"], b["comparison"]) for b in summary}
    assert ("training_mode", "streamed_truncated vs full_bptt") in combos
    assert ("activation_checkpointing", "checkpointing on vs off") in combos
    assert len(summary) == 2 * 2 * 2  # factor x kind x K
    for block in summary:
        assert block["groups"] >= 2


def test_script_never_spawns_a_subprocess():
    """Process isolation belongs to the shell driver; the Python module is pure."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    banned = {"subprocess", "os.system", "popen", "spawn"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            used.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert not (used & banned), f"module must not spawn processes: {used & banned}"


def test_driver_runs_one_cell_per_process_and_aggregates():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'SCRIPT="scripts/compare_r7_gpu_multiseed.py"' in text
    assert text.count('"${SCRIPT}"') == 3, "plan, per-cell and aggregate"
    assert "--cell-out" in text and "--plan" in text and "--aggregate" in text
    assert "ONE CELL PER PROCESS" in text
    # One interpreter per cell: the measurement invocation appears exactly once
    # inside the loop, so no cell shares an interpreter with another.
    loop = text.split("while IFS= read -r CELL; do", 1)[1]
    assert loop.count('--cell "$CELL"') == 1
    assert "--aggregate" in loop, "the loop must be followed by aggregation"
    assert "timeout" in loop, "a stuck cell must not hang the whole sweep"
    assert "skip ${TAG} (exists)" in text, "a resumed sweep must not re-measure cells"


def test_driver_uses_one_shared_flag_set_for_plan_and_cells():
    """The plan and the measured cells must not diverge in measurement flags.

    A first version passed `--bf16` only to the per-cell invocation, so
    protocol.json froze `dtype: fp32` while bf16 was actually measured. That
    silently invalidates the frozen record, so the shared array is asserted
    here rather than trusted.
    """
    text = DRIVER.read_text(encoding="utf-8")
    assert "MEASURE_ARGS" in text, "flags must live in one shared array"
    # Both invocations must expand the same array.
    assert text.count('"${MEASURE_ARGS[@]}"') == 2, "plan and cell must share MEASURE_ARGS"
    assert "--plan --out \"${OUT}\" \"${MEASURE_ARGS[@]}\"" in text
    assert '"${MEASURE_ARGS[@]}" \\' in text
    # bf16 must not be appended at a call site; it belongs to the shared set.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".venv/bin/python") or stripped.startswith("--measure"):
            assert "--bf16" not in stripped, f"bf16 must not be added per call site: {line}"
    # A frozen protocol whose dtype is overridden is exactly the old bug.
    assert "same measurement flags" in text.lower() or "same measurement flags" in text


def test_protocol_records_the_dtype_that_cells_actually_measured():
    module = _module()
    import argparse

    base = dict(out="/tmp/irrelevant", device="cpu", data="synthetic",
                seeds=[41], steps=[1], kinds=["generic"], modes=["full_bptt"],
                ckpt=[0], batch=4, grid=[12, 12], channels=11, dim=32, depth=2,
                heads=4, window=8, latent=16, anchored=4, free=4,
                warmup=1, measure=1, cell=None, cell_out=None,
                max_seconds=600.0, cell_timeout=60.0)
    cells = module.build_cells((41,), ("generic",), ("full_bptt",), (0,), (1,))
    fp32 = module.protocol_payload(argparse.Namespace(**dict(base, bf16=False)), cells)
    bf16 = module.protocol_payload(argparse.Namespace(**dict(base, bf16=True)), cells)
    assert fp32["dtype"] == "fp32"
    assert bf16["dtype"] == "bf16-autocast/fp32-master"
    # No CUDA is required to freeze a protocol: CI is CPU-only.
    assert fp32["gpu_name"] is None and bf16["gpu_name"] is None
    assert fp32["scientific_claim"] is False


def test_plan_freezes_protocol_before_any_cell_and_declares_the_bounds(tmp_path):
    """R-006 mechanism: protocol.json is persisted by --plan, before any step."""
    module = _module()
    args = _arg_namespace(module, tmp_path)
    cells = module.write_plan(args)
    protocol_path = tmp_path / "protocol.json"
    assert protocol_path.is_file()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["frozen_before_any_step"] is True
    assert protocol["scientific_claim"] is False
    assert protocol["seeds"] == list(args.seeds)
    assert protocol["case_count"] == len(cells)
    assert protocol["reserved_isolation"]
    assert any("not a significance test" in line for line in protocol["limitations"])
    assert any("no convergence" in line for line in protocol["limitations"])
    # The header digest must describe the frozen protocol that was written.
    from training.r7_experiment import canonical_digest
    header = json.loads((tmp_path / "cells.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert header["protocol_digest"] == canonical_digest(protocol)


def test_aggregate_omits_the_summary_when_a_cell_is_missing(tmp_path):
    """An incomplete sweep must not publish a comparison table."""
    module = _module()
    args = _arg_namespace(module, tmp_path, seeds=(41, 42))
    cells = module.write_plan(args)
    # Measure only the first cell.
    first = cells[0]
    cell_dir = tmp_path / "cells" / module.cell_tag(first)
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell.json").write_text(json.dumps(
        _row(first["kind"], first["k"], first["training_mode"],
             first["activation_checkpointing"], first["seed"], 1000, 900, 0.1)),
        encoding="utf-8")
    report = module.aggregate(args)
    assert report["complete"] is False
    assert report["summary"] == []
    assert report["missing_cells"] == [module.cell_tag(c) for c in cells[1:]]
    assert report["measured_cells"] == 1
    assert report["protocol_digest"] == json.loads(
        (tmp_path / "cells.jsonl").read_text(encoding="utf-8").splitlines()[0])["protocol_digest"]


def test_aggregate_publishes_a_summary_when_every_cell_is_present(tmp_path):
    module = _module()
    args = _arg_namespace(module, tmp_path, seeds=(41, 42))
    cells = module.write_plan(args)
    for cell in cells:
        cell_dir = tmp_path / "cells" / module.cell_tag(cell)
        cell_dir.mkdir(parents=True)
        base = 1000 if cell["training_mode"] == "full_bptt" else 900
        (cell_dir / "cell.json").write_text(json.dumps(
            _row(cell["kind"], cell["k"], cell["training_mode"],
                 cell["activation_checkpointing"], cell["seed"], base, base, 0.1)),
            encoding="utf-8")
    report = module.aggregate(args)
    assert report["complete"] is True
    assert report["failed_cells"] == []
    assert report["summary"], "a complete sweep must publish comparisons"
    assert report["scientific_claim"] is False
    # Each comparison holds the other factor fixed, so one group per
    # (seed, checkpointing state).
    for block in report["summary"]:
        assert block["groups"] == len(args.seeds) * len(args.ckpt), block["factor"]
        assert block["metrics"]["peak_allocated_bytes"]["n"] == block["groups"]
        assert block["paired_on"] and block["factor"] not in block["paired_on"]
        # A published block must say which model and K it describes.
        assert block["kind"] in args.kinds
        assert block["k"] in args.steps


def test_aggregate_marks_an_oom_cell_as_failed(tmp_path):
    module = _module()
    args = _arg_namespace(module, tmp_path, seeds=(41, 42))
    cells = module.write_plan(args)
    for index, cell in enumerate(cells):
        cell_dir = tmp_path / "cells" / module.cell_tag(cell)
        cell_dir.mkdir(parents=True)
        row = _row(cell["kind"], cell["k"], cell["training_mode"],
                   cell["activation_checkpointing"], cell["seed"], 1000, 900, 0.1)
        if index == 0:
            row["status"] = "oom"
        (cell_dir / "cell.json").write_text(json.dumps(row), encoding="utf-8")
    report = module.aggregate(args)
    assert report["complete"] is False
    assert report["summary"] == []
    assert report["failed_cells"] == [module.cell_tag(cells[0])]


def test_existing_output_directory_is_refused(tmp_path):
    module = _module()
    args = _arg_namespace(module, tmp_path)
    module.write_plan(args)
    with pytest.raises(FileExistsError):
        module.write_plan(args)


def _arg_namespace(module, tmp_path, seeds=(41, 42, 43)):
    import argparse

    args = argparse.Namespace(**{
        "out": str(tmp_path), "device": "cuda", "data": "synthetic",
        "seeds": list(seeds), "steps": [1], "kinds": ["generic"],
        "modes": ["full_bptt", "streamed_truncated"], "ckpt": [0, 1],
        "batch": 4, "grid": [12, 12], "channels": 11, "dim": 32, "depth": 2,
        "heads": 4, "window": 8, "latent": 16, "anchored": 4, "free": 4,
        "warmup": 1, "measure": 1, "bf16": False, "cell": None, "cell_out": None,
        "max_seconds": 600.0, "cell_timeout": 60.0,
    })
    return args
