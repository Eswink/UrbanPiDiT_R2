"""Tests for the checkpoint pruner used to stay inside the artifact cap.

The pruner deletes files, so the properties that matter are the refusals: it must
never remove a selected endpoint, never act on an arm whose endpoint is missing
or unrecorded, and never run without ``--apply``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prune_r7_run_checkpoints import plan_prune  # noqa: E402


def _arm(run: Path, seed: str, arm: str, selected_update: int, updates=(100, 200, 300)):
    arm_dir = run / "training" / seed / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    for update in updates:
        (arm_dir / f"update_{update:07d}.pt").write_bytes(b"x" * 8)
    selected = arm_dir / f"update_{selected_update:07d}.pt"
    (arm_dir / "training_report.json").write_text(
        json.dumps({"selected_checkpoint": str(selected)}), encoding="utf-8")
    return arm_dir


def test_plan_keeps_exactly_the_selected_endpoint(tmp_path):
    run = tmp_path / "run"
    _arm(run, "seed41", "generic", 200)
    keep, remove = plan_prune(run)
    assert [path.name for path in keep] == ["update_0000200.pt"]
    assert sorted(path.name for path in remove) == [
        "update_0000100.pt", "update_0000300.pt"]


def test_plan_covers_every_arm_in_every_seed(tmp_path):
    run = tmp_path / "run"
    for seed in ("seed41", "seed42"):
        for arm, pick in (("a", 100), ("b", 300)):
            _arm(run, seed, arm, pick)
    keep, remove = plan_prune(run)
    assert len(keep) == 4
    assert len(remove) == 8


def test_plan_refuses_when_the_selected_endpoint_is_missing(tmp_path):
    """A run whose recorded endpoint is absent must not be pruned at all: the
    remaining files might be the only copy of a published number's source."""
    run = tmp_path / "run"
    arm_dir = _arm(run, "seed41", "generic", 200)
    (arm_dir / "update_0000200.pt").unlink()
    with pytest.raises(ValueError, match="endpoint is missing"):
        plan_prune(run)


def test_plan_refuses_a_report_without_a_selected_checkpoint(tmp_path):
    run = tmp_path / "run"
    arm_dir = run / "training" / "seed41" / "generic"
    arm_dir.mkdir(parents=True)
    (arm_dir / "update_0000200.pt").write_bytes(b"x")
    (arm_dir / "training_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="selected_checkpoint"):
        plan_prune(run)


def test_plan_refuses_a_run_with_no_reports(tmp_path):
    run = tmp_path / "run"
    (run / "training" / "seed41").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to guess"):
        plan_prune(run)


def test_plan_refuses_a_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        plan_prune(tmp_path / "absent")


def test_dry_run_removes_nothing(tmp_path, capsys):
    from tools.prune_r7_run_checkpoints import main

    run = tmp_path / "run"
    _arm(run, "seed41", "generic", 200)
    argv = sys.argv
    sys.argv = ["prune", "--run", str(run)]
    try:
        main()
    finally:
        sys.argv = argv
    assert sorted(path.name for path in (run / "training/seed41/generic").glob("*.pt")) == [
        "update_0000100.pt", "update_0000200.pt", "update_0000300.pt"]
    assert not (run / "checkpoint_prune_receipt.json").exists()
    assert "dry run" in capsys.readouterr().out


def test_apply_removes_only_the_unselected_and_writes_a_receipt(tmp_path):
    from tools.prune_r7_run_checkpoints import main

    run = tmp_path / "run"
    _arm(run, "seed41", "generic", 200)
    argv = sys.argv
    sys.argv = ["prune", "--run", str(run), "--apply"]
    try:
        main()
    finally:
        sys.argv = argv
    arm_dir = run / "training" / "seed41" / "generic"
    assert [path.name for path in arm_dir.glob("*.pt")] == ["update_0000200.pt"]
    receipt = json.loads((run / "checkpoint_prune_receipt.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in receipt["kept"]] == [
        "training/seed41/generic/update_0000200.pt"]
    assert sorted(Path(item["path"]).name for item in receipt["removed"]) == [
        "update_0000100.pt", "update_0000300.pt"]
    # the receipt records what it destroyed, so the removal is auditable
    assert all(len(item["sha256"]) == 64 for item in receipt["removed"])


def test_apply_preserves_every_non_checkpoint_artifact(tmp_path):
    from tools.prune_r7_run_checkpoints import main

    run = tmp_path / "run"
    arm_dir = _arm(run, "seed41", "generic", 200)
    evaluation = run / "evaluation" / "seed41" / "generic" / "lead_006h"
    evaluation.mkdir(parents=True)
    (evaluation / "rmse.csv").write_text("lead_hours,variable,rmse\n6,t2m,1.0\n",
                                         encoding="utf-8")
    (run / "protocol.json").write_text("{}", encoding="utf-8")
    (run / "ablation_result.json").write_text("{}", encoding="utf-8")
    argv = sys.argv
    sys.argv = ["prune", "--run", str(run), "--apply"]
    try:
        main()
    finally:
        sys.argv = argv
    assert (evaluation / "rmse.csv").is_file()
    assert (run / "protocol.json").is_file()
    assert (run / "ablation_result.json").is_file()
    assert (arm_dir / "training_report.json").is_file()
