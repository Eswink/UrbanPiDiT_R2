from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def verify_policy(root):
    decision = root / "docs/decisions/0026-shared-gpu-coresidency-policy.md"
    rule = root / "docs/rules/gpu-resources.md"
    required = {
        decision: ("## Context", "## Decision", "## Consequences", "代价", "共驻", "任何信号", "具名授权"),
        rule: ("R-054", "E-216", "2048", "spawn", "禁止", "非本实验", "无全仓机械检查"),
        root / "AGENTS.md": ("默认共驻", "非本实验进程", "决策 0026"),
        root / "docs/decisions/README.md": (decision.name,),
        root / "docs/rules/README.md": (rule.name, "R-054"),
        root / "docs/rules/CHANGELOG.md": ("GPU 默认共驻政策", "E-216"),
        root / "docs/rules/EVIDENCE.md": ("| E-216 |", decision.name),
    }
    for path, fragments in required.items():
        text = path.read_text(encoding="utf-8")
        if not all(fragment in text for fragment in fragments):
            raise ValueError(f"co-residency policy metadata incomplete: {path.name}")


def test_gpu_coresidency_policy_and_indexes_are_bound():
    verify_policy(ROOT)


def test_gpu_coresidency_policy_checker_rejects_missing_signal_boundary(tmp_path):
    paths = ("AGENTS.md", "docs/decisions/0026-shared-gpu-coresidency-policy.md",
             "docs/rules/gpu-resources.md", "docs/decisions/README.md", "docs/rules/README.md",
             "docs/rules/CHANGELOG.md", "docs/rules/EVIDENCE.md")
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith("0026-shared-gpu-coresidency-policy.md"):
            text = text.replace("任何信号", "信号")
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="policy metadata incomplete"):
        verify_policy(tmp_path)
