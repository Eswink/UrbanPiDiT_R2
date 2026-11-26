"""The headroom gate ``run_stage`` applies to a multi-card protocol.

Split out of ``tests/test_r7_round_stages.py`` when those tests pushed it past the 600-line
hard cap (R-051). The seam is real: everything here is about *which card* the stage gate
probes, which is a different question from the deadline/receipt/bookkeeping tests that file
keeps.

Why this file had to exist at all: ``training.r7_device_selection.protocol_fields`` emits
``gpu_uuids`` (plural) and never the singular ``gpu_uuid``, so every multi-card round frozen
through it lacks the singular key. ``run_stage`` read ``body["gpu_uuid"]`` directly and
raised ``KeyError`` — which is exactly how the G1-B visit-gain reading stage died on
2026-11-26 after six arms had already trained. No existing test could see it because every
fixture in the older file hand-wrote ``gpu_uuid``. A fixture that always supplies the key a
code path reads cannot test what happens when the key is absent.

No GPU, no network, no ``outputs/``: every outside call is injected.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def support():
    return _load("r7_round_stages_gate_under_test", "tools/r7_round_stages.py")


def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


def _multi_card_protocol(output: Path) -> Path:
    """A protocol shaped the way ``protocol_fields`` actually writes one: no ``gpu_uuid``."""
    body = {"protocol_sha256": "a" * 64, "deadline_perf_counter": 10_000.0,
            "per_seed_seconds": 1800.0, "planned_seconds": 5400.0, "hard_cap_seconds": 7200.0,
            "gpu_uuids": ["GPU-second", "GPU-first"],
            "estimated_peak_bytes": 2 ** 31, "headroom_margin_bytes": 2 ** 31}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "protocol.json"
    target.write_text(json.dumps(body), encoding="utf-8")
    return target


def test_run_stage_gates_a_multi_card_protocol_on_its_declared_primary(support, tmp_path):
    """The gate must probe the frozen list's first entry, not a second source of truth.

    The assertion is on *which* card was probed. A test that only checked "did not raise"
    would also pass if the gate silently fell back to the second card, which would admit a
    stage on a card the freeze never designated.
    """
    _multi_card_protocol(tmp_path)

    def runner(argv, cwd=None, env=None, timeout=None):
        if argv[0] == "nvidia-smi":
            # 20,480 MiB free clears the fixture's 4,096 MiB requirement, so the verdict is
            # about which card was probed rather than about capacity.
            return _ok("GPU-second, 20480, 24576\nGPU-first, 20480, 24576\n")
        return _ok()

    record = support.run_stage(tmp_path, "scripts/study_fake.py", "reading", seed=None,
                               runner=runner, clock=lambda: 1000.0)
    assert record["gate"]["passed"] is True, record
    assert record["gate"]["uuid"] == "GPU-second"


def test_run_stage_counterproof_a_protocol_with_no_device_is_refused(support, tmp_path):
    """No declared device at all is an error, never a silent gate on a visible card."""
    _multi_card_protocol(tmp_path)
    target = tmp_path / "protocol.json"
    body = json.loads(target.read_text(encoding="utf-8"))
    body.pop("gpu_uuids")
    target.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(support.RoundError, match="declares no device"):
        support.run_stage(tmp_path, "scripts/study_fake.py", "reading", seed=None,
                          runner=lambda *a: _ok(), clock=lambda: 1000.0)
