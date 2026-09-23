from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.run_v531_reviewer_evidence_checks import build_report
from urbanpidit_version import VERSION


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "urbanpidit_v53_morpho_process.yaml"


def _small_cfg() -> dict[str, Any]:
    cfg = dict(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {})
    cfg["model"] = {
        **dict(cfg.get("model", {}) or {}),
        "D": 32,
        "depth": 1,
        "heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    morpho = dict(cfg.get("morpho_process", {}) or {})
    morpho["proxy_encoder"] = {
        **dict(morpho.get("proxy_encoder", {}) or {}),
        "hidden_channels": [8, 16, 32],
        "use_diurnal_modulation": False,
    }
    morpho["anisotropic_process_graph"] = {
        **dict(morpho.get("anisotropic_process_graph", {}) or {}),
        "k": 2,
        "enable_cache": False,
    }
    cfg["morpho_process"] = morpho
    cfg["micromet_coupling"] = {
        **dict(cfg.get("micromet_coupling", {}) or {}),
        "hidden_channels": 8,
        "interleaved_interval": 1,
        "graph_cfg": {"k": 2, "enable_cache": False},
    }
    return cfg


def test_v531_reviewer_evidence_report_schema() -> None:
    report = build_report(_small_cfg(), config_path=CONFIG_PATH, batch_size=1)

    assert report["version"] == VERSION
    assert report["trained_checkpoint"] is False
    assert report["batch_report"]["target_tensor_present"] is False
    assert report["forward_report"]["prediction_finite"] is True
    assert report["diagnostic_prefix_contract"]["missing_required_prefixes"] == []
    assert "proxy/roughness_proxy_mean" in report["diagnostics"]
    assert "leakage/target_tensor_provided" in report["diagnostics"]