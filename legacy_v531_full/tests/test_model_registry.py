from __future__ import annotations

from pathlib import Path

import yaml

from models.model_registry import (
    build_model_from_config,
    build_model_v53_from_config,
    build_model_v531_from_config,
    dropped_model_kwargs,
    filter_model_kwargs,
)
from models.urban_pidit import UrbanPiDiT
from utils.config_builder import build_model_cfg


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_config(name: str) -> dict:
    return dict(yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) or {})


def _small_cfg(cfg: dict) -> dict:
    out = dict(cfg)
    out.update({"D": 32, "depth": 1, "heads": 4, "dropout": 0.0, "drop_path_rate": 0.0})
    return out


def test_registry_drops_legacy_and_future_keys() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v5_canopy.yaml")))
    cfg["unsupported_future_key"] = 1

    dropped = dropped_model_kwargs(cfg)
    filtered = filter_model_kwargs(cfg)

    assert "use_legacy_urban_graph" in dropped
    assert "unsupported_future_key" in dropped
    assert "use_legacy_urban_graph" not in filtered
    assert "unsupported_future_key" not in filtered
    assert "micromet_coupling_cfg" in filtered


def test_registry_builds_v4_legacy_model() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("beijing_v4_two_stage.yaml")))
    model = build_model_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_urban_graph is True
    assert model.use_morphology_graph is False


def test_registry_builds_v5_model_without_constructor_error() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v5_canopy.yaml")))
    model = build_model_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_static_morphology_encoder is True
    assert model.use_morphology_graph is True
    assert model.use_wind_aware_graph is True
    assert model.use_urban_canopy_coupling is True


def test_registry_builds_v51_micromet_model_without_constructor_error() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v51_micromet.yaml")))
    model = build_model_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_static_morphology_encoder is True
    assert model.use_morphology_graph is True
    assert model.use_wind_aware_graph is True
    assert model.use_urban_canopy_coupling is False
    assert model.use_micromet_coupling is True


def test_registry_builds_v52_micromet_refine_model_without_constructor_error() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v52_micromet_refine.yaml")))
    model = build_model_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_static_morphology_encoder is True
    assert model.use_morphology_graph is True
    assert model.use_wind_aware_graph is True
    assert model.use_micromet_coupling is True
    assert model.micromet_coupling_mode == "interleaved"


def test_registry_builds_v53_morpho_process_model_without_constructor_error() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v53_morpho_process.yaml")))
    model = build_model_v53_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_process_proxy_encoder is True
    assert model.use_process_adaln is True
    assert model.use_urban_control_branch is True
    assert model.use_anisotropic_process_graph is True
    assert model.use_morphology_residual_head is True


def test_registry_builds_v531_reviewer_evidence_model_without_constructor_error() -> None:
    cfg = _small_cfg(build_model_cfg(load_config("urbanpidit_v531_reviewer_evidence.yaml")))
    model = build_model_v531_from_config(cfg)

    assert isinstance(model, UrbanPiDiT)
    assert model.use_process_proxy_encoder is True
    assert model.use_process_adaln is True
    assert model.use_urban_control_branch is True
    assert model.use_anisotropic_process_graph is True
    assert model.use_micromet_token_branches is True
    assert model.use_morphology_residual_head is True
    assert model.use_urban_graph is False


def test_registry_builds_v51_experiment_configs_without_constructor_error() -> None:
    for name in [
        "mechanism_ablation.yaml",
        "mechanism_ablation_v52.yaml",
        "graph_diagnostics.yaml",
        "process_consistency_eval.yaml",
        "static_perturbation.yaml",
    ]:
        cfg = _small_cfg(build_model_cfg(load_config(name)))
        model = build_model_from_config(cfg)

        assert isinstance(model, UrbanPiDiT)
        assert model.use_urban_graph is False
