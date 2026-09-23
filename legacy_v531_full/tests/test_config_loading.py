from pathlib import Path

import yaml

from baselines.forecast_models import build_baselines_from_config
from data.splits import leave_one_region_splits, normalize_region_specs
from experiments.ablation import ABLATION_OVERRIDES
from experiments.fair_benchmark import build_fair_benchmark_manifest
from experiments.leave_one_city_out import build_leave_one_city_out_configs
from utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs
from urbanpidit_version import RELEASE_NAME, VERSION_NAME


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BASELINE_CONFIGS = [
    "persistence.yaml",
    "climatology.yaml",
    "linear_trend.yaml",
    "moving_average.yaml",
    "exp_smoothing.yaml",
    "residual_climatology.yaml",
    "spatial_mean.yaml",
    "local_mean.yaml",
]


def load_config(path: Path) -> dict:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def test_core_configs_parse_with_unified_builders():
    for name in [
        "beijing.yaml",
        "beijing_v4_two_stage.yaml",
        "complete_config.yaml",
        "urbanpidit_v5_canopy.yaml",
        "urbanpidit_v51_micromet.yaml",
        "urbanpidit_v52_micromet_refine.yaml",
        "mechanism_ablation.yaml",
        "mechanism_ablation_v52.yaml",
        "graph_diagnostics.yaml",
        "process_consistency_eval.yaml",
        "static_perturbation.yaml",
    ]:
        cfg = load_config(CONFIG_DIR / name)
        data_kwargs = build_datamodule_kwargs(cfg)
        lit_kwargs = build_litmodule_kwargs(cfg)
        assert data_kwargs["dynamic_vars"]
        assert data_kwargs["k"] > 0
        assert "model_cfg" in lit_kwargs
        assert lit_kwargs["model_cfg"]["in_channels"] == len(data_kwargs["dynamic_vars"])


def test_v51_experiment_configs_keep_manifest_scope_and_hour_gate():
    expected = {
        "mechanism_ablation.yaml": "v51_mechanism_ablation",
        "mechanism_ablation_v52.yaml": "v52_mechanism_ablation",
        "graph_diagnostics.yaml": "v51_graph_diagnostics",
        "process_consistency_eval.yaml": "v51_process_consistency_eval",
        "static_perturbation.yaml": "v52_static_perturbation",
    }
    for name, protocol in expected.items():
        cfg = load_config(CONFIG_DIR / name)
        data_kwargs = build_datamodule_kwargs(cfg)
        lit_kwargs = build_litmodule_kwargs(cfg)

        assert cfg["experiment"]["protocol"] == protocol
        assert cfg["experiment"]["default_dry_run"] is True
        assert data_kwargs["expose_hour"] is True
        assert lit_kwargs["model_cfg"]["use_legacy_urban_graph"] is False


def test_v5_config_enables_only_explicit_new_modules():
    cfg = load_config(CONFIG_DIR / "urbanpidit_v5_canopy.yaml")
    model_cfg = build_litmodule_kwargs(cfg)["model_cfg"]
    assert model_cfg["use_static_morphology_encoder"] is True
    assert model_cfg["use_morphology_graph"] is True
    assert model_cfg["use_wind_aware_graph"] is True
    assert model_cfg["use_dynamic_vg"] is True
    assert model_cfg["use_static_vg"] is True
    assert model_cfg["use_urban_canopy_coupling"] is True
    assert model_cfg["use_legacy_urban_graph"] is False


def test_v51_config_enables_micromet_and_hour_gate():
    cfg = load_config(CONFIG_DIR / "urbanpidit_v51_micromet.yaml")
    built = build_litmodule_kwargs(cfg)
    data_kwargs = build_datamodule_kwargs(cfg)
    model_cfg = built["model_cfg"]

    assert data_kwargs["expose_hour"] is True
    assert model_cfg["use_micromet_coupling"] is True
    assert model_cfg["use_urban_canopy_coupling"] is False
    assert model_cfg["micromet_coupling_cfg"]["enabled"] is True
    assert model_cfg["micromet_coupling_cfg"]["mode"] == "pre"


def test_v52_config_enables_interleaved_micromet_and_nonzero_process():
    cfg = load_config(CONFIG_DIR / "urbanpidit_v52_micromet_refine.yaml")
    built = build_litmodule_kwargs(cfg)
    data_kwargs = build_datamodule_kwargs(cfg)
    model_cfg = built["model_cfg"]
    physics_cfg = built["physics_cfg"]

    assert data_kwargs["expose_hour"] is True
    assert model_cfg["use_micromet_coupling"] is True
    assert model_cfg["use_morphology_graph"] is True
    assert model_cfg["use_wind_aware_graph"] is True
    assert model_cfg["micromet_coupling_cfg"]["enabled"] is True
    assert model_cfg["micromet_coupling_cfg"]["mode"] == "interleaved"
    assert model_cfg["micromet_coupling_cfg"]["interleaved_interval"] == 3
    assert physics_cfg["use_process_consistency"] is True
    assert physics_cfg["lambda_process_rh"] == 0.02
    assert physics_cfg["lambda_process_drag"] == 0.01
    assert physics_cfg["lambda_process_diurnal"] == 0.01


def test_v531_reviewer_evidence_config_uses_version_authority_names():
    cfg = load_config(CONFIG_DIR / "urbanpidit_v531_reviewer_evidence.yaml")
    built = build_litmodule_kwargs(cfg)
    data_kwargs = build_datamodule_kwargs(cfg)
    model_cfg = built["model_cfg"]
    physics_cfg = built["physics_cfg"]

    assert cfg["version"]["name"] == VERSION_NAME
    assert cfg["version"]["release"] == RELEASE_NAME
    assert data_kwargs["expose_hour"] is True
    assert model_cfg["use_process_proxy_encoder"] is True
    assert model_cfg["use_process_adaln"] is True
    assert model_cfg["use_urban_control_branch"] is True
    assert model_cfg["use_anisotropic_process_graph"] is True
    assert model_cfg["use_micromet_token_branches"] is True
    assert model_cfg["use_morphology_residual_head"] is True
    assert model_cfg["use_urban_graph"] is False
    assert physics_cfg["use_process_consistency"] is True
    assert physics_cfg["lambda_process_proxy"] == 0.01


def test_fair_benchmark_config_builds_controlled_baselines(tmp_path):
    cfg_path = CONFIG_DIR / "fair_benchmark.yaml"
    cfg = load_config(cfg_path)
    models = build_baselines_from_config(cfg)
    assert "persistence_dynamic" in models
    assert "static_analog_same_static" in models
    assert models["persistence_dynamic"].static_policy == "dynamic_only"
    assert models["static_analog_same_static"].static_policy == "same_static"

    manifest = build_fair_benchmark_manifest(str(cfg_path), str(tmp_path / "fair"))
    assert manifest["protocol"] == "fair_static_information_benchmark"
    assert len(manifest["commands"]) >= 3


def test_eight_single_baseline_configs_are_loadable():
    baseline_dir = CONFIG_DIR / "baselines"
    for filename in BASELINE_CONFIGS:
        cfg = load_config(baseline_dir / filename)
        models = build_baselines_from_config(cfg)
        assert len(models) == 1
        alias, model = next(iter(models.items()))
        assert alias
        assert model.static_policy == "dynamic_only"
        assert model.dynamic_vars == cfg["dynamic_vars"]


def test_multi_city_config_builds_leave_one_city_splits(tmp_path):
    cfg_path = CONFIG_DIR / "multi_city.yaml"
    cfg = load_config(cfg_path)
    regions = normalize_region_specs(cfg)
    splits = leave_one_region_splits(regions)
    assert set(splits) == {"leave_beijing_out", "leave_shanghai_out", "leave_guangzhou_out"}

    generated = build_leave_one_city_out_configs(str(cfg_path), str(tmp_path / "loco"))
    assert set(generated) == set(splits)
    for path in generated.values():
        assert Path(path).exists()


def test_ablation_config_names_cover_claimed_module_scope():
    expected = {
        "base",
        "static_encoder_only",
        "morphology_graph_only",
        "wind_aware_graph_only",
        "dynamic_vg_only",
        "static_vg_only",
        "urban_canopy_only",
        "combined",
    }
    assert expected.issubset(set(ABLATION_OVERRIDES))