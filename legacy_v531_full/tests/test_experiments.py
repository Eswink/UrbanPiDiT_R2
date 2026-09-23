import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.ablation import ABLATION_OVERRIDES, build_ablation_configs
from experiments.collect import collect_results
from experiments.event_eval import evaluate_event_tensor_files
from experiments.export import export_json_to_csv
from experiments.common import load_yaml
from experiments.fair_benchmark import build_fair_benchmark_commands
from experiments.leave_one_city_out import build_leave_one_city_out_configs
from experiments.perturbation import build_perturbation_configs
from experiments.run_graph_diagnostics import build_graph_diagnostics
from experiments.run_mechanism_ablation import build_mechanism_ablation_commands, build_mechanism_ablation_configs
from experiments.run_process_consistency_eval import build_process_consistency_eval
from experiments.run_static_perturbation import (
    build_static_perturbation_commands,
    build_static_perturbation_configs,
    compute_degradation_rows,
    default_static_perturbation_modes,
)
from experiments.smoke_test_v51 import make_smoke_model_cfg, run_smoke
from utils.config_builder import build_model_cfg


BASE_CFG = """
data_root: /tmp/beijing
dynamic_vars: [d2m, sp, t2m, tcc, tp, u10, v10]
static_vars: [landcover, building_surface, buildings, building_volume, population]
include_static: true
broadcast_static: true
k: 4
delta_t: 1
H: 4
W: 4
static_schema:
  categorical: [landcover]
  continuous: [building_surface, buildings, building_volume, population]
  categorical_cardinality:
    landcover: 20
forecast:
  lead_times: [1, 2]
model:
  D: 32
  depth: 2
  heads: 4
  mlp_ratio: 2.0
  dropout: 0.0
  drop_path_rate: 0.0
morphology_graph:
  k: 2
  enable_cache: false
micromet_coupling:
  enabled: true
  mode: pre
  hidden_channels: 8
  graph_cfg:
    k: 2
    enable_cache: false
losses:
  feasibility:
    enabled: true
    lambda: 0.1
  structure:
    enabled: true
    fft_enabled: false
    grad_enabled: false
  process:
    enabled: true
    lambda_rh: 0.1
    lambda_drag: 0.1
    lambda_diurnal: 0.1
train:
  batch_size: 2
  num_workers: 0
ablation:
  use_variable_graph: false
  use_lead_time_conditioning: true
multi_region:
  regions:
    - name: beijing
      data_root: /tmp/beijing
    - name: shanghai
      data_root: /tmp/shanghai
"""


class TestExperiments(unittest.TestCase):
    def test_fair_benchmark_commands_are_dry_run_safe(self):
        commands = build_fair_benchmark_commands("configs/fair_baselines.yaml", "outputs/fair")
        self.assertGreaterEqual(len(commands), 3)
        self.assertIn("baselines.forecast_runner", commands[0].command)

    def test_ablation_and_perturbation_config_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "base.yaml"
            cfg_path.write_text(BASE_CFG, encoding="utf-8")
            ablations = build_ablation_configs(str(cfg_path), str(Path(tmp) / "ab"), names=["base", "combined"])
            self.assertEqual(set(ablations), {"base", "combined"})
            self.assertTrue(Path(ablations["combined"]).exists())
            perturbs = build_perturbation_configs(str(cfg_path), str(Path(tmp) / "pt"), modes=["none", "fill_zero"])
            self.assertEqual(set(perturbs), {"none", "fill_zero"})
            self.assertIn("combined", ABLATION_OVERRIDES)

    def test_leave_one_city_out_config_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "multi.yaml"
            cfg_path.write_text(BASE_CFG, encoding="utf-8")
            configs = build_leave_one_city_out_configs(str(cfg_path), str(Path(tmp) / "loco"))
            self.assertEqual(set(configs), {"leave_beijing_out", "leave_shanghai_out"})
            self.assertTrue(Path(configs["leave_beijing_out"]).exists())

    def test_event_eval_from_tensor_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = torch.tensor([0.1, 0.8, 0.9, 0.2])
            target = torch.tensor([0.0, 1.0, 0.0, 1.0])
            pred_path = Path(tmp) / "pred.pt"
            target_path = Path(tmp) / "target.pt"
            torch.save(pred, pred_path)
            torch.save(target, target_path)
            result = evaluate_event_tensor_files(pred_path, target_path, thresholds=[0.5])
            self.assertAlmostEqual(result["thresholds"]["0.5"]["f1"], 0.5, places=6)

    def test_collect_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            (root / "a.json").write_text(json.dumps({"metric": 1.0}), encoding="utf-8")
            collected = collect_results(root)
            self.assertEqual(len(collected["files"]), 1)
            collected_path = Path(tmp) / "collected.json"
            collected_path.write_text(json.dumps(collected), encoding="utf-8")
            csv_path = Path(tmp) / "out.csv"
            rows = export_json_to_csv(collected_path, csv_path)
            self.assertEqual(rows, 1)
            self.assertTrue(csv_path.exists())

    def test_v51_mechanism_manifest_is_dry_run_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "base.yaml"
            cfg_path.write_text(BASE_CFG, encoding="utf-8")
            configs = build_mechanism_ablation_configs(
                str(cfg_path),
                str(Path(tmp) / "mechanism"),
                names=["v4_base", "micromet_only", "full_v51"],
            )
            commands = build_mechanism_ablation_commands(configs)

            self.assertEqual(set(configs), {"v4_base", "micromet_only", "full_v51"})
            self.assertEqual(len(commands), 3)
            self.assertTrue(all(cmd.command[:2] == ["python", "train.py"] for cmd in commands))
            self.assertNotIn("--execute", commands[0].command)
            self.assertTrue(Path(configs["full_v51"]).exists())

    def test_v51_graph_and_process_diagnostics_are_manifest_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "base.yaml"
            cfg_path.write_text(BASE_CFG, encoding="utf-8")
            graph = build_graph_diagnostics(str(cfg_path), dry_run=True)
            process = build_process_consistency_eval(str(cfg_path), dry_run=True)

            self.assertEqual(graph["protocol"], "v51_graph_diagnostics")
            self.assertTrue(graph["dry_run"])
            self.assertEqual(graph["spatial_hw"], [4, 4])
            self.assertIn("graph_entropy", graph["base_graph"])
            self.assertIn("wind_anisotropy_mean", graph["wind_aware_graph"])
            self.assertEqual(process["protocol"], "v51_process_consistency_eval")
            self.assertTrue(process["dry_run"])
            self.assertTrue(process["enabled"]["process_consistency"])
            self.assertIn("rh_consistency", process["process"])
            self.assertIn("tp_negative_rate", process["violation_rates"])

    def test_v51_smoke_dummy_forward_is_lightweight(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "base.yaml"
            cfg_text = BASE_CFG.replace(
                "  use_lead_time_conditioning: true\n",
                "  use_lead_time_conditioning: true\n  use_micromet_coupling: true\n",
            )
            cfg_path.write_text(cfg_text, encoding="utf-8")
            raw_model_cfg = build_model_cfg(load_yaml(cfg_path))
            smoke_cfg = make_smoke_model_cfg(raw_model_cfg)
            self.assertEqual(smoke_cfg["H"], 4)
            self.assertEqual(smoke_cfg["D"], 32)
            result = run_smoke(str(cfg_path))

            self.assertEqual(result["protocol"], "v51_smoke_forward")
            self.assertTrue(result["finite"])
            self.assertEqual(result["output_shape"], [2, 7, 4, 4])
            self.assertIn("micromet_enabled", result["diagnostics"]["micromet"])

    def test_v51_static_perturbation_manifest_and_degradation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "base.yaml"
            cfg_path.write_text(BASE_CFG, encoding="utf-8")
            modes = default_static_perturbation_modes(["landcover", "building_surface", "population"])
            configs = build_static_perturbation_configs(
                str(cfg_path),
                str(Path(tmp) / "static"),
                modes=["none", "landcover_only", "continuous_only", "remove_population"],
            )
            commands = build_static_perturbation_commands(configs, ckpt="/tmp/model.ckpt")
            rows = compute_degradation_rows(
                {
                    "none": {"RMSE": 2.0, "ACC": 0.8},
                    "landcover_only": {"RMSE": 2.5, "ACC": 0.7},
                }
            )

            self.assertIn("remove_population", modes)
            self.assertEqual(set(configs), {"none", "landcover_only", "continuous_only", "remove_population"})
            self.assertTrue(all(cmd.command[0:2] == ["python", "evaluate.py"] for cmd in commands))
            self.assertIn("/tmp/model.ckpt", commands[0].command)
            self.assertTrue(any(row["mode"] == "landcover_only" and row["metric"] == "RMSE" for row in rows))
            self.assertTrue(Path(configs["remove_population"]).exists())


if __name__ == "__main__":
    unittest.main()