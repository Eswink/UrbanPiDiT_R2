import tempfile
import unittest
from pathlib import Path

import torch

from data.multi_region_loader import MultiRegionWeatherDataModule, RegionTaggedDataset
from data.splits import RegionSpec, leave_one_region_splits, normalize_region_specs
from data.static_preprocess import describe_static_schema, merge_static_schemas
from run_leave_one_city_out import build_leave_one_city_out_manifest


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"x_ctx": torch.zeros(1, 1, 1), "meta": {"idx": idx}}


class TestMultiRegion(unittest.TestCase):
    def test_leave_one_region_splits(self):
        regions = [RegionSpec("a", Path("/tmp/a"), {}), RegionSpec("b", Path("/tmp/b"), {}), RegionSpec("c", Path("/tmp/c"), {})]
        splits = leave_one_region_splits(regions)
        self.assertEqual(splits["leave_a_out"]["train"], ["b", "c"])
        self.assertEqual(splits["leave_b_out"]["test"], ["b"])
        self.assertEqual(splits["leave_c_out"]["val"], ["c"])

    def test_region_tagged_dataset_adds_region_metadata(self):
        ds = RegionTaggedDataset(_TinyDataset(), region_name="beijing", region_index=3)
        sample = ds[0]
        self.assertEqual(sample["region"], "beijing")
        self.assertEqual(int(sample["region_index"]), 3)
        self.assertEqual(sample["meta"]["region"], "beijing")
        self.assertEqual(sample["meta"]["region_index"], 3)

    def test_static_schema_merge(self):
        static_vars = ["landcover", "building_surface", "population"]
        schema = merge_static_schemas(
            static_vars,
            [
                {"categorical": ["landcover"], "categorical_cardinality": {"landcover": 10}},
                {"categorical": ["landcover"], "categorical_cardinality": {"landcover": 20}},
            ],
        )
        self.assertEqual(schema["categorical"], ["landcover"])
        self.assertEqual(schema["continuous"], ["building_surface", "population"])
        self.assertEqual(schema["categorical_cardinality"]["landcover"], 20)
        desc = describe_static_schema(static_vars, schema)
        self.assertEqual(desc["num_categorical"], 1)
        self.assertEqual(desc["num_continuous"], 2)

    def test_config_normalization_and_manifest_without_data_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "multi_city.yaml"
            cfg_path.write_text(
                """
dynamic_vars: [t2m, u10]
static_vars: [landcover, population]
include_static: true
broadcast_static: true
k: 2
delta_t: 1
static_schema:
  categorical: [landcover]
  categorical_cardinality:
    landcover: 8
forecast:
  lead_times: [1, 2]
train:
  batch_size: 2
  num_workers: 0
multi_region:
  regions:
    - name: beijing
      data_root: /tmp/beijing
    - name: shanghai
      data_root: /tmp/shanghai
      static_schema:
        categorical_cardinality:
          landcover: 16
""",
                encoding="utf-8",
            )
            manifest = build_leave_one_city_out_manifest(cfg_path, validate_data=False)
            self.assertEqual(manifest["num_regions"], 2)
            self.assertIn("leave_beijing_out", manifest["splits"])
            self.assertEqual(manifest["static_schema"]["categorical_cardinality"]["landcover"], 16)

            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            regions = normalize_region_specs(cfg)
            dm = MultiRegionWeatherDataModule.from_config(cfg, split={"train": ["beijing"], "val": ["shanghai"], "test": ["shanghai"]})
            self.assertEqual([r.name for r in regions], ["beijing", "shanghai"])
            self.assertEqual(dm.split["train"], ["beijing"])
            self.assertEqual(dm.static_schema["categorical_cardinality"]["landcover"], 16)


if __name__ == "__main__":
    unittest.main()