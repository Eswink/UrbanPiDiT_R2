"""数据子模块。"""

from .loader import BeijingWeatherDataset, MetroWeatherDataModule
from .multi_region_loader import MultiRegionWeatherDataModule, RegionTaggedDataset, build_leave_one_region_datamodules
from .splits import RegionSpec, explicit_region_split, leave_one_region_splits, normalize_region_specs
from .static_preprocess import describe_static_schema, merge_static_schemas, static_schema_for_region

__all__ = [
    "BeijingWeatherDataset",
    "MetroWeatherDataModule",
    "RegionSpec",
    "RegionTaggedDataset",
    "MultiRegionWeatherDataModule",
    "build_leave_one_region_datamodules",
    "normalize_region_specs",
    "leave_one_region_splits",
    "explicit_region_split",
    "merge_static_schemas",
    "static_schema_for_region",
    "describe_static_schema",
]
