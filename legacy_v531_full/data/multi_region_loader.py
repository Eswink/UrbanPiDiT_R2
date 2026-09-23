from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    import pytorch_lightning as pl
except Exception:
    pl = None  # type: ignore

from .loader import BeijingWeatherDataset, compute_train_climatology_mean_field, compute_train_normalization_stats
from .splits import RegionSpec, explicit_region_split, normalize_region_specs, region_map
from .static_preprocess import merge_static_schemas


def _as_list(value: Optional[Iterable[Any]]) -> List[Any]:
    return list(value or [])


def _with_region(sample: Dict[str, Any], *, name: str, index: int) -> Dict[str, Any]:
    out = dict(sample)
    meta = dict(out.get("meta", {}) or {})
    meta["region"] = name
    meta["region_index"] = int(index)
    out["meta"] = meta
    out["region"] = name
    out["region_index"] = torch.tensor(int(index), dtype=torch.long)
    return out


class RegionTaggedDataset(Dataset):
    def __init__(self, dataset: Dataset, *, region_name: str, region_index: int) -> None:
        self.dataset = dataset
        self.region_name = str(region_name)
        self.region_index = int(region_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return _with_region(self.dataset[idx], name=self.region_name, index=self.region_index)


def _ensure_region_stats(
    region: RegionSpec,
    *,
    dynamic_vars: Sequence[str],
    static_vars: Sequence[str],
    include_static: bool,
    static_schema: Mapping[str, Any],
) -> None:
    root = region.data_root
    mean_path = root / "normalize_mean_train.npz"
    std_path = root / "normalize_std_train.npz"
    smean_path = root / "normalize_static_mean_train.npz"
    sstd_path = root / "normalize_static_std_train.npz"
    clim_path = root / "normalize_clim_train.npz"

    need_norm = not (mean_path.exists() and std_path.exists() and (not include_static or (smean_path.exists() and sstd_path.exists())))
    need_clim = not clim_path.exists()
    if not need_norm and not need_clim:
        return

    import numpy as np

    if need_norm:
        dyn_mean, dyn_std, stat_mean, stat_std = compute_train_normalization_stats(
            root,
            dynamic_vars,
            static_vars,
            include_static=include_static,
            static_schema=dict(static_schema),
        )
        np.savez(mean_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_mean.items()})
        np.savez(std_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_std.items()})
        if include_static:
            np.savez(smean_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_mean.items()})
            np.savez(sstd_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_std.items()})

    if need_clim:
        clim = compute_train_climatology_mean_field(root, dynamic_vars)
        if len(clim) > 0:
            np.savez(clim_path, **{k: np.asarray(v, dtype=np.float32) for k, v in clim.items()})


def build_region_dataset(
    region: RegionSpec,
    *,
    split: str,
    region_index: int,
    k: int,
    delta_t: int,
    lead_times: Optional[Sequence[int]],
    dynamic_vars: Sequence[str],
    static_vars: Sequence[str],
    include_static: bool,
    broadcast_static: bool,
    static_perturb: Optional[Mapping[str, Any]] = None,
    static_schema: Optional[Mapping[str, Any]] = None,
) -> RegionTaggedDataset:
    schema = dict(static_schema or region.static_schema or {})
    ds = BeijingWeatherDataset(
        region.data_root / str(split),
        k=k,
        delta_t=delta_t,
        lead_times=lead_times,
        dynamic_vars=dynamic_vars,
        static_vars=static_vars,
        include_static=include_static,
        broadcast_static=broadcast_static,
        normalize_root=region.data_root,
        static_perturb=dict(static_perturb or {}),
        static_schema=schema,
    )
    return RegionTaggedDataset(ds, region_name=region.name, region_index=region_index)


class MultiRegionWeatherDataModule(pl.LightningDataModule if pl is not None else object):
    def __init__(
        self,
        *,
        regions: Sequence[RegionSpec | Mapping[str, Any]],
        split: Optional[Mapping[str, Sequence[str]]] = None,
        k: int,
        delta_t: int,
        lead_times: Optional[Sequence[int]],
        batch_size: int,
        num_workers: int,
        dynamic_vars: Sequence[str],
        static_vars: Sequence[str],
        include_static: bool = True,
        broadcast_static: bool = True,
        static_perturb: Optional[Mapping[str, Any]] = None,
        static_schema: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if pl is not None:
            super().__init__()
        normalized: List[RegionSpec] = []
        for item in regions:
            if isinstance(item, RegionSpec):
                normalized.append(item)
            else:
                normalized.extend(normalize_region_specs({"multi_region": {"regions": [item]}}))
        self.regions = normalized
        self.region_by_name = region_map(self.regions)
        self.region_indices = {r.name: i for i, r in enumerate(self.regions)}
        self.split = {key: [str(v) for v in values] for key, values in dict(split or {}).items()}
        if not self.split:
            names = [r.name for r in self.regions]
            self.split = {"train": names, "val": names, "test": names}
        self.k = int(k)
        self.delta_t = int(delta_t)
        self.lead_times = [int(x) for x in lead_times] if lead_times is not None else None
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dynamic_vars = list(dynamic_vars)
        self.static_vars = list(static_vars)
        self.include_static = bool(include_static)
        self.broadcast_static = bool(broadcast_static)
        self.static_perturb = dict(static_perturb or {})
        base_schema = dict(static_schema or {})
        region_schemas = [r.static_schema for r in self.regions]
        self.static_schema = merge_static_schemas(self.static_vars, [base_schema, *region_schemas])

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, split: Optional[Mapping[str, Sequence[str]]] = None):
        regions = normalize_region_specs(cfg)
        chosen_split = dict(split or explicit_region_split(cfg, regions))
        forecast = dict(cfg.get("forecast", {}) or {})
        train = dict(cfg.get("train", {}) or {})
        data = dict(cfg.get("data", {}) or {})
        return cls(
            regions=regions,
            split=chosen_split,
            k=int(data.get("k", cfg.get("k"))),
            delta_t=int(data.get("delta_t", cfg.get("delta_t"))),
            lead_times=forecast.get("lead_times", None),
            batch_size=int(train.get("batch_size", 1)),
            num_workers=int(train.get("num_workers", 0)),
            dynamic_vars=list(data.get("dynamic_vars", cfg.get("dynamic_vars", [])) or []),
            static_vars=list(data.get("static_vars", cfg.get("static_vars", [])) or []),
            include_static=bool(data.get("include_static", cfg.get("include_static", True))),
            broadcast_static=bool(data.get("broadcast_static", cfg.get("broadcast_static", True))),
            static_perturb=data.get("static_perturb", cfg.get("static_perturb", None)),
            static_schema=dict(data.get("static_schema", cfg.get("static_schema", {})) or {}),
        )

    def prepare_data(self) -> None:
        for region in self.regions:
            _ensure_region_stats(
                region,
                dynamic_vars=self.dynamic_vars,
                static_vars=self.static_vars,
                include_static=self.include_static,
                static_schema=self.static_schema,
            )

    def _dataset_for_names(self, names: Sequence[str], split_name: str) -> Dataset:
        datasets = []
        for name in names:
            region = self.region_by_name[name]
            datasets.append(
                build_region_dataset(
                    region,
                    split=split_name,
                    region_index=self.region_indices[name],
                    k=self.k,
                    delta_t=self.delta_t,
                    lead_times=self.lead_times,
                    dynamic_vars=self.dynamic_vars,
                    static_vars=self.static_vars,
                    include_static=self.include_static,
                    broadcast_static=self.broadcast_static,
                    static_perturb=self.static_perturb,
                    static_schema=self.static_schema,
                )
            )
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = self._dataset_for_names(self.split.get("train", []), "train")
        self.val_ds = self._dataset_for_names(self.split.get("val", []), "val")
        self.test_ds = self._dataset_for_names(self.split.get("test", []), "test")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


def build_leave_one_region_datamodules(cfg: Mapping[str, Any]) -> Dict[str, MultiRegionWeatherDataModule]:
    from .splits import leave_one_region_splits

    regions = normalize_region_specs(cfg)
    splits = leave_one_region_splits(regions)
    return {name: MultiRegionWeatherDataModule.from_config(cfg, split=split) for name, split in splits.items()}


__all__ = [
    "RegionTaggedDataset",
    "MultiRegionWeatherDataModule",
    "build_region_dataset",
    "build_leave_one_region_datamodules",
]