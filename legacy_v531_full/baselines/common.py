from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.loader import (
    BeijingWeatherDataset,
    compute_train_climatology_mean_field,
    compute_train_normalization_stats,
)


@dataclass
class DataConfig:
    data_root: Path
    dynamic_vars: List[str]
    static_vars: List[str]
    static_schema: Dict[str, Any]
    include_static: bool
    broadcast_static: bool
    expose_hour: bool
    k: int
    delta_t: int
    lead_times: Optional[List[int]]
    time_step_hours: float

    train_batch_size: int
    eval_batch_size: int
    num_workers: int


def load_config(path: str | Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _data_section(cfg: Dict) -> Dict:
    return dict(cfg.get("data", {}) or {})


def _get_data_value(cfg: Dict, key: str, default=None):
    data = _data_section(cfg)
    if key in data:
        return data[key]
    return cfg.get(key, default)


def _deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _get_static_schema(cfg: Dict) -> Dict[str, Any]:
    data_schema = dict(_data_section(cfg).get("static_schema", {}) or {})
    root_schema = dict(cfg.get("static_schema", {}) or {})
    return _deep_update(data_schema, root_schema)


def parse_data_config(cfg: Dict) -> DataConfig:
    data_root = Path(_get_data_value(cfg, "data_root")).expanduser()
    dyn = list(_get_data_value(cfg, "dynamic_vars"))
    stat = list(_get_data_value(cfg, "static_vars", []) or [])
    static_schema = _get_static_schema(cfg)
    include_static = bool(_get_data_value(cfg, "include_static", True))
    broadcast_static = bool(_get_data_value(cfg, "broadcast_static", False))
    expose_hour = bool(_get_data_value(cfg, "expose_hour", False))
    k = int(_get_data_value(cfg, "k"))
    delta_t = int(_get_data_value(cfg, "delta_t"))

    forecast_cfg = dict(cfg.get("forecast", {}) or {})
    lead_times = forecast_cfg.get("lead_times", None)
    if lead_times is not None:
        lead_times = [int(x) for x in list(lead_times)]
        # de-dup while keeping order
        seen = set()
        lead_times = [x for x in lead_times if not (x in seen or seen.add(x))]
        if len(lead_times) == 0:
            lead_times = None

    time_step_hours = float(forecast_cfg.get("time_step_hours", 6.0))

    train_cfg = dict(cfg.get("train", {}) or {})
    eval_cfg = dict(cfg.get("eval", {}) or {})
    train_bs = int(train_cfg.get("batch_size", 8))
    eval_bs = int(eval_cfg.get("batch_size", train_bs))
    num_workers = int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 8)))

    return DataConfig(
        data_root=data_root,
        dynamic_vars=dyn,
        static_vars=stat,
        static_schema=static_schema,
        include_static=include_static,
        broadcast_static=broadcast_static,
        expose_hour=expose_hour,
        k=k,
        delta_t=delta_t,
        lead_times=lead_times,
        time_step_hours=time_step_hours,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=num_workers,
    )


def ensure_train_only_stats(data_cfg: DataConfig) -> None:
    """Create train-only mean/std + climatology cache if missing.

    This mirrors :meth:`MetroWeatherDataModule.prepare_data` but does NOT
    require installing pytorch_lightning.
    """

    root = data_cfg.data_root
    mean_path = root / "normalize_mean_train.npz"
    std_path = root / "normalize_std_train.npz"
    smean_path = root / "normalize_static_mean_train.npz"
    sstd_path = root / "normalize_static_std_train.npz"
    clim_path = root / "normalize_clim_train.npz"

    need_norm = not (
        mean_path.exists()
        and std_path.exists()
        and (not data_cfg.include_static or (smean_path.exists() and sstd_path.exists()))
    )
    need_clim = not clim_path.exists()

    if (not need_norm) and (not need_clim):
        return

    if need_norm:
        dyn_mean, dyn_std, stat_mean, stat_std = compute_train_normalization_stats(
            root,
            data_cfg.dynamic_vars,
            data_cfg.static_vars,
            include_static=data_cfg.include_static,
            static_schema=data_cfg.static_schema,
        )
        np.savez(mean_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_mean.items()})
        np.savez(std_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_std.items()})
        if data_cfg.include_static:
            np.savez(smean_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_mean.items()})
            np.savez(sstd_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_std.items()})

    if need_clim:
        clim = compute_train_climatology_mean_field(root, data_cfg.dynamic_vars)
        if len(clim) > 0:
            np.savez(clim_path, **{k: np.asarray(v, dtype=np.float32) for k, v in clim.items()})


def build_datasets(data_cfg: DataConfig) -> Tuple[BeijingWeatherDataset, BeijingWeatherDataset, BeijingWeatherDataset]:
    ensure_train_only_stats(data_cfg)
    train_ds = BeijingWeatherDataset(
        data_cfg.data_root / "train",
        k=data_cfg.k,
        delta_t=data_cfg.delta_t,
        lead_times=data_cfg.lead_times,
        dynamic_vars=data_cfg.dynamic_vars,
        static_vars=data_cfg.static_vars,
        include_static=data_cfg.include_static,
        broadcast_static=data_cfg.broadcast_static,
        normalize_root=data_cfg.data_root,
        static_schema=data_cfg.static_schema,
        expose_hour=data_cfg.expose_hour,
    )
    val_ds = BeijingWeatherDataset(
        data_cfg.data_root / "val",
        k=data_cfg.k,
        delta_t=data_cfg.delta_t,
        lead_times=data_cfg.lead_times,
        dynamic_vars=data_cfg.dynamic_vars,
        static_vars=data_cfg.static_vars,
        include_static=data_cfg.include_static,
        broadcast_static=data_cfg.broadcast_static,
        normalize_root=data_cfg.data_root,
        static_schema=data_cfg.static_schema,
        expose_hour=data_cfg.expose_hour,
    )
    test_ds = BeijingWeatherDataset(
        data_cfg.data_root / "test",
        k=data_cfg.k,
        delta_t=data_cfg.delta_t,
        lead_times=data_cfg.lead_times,
        dynamic_vars=data_cfg.dynamic_vars,
        static_vars=data_cfg.static_vars,
        include_static=data_cfg.include_static,
        broadcast_static=data_cfg.broadcast_static,
        normalize_root=data_cfg.data_root,
        static_schema=data_cfg.static_schema,
        expose_hour=data_cfg.expose_hour,
    )
    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_cfg: DataConfig,
    *,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(data_cfg)
    train_dl = DataLoader(
        train_ds,
        batch_size=data_cfg.train_batch_size,
        shuffle=shuffle_train,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=data_cfg.eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=data_cfg.eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


def lead_steps_to_tag(lead_steps: int, *, time_step_hours: float = 6.0) -> str:
    hours = float(lead_steps) * float(time_step_hours)
    if abs(hours - round(hours)) < 1e-6:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def maybe_limit_rows(arr: np.ndarray, max_rows: Optional[int]) -> np.ndarray:
    if max_rows is None or max_rows <= 0:
        return arr
    return arr[: int(max_rows)]
