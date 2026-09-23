from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _as_list(value: Optional[Iterable[Any]]) -> List[Any]:
    return list(value or [])


@dataclass(frozen=True)
class RegionSpec:
    name: str
    data_root: Path
    static_schema: Dict[str, Any]
    weight: float = 1.0


def normalize_region_specs(cfg: Mapping[str, Any]) -> List[RegionSpec]:
    multi = dict(cfg.get("multi_region", {}) or {})
    regions = multi.get("regions", cfg.get("regions", []))
    default_schema = dict(cfg.get("static_schema", {}) or {})
    out: List[RegionSpec] = []
    for item in _as_list(regions):
        if isinstance(item, str):
            name = item
            data_root = Path(item)
            schema = default_schema
            weight = 1.0
        else:
            name = str(item.get("name", item.get("city", ""))).strip()
            if not name:
                raise ValueError(f"region name is required: {item}")
            data_root = Path(item.get("data_root", item.get("root", ""))).expanduser()
            if str(data_root) in {"", "."} and "data_root" not in item and "root" not in item:
                raise ValueError(f"data_root is required for region={name}")
            schema = dict(default_schema)
            schema.update(dict(item.get("static_schema", {}) or {}))
            weight = float(item.get("weight", 1.0))
        out.append(RegionSpec(name=name, data_root=data_root, static_schema=schema, weight=weight))
    return out


def leave_one_region_splits(regions: Sequence[RegionSpec | str]) -> Dict[str, Dict[str, List[str]]]:
    names = [r.name if isinstance(r, RegionSpec) else str(r) for r in regions]
    if len(names) < 2:
        raise ValueError("leave-one-region-out requires at least two regions")
    splits: Dict[str, Dict[str, List[str]]] = {}
    for name in names:
        train = [x for x in names if x != name]
        splits[f"leave_{name}_out"] = {"train": train, "val": [name], "test": [name]}
    return splits


def explicit_region_split(cfg: Mapping[str, Any], regions: Sequence[RegionSpec]) -> Dict[str, List[str]]:
    multi = dict(cfg.get("multi_region", {}) or {})
    split = dict(multi.get("split", cfg.get("split", {})) or {})
    names = {r.name for r in regions}
    if not split:
        return {"train": sorted(names), "val": sorted(names), "test": sorted(names)}
    out: Dict[str, List[str]] = {}
    for key in ("train", "val", "test"):
        values = [str(v) for v in _as_list(split.get(key, []))]
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown regions in {key} split: {unknown}")
        out[key] = values
    return out


def region_map(regions: Sequence[RegionSpec]) -> Dict[str, RegionSpec]:
    return {r.name: r for r in regions}


__all__ = ["RegionSpec", "normalize_region_specs", "leave_one_region_splits", "explicit_region_split", "region_map"]