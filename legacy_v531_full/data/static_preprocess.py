from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .loader import parse_static_schema


def merge_static_schemas(static_vars: Sequence[str], schemas: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    categorical: List[str] = []
    continuous: List[str] = []
    cardinality: Dict[str, int] = {}

    for schema in schemas:
        parsed = parse_static_schema(static_vars, dict(schema or {}))
        for name in parsed["categorical"]:
            if name not in categorical:
                categorical.append(name)
        for name in parsed["continuous"]:
            if name not in continuous:
                continuous.append(name)
        for name, value in parsed["categorical_cardinality"].items():
            cardinality[name] = max(int(cardinality.get(name, 0)), int(value))

    categorical_set = set(categorical)
    continuous = [name for name in static_vars if name in set(continuous) and name not in categorical_set]
    categorical = [name for name in static_vars if name in categorical_set]
    return {
        "categorical": categorical,
        "continuous": continuous,
        "categorical_cardinality": {name: int(cardinality.get(name, 20 if name == "landcover" else 256)) for name in categorical},
    }


def static_schema_for_region(static_vars: Sequence[str], base_schema: Mapping[str, Any], region_schema: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base_schema or {})
    merged.update(dict(region_schema or {}))
    return parse_static_schema(static_vars, merged)


def describe_static_schema(static_vars: Sequence[str], schema: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = parse_static_schema(static_vars, dict(schema or {}))
    return {
        "num_static_vars": len(static_vars),
        "num_continuous": len(parsed["continuous"]),
        "num_categorical": len(parsed["categorical"]),
        "continuous": parsed["continuous"],
        "categorical": parsed["categorical"],
        "categorical_cardinality": parsed["categorical_cardinality"],
    }


__all__ = ["merge_static_schemas", "static_schema_for_region", "describe_static_schema"]