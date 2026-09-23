"""YAML 配置加载、保存和表单映射。"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR

DYNAMIC_VARS = ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]
STATIC_VARS = ["landcover", "building_surface", "buildings", "building_volume", "population"]

CONFIG_SECTIONS = [
    "data_root",
    "dynamic_vars",
    "static_vars",
    "include_static",
    "broadcast_static",
    "k",
    "delta_t",
    "H",
    "W",
    "forecast",
    "model",
    "diffusion",
    "physics",
    "train",
    "inference",
    "ablation",
    "urban_graph",
    "logging",
    "eval",
    "baselines",
]


def list_config_files(config_dir: Path = CONFIG_DIR) -> list[Path]:
    """列出项目 YAML 配置文件。"""

    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("**/*.yaml"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 mapping：{path}")
    return data


def dump_yaml(data: dict[str, Any]) -> str:
    """序列化 YAML。"""

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """保存 YAML 配置。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_yaml(data), encoding="utf-8")


def yaml_text_to_dict(text: str) -> dict[str, Any]:
    """把 YAML 文本转为字典。"""

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是 mapping")
    return data


def config_hash(data: dict[str, Any]) -> str:
    """生成配置内容哈希。"""

    payload = dump_yaml(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def deep_get(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """按点分路径读取嵌套值。"""

    cur: Any = data
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """按点分路径写入嵌套值。"""

    cur = data
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return data


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """将嵌套字典展平成点分键。"""

    out: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def nested_from_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """从点分键恢复嵌套字典。"""

    data: dict[str, Any] = {}
    for key, value in flat.items():
        deep_set(data, key, value)
    return data


def normalize_save_name(name: str) -> str:
    """规范化配置文件名。"""

    cleaned = name.strip().replace(" ", "_")
    if not cleaned.endswith(".yaml"):
        cleaned += ".yaml"
    return cleaned


def build_edited_config(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """在原配置上应用点分路径更新。"""

    data = copy.deepcopy(base)
    for key, value in updates.items():
        deep_set(data, key, value)
    return data


def validate_config(data: dict[str, Any]) -> list[str]:
    """返回配置风险提示。"""

    warnings: list[str] = []
    required = ["data_root", "dynamic_vars", "k", "delta_t", "model", "diffusion", "train"]
    for key in required:
        if key not in data:
            warnings.append(f"缺少关键配置：{key}")
    dynamic_vars = data.get("dynamic_vars", [])
    if not isinstance(dynamic_vars, list) or len(dynamic_vars) == 0:
        warnings.append("dynamic_vars 不能为空")
    model = data.get("model", {}) or {}
    if int(model.get("D", 0) or 0) <= 0:
        warnings.append("model.D 必须为正数")
    if int(model.get("heads", 0) or 0) <= 0:
        warnings.append("model.heads 必须为正数")
    train = data.get("train", {}) or {}
    if float(train.get("lr", 0.0) or 0.0) <= 0:
        warnings.append("train.lr 必须为正数")
    if int(train.get("batch_size", 0) or 0) <= 0:
        warnings.append("train.batch_size 必须为正数")
    return warnings


def relative_to_project(path: str | Path, project_root: Path) -> str:
    """尽量显示相对路径。"""

    p = Path(path)
    try:
        return str(p.relative_to(project_root))
    except ValueError:
        return str(p)