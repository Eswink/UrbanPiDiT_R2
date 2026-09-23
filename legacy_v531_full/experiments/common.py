from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


@dataclass(frozen=True)
class ExperimentCommand:
    name: str
    command: List[str]
    description: str = ""

    def shell(self) -> str:
        return " ".join(str(x) for x in self.command)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def write_yaml(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(obj), sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in dict(updates or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def set_path(obj: MutableMapping[str, Any], path: str, value: Any) -> None:
    cur: MutableMapping[str, Any] = obj
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise ValueError("empty override path")
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_path_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    out = deepcopy(dict(cfg))
    for path, value in dict(overrides or {}).items():
        set_path(out, str(path), value)
    return out


def command_manifest(commands: Sequence[ExperimentCommand]) -> List[Dict[str, Any]]:
    return [{"name": c.name, "description": c.description, "command": c.command, "shell": c.shell()} for c in commands]


def run_commands(commands: Sequence[ExperimentCommand], *, execute: bool = False) -> List[Dict[str, Any]]:
    results = []
    for cmd in commands:
        item = {"name": cmd.name, "command": cmd.command, "executed": bool(execute)}
        if execute:
            proc = subprocess.run(cmd.command, check=False, text=True, capture_output=True)
            item.update({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        results.append(item)
    return results


def python_module_command(module: str, args: Mapping[str, Any]) -> List[str]:
    cmd = ["python", "-m", module]
    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def write_manifest(commands: Sequence[ExperimentCommand], path: str | Path, *, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    manifest = {"commands": command_manifest(commands)}
    if extra:
        manifest.update(dict(extra))
    save_json(manifest, path)
    return manifest


__all__ = [
    "ExperimentCommand",
    "load_yaml",
    "write_yaml",
    "save_json",
    "deep_update",
    "set_path",
    "apply_path_overrides",
    "command_manifest",
    "run_commands",
    "python_module_command",
    "write_manifest",
]