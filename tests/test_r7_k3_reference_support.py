"""Explicit synthetic-fixture support; no active conftest replacement or global torch/network patch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pytest
import torch


def denied(*args, **kwargs):
    raise RuntimeError("engineering forbids network/actual checkpoint/GPU")


@pytest.fixture(autouse=True)
def no_actual_runtime(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(torch, "load", denied)
    for name in ("init", "set_device", "synchronize", "manual_seed_all", "empty_cache", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, denied)


def archive_fixture():
    directory = Path(__file__).resolve().parent / "fixtures"
    identity = json.loads((directory / "r7_k3_reference_code_identity.json").read_bytes())
    path = directory / "r7_k3_reference_code.zip"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["code_zip_sha256"]
    return path, identity


@pytest.fixture
def archive_source(tmp_path):
    path, identity = archive_fixture()
    source = tmp_path / "archived_code"
    from r7_k3_reference_identity import extract_archive
    extract_archive(path, identity["files"], source)
    return source


@pytest.fixture
def archived_leaves(monkeypatch, archive_source):
    """Unmodified pinned leaf bytes on synthetic tensors; no actual dataset/model constructor imports."""
    source, modules = archive_source, {}
    for name in ("model", "training", "data"):
        package = types.ModuleType(name)
        package.__path__ = [str(source / name)]
        package.__file__ = str(source / name / "__init__.py")
        monkeypatch.setitem(sys.modules, name, package)
    spacetime = types.ModuleType("model.spacetime_conditioning_r7")
    spacetime.SPACETIME_INPUT_FIELDS = ("latitude", "longitude", "init_utc_hour", "init_day_of_year")
    monkeypatch.setitem(sys.modules, spacetime.__name__, spacetime)
    halting = types.ModuleType("model.r7_halting")
    def positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name)
        return value
    halting.positive_int = positive_int
    halting.forecast_inputs = lambda batch: {k: batch[k] for k in ("coarse_history", "lead_time_hours", *spacetime.SPACETIME_INPUT_FIELDS) if k in batch}
    monkeypatch.setitem(sys.modules, halting.__name__, halting)
    names = ("model.r7_rollout", "training.r7_rollout_metrics", "training.r7_acc", "training.r7_climatology_skill", "training.r7_boundary_metrics", "training.r7_experiment")
    for name in names:
        path = source / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.fixture
def synthetic_sample():
    h, w, channels = 7, 8, 17
    history = torch.arange(2 * channels * h * w, dtype=torch.float32).reshape(2, channels, h, w) / 100
    return {"coarse_history": history, "rollout_targets": history[-1:].clone() + 2,
            "latitude": torch.linspace(20, 50, h), "longitude": torch.linspace(100, 110, w),
            "init_time": "2016-02-17T06:00:00", "valid_times": ["2016-02-17T18:00:00"],
            "lead_time_hours": torch.tensor(6.), "init_utc_hour": torch.tensor(6.), "init_day_of_year": torch.tensor(48.),
            "init_year": torch.tensor(2016.), "future_target": torch.full((channels, h, w), float("nan")),
            "atmos_target": torch.full((channels, h, w), float("nan")),
            "process_targets": torch.full((8,), float("nan")), "future_process_targets": torch.full((8,), float("nan"))}
