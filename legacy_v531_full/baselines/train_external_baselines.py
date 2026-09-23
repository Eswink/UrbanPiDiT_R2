from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from .common import build_dataloaders, load_config, parse_data_config, save_json
from .forecast_models import build_forecast_baseline
from .forecast_runner import evaluate_forecast_baseline, model_complexity_summary
from .external_models.urban_loader_adapter import move_batch_to_device


try:  # TensorBoard is optional in lightweight CI environments.
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except Exception:  # pragma: no cover - exercised when tensorboard is not installed.
    _TorchSummaryWriter = None


class JsonlSummaryWriter:
    """Tiny fallback writer used when tensorboard is not installed.

    It preserves resumable scalar logs in a JSONL file so that training does not
    fail on minimal machines.  Install ``tensorboard`` to also create native
    TensorBoard event files.  The public API mirrors the subset of SummaryWriter
    used by this training script.
    """

    def __init__(self, log_dir: str | Path, *, purge_step: Optional[int] = None, flush_secs: int = 30, **_: Any) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "scalars.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.purge_step = purge_step
        if purge_step is not None:
            self._fh.write(json.dumps({"event": "resume", "purge_step": int(purge_step), "time": time.time()}) + "\n")
            self._fh.flush()

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None) -> None:
        self._fh.write(json.dumps({
            "tag": str(tag),
            "value": float(scalar_value),
            "global_step": None if global_step is None else int(global_step),
            "time": time.time(),
        }, ensure_ascii=False) + "\n")

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        self._fh.write(json.dumps({
            "tag": str(tag),
            "text": str(text_string),
            "global_step": None if global_step is None else int(global_step),
            "time": time.time(),
        }, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class NullSummaryWriter:
    def add_scalar(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        return None

    def add_text(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        return None

    def flush(self) -> None:  # pragma: no cover - trivial
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None


def _target(batch: Dict, device: torch.device) -> torch.Tensor:
    if "y" in batch:
        y = batch["y"]
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        return y.to(device=device, dtype=torch.float32)
    y = batch["x0"]
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    return y.to(device=device, dtype=torch.float32).unsqueeze(1)


def _as_int_list(value: Any, default: Optional[Iterable[int]] = None) -> List[int]:
    if value is None:
        return [int(x) for x in list(default or [])]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            values = [int(value.detach().cpu().item())]
        elif value.ndim >= 2:
            values = [int(x) for x in value.detach().cpu().reshape(value.shape[0], -1)[0].tolist()]
        else:
            values = [int(x) for x in value.detach().cpu().flatten().tolist()]
    elif isinstance(value, (list, tuple)):
        values = [int(x) for x in value]
    else:
        values = [int(value)]
    return values or [int(x) for x in list(default or [])]


def _target_for_leads(batch: Dict, device: torch.device, lead_times: Sequence[int]) -> torch.Tensor:
    target = _target(batch, device)
    if target.ndim == 4:
        target = target.unsqueeze(1)
    if target.ndim != 5:
        raise ValueError(f"target must be [B,L,C,H,W] or [B,C,H,W], got shape={tuple(target.shape)}")
    requested = [int(x) for x in list(lead_times)]
    if len(requested) == target.shape[1]:
        return target
    batch_leads = _as_int_list(batch.get("lead_times", None), default=range(1, target.shape[1] + 1))
    if target.shape[1] == 1:
        return target.expand(-1, len(requested), -1, -1, -1).contiguous()
    indices = []
    for lead in requested:
        if lead not in batch_leads:
            raise ValueError(f"Requested lead {lead} is unavailable in batch lead_times={batch_leads}")
        idx = batch_leads.index(lead)
        if idx >= target.shape[1]:
            raise ValueError(f"Batch lead index {idx} is out of target range shape={tuple(target.shape)}")
        indices.append(idx)
    return target[:, indices]


def _protocol_value(cfg: Dict, item: Dict, key: str, default: Any = None) -> Any:
    training = dict(cfg.get("external_baseline_training", {}) or {})
    forecast = dict(cfg.get("forecast", {}) or {})
    params = dict(item.get("params", {}) or {})
    if key in item:
        return item[key]
    if key in params:
        return params[key]
    if key in training:
        return training[key]
    return forecast.get(key, default)


def _training_protocol_config(cfg: Dict, item: Dict, model: torch.nn.Module, default_eval_leads: Sequence[int]) -> Dict[str, Any]:
    training = dict(cfg.get("external_baseline_training", {}) or {})
    forecast = dict(cfg.get("forecast", {}) or {})
    name = str(item.get("name", getattr(getattr(model, "spec", None), "name", ""))).lower()
    forecast_protocol = str(getattr(model, "forecast_protocol", _protocol_value(cfg, item, "forecast_protocol", "direct")))
    one_step = int(_protocol_value(cfg, item, "one_step_lead", getattr(model, "one_step_lead", 1)))
    eval_leads = _as_int_list(
        item.get("eval_lead_times", training.get("eval_lead_times", forecast.get("eval_lead_times", default_eval_leads))),
        default=default_eval_leads,
    )
    single_step_leads = _as_int_list(
        item.get("train_lead_times", training.get("train_lead_times", [one_step])),
        default=[one_step],
    )
    rollout_loss_leads = _as_int_list(
        item.get("rollout_loss_lead_times", training.get("rollout_loss_lead_times", eval_leads)),
        default=eval_leads,
    )
    train_protocol = str(item.get("train_protocol", training.get("train_protocol", "auto"))).strip().lower()
    if train_protocol in {"auto", "auto_by_model"}:
        if hasattr(model, "training_loss") and name in {"gencast", "faithful_gencast"}:
            train_protocol = "diffusion"
        elif hasattr(model, "training_loss") and name in {"corrdiff", "faithful_corrdiff"}:
            stage = str(getattr(model, "training_stage", "regression")).lower()
            train_protocol = "diffusion" if stage == "diffusion" else "direct"
        elif forecast_protocol == "official_rollout" and name in {"graphcast", "faithful_graphcast", "faithful_fourcastnet"}:
            train_protocol = "rollout_loss"
        elif forecast_protocol == "official_rollout":
            train_protocol = "one_step"
        else:
            train_protocol = "direct"
    aliases = {
        "single_step": "one_step",
        "6h": "one_step",
        "one_step_mse": "one_step",
        "autoregressive": "rollout_loss",
        "autoregressive_rollout": "rollout_loss",
        "gencast_diffusion": "diffusion",
        "edm": "diffusion",
    }
    train_protocol = aliases.get(train_protocol, train_protocol)
    if train_protocol == "rollout_loss":
        train_leads = rollout_loss_leads
    elif train_protocol == "one_step":
        train_leads = [one_step]
    elif train_protocol == "diffusion":
        train_leads = single_step_leads
    else:
        train_leads = single_step_leads if train_protocol != "direct" else eval_leads
    if train_protocol not in {"direct", "one_step", "rollout_loss", "diffusion"}:
        raise ValueError(f"Unknown train_protocol={train_protocol!r}; expected direct/one_step/rollout_loss/diffusion")
    return {
        "forecast_protocol": forecast_protocol,
        "train_protocol": train_protocol,
        "one_step_lead": one_step,
        "train_lead_times": train_leads,
        "eval_lead_times": eval_leads,
        "rollout_loss_lead_times": rollout_loss_leads,
    }


def _training_loss_for_batch(model: torch.nn.Module, batch: Dict, device: torch.device, protocol_cfg: Dict[str, Any]) -> torch.Tensor:
    train_protocol = str(protocol_cfg.get("train_protocol", "direct"))
    train_leads = [int(x) for x in protocol_cfg.get("train_lead_times", [])]
    if train_protocol == "diffusion" and hasattr(model, "training_loss"):
        return model.training_loss(batch, lead_times=train_leads)
    pred = model(batch, lead_times=train_leads)
    target = _target_for_leads(batch, device, train_leads)
    return F.mse_loss(pred, target)


def _validation_losses_for_batch(model: torch.nn.Module, batch: Dict, device: torch.device, protocol_cfg: Dict[str, Any]) -> Dict[str, float]:
    one_step = int(protocol_cfg.get("one_step_lead", 1))
    eval_leads = [int(x) for x in protocol_cfg.get("eval_lead_times", [one_step])]
    pred_eval = model(batch, lead_times=eval_leads)
    target_eval = _target_for_leads(batch, device, eval_leads)
    losses = {"val_rollout_mse": float(F.mse_loss(pred_eval, target_eval).detach().cpu())}
    if one_step in eval_leads:
        idx = eval_leads.index(one_step)
        losses["val_6h_mse"] = float(F.mse_loss(pred_eval[:, idx : idx + 1], target_eval[:, idx : idx + 1]).detach().cpu())
    else:
        pred_one = model(batch, lead_times=[one_step])
        target_one = _target_for_leads(batch, device, [one_step])
        losses["val_6h_mse"] = float(F.mse_loss(pred_one, target_one).detach().cpu())
    train_protocol = str(protocol_cfg.get("train_protocol", "direct"))
    losses["val_mse"] = losses["val_rollout_mse"] if train_protocol == "rollout_loss" else losses["val_6h_mse"]
    return losses


def _iter_trainable_specs(cfg: Dict) -> List[Dict]:
    baseline_cfg = dict(cfg.get("baselines", {}) or {})
    items = baseline_cfg.get("models", []) or []
    selected = []
    for item in items:
        if isinstance(item, str):
            item = {"name": item, "alias": item}
        family = str(item.get("family", "")).lower()
        name = str(item.get("name", "")).lower()
        if family in {"external_weather_baseline", "faithful_weather_baseline"} or name in {
            "fourcastnet",
            "graphcast",
            "gencast",
            "corrdiff",
            "faithful_fourcastnet",
            "faithful_graphcast",
            "faithful_gencast",
            "faithful_corrdiff",
        }:
            selected.append(dict(item))
    return selected


def _make_model(cfg: Dict, item: Dict):
    data = dict(cfg.get("data", {}) or {})
    forecast = dict(cfg.get("forecast", {}) or {})
    training = dict(cfg.get("external_baseline_training", {}) or {})
    baseline_cfg = dict(cfg.get("baselines", {}) or {})
    params = dict(item.get("params", {}) or {})
    for alias_key in ("prediction_protocol", "protocol", "rollout_mode"):
        if alias_key in params:
            params.setdefault("forecast_protocol", params.pop(alias_key))
    protocol = item.get(
        "forecast_protocol",
        item.get(
            "prediction_protocol",
            item.get(
                "protocol",
                training.get("forecast_protocol", training.get("protocol", forecast.get("forecast_protocol", forecast.get("protocol", None)))),
            ),
        ),
    )
    if protocol is not None:
        params.setdefault("forecast_protocol", protocol)
    one_step = item.get("one_step_lead", training.get("one_step_lead", forecast.get("one_step_lead", None)))
    if one_step is not None:
        params.setdefault("one_step_lead", int(one_step))
    time_step_hours = forecast.get("time_step_hours", cfg.get("time_step_hours", None))
    if time_step_hours is not None:
        params.setdefault("time_step_hours", float(time_step_hours))
    for training_only_key in (
        "train_protocol",
        "eval_protocol",
        "train_lead_times",
        "eval_lead_times",
        "rollout_loss_lead_times",
        "rollout_eval",
    ):
        params.pop(training_only_key, None)
    return build_forecast_baseline(
        str(item["name"]),
        dynamic_vars=list(data.get("dynamic_vars", cfg.get("dynamic_vars", [])) or []),
        static_vars=list(data.get("static_vars", cfg.get("static_vars", [])) or []),
        k=int(data.get("k", cfg.get("k", 1))),
        lead_times=forecast.get("lead_times", cfg.get("lead_times", None)),
        static_policy=str(item.get("static_policy", baseline_cfg.get("static_policy", "same_static"))),
        params=params,
    )


def _early_stopping_config(training: Dict) -> Dict:
    raw = dict(training.get("early_stopping", {}) or {})
    # Backward-compatible flat keys are also accepted.
    enabled = raw.get("enabled", training.get("early_stopping", False))
    return {
        "enabled": bool(enabled),
        "monitor": str(raw.get("monitor", training.get("early_stopping_monitor", "val_6h_mse"))),
        "mode": str(raw.get("mode", training.get("early_stopping_mode", "min"))).lower(),
        "patience": int(raw.get("patience", training.get("early_stopping_patience", 10))),
        "min_delta": float(raw.get("min_delta", training.get("early_stopping_min_delta", 0.0))),
        "restore_best": bool(raw.get("restore_best", training.get("restore_best", True))),
    }


def _checkpoint_config(training: Dict) -> Dict:
    raw = dict(training.get("checkpoint", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", training.get("checkpoint_enabled", True))),
        "save_last": bool(raw.get("save_last", training.get("save_last", True))),
        "save_every": int(raw.get("save_every", training.get("save_every", 1))),
        "save_top_k": int(raw.get("save_top_k", training.get("save_top_k", 0))),
        "monitor": raw.get("monitor", training.get("checkpoint_monitor", None)),
        "mode": raw.get("mode", training.get("checkpoint_mode", None)),
        "resume": bool(raw.get("resume", training.get("resume", False))),
        "auto_resume": bool(raw.get("auto_resume", training.get("auto_resume", False))),
        "resume_path": raw.get("resume_path", training.get("resume_path", None)),
        "strict": bool(raw.get("strict", training.get("resume_strict", True))),
        "load_optimizer": bool(raw.get("load_optimizer", training.get("resume_optimizer", True))),
    }


def _tensorboard_config(training: Dict) -> Dict:
    raw = dict(training.get("tensorboard", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", training.get("tensorboard", True))),
        "log_dir": raw.get("log_dir", training.get("tensorboard_log_dir", None)),
        "resume": bool(raw.get("resume", training.get("tensorboard_resume", True))),
        "flush_secs": int(raw.get("flush_secs", training.get("tensorboard_flush_secs", 30))),
        "log_every_n_steps": int(raw.get("log_every_n_steps", training.get("log_every_n_steps", 20))),
        "write_jsonl_fallback": bool(raw.get("write_jsonl_fallback", training.get("tensorboard_jsonl_fallback", True))),
    }


def _is_improvement(value: float, best: float, *, mode: str = "min", min_delta: float = 0.0) -> bool:
    if not math.isfinite(value):
        return False
    if mode == "max":
        return value > best + float(min_delta)
    if mode != "min":
        raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {mode!r}")
    return value < best - float(min_delta)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:  # pragma: no cover - defensive only
        print(json.dumps({"event": "rng_state_restore_warning", "warning": str(exc)}, ensure_ascii=False))


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    item: Dict,
    epoch: int,
    global_step: int,
    history: List[Dict],
    best_val: float,
    best_epoch: int,
    bad_epochs: int,
    early_cfg: Dict,
    alias: str,
) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "model_item": item,
        "alias": alias,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "global_step": int(global_step),
        "history": list(history),
        "best_val": float(best_val) if math.isfinite(float(best_val)) else best_val,
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "early_stopping": dict(early_cfg),
        "rng_state": _rng_state(),
    }


def _save_training_checkpoint(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _checkpoint_top_k_config(ckpt_cfg: Dict, early_cfg: Dict) -> Dict[str, Any]:
    mode = str(ckpt_cfg.get("mode") or early_cfg.get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ValueError(f"checkpoint.mode must be 'min' or 'max', got {mode!r}")
    return {
        "save_top_k": max(int(ckpt_cfg.get("save_top_k", 0)), 0),
        "monitor": str(ckpt_cfg.get("monitor") or early_cfg.get("monitor", "val_6h_mse")),
        "mode": mode,
    }


def _sanitize_checkpoint_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _score_checkpoint_token(score: float) -> str:
    if not math.isfinite(float(score)):
        return "nan"
    return f"{float(score):.8g}".replace("-", "neg").replace("+", "").replace(".", "p")


def _top_k_checkpoint_path(model_dir: Path, epoch: int, monitor: str, score: float) -> Path:
    metric = _sanitize_checkpoint_token(monitor)
    score_token = _score_checkpoint_token(score)
    return model_dir / "top_k" / f"epoch_{int(epoch):04d}_{metric}_{score_token}.pt"


def _sort_top_k_entries(entries: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    def _rank_key(entry: Dict[str, Any]) -> Tuple[float, int]:
        score = float(entry.get("score", math.nan))
        ranked_score = -score if mode == "max" else score
        return ranked_score, int(entry.get("epoch", 0))

    valid = [entry for entry in entries if math.isfinite(float(entry.get("score", math.nan)))]
    return sorted(valid, key=_rank_key)


def _normalise_top_k_entries(entries: Iterable[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for entry in entries:
        path = Path(str(entry.get("path", "")))
        if path.exists():
            kept.append({**entry, "path": str(path)})
    return _sort_top_k_entries(kept, mode)


def _load_top_k_checkpoints(model_dir: Path, top_k_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    index_path = model_dir / "top_k" / "top_k_checkpoints.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = data.get("checkpoints", []) if isinstance(data, dict) else data
    return _normalise_top_k_entries(entries or [], str(top_k_cfg.get("mode", "min")))


def _write_top_k_index(model_dir: Path, top_k_cfg: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
    save_json({**top_k_cfg, "checkpoints": entries}, model_dir / "top_k" / "top_k_checkpoints.json")


def _remove_checkpoint_file(path_value: Any) -> None:
    if not path_value:
        return
    path = Path(str(path_value))
    if path.exists():
        path.unlink()


def _update_top_k_checkpoints(
    *,
    entries: List[Dict[str, Any]],
    model_dir: Path,
    payload: Dict[str, Any],
    epoch: int,
    score: float,
    monitor: str,
    mode: str,
    save_top_k: int,
) -> List[Dict[str, Any]]:
    if save_top_k <= 0 or not math.isfinite(float(score)):
        return _normalise_top_k_entries(entries, mode)
    current = []
    for entry in entries:
        if int(entry.get("epoch", -1)) == int(epoch):
            _remove_checkpoint_file(entry.get("path"))
            continue
        current.append(entry)
    path = _top_k_checkpoint_path(model_dir, epoch, monitor, score)
    new_entry = {
        "epoch": int(epoch),
        "score": float(score),
        "monitor": str(monitor),
        "mode": str(mode),
        "path": str(path),
    }
    ranked = _sort_top_k_entries(current + [new_entry], mode)
    retained = ranked[:save_top_k]
    dropped = ranked[save_top_k:]
    retained_paths = {str(entry.get("path")) for entry in retained}
    if str(path) in retained_paths:
        save_payload = {**payload, "top_k_checkpoints": retained}
        _save_training_checkpoint(path, save_payload)
    for entry in dropped:
        _remove_checkpoint_file(entry.get("path"))
    _write_top_k_index(model_dir, {
        "save_top_k": int(save_top_k),
        "monitor": str(monitor),
        "mode": str(mode),
    }, retained)
    return retained


def _find_checkpoint_in_dir(dir_path: Path) -> Optional[Path]:
    r"""
    在目录内按优先级查找可用的 checkpoint：last.pt → best.pt → top_k 中最新 epoch。

    Parameters
    ----
    dir_path : Path
        模型输出目录。

    Returns
    ----
    Optional[Path]
        找到的 checkpoint 路径，若不存在则返回 None。
    """
    dir_path = Path(dir_path)
    last = dir_path / "last.pt"
    if last.exists():
        return last
    best = dir_path / "best.pt"
    if best.exists():
        return best
    index_path = dir_path / "top_k" / "top_k_checkpoints.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        entries = data.get("checkpoints", []) if isinstance(data, dict) else data
        valid: list[dict] = []
        for entry in (entries or []):
            path = Path(str(entry.get("path", "")))
            if path.exists():
                valid.append({"epoch": int(entry.get("epoch", 0)), "path": path})
        if valid:
            valid.sort(key=lambda e: e["epoch"], reverse=True)
            return valid[0]["path"]
    return None


def _resolve_resume_checkpoint(
    *,
    model_dir: str | Path,
    alias: str,
    ckpt_cfg: Dict,
    explicit_resume: Optional[bool] = None,
    explicit_resume_path: Optional[str | Path] = None,
) -> Optional[Path]:
    model_dir = Path(model_dir)
    resume = bool(ckpt_cfg.get("resume", False) or ckpt_cfg.get("auto_resume", False))
    if explicit_resume is not None:
        resume = bool(explicit_resume)
    path_value = explicit_resume_path if explicit_resume_path is not None else ckpt_cfg.get("resume_path")
    if path_value:
        candidate = Path(path_value).expanduser()
        if candidate.is_dir():
            alias_dir = candidate / alias
            if alias_dir.is_dir():
                found = _find_checkpoint_in_dir(alias_dir)
                if found is not None:
                    return found
            return _find_checkpoint_in_dir(candidate)
        return candidate
    if resume or bool(ckpt_cfg.get("auto_resume", False)):
        found = _find_checkpoint_in_dir(model_dir)
        if found is not None:
            return found
    return None


def _load_training_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    strict: bool = True,
    load_optimizer: bool = True,
) -> Dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=bool(strict))
    if optimizer is not None and load_optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    _set_rng_state(ckpt.get("rng_state"))
    return ckpt


def _make_summary_writer(tb_cfg: Dict, *, model_dir: Path, alias: str, global_step: int, resumed: bool):
    if not tb_cfg.get("enabled", True):
        return NullSummaryWriter(), {"enabled": False, "log_dir": None, "native_tensorboard": False, "jsonl_fallback": False}
    log_root = tb_cfg.get("log_dir") or str(model_dir.parent / "tensorboard")
    log_dir = Path(log_root) / alias
    purge_step = int(global_step) if resumed and tb_cfg.get("resume", True) else None
    if _TorchSummaryWriter is not None:
        writer = _TorchSummaryWriter(log_dir=str(log_dir), purge_step=purge_step, flush_secs=int(tb_cfg.get("flush_secs", 30)))
        return writer, {"enabled": True, "log_dir": str(log_dir), "native_tensorboard": True, "jsonl_fallback": False, "purge_step": purge_step}
    if tb_cfg.get("write_jsonl_fallback", True):
        writer = JsonlSummaryWriter(log_dir=log_dir, purge_step=purge_step, flush_secs=int(tb_cfg.get("flush_secs", 30)))
        return writer, {"enabled": True, "log_dir": str(log_dir), "native_tensorboard": False, "jsonl_fallback": True, "purge_step": purge_step}
    return NullSummaryWriter(), {"enabled": False, "log_dir": str(log_dir), "native_tensorboard": False, "jsonl_fallback": False, "purge_step": purge_step}


def _log_epoch(writer: Any, *, alias: str, epoch: int, global_step: int, item_hist: Dict) -> None:
    prefix = f"{alias}"
    for key in ("train_mse", "val_mse", "val_6h_mse", "val_rollout_mse", "monitor_value", "best_monitor", "bad_epochs"):
        if key in item_hist and item_hist[key] is not None:
            try:
                writer.add_scalar(f"{prefix}/{key}", float(item_hist[key]), global_step)
                writer.add_scalar(f"{prefix}_epoch/{key}", float(item_hist[key]), epoch)
            except Exception:
                pass


def _log_metric_tree(writer: Any, *, alias: str, split: str, metrics: Dict, global_step: int) -> None:
    for lead_tag, item in dict(metrics or {}).items():
        for metric_name in ("RMSE", "MAE", "Bias", "CRPS", "ACC"):
            if metric_name in item:
                writer.add_scalar(f"{alias}/{split}/{lead_tag}/{metric_name}", float(item[metric_name]), global_step)
        for metric_name in ("RMSE_per_var", "MAE_per_var", "Bias_per_var", "CRPS_per_var", "ACC_per_var"):
            for var_name, value in dict(item.get(metric_name, {}) or {}).items():
                writer.add_scalar(f"{alias}/{split}/{lead_tag}/{metric_name}/{var_name}", float(value), global_step)


def train_one_model(
    cfg: Dict,
    item: Dict,
    out_dir: Path,
    device: torch.device,
    *,
    resume: Optional[bool] = None,
    resume_path: Optional[str | Path] = None,
) -> Dict:
    data_cfg = parse_data_config(cfg)
    train_dl, val_dl, test_dl = build_dataloaders(data_cfg, shuffle_train=True)
    lead_times = data_cfg.lead_times or [data_cfg.delta_t]
    training = dict(cfg.get("external_baseline_training", {}) or {})
    epochs = int(training.get("epochs", 10))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-4))
    grad_clip = float(training.get("grad_clip", 1.0))
    save_best = bool(training.get("save_best", True))
    early_cfg = _early_stopping_config(training)
    ckpt_cfg = _checkpoint_config(training)
    tb_cfg = _tensorboard_config(training)
    alias = str(item.get("alias", item.get("name", "external_baseline")))

    model = _make_model(cfg, item).to(device)
    protocol_cfg = _training_protocol_config(cfg, item, model, lead_times)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({
        "model": alias,
        "event": "start_training",
        "parameters": n_params,
        "trainable": n_trainable,
        "train_protocol": protocol_cfg["train_protocol"],
        "forecast_protocol": protocol_cfg["forecast_protocol"],
        "train_lead_times": protocol_cfg["train_lead_times"],
        "eval_lead_times": protocol_cfg["eval_lead_times"],
        "one_step_lead": protocol_cfg["one_step_lead"],
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
    }, ensure_ascii=False))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_dir = out_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    top_k_cfg = _checkpoint_top_k_config(ckpt_cfg, early_cfg)
    top_k_entries: List[Dict[str, Any]] = _load_top_k_checkpoints(model_dir, top_k_cfg)

    history: List[Dict] = []
    best_val = -math.inf if early_cfg["mode"] == "max" else math.inf
    best_epoch = 0
    bad_epochs = 0
    stopped_epoch = 0
    early_stopped = False
    global_step = 0
    start_epoch = 1
    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"
    resumed_from: Optional[str] = None

    resume_ckpt = _resolve_resume_checkpoint(
        model_dir=model_dir,
        alias=alias,
        ckpt_cfg=ckpt_cfg,
        explicit_resume=resume,
        explicit_resume_path=resume_path,
    )
    if resume_ckpt is not None:
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_ckpt}")
        ckpt = _load_training_checkpoint(
            path=resume_ckpt,
            model=model,
            optimizer=opt,
            device=device,
            strict=bool(ckpt_cfg.get("strict", True)),
            load_optimizer=bool(ckpt_cfg.get("load_optimizer", True)),
        )
        resumed_from = str(resume_ckpt)
        start_epoch = int(ckpt.get("next_epoch", int(ckpt.get("epoch", 0)) + 1))
        global_step = int(ckpt.get("global_step", 0))
        history = list(ckpt.get("history", []) or [])
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        bad_epochs = int(ckpt.get("bad_epochs", bad_epochs))
        if not top_k_entries:
            top_k_entries = _normalise_top_k_entries(
                ckpt.get("top_k_checkpoints", []) or [],
                str(top_k_cfg.get("mode", "min")),
            )
        print(json.dumps({
            "model": alias,
            "event": "resume_training",
            "checkpoint": resumed_from,
            "start_epoch": start_epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_monitor": best_val,
        }, ensure_ascii=False))

    writer, tb_info = _make_summary_writer(tb_cfg, model_dir=model_dir, alias=alias, global_step=global_step, resumed=resumed_from is not None)
    writer.add_text("run/config", json.dumps({"alias": alias, "model_item": item}, ensure_ascii=False), global_step)

    if start_epoch > epochs:
        print(json.dumps({
            "model": alias,
            "event": "already_complete",
            "start_epoch": start_epoch,
            "configured_epochs": epochs,
        }, ensure_ascii=False))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        with Progress(BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
            task = progress.add_task(f"train {alias} epoch {epoch}/{epochs}", total=len(train_dl))
            for batch_idx, batch in enumerate(train_dl, start=1):
                batch = move_batch_to_device(batch, device)
                loss = _training_loss_for_batch(model, batch, device, protocol_cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                global_step += 1
                loss_value = float(loss.detach().cpu())
                train_loss += loss_value
                train_count += 1
                if tb_cfg.get("enabled", True) and int(tb_cfg.get("log_every_n_steps", 20)) > 0:
                    if global_step == 1 or global_step % int(tb_cfg.get("log_every_n_steps", 20)) == 0:
                        writer.add_scalar(f"{alias}/train_step/mse", loss_value, global_step)
                        writer.add_scalar(f"{alias}/train_step/epoch", float(epoch), global_step)
                progress.advance(task)

        model.eval()
        val_totals: Dict[str, float] = {}
        val_count = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = move_batch_to_device(batch, device)
                batch_losses = _validation_losses_for_batch(model, batch, device, protocol_cfg)
                for key, value in batch_losses.items():
                    val_totals[key] = val_totals.get(key, 0.0) + float(value)
                val_count += 1
        item_hist = {
            "epoch": epoch,
            "global_step": global_step,
            "train_mse": train_loss / max(train_count, 1),
            "train_protocol": protocol_cfg["train_protocol"],
            "forecast_protocol": protocol_cfg["forecast_protocol"],
            "train_lead_times": protocol_cfg["train_lead_times"],
            "eval_lead_times": protocol_cfg["eval_lead_times"],
        }
        for key, total in val_totals.items():
            item_hist[key] = total / max(val_count, 1)
        item_hist.setdefault("val_mse", item_hist.get("val_6h_mse", item_hist.get("val_rollout_mse", math.inf)))
        history.append(item_hist)

        monitor = str(early_cfg["monitor"])
        monitor_value = float(item_hist.get(monitor, item_hist["val_mse"]))
        improved = _is_improvement(
            monitor_value,
            best_val,
            mode=str(early_cfg["mode"]),
            min_delta=float(early_cfg["min_delta"]),
        )
        if improved:
            best_val = monitor_value
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        item_hist.update({
            "monitor": monitor,
            "monitor_value": monitor_value,
            "best_monitor": best_val,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
        })
        top_k_monitor = str(top_k_cfg["monitor"])
        top_k_score = float(item_hist.get(top_k_monitor, item_hist["val_mse"]))
        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=opt,
            cfg=cfg,
            item=item,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_val=best_val,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            early_cfg=early_cfg,
            alias=alias,
        )
        if save_best and improved:
            _save_training_checkpoint(best_path, checkpoint_payload)
        if ckpt_cfg.get("enabled", True):
            top_k_entries = _update_top_k_checkpoints(
                entries=top_k_entries,
                model_dir=model_dir,
                payload=checkpoint_payload,
                epoch=epoch,
                score=top_k_score,
                monitor=top_k_monitor,
                mode=str(top_k_cfg["mode"]),
                save_top_k=int(top_k_cfg["save_top_k"]),
            )
            item_hist["top_k_ranked_epochs"] = [int(entry["epoch"]) for entry in top_k_entries]
        _log_epoch(writer, alias=alias, epoch=epoch, global_step=global_step, item_hist=item_hist)
        writer.flush()
        print(json.dumps({"model": alias, **item_hist}, ensure_ascii=False))

        if ckpt_cfg.get("enabled", True) and ckpt_cfg.get("save_last", True):
            save_every = max(int(ckpt_cfg.get("save_every", 1)), 1)
            if epoch == epochs or epoch % save_every == 0:
                _save_training_checkpoint(last_path, {**checkpoint_payload, "top_k_checkpoints": top_k_entries})

        if early_cfg["enabled"] and bad_epochs >= int(early_cfg["patience"]):
            stopped_epoch = epoch
            early_stopped = True
            print(json.dumps({
                "model": alias,
                "event": "early_stopping",
                "stopped_epoch": stopped_epoch,
                "best_epoch": best_epoch,
                "best_monitor": best_val,
                "patience": int(early_cfg["patience"]),
            }, ensure_ascii=False))
            break

    if save_best and early_cfg.get("restore_best", True) and best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.6
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    eval_lead_times = [int(x) for x in protocol_cfg.get("eval_lead_times", lead_times)]
    metrics = {
        "val": evaluate_forecast_baseline(
            model,
            val_dl,
            lead_times=eval_lead_times,
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            progress_desc=f"val {alias}",
        ),
        "test": evaluate_forecast_baseline(
            model,
            test_dl,
            lead_times=eval_lead_times,
            time_step_hours=data_cfg.time_step_hours,
            var_names=data_cfg.dynamic_vars,
            progress_desc=f"test {alias}",
        ),
    }
    model_complexity = dict(metrics["val"].get("__model_complexity__", model_complexity_summary(model)))
    _log_metric_tree(writer, alias=alias, split="val", metrics=metrics["val"], global_step=global_step)
    _log_metric_tree(writer, alias=alias, split="test", metrics=metrics["test"], global_step=global_step)
    top_k_entries = _normalise_top_k_entries(top_k_entries, str(top_k_cfg.get("mode", "min")))
    if int(top_k_cfg.get("save_top_k", 0)) > 0:
        _write_top_k_index(model_dir, top_k_cfg, top_k_entries)

    result = {
        "alias": alias,
        "name": str(item.get("name")),
        "protocol": dict(protocol_cfg),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "resumed_from": resumed_from,
        "global_step": global_step,
        "start_epoch": start_epoch,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "early_stopping": {**early_cfg, "early_stopped": early_stopped, "stopped_epoch": stopped_epoch},
        "checkpointing": {
            **ckpt_cfg,
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "top_k": {**top_k_cfg, "checkpoints": top_k_entries, "count": len(top_k_entries)},
        },
        "top_k_checkpoints": top_k_entries,
        "tensorboard": tb_info,
        "model_complexity": model_complexity,
        "history": history,
        "metrics": metrics,
    }
    save_json(result, model_dir / "result.json")
    return result


def _build_baseline_comparison(results: Dict) -> Dict:
    records: List[Dict[str, Any]] = []
    for alias, result in dict(results.get("models", {}) or {}).items():
        complexity = dict(result.get("model_complexity", {}) or {})
        protocol = dict(result.get("protocol", {}) or {})
        top_k_checkpoints = list(result.get("top_k_checkpoints", []) or [])
        checkpointing = dict(result.get("checkpointing", {}) or {})
        for split in ("val", "test"):
            split_metrics = dict(result.get("metrics", {}).get(split, {}) or {})
            for lead_tag, metrics in split_metrics.items():
                if str(lead_tag).startswith("__"):
                    continue
                records.append({
                    "alias": alias,
                    "name": result.get("name"),
                    "forecast_protocol": protocol.get("forecast_protocol"),
                    "train_protocol": protocol.get("train_protocol"),
                    "train_lead_times": protocol.get("train_lead_times"),
                    "eval_lead_times": protocol.get("eval_lead_times"),
                    "best_checkpoint": checkpointing.get("best_checkpoint"),
                    "top_k_count": len(top_k_checkpoints),
                    "top_k_best_checkpoint": top_k_checkpoints[0].get("path") if top_k_checkpoints else None,
                    "top_k_best_score": top_k_checkpoints[0].get("score") if top_k_checkpoints else None,
                    "split": split,
                    "lead_tag": lead_tag,
                    "lead_steps": metrics.get("lead_steps"),
                    "num_parameters": complexity.get("num_parameters"),
                    "num_trainable": complexity.get("num_trainable"),
                    "estimated_gflops": complexity.get("estimated_gflops"),
                    "estimated_gflops_per_sample": complexity.get("estimated_gflops_per_sample"),
                    "one_step_gflops": complexity.get("one_step_gflops"),
                    "one_step_gflops_per_sample": complexity.get("one_step_gflops_per_sample"),
                    "rollout_steps": complexity.get("rollout_steps"),
                    "rollout_gflops": complexity.get("rollout_gflops"),
                    "rollout_gflops_per_sample": complexity.get("rollout_gflops_per_sample"),
                    "num_diffusion_steps": complexity.get("num_diffusion_steps"),
                    "RMSE": metrics.get("RMSE"),
                    "MAE": metrics.get("MAE"),
                    "Bias": metrics.get("Bias"),
                    "CRPS": metrics.get("CRPS"),
                    "ACC": metrics.get("ACC"),
                })
    return {"records": records}


def train_external_baselines(
    config_path: str | Path,
    out_dir: str | Path,
    device: str = "auto",
    *,
    resume: Optional[bool] = None,
    resume_path: Optional[str | Path] = None,
) -> Dict:
    cfg = load_config(config_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _iter_trainable_specs(cfg)
    if not specs:
        raise ValueError("No external baseline specs found under baselines.models")
    results = {
        "config": str(config_path),
        "device": str(device_obj),
        "resume": resume,
        "resume_path": None if resume_path is None else str(resume_path),
        "models": {},
    }
    for item in specs:
        alias = str(item.get("alias", item.get("name", "unknown")))
        result_path = out_dir / alias / "result.json"
        if result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            print(json.dumps({
                "model": alias,
                "event": "skip_completed",
                "reason": "result.json already exists",
                "result_path": str(result_path),
            }, ensure_ascii=False))
            results["models"][alias] = existing
            continue
        res = train_one_model(cfg, item, out_dir, device_obj, resume=resume, resume_path=resume_path)
        results["models"][res["alias"]] = res
    comparison = _build_baseline_comparison(results)
    save_json(results, out_dir / "external_baselines_results.json")
    save_json(comparison, out_dir / "baseline_comparison.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UrbanPiDiT-loader-compatible FourCastNet/GraphCast/GenCast baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="outputs/baselines/external_weather")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume each baseline from its last checkpoint in --out_dir/<alias>/last.pt unless --resume_path is set.")
    parser.add_argument("--resume_path", default=None, help="Checkpoint file or output directory to resume from. A directory is resolved as <dir>/<alias>/last.pt when possible.")
    args = parser.parse_args()
    results = train_external_baselines(args.config, args.out_dir, args.device, resume=args.resume, resume_path=args.resume_path)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
