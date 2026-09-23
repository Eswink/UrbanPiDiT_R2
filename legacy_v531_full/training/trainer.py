"""UrbanPiDiT 训练编排工具。

本模块集中管理单阶段、两阶段训练、断点恢复与 checkpoint 初始化，供
`train.py` 和 `train_suite.py` 复用。
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception:
        WandbLogger = None
except Exception as e:
    raise ImportError("训练脚本需要安装 pytorch_lightning。请先安装后再运行。") from e

from data.loader import MetroWeatherDataModule
from pidit_lit import UrbanPiDiTLitModule
from urbanpidit_version import DEFAULT_RUN_NAME
from utils.config_builder import build_datamodule_kwargs, build_litmodule_kwargs, deep_update


def load_config(path: str) -> Dict[str, Any]:
    r"""读取 YAML 配置，自动解析 ``base`` 和 ``base_overrides``。

    若配置中包含 ``base`` 键，则先加载基础配置（支持递归），
    再应用 ``base_overrides`` 覆盖，最后合并子配置中的其余顶层键。

    Parameters
    ----
    path : str
        配置文件路径。

    Returns
    ----
    Dict[str, Any]
        合并后的完整配置字典。
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg or {})

    base_path = cfg.pop("base", None)
    overrides = dict(cfg.pop("base_overrides", {}) or {})

    if base_path is not None:
        # 基础配置路径相对于项目根目录（cwd）解析，与 train_suite.py 行为一致
        resolved_base = os.path.normpath(os.path.join(os.getcwd(), base_path))
        if not os.path.exists(resolved_base):
            # fallback: 相对于当前配置所在目录
            config_dir = os.path.dirname(os.path.abspath(path))
            resolved_base = os.path.normpath(os.path.join(config_dir, base_path))
        merged = load_config(resolved_base)
        merged = deep_update(merged, overrides)
        merged = deep_update(merged, cfg)
    else:
        merged = cfg

    # 将顶层 two_stage 注入到 train.two_stage，保持与 run_training_from_config 兼容
    two_stage = merged.pop("two_stage", None)
    if two_stage is not None:
        merged = deep_update(merged, {"train": {"two_stage": dict(two_stage)}})

    return merged


def build_datamodule(cfg: Dict[str, Any]) -> MetroWeatherDataModule:
    """根据配置构建 DataModule。"""

    return MetroWeatherDataModule(**build_datamodule_kwargs(cfg))


def build_litmodule(cfg: Dict[str, Any]) -> UrbanPiDiTLitModule:
    """根据配置构建 LightningModule。"""

    return UrbanPiDiTLitModule(**build_litmodule_kwargs(cfg))


def build_loggers(cfg: Dict[str, Any], *, stage_name: str, resume: bool = False) -> Any:
    """构建 TensorBoard/W&B logger。"""

    logging_cfg = dict(cfg.get("logging", {}) or {})
    use_tb = bool(logging_cfg.get("use_tensorboard", True))
    use_wandb = bool(logging_cfg.get("use_wandb", False))

    log_dir = str(logging_cfg.get("log_dir", "logs"))
    run_name = str(logging_cfg.get("run_name", DEFAULT_RUN_NAME))

    loggers = []
    if use_tb:
        loggers.append(TensorBoardLogger(save_dir=log_dir, name=run_name, version=stage_name))

    if use_wandb:
        if WandbLogger is None:
            raise ImportError(
                "已在配置中开启 use_wandb，但当前环境未安装 wandb 或 WandbLogger 不可用。\n"
                "请执行：pip install wandb\n然后重试。"
            )
        loggers.append(_build_wandb_logger(logging_cfg, log_dir, run_name, stage_name, resume))

    if len(loggers) == 0:
        return False
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def _build_wandb_logger(
    logging_cfg: Dict[str, Any],
    log_dir: str,
    run_name: str,
    stage_name: str,
    resume: bool,
) -> Any:
    wandb_project = str(logging_cfg.get("wandb_project", "UrbanPiDiT"))
    wandb_entity = logging_cfg.get("wandb_entity", None)
    wandb_group = logging_cfg.get("wandb_group", run_name)
    wandb_name = str(logging_cfg.get("wandb_name", run_name)) + f"_{stage_name}"
    wandb_tags = logging_cfg.get("wandb_tags", [])
    wandb_save_dir = logging_cfg.get("wandb_save_dir", log_dir)
    wandb_log_model = bool(logging_cfg.get("wandb_log_model", False))
    wandb_offline = bool(logging_cfg.get("wandb_offline", False))
    wandb_id = _resolve_wandb_id(log_dir, run_name, stage_name, resume)

    return WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_name,
        tags=wandb_tags,
        save_dir=wandb_save_dir,
        log_model=wandb_log_model,
        offline=wandb_offline,
        id=wandb_id,
        resume="allow" if resume else None,
    )


def _resolve_wandb_id(log_dir: str, run_name: str, stage_name: str, resume: bool) -> str:
    wandb_id_file = os.path.join(log_dir, run_name, stage_name, "wandb_id.txt")
    if resume and os.path.isfile(wandb_id_file):
        try:
            with open(wandb_id_file, "r") as f:
                value = f.read().strip()
            if value:
                return value
        except Exception:
            pass

    value = uuid.uuid4().hex
    os.makedirs(os.path.dirname(wandb_id_file), exist_ok=True)
    try:
        with open(wandb_id_file, "w") as f:
            f.write(value)
    except Exception:
        pass
    return value


def get_ckpt_dir(cfg: Dict[str, Any], stage_name: str) -> str:
    """返回某个训练阶段的 checkpoint 目录。"""

    logging_cfg = dict(cfg.get("logging", {}) or {})
    log_dir = str(logging_cfg.get("log_dir", "logs"))
    run_name = str(logging_cfg.get("run_name", DEFAULT_RUN_NAME))
    return str(logging_cfg.get("ckpt_dir", os.path.join(log_dir, run_name, stage_name, "checkpoints")))


def find_last_ckpt(ckpt_dir: str) -> Optional[str]:
    """查找 last checkpoint。"""

    path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(path):
        return path
    return None


def load_ckpt_weights(
    lit: UrbanPiDiTLitModule,
    ckpt_path: str,
    *,
    only_net: bool = True,
    strict: bool = False,
) -> None:
    """从 Lightning checkpoint 加载预训练权重。"""

    if not ckpt_path:
        raise ValueError("ckpt_path 为空，无法加载权重")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state

    if only_net:
        if any(k.startswith("net.") for k in sd.keys()):
            sd_net = {k[len("net.") :]: v for k, v in sd.items() if k.startswith("net.")}
        else:
            sd_net = sd
        missing, unexpected = lit.net.load_state_dict(sd_net, strict=strict)
    else:
        missing, unexpected = lit.load_state_dict(sd, strict=strict)

    _print_load_report(missing, unexpected)


def _print_load_report(missing: Any, unexpected: Any) -> None:
    if missing:
        suffix = "..." if len(missing) > 8 else ""
        print(f"[LoadCkpt] Missing keys ({len(missing)}): {missing[:8]}{suffix}")
    if unexpected:
        suffix = "..." if len(unexpected) > 8 else ""
        print(f"[LoadCkpt] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{suffix}")


def build_trainer_and_callbacks(
    cfg: Dict[str, Any],
    *,
    stage_name: str,
    resume: bool = False,
) -> Tuple[pl.Trainer, ModelCheckpoint]:
    """构建 Trainer 与 checkpoint 回调。"""

    train_cfg = dict(cfg.get("train", {}) or {})
    monitor_metric = str(train_cfg.get("monitor", "val/RMSE"))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    patience = int(train_cfg.get("patience", 20))
    save_top_k = int(train_cfg.get("save_top_k", 5))

    ckpt_dir = get_ckpt_dir(cfg, stage_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor_metric,
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
        filename="{epoch:03d}-{step:06d}",
    )
    es = EarlyStopping(monitor=monitor_metric, mode="min", patience=patience, min_delta=min_delta)
    lrm = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        max_epochs=int(train_cfg.get("max_epochs", 200)),
        check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
        logger=build_loggers(cfg, stage_name=stage_name, resume=resume),
        callbacks=[ckpt, es, lrm],
        deterministic=True,
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 0.0)),
        accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
        limit_train_batches=train_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
        limit_test_batches=train_cfg.get("limit_test_batches", 1.0),
    )
    return trainer, ckpt


def run_one_stage(
    cfg: Dict[str, Any],
    *,
    stage_name: str,
    init_ckpt_path: Optional[str] = None,
    init_only_net: bool = True,
    init_strict: bool = False,
    resume_fit_ckpt_path: Optional[str] = None,
    test_after_fit: bool = True,
) -> str:
    """运行一个训练阶段并返回 best checkpoint 路径。"""

    seed = int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    dm = build_datamodule(cfg)
    lit = build_litmodule(cfg)

    if init_ckpt_path is not None and resume_fit_ckpt_path is None:
        print(f"[Stage:{stage_name}] 初始化权重 <- {init_ckpt_path}")
        load_ckpt_weights(lit, init_ckpt_path, only_net=init_only_net, strict=init_strict)

    resume_flag = resume_fit_ckpt_path is not None
    trainer, ckpt_cb = build_trainer_and_callbacks(cfg, stage_name=stage_name, resume=resume_flag)
    if resume_flag:
        print(f"[Stage:{stage_name}] 断点续训 <- {resume_fit_ckpt_path}")
        trainer.fit(lit, datamodule=dm, ckpt_path=resume_fit_ckpt_path)
    else:
        trainer.fit(lit, datamodule=dm)

    if test_after_fit:
        trainer.test(lit, datamodule=dm, ckpt_path="best")

    best_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    print(f"[Stage:{stage_name}] best checkpoint: {best_path}")
    return best_path


def run_training_from_config(
    cfg: Dict[str, Any],
    *,
    resume: bool = False,
    resume_ckpt: Optional[str] = None,
    test_after_fit: bool = True,
) -> str:
    """按配置运行单阶段或两阶段训练，返回最终阶段 best checkpoint。"""

    two_stage_cfg = dict(cfg.get("train", {}).get("two_stage", {}) or {})
    if not bool(two_stage_cfg.get("enabled", False)):
        return _run_single_stage_from_config(
            cfg,
            resume=resume,
            resume_ckpt=resume_ckpt,
            test_after_fit=test_after_fit,
        )
    return _run_two_stage_from_config(
        cfg,
        two_stage_cfg,
        resume=resume,
        resume_ckpt=resume_ckpt,
        test_after_fit=test_after_fit,
    )


def _run_single_stage_from_config(
    cfg: Dict[str, Any],
    *,
    resume: bool,
    resume_ckpt: Optional[str],
    test_after_fit: bool,
) -> str:
    stage_name = "single"
    fit_ckpt = resume_ckpt
    if fit_ckpt is None and resume:
        fit_ckpt = find_last_ckpt(get_ckpt_dir(cfg, stage_name))
    return run_one_stage(cfg, stage_name=stage_name, resume_fit_ckpt_path=fit_ckpt, test_after_fit=test_after_fit)


def _run_two_stage_from_config(
    cfg: Dict[str, Any],
    two_stage_cfg: Dict[str, Any],
    *,
    resume: bool,
    resume_ckpt: Optional[str],
    test_after_fit: bool,
) -> str:
    stage1 = dict(two_stage_cfg.get("stage1", {}) or {})
    stage2 = dict(two_stage_cfg.get("stage2", {}) or {})
    stage1_name = str(stage1.get("name", "stage1_6h"))
    stage2_name = str(stage2.get("name", "stage2_rollout"))

    cfg_stage1 = deep_update(cfg, dict(stage1.get("overrides", {}) or {}))
    cfg_stage2 = deep_update(cfg, dict(stage2.get("overrides", {}) or {}))
    resume_stage, fit_ckpt = _resolve_resume_stage(
        cfg_stage1,
        cfg_stage2,
        stage1_name,
        stage2_name,
        resume=resume,
        resume_ckpt=resume_ckpt,
    )

    if resume_stage == stage1_name:
        ckpt_stage1 = run_one_stage(
            cfg_stage1,
            stage_name=stage1_name,
            resume_fit_ckpt_path=fit_ckpt,
            test_after_fit=test_after_fit,
        )
    elif resume_stage == stage2_name:
        ckpt_stage1 = None
    else:
        ckpt_stage1 = run_one_stage(cfg_stage1, stage_name=stage1_name, test_after_fit=test_after_fit)

    if resume_stage == stage2_name:
        return run_one_stage(
            cfg_stage2,
            stage_name=stage2_name,
            resume_fit_ckpt_path=fit_ckpt,
            test_after_fit=test_after_fit,
        )

    ckpt_init = ckpt_stage1 if bool(stage2.get("init_from_stage1", True)) else None
    return run_one_stage(
        cfg_stage2,
        stage_name=stage2_name,
        init_ckpt_path=ckpt_init,
        init_only_net=bool(stage2.get("init_only_net", True)),
        init_strict=bool(stage2.get("init_strict", False)),
        test_after_fit=test_after_fit,
    )


def _resolve_resume_stage(
    cfg_stage1: Dict[str, Any],
    cfg_stage2: Dict[str, Any],
    stage1_name: str,
    stage2_name: str,
    *,
    resume: bool,
    resume_ckpt: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if resume_ckpt:
        if stage2_name in resume_ckpt:
            return stage2_name, resume_ckpt
        if stage1_name in resume_ckpt:
            return stage1_name, resume_ckpt
        return None, resume_ckpt
    if not resume:
        return None, None

    ckpt2 = find_last_ckpt(get_ckpt_dir(cfg_stage2, stage2_name))
    if ckpt2:
        return stage2_name, ckpt2
    ckpt1 = find_last_ckpt(get_ckpt_dir(cfg_stage1, stage1_name))
    if ckpt1:
        return stage1_name, ckpt1
    return None, None