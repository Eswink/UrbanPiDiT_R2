"""子进程启动、日志读取和进程状态管理。"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from . import db
from .paths import CONDA_ENV, PROJECT_ROOT, STREAMLIT_ROOT

RUN_LOG_DIR = STREAMLIT_ROOT / "run_logs"


def timestamp_slug() -> str:
    """生成适合路径的时间戳。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_python_command(args: list[str]) -> list[str]:
    """构建 conda 环境下的 Python 命令。"""

    return ["conda", "run", "-p", str(CONDA_ENV), "python", *args]


def command_to_text(command: list[str]) -> str:
    """生成便于展示的命令文本。"""

    return " ".join(shlex.quote(str(part)) for part in command)


def _open_log(log_file: Path) -> Any:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file.open("a", encoding="utf-8", buffering=1)


def spawn_experiment(
    *,
    name: str,
    kind: str,
    config_path: str,
    config_hash: str,
    stage: str,
    command: list[str],
    log_dir: str = "",
    out_dir: str = "",
    notes: str = "",
) -> int:
    """启动实验子进程并记录到 SQLite。"""

    log_file = RUN_LOG_DIR / f"{timestamp_slug()}_{kind}_{safe_name(name)}.log"
    experiment_id = db.create_experiment(
        name=name,
        kind=kind,
        config_path=config_path,
        config_hash=config_hash,
        stage=stage,
        command=command_to_text(command),
        log_file=str(log_file),
        log_dir=log_dir,
        out_dir=out_dir,
        notes=notes,
    )

    with _open_log(log_file) as handle:
        handle.write(f"[UI] experiment_id={experiment_id}\n")
        handle.write(f"[UI] command={command_to_text(command)}\n\n")
        proc = subprocess.Popen(
            _wrap_command(command, experiment_id),
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    db.update_experiment(experiment_id, status="running", pid=int(proc.pid))
    return experiment_id


def _wrap_command(command: list[str], experiment_id: int) -> list[str]:
    """包一层状态更新命令。"""

    python = str(CONDA_ENV / "bin" / "python")
    code = "\n".join(
        [
            "import os, subprocess, sys",
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
            "from streamlit_app.utils import db",
            f"cmd = {command!r}",
            "proc = subprocess.Popen(cmd)",
            "return_code = proc.wait()",
            f"status = 'completed' if return_code == 0 else 'failed'",
            f"db.update_experiment({experiment_id}, status=status, return_code=return_code, end_time=db.now_iso())",
            "sys.exit(return_code)",
        ]
    )
    return [python, "-c", code]


def refresh_experiment_status(experiment: dict[str, Any]) -> dict[str, Any]:
    """同步单条实验的进程状态。"""

    status = str(experiment.get("status") or "")
    pid = experiment.get("pid")
    if status != "running" or not pid:
        return experiment

    alive = psutil.pid_exists(int(pid))
    if alive:
        try:
            proc = psutil.Process(int(pid))
            if proc.status() != psutil.STATUS_ZOMBIE:
                return experiment
        except psutil.Error:
            return experiment

    db.update_experiment(experiment["id"], status="completed", end_time=db.now_iso())
    updated = db.get_experiment(int(experiment["id"]))
    return updated or experiment


def refresh_all_running() -> None:
    """同步全部运行中实验状态。"""

    for item in db.list_experiments(limit=1000):
        if item.get("status") == "running":
            refresh_experiment_status(item)


def stop_experiment(experiment_id: int) -> bool:
    """终止实验进程。"""

    item = db.get_experiment(experiment_id)
    if not item or not item.get("pid"):
        return False
    pid = int(item["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        db.update_experiment(experiment_id, status="completed", end_time=db.now_iso())
        return True
    except PermissionError:
        return False
    db.update_experiment(experiment_id, status="killed", end_time=db.now_iso())
    return True


def read_log_tail(log_file: str | Path | None, *, max_lines: int = 200) -> str:
    """读取日志尾部。"""

    if not log_file:
        return ""
    path = Path(log_file)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def safe_name(name: str) -> str:
    """生成安全文件名片段。"""

    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    value = "".join(keep).strip("_")
    return value or "experiment"


def make_train_command(config_path: str, *, resume: bool = False, resume_ckpt: str | None = None) -> list[str]:
    """生成训练命令。"""

    args = ["train.py", "--config", config_path]
    if resume_ckpt:
        args += ["--resume_ckpt", resume_ckpt]
    elif resume:
        args.append("--resume")
    return build_python_command(args)


def make_eval_command(config_path: str, ckpt_path: str) -> list[str]:
    """生成评估命令。"""

    return build_python_command(["evaluate.py", "--config", config_path, "--ckpt", ckpt_path])


def make_forecast_baseline_command(config_path: str, out_dir: str) -> list[str]:
    """生成公平基线命令。"""

    return build_python_command(["-m", "baselines.forecast_runner", "--config", config_path, "--out_dir", out_dir])


def make_legacy_baseline_command(
    *,
    family: str,
    config_path: str,
    out_dir: str,
    options: dict[str, Any],
) -> list[str]:
    """生成可训练传统基线命令。"""

    if family == "linear":
        args = [
            "-m",
            "baselines.linear",
            "--config",
            config_path,
            "--out_dir",
            out_dir,
            "--kind",
            str(options.get("kind", "ridge")),
            "--alpha",
            str(options.get("alpha", 1.0)),
            "--max_train_samples",
            str(options.get("max_train_samples", 0)),
        ]
    elif family == "tree":
        args = [
            "-m",
            "baselines.tree",
            "--config",
            config_path,
            "--out_dir",
            out_dir,
            "--model",
            str(options.get("model", "rf")),
            "--max_train_rows",
            str(options.get("max_train_rows", 0)),
        ]
        if bool(options.get("tree_no_xy", False)):
            args.append("--no_xy")
    elif family == "mlp":
        args = [
            "-m",
            "baselines.mlp",
            "--config",
            config_path,
            "--out_dir",
            out_dir,
            "--epochs",
            str(options.get("epochs", 50)),
            "--lr",
            str(options.get("lr", 1e-3)),
            "--hidden",
            str(options.get("hidden", 1024)),
            "--dropout",
            str(options.get("dropout", 0.0)),
        ]
    else:
        raise ValueError(f"未知基线类型：{family}")
    return build_python_command(args)