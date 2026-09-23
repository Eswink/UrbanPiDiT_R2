r"""UrbanPiDiT V5.3.1 一键实验套件入口。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from training.evaluator import evaluate_checkpoint
from training.trainer import find_last_ckpt, get_ckpt_dir, load_config, run_training_from_config
from utils.config_builder import deep_update

BatchLimit = Union[int, float]


@dataclass(frozen=True)
class SuiteContext:
    suite_path: Path
    out_dir: Path
    suite_cfg: dict[str, Any]
    base_cfg: dict[str, Any]
    aliases: Optional[set[str]]
    resume: bool
    eval_only: bool
    skip_existing: bool
    max_epochs: Optional[int]
    limit_train_batches: Optional[BatchLimit]
    limit_val_batches: Optional[BatchLimit]
    limit_test_batches: Optional[BatchLimit]


def parse_batch_limit(value: str) -> BatchLimit:
    raw = value.strip()
    number = float(raw)
    if number <= 0:
        raise argparse.ArgumentTypeError("batch limit 必须大于 0")
    if number.is_integer() and "." not in raw and "e" not in raw.lower():
        return int(number)
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UrbanPiDiT V5.3.1 experiment suite")
    parser.add_argument("--config", type=str, required=True, help="suite YAML 路径")
    parser.add_argument("--out_dir", "--out-dir", type=str, required=True, help="实验输出根目录")
    parser.add_argument("--resume", action="store_true", help="跳过已完成实验，并续训未完成实验")
    parser.add_argument("--eval_only", "--eval-only", action="store_true", help="只评估已有 checkpoint，不训练")
    parser.add_argument("--experiments", type=str, default=None, help="逗号分隔的实验 alias 列表")
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true", help="跳过已有 summary.json 的实验")
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=None, help="smoke test 覆盖每阶段最大 epoch")
    parser.add_argument(
        "--limit_train_batches",
        "--limit-train-batches",
        type=parse_batch_limit,
        default=None,
        help="限制每个训练 epoch 的 batch 数/比例",
    )
    parser.add_argument(
        "--limit_val_batches",
        "--limit-val-batches",
        type=parse_batch_limit,
        default=None,
        help="限制每次验证的 batch 数/比例",
    )
    parser.add_argument(
        "--limit_test_batches",
        "--limit-test-batches",
        type=parse_batch_limit,
        default=None,
        help="限制自动评估的 batch 数/比例",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="只打印将要执行的实验，不训练也不评估")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def resolve_path(raw_path: str, suite_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_root = suite_path.parent.parent if suite_path.parent.name == "configs" else suite_path.parent
    candidates = [Path.cwd() / path, suite_path.parent / path, project_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (project_root / path).resolve()


def parse_aliases(value: Optional[str]) -> Optional[set[str]]:
    if value is None or not value.strip():
        return None
    aliases = {item.strip() for item in value.split(",") if item.strip()}
    return aliases or None


def load_suite_context(args: argparse.Namespace) -> SuiteContext:
    suite_path = Path(args.config).resolve()
    suite_cfg = read_yaml(suite_path)
    base_path = resolve_path(str(suite_cfg.get("base", "")), suite_path)
    base_cfg = load_config(str(base_path))
    base_cfg = deep_update(base_cfg, suite_cfg.get("base_overrides", {}))
    base_cfg = inject_two_stage(base_cfg, suite_cfg)
    return SuiteContext(
        suite_path=suite_path,
        out_dir=Path(args.out_dir).resolve(),
        suite_cfg=suite_cfg,
        base_cfg=base_cfg,
        aliases=parse_aliases(args.experiments),
        resume=bool(args.resume),
        eval_only=bool(args.eval_only),
        skip_existing=bool(args.skip_existing),
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )


def inject_two_stage(base_cfg: dict[str, Any], suite_cfg: dict[str, Any]) -> dict[str, Any]:
    two_stage = suite_cfg.get("two_stage", None)
    if two_stage is None:
        return base_cfg
    return deep_update(base_cfg, {"train": {"two_stage": dict(two_stage or {})}})


def selected_experiments(ctx: SuiteContext) -> list[dict[str, Any]]:
    experiments = [dict(item) for item in ctx.suite_cfg.get("experiments", [])]
    aliases = [str(item.get("alias", "")) for item in experiments]
    if len(set(aliases)) != len(aliases):
        raise ValueError("suite 配置中存在重复 alias")
    if ctx.aliases is None:
        return experiments
    missing = sorted(ctx.aliases - set(aliases))
    if missing:
        raise ValueError(f"未找到实验 alias: {missing}")
    return [item for item in experiments if str(item.get("alias")) in ctx.aliases]


def build_variant_cfg(ctx: SuiteContext, experiment: dict[str, Any]) -> dict[str, Any]:
    alias = str(experiment["alias"])
    exp_dir = ctx.out_dir / alias
    cfg = deep_update(ctx.base_cfg, experiment.get("overrides", {}))
    cfg = deep_update(
        cfg,
        {
            "logging": {
                "log_dir": str(exp_dir / "logs"),
                "run_name": alias,
            },
            "suite": {
                "config": str(ctx.suite_path),
                "alias": alias,
                "group": experiment.get("group", ""),
                "desc": experiment.get("desc", ""),
            },
        },
    )
    return apply_runtime_overrides(cfg, ctx)


def apply_runtime_overrides(cfg: dict[str, Any], ctx: SuiteContext) -> dict[str, Any]:
    train_updates = build_runtime_train_updates(ctx)
    eval_updates = build_runtime_eval_updates(ctx)
    updates: dict[str, Any] = {}
    if train_updates:
        updates["train"] = train_updates
    if eval_updates:
        updates["eval"] = eval_updates
    out = deep_update(cfg, updates)
    return apply_runtime_stage_overrides(out, train_updates)


def build_runtime_train_updates(ctx: SuiteContext) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if ctx.max_epochs is not None:
        updates["max_epochs"] = int(ctx.max_epochs)
    if ctx.limit_train_batches is not None:
        updates["limit_train_batches"] = ctx.limit_train_batches
    if ctx.limit_val_batches is not None:
        updates["limit_val_batches"] = ctx.limit_val_batches
    if ctx.limit_test_batches is not None:
        updates["limit_test_batches"] = ctx.limit_test_batches
    return updates


def build_runtime_eval_updates(ctx: SuiteContext) -> dict[str, Any]:
    if ctx.limit_test_batches is None:
        return {}
    return {"limit_test_batches": ctx.limit_test_batches}


def apply_runtime_stage_overrides(cfg: dict[str, Any], train_updates: dict[str, Any]) -> dict[str, Any]:
    if not train_updates:
        return cfg
    two_stage = dict(cfg.get("train", {}).get("two_stage", {}) or {})
    if not bool(two_stage.get("enabled", False)):
        return cfg
    return deep_update(
        cfg,
        {
            "train": {
                "two_stage": {
                    "stage1": {"overrides": {"train": train_updates}},
                    "stage2": {"overrides": {"train": train_updates}},
                }
            }
        },
    )


def final_stage_cfg(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    two_stage = dict(cfg.get("train", {}).get("two_stage", {}) or {})
    if not bool(two_stage.get("enabled", False)):
        return "single", cfg
    stage2 = dict(two_stage.get("stage2", {}) or {})
    stage_name = str(stage2.get("name", "stage2_multi_lead"))
    return stage_name, deep_update(cfg, stage2.get("overrides", {}) or {})


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(exp_dir: Path, data: dict[str, Any]) -> None:
    payload = {"updated_at": utc_now(), **data}
    write_json(exp_dir / "status.json", payload)


def append_event(ctx: SuiteContext, alias: str, event: str, **fields: Any) -> None:
    payload = {"time": utc_now(), "alias": alias, "event": event, **fields}
    path = ctx.out_dir / "suite_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def planned_resume_checkpoint(cfg: dict[str, Any]) -> dict[str, Any]:
    two_stage = dict(cfg.get("train", {}).get("two_stage", {}) or {})
    if not bool(two_stage.get("enabled", False)):
        stage_name = "single"
        ckpt_dir = get_ckpt_dir(cfg, stage_name)
        return {"stage": stage_name, "ckpt_dir": ckpt_dir, "ckpt_path": find_last_ckpt(ckpt_dir)}

    stage1 = dict(two_stage.get("stage1", {}) or {})
    stage2 = dict(two_stage.get("stage2", {}) or {})
    stage1_name = str(stage1.get("name", "stage1_6h"))
    stage2_name = str(stage2.get("name", "stage2_rollout"))
    cfg_stage1 = deep_update(cfg, dict(stage1.get("overrides", {}) or {}))
    cfg_stage2 = deep_update(cfg, dict(stage2.get("overrides", {}) or {}))
    stage2_dir = get_ckpt_dir(cfg_stage2, stage2_name)
    stage2_ckpt = find_last_ckpt(stage2_dir)
    if stage2_ckpt:
        return {"stage": stage2_name, "ckpt_dir": stage2_dir, "ckpt_path": stage2_ckpt}
    stage1_dir = get_ckpt_dir(cfg_stage1, stage1_name)
    stage1_ckpt = find_last_ckpt(stage1_dir)
    return {"stage": stage1_name, "ckpt_dir": stage1_dir, "ckpt_path": stage1_ckpt}


def resolve_checkpoint(exp_dir: Path, cfg: dict[str, Any]) -> str:
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        value = str(read_json(summary_path).get("ckpt_path", ""))
        if value and Path(value).is_file():
            return value

    stage_name, stage_cfg = final_stage_cfg(cfg)
    ckpt_dir = Path(get_ckpt_dir(stage_cfg, stage_name))
    last_ckpt = find_last_ckpt(str(ckpt_dir))
    if last_ckpt:
        return last_ckpt

    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"找不到可评估 checkpoint: {ckpt_dir}")


def run_experiment(ctx: SuiteContext, experiment: dict[str, Any]) -> dict[str, Any]:
    alias = str(experiment["alias"])
    exp_dir = ctx.out_dir / alias
    summary_path = exp_dir / "summary.json"
    if summary_path.exists() and (ctx.resume or ctx.skip_existing):
        print(f"[Suite] 跳过已完成实验: {alias}")
        summary = read_json(summary_path)
        write_status(exp_dir, {"status": "skipped_completed", "summary_path": str(summary_path)})
        append_event(ctx, alias, "skip_completed", summary_path=str(summary_path))
        return summary

    cfg = build_variant_cfg(ctx, experiment)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(exp_dir / "config.yaml", cfg)
    resume_plan = planned_resume_checkpoint(cfg) if ctx.resume else None
    write_status(
        exp_dir,
        {
            "status": "started",
            "resume_enabled": ctx.resume,
            "eval_only": ctx.eval_only,
            "planned_resume": resume_plan,
        },
    )
    append_event(ctx, alias, "experiment_start", resume_enabled=ctx.resume, eval_only=ctx.eval_only, planned_resume=resume_plan)

    started_at = time.time()
    try:
        ckpt_path = resolve_checkpoint(exp_dir, cfg) if ctx.eval_only else train_variant(ctx, alias, cfg)
        stage_name, eval_cfg = final_stage_cfg(cfg)
        write_status(exp_dir, {"status": "evaluating", "ckpt_path": ckpt_path, "stage": stage_name})
        append_event(ctx, alias, "eval_start", ckpt_path=ckpt_path, stage=stage_name)
        metrics = evaluate_checkpoint(eval_cfg, ckpt_path, output_path=str(exp_dir / "metrics.json"))
        summary = build_experiment_summary(experiment, ckpt_path, stage_name, metrics, started_at)
        write_json(summary_path, summary)
        write_status(exp_dir, {"status": "completed", "ckpt_path": ckpt_path, "summary_path": str(summary_path)})
        append_event(ctx, alias, "experiment_complete", ckpt_path=ckpt_path, summary_path=str(summary_path))
        return summary
    except Exception as exc:
        write_status(exp_dir, {"status": "failed", "error": repr(exc)})
        append_event(ctx, alias, "experiment_failed", error=repr(exc))
        raise


def train_variant(ctx: SuiteContext, alias: str, cfg: dict[str, Any]) -> str:
    print(f"[Suite] 训练实验: {alias}")
    exp_dir = ctx.out_dir / alias
    resume_plan = planned_resume_checkpoint(cfg) if ctx.resume else None
    write_status(exp_dir, {"status": "training", "resume_enabled": ctx.resume, "planned_resume": resume_plan})
    append_event(ctx, alias, "train_start", resume_enabled=ctx.resume, planned_resume=resume_plan)
    ckpt_path = run_training_from_config(cfg, resume=ctx.resume, test_after_fit=False)
    if not ckpt_path or not Path(ckpt_path).is_file():
        fallback_ckpt = resolve_checkpoint(exp_dir, cfg)
        append_event(ctx, alias, "train_checkpoint_fallback", returned_ckpt=ckpt_path, resolved_ckpt=fallback_ckpt)
        ckpt_path = fallback_ckpt
    write_status(exp_dir, {"status": "trained", "resume_enabled": ctx.resume, "ckpt_path": ckpt_path})
    append_event(ctx, alias, "train_complete", ckpt_path=ckpt_path)
    return ckpt_path


def build_experiment_summary(
    experiment: dict[str, Any],
    ckpt_path: str,
    stage_name: str,
    metrics: dict[str, float],
    started_at: float,
) -> dict[str, Any]:
    return {
        "alias": str(experiment["alias"]),
        "group": str(experiment.get("group", "")),
        "desc": str(experiment.get("desc", "")),
        "ckpt_path": str(ckpt_path),
        "final_stage": str(stage_name),
        "elapsed_sec": round(time.time() - started_at, 3),
        "metrics": dict(metrics),
    }


def write_suite_summary(ctx: SuiteContext, summaries: list[dict[str, Any]]) -> None:
    payload = {
        "suite_config": str(ctx.suite_path),
        "out_dir": str(ctx.out_dir),
        "experiment_count": len(summaries),
        "experiments": summaries,
    }
    write_json(ctx.out_dir / "summary.json", payload)
    write_json(ctx.out_dir / "suite_summary.json", payload)
    write_summary_csv(ctx.out_dir / "summary.csv", summaries)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    metric_keys = sorted({key for item in summaries for key in dict(item.get("metrics", {})).keys()})
    fieldnames = ["alias", "group", "desc", "ckpt_path", "final_stage", "elapsed_sec", *metric_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            row = {key: item.get(key, "") for key in fieldnames}
            row.update(dict(item.get("metrics", {})))
            writer.writerow(row)


def print_dry_run(experiments: list[dict[str, Any]]) -> None:
    print(f"[Suite] dry-run 实验数: {len(experiments)}")
    for item in experiments:
        keys = sorted(dict(item.get("overrides", {}) or {}).keys())
        print(f"- {item.get('alias')} [{item.get('group')}] overrides={keys}")


def main() -> None:
    args = parse_args()
    ctx = load_suite_context(args)
    experiments = selected_experiments(ctx)
    if bool(args.dry_run):
        print_dry_run(experiments)
        return

    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_experiment(ctx, experiment) for experiment in experiments]
    write_suite_summary(ctx, summaries)
    print(f"[Suite] 完成 {len(summaries)} 个实验，汇总已写入: {ctx.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()