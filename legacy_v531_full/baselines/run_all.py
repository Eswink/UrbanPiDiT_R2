from __future__ import annotations

import argparse
import json
from pathlib import Path

from .linear import run_linear_baseline
from .mlp import TrainCfg, run_mlp_baseline
from .tree import run_tree_baseline


def main():
    p = argparse.ArgumentParser(description="Run all requested baselines (linear/tree/mlp)")
    p.add_argument("--config", required=True, type=str, help="yaml config path")
    p.add_argument("--out_dir", type=str, default="outputs/baselines", help="output root")

    # Linear
    p.add_argument("--linear_kind", type=str, default="ridge", choices=["ridge", "lasso"])
    p.add_argument("--linear_alpha", type=float, default=1.0)
    p.add_argument("--linear_max_train_samples", type=int, default=0)

    # Tree
    p.add_argument("--tree_model", type=str, default="xgb", choices=["xgb", "rf"])
    p.add_argument("--tree_max_train_rows", type=int, default=0)
    p.add_argument("--tree_no_xy", action="store_true")

    # MLP
    p.add_argument("--mlp_epochs", type=int, default=50)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    p.add_argument("--mlp_hidden", type=int, default=1024)
    p.add_argument("--mlp_dropout", type=float, default=0.0)
    p.add_argument("--mlp_grad_clip", type=float, default=1.0)
    p.add_argument("--mlp_early_stop_patience", type=int, default=10)
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    linear_max = args.linear_max_train_samples if args.linear_max_train_samples and args.linear_max_train_samples > 0 else None
    tree_max = args.tree_max_train_rows if args.tree_max_train_rows and args.tree_max_train_rows > 0 else None

    results = {}
    results["linear"] = run_linear_baseline(
        config_path=args.config,
        kind=args.linear_kind,
        alpha=args.linear_alpha,
        max_train_samples=linear_max,
        out_dir=out_root / "linear",
    )
    results["tree"] = run_tree_baseline(
        config_path=args.config,
        model=args.tree_model,
        max_train_rows=tree_max,
        add_xy=(not bool(args.tree_no_xy)),
        out_dir=out_root / "tree",
    )
    train_cfg = TrainCfg(
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        hidden=args.mlp_hidden,
        dropout=args.mlp_dropout,
        grad_clip=args.mlp_grad_clip,
        early_stop_patience=args.mlp_early_stop_patience,
    )
    results["mlp"] = run_mlp_baseline(
        config_path=args.config,
        train_cfg=train_cfg,
        out_dir=out_root / "mlp",
        device=args.device,
    )

    with open(out_root / "summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
