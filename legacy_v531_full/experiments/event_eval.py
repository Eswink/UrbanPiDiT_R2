from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from metrics.event_metrics import event_scores


def _load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    raise ValueError(f"unsupported tensor file: {path}")


def evaluate_event_tensor_files(pred_path: str | Path, target_path: str | Path, *, thresholds: Iterable[float], op: str = ">=") -> Dict:
    pred = _load_tensor(pred_path).float()
    target = _load_tensor(target_path).float()
    results = {}
    for threshold in thresholds:
        scores = event_scores(pred, target, threshold=float(threshold), op=op)
        results[str(threshold)] = {key: float(value.detach().cpu()) for key, value in scores.items()}
    return {"pred": str(pred_path), "target": str(target_path), "op": op, "thresholds": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate threshold event metrics from saved tensors")
    parser.add_argument("--pred", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--thresholds", required=True, type=str, help="comma separated thresholds")
    parser.add_argument("--op", type=str, default=">=")
    parser.add_argument("--out", type=str, default="outputs/experiments/event_eval/results.json")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    result = evaluate_event_tensor_files(args.pred, args.target, thresholds=thresholds, op=args.op)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()