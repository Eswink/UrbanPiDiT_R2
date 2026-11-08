"""Reproduce Stage C1's preflight measurement, which the repository could not recompute.

Why this exists. Decision 0094 permanently closed Stage C1 on a measured claim: the divergence
term charges the analysis field 10.2x what it charges the model, 8/8 cases. But the C1 test file's
24 cases are all unit properties -- non-negative, zero on a satisfying field, channels matched by
name -- and nothing in the repository recomputes the 8-case comparison. Three different values for
the analysis field's mean |div| were recorded in this repository (5.64e-05 measured independently,
2.51e-05 on the result page, 9.4e-06 in the module comment), and none agreed.

This script recomputes the *analysis-field* half exactly, using the production functions:

  * ``divergence_penalty`` -- the hinge the round would have optimised, so the numbers are the
    ones C1's page reports (not the raw mean |div|, which is a different quantity).
  * ``spherical_divergence`` -- the repository's single operator.

It reads the store only; it does not load a model. The model half of the 10.2x needs a forward pass
and is deliberately out of scope: this script's job is to make the recorded analysis numbers
recomputable and to say precisely which quantity each one is.

    python tools/reproduce_c1_preflight.py                     # summary of all conventions
    python tools/reproduce_c1_preflight.py --cases 1000 5000   # specific window indices

Read-only: opens the store in mode "r", writes nothing, takes no GPU, changes no frozen criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The eight window indices C1's result page lists, verbatim from its per-case table.
C1_CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)

#: Values recorded in this repository for the analysis field, for side-by-side comparison.
RECORDED = {
    "result_page_mean_abs_div": 2.51e-05,
    "result_page_p95_abs_div": 7.30e-05,
    "module_comment_mean": 9.4e-06,
    "module_comment_p95": 5.6e-05,
}


def _store_and_fields(store: Path):
    """Open the store read-only and locate the wind channels and coordinates."""
    import numpy as np
    import torch
    import zarr

    root = zarr.open_group(str(store), mode="r")
    channels = list(root.attrs["channels"])
    for name in ("u10", "v10"):
        if name not in channels:
            raise SystemExit(f"store has no {name} channel; channels={channels}")
    latitude = torch.as_tensor(np.asarray(root["latitude"][:], dtype=np.float32))
    longitude = torch.as_tensor(np.asarray(root["longitude"][:], dtype=np.float32))
    return root, channels, latitude, longitude


def _denormalised_winds(root, channels, frame: int):
    """One frame's u10/v10 back in physical units (m/s)."""
    import numpy as np
    import torch

    mean = np.asarray(root["normalization_mean"][:], dtype=np.float64)
    std = np.asarray(root["normalization_std"][:], dtype=np.float64)
    iu, iv = channels.index("u10"), channels.index("v10")
    raw = np.asarray(root["state"][frame], dtype=np.float64)
    u = torch.as_tensor(raw[iu] * std[iu] + mean[iu], dtype=torch.float64).unsqueeze(0)
    v = torch.as_tensor(raw[iv] * std[iv] + mean[iv], dtype=torch.float64).unsqueeze(0)
    return u, v


def measure(store: Path, frames: tuple[int, ...]) -> list[dict]:
    """For each frame: the hinge the round optimises, and the field's raw distribution."""
    import numpy as np
    import torch

    from model.r7_process_tensor_diagnostics import spherical_divergence
    from training.r7_physical_constraints import (
        DIVERGENCE_SCALE,
        DIVERGENCE_TOLERANCE,
        divergence_penalty,
    )

    root, channels, latitude, longitude = _store_and_fields(store)
    total = int(root["state"].shape[0])
    rows = []
    for frame in frames:
        if not 0 <= frame < total:
            raise SystemExit(f"frame {frame} outside the store's {total} frames")
        u, v = _denormalised_winds(root, channels, frame)
        field = spherical_divergence(u, v, latitude, longitude)[0].abs()
        flat = field.flatten()
        rows.append({
            "frame": frame,
            "hinge": float(divergence_penalty(u, v, latitude, longitude)),
            "mean_abs_div": float(flat.mean()),
            "p95_abs_div": float(torch.quantile(flat, 0.95)),
            "fraction_above_deadband": float((flat > DIVERGENCE_SCALE).double().mean()),
            "scale": float(DIVERGENCE_SCALE),
            "tolerance": float(DIVERGENCE_TOLERANCE),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    """Averages over the frames, plus the comparison against every recorded value."""
    import numpy as np

    summary = {
        "frames": len(rows),
        "mean_abs_div": float(np.mean([r["mean_abs_div"] for r in rows])),
        "p95_abs_div": float(np.mean([r["p95_abs_div"] for r in rows])),
        "mean_hinge": float(np.mean([r["hinge"] for r in rows])),
        "mean_fraction_above_deadband": float(np.mean([r["fraction_above_deadband"] for r in rows])),
    }
    ratios = {}
    for name, value in RECORDED.items():
        mine = summary["mean_abs_div"] if "mean" in name else summary["p95_abs_div"]
        ratios[name] = {"recorded": value, "measured": mine, "ratio": mine / value}
    summary["recorded_comparison"] = ratios
    summary["scale_units_from_non_divergent"] = summary["mean_abs_div"] / rows[0]["scale"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path,
                        default=ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr")
    parser.add_argument("--cases", type=int, nargs="*", default=list(C1_CASES))
    parser.add_argument("--frame-offset", type=int, default=13,
                        help="targets of window i are frames i+2..i+13, so +13 is its last lead")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    frames = tuple(case + args.frame_offset for case in args.cases)
    rows = measure(args.store, frames)
    summary = summarise(rows)
    report = {"store": str(args.store), "quantity_measured": "analysis field (store's own winds)",
              "cases": list(args.cases), "rows": rows, "summary": summary}

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
