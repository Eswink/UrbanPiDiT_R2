"""Derive the wide train-only valid-time climatology anchor table for the H-D round (D2).

``docs/goals/s3-anchor-isolation.md`` needs ``C_valid`` on the registered 129x129 instance. The only
qualified anchor table in the repository is the 2026-10-07 65x65 one, which predates the wide store
and cannot be reused on this grid. This tool derives a new one from the **existing** wide store's
train split only -- no download, no store rebuild (decision 0038's D3 deferral is respected) -- and
writes it to a fresh exclusive ``outputs/`` path with the full identity the anchor spec requires.

The bucket is the ``(month, synoptic UTC hour)`` of the **valid** time, because that is what
``TrainClimatologyAnchor.forward`` queries (it advances init by the declared lead and then indexes
the bucket). The table is stored in the model's **normalized** space, which is what
``install`` documents and what the read path applies, so the physical mean is normalized here with
the store's own train-only normalization constants.

Two modes, and the read-only one is not optional in spirit: ``--preflight`` validates the store,
computes the bucket counts and prints every digest while writing nothing; ``--write`` repeats that
work and then publishes. A bucket with no train frame, a non-synoptic hour, or a normalization
constant that does not match the store is an error, never a filled-in default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STORE = ROOT / "outputs/r7_s3_wide_instance_v4_20261009_attempt01/store"
CACHE = STORE / "cache.zarr"
TRAIN_MANIFEST = STORE / "manifests/train.jsonl"
SOURCE_PREFLIGHT = STORE / "manifests/source_preflight.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_anchor_table_20261015_attempt01"
SYNOPTIC_HOURS = (0, 6, 12, 18)
FORMAT = "r7-train-climatology-anchor-v1"
KIND = "train-only-month-hour-grid-mean-v1"


def source_sha256() -> str:
    """The source the wide store was built from, read from the build's own record."""
    report = json.loads(SOURCE_PREFLIGHT.read_text(encoding="utf-8"))
    fingerprint = report["fingerprint"]
    if fingerprint.get("scope") != "full-local-file":
        raise ValueError("the store source fingerprint is not the full local file")
    return fingerprint["sha256"]


def table_digest(table) -> str:
    """Exactly the digest ``TrainClimatologyAnchor._table_digest`` computes."""
    return hashlib.sha256(table.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def _open_store():
    import zarr

    from data.r7_store import require_complete_manifest, validate_store
    require_complete_manifest(TRAIN_MANIFEST)
    root = zarr.open_group(str(CACHE), mode="r")
    validate_store(root)
    return root


def bucket_index(time_ns):
    """Map each store timestamp to its ``(month, synoptic hour)`` bucket.

    Uses UTC month/hour of the stored ns timestamps directly; the anchor's own month arithmetic is
    calendar-month, which for a UTC instant is the same month.
    """
    import numpy as np
    stamps = np.asarray(time_ns, dtype="datetime64[ns]")
    months = stamps.astype("datetime64[M]").astype(int) % 12 + 1
    hours = (stamps.astype("datetime64[h]").astype(int) % 24).astype(int)
    return months.astype(int), hours


def derive():
    """Read the store, average the train years per bucket, normalize, and return the payload."""
    import numpy as np
    import torch

    from data.r7_store import normalization
    from training.r7_experiment import dataset_identity
    root = _open_store()
    attrs = dict(root.attrs)
    channels = list(attrs["channels"])
    units = list(attrs["units"])
    state = root["state"]
    frames, n_channels, height, width = state.shape
    if (n_channels, height, width) != (len(channels), 129, 129):
        raise ValueError(f"unexpected wide store shape {state.shape} for {len(channels)} channels")
    latitude = [float(value) for value in np.asarray(root["latitude"][:])]
    longitude = [float(value) for value in np.asarray(root["longitude"][:])]
    mean, std = normalization(root, count=n_channels)
    train_years = sorted(int(year) for year in attrs["normalization_years"])
    if train_years != sorted(int(year) for year in attrs["split_years"]["train"]):
        raise ValueError("normalization years are not exactly the training years")

    time_ns = np.asarray(root["time_ns"][:])
    months, hours = bucket_index(time_ns)
    # ``time_ns`` is an integer ns array (validate_store requires kind 'i'), so it must be cast
    # before any datetime64 view; casting it directly to years would read the integers as years.
    stamps = time_ns.astype("datetime64[ns]")
    years = stamps.astype("datetime64[Y]").astype(int) + 1970
    train_mask = np.isin(years, np.asarray(train_years, dtype=years.dtype))
    if not train_mask.any():
        raise ValueError("the store holds no timestamp in the declared training years")
    odd = sorted({int(hour) for hour in hours[train_mask]} - set(SYNOPTIC_HOURS))
    if odd:
        raise ValueError(f"training timestamps include non-synoptic UTC hours {odd}")

    buckets = sorted({(int(month), int(hour)) for month, hour in zip(months[train_mask], hours[train_mask])})
    totals = {key: torch.zeros(n_channels, height, width, dtype=torch.float64) for key in buckets}
    counts = {key: 0 for key in buckets}
    for position in np.nonzero(train_mask)[0]:
        key = (int(months[position]), int(hours[position]))
        field = np.asarray(state[int(position)], dtype=np.float64)
        if not np.isfinite(field).all():
            raise ValueError(f"nonfinite store frame at index {int(position)}")
        totals[key] += torch.from_numpy(field)
        counts[key] += 1
    for key, count in counts.items():
        if count <= 0:
            raise ValueError(f"bucket {key} has no training frame")

    physical = torch.stack([(totals[key] / counts[key]).to(torch.float32) for key in buckets])
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32)
    table = ((physical - mean_t[None, :, None, None]) / std_t[None, :, None, None]).contiguous()
    if table.dtype != torch.float32 or not torch.isfinite(table).all():
        raise ValueError("the normalized climatology table is not finite FP32")
    spec = {
        "format": FORMAT, "kind": KIND, "channels": channels, "units": units,
        "source_sha256": source_sha256(),
        "train_data_identity": dataset_identity(TRAIN_MANIFEST)[0],
        "climatology_mean_identity_sha256": hashlib.sha256(
            physical.contiguous().numpy().tobytes(order="C")).hexdigest(),
        "normalization_mean": [float(value) for value in mean],
        "normalization_std": [float(value) for value in std],
        "latitude": latitude, "longitude": longitude,
        "bucket_keys": [[month, hour] for month, hour in buckets],
        "bucket_counts": [counts[key] for key in buckets],
        "training_years": train_years, "n_selected_steps": int(sum(counts.values())),
        "selection": "declared_train_years",
        "table_sha256": table_digest(table), "table_shape": list(table.shape),
    }
    receipt = {
        "format": "r7-s3-anchor-table-build-v1", "scientific_claim": False, "test_read": False,
        "store": str(CACHE), "train_manifest": str(TRAIN_MANIFEST),
        "store_attributes": {"channels": channels, "units": units, "shape": list(state.shape),
                             "normalization_years": train_years,
                             "split_years": attrs["split_years"]},
        "source_preflight_sha256": hashlib.sha256(SOURCE_PREFLIGHT.read_bytes()).hexdigest(),
        "n_train_timestamps": int(train_mask.sum()), "n_store_timestamps": int(frames),
        "bucket_counts": {f"{month:02d}T{hour:02d}": counts[(month, hour)] for month, hour in buckets},
        "selection": "store timestamps whose UTC year is a declared training year; no val/test year",
        "units_note": "physical_units_retained=true in the store; the table is published normalized",
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest(),
        "limitations": [
            "the climatology is a 4-month seasonal mean (the store's train split covers only "
            "Jan/Apr/Jul/Oct), so it is not a 12-month annual climatology",
            "the table is a mean of the store's train-year frames; it is a derived artifact and "
            "carries no scientific claim about forecast skill",
        ],
    }
    return spec, table, receipt


def self_check(spec, table):
    """Construct the anchor from the spec, install the table, and query one bucket."""
    import torch

    from model.climatology_anchor_r7 import TrainClimatologyAnchor
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(table)
    month, hour = spec["bucket_keys"][0]
    history = torch.zeros(1, 2, len(spec["channels"]), len(spec["latitude"]), len(spec["longitude"]))
    batch = {"latitude": torch.tensor(spec["latitude"]), "longitude": torch.tensor(spec["longitude"]),
             "init_calendar_year": torch.tensor([2017.]), "init_day_of_year": torch.tensor([1.]),
             "init_utc_hour": torch.tensor([float(hour)]), "lead_time_hours": torch.tensor([6.])}
    result = anchor(batch, history)
    return {"ready": bool(anchor.ready), "query_shape": list(result.shape),
            "queried_bucket": [month, hour], "query_finite": bool(torch.isfinite(result).all())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only: report, write nothing")
    mode.add_argument("--write", action="store_true", help="publish to a fresh exclusive path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    spec, table, receipt = derive()
    check = self_check(spec, table)
    summary = {"status": "preflight", "n_buckets": len(spec["bucket_keys"]),
               "table_shape": spec["table_shape"], "table_sha256": spec["table_sha256"],
               "climatology_mean_identity_sha256": spec["climatology_mean_identity_sha256"],
               "source_sha256": spec["source_sha256"], "train_data_identity": spec["train_data_identity"],
               "n_selected_steps": spec["n_selected_steps"],
               "bucket_counts": receipt["bucket_counts"], "self_check": check}
    if args.preflight:
        summary["out"] = None
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    import torch
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"fresh exclusive output only: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "anchor_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    torch.save(table, out / "anchor_table.pt")
    receipt["table_file_sha256"] = hashlib.sha256((out / "anchor_table.pt").read_bytes()).hexdigest()
    (out / "build_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "BUILD_COMPLETE.json").write_text(json.dumps(
        {"schema_version": 1, "build_complete": True, "format": receipt["format"],
         "table_sha256": spec["table_sha256"], "spec_sha256": receipt["spec_sha256"]},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reloaded = torch.load(out / "anchor_table.pt", map_location="cpu", weights_only=True)
    if table_digest(reloaded) != spec["table_sha256"] or not torch.equal(reloaded, table):
        raise ValueError("published table failed its own read-back")
    summary["status"] = "written"
    summary["out"] = str(out)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
