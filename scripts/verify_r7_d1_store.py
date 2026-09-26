"""Offline replay verification for the acquired D1 store (#63/#64).

Re-derives everything that can be re-derived without the network and fails
closed on any mismatch: source artifact hash against the acquisition receipt,
per-channel payload hashes against the store's physical state, time coverage
against the frozen D1 request, window placement against the declared time
ranges (windows never cross a boundary), and train-only normalization
statistics recomputed from the declared train range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(source_nc, receipt, manifest_dir, output_path):
    import pandas as pd
    import zarr

    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from data.download.read_plan_frozen import channel_plan, d1_request
    from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS

    checks, failures = {}, []
    receipt = _load_json(receipt)
    if receipt.get("status") != "downloaded-real-source":
        failures.append(f"receipt status {receipt.get('status')!r} is not a completed acquisition")
    if receipt.get("synthetic_fallback") is not False:
        failures.append("receipt does not declare synthetic_fallback=False")
    checks["receipt_status"] = receipt.get("status")

    derived = _sha256_file(source_nc)
    recorded = receipt["local_artifact"]["sha256"]
    checks["source_sha256_match"] = derived == recorded
    if derived != recorded:
        failures.append("source artifact bytes do not match the receipt SHA256")

    marker = Path(manifest_dir) / "BUILD_COMPLETE.json"
    complete = _load_json(marker) if marker.is_file() else {}
    checks["build_complete"] = complete.get("build_complete") is True
    if not checks["build_complete"]:
        failures.append("manifests lack a BUILD_COMPLETE.json with build_complete=true")

    store_path = Path(manifest_dir).parent / "cache.zarr"
    root = zarr.open_group(str(store_path), mode="r")
    if root.attrs.get("build_complete") is not True or root.attrs.get("schema_version") != 1:
        failures.append("store is not a complete schema_version=1 R7 store")
    times = pd.DatetimeIndex(root["time_ns"][:].astype("datetime64[ns]"))
    request = d1_request()
    expected = [pd.Timestamp(request["first_time"]) + pd.Timedelta(hours=6 * step)
                for step in range(request["steps"])]
    checks["time_coverage"] = [str(times[0]), str(times[-1])] == \
        [str(expected[0]), str(expected[-1])] and len(times) == request["steps"]
    if not checks["time_coverage"]:
        failures.append("store time axis does not match the frozen D1 request")

    names = list(root.attrs["channels"])
    planned = [row["channel"] for row in channel_plan()]
    checks["channel_order"] = names == planned
    if names != planned:
        failures.append("store channel order differs from the published 17-channel plan")

    state = np.asarray(root["state"][:], dtype=np.float32)
    mismatched = []
    for index, name in enumerate(names):
        payload = hashlib.sha256(
            np.ascontiguousarray(state[:, index], dtype="<f4").tobytes()).hexdigest()
        if payload != receipt["payloads"][name]["payload_sha256"]:
            mismatched.append(name)
    checks["payload_sha256_all_17"] = not mismatched
    if mismatched:
        failures.append(f"payload hash mismatch for channels: {mismatched}")

    ranges = root.attrs.get("split_time_ranges")
    checks["split_mode_time_ranges"] = root.attrs.get("split_mode") == "time_ranges" and bool(ranges)
    if not checks["split_mode_time_ranges"]:
        failures.append("store does not declare the time-range split mode")

    # Ownership comes from the shared reader helper, not a local re-implementation:
    # a replay that recomputes the rule can stay green while the readers disagree
    # (the 0/10 val/test validate_record defect this script used to miss).
    from data.r7_store import split_time_labels, validate_record
    from data.r7_evaluation import fit_training_climatology
    labels = split_time_labels(root)
    if labels is None:
        failures.append("store declares the time-range mode but yields no range labels")
        labels = np.full(len(times), "", dtype=object)
    split_of = labels
    stamps_ns = times.asi8
    train_mask = split_of == "train"
    # fp64 accumulation: fp32 means drift ~1e-4 on 400k rows and would false-fail
    train_values = state[train_mask].astype(np.float64)
    mean_expected = train_values.transpose(0, 2, 3, 1).reshape(-1, state.shape[1]).mean(0)
    std_expected = train_values.transpose(0, 2, 3, 1).reshape(-1, state.shape[1]).std(0)
    mean_stored = np.asarray(root["normalization_mean"][:])
    std_stored = np.asarray(root["normalization_std"][:])
    checks["normalization_train_only"] = bool(
        np.allclose(mean_stored, mean_expected, rtol=1e-4, atol=1e-5)
        and np.allclose(std_stored, std_expected, rtol=1e-4, atol=1e-5))
    if not checks["normalization_train_only"]:
        failures.append("stored normalization does not match train-range-only statistics")

    window_failures = []
    total_by_split = {}
    for split in ("train", "val", "test"):
        records = [json.loads(line) for line in
                   (Path(manifest_dir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        total_by_split[split] = len(records)
        for record in records:
            stamps = [pd.Timestamp(value) for value in record["history_times"]] + \
                [pd.Timestamp(record["target_time"])]
            indices = record["history_indices"] + [record["target_index"]]
            declared = split_of[indices[1]]  # init index
            if declared != split:
                window_failures.append(f"{record['sample_id']}: init belongs to {declared}")
            for stamp, index in zip(stamps, indices):
                if stamps_ns[index] != stamp.value:
                    window_failures.append(f"{record['sample_id']}: index/time mismatch")
                if split_of[index] != split:
                    window_failures.append(f"{record['sample_id']}: window crosses a split boundary")
            if (stamps[-1] - stamps[-2]) != pd.Timedelta(hours=6):
                window_failures.append(f"{record['sample_id']}: lead != +6h")
    checks["windows_within_ranges_and_exact"] = not window_failures
    checks["window_counts_by_split"] = total_by_split
    if window_failures:
        failures.append(f"window placement violations: {window_failures[:3]}")

    # The gap this closes: every published record must pass the *readers'* own
    # validator, and the train-only climatology must fit on the declared train
    # window rather than the whole store. Both used to fail on a range-mode store.
    reader_failures = []
    for split in ("train", "val", "test"):
        for record in [json.loads(line) for line in
                       (Path(manifest_dir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]:
            try:
                validate_record(root, record)
            except ValueError as error:
                reader_failures.append(f"{record['sample_id']}: {error}")
    checks["validate_record_all_splits"] = not reader_failures
    if reader_failures:
        failures.append(f"reader validation rejected published records: {reader_failures[:3]}")

    climatology = fit_training_climatology(store_path)
    counted = int(sum(climatology["counts"].values()))
    checks["climatology_train_steps"] = {
        "counted_steps": counted, "train_range_steps": int(train_mask.sum()),
        "store_steps": int(len(times))}
    if counted != int(train_mask.sum()):
        failures.append(
            f"training climatology counted {counted} steps but the declared train "
            f"range holds {int(train_mask.sum())} of {len(times)}")

    result = {
        "format": "r7-d1-replay-verification-v1",
        "scientific_claim": False,
        "verified": not failures,
        "checks": checks,
        "failures": failures,
        "source_artifact": {"path": str(source_nc), "sha256": derived,
                            "bytes": Path(source_nc).stat().st_size},
        "receipt_network_body_bytes": receipt.get("network_body_bytes"),
        "receipt_elapsed_seconds": receipt.get("elapsed_seconds"),
        "snapshot_id": receipt.get("snapshot_id"),
        "limitations": ["offline replay verifies identity and contracts, not scientific fitness"],
    }
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline replay verification for the D1 store.")
    parser.add_argument("--source", required=True, help="acquired D1 NetCDF")
    parser.add_argument("--receipt", required=True, help="acquisition receipt JSON")
    parser.add_argument("--manifests", required=True, help="published store manifest directory")
    parser.add_argument("--out", required=True, help="new verification JSON path")
    args = parser.parse_args()
    result = verify(args.source, args.receipt, args.manifests, args.out)
    print(json.dumps({"verified": result["verified"], "checks": result["checks"],
                      "failures": result["failures"]}, ensure_ascii=False, indent=1))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
