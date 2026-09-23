"""Original seasonal checkpoint correction audit: validation only, no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from data.restore_pilot_cache import restore_pilot_cache
    from training.r7_correction_diagnostic import run_correction_diagnostic
    from training.r7_calibration_runner import file_sha256
    root, out = Path(args.artifact_root)/"forecasts", Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    checkpoints = {s:root/"training"/f"process_{s}"/"update_0000200.pt" for s in (41,42,43)}
    original_hashes = {s:file_sha256(p) for s,p in checkpoints.items()}
    _, restoration = restore_pilot_cache(root/"source"/"era5_pressure_pilot.nc",
        root/"source"/"receipt.json", root/"manifests", checkpoints[41], out/"restored")
    manifest = out/"restored"/"manifests"/"val_balanced.jsonl"
    reports = {}
    for seed, checkpoint in checkpoints.items():
        reports[str(seed)] = run_correction_diagnostic(manifest, checkpoint=checkpoint,
            output=out/f"correction_{seed}.json", max_steps=3, max_samples=24)
        if len(reports[str(seed)]["initializations"]) != 24:
            raise ValueError("all predeclared validation cases required")
    if original_hashes != {s:file_sha256(p) for s,p in checkpoints.items()}:
        raise ValueError("original checkpoints changed")
    result = dict(format="r7-real-correction-audit-v1", scientific_claim=False,
        training_executed=False, test_evaluated=False, deployable=False, gpu_used=False,
        restoration=restoration, reports=reports, elapsed_seconds=time.monotonic()-started)
    with (out/"correction_audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(complete=True, validation_cases=24, seeds=[41,42,43],
        training_executed=False, elapsed_seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
