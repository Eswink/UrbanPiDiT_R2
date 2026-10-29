"""The pre-change reference reading for the static-conditioning wiring round (decision 0069).

Why this exists. Decision 0069 clause 2 says the ``static_conditioning=off`` path must be
bitwise the behaviour that existed before the pathway: not "looks the same". The only
comparison that can prove that is **before vs after the ``model/**`` edit** — two arms both
running the new code prove only that the switch is self-consistent.

So this tool records, on CPU only, in one deterministic pass:

- ``model_code_sha256`` of the tree it ran on (the edit changes it, by design);
- a digest of every parameter *at a fixed seed*, so a shifted initialization shows up;
- the loss of one fixed batch at a fixed seed, and
- a digest of every parameter after one optimizer step on that batch.

After the model edit the same command must reproduce the same *values*; only the recorded
``model_code_sha256`` may differ. The ordering is part of the evidence: a reference captured
after the edit proves nothing, so the record carries the git revision and the wall-clock
stamp of when it was produced.

It reads no real data, opens no store, touches no GPU, and its record is
``scientific_claim: false``. The batch is synthetic zeros-plus-ramp on the frozen 65x65 grid:
this is an implementation fingerprint, not a weather reading.

    .venv/bin/python scripts/r7_static_wiring_reference.py --out <path.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 41
LEAD_HOURS = 6.0
GRID = (65, 65)
CHANNELS, HISTORY = 17, 2
REASONING_STEPS = 1
LEARNING_RATE = 1e-4
#: A fixed, byte-stable revision of the batch: any change here changes the reference.
BATCH_SPEC = {
    "coarse_history": "zeros except channel 0 = ramp(0..4095)/4096, broadcast over T",
    "atmos_target": "zeros except channel 0 = ramp reversed /4096",
    "latitude": "linspace(43, 27, 65)", "longitude": "linspace(107, 123, 65)",
    "lead_time_hours": LEAD_HOURS, "init_utc_hour": 0.0,
    "init_day_of_year": 100.0, "init_calendar_year": 2017.0,
    "history_offsets_hours": [-6.0, 0.0],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(revision: str) -> dict:
    """The revision this reference was captured on: a dirty tree is not a reference."""
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "model_dirty": bool(run("status", "--porcelain", "--", "model")),
            "requested_revision": revision}


def fixed_batch():
    """The frozen fingerprint batch. Built here, not read, so the bytes cannot drift."""
    import torch
    rows, columns = GRID
    ramp = (torch.arange(rows * columns, dtype=torch.float32) / (rows * columns - 1))
    history = torch.zeros(1, HISTORY, CHANNELS, rows, columns)
    history[0, :, 0] = ramp.reshape(rows, columns)
    target = torch.zeros(1, CHANNELS, rows, columns)
    target[0, 0] = ramp.flip(0).reshape(rows, columns)
    return {
        "coarse_history": history, "atmos_target": target,
        "latitude": torch.linspace(43., 27., rows), "longitude": torch.linspace(107., 123., columns),
        "lead_time_hours": torch.tensor(LEAD_HOURS), "init_utc_hour": torch.tensor(0.),
        "init_day_of_year": torch.tensor(100.), "init_calendar_year": torch.tensor(2017.),
        "history_offsets_hours": torch.tensor([[-6., 0.]]),
    }


def parameter_digest(model) -> str:
    """Byte-stable digest over names, shapes, dtypes and values of every tensor."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().to(torch.float64).contiguous().numpy()
        digest.update(name.encode() + b"\0" + str(tuple(array.shape)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def measure() -> dict:
    """One deterministic build -> forward -> backward -> step, recorded as digests."""
    import torch
    from model.weather_forecaster_r7 import NativeAtmosForecaster  # noqa: F401 (import surface check)
    from scripts.study_r7_s3_fy_year_coverage_matched import _model_spec
    from training.r7_experiment import make_model, seed_everything, model_code_digest
    from training.r7_halting import per_sample_latitude_mse
    spec = _model_spec()
    batch = fixed_batch()
    with torch.random.fork_rng(devices=[]):
        seed_everything(SEED)
        model = make_model("process", spec)
    initial = parameter_digest(model)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch, reasoning_steps=REASONING_STEPS)
    loss = per_sample_latitude_mse(output.forecast, batch["atmos_target"],
                                   batch.get("latitude")).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite reference loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"model_code_sha256": model_code_digest(), "model_spec": spec,
            "model_spec_canonical_sha256": _spec_digest(spec), "n_parameters": n_parameters,
            "seed": SEED, "initial_parameter_digest": initial,
            "first_update_loss": float(loss.detach()),
            "post_update_parameter_digest": parameter_digest(model),
            "forecast_shape": list(output.forecast.shape)}


def _spec_digest(spec) -> str:
    from training.r7_experiment import canonical_digest
    return canonical_digest(spec)


def record(revision: str) -> dict:
    values = measure()
    return {"format": "r7-static-wiring-reference-v1", "purpose": (
                "pre/post-edit fingerprint for decision 0069 W2: the static_conditioning=off "
                "path must reproduce these values exactly"),
            "scientific_claim": False, "test_read": False, "gpu_used": False,
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "batch_spec": BATCH_SPEC, "learning_rate": LEARNING_RATE,
            "reasoning_steps": REASONING_STEPS, "grid": list(GRID),
            "revision": _git(revision), "values": values,
            "limitations": [
                "this is an implementation fingerprint on a synthetic fixed batch, not a "
                "weather reading; no real data, store or GPU is involved",
                "it pins the initial parameters and one optimizer step of the frozen R1 spec; "
                "it does not pin any trained-checkpoint arithmetic",
                "it proves nothing about the static-conditioning=on path, which is a new "
                "computation by construction",
            ]}


def compare(reference: dict, observed: dict) -> list[str]:
    """Field-by-field differences; an empty list means the off path is unchanged."""
    ref, now = reference["values"], observed["values"]
    keys = ("model_spec_canonical_sha256", "n_parameters", "seed", "initial_parameter_digest",
            "first_update_loss", "post_update_parameter_digest", "forecast_shape")
    return [f"{key}: {ref.get(key)!r} -> {now.get(key)!r}"
            for key in keys if ref.get(key) != now.get(key)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the reference record here (exclusive)")
    parser.add_argument("--compare", type=Path, metavar="REFERENCE",
                        help="compare the live tree against a captured reference instead")
    parser.add_argument("--revision", default="HEAD", help="recorded git revision (default HEAD)")
    args = parser.parse_args(argv)
    if args.compare is not None:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        differences = compare(reference, record(args.revision))
        print(json.dumps({"reference": str(args.compare), "differences": differences,
                          "passed": not differences}, ensure_ascii=False))
        return 0 if not differences else 1
    if args.out is None:
        parser.error("one of --out or --compare is required")
    payload = record(args.revision)
    with Path(args.out).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"out": str(args.out), "model_code_sha256": payload["values"]["model_code_sha256"],
                      "initial": payload["values"]["initial_parameter_digest"][:16],
                      "loss": payload["values"]["first_update_loss"],
                      "sha256": _sha256(Path(args.out))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
