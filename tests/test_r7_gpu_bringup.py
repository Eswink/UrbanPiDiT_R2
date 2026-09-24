"""GPU bring-up invariants. Every GPU test self-skips when CUDA is unavailable.

CI runs a bare `pytest -q` with no marker filtering, so these must skip on the
CPU-only container rather than fail. Nothing here asserts forecast skill; the
checks are memory/timing-plumbing and distributed-correctness invariants.
"""
from __future__ import annotations
import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")


def _batch(batch_size, channels, height, width, device, anchored=None):
    history = torch.randn(batch_size, 2, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)
    batch = {"coarse_history": history, "atmos_target": target}
    if anchored:
        batch["process_targets"] = torch.randn(batch_size, anchored, device=device)
    return batch


@CUDA
def test_peak_memory_resets_and_streamed_stays_flat_in_k():
    """Streamed truncated training must not grow its peak with reasoning depth.

    This is the whole engineering claim of the streamed path, so it is asserted
    on real measurements rather than inferred from saved-tensor bookkeeping.
    """
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from training.r7_streaming import train_streamed_update

    device = torch.device("cuda:0")
    peaks = {}
    for steps in (1, 8):
        torch.manual_seed(0)
        model = GenericRecursiveWeatherForecaster(in_channels=4, dim=64, depth=2, heads=4,
            window_size=4, latent_tokens=8, default_reasoning_steps=1).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _batch(4, 4, 12, 12, device)
        train_streamed_update(model, optimizer, [batch], reasoning_steps=steps,
            process_weight=0.0, amp_dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        train_streamed_update(model, optimizer, [batch], reasoning_steps=steps,
            process_weight=0.0, amp_dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        peaks[steps] = torch.cuda.max_memory_allocated(device)
        del model, optimizer, batch
    # K=8 may exceed K=1 slightly (allocator granularity), but must not scale
    # the way retained activations would.
    assert peaks[8] < peaks[1] * 1.5, peaks


@CUDA
def test_full_bptt_peak_grows_where_streamed_stays_bounded():
    """The comparison the whole sweep rests on, asserted directly."""
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from model.r7_halting import forecast_inputs
    from training.r7_recursive_losses import deep_supervised_forecast_mse
    from training.r7_streaming import train_streamed_update

    device = torch.device("cuda:0")

    def peaks_for(steps, streamed):
        torch.manual_seed(0)
        model = GenericRecursiveWeatherForecaster(in_channels=4, dim=64, depth=2, heads=4,
            window_size=4, latent_tokens=8, default_reasoning_steps=1).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _batch(4, 4, 12, 12, device)
        for _ in range(2):
            if streamed:
                train_streamed_update(model, optimizer, [batch], reasoning_steps=steps,
                    process_weight=0.0, amp_dtype=torch.bfloat16)
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(forecast_inputs(batch), reasoning_steps=steps)
                    loss = deep_supervised_forecast_mse(
                        out.draft_forecasts, batch["atmos_target"], None, final_weight=2.0)
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        if streamed:
            train_streamed_update(model, optimizer, [batch], reasoning_steps=steps,
                process_weight=0.0, amp_dtype=torch.bfloat16)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(forecast_inputs(batch), reasoning_steps=steps)
                loss = deep_supervised_forecast_mse(
                    out.draft_forecasts, batch["atmos_target"], None, final_weight=2.0)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        value = torch.cuda.max_memory_allocated(device)
        del model, optimizer, batch
        return value

    streamed_k1 = peaks_for(1, True)
    streamed_k8 = peaks_for(8, True)
    bptt_k1 = peaks_for(1, False)
    bptt_k8 = peaks_for(8, False)
    assert streamed_k8 / streamed_k1 < bptt_k8 / bptt_k1


@CUDA
def test_single_gpu_checkpoint_resume_is_bitwise_stable(tmp_path):
    """Resume on GPU must reproduce the uninterrupted run exactly."""
    import torch as th
    from training.r7_local_runner import run_local_updates
    from data.synthetic_atmos import SyntheticAtmosDataset

    dataset = SyntheticAtmosDataset(length=16, hw=(8, 8), channels=4)
    config = {"in_channels": 4, "dim": 32, "depth": 1, "heads": 4, "window_size": 4,
              "latent_tokens": 8, "default_reasoning_steps": 2}
    common = {"kind": "generic", "model_config": config, "data_identity": "test-fixture",
        "batch_size": 2, "steps": 4, "process_weight": 0.0, "bf16": True,
        "device_name": "cuda"}
    half_checkpoint, _ = run_local_updates(dataset, output_dir=str(tmp_path / "half"),
        total_updates=2, **common)
    resumed, _ = run_local_updates(dataset, output_dir=str(tmp_path / "resumed"),
        total_updates=4, resume=str(half_checkpoint), **common)
    reference, _ = run_local_updates(dataset, output_dir=str(tmp_path / "reference"),
        total_updates=4, **common)
    a = th.load(resumed, map_location="cpu", weights_only=True)
    b = th.load(reference, map_location="cpu", weights_only=True)
    assert a["updates"] == b["updates"] == 4
    assert all(th.equal(a["model"][k], b["model"][k]) for k in a["model"])


@CUDA
def test_ddp_sampler_partitions_dataset_without_duplication():
    """The two-rank index split must cover the dataset exactly once."""
    from torch.utils.data import DistributedSampler
    from data.synthetic_atmos import SyntheticAtmosDataset

    dataset = SyntheticAtmosDataset(length=32, hw=(8, 8), channels=4)
    pooled = []
    for rank in range(2):
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True,
            seed=42, drop_last=False)
        sampler.set_epoch(0)
        pooled.extend(list(sampler))
    assert sorted(pooled) == list(range(len(dataset)))
    assert len(pooled) == len(set(pooled))


def test_bringup_scripts_declare_expected_cli_flags():
    """CLI contract for the bring-up scripts, checked statically (no subprocess)."""
    expected = {"bench_r7_gpu_memory.py": {"--out", "--device", "--steps", "--bf16"},
                "bench_r7_ddp_smoke.py": {"--out", "--mode", "--steps", "--world-size"}}
    for name, flags in expected.items():
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        declared = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        declared.add(arg.value)
        assert flags <= declared, f"{name} missing {flags - declared}"


def test_sweep_freezes_protocol_before_any_step_and_reports_peaks():
    """R-006 mechanism: protocol persisted before the measured loop starts."""
    source = (ROOT / "scripts" / "bench_r7_gpu_memory.py").read_text(encoding="utf-8")
    write_at = source.index('folder / "protocol.json"')
    first_measure = source.index("rows, stopped_early = []")
    assert write_at < first_measure
    assert "scientific_claim" in source
    assert "reset_peak_memory_stats" in source
    assert "synchronize" in source


def test_sweep_supports_single_cell_isolation():
    """`reserved` carries the allocator high-water mark across in-process cases.

    The filters plus the isolating driver are what let a single cell be measured
    in a fresh process; without them the reserved column is contaminated. Both
    must stay, or the published table silently becomes wrong.
    """
    script = (ROOT / "scripts" / "bench_r7_gpu_memory.py").read_text(encoding="utf-8")
    for flag in ("--kinds", "--modes", "--ckpt"):
        assert flag in script, flag
    assert "args.kinds" in script and "args.modes" in script
    driver = (ROOT / "scripts" / "sweep_r7_gpu_isolated.sh").read_text(encoding="utf-8")
    assert "for KIND in" in driver and "for MODE in" in driver and "for CK in" in driver
    assert "fresh process" in driver.lower()
    # one invocation per cell, so no cell shares an interpreter with another
    assert driver.count("bench_r7_gpu_memory.py") == 1


def test_gpu_scripts_skip_cleanly_when_cuda_is_absent():
    """The CUDA guard in this module must be a skipif, never a hard failure."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert "skipif(not torch.cuda.is_available()" in source
