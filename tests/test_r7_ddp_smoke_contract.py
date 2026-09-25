"""CPU tests for the #61 DDP smoke contract.

Covers the acceptance items that do not need a GPU: the eval depth comes only
from the reasoning parameter (AST + a dynamic fake model with a forward-hook
spy), resume rejects every run-contract change, sampler padding for lengths
that do not divide the world size is reported instead of denied, training-mode
labels are distinct, streamed+DDP is refused rather than substituted, and the
no_sync accumulation rule defers the all-reduce to the final microbatch.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "bench_r7_ddp_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("bench_r7_ddp_smoke_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ast_validation_depth_comes_only_from_the_reasoning_parameter():
    """No batch_loss call may pass the optimizer-update count as the depth."""
    module = _module()
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    batch_loss_calls = [node for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "batch_loss"]
    assert batch_loss_calls, "the AST must contain the batch_loss call sites"
    for node in batch_loss_calls:
        depth_arg = node.args[2] if len(node.args) >= 3 else None
        assert not (isinstance(depth_arg, ast.Attribute) and depth_arg.attr == "steps"), \
            f"batch_loss must never receive args.steps as the depth: {ast.dump(node)}"
    validation_calls = [node for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "run_validation"]
    assert validation_calls
    for node in validation_calls:
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        assert "reasoning_steps" in keywords, "validation must bind the depth explicitly"
        value = keywords["reasoning_steps"]
        assert isinstance(value, ast.Attribute) and value.attr == "reasoning_steps"
    # And the module is importable, so the AST assertions describe live code.
    assert hasattr(module, "run_validation")


class _CounterCell(nn.Module):
    def forward(self, x, context):
        return x


class _FakeRecursion(nn.Module):
    """Dynamic fake model: its cell fires exactly `reasoning_steps` times."""

    def __init__(self, honor_depth=True):
        super().__init__()
        self.cell = _CounterCell()
        self.honor_depth = honor_depth

    def forward(self, batch, reasoning_steps=None, **kwargs):
        steps = int(reasoning_steps) if self.honor_depth else 0
        x = batch["x"]
        for _ in range(steps):
            x = self.cell(x, None)
        return x


def test_validation_observes_the_requested_depth_not_the_update_count(monkeypatch):
    """`--steps 10 --reasoning-steps 4` must validate K=4, observed by the hook."""
    module = _module()
    seen_depths = []

    def spy_batch_loss(model, batch, steps, bf16):
        seen_depths.append(steps)
        out = model(batch, reasoning_steps=steps)
        return out.sum() * 0.0 + 0.5

    monkeypatch.setattr(module, "batch_loss", spy_batch_loss)
    model = _FakeRecursion().eval()
    batches = [{"sample_id": ["s0", "s1"], "x": torch.zeros(2, 3)},
               {"sample_id": ["s2"], "x": torch.zeros(1, 3)}]
    result = module.run_validation(model, batches, torch.device("cpu"),
                                   reasoning_steps=4, bf16=False)
    # The decoy update count (10) appears nowhere; only the reasoning depth does.
    assert seen_depths == [4, 4]
    assert result["observed_reasoning_steps"] == [4, 4]
    assert result["requested_reasoning_steps"] == 4
    assert result["seen_val_ids"] == ["s0", "s1", "s2"]
    assert result["val_losses"] == [0.5, 0.5]


def test_validation_refuses_a_depth_it_did_not_observe(monkeypatch):
    module = _module()

    def spy_batch_loss(model, batch, steps, bf16):
        model(batch, reasoning_steps=steps)
        return torch.tensor(0.5)

    monkeypatch.setattr(module, "batch_loss", spy_batch_loss)
    model = _FakeRecursion(honor_depth=False).eval()
    batches = [{"sample_id": ["s0"], "x": torch.zeros(1, 3)}]
    with pytest.raises(SystemExit, match="did not observe|refuses"):
        module.run_validation(model, batches, torch.device("cpu"),
                              reasoning_steps=4, bf16=False)


def _valid_checkpoint(module):
    training_mode = "full_bptt"
    dataset_signature_value = module.dataset_signature(32, 12)
    model_config = module.model_config_for_mode(training_mode)
    saved = {"format": module.CHECKPOINT_FORMAT,
             "model_code_sha256": module.model_code_digest(),
             "model_config": model_config,
             "world_size": 2, "seed": 42, "per_gpu_batch": 2, "accumulation": 1,
             "reasoning_steps": 4, "training_mode": training_mode,
             "dataset_signature": dataset_signature_value,
             "target_steps": 10, "updates": 5,
             "signature": module.canonical_digest({
                 "model": model_config, "seed": 42, "steps": 10,
                 "per_gpu_batch": 2, "accumulation": 1, "world_size": 2,
                 "reasoning_steps": 4, "training_mode": training_mode,
                 "dataset_signature": dataset_signature_value})}
    return saved, dict(world_size=2, model_code_sha256=module.model_code_digest(),
                       seed=42, per_gpu_batch=2, accumulation=1, reasoning_steps=4,
                       training_mode=training_mode,
                       dataset_signature_value=dataset_signature_value,
                       target_updates=10)


def test_resume_accepts_the_same_contract():
    module = _module()
    saved, kwargs = _valid_checkpoint(module)
    assert module.validate_resume_contract(saved, **kwargs) == 5


@pytest.mark.parametrize("field,changed", [
    ("seed", 43), ("per_gpu_batch", 4), ("accumulation", 2),
    ("reasoning_steps", 3), ("training_mode", "retained_truncated"),
    ("world_size", 1), ("dataset_signature", "0" * 8),
    ("model_code_sha256", "f" * 64),
])
def test_resume_rejects_every_contract_change(field, changed):
    module = _module()
    saved, kwargs = _valid_checkpoint(module)
    saved[field] = changed
    with pytest.raises(SystemExit, match=f"resume contract change rejected: {field}"):
        module.validate_resume_contract(saved, **kwargs)


def test_resume_rejects_missing_fields_old_formats_and_bad_endpoints():
    module = _module()
    saved, kwargs = _valid_checkpoint(module)
    incomplete = dict(saved)
    del incomplete["reasoning_steps"]
    with pytest.raises(SystemExit, match="cannot be verified"):
        module.validate_resume_contract(incomplete, **kwargs)
    stale = dict(saved, format="r7-ddp-smoke-v1")
    with pytest.raises(SystemExit, match="r7-ddp-smoke-v1"):
        module.validate_resume_contract(stale, **kwargs)
    tampered = dict(saved, signature="0" * 64)
    with pytest.raises(SystemExit, match="self-inconsistent"):
        module.validate_resume_contract(tampered, **kwargs)
    finished = dict(saved, updates=10)
    with pytest.raises(SystemExit, match="must exceed the saved updates"):
        module.validate_resume_contract(finished, **kwargs)


@pytest.mark.parametrize("length,padding,padded", [
    (32, 0, False),   # divisible: the historic smoke case
    (31, 1, True),    # odd length: DistributedSampler repeats one index
    (33, 1, True),
])
def test_sampler_partition_reports_padding_explicitly(length, padding, padded):
    module = _module()
    audit = module.check_sampler_partition(2, "train", shuffle=True, length=length)
    assert audit["dataset_length"] == length
    assert audit["covers_dataset"] is True, "padding must never drop a sample"
    assert audit["padding_samples"] == padding
    assert audit["count"] == length + padding
    assert audit["no_duplicates"] is (not padded)
    assert bool(audit["padded_indices"]) is padded
    assert sum(audit["per_rank_counts"]) == audit["count"]


def test_sampler_partition_without_padding_for_divisible_val_length():
    module = _module()
    audit = module.check_sampler_partition(2, "val", shuffle=False, length=12)
    assert audit["no_duplicates"] is True and audit["padding_samples"] == 0
    assert audit["per_rank_counts"] == [6, 6]


def test_training_mode_mapping_keeps_the_three_labels_distinct():
    module = _module()
    full = module.model_config_for_mode("full_bptt")
    retained = module.model_config_for_mode("retained_truncated")
    assert full["detach_between_steps"] is False
    assert retained["detach_between_steps"] is True
    # The mapping mutates a copy: the module config must stay untouched.
    assert module.MODEL_CONFIG["detach_between_steps"] is False
    with pytest.raises(ValueError, match="streamed_truncated"):
        module.model_config_for_mode("streamed_truncated")


@pytest.mark.parametrize("mode", ["ddp", "reference", "compare"])
def test_streamed_truncated_is_refused_not_silently_substituted(mode):
    module = _module()
    with pytest.raises(SystemExit, match="refusing"):
        module.refuse_unverified_combinations(mode, "streamed_truncated")
    assert module.refuse_unverified_combinations(mode, "full_bptt") is True
    assert module.refuse_unverified_combinations(mode, "retained_truncated") is True


@pytest.mark.parametrize("position,last,sync,distributed,deferred", [
    (0, 2, True, True, True),    # first of three: defer
    (1, 2, True, True, True),    # second of three: defer
    (2, 2, True, True, False),   # final microbatch: synchronize
    (0, 0, True, True, False),   # single-microbatch group: synchronize
    (0, 1, False, True, False),  # incomplete group: must synchronize
    (0, 2, True, False, False),  # world_size 1: no collective to defer
])
def test_no_sync_rule_defers_all_but_the_last_microbatch(position, last, sync,
                                                         distributed, deferred):
    module = _module()
    assert module.no_sync_for(position, last, sync, distributed) is deferred


def test_dataset_signature_binds_the_exact_lengths():
    module = _module()
    assert module.dataset_signature(32, 12) == module.dataset_signature(32, 12)
    assert module.dataset_signature(31, 12) != module.dataset_signature(32, 12)
    assert module.dataset_signature(32, 13) != module.dataset_signature(32, 12)


def test_accumulate_step_weights_samples_and_records_observed_depth():
    """CPU end-to-end microbatch step: per-sample weighting, observed K, grads."""
    module = _module()
    dataset = module.SyntheticAtmosDataset(**dict(module.DATASET_CONFIG, length=4))
    batches = [default_collate([dataset[0], dataset[1]]),
               default_collate([dataset[2], dataset[3]])]
    model_config = dict(module.MODEL_CONFIG, dim=8, depth=1, heads=2, window_size=2,
                        latent_tokens=4)
    model = module.GenericRecursiveWeatherForecaster(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SimpleNamespace(reasoning_steps=4, seed=42, per_gpu_batch=2)
    before = next(model.parameters()).detach().clone()
    handle, counts = module.attach_reasoning_counter(model)
    try:
        total, consumed, depth_used = module._accumulate_step(
            model, optimizer, batches, torch.device("cpu"), config, bf16=False,
            sync=False, counts=counts)
    finally:
        handle.remove()
    assert depth_used == 8, "two microbatches at K=4 must fire the cell 8 times"
    assert consumed == [0, 1, 2, 3]
    assert total > 0.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    # The step already ran inside _accumulate_step; the weights must have moved.
    assert not torch.equal(before, next(model.parameters()).detach())


def test_cli_keeps_the_original_surface_and_adds_the_contract_knobs():
    source = MODULE.read_text(encoding="utf-8")
    for flag in ("--mode", "--out", "--steps", "--per-gpu-batch", "--accumulation",
                 "--world-size", "--seed", "--reasoning-steps", "--resume", "--bf16"):
        assert flag in source, flag
    for flag in ("--training-mode", "--length", "--val-length"):
        assert flag in source, flag
