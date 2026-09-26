"""Offline tests for the B2 scheduled runner (#64 B2).

No network, no real data and no CUDA: the schedule arithmetic and the
selection/early-stopping contract are exercised on synthetic fixtures, and the
failure modes that matter for the B2 protocol (training that runs *longer* than
declared, validation that is never scored, a checkpoint whose update number
disagrees with the report) must fail closed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from data.synthetic_atmos import SyntheticAtmosDataset
from training.r7_scheduled_runner import run_scheduled_updates, warmup_cosine_factor


class RolloutFixture:
    """Held-out validation fixture with ZarrRolloutDataset's sample contract.

    The runner refuses the training dataset as validation, so these tests need a
    dataset that actually carries ``rollout_targets``. This is a fixture, not a
    data source: nothing here is weather truth.
    """

    def __init__(self, length=3, hw=(8, 12), channels=4, lead_hours=(6,)):
        self.length = int(length)
        self.hw = tuple(hw)
        self.channels = int(channels)
        self.lead_hours = tuple(int(lead) for lead in lead_hours)
        generator = torch.Generator().manual_seed(11)
        self.data = torch.randn(self.length, 3, self.channels, *self.hw,
                                generator=generator)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frames = self.data[index]
        latitude = torch.linspace(43.0, 27.0, self.hw[0])
        return {
            "coarse_history": frames[:2],
            "rollout_targets": frames[2:2 + len(self.lead_hours)],
            "latitude": latitude,
            "longitude": torch.linspace(107.0, 123.0, self.hw[1]),
            "lead_time_hours": torch.tensor(float(6)),
            "init_time": f"2016-01-{index + 1:02d}T00:00:00",
            "valid_times": [f"2016-01-{index + 1:02d}T0{h}:00:00" for h in range(1, 2)],
        }


def validation_dataset():
    return RolloutFixture()


def options(tmp_path, kind="native"):
    """Runner options pointing at a fresh run directory.

    The runner refuses an existing output directory (a published endpoint is
    write-once), so tests must not hand it the pytest tmp_path itself.
    """
    model = dict(in_channels=4, history_steps=2, dim=16, depth=1, heads=4,
                 window_size=4, dropout=0.0)
    if kind == "generic":
        model["latent_tokens"] = 4
    if kind == "process":
        model.update(anchored_processes=2, free_processes=2)
    if kind == "generic" or kind == "process":
        model["default_reasoning_steps"] = 2
    return dict(kind=kind, model_config=model, data_identity="synthetic-b2-test",
                output_dir=Path(tmp_path) / "run", batch_size=2, steps=2, seed=7, lr=2e-4,
                clip=1.0, process_weight=0.0, device_name="cpu")


def test_warmup_cosine_rises_then_decays_to_the_declared_floor():
    peak = warmup_cosine_factor(10, total_updates=100, warmup_updates=10, minimum_ratio=0.1)
    assert peak == pytest.approx(1.0)
    # warmup is monotone increasing and starts strictly below the peak
    series = [warmup_cosine_factor(u, total_updates=100, warmup_updates=10,
                                   minimum_ratio=0.1)
              for u in range(1, 11)]
    assert series[0] == pytest.approx(0.1)
    assert all(a < b for a, b in zip(series, series[1:]))
    # after warmup it decays and lands exactly on the floor at the endpoint
    post = [warmup_cosine_factor(u, total_updates=100, warmup_updates=10,
                                 minimum_ratio=0.1)
            for u in range(10, 101)]
    assert all(a >= b for a, b in zip(post, post[1:]))
    assert post[-1] == pytest.approx(0.1)


def test_warmup_cosine_rejects_impossible_schedules():
    with pytest.raises(ValueError, match="warmup_updates cannot exceed"):
        warmup_cosine_factor(1, total_updates=5, warmup_updates=10, minimum_ratio=0.1)
    with pytest.raises(ValueError, match="minimum_ratio"):
        warmup_cosine_factor(1, total_updates=5, warmup_updates=1, minimum_ratio=0.0)
    with pytest.raises(ValueError, match="minimum_ratio"):
        warmup_cosine_factor(1, total_updates=5, warmup_updates=1, minimum_ratio=1.5)


def test_training_stops_at_the_declared_endpoint_and_selects_a_checkpoint(tmp_path):
    ds = SyntheticAtmosDataset(length=4, hw=(8, 12), channels=4)
    checkpoint, report = run_scheduled_updates(
        ds, total_updates=6, warmup_updates=2, minimum_lr_ratio=0.25,
        validation_every=2, early_stopping_patience=None,
        validation_dataset=validation_dataset(), **options(tmp_path))
    assert checkpoint.is_file()
    assert checkpoint.parent == Path(tmp_path) / "run"
    assert report["updates_this_run"] == 6
    assert len(report["validations"]) == 3
    assert report["selected_update"] in (2, 4, 6)
    assert report["selection_split"] == "val"
    # the selected checkpoint is the one whose update number the report claims
    assert checkpoint.name == f"update_{report['selected_update']:07d}.pt"
    assert report["selected_validation_mse"] == min(
        entry["mean_mse"] for entry in report["validations"])
    # the learning rate followed the frozen schedule
    rates = [entry["lr"] for entry in report["losses"]]
    assert rates[0] == pytest.approx(report["contract"]["lr"] * 0.5)
    assert max(rates) <= report["contract"]["lr"] + 1e-12
    assert rates[-1] == pytest.approx(report["contract"]["lr"] * 0.25, rel=1e-6)


def test_validation_runs_do_not_perturb_the_training_stream(tmp_path):
    """Scoring validation must not consume the training RNG stream.

    If it did, the same seed would train differently depending on how often the
    schedule happened to look at validation, and the "same seed" comparison at
    the heart of B2 would be false.
    """
    ds = SyntheticAtmosDataset(length=4, hw=(8, 12), channels=4)
    _, with_validation = run_scheduled_updates(
        ds, total_updates=4, warmup_updates=1, validation_every=1,
        early_stopping_patience=None, validation_dataset=validation_dataset(),
        **options(tmp_path / "a"))
    _, without_validation = run_scheduled_updates(
        ds, total_updates=4, warmup_updates=1, validation_every=0,
        early_stopping_patience=None, **options(tmp_path / "b"))
    losses_a = [entry["loss"] for entry in with_validation["losses"]]
    losses_b = [entry["loss"] for entry in without_validation["losses"]]
    assert losses_a == pytest.approx(losses_b, rel=0, abs=0)


def test_early_stopping_reads_validation_and_only_shortens(tmp_path, monkeypatch):
    """Early stopping may stop shorter than the endpoint, never longer."""
    ds = SyntheticAtmosDataset(length=4, hw=(8, 12), channels=4)
    import training.r7_scheduled_runner as module

    calls = {"n": 0}
    real_score = module.score_validation

    def flat_score(*args, **kwargs):
        calls["n"] += 1
        score = real_score(*args, **kwargs)
        # Flat validation: no improvement ever, so patience must expire.
        return dict(score, mean_mse=1.0)

    monkeypatch.setattr(module, "score_validation", flat_score)
    _, report = run_scheduled_updates(
        ds, total_updates=20, warmup_updates=1, validation_every=2,
        early_stopping_patience=3, validation_dataset=validation_dataset(),
        **options(tmp_path))
    assert report["updates_this_run"] < 20
    assert report["early_stopped"] is True
    assert "early stop" in report["stopped_reason"]
    # The first check establishes the best score; patience then expires after
    # `patience` further non-improving checks, so 4 checks in total.
    assert calls["n"] == 1 + 3
    assert report["updates_this_run"] <= 20


def _generic_state(latent_tokens=4):
    from training.r7_experiment import make_model

    anchor = make_model("generic", {"in_channels": 4, "out_channels": 4, "history_steps": 2,
                                    "dim": 16, "depth": 1, "heads": 4, "window_size": 4,
                                    "patch_size": 2, "dropout": 0.0,
                                    "latent_tokens": latent_tokens})
    return {name: tensor.detach().clone() for name, tensor in anchor.state_dict().items()}


def test_shared_initial_state_transfers_every_common_parameter(tmp_path):
    """Same architecture: every parameter is common, so all of them transfer."""
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    _, report = run_scheduled_updates(
        ds, total_updates=1, warmup_updates=1, validation_every=0,
        shared_initial_state=_generic_state(), **options(tmp_path, kind="generic"))
    shared = report["shared_initial_state"]
    assert shared["provided"] is True
    assert shared["ignored_count"] == 0
    assert shared["applied_count"] > 0
    assert "backbone" in " ".join(shared["applied_parameters"])


def test_shared_initial_state_records_parameters_that_cannot_transfer(tmp_path):
    """The real B2 pair (generic -> process): the common part transfers and the
    arm-specific part is recorded as ignored.

    This is the property #64 B2 sentence 3 needs to be auditable - the alignment
    is reported with the names it applied and the names it could not apply, so
    "aligned common weights" is measured rather than assumed. generic and process
    share a backbone/draft-encoder/correction-head but not the latent tokens or
    the process queries.
    """
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    _, report = run_scheduled_updates(
        ds, total_updates=1, warmup_updates=1, validation_every=0,
        shared_initial_state=_generic_state(), **options(tmp_path, kind="process"))
    shared = report["shared_initial_state"]
    assert shared["applied_count"] > 0
    assert shared["ignored_count"] > 0
    applied = " ".join(shared["applied_parameters"])
    ignored = " ".join(shared["ignored_parameters"])
    assert "backbone" in applied and "draft_encoder" in applied
    # "ignored" is the anchor's generic-only part: the latent tokens and the
    # generic cell/latent_to_context have no counterpart in the process arm, so
    # they cannot be aligned. The process arm's own-only parameters are simply
    # not in the anchor and keep the seeded initialization.
    assert "latent" in ignored
    assert "cell." in ignored or "latent_to_context" in ignored
    # the two arms' shared backbone really was aligned, not merely attempted
    assert "backbone.encoder" in applied


def test_arms_with_no_shared_structure_cannot_be_force_aligned(tmp_path):
    """native and generic share no parameter names, so alignment fails closed.

    The B2 protocol therefore never claims weight alignment for U-Net/window/AFNO
    against the recursive arms - those are matched on data and budget only.
    """
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="matched no parameter"):
        run_scheduled_updates(ds, total_updates=1, warmup_updates=1, validation_every=0,
                              shared_initial_state=_generic_state(),
                              **options(tmp_path / "run", kind="native"))


def test_shared_initial_state_that_matches_nothing_fails_closed(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="matched no parameter"):
        run_scheduled_updates(ds, total_updates=1, warmup_updates=1, validation_every=0,
                              shared_initial_state={"not_a_parameter": torch.zeros(3)},
                              **options(tmp_path))


def test_published_endpoint_refuses_a_second_run(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=0,
                          **options(tmp_path))
    with pytest.raises(FileExistsError, match="already published"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=0,
                              **options(tmp_path))


def test_invalid_schedule_arguments_fail_closed(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="minimum_lr_ratio"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, minimum_lr_ratio=0.0,
                              validation_every=0, **options(tmp_path / "x"))
    with pytest.raises(ValueError, match="minimum_improvement"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1,
                              minimum_improvement=-1.0, validation_every=0,
                              **options(tmp_path / "y"))
    bad = options(tmp_path / "z")
    bad["lr"] = float("nan")
    with pytest.raises(ValueError, match="positive finite"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=0,
                              **bad)


def test_report_is_json_serializable_and_declares_no_scientific_claim(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    _, report = run_scheduled_updates(ds, total_updates=2, warmup_updates=1,
                                      validation_every=1,
                                      validation_dataset=validation_dataset(),
                                      **options(tmp_path))
    payload = json.loads((Path(tmp_path) / "run" / "training_report.json").read_text(
        encoding="utf-8"))
    assert payload["scientific_claim"] is False
    assert payload["selection_split"] == "val"
    assert payload["contract"]["optimization"] == "native"
    assert math.isfinite(payload["seconds_per_update"])


def test_training_dataset_is_refused_as_validation(tmp_path):
    """Validation scoring must reject a dataset without held-out targets."""
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="held-out"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=1,
                              validation_dataset=ds, **options(tmp_path))


def test_validation_required_when_checks_are_scheduled(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="requires an explicit held-out"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=1,
                              validation_dataset=None, **options(tmp_path))


def test_early_stopping_without_validation_checks_is_refused(tmp_path):
    ds = SyntheticAtmosDataset(length=3, hw=(8, 12), channels=4)
    with pytest.raises(ValueError, match="early stopping without validation"):
        run_scheduled_updates(ds, total_updates=2, warmup_updates=1, validation_every=0,
                              early_stopping_patience=2, **options(tmp_path))


def test_validation_scoring_is_denormalized_free_and_finite(tmp_path):
    """The scoring space is stated, deterministic and free of physical units."""
    from training.r7_experiment import make_model
    from training.r7_scheduled_runner import score_validation

    model = make_model("native", {"in_channels": 4, "out_channels": 4, "history_steps": 2,
                                  "dim": 16, "depth": 1, "heads": 4, "window_size": 4,
                                  "dropout": 0.0})
    dataset = validation_dataset()
    first = score_validation(model, dataset, kind="native", device=torch.device("cpu"),
                             steps=2, lead_hours=(6,))
    second = score_validation(model, dataset, kind="native", device=torch.device("cpu"),
                              steps=2, lead_hours=(6,))
    assert first["mean_mse"] == pytest.approx(second["mean_mse"], rel=0, abs=0)
    assert first["n_cases"] == len(dataset)
    assert first["split"] == "val"
    assert "normalized" in first["space"]
    assert math.isfinite(first["mean_mse"]) and first["mean_mse"] >= 0
    # horizons are integers; a float would silently mean a different cadence
    with pytest.raises(ValueError, match="positive integers"):
        score_validation(model, dataset, kind="native", device=torch.device("cpu"),
                         steps=2, lead_hours=(6.5,))
