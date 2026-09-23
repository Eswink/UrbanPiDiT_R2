from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from baselines.train_external_baselines import (
    JsonlSummaryWriter,
    _checkpoint_config,
    _checkpoint_payload,
    _checkpoint_top_k_config,
    _load_top_k_checkpoints,
    _load_training_checkpoint,
    _make_summary_writer,
    _resolve_resume_checkpoint,
    _save_training_checkpoint,
    _tensorboard_config,
    _update_top_k_checkpoints,
)


def test_checkpoint_config_and_resume_resolution(tmp_path: Path):
    cfg = _checkpoint_config({
        "checkpoint": {
            "enabled": True,
            "save_last": True,
            "resume": True,
            "resume_path": None,
            "load_optimizer": True,
        }
    })
    assert cfg["enabled"] is True
    assert cfg["resume"] is True

    model_dir = tmp_path / "runs" / "fourcastnet_same_static"
    model_dir.mkdir(parents=True)
    last = model_dir / "last.pt"
    last.write_bytes(b"placeholder")

    resolved = _resolve_resume_checkpoint(
        model_dir=model_dir,
        alias="fourcastnet_same_static",
        ckpt_cfg=cfg,
    )
    assert resolved == last

    resolved_from_dir = _resolve_resume_checkpoint(
        model_dir=model_dir,
        alias="fourcastnet_same_static",
        ckpt_cfg={**cfg, "resume_path": str(tmp_path / "runs")},
    )
    assert resolved_from_dir == last


def test_checkpoint_roundtrip_restores_model_optimizer_and_global_step(tmp_path: Path):
    device = torch.device("cpu")
    model = torch.nn.Linear(3, 2)
    opt = AdamW(model.parameters(), lr=1e-3)
    x = torch.ones(4, 3)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    payload = _checkpoint_payload(
        model=model,
        optimizer=opt,
        cfg={"external_baseline_training": {"epochs": 2}},
        item={"name": "fourcastnet", "alias": "fourcastnet_same_static"},
        epoch=3,
        global_step=17,
        history=[{"epoch": 1, "global_step": 5, "val_mse": 1.0}],
        best_val=0.5,
        best_epoch=2,
        bad_epochs=1,
        early_cfg={"enabled": True, "monitor": "val_mse"},
        alias="fourcastnet_same_static",
    )
    path = tmp_path / "last.pt"
    _save_training_checkpoint(path, payload)

    restored = torch.nn.Linear(3, 2)
    restored_opt = AdamW(restored.parameters(), lr=1e-3)
    ckpt = _load_training_checkpoint(
        path=path,
        model=restored,
        optimizer=restored_opt,
        device=device,
        strict=True,
        load_optimizer=True,
    )

    assert ckpt["epoch"] == 3
    assert ckpt["next_epoch"] == 4
    assert ckpt["global_step"] == 17
    assert ckpt["bad_epochs"] == 1
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(p1, p2)
    assert restored_opt.state_dict()["state"]


def test_tensorboard_config_and_resumable_jsonl_fallback(tmp_path: Path, monkeypatch):
    tb_cfg = _tensorboard_config({
        "tensorboard": {
            "enabled": True,
            "log_dir": str(tmp_path / "tb"),
            "resume": True,
            "flush_secs": 1,
            "log_every_n_steps": 1,
            "write_jsonl_fallback": True,
        }
    })
    assert tb_cfg["enabled"] is True
    assert tb_cfg["resume"] is True

    # Force fallback path so this test does not require tensorboard to be installed.
    import baselines.train_external_baselines as trainer

    monkeypatch.setattr(trainer, "_TorchSummaryWriter", None)
    writer, info = _make_summary_writer(
        tb_cfg,
        model_dir=tmp_path / "out" / "fourcastnet_same_static",
        alias="fourcastnet_same_static",
        global_step=42,
        resumed=True,
    )
    assert isinstance(writer, JsonlSummaryWriter)
    assert info["jsonl_fallback"] is True
    assert info["purge_step"] == 42

    writer.add_scalar("fourcastnet/train_step/mse", 0.123, 43)
    writer.flush()
    writer.close()
    log_file = Path(info["log_dir"]) / "scalars.jsonl"
    text = log_file.read_text()
    assert '"purge_step": 42' in text
    assert '"global_step": 43' in text


def test_top_k_checkpoint_keeps_best_scores_and_deletes_dropped_files(tmp_path: Path):
    top_k_cfg = _checkpoint_top_k_config(
        {"save_top_k": 2, "monitor": "val_mse", "mode": "min"},
        {"monitor": "val_mse", "mode": "min"},
    )
    entries = []
    payload = {"model_state_dict": {}, "optimizer_state_dict": {}}
    scores = {1: 0.5, 2: 0.4, 3: 0.6, 4: 0.3}

    for epoch, score in scores.items():
        entries = _update_top_k_checkpoints(
            entries=entries,
            model_dir=tmp_path,
            payload={**payload, "epoch": epoch},
            epoch=epoch,
            score=score,
            monitor=top_k_cfg["monitor"],
            mode=top_k_cfg["mode"],
            save_top_k=top_k_cfg["save_top_k"],
        )

    assert [entry["epoch"] for entry in entries] == [4, 2]
    assert [entry["score"] for entry in entries] == [0.3, 0.4]
    for entry in entries:
        assert Path(entry["path"]).exists()
    assert not any("epoch_0001" in str(path) for path in (tmp_path / "top_k").glob("*.pt"))
    assert not any("epoch_0003" in str(path) for path in (tmp_path / "top_k").glob("*.pt"))

    loaded = _load_top_k_checkpoints(tmp_path, top_k_cfg)
    assert [entry["epoch"] for entry in loaded] == [4, 2]


def test_top_k_checkpoint_supports_max_mode(tmp_path: Path):
    top_k_cfg = _checkpoint_top_k_config(
        {"save_top_k": 2, "monitor": "val_acc", "mode": "max"},
        {"monitor": "val_mse", "mode": "min"},
    )
    entries = []
    for epoch, score in {1: 0.2, 2: 0.8, 3: 0.5}.items():
        entries = _update_top_k_checkpoints(
            entries=entries,
            model_dir=tmp_path,
            payload={"epoch": epoch},
            epoch=epoch,
            score=score,
            monitor=top_k_cfg["monitor"],
            mode=top_k_cfg["mode"],
            save_top_k=top_k_cfg["save_top_k"],
        )

    assert [entry["epoch"] for entry in entries] == [2, 3]
    assert [entry["score"] for entry in entries] == [0.8, 0.5]
