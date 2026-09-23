from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_ROOT = ROOT / "streamlit_app"
sys.path.insert(0, str(STREAMLIT_ROOT))

from utils import db
from utils.config import build_edited_config, config_hash, load_yaml, save_yaml, validate_config
from utils.metrics import parse_result_json, rows_for_db
from utils.runner import make_eval_command, make_train_command, read_log_tail, safe_name
from utils.viz import pivot_metric_matrix, plot_metric_curve, plot_metric_heatmap


@pytest.fixture()
def isolated_ui_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ui.sqlite3"
    monkeypatch.setenv(db.DB_PATH_ENV, str(db_path))
    db.init_db()
    return db_path


def test_streamlit_management_flow_without_real_training(tmp_path, isolated_ui_db):
    base_config = ROOT / "configs" / "beijing.yaml"
    cfg = load_yaml(base_config)
    edited = build_edited_config(
        cfg,
        {
            "train.batch_size": 1,
            "train.num_workers": 0,
            "train.max_epochs": 1,
            "logging.run_name": "ui_integration_test",
        },
    )
    assert validate_config(edited) == []

    saved_config = tmp_path / "ui_config.yaml"
    save_yaml(saved_config, edited)
    assert load_yaml(saved_config)["train"]["batch_size"] == 1

    command = make_train_command(str(saved_config), resume=True)
    assert command[:5] == ["conda", "run", "-p", "/3240608030/weather-q/weather", "python"]
    assert command[-3:] == ["--config", str(saved_config), "--resume"]

    log_file = tmp_path / "train.log"
    log_file.write_text("epoch=0\nloss=1.0\nepoch=1\nloss=0.5\n", encoding="utf-8")
    exp_id = db.create_experiment(
        name="ui integration train",
        kind="train",
        config_path=str(saved_config),
        config_hash=config_hash(edited),
        stage="single",
        command=" ".join(command),
        log_file=str(log_file),
        log_dir=str(tmp_path / "logs"),
        notes="isolated integration test",
    )
    db.update_experiment(exp_id, status="completed", return_code=0, end_time=db.now_iso(), best_ckpt=str(tmp_path / "best.ckpt"))

    experiment = db.get_experiment(exp_id)
    assert experiment is not None
    assert experiment["status"] == "completed"
    assert "loss=0.5" in read_log_tail(experiment["log_file"], max_lines=2)

    eval_command = make_eval_command(str(saved_config), experiment["best_ckpt"])
    assert eval_command[-4:] == ["--config", str(saved_config), "--ckpt", experiment["best_ckpt"]]

    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "test/RMSE_t2m@6h": 1.25,
                "test/MAE_t2m@6h": 0.75,
                "test/RMSE_t2m@12h": 1.5,
                "val/RMSE_t2m@6h": 1.1,
            }
        ),
        encoding="utf-8",
    )
    rows = parse_result_json(metrics_json)
    db.replace_metrics(exp_id, rows_for_db(rows))
    stored = db.list_metrics([exp_id])

    assert len(stored) == 4
    assert {row["metric_name"] for row in stored} == {"RMSE", "MAE"}
    assert {row["lead_time"] for row in stored} == {"6h", "12h"}

    variables, leads, matrix = pivot_metric_matrix(stored, metric_name="RMSE", split="test", model_name="UrbanPiDiT")
    assert variables == ["t2m"]
    assert leads == ["6h", "12h"]
    assert matrix.shape == (1, 2)

    heatmap = plot_metric_heatmap(stored, metric_name="RMSE", split="test", model_name="UrbanPiDiT")
    curve = plot_metric_curve(stored, metric_name="RMSE", split="test", variable="t2m", model_name="UrbanPiDiT")
    assert heatmap.axes
    assert curve.axes

    assert safe_name("ui integration train") == "ui_integration_train"
    assert len(db.list_experiments(limit=10, kind="train")) == 1
    assert isolated_ui_db.exists()
    assert os.environ[db.DB_PATH_ENV] == str(isolated_ui_db)