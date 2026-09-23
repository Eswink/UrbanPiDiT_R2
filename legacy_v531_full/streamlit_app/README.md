# UrbanPiDiT Streamlit 实验管理台

## 启动

```bash
conda activate /3240608030/weather-q/weather
cd /3240608030/weather-q/UrbanPiDiT_V4_with_baselines
streamlit run streamlit_app/app.py
```

如果环境中没有 Streamlit：

```bash
conda activate /3240608030/weather-q/weather
pip install -r streamlit_app/requirements.txt
```

## 功能范围

- 配置编辑：基于 `configs/*.yaml` 创建或覆盖实验配置。
- 训练管理：后台启动 `train.py`，实时查看日志，支持终止进程。
- 评估管理：后台启动 `evaluate.py`，支持选择已有 checkpoint。
- 基线实验：支持 fair forecast、linear、tree、mlp 基线入口。
- 结果看板：解析 `results.json`、`summary.json`、`metrics.json` 并写入 SQLite，展示表格、热力图和趋势曲线。

## 运行数据

- SQLite：`streamlit_app/experiments.sqlite3`
- UI 日志：`streamlit_app/run_logs/`
- 实验输出：沿用项目原有 `logs/` 和 `outputs/` 约定。