"""基线实验运行页。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import db
from utils.config import config_hash, list_config_files, load_yaml
from utils.metrics import find_result_files, parse_result_json, rows_for_db
from utils.paths import CONFIG_DIR, OUTPUT_DIR, PROJECT_ROOT
from utils.runner import (
    make_forecast_baseline_command,
    make_legacy_baseline_command,
    read_log_tail,
    refresh_all_running,
    spawn_experiment,
    timestamp_slug,
)

st.set_page_config(page_title="Baselines", layout="wide")


def main() -> None:
    """渲染基线页。"""

    db.init_db()
    refresh_all_running()
    st.title("基线运行器")

    configs = list_config_files()
    default_idx = _default_config_index(configs)

    with st.form("baseline_form"):
        config_path = st.selectbox(
            "配置文件",
            configs,
            index=default_idx,
            format_func=lambda p: str(Path(p).relative_to(CONFIG_DIR)),
        )
        family = st.selectbox(
            "基线类型",
            ["fair_forecast", "linear", "tree", "mlp"],
            format_func=_family_label,
        )
        options = _render_family_options(family)
        name = st.text_input("实验名称", value=f"baseline_{family}_{timestamp_slug()}")
        out_dir = st.text_input("输出目录", value=str(OUTPUT_DIR / "baselines" / name))
        notes = st.text_area("备注", value="", height=80)
        submitted = st.form_submit_button("开始基线实验", type="primary", use_container_width=True)

    if submitted:
        cfg = load_yaml(config_path)
        if family == "fair_forecast":
            command = make_forecast_baseline_command(str(config_path), out_dir)
        else:
            command = make_legacy_baseline_command(
                family=family,
                config_path=str(config_path),
                out_dir=out_dir,
                options=options,
            )
        exp_id = spawn_experiment(
            name=name,
            kind="baseline",
            config_path=str(config_path),
            config_hash=config_hash(cfg),
            stage=family,
            command=command,
            out_dir=out_dir,
            notes=notes,
        )
        st.success(f"基线实验已启动，实验 ID：{exp_id}")
        st.code(" ".join(command), language="bash")

    st.subheader("基线历史")
    rows = db.list_experiments(limit=200, kind="baseline")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        selected_id = st.selectbox("查看日志", [int(item["id"]) for item in rows])
        item = db.get_experiment(selected_id)
        if item:
            st.code(read_log_tail(item.get("log_file"), max_lines=180), language="text")
    else:
        st.info("暂无基线实验记录。")

    st.subheader("导入基线结果 JSON")
    result_files = find_result_files(PROJECT_ROOT / "outputs")
    if not result_files:
        st.info("outputs/ 下还没有可导入的 results.json 或 summary.json。")
        return
    json_path = st.selectbox("结果 JSON", result_files, format_func=lambda p: str(Path(p).relative_to(PROJECT_ROOT)))
    target_ids = [int(item["id"]) for item in rows]
    if target_ids:
        target_id = st.selectbox("写入到基线实验", target_ids)
        if st.button("解析并写入指标", use_container_width=True):
            parsed = parse_result_json(json_path)
            db.replace_metrics(int(target_id), rows_for_db(parsed))
            st.success(f"已写入 {len(parsed)} 条指标。")
            st.dataframe(parsed, use_container_width=True, hide_index=True)


def _default_config_index(configs: list[Path]) -> int:
    for idx, path in enumerate(configs):
        if path.name == "fair_baselines.yaml":
            return idx
    return 0


def _family_label(value: str) -> str:
    labels = {
        "fair_forecast": "公平静态对照基线",
        "linear": "Linear / Ridge / Lasso",
        "tree": "Tree / XGBoost / RandomForest",
        "mlp": "Simple MLP",
    }
    return labels.get(value, value)


def _render_family_options(family: str) -> dict:
    if family == "linear":
        col1, col2, col3 = st.columns(3)
        return {
            "kind": col1.selectbox("kind", ["ridge", "lasso"]),
            "alpha": col2.number_input("alpha", min_value=0.0, value=1.0),
            "max_train_samples": col3.number_input("max_train_samples", min_value=0, value=0),
        }
    if family == "tree":
        col1, col2, col3 = st.columns(3)
        return {
            "model": col1.selectbox("model", ["rf", "xgb", "rf_gpu"]),
            "max_train_rows": col2.number_input("max_train_rows", min_value=0, value=0),
            "tree_no_xy": col3.checkbox("禁用 xy 坐标特征", value=False),
        }
    if family == "mlp":
        col1, col2, col3, col4 = st.columns(4)
        return {
            "epochs": col1.number_input("epochs", min_value=1, value=50),
            "lr": col2.number_input("lr", min_value=0.0, value=1e-3, format="%.6f"),
            "hidden": col3.number_input("hidden", min_value=1, value=1024),
            "dropout": col4.number_input("dropout", min_value=0.0, max_value=1.0, value=0.0),
        }
    st.info("将运行 configs/fair_baselines.yaml 中定义的公平静态对照基线。")
    return {}


if __name__ == "__main__":
    main()