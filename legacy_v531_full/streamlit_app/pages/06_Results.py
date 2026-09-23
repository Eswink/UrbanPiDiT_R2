"""实验结果看板页。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils import db
from utils.metrics import find_result_files, parse_result_json, rows_for_db
from utils.paths import PROJECT_ROOT
from utils.viz import plot_metric_curve, plot_metric_heatmap

st.set_page_config(page_title="Results", layout="wide")


def main() -> None:
    """渲染结果页。"""

    db.init_db()
    st.title("结果看板")

    experiments = db.list_experiments(limit=1000)
    if not experiments:
        st.info("暂无实验记录。")
        return

    selected_ids = st.multiselect(
        "选择实验",
        [int(item["id"]) for item in experiments],
        default=[int(experiments[0]["id"])],
        format_func=lambda x: _format_experiment(x, experiments),
    )
    if not selected_ids:
        st.info("请选择至少一个实验。")
        return

    metrics = db.list_metrics(selected_ids)
    if not metrics:
        st.warning("选中的实验还没有结构化指标。可以先在 Evaluation/Baselines 页面导入 JSON。")
        _render_import_any_json(selected_ids)
        return

    metrics_df = pd.DataFrame(metrics)
    meta_df = pd.DataFrame(experiments)[["id", "name", "kind", "status", "config_path"]]
    metrics_df = metrics_df.merge(meta_df, left_on="experiment_id", right_on="id", how="left", suffixes=("", "_exp"))

    st.subheader("指标表")
    split_filter = st.multiselect("split", sorted(metrics_df["split"].dropna().unique()), default=sorted(metrics_df["split"].dropna().unique()))
    metric_filter = st.multiselect(
        "metric",
        sorted(metrics_df["metric_name"].dropna().unique()),
        default=sorted(metrics_df["metric_name"].dropna().unique())[:3],
    )
    shown = metrics_df[
        metrics_df["split"].isin(split_filter) & metrics_df["metric_name"].isin(metric_filter)
    ].copy()
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button(
        "下载当前表格 CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="urbanpidit_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("交互图表")
    plot_rows = shown.to_dict(orient="records")
    if not plot_rows:
        st.info("当前筛选条件下没有指标。")
        return

    col1, col2, col3, col4 = st.columns(4)
    metric_name = col1.selectbox("图表指标", sorted(shown["metric_name"].dropna().unique()))
    split = col2.selectbox("图表 split", sorted(shown["split"].dropna().unique()))
    variable = col3.selectbox("趋势变量", sorted(shown["variable"].dropna().unique()))
    model_name = col4.selectbox("子模型", sorted(shown["model_name"].dropna().unique()))

    tab1, tab2 = st.tabs(["变量 × lead 热力图", "lead 趋势曲线"])
    with tab1:
        fig = plot_metric_heatmap(plot_rows, metric_name=metric_name, split=split, model_name=model_name)
        st.pyplot(fig, use_container_width=True)
    with tab2:
        fig = plot_metric_curve(
            plot_rows,
            metric_name=metric_name,
            split=split,
            variable=variable,
            model_name=model_name,
        )
        st.pyplot(fig, use_container_width=True)

    _render_import_any_json(selected_ids)


def _render_import_any_json(selected_ids: list[int]) -> None:
    st.subheader("导入任意结果 JSON")
    result_files = find_result_files(PROJECT_ROOT)
    if not result_files:
        st.info("项目目录下没有找到可导入 JSON。")
        return
    json_path = st.selectbox("JSON 文件", result_files, format_func=lambda p: str(Path(p).relative_to(PROJECT_ROOT)))
    target_id = st.selectbox("写入实验", selected_ids)
    if st.button("解析并写入", use_container_width=True):
        parsed = parse_result_json(json_path)
        db.replace_metrics(int(target_id), rows_for_db(parsed))
        st.success(f"已写入 {len(parsed)} 条指标。")
        st.rerun()


def _format_experiment(experiment_id: int, rows: list[dict]) -> str:
    for row in rows:
        if int(row["id"]) == int(experiment_id):
            return f"#{row['id']} | {row.get('kind')} | {row.get('name')}"
    return str(experiment_id)


if __name__ == "__main__":
    main()