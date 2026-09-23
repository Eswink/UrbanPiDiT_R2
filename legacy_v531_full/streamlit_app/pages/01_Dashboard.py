"""实验总览页。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import db
from utils.runner import read_log_tail, refresh_all_running, stop_experiment

st.set_page_config(page_title="Dashboard", layout="wide")


def main() -> None:
    """渲染实验总览。"""

    db.init_db()
    refresh_all_running()
    st.title("实验总览")

    experiments = db.list_experiments(limit=1000)
    total = len(experiments)
    running = sum(1 for item in experiments if item.get("status") == "running")
    completed = sum(1 for item in experiments if item.get("status") == "completed")
    failed = sum(1 for item in experiments if item.get("status") == "failed")
    killed = sum(1 for item in experiments if item.get("status") == "killed")

    cols = st.columns(5)
    cols[0].metric("总实验", total)
    cols[1].metric("运行中", running)
    cols[2].metric("已完成", completed)
    cols[3].metric("失败", failed)
    cols[4].metric("已终止", killed)

    if not experiments:
        st.info("暂无实验记录。")
        return

    status_filter = st.multiselect(
        "状态筛选",
        options=sorted({str(item.get("status")) for item in experiments}),
        default=sorted({str(item.get("status")) for item in experiments}),
    )
    kind_filter = st.multiselect(
        "类型筛选",
        options=sorted({str(item.get("kind")) for item in experiments}),
        default=sorted({str(item.get("kind")) for item in experiments}),
    )
    filtered = [
        item
        for item in experiments
        if str(item.get("status")) in status_filter and str(item.get("kind")) in kind_filter
    ]

    st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)

    st.subheader("实验操作")
    ids = [int(item["id"]) for item in filtered]
    selected_id = st.selectbox("选择实验", ids, format_func=lambda x: _format_experiment(x, filtered))
    selected = db.get_experiment(int(selected_id)) if selected_id else None
    if not selected:
        return

    col1, col2 = st.columns([1, 4])
    if col1.button("刷新状态", use_container_width=True):
        refresh_all_running()
        st.rerun()
    if selected.get("status") == "running" and col1.button("终止实验", type="primary", use_container_width=True):
        ok = stop_experiment(int(selected["id"]))
        if ok:
            st.success("已发送终止信号。")
            st.rerun()
        else:
            st.error("终止失败，请检查进程权限。")

    with col2.expander("日志尾部", expanded=True):
        st.code(read_log_tail(selected.get("log_file"), max_lines=160), language="text")


def _format_experiment(experiment_id: int, rows: list[dict]) -> str:
    for row in rows:
        if int(row["id"]) == int(experiment_id):
            return f"#{row['id']} | {row.get('kind')} | {row.get('status')} | {row.get('name')}"
    return str(experiment_id)


if __name__ == "__main__":
    main()