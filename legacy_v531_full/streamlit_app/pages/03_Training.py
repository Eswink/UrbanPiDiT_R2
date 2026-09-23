"""训练启动与监控页。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import db
from utils.config import config_hash, list_config_files, load_yaml
from utils.paths import CONFIG_DIR, PROJECT_ROOT
from utils.runner import make_train_command, read_log_tail, refresh_all_running, spawn_experiment, stop_experiment, timestamp_slug

st.set_page_config(page_title="Training", layout="wide")


def main() -> None:
    """渲染训练页。"""

    db.init_db()
    refresh_all_running()
    st.title("训练启动与监控")

    configs = list_config_files()
    if not configs:
        st.error(f"没有找到配置文件：{CONFIG_DIR}")
        return

    with st.form("train_form"):
        selected = st.selectbox(
            "配置文件",
            configs,
            format_func=lambda p: str(Path(p).relative_to(CONFIG_DIR)),
        )
        cfg = load_yaml(selected)
        default_name = f"train_{Path(selected).stem}_{timestamp_slug()}"
        name = st.text_input("实验名称", value=default_name)
        stage = st.selectbox("阶段", ["single", "two_stage_auto"])
        resume = st.checkbox("自动断点续训", value=False)
        resume_ckpt = st.text_input("指定 resume_ckpt（可选）", value="")
        notes = st.text_area("备注", value="", height=80)
        submitted = st.form_submit_button("开始训练", type="primary", use_container_width=True)

    if submitted:
        config_path = str(selected)
        command = make_train_command(config_path, resume=resume, resume_ckpt=resume_ckpt.strip() or None)
        exp_id = spawn_experiment(
            name=name,
            kind="train",
            config_path=config_path,
            config_hash=config_hash(cfg),
            stage=stage,
            command=command,
            log_dir=str(PROJECT_ROOT / str(cfg.get("logging", {}).get("log_dir", "logs"))),
            notes=notes,
        )
        st.success(f"训练已启动，实验 ID：{exp_id}")
        st.code(" ".join(command), language="bash")

    st.subheader("运行中训练")
    rows = [item for item in db.list_experiments(limit=200, kind="train") if item.get("status") == "running"]
    if not rows:
        st.info("当前没有运行中的训练。")
    else:
        for item in rows:
            _render_running_item(item)

    st.subheader("训练历史")
    history = db.list_experiments(limit=100, kind="train")
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("暂无训练历史。")


def _render_running_item(item: dict) -> None:
    with st.expander(f"#{item['id']} | {item['name']} | PID {item.get('pid')}", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 4])
        col1.metric("状态", item.get("status"))
        col2.metric("开始时间", item.get("start_time"))
        if col3.button("终止", key=f"stop_{item['id']}"):
            if stop_experiment(int(item["id"])):
                st.success("已发送终止信号。")
                st.rerun()
            st.error("终止失败。")
        st.code(read_log_tail(item.get("log_file"), max_lines=180), language="text")


if __name__ == "__main__":
    main()