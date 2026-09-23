"""评估运行页。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import db
from utils.config import config_hash, list_config_files, load_yaml
from utils.metrics import find_result_files, parse_result_json, rows_for_db
from utils.paths import CONFIG_DIR, PROJECT_ROOT
from utils.runner import make_eval_command, read_log_tail, refresh_all_running, spawn_experiment, timestamp_slug

st.set_page_config(page_title="Evaluation", layout="wide")


def main() -> None:
    """渲染评估页。"""

    db.init_db()
    refresh_all_running()
    st.title("评估运行器")

    configs = list_config_files()
    train_experiments = db.list_experiments(limit=500, kind="train")
    ckpt_candidates = _checkpoint_candidates(train_experiments)

    with st.form("eval_form"):
        selected = st.selectbox(
            "配置文件",
            configs,
            format_func=lambda p: str(Path(p).relative_to(CONFIG_DIR)),
        )
        ckpt_mode = st.radio("Checkpoint 来源", ["从候选列表选择", "手动输入"], horizontal=True)
        if ckpt_mode == "从候选列表选择" and ckpt_candidates:
            ckpt_path = st.selectbox("checkpoint", ckpt_candidates)
        else:
            ckpt_path = st.text_input("checkpoint 路径", value="")
        name = st.text_input("评估名称", value=f"eval_{Path(selected).stem}_{timestamp_slug()}")
        notes = st.text_area("备注", value="", height=80)
        submitted = st.form_submit_button("开始评估", type="primary", use_container_width=True)

    if submitted:
        if not ckpt_path:
            st.error("请提供 checkpoint 路径。")
        else:
            cfg = load_yaml(selected)
            command = make_eval_command(str(selected), str(ckpt_path))
            exp_id = spawn_experiment(
                name=name,
                kind="eval",
                config_path=str(selected),
                config_hash=config_hash(cfg),
                stage="test",
                command=command,
                log_dir=str(PROJECT_ROOT / str(cfg.get("logging", {}).get("log_dir", "logs"))),
                notes=notes,
            )
            st.success(f"评估已启动，实验 ID：{exp_id}")
            st.code(" ".join(command), language="bash")

    st.subheader("评估历史")
    rows = db.list_experiments(limit=200, kind="eval")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        selected_id = st.selectbox("查看日志", [int(item["id"]) for item in rows])
        item = db.get_experiment(selected_id)
        if item:
            st.code(read_log_tail(item.get("log_file"), max_lines=180), language="text")
    else:
        st.info("暂无评估记录。")

    st.subheader("导入已有 JSON 指标")
    result_files = find_result_files(PROJECT_ROOT)
    if not result_files:
        st.info("没有找到 results.json / summary.json / metrics.json。")
        return
    json_path = st.selectbox("指标 JSON", result_files, format_func=lambda p: str(Path(p).relative_to(PROJECT_ROOT)))
    target_ids = [int(item["id"]) for item in rows]
    if target_ids:
        target_id = st.selectbox("写入到评估实验", target_ids)
        if st.button("解析并写入指标", use_container_width=True):
            parsed = parse_result_json(json_path)
            db.replace_metrics(int(target_id), rows_for_db(parsed))
            st.success(f"已写入 {len(parsed)} 条指标。")
            st.dataframe(parsed, use_container_width=True, hide_index=True)


def _checkpoint_candidates(experiments: list[dict]) -> list[str]:
    candidates = []
    for item in experiments:
        if item.get("best_ckpt"):
            candidates.append(str(item["best_ckpt"]))
    for path in (PROJECT_ROOT / "logs").rglob("*.ckpt") if (PROJECT_ROOT / "logs").exists() else []:
        candidates.append(str(path))
    return sorted(dict.fromkeys(candidates))


if __name__ == "__main__":
    main()