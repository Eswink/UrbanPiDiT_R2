"""配置编辑页。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from utils.config import (
    DYNAMIC_VARS,
    STATIC_VARS,
    build_edited_config,
    config_hash,
    dump_yaml,
    list_config_files,
    load_yaml,
    normalize_save_name,
    save_yaml,
    validate_config,
    yaml_text_to_dict,
)
from utils.paths import CONFIG_DIR

st.set_page_config(page_title="Config Editor", layout="wide")


def main() -> None:
    """渲染配置编辑器。"""

    st.title("配置编辑器")
    configs = list_config_files()
    if not configs:
        st.error(f"没有找到配置文件：{CONFIG_DIR}")
        return

    selected = st.selectbox(
        "选择基础配置",
        configs,
        format_func=lambda p: str(Path(p).relative_to(CONFIG_DIR)),
    )
    cfg = load_yaml(selected)
    mode = st.radio("编辑模式", ["结构化表单", "原始 YAML"], horizontal=True)

    if mode == "结构化表单":
        edited = _structured_form(cfg)
    else:
        edited = _raw_yaml_editor(cfg)

    warnings = validate_config(edited)
    if warnings:
        st.warning("\n".join(f"- {item}" for item in warnings))
    else:
        st.success(f"配置校验通过，hash={config_hash(edited)}")

    st.subheader("保存")
    col1, col2 = st.columns([2, 1])
    default_name = f"ui_{Path(selected).stem}.yaml"
    save_name = col1.text_input("另存为文件名", value=default_name)
    target = CONFIG_DIR / normalize_save_name(save_name)
    col2.write("目标路径")
    col2.code(str(target), language="text")

    save_col1, save_col2 = st.columns(2)
    if save_col1.button("另存为新配置", type="primary", use_container_width=True):
        if warnings:
            st.error("配置仍存在风险提示，请修正后保存。")
        else:
            save_yaml(target, edited)
            st.success(f"已保存：{target}")
    if save_col2.button("覆盖当前配置", use_container_width=True):
        if warnings:
            st.error("配置仍存在风险提示，请修正后保存。")
        else:
            save_yaml(selected, edited)
            st.success(f"已覆盖：{selected}")

    with st.expander("预览 YAML", expanded=False):
        st.code(dump_yaml(edited), language="yaml")


def _structured_form(cfg: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    with st.expander("数据与时空范围", expanded=True):
        col1, col2, col3 = st.columns(3)
        updates["data_root"] = col1.text_input("data_root", value=str(cfg.get("data_root", "")))
        updates["dynamic_vars"] = col2.multiselect(
            "dynamic_vars",
            DYNAMIC_VARS,
            default=[v for v in cfg.get("dynamic_vars", DYNAMIC_VARS) if v in DYNAMIC_VARS],
        )
        updates["static_vars"] = col3.multiselect(
            "static_vars",
            STATIC_VARS,
            default=[v for v in cfg.get("static_vars", STATIC_VARS) if v in STATIC_VARS],
        )
        col4, col5, col6, col7 = st.columns(4)
        updates["include_static"] = col4.checkbox("include_static", value=bool(cfg.get("include_static", True)))
        updates["broadcast_static"] = col5.checkbox("broadcast_static", value=bool(cfg.get("broadcast_static", True)))
        updates["k"] = col6.number_input("k", min_value=1, value=int(cfg.get("k", 4)))
        updates["delta_t"] = col7.number_input("delta_t", min_value=1, value=int(cfg.get("delta_t", 1)))
        col8, col9 = st.columns(2)
        updates["H"] = col8.number_input("H", min_value=1, value=int(cfg.get("H", 8)))
        updates["W"] = col9.number_input("W", min_value=1, value=int(cfg.get("W", 8)))

    with st.expander("Forecast", expanded=True):
        forecast = dict(cfg.get("forecast", {}) or {})
        col1, col2, col3 = st.columns(3)
        updates["forecast.time_step_hours"] = col1.number_input(
            "time_step_hours", min_value=1, value=int(forecast.get("time_step_hours", 6))
        )
        updates["forecast.lead_times"] = _int_list_input(col2, "lead_times", forecast.get("lead_times", [1, 2, 3, 4]))
        updates["forecast.eval_lead_times"] = _int_list_input(
            col3, "eval_lead_times", forecast.get("eval_lead_times", [1, 2, 3, 4])
        )
        col4, col5 = st.columns(2)
        updates["forecast.train_lead_time_sampling"] = col4.selectbox(
            "train_lead_time_sampling",
            ["uniform", "curriculum"],
            index=0 if forecast.get("train_lead_time_sampling") == "uniform" else 1,
        )
        updates["forecast.multi_horizon_inference"] = col5.selectbox(
            "multi_horizon_inference",
            ["direct", "autoregressive"],
            index=0 if forecast.get("multi_horizon_inference", "direct") == "direct" else 1,
        )

    with st.expander("模型与扩散", expanded=True):
        model = dict(cfg.get("model", {}) or {})
        diffusion = dict(cfg.get("diffusion", {}) or {})
        col1, col2, col3, col4 = st.columns(4)
        updates["model.D"] = col1.number_input("D", min_value=1, value=int(model.get("D", 256)))
        updates["model.depth"] = col2.number_input("depth", min_value=1, value=int(model.get("depth", 12)))
        updates["model.heads"] = col3.number_input("heads", min_value=1, value=int(model.get("heads", 4)))
        updates["model.mlp_ratio"] = col4.number_input("mlp_ratio", min_value=1.0, value=float(model.get("mlp_ratio", 4.0)))
        col5, col6, col7, col8 = st.columns(4)
        updates["model.drop_path_rate"] = col5.number_input(
            "drop_path_rate", min_value=0.0, max_value=1.0, value=float(model.get("drop_path_rate", 0.1))
        )
        updates["model.dropout"] = col6.number_input(
            "dropout", min_value=0.0, max_value=1.0, value=float(model.get("dropout", 0.1))
        )
        updates["diffusion.P_mean"] = col7.number_input("P_mean", value=float(diffusion.get("P_mean", -0.8)))
        updates["diffusion.P_std"] = col8.number_input("P_std", min_value=0.0, value=float(diffusion.get("P_std", 0.8)))
        updates["diffusion.sample_steps"] = st.number_input(
            "sample_steps", min_value=1, value=int(diffusion.get("sample_steps", 8))
        )

    with st.expander("训练与推理", expanded=True):
        train = dict(cfg.get("train", {}) or {})
        inference = dict(cfg.get("inference", {}) or {})
        col1, col2, col3, col4 = st.columns(4)
        updates["train.seed"] = col1.number_input("seed", value=int(train.get("seed", 42)))
        updates["train.batch_size"] = col2.number_input("batch_size", min_value=1, value=int(train.get("batch_size", 128)))
        updates["train.num_workers"] = col3.number_input("num_workers", min_value=0, value=int(train.get("num_workers", 8)))
        updates["train.max_epochs"] = col4.number_input("max_epochs", min_value=1, value=int(train.get("max_epochs", 200)))
        col5, col6, col7, col8 = st.columns(4)
        updates["train.lr"] = col5.number_input("lr", min_value=0.0, value=float(train.get("lr", 5e-4)), format="%.6f")
        updates["train.weight_decay"] = col6.number_input(
            "weight_decay", min_value=0.0, value=float(train.get("weight_decay", 1e-4)), format="%.6f"
        )
        updates["train.patience"] = col7.number_input("patience", min_value=1, value=int(train.get("patience", 20)))
        updates["inference.ensemble_size"] = col8.number_input(
            "ensemble_size", min_value=1, value=int(inference.get("ensemble_size", 20))
        )

    with st.expander("Ablation 开关", expanded=True):
        ablation = dict(cfg.get("ablation", {}) or {})
        keys = sorted(ablation.keys())
        cols = st.columns(4)
        for idx, key in enumerate(keys):
            updates[f"ablation.{key}"] = cols[idx % 4].checkbox(key, value=bool(ablation.get(key)))

    with st.expander("Logging", expanded=False):
        logging = dict(cfg.get("logging", {}) or {})
        col1, col2, col3 = st.columns(3)
        updates["logging.log_dir"] = col1.text_input("log_dir", value=str(logging.get("log_dir", "logs")))
        updates["logging.run_name"] = col2.text_input("run_name", value=str(logging.get("run_name", "urbanx_baseline")))
        updates["logging.use_tensorboard"] = col3.checkbox(
            "use_tensorboard", value=bool(logging.get("use_tensorboard", True))
        )
        updates["logging.use_wandb"] = st.checkbox("use_wandb", value=bool(logging.get("use_wandb", False)))

    return build_edited_config(cfg, updates)


def _raw_yaml_editor(cfg: dict[str, Any]) -> dict[str, Any]:
    text = st.text_area("YAML", value=dump_yaml(cfg), height=650)
    try:
        return yaml_text_to_dict(text)
    except Exception as exc:
        st.error(f"YAML 解析失败：{exc}")
        return cfg


def _int_list_input(container, label: str, value: list[int]) -> list[int]:
    text = container.text_input(label, value=", ".join(str(v) for v in value))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()