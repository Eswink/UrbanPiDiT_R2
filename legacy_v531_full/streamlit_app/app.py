"""UrbanPiDiT Streamlit 实验管理入口。"""

from __future__ import annotations

import streamlit as st

from utils import db
from utils.paths import PROJECT_ROOT, STREAMLIT_ROOT
from utils.runner import refresh_all_running

st.set_page_config(
    page_title="UrbanPiDiT Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """渲染首页。"""

    db.init_db()
    refresh_all_running()

    st.title("UrbanPiDiT 实验管理台")
    st.caption("配置、训练、评估、基线和结果看板的统一入口")

    col1, col2, col3 = st.columns(3)
    col1.metric("项目目录", str(PROJECT_ROOT))
    col2.metric("UI 数据库", str(STREAMLIT_ROOT / "experiments.sqlite3"))
    col3.metric("运行环境", "/3240608030/weather-q/weather")

    st.subheader("推荐工作流")
    st.markdown(
        """
        1. 在 `Config Editor` 基于现有 YAML 创建实验配置。  
        2. 在 `Training` 启动训练，并实时查看日志。  
        3. 在 `Evaluation` 选择 checkpoint 运行测试。  
        4. 在 `Baselines` 运行公平基线或传统 ML 基线。  
        5. 在 `Results` 汇总对比指标和图表。
        """
    )

    experiments = db.list_experiments(limit=10)
    st.subheader("最近实验")
    if experiments:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
    else:
        st.info("还没有实验记录。请先进入 Config Editor 或 Training 页面。")


if __name__ == "__main__":
    main()