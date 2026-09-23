"""Streamlit UI 的路径约定。"""

from __future__ import annotations

from pathlib import Path

STREAMLIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STREAMLIT_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = STREAMLIT_ROOT / "experiments.sqlite3"
CONDA_ENV = Path("/3240608030/weather-q/weather")