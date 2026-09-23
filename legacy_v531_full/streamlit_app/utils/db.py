"""SQLite 实验追踪层。"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .paths import DB_PATH

SCHEMA_VERSION = 2
DB_PATH_ENV = "URBANPIDIT_STREAMLIT_DB"


def resolve_db_path(db_path: Path | None = None) -> Path:
    """解析当前数据库路径。"""

    if db_path is not None:
        return Path(db_path)
    override = os.environ.get(DB_PATH_ENV)
    return Path(override) if override else DB_PATH


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """打开 SQLite 连接。"""

    resolved = resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """初始化实验数据库。"""

    with connect(resolve_db_path(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'train',
                config_path TEXT NOT NULL,
                config_hash TEXT NOT NULL DEFAULT '',
                stage TEXT NOT NULL DEFAULT 'single',
                status TEXT NOT NULL DEFAULT 'created',
                pid INTEGER,
                return_code INTEGER,
                command TEXT NOT NULL DEFAULT '',
                start_time TEXT,
                end_time TEXT,
                log_file TEXT,
                log_dir TEXT,
                out_dir TEXT,
                best_ckpt TEXT,
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'experiment',
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                lead_time TEXT,
                variable TEXT,
                split TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(conn, "experiment_metrics", "model_name", "TEXT NOT NULL DEFAULT 'experiment'")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_status
            ON experiments(status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_experiment
            ON experiment_metrics(experiment_id)
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)
            """,
            (str(SCHEMA_VERSION),),
        )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """缺失字段时做轻量 migration。"""

    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {str(row["name"]) for row in rows}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def now_iso() -> str:
    """返回本地 ISO 时间戳。"""

    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def create_experiment(
    *,
    name: str,
    kind: str,
    config_path: str,
    config_hash: str,
    stage: str,
    command: str,
    log_file: str,
    log_dir: str = "",
    out_dir: str = "",
    notes: str = "",
) -> int:
    """写入一条实验记录。"""

    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments(
                name, kind, config_path, config_hash, stage, status,
                command, start_time, log_file, log_dir, out_dir, notes
            ) VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?, ?, ?, ?)
            """,
            (name, kind, config_path, config_hash, stage, command, now_iso(), log_file, log_dir, out_dir, notes),
        )
        return int(cur.lastrowid)


def update_experiment(experiment_id: int, **fields: Any) -> None:
    """更新实验记录。"""

    if not fields:
        return
    allowed = {
        "status",
        "pid",
        "return_code",
        "end_time",
        "log_file",
        "log_dir",
        "out_dir",
        "best_ckpt",
        "notes",
    }
    clean = {key: value for key, value in fields.items() if key in allowed}
    if not clean:
        return
    init_db()
    assignments = ", ".join(f"{key} = ?" for key in clean)
    values = list(clean.values()) + [experiment_id]
    with connect() as conn:
        conn.execute(f"UPDATE experiments SET {assignments} WHERE id = ?", values)


def get_experiment(experiment_id: int) -> dict[str, Any] | None:
    """读取单条实验记录。"""

    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    return dict(row) if row else None


def list_experiments(*, limit: int = 500, kind: str | None = None) -> list[dict[str, Any]]:
    """读取实验列表。"""

    init_db()
    sql = "SELECT * FROM experiments"
    params: list[Any] = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    sql += " ORDER BY COALESCE(start_time, '') DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def delete_experiment(experiment_id: int) -> None:
    """删除实验记录。"""

    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))


def replace_metrics(experiment_id: int, metrics: list[dict[str, Any]]) -> None:
    """替换某个实验的指标记录。"""

    init_db()
    created_at = now_iso()
    rows = [
        (
            experiment_id,
            str(item.get("model_name", item.get("model", "experiment"))),
            str(item.get("metric_name", "")),
            float(item.get("metric_value", 0.0)),
            item.get("lead_time"),
            item.get("variable"),
            item.get("split"),
            created_at,
        )
        for item in metrics
        if item.get("metric_name") is not None and item.get("metric_value") is not None
    ]
    with connect() as conn:
        conn.execute("DELETE FROM experiment_metrics WHERE experiment_id = ?", (experiment_id,))
        conn.executemany(
            """
            INSERT INTO experiment_metrics(
                experiment_id, model_name, metric_name, metric_value, lead_time,
                variable, split, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def list_metrics(experiment_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """读取指标记录。"""

    init_db()
    sql = "SELECT * FROM experiment_metrics"
    params: list[Any] = []
    if experiment_ids:
        placeholders = ",".join("?" for _ in experiment_ids)
        sql += f" WHERE experiment_id IN ({placeholders})"
        params.extend(experiment_ids)
    sql += " ORDER BY experiment_id, model_name, split, lead_time, variable, metric_name"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]