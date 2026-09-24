# Python 实现

范围：`data/**`、`model/**`、`training/**`、`scripts/**`（归档目录除外）。

本项目的实现风格已经相当统一：0 处 logging、0 处裸 except、93 个文件用 pathlib、0 处硬编码宿主路径。
以下规则把这个现状固定下来。

## R-013 文本 IO 必须显式 encoding

- **级别**：必须
- **范围**：`data/**`、`model/**`、`training/**`、`scripts/**`、`tests/**`
- **陈述**：文本模式打开的 `open()` 必须传 `encoding='utf-8'`。
- **依据**：E-071（AST 扫描：活跃区 0 处文本 open 缺 encoding；`contracts.py:37`、`r7_era5_zarr.py` 等均显式传入）、
  E-072（全部活跃源文件可 UTF-8 解码，0 个非 UTF-8 文件）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-013`，AST 判定，二进制模式自动豁免）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入非 UTF-8 外部数据源时（应在读取边界显式声明编码，而不是改这条规则）

## R-014 禁止硬编码宿主绝对路径

- **级别**：禁止
- **范围**：`data/**`、`model/**`、`training/**`、`scripts/**`、`tests/**`
- **陈述**：代码不得内嵌 `/home/...`、`/data/...`、`C:\...` 等宿主绝对路径。
- **依据**：E-073（正则扫描 0 处命中，覆盖 .py 与 .yaml/.json/.md/.toml）、
  E-074（路径均以 `Path(__file__).resolve().parents[1]` 相对定位，如 `scripts/real_r7_seasonal_pilot.py:9`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-014`）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入必须在特定挂载点运行的流程时（应改为参数或环境变量）

## R-015 禁止裸 except

- **级别**：禁止
- **范围**：同上
- **陈述**：禁止 `except:` 无类型捕获；异常必须可分类。
- **依据**：E-075（AST 扫描：活跃区 `ExceptHandler.type is None` 命中 0 处）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-015`）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：无（这条不应被放宽）

## R-016 出网客户端只允许在下载层

- **级别**：必须
- **范围**：`data/**`、`model/**`、`training/**`
- **陈述**：`requests`/`urllib`/`socket`/`s3fs`/`gcsfs`/`fsspec` 等出网客户端的导入只允许出现在
  `data/download/**` 内；其余层必须通过已下载的本地文件工作。
- **依据**：E-076（精确 AST 扫描：活跃区仅 `data/download/` 下的 4 个文件 import 出网模块，
  另有 `urllib.parse` 用于解析 URL）、E-077（10 个离线 workflow 用 monkeypatch 禁用
  `socket.socket.connect`/`create_connection` 来强制不联网）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-016`）；离线强制另见 `ci-and-verification.md` R-028
- **例外**：无（`tests/` 不在本规则范围，但测试另有 R-025 约束）
- **引入日期**：2026-09-24
- **复核触发**：当某个训练流程确实需要在线取数时（那说明它应属于下载层）

## R-017 库与脚本模块启用 postponed annotations

- **级别**：必须
- **范围**：`data/**`、`model/**`、`training/**`、`scripts/**`
- **陈述**：模块必须在文件头部 `from __future__ import annotations`。
- **依据**：E-078（活跃区按目录覆盖：data 33/35、model 22/24、training 36/37、scripts 35/36；
  6 个例外全是空 `__init__.py` 与 `train.py`）
- **现状**：B 类 —— 覆盖 96.7%，遗漏 6 个文件（见例外）
- **执行方式**：脚本（`--rule R-017`）
- **例外**（精确路径）：
  - `data/__init__.py`、`data/download/__init__.py`、`model/__init__.py`、
    `model/layers/__init__.py`、`training/__init__.py`（空包标记文件，无注解需求）
  - `train.py`（6 行的 V6 遗留入口，见 OPEN_QUESTIONS Q-002）
- **引入日期**：2026-09-24
- **复核触发**：当上述 `__init__.py` 开始包含类型注解时

## R-018 路径操作使用 pathlib

- **级别**：默认
- **范围**：同上
- **陈述**：路径拼接与文件系统操作默认使用 `pathlib.Path`；唯一允许的 `os.path` 用法是
  `os.path.relpath`（用于把路径序列化为清单里的相对路径）。
- **依据**：E-079（`Path` 出现在 93 个活跃文件；`os.path.*` 仅 4 处，全部是 `relpath`：
  `r7_era5.py:170`、`r7_era5_zarr.py:34,141`、`real_station_smoke.py:209`）
- **现状**：B 类 —— 4 处例外，均为同一惯用法
- **执行方式**：脚本（`--rule R-018`，AST 判定 `os.path.<attr>`）
- **例外**（精确路径）：
  - `data/preprocess/r7_era5.py:170`
  - `data/preprocess/r7_era5_zarr.py:34`
  - `data/preprocess/r7_era5_zarr.py:141`
  - `data/preprocess/real_station_smoke.py:209`
- **引入日期**：2026-09-24
- **复核触发**：当 `Path.relative_to` 能替代该惯用法时（可把例外清零）
