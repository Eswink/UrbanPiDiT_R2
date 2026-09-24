# 数据与产物

范围：`data/**`、`legacy_v531_full/**` 的活跃侧用法、任何写入产物的脚本。

## R-001 新数据输出排他创建

- **级别**：必须
- **范围**：`data/preprocess/**`、`data/download/**`、`training/r7_*.py`、`scripts/real_r7_*.py`
- **陈述**：新数据输出必须以排他模式（`'x'` / `'xb'` / `mode='x'`）创建；禁止用 `'w'` 截断一个数据输出路径。
- **依据**：E-041（41 处排他写入分布在 30 个文件）、E-042（`contracts.py:37` 用 `open('x')` 占位）、E-043（`r7_era5.py:165`、`pressure_pilot_replay.py:96` 用 `'xb'`）
- **现状**：A 类 —— 全仓扫描 0 处截断写入数据路径
- **执行方式**：脚本（`tools/check_conventions.py --rule R-001`）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入数据版本管理（DVC/git-lfs）或允许原地更新缓存时

## R-002 归档快照只读

- **级别**：禁止
- **范围**：`legacy_v531_full/**`、`data/legacy_v531/**`、`model/legacy_v531/**`
- **陈述**：活跃代码不得向归档快照路径写入、不得 `import` 归档模块、不得修改归档内容。
- **依据**：E-044（`model/legacy_v531/README.md` 声明"不属于 V6 import 主路径"）、E-045（全仓 0 处活跃代码 import `legacy_v531`）、E-046（`pyproject.toml:29` exclude `*.legacy*`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-002` 管写入；`--rule R-031` 管 import）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当 V5.3.1 对照工作结束、决定删除归档时

## R-003 派生数据必须可重建

- **级别**：必须
- **范围**：`data/processed/**`、`data/interim/**`、`data/manifests/**`
- **陈述**：任何派生数据必须能由「已固定的源 + 已归档的代码 + 记录的协议」重建；派生数据不得成为唯一真相。
- **依据**：E-047（`R7_DATA_PUBLICATION.md` 声明"旧的无版本 R7 缓存必须在新目标重建，不得靠加标记改装"）、E-048（`data/manifests/real_smoke/provenance.json` 记录 source_sha256 与 split 策略）
- **现状**：A 类（按设计满足：`data/processed/` 与 `data/interim/` 在版本控制中只有 `.gitkeep`）
- **执行方式**：人工自觉（无机械判据；"可重建"需要语义判断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当某个派生数据被外部引用且无法重建时

## R-004 原始与派生数据目录禁止原地写入

- **级别**：禁止
- **范围**：`data/raw/**`、`data/interim/**`、`data/processed/**`
- **陈述**：代码不得写入、重命名或删除这些目录下的内容。新数据一律写到新的目标目录。
- **依据**：E-049（`data/raw`、`data/interim`、`data/processed` 在版本控制中只有 `.gitkeep`）、E-050（`.gitignore:8-13` 显式忽略这三个目录的内容）、E-051（`R7_REAL_DATA_PREFLIGHT.md` 声明"I输入文件永不被修改"）
- **现状**：A 类 —— 全仓扫描 0 处写入这三个目录的调用
- **执行方式**：脚本（`--rule R-004`）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入需要原地更新的数据格式时

## R-005 本地数据集必须经发布契约

- **级别**：必须
- **范围**：`data/preprocess/**`、`data/r7_store.py`
- **陈述**：本地数据集必须经 `fresh_outputs` 预约新路径，并在成功后写 `BUILD_COMPLETE.json`；读取方必须要求该标记。
- **依据**：E-052（`contracts.py:10-50` fresh_outputs 拒绝已存在/符号链接/嵌套路径，成功后写 BUILD_COMPLETE）、E-053（`r7_store.py:12-19,31-35` 读取方拒绝不完整/无版本 store）、E-054（`r7_era5_zarr.py:93,147` `build_complete` 起始 False，末尾才翻转）
- **现状**：A 类 —— 契约存在且读取方强制
- **执行方式**：脚本（`--rule R-005` 检查契约文件与读取方标记）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当并行/分布式构建被引入时（当前明确不是分布式锁）
