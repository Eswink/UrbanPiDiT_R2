# legacy_v6 归档快照

V6 物理一致性损失源码快照，仅用于对照/迁移，不属于 R7 import 主路径。

## 内容

- `legacy_physical_consistency.py` — V6 的 `PhysicalConsistencyLoss` 及其四层代理
  （feasibility / structure / process_consistency / process_proxy）。833 行。

## 为什么在这里

该模块在 V6 时期被 Lightning 模块使用；迁移到 R7 后**未被任何活跃模块导入**
（见 `docs/rules/EVIDENCE.md` E-084）。按 Q-001 的决定（见
`docs/rules/OPEN_QUESTIONS.md`），它被归档而非删除，因为实验项目中看起来"死"的代码
常是方法学记录。

## 边界（与其它归档一致）

- **只读**：不得被活跃代码 import，不得写入（规则 R-002、R-031）。
- 不参与打包：`pyproject.toml` 的 `exclude` 含 `legacy_v6*`。
- 不做命名与规模规则约束：`tools/check_conventions.py` 的 `ARCHIVAL_PREFIXES` 含 `legacy_v6/`，
  因此本目录不适用行长、函数长、文件长等阈值。
- 需要恢复使用时，走迁移评审把它正式迁回主路径，而不是直接 import 归档路径。

## 来源

移动自 `training/legacy_physical_consistency.py`（`git log` 保留了完整历史）。
归档时内容的 833 行与原文逐行一致；仅补了一个结尾换行（原文件末尾无换行符）。
