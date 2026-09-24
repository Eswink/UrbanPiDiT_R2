# 仓库与目录契约

范围：顶层布局、版本控制边界、归档边界。

顶层职责（实测，非声明）：

| 路径 | 职责 | 活跃度 |
| --- | --- | --- |
| `data/` | 数据契约、下载、预处理、数据集与 store | 活跃（36 个 .py） |
| `model/` | V6/R7 模型代码 | 活跃（29 个 .py） |
| `training/` | 训练、评估、实验编排、指标 | 活跃（37 个 .py） |
| `tests/` | 单元与集成测试 | 活跃（67 个 .py，280 个测试函数） |
| `scripts/` | CLI 入口与 smoke | 活跃（25 个 .py） |
| `configs/` | YAML 配置 | 活跃（10 个文件） |
| `docs/` | 设计、实验记录、SOP | 活跃（44 个 .md） |
| `audit/` | 自审记录与运行日志 | 归档式追加 |
| `legacy_v531_full/` | V5.3.1 完整快照 | **只读归档**（202 个 .py） |
| `legacy_v6/` | V6 物理一致性损失快照 | **只读归档**（1 个 .py，833 行） |
| `data/legacy_v531/`, `model/legacy_v531/` | V5.3.1 局部快照 | **只读归档** |
| `outputs/` | 运行产物 | 未纳入版本控制（当前不存在） |
| `tools/` | 只读的约定检查脚本与 agent hooks | 活跃（接入 CI 与 hooks） |
| `docs/plans/` | 归档后的计划 | 活跃（提交） |
| `docs/decisions/` | 决策记录（ADR） | 活跃（提交） |
| `docs/goals/` | goal 模式的长文目标源 | 活跃（提交） |
| `.agents/skills/` | 能力（SOP），代理自动加载 | 活跃（提交） |

## R-012 原始数据与产物不得进入版本控制

- **级别**：禁止
- **范围**：`data/raw/**`、`data/interim/**`、`data/processed/**`、`outputs/**`、`logs/**`、`*.ckpt`
- **陈述**：这些路径下的内容不得被提交；只允许保留 `.gitkeep` 占位。
- **依据**：E-049/E-050（三个数据目录仅有 `.gitkeep` 被跟踪，`.gitignore:8-13` 显式忽略内容）、
  E-070（活跃区 205 个被跟踪 .py 文件中无 `.ckpt`/`.npy`/`.npz`/`.png` 产物；`git status` 干净）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-012`，读 `git ls-files`；无 git 时报告 UNKNOWN）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入需要提交的小型参考数据集时（需显式豁免并说明理由）

## R-031 活跃代码不得 import 归档快照

- **级别**：禁止
- **范围**：除归档目录本身外的全部 `.py`
- **陈述**：`legacy_v531_full/`、`data/legacy_v531/`、`model/legacy_v531/` 不得被活跃代码导入。
- **依据**：E-044（`model/legacy_v531/README.md` 明确"不属于 V6 import 主路径"）、
  E-045（全仓 0 处匹配 `^\s*(from|import)\s+\S*legacy_v531`）、E-046（打包时 exclude `*.legacy*`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-031`）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当需要把归档中的某个模块正式迁回主路径时（需走迁移评审，不是直接 import）

### 观察（不立规）

- 项目**没有** `LICENSE`、`CITATION.cff`、`AUTHORS`。这不是违规，是当前状态；论文投稿前需要补齐。
- `.mimosa/`（Mimosa 扫描状态）与 `.zcode/plans/`（会话计划）都是工具运行态，
  已被 `.gitignore` 忽略。归入 R-035 的「工具运行态」范畴，不逐个工具立规。
- 归档区包含 3 个第三方 notebook（GraphCast 上游示例），其 `outputs` 已清空、`execution_count` 为 null。
  归档内的第三方材料不适用本项目的命名与规模规则。
- 命名抽查：活跃区 205 个模块全部符合 `snake_case`（`__init__.py` 除外），无拼音、无空格、
  无 `utils/misc/common/helpers` 杂物桶目录。因此**不为此立规**——现状已经满足，
  立一条无人会违反的规则没有价值。
