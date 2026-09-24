# 规则体系索引

本目录是 UrbanPiDiT-R² 的分类细则。常驻契约（硬约束 + 路由表）在根 `AGENTS.md`，
不在这里重复。

- 每条规则含：级别 / 范围 / 陈述 / 依据（E-xxx）/ 现状（A/B/C/D）/ 执行方式 / 例外 / 引入日期 / 复核触发。
- 「依据」指向 `EVIDENCE.md` 的台账编号；无法追溯证据的规则不应存在。
- 现状分类决定能否做门禁：**A/B 类才可阻断，C 类只能报告**（原因见 `MIGRATION.md`）。
- 规则变更记入 `CHANGELOG.md`；需要人拍板的进 `OPEN_QUESTIONS.md`。

## 分类文件

| 文件 | 覆盖内容 | 规则 |
| --- | --- | --- |
| [`data-and-artifacts.md`](data-and-artifacts.md) | 原始数据不可变、派生数据发布契约、产物归档 | R-001 – R-005 |
| [`reproducibility.md`](reproducibility.md) | 协议冻结、非科学声明、代码身份、种子 | R-006 – R-011 |
| [`repository-layout.md`](repository-layout.md) | 目录契约、什么进版本控制、归档边界 | R-012, R-031 |
| [`python-implementation.md`](python-implementation.md) | 编码、路径、异常、依赖方向、出网位置 | R-013 – R-018 |
| [`size-thresholds.md`](size-thresholds.md) | 行长、函数长、文件长、嵌套、参数数 | R-019, R-019b, R-020 – R-023 |
| [`testing.md`](testing.md) | 必测项、测试隔离、离线、反证 | R-024 – R-026 |
| [`ci-and-verification.md`](ci-and-verification.md) | workflow 契约、门禁接入、失败纪律 | R-027 – R-030 |
| [`artifact-storage.md`](artifact-storage.md) | 计划 / 决策 / 目标 / 工具态的存放与提交边界 | R-032 – R-037 |
| [`naming.md`](naming.md) | 文件/模块/类/函数/常量/测试/文档/配置命名 | R-038 – R-048 |

配套：[`environment.md`](environment.md) 记录虚拟环境与三份 requirements 未覆盖的依赖缺口（非规则）。

归档目录清单（只读，不适用规模与命名规则）：
`legacy_v531_full/`、`data/legacy_v531/`、`model/legacy_v531/`、`legacy_v6/`。

## 现状分类分布

下表由各规则文件里声明的「现状」栏统计得出（不是手工维护的清单）。

| 分类 | 含义 | 数量 |
| --- | --- | --- |
| A 全体遵守 | 0 处违反，可直接做门禁 | 31 |
| B 少数例外 | 有精确例外清单，可做门禁 | 10 |
| C 目标态 | 只报告不阻断 | 8 |
| D 未涉及 | 项目无此场景，不立规 | 0 |

合计 **49** 条规则条目（含 R-019 拆出的 R-019b 子检查）。

- **A 类**（31）：R-001, R-002, R-003, R-004, R-005, R-008, R-009, R-011, R-012, R-013,
  R-014, R-015, R-016, R-024, R-025, R-026, R-028, R-029, R-031, R-035, R-036,
  R-038 – R-047
- **B 类**（10）：R-006（3 例外）、R-007、R-010（3 例外）、R-017（6 例外）、R-018（4 例外）、
  R-027（1 历史例外）、R-032、R-033、R-034、R-037
- **C 类**（8）：R-019、R-019b、R-020、R-021、R-022、R-023、R-030、R-048

## 机械化状态

`tools/check_conventions.py` 实现 49 条中的 **45** 条：

- **34 条阻断**（A/B 类中可机械化的部分，当前在仓库上 0 违规）：失败即代表违反契约；
- **11 条报告**（C 类与趋势指标）：只在 `--report` 下运行，永不阻断。

未机械化的 4 条，执行方式如实写作「人工自觉」：

| 规则 | 未机械化原因 |
| --- | --- |
| R-003 | 派生数据可重建——语义判断 |
| R-011 | 种子显式记录——部分由测试断言覆盖 |
| R-026 | 每个判据必须有反证——需评审判断 |
| R-034 | goal 长文的 objective 写法——人工流程；可判定部分由 `docs/goals/README.md` 的固定结构承载 |

**已接入 CI**：`.github/workflows/ci.yml` 的 `Check repository conventions` 步骤。

```bash
python tools/check_conventions.py            # 阻断规则（CI 跑的就是这条）
python tools/check_conventions.py --report   # 含 C 类报告与趋势
pytest tests/test_check_conventions.py -q    # 检查脚本自测（含反证 + 结构性防回归）
```

**已接入 CI**：`.github/workflows/ci.yml` 的 `Check repository conventions` 步骤。

```bash
python tools/check_conventions.py            # 阻断规则（CI 跑的就是这条）
python tools/check_conventions.py --report   # 含 C 类报告与趋势
pytest tests/test_check_conventions.py -q    # 检查脚本自测（含反证 + 结构性防回归）
```
