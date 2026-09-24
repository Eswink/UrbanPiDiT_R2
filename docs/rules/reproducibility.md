# 可复现性

范围：`training/**`、`scripts/real_r7_*.py`、`.github/workflows/**`、任何产生结论的执行。

本项目的核心纪律是「先冻结判据、再跑实验、失败照实记录」。这几条规则就是它的机械化表达。

## R-006 批量实验先冻结协议再训练

- **级别**：必须
- **范围**：`training/r7_*study.py`、`training/r7_*control.py`、`scripts/real_r7_*.py`
- **陈述**：有界实验必须在任何训练或评估调用之前，把协议写入 `protocol.json`（或
  `experiment_protocol.json`），并在结果里记录其 digest。
- **依据**：E-055（6 个 study/control 模块中协议写入行均在首个训练调用之前，含
  `r7_cpu_study.py:184` 注释 "Frozen BEFORE preprocessing/training"）、E-056（协议 digest 被
  `test_r7_continuous_control.py:94`、`test_r7_cpu_study.py:20`、`test_r7_spatial_study.py:17` 断言）
- **现状**：B 类 —— 6 个模块合规，3 个例外（见下）
- **执行方式**：脚本（`--rule R-006`，按顶层函数比较写入行与首个训练调用行）
- **例外**（精确路径，共 3 处，均为协议约定确立之前的早期 pilot）：
  - `scripts/real_r7_bounded_smoke.py`（#43 之前的单变量工程 smoke）
  - `scripts/real_r7_pressure_pilot.py`（#44 早期真实数据 pilot）
  - `scripts/real_r7_surface_pilot.py`（#43 surface pilot）
- **引入日期**：2026-09-24
- **复核触发**：当这三个早期 pilot 被重跑或替换时，应改为合规或移入归档

## R-007 实验产物必须带显式非科学声明

- **级别**：必须
- **范围**：同 R-006
- **陈述**：有界实验的结果必须带 `scientific_claim: false` 或等价显式标志
  （`scientific_forecast_claim`/`scientific_training_ready`/`not_claimed`），并列出 `limitations`。
- **依据**：E-057（22 个结果写入模块中 18 个带 `scientific_claim`，另 4 个用等价字段）、E-058
  （`test_r7_inference_profile.py:37`、`test_r7_policy_selection.py:27`、`test_r7_pinned_real_fixture.py:25`
  直接断言该标志为 False）
- **现状**：B 类 —— 全部命中；等价字段使判定需要放宽到多词表（已在脚本中实现）
- **执行方式**：脚本（`--rule R-007`）
- **例外**：无（4 个使用等价字段的模块不算例外，它们已满足陈述）
- **引入日期**：2026-09-24
- **复核触发**：当某个结果确实要申请科学结论时（需要单独的证据链，不是改标志）

## R-008 下载失败必须留审计，禁止合成回退

- **级别**：禁止
- **范围**：`data/download/**`
- **陈述**：真实数据获取失败时，必须写出机器可读的失败记录（`status='failed-no-fallback'`,
  `synthetic_fallback=False`），不得静默改用合成数据继续。
- **依据**：E-059（`arco_tiny_bounded.py:152-159` 失败写 receipts.json 后重新抛出）、E-060
  （`DELIVERY.md` 声明"下载失败机器可读审计，禁止 synthetic silent fallback"）、E-061
  （`test_r7_bounded_download.py:54-62` 断言无回退记录存在）
- **现状**：A 类 —— 已实现该审计的下载路径均合规
- **执行方式**：脚本（`--rule R-008`，检查含 `try` 的下载模块是否带审计字段）+ 测试断言
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当新增下载器时（新模块必须自带该字段）

## R-009 不得为了让门禁通过而弱化判据

- **级别**：禁止
- **范围**：全仓
- **陈述**：禁止删除测试、跳过测试、降低断言强度、放宽已冻结阈值，或未实跑就记录 PASS。
- **依据**：E-062（`R7_MANUAL_ITERATION.md:17` "A cancelled or queued run is not a pass"）、
  E-063（`R7_MANUAL_ITERATION.md:15` "fix the actual failure without weakening scientific/tests requirements"）、
  E-064（全仓 262 个测试函数中仅 2 处 skipif，且理由注明"clean checkout 无 fixture，禁止合成回退"）
- **现状**：A 类 —— 无被注释掉的断言、无长期 xfail、无降级记录；测试强度计数有基线监控
- **执行方式**：**脚本（`--rule R-009`，报告型）**对照记录基线（280 个测试函数 / 560 个断言），
  低于基线即报告；语义层面的削弱（降低断言强度、事后放宽阈值）仍为**人工评审**
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当**有意**增删测试时（应同步更新 `tools/check_conventions.py` 的
  `TEST_FUNCTION_BASELINE`/`ASSERT_BASELINE` 并在 CHANGELOG 说明原因）

## R-010 跑实验的 workflow 必须归档代码身份

- **级别**：必须
- **范围**：`.github/workflows/*.yml`
- **陈述**：任何执行有界实验的 workflow 必须在产物中归档 `code_commit.txt`（`git rev-parse HEAD`）
  与 `code.zip`（`git archive`）。
- **依据**：E-065（18 个 workflow 中 11 个同时归档 commit 与 zip）、E-066（`R7_CPU_REFINEMENT_RESULTS.md:163`
  记录"用归档的 code.zip 评估旧产物"的必要性）、E-067（`R7_CASE_ALIGNMENT.md` 要求跨配置比较需显式案例审计）
- **现状**：B 类 —— 11 个合规，3 个例外（见下）
- **执行方式**：脚本（`--rule R-010`）
- **例外**（精确路径）：
  - `.github/workflows/r7-pressure-pilot.yml`（数据获取 pilot，非结论性实验）
  - `.github/workflows/r7-surface-pilot.yml`（同上）
  - `.github/workflows/r7-pressure-replay.yml`（复用已归档产物，不产生新实验结论）
- **引入日期**：2026-09-24
- **复核触发**：当上述三个 workflow 产生需要长期引用的结论时

## R-011 随机性与精度必须显式且可记录

- **级别**：必须
- **范围**：`training/**`、`model/**`、`data/**`
- **陈述**：随机种子必须显式传入并写入运行记录；不得依赖全局隐式状态。
- **依据**：E-068（`r7_experiment.py:56-61` seed_everything 覆盖 random/numpy/torch/cuda；
  `:64-78` 保存与恢复 RNG 状态，CUDA 拓扑不匹配时抛错）、E-069（`r7_local_runner.py:74` 把 seed 写进
  checkpoint contract；`:141` 存完整 RNG 状态；`test_r7_local_runner.py:36-56` 断言 resume 等价）
- **现状**：A 类 —— 训练路径全部显式传种子；评估路径已补 `deterministic: true` 与
  `determinism_scope` 说明字段（Q-004，见 `training/r7_evaluate.py`）
- **执行方式**：人工自觉 + 测试断言（`test_r7_local_runner.py` 覆盖 resume 等价性）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入多进程/分布式数据加载时

### 已知缺口（不立规，记录在此）

- 未设置 `torch.use_deterministic_algorithms`、`cudnn.deterministic`，也未固定 TF32 开关；
  `r7_inference_profile.py:136` 只**报告**这些状态而不设置。因此当前只能声称
  config-reproducible，**不能**声称 bit-reproducible。
- `r7_evaluate.py` 与 `r7_inference_profile.py` 的记录中没有 seed 字段（推理按构造确定性，
  靠 digest 校验）。见 OPEN_QUESTIONS Q-004。
