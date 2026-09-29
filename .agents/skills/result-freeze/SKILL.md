---
name: result-freeze
description: 当一个计算结果需要被引用、写进报告或归档时使用；覆盖复现性核查、digest 核对与产物登记。
---

# 结果定稿

## 何时使用

- 某条结果将被写进报告 / 论文 / 对外交付；
- 某条结果将被其他分析引用为输入；
- 你要在 `docs/R7_*.md` 里声称"已验证"。

**不适用**：
- 探索性中间结果（还没确定要保留的）；
- 工程 smoke（它的产物理当标注为工程证据而非科学结果）。

## 前置条件

1. 该结果已有对应的运行记录（`protocol.json` + 结果 JSON），否则先补记录（R-006）；
2. 输入数据身份已知（source SHA256 + receipt）；
3. 代码身份已知（产生该结果的 commit，或它归档的 `code.zip`）；
4. 工作树状态已知：`git status` 无意外改动。

## 机器可读验证回执（CPU study 试点）

`r7-cpu-study` 的 `always()` 收尾步骤会在运行目录写入
`verification_receipt.json`，并调用 `tools/verify_experiment_receipt.py` 做只读核验。回执至少记录：

- `run_id`、`workflow`、完整 `commit_sha`；
- `protocol_sha256`、`source_sha256`、`data_identity`；
- `status`：`success`、`partial`、`failed`、`cancelled`、`queued` 或 `skipped`；
- `required_artifacts` 与逐文件 `output_artifacts`（含相对路径和 SHA256）；
- `scientific_claim: false` 与非空 `limitations`，以及非成功状态的 `failure_reason`。

只有 `status: success` 且所有必需文件、结果声明和 digest 都通过校验，才会被命令行的
`--require-success` 接受。`partial`、`failed`、`cancelled`、`queued`、`skipped` 都是有用的工程记录，
但不能作为通过、科学结论或训练授权。job-level 的标签门控如果使整个 workflow 显示 skipped，步骤不会启动，
因此不会伪造一份回执；该 CI 状态仍按本文件的“取消/排队不算通过”纪律处理。

这是第一条试点 workflow 的最小契约；严格 profile 使用 `urbanpidit-verification-receipt-v1.1`，要求 source bytes、protocol source digest、source receipt 和结果身份交叉一致。它不替代已有的 `protocol.json`、source receipt、代码归档或结果冻结步骤；只验证身份和产物完整性，不判断指标好坏、seed 一致性或科学 gate。

## 步骤

1. **核对记录完整性。** 打开结果目录，确认存在：
   `protocol.json`（含 `protocol_sha256`）、结果 JSON（含 `scientific_claim: false` 与 `limitations`）、
   `code_commit.txt`、`code.zip`（CI 运行的话）。
   - 缺 `protocol.json` ⇒ 这条结果不满足 R-006，不能作为可引用结论。
2. **核对输入身份。** 用 `data/download/*_replay.py` 的 `verify_*_pilot` 对源做校验；
   或确认 receipt 中的 `source_netcdf_sha256` 与你手上的文件一致。
3. **重放一次。** 取记录中的命令，在**临时目标目录**重跑。
   - 期望：与归档产物的 digest 或关键指标一致；
   - 不一致 ⇒ 走「差异归因」，**不得**继续定稿。
4. **核对字段语义。** 确认：
   - 单位没有混用（RMSE 不能跨物理单位平均，见 `r7_cpu_study.py:104-109` 的 docstring）；
   - 声明的 case 数与实际评估数一致（`n_evaluated == len(initializations)`）；
   - 若做了跨配置比较，用的是共同案例（`docs/R7_CASE_ALIGNMENT.md`）。
5. **登记。** 在对应 `docs/R7_*.md` 中记录：issue 编号、实际运行 id、CI 运行 id、
   结论与**限制**。若结果是负面或混合的，照实写（`docs/R7_TASK_QUEUE.md:50` 要求保留全部负面结果）。

## 检查点

- **步骤 1 后**：记录不完整 ⇒ 停止，先补齐。不要把"我记得跑过"当作记录。
- **步骤 3 后**：digest 不一致 ⇒ **不得**继续定稿。按以下顺序排查：
  种子是否显式传入 → 依赖版本是否一致 → 数据版本是否一致 → 是否存在并行非确定性。
  并在记录中写明差异性质。
- **步骤 4 后**：若发现评估只覆盖了部分初始化（例如某变体在中途失败），
  **必须**在记录中写明；不得只报告通过的部分。
- **最终**：每一个数字都应能指到结果记录与产物 digest。指不到的数字不能进报告。

## 常见失败

- **把"指标接近"当成"可复现"**：没有 digest 一致时，最高只能声称
  config-reproducible。本项目当前**未设置** `use_deterministic_algorithms` 与
  `cudnn.deterministic`，也没有固定 TF32 开关，因此**不能**声称 bit-reproducible。
- **用推理步数当速度指标**：`R7_MANUAL_ITERATION.md:38-40` 明确禁止把 reasoning-step
  count 说成 wall-clock speedup。要报速度就分别测模型前向延迟。
- **把合成 fixture 当天气真值**：`docs/REAL_DATA_SMOKE.md` 与
  `data/manifests/real_smoke/provenance.json` 的 `limitations` 已列明该 fixture 的边界
  （单站观测广播到网格，不代表空间超分真值）。
- **事后放宽判据**：判据必须实验前写死（R-006），失败照实记录，不删不改（R-009）。
- **测试数据被用于选择**：先冻结候选、再在 held-out 上评估。
  `docs/R7_EXPERIMENT_HANDOFF.md:51-55` 要求测试数据在选定冻结前保持不动。

## 完成判据

- 结果目录含 `protocol.json`（含 digest）、结果 JSON（含非科学声明与 limitations）、
  代码身份文件；
- 能在临时目标重跑并得到一致（或差异已归因）的结果；
- `docs/` 下有记录：issue、实际运行 id、CI 运行 id、结论、限制、以及负面结果（若有）；
- 记录中明确写出可复现等级（当前最高为 config-reproducible）。

## 明确不覆盖

- 不做结果的科学解释与统计检验（那是分析流程的职责）；
- 不认证科学价值：本流程只保证"这条数字可追溯、可重跑"，不保证模型有预报技巧；
- 不覆盖尚未记录的临时探索；
- 不覆盖发布与合并（`main` 合并需要单独授权）。
