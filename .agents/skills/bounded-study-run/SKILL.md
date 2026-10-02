---
name: bounded-study-run
description: 当要运行一个受预算约束的 CPU 实验并为其留下可核查证据时使用；覆盖协议冻结、离线执行、代码身份归档与结果登记。
---

# 运行一个有界实验

## 何时使用

- 你要跑一个受明确的更新数/样本数/时间预算约束的实验（消融、对照、诊断）；
- 结果需要被后续引用或写进记录；
- 或者你要新增一个 `.github/workflows/r7-*-study.yml` 类型的 CI 实验。

**不适用**：
- 探索性随手试探（还不确定要不要保留的）——先跑，确定了再走本流程；
- 完整训练的编排不由本流程覆盖，但本地训练规模本身已按决策 0030（继承 0029） 常设下放；新/多年度数据下载
  与数据发布仍走 `real-data-acquisition` 并逐次取得授权，不能把本地训练自主权扩成取数许可。

## 授权

按决策 0030（继承 0029），本地 GPU 运行、训练/评估、触发实验 workflow，以及每次实验的时长与预算分配已
常设下放，由执行者自主决定，不再逐次询问；范围、预算端点、产物与证据、失败与 skip 处理仍须
在执行前写进长文/协议并记入运行记录。

保留逐次授权项：GPU 租赁与付费资源、多年度/新数据下载与数据发布 `--write`、`main` 合并、
force push、破坏性数据操作；本决策不改变这些原有边界与禁令。授权不豁免 R-006（先冻结协议）、
R-028（离线禁网）、R-054（共驻且不干预邻居）、R-009（不弱化判据）、身份校验、
`scientific_claim: false` 与 limitations。如实保留 `skip`/`cancelled`/`queued`/`partial`/`failed`，
它们不算通过。总体研究方向归用户；执行者自主普通决策和节点推进，按既有前置/出口对表、登记
证据后可在同一 goal 内继续，不逐轮等待用户。独立审阅核证据质量，不是推进许可；最终 goal 完成
仍由用户或运行时独立校验裁定，不可由执行者自宣。

## 前置条件

1. 已确认数据源身份：本地文件路径 + 其 SHA256（真实数据必须能在 `data/download/*_replay.py`
   的 pin 中被核验）。
2. 已决定预算端点：更新数上限、样本数上限、`planned_seconds`（软预算）与 `hard_cap_seconds`
   （宽松硬上限，默认约 2× 计划，整轮墙钟）。两个时间端点在该轮长文写死并纳入冻结协议，
   **写下来之后再开始**。
3. 工作树状态已知：`git status` 无意外改动，`git rev-parse HEAD` 记下。
4. 环境一致：CI 上跑的话确认 workflow 里装的是 `requirements.txt` + `requirements-r7-data.txt`。

## 步骤

1. **冻结协议。** 在训练代码里，把全部实验设定写入 `<out>/protocol.json`（用排他模式 `'x'`），
   并在结果中记录其 digest。协议必须包含：数据源 sha256、变体定义、种子、
   更新数端点、批大小/学习率/steps、评估样本数、lead hours、选择规则、以及
   `scientific_claim: false`。
   - 样例参照：`training/r7_cpu_study.py:184`（注释 "Frozen BEFORE preprocessing/training"）、
     `training/r7_continuous_control.py:80`、`training/r7_spatial_study.py:93`。
   - 约束依据：R-006。
2. **准备数据（离线）。** 从已核验的本地源构建数据集，走发布契约（`fresh_outputs` →
   `BUILD_COMPLETE.json`）。不得联网。
   - 样例：`prepare_local(source, cfg, write=True, store_path=..., manifest_dir=..., max_raw_gib=...)`。
   - 约束依据：R-001、R-004、R-005。
3. **执行。** 跑训练与评估，循环内检查整轮墙钟的**硬上限**；超过软预算不中止，继续等待并记录
   `soft_overrun_seconds = max(0, whole_elapsed_seconds - planned_seconds)`，只允许在硬上限因时长
   截断。硬截断记 `budget_limited`/`failed`、保留证据、全额记账，不算科学通过；真实失败与其它
   冻结停止条件仍可提前停止。
   - 历史样例：`r7_continuous_control.py:92-93` 的 1080 秒检查（不回溯改其冻结语义）。
   - CI 必须满足 `timeout-minutes * 60 > hard_cap_seconds` 并留收尾余量，不等 GitHub 超时才失败。
   - 约束依据：R-030、决策 0030（继承 0029）；新契约不回溯修改冻结常量或历史结果。
4. **归档代码身份。** 在 workflow 里把 `code.zip`（`git archive HEAD`）与
   `code_commit.txt`（`git rev-parse HEAD`）写入产物目录，并把产物 upload 为 artifact。
   - 样例：`r7-cpu-study.yml` 的 "Preserve exact source code" 步骤。
   - 约束依据：R-010。
5. **登记结果。** 结果 JSON 必须带 `scientific_claim: false` 与 `limitations` 列表；
   写一份 `docs/R7_*.md` 记录 issue、实际运行 id、CI 运行 id、以及**混合/负面结果**。

## 检查点

- **步骤 1 后**：确认 `protocol.json` 存在且**早于**任何训练调用。
  若你的代码里训练调用出现在协议写入之前，停下来先调整顺序——这是 R-006 的实质要求，
  不是形式要求。
- **步骤 2 后**：确认 `BUILD_COMPLETE.json` 存在；确认窗口数与预期一致
  （例如 `if prepared['windows'] != dict(train=120,val=120,test=120): raise`）。
  不一致 ⇒ 停止，不要放宽预期值（R-009）。
- **步骤 3 后**：确认评估覆盖你声明的全部 case。若某变体失败，
  **保留失败记录**并把该变体标记为失败；不得静默丢弃它再报告"其余变体通过"。
- **步骤 4 后**：确认 artifact 内确实有 `code_commit.txt` 与 `code.zip`。缺任一项，
  这次运行事后无法被审计。
- **步骤 5 后**：结果里若出现负面或混合结论，**照实记录**。本项目的
  `docs/R7_TASK_QUEUE.md:50` 明确要求保留全部负面结果。

## 常见失败

- **协议写入用了 `'w'`**：会覆盖已存在的协议，导致"事后调整判据"无法被发现。用 `'x'`。
- **把评估集当验证集调参**：`R7_EXPERIMENT_HANDOFF.md:51-55` 要求先冻结候选、
  再用同一批 held-out case 评估；测试数据在选定前不得触碰。
- **禁网没生效**：离线实验必须在进程内 monkeypatch `socket.socket.connect` 与
  `socket.create_connection`（R-028）。它是"防意外"，不是沙箱。
- **artifact 下载用错 run-id**：workflow 里 `download-artifact` 的 `run-id` 是硬编码的。
  改动源数据后必须同步更新 pin，否则会在旧产物上跑出新结论。
- **软预算写成硬上限**：计划没写软/硬两个数字，事后无法区分 overrun 与截断；循环内部只按
  已冻结的硬上限因时长截断，不能把软预算超出当成失败。

## 完成判据

- 产物目录含 `protocol.json`（含 digest）、结果 JSON（含 `scientific_claim: false` 与 `limitations`）、
  `code_commit.txt`、`code.zip`；
- 评估覆盖声明的全部 case，或失败原因被显式记录；
- `docs/` 下有一条记录：issue 编号、实际运行 id、CI 运行 id、结论（含负面结论）。

## 明确不覆盖

- 不覆盖科学结论的判定（本流程只保证工程可核查，不证明模型有预报技巧）；
- 不覆盖 GPU 资源的申请与租赁（那需要单独授权）；
- 不覆盖控制器标定与策略选择的调参（那是另一个流程，且当前尚未形成稳定协议）；
- 不覆盖真实数据的首次获取（见 `real-data-acquisition`）。
