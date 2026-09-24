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
- 已在预算是"完整训练/多年度数据"级别的实验——那需要单独授权（见主契约的硬约束）。

## 前置条件

1. 已确认数据源身份：本地文件路径 + 其 SHA256（真实数据必须能在 `data/download/*_replay.py`
   的 pin 中被核验）。
2. 已决定预算端点：更新数上限、样本数上限、时间上限。**写下来之后再开始**。
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
3. **执行。** 按预算跑训练与评估，循环内检查墙钟截止时间，超时立即抛错（不要等 CI 超时）。
   - 样例：`r7_continuous_control.py:92-93` 的 1080 秒检查。
   - 约束依据：R-030。
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
- **超时才发现预算错**：把截止检查放在循环**内部**，不是循环之后。

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
