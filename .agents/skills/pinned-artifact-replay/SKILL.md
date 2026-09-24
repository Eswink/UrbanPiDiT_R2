---
name: pinned-artifact-replay
description: 当需要用已归档的实验产物重跑或核对历史结果时使用；覆盖源 hash 校验、receipt 交叉核对、离线重放与接受标记。
---

# 重放已固定的产物

## 何时使用

- 你要在**不重新下载数据**的前提下重跑一个历史实验；
- 你要核对某条历史结论是否仍可复现；
- 旧产物需要用**它当时的代码**（`code.zip` / commit）而不是当前 HEAD 来评估；

**不适用**：
- 首次获取真实数据（见 `real-data-acquisition`）；
- 拿旧产物去支撑一个新结论（那需要新的实验，旧产物只能作为对照）。

## 前置条件

1. 知道目标产物的 pin：源文件的 SHA256（在 `data/download/*_replay.py` 的 `PINNED_*_SHA256` 中）
   与 receipt 的期望字段。
2. 拿到匹配的 artifact（例如 `r7-real-pressure-cpu-4cee1f85...`，通过 workflow 的
   `download-artifact` + 固定 `run-id`）。
3. 确认你要用的代码身份：如果产物是旧 commit 产生的，用它的 `code.zip`，不要用当前 HEAD
   （`docs/R7_CPU_REFINEMENT_RESULTS.md:163` 明确记录过这个坑）。

## 步骤

1. **定位 pin。** 读 `data/download/pressure_pilot_replay.py`（或 `seasonal_pilot_replay.py`、
   `continuous_pilot_replay.py`），确认 `PINNED_*_SHA256` 与你手上的源文件对应。
   - 三个模块分别是：pressure（`13fb7280...`）、seasonal（`3b5d7df8...`）、
     continuous（`0609fa38...`）。
2. **校验源。** 调用对应的 `verify_*_pilot(source, receipt)`。它会：
   - 比对源文件 SHA256 与 pin；
   - 交叉核对 receipt 的 source/receipt 声明、时间选择、季节配置、变量顺序；
   - 从 NetCDF 本身重新校验坐标、时间、dtype、有限性与 payload hash。
   - 样例：`pressure_pilot_replay.py:17-83`、`continuous_pilot_replay.py:12-30`。
3. **复制到新目标。** 用 `copy_verified_*_pilot(...)`，它会在复制后**再次**计算 hash
   以检测 TOCTOU（`pressure_pilot_replay.py:91-94`）。
4. **离线重放。** 在禁网进程内跑评估。样例（workflow 内联）：
   ```python
   import runpy, socket, sys
   def deny(*a, **k): raise RuntimeError('network access prohibited during replay')
   socket.socket.connect = deny
   socket.create_connection = deny
   sys.argv = ['scripts/real_r7_pressure_pilot.py', '--out', 'outputs/offline_pressure',
               '--source', 'input_pilot/source/era5_pressure_pilot.nc',
               '--receipt', 'input_pilot/source/receipt.json']
   runpy.run_path(sys.argv[0], run_name='__main__')
   ```
   参照：`.github/workflows/r7-pressure-replay.yml:33-44`。
5. **写接受标记。** 恢复类流程成功时写出 `RESTORATION_ACCEPTED.json`，
   并声明 `source_network_requests=0`（`data/restore_pilot_cache.py:86-95`）。

## 检查点

- **步骤 2 后**：hash 不匹配 ⇒ **立即停止**。不要"大概是对的那份数据"就继续。
  报错信息本身是证据（"source hash does not match the pinned successful real-data artifact"）。
- **步骤 4 后**：确认 `receipt['network_body_bytes'] is None` 或等价字段存在
  （continuous 的校验会强制这一点）。若出现网络字节数，说明禁网没生效。
- **步骤 4 后**：窗口数/样本数必须与原始声明一致。**不一致 ⇒ 停止**，
  不要用新数字解释旧结论（R-009）。
- **步骤 5 后**：确认接受标记存在；没有标记说明本次重放不算成功完成。

## 常见失败

- **用当前 HEAD 评估旧产物**：模型源码 digest 变了就会失败
  （`training/r7_experiment.py:120-130` 会因 code-digest 或 contract 不匹配而拒绝加载）。
  正确做法是用产物自带的 `code.zip`。
- **忘记更新 workflow 的 `run-id`**：改了源数据后 pin 与 artifact 不再对应，
  但下载仍会成功，于是你在旧数据上得出结论。改源数据时同步改 pin 与 run-id。
- **TOCTOU**：校验通过后文件被改写。`copy_verified_*_pilot` 复制后会再 hash 一次，
  不要绕过它自己写复制逻辑。
- **把 replay 当成新实验**：replay 只验证"同一输入产生同一结论"，
  它不产生新的科学证据。

## 完成判据

- 源 SHA256 与 pin 一致；
- 契约（时间/变量/坐标/预算声明）与 receipt 一致；
- 离线重放的窗口数与原始声明一致；
- 接受标记（如 `RESTORATION_ACCEPTED.json`）存在。

## 明确不覆盖

- 不覆盖新数据的获取与核验（见 `real-data-acquisition`）；
- 不替代新实验：replay 成功只说明历史结果可复现，不说明它科学上正确；
- 不覆盖跨配置比较的"共同案例"审计（那是 `training/r7_profile_comparison.py` 的职责，
  参见 `docs/R7_CASE_ALIGNMENT.md`）。
