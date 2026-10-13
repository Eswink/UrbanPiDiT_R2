# S3 H-A（P_0 逐样本输入条件化）单因素臂：**运行尝试失败**（`load_checkpoint` 以模型身份变更拒绝迁移父）

- **日期**：2026-10-14
- **工作分支**：`r7/weather-reasoning`
- **状态**：`outcome_class: unresolved`、`candidate_state: blocked`、`scientific_claim: false`、`test_read: false`
- **一句话**：本轮把 `model/*.py` 改了（新增 `process_init_proj` + 一处加性注入点），因此
  `model_code_digest()` 变化，注册时迁移好的父 checkpoint 被 `training/r7_experiment.py` 的身份门**按设计**
  拒绝；`stage-seeds` 在**训练开始之前**失败，**0 GPU-h、无任何读数**。修复路径是**已提交的**身份翻译工具
  （不是绕过身份门），已执行完毕。

## 1. 实际发生了什么

- 编排 workflow `dwfrun-13d26a6b`，草稿 `.zcode/workflow-drafts/r7-s3-process-init-run.dwf.ts`。
- **独立复核 clean**：与剂量驱动逐行 diff，差异全部为**预注册**的单因素改动（唯一因子
  `process_init_conditioning` False→True；identity-gated 扩张；off 路径逐位一致）。复核者实跑：
  `all_shared_bit_identical: True`、新增参数 37440、`Linear` 清零后输出恒零、pin 表 15/15 匹配、
  在**干净 git worktree（无 `outputs/`）**上 `tests/test_r7_s3_process_init.py` **39 passed**。
- `code.zip` 按 HEAD `7539bb32ba385bc05b9007e0667c15c6ca8b0ca7` 重建
  （SHA256 `00f447d6816eeb8257e01c140ae6b3521724df0ec6200a9af368ace7c94f6920`）。
- `freeze` / `prepare` / `archive` **全 success**（`outputs/r7_s3_process_init_20261014_attempt01/`）。
- **`stage-seeds` 失败（exit 1）**：seed 41 与 seed 42 **都在父加载处**抛
  `ValueError: checkpoint model implementation differs; use the recorded code revision for exact
  evaluation/resume`（`training/r7_experiment.py:129`，经
  `scripts/study_r7_s3_process_init.py:373 _parent_model`）。**seed 43 从未启动**
  （`device_mode: "dual"`，一卡一 seed，两块卡各跑一个 seed、两个都失败）。
- 失败发生在**训练之前**：无 optimizer step、无 checkpoint、无评分、**无 `readings.json`**。

## 2. 根因（已确认，非推测）

`training/r7_experiment.py:126-131`：

```python
def load_checkpoint(path,*,expected=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='r7-local-v1':
        raise ValueError('unsupported R7 checkpoint format')
    if checkpoint.get('model_code_sha256')!=model_code_digest():
        raise ValueError('checkpoint model implementation differs; use the recorded code revision for exact evaluation/resume')
```

- 本臂**确实修改了 `model/process_forecast_r7.py`**（这是本轮相对前两轮的实质差异：depth/width 轮只改
  **spec**，本轮改**前向**）。因此 `model_code_digest()`（对全部 `model/*.py` 取哈希）由
  `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab` 变为
  `223b2e5b82093bb5c1b75ffa25ff63983f44e6cdf6a96da772b17e88a748750e`（本协议顶层
  `model_code_sha256` 即后者）。
- 本轮父是**已迁移** checkpoint：`outputs/r7_s3_rollout_dose_parent_migrated_20261009_attempt02/seed{41,42,43}/parent_update_0001600.pt`，
  其 `migration_receipt.json` 记的 `current_model_code_sha256` 是**旧** digest `d3fb58db…`。
- 于是身份门拒绝——**这不是缺陷，是门在正确工作**。父文件本身完好（`sha256_file` pin 校验在拒绝之前已通过）。

## 3. 这条失败能支持什么、不能支持什么

- **能支持**：本轮 H-A 假设**未被检验**；0 科学读数；§3 预声明出口（72h 三 seed 全正）**既未触发也未满足**。
- **不能支持**：**不能**说 H-A 无效、**不能**据此关闭 H-A 支、**不能**写任何通过/失败。
- **结构性发现（已确认，本页主要产出）**：**任何 `model/*.py` 改动都会改变 `model_code_digest()`，
  从而让所有"迁移父"在「微调自注册父」协议下不可加载**。spec-only 改动（depth、dim）不动 digest，
  所以容量两轮能跑；H-A 需要新代码，所以被身份门挡住。这条约束对后续任何架构方向都成立，
  是架构轴的真实成本项。

## 4. 合法修复（已执行，**未**绕过身份门）

- 使用**仓库内已提交的** `scripts/migrate_r7_parent_state.py`。其 docstring 明确写着它的用途：
  "performs the documented model-only migration **instead of weakening that guard**"。它做的检查包括：
  源 checkpoint SHA256 pin、归档 revision 自己的 loader 导出、tensor keys/shapes/dtypes/finite、
  当前模块树语义、`strict=True` 的 `load_state_dict`，并在写盘后**读回自检**；**不**携带
  optimizer/cursor/RNG。
- 命令（在当前代码下重跑）：
  `.venv/bin/python scripts/migrate_r7_parent_state.py --out outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt01`
- 结果 `{"status":"success"}`，新父 SHA256：
  seed41 `3bb09c1968377ece37b5cf8d48eef34e86f81d527c7e76b2f8bf8c67da1dc05d`、
  seed42 `91667a680d10dc67ca9fa2ddaa33f445e988fb8cd3515b4cc15f0fe21e09b05d`、
  seed43 `b1a716bc7e36b83420e410b3860d72d637874418b538de577a5359bf20ca647a`。
- `semantics_equal: false`（三 seed）是**预期**的：当前模块树多了一个 gated 模块，模块树语义摘要必然变化；
  "off 路径逐位一致"由 H-A 自身的正控断言（`_positive_control`）保证，不由该字段保证。
- **未**修改任何比对逻辑、**未**放宽任何阈值或门、**未**改动 `data/raw|interim|processed`、
  **未**改动归档快照、**未**对任何非本实验进程发送信号。

## 5. 成本

- **0 GPU-h**（失败发生在父加载、训练之前；两块 GPU 全程只有邻居作业）。
- 磁盘：attempt 目录 15 MB；网络 0；soft/hard overrun 均 0（本轮 GPU 未启动）。

## 6. 身份与可核查证据

| 项 | 值 |
| --- | --- |
| code commit | `7539bb32ba385bc05b9007e0667c15c6ca8b0ca7` |
| code.zip SHA256 | `00f447d6816eeb8257e01c140ae6b3521724df0ec6200a9af368ace7c94f6920` |
| `protocol.json` digest | `3ac318c1d5f82186c687c7109965f589dd957a7a6b27cbfc7f2123d25a35c310` |
| 本轮 `model_code_sha256` | `223b2e5b82093bb5c1b75ffa25ff63983f44e6cdf6a96da772b17e88a748750e` |
| 父 receipt digest（旧） | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab` |
| source SHA256 | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| 失败点 | `training/r7_experiment.py:129`（经 `scripts/study_r7_s3_process_init.py:373`） |
| 修复工具 | `scripts/migrate_r7_parent_state.py`（未修改，`git status` 干净） |

## 7. limitations

- **无读数**：无 `readings.json`、无 seed receipt、无 `attempt.json`；无 seed 均值、无同时区间、
  无 primary/gate 判定；§3 预声明出口**既未触发也未满足**。
- **不是科学判定**：本页不改变 r、不读 test、不写通过/失败、不关闭 H-A 支。
- **迁移是身份翻译**：其科学价值为零（工具 docstring 亦如此声明），只是让当前 harness 允许加载。
- **训练稳定性无证据**：本轮 GPU 未使用，无法对 H-A 的显存/收敛提供任何实测。
- 本轮结论只对**本代码 revision** 成立：下一次 `model/*.py` 改动会再次使本轮迁移父失效，
  需再次迁移（这是结构性的，不是偶发）。

## 8. 下一动作（S3）

- 修复 H-A 驱动对**当前 digest** 迁移父的解析（切到
  `outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt01` 与上表新 SHA pin），
  重新 `freeze`（新 `protocol_sha256`）后跑 GPU。
- 第一条命令：`.venv/bin/python tools/r7_round.py round-guard`。
- 预声明出口不变：**72h 三 seed 全正 → 提请独立验收；否则关闭 H-A 这一支**。
