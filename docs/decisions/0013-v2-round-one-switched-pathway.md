# 0013 V2 第一轮：时空输入与位置化 process 读写，双开关默认关 + 逐位等价证明

- **日期**：2026-09-28
- **状态**：accepted
- **代码 SHA 范围**：`93d89aa`（起点）→ 本轮提交
- **依据证据**：`docs/R7_71_72_M1_AND_RWA.md`；`tests/test_r7_switched_path_equivalence.py`（23 个 digest 的逐位等价 + 反证）；
  `tests/test_r7_input_path_consistency.py`；`tests/test_r7_spacetime_inputs.py`；
  `tests/test_r7_process_readout_positional.py`；`outputs/r7_71_72_m1_rwa/`

## Context

#71 要求把**已知时空条件**（init 时刻、年内/日内相位、地理坐标）真正接进主模型，
#72 的 RW-A 要求把 `process.mean(dim=1)` 后的 **P.mean 广播**换成**位置化读写**。
本轮开工时仓库的三条既成事实决定了怎么做：

1. 任何 `model/*.py` 的字节改动都会改变 `model_code_sha256`，旧 checkpoint 在新代码下
   **拒绝加载**（`training/r7_experiment.py:124-134`）。所以"让新通路可用"不能靠兼容旧产物，
   旧分数与新分数也不能混用——**要么新通路显式开，要么旧路径不动**。
2. 已发布的 B2 / M2 数字全部建立在**当前默认路径**上。一个重构如果悄悄改了默认行为，
   那些数字的来源就不再是可见的代码，而"看起来没变"不是证据。
3. 数据侧已经给出 `latitude`/`longitude`/`grid_spacing_deg`，但**没有 init 时刻**；
   而 `latitude` 在 `forecast_inputs` 白名单里被剥掉，从来没有任何模型读过它。

## Decision

**采用：两个新机制各自独立、显式、默认关闭；关闭时必须与改动前的实现逐位相同；
开启时新增的参数量与算力如实报告。** 具体：

1. **开关**：`NativeAtmosForecaster.spacetime_inputs` 与
   `ProcessForecastCoReasoner.positional_process_readout`，默认均为 `False`；
   非布尔值即报错。
2. **逐位等价是测试，不是声明**：`tests/test_r7_switched_path_equivalence.py` 用
   `git archive <起点 SHA> model training` 把**改动前的实现**读进同一进程（包名改为
   `pre_change_model`/`pre_change_training`，唯一改动是 import 前缀，且该改写可逆校验），
   再对 forward / rollout / 自适应 / 流式训练反向传播共 **23 个 digest** 做逐位比较；
   另有反证：故意扰动冻结副本的一个常量后比较必须失败。
3. **字段集合只有一处声明**：`DECLARED_MODEL_INPUTS = coarse_history, lead_time_hours,
   latitude, longitude, init_utc_hour, init_day_of_year`。白名单、固定/截断/验证/
   评估/自适应/rollout/profiler 七条路径由测试钉住"递给模型的键集合完全相同"，
   且目标类字段（`atmos_target`/`rollout_targets`/`process_targets`）永不出现。
4. **相位只由 init + lead 计算**，从 store 自带的 `time_ns` 整数推导
   （`data/r7_store.init_time_fields`，非整点即拒绝）；模块不 import 任何时钟源（AST 反证）。
   `init_year` 仍随样本携带以供审计，但**不在**读取集合内——年相位是周期量，
   该事实由"改 init_year 输出逐位不变"的探针钉住。
5. **RW-A 仍是加法式**：每个输出位置用自己的 query（自身 context token + 固定的二维
   正弦位置编码）对 M 个 process token 做 cross-attention，得到 `[B,N,D]` 摘要，
   由 `solver_conditioning` **直接相加**；无门控、无逐位置状态、process 递推本身不变。
6. **新增模块在构造时不消耗随机流**（`model/spacetime_conditioning_r7.isolated_stream`），
   因此同种子下两条臂的**全部既有参数逐位相同**；运行里实测 `shared_tensors_identical`
   并写进结果，而不是假定。
7. 本轮对照实验**同时打开两个开关**，报告实测参数增量（+168,480，+6.0%）与
   前向 FLOPs 增量（+8.3%）；本设计**不能**分离两个机制各自的贡献。

## Consequences

**变容易的：**

- 新机制的引入有了可复制模板：默认关 + 冻结实现逐位等价 + 字段集合单点声明 + 臂配对实测；
  以后任何"加一条通路"的改动都能按同一套证据要求提交。
- 旧数字仍可复现：旧 checkpoint 在旧修订下仍能加载，新代码不会静默接受它们；
  "默认路径没变"由测试而非记忆保证。
- 七条路径的输入集合由测试互锁，`latitude` 这类"样本里有、模型收不到"的历史缺陷类型
  会被立刻发现（`missed` 集合被钉住为 `{atmos_baseline}`）。

**变难 / 代价（如实列出）：**

- 开启时模型**更贵**：+168,480 参数（+6.0%）、前向 FLOPs +8.3%（同批次同深度实测），
  因此两臂**不是**参数/算力对齐的对照；单靠本轮的胜负**不能**归因到机制而非容量。
- `spacetime_inputs=True` 使四个字段变成**必需**：任何不能提供它们的路径会**报错而不是降级**。
  这是刻意的（禁止用服务器时钟/占位值兜底），代价是新调用方必须同步携带字段，
  否则在运行期失败——白名单与 rollout 的测试把这件事显式化。
- RW-A 的读取仍然是**集合语义**：query 不含 key，permute process token 的输出差在
  浮点求和噪声内（测试钉住），所以"positional"**不能**读成"ordered"；
  位置基是固定正弦而非可学的每位置表（换来的是任意 token 网格都能定义）。
- 两个开关本轮一起开，**两个机制的分别贡献 unresolved**；两个 seed 的一致性不是显著性。
- 等价性测试依赖仓库历史（`git archive <SHA>`）：CI 已经 `fetch-depth: 0`，
  但没有历史的克隆上它会**失败而不是跳过**（刻意的：未测量的等价不算通过）。
- 新增的 M1 字段使训练样本字典变大（每样本 +3 个标量），并让 `data/synthetic_atmos.py`
  这个纯 fixture 也需要合成一个 init 时刻——它只作为**输入**，不构成任何天气真值。

**备选方案与否决理由：**

- *常开 + 兼容旧 checkpoint 的 digest 分支*：否决。这正是硬约束禁止的"静默兼容"，
  会使旧分数与新代码混用且无法区分。
- *可学的每位置 query 表（Perceiver-IO 的 learned latents）*：否决。它必须声明一个
  最大 token 网格（模型其它部分没有这个分辨率上限），并为 N 个位置各存一个 D 维向量，
  在没有证据表明需要这种自由度时徒增参数。
- *把 lat/lon 压成域均标量再 `Linear(2,dim)`*：否决。这会退化成"另一个 lead-time embedding"，
  正是 #71 要消除的缺陷（此点在目标源 §6 已作为 planner 草案错误更正）。
- *把新增读数零初始化（LoRA/ControlNet 式，使开启臂初始等价于基线）*：否决。
  那会让"未训练的新通路已经对相位与 process 扰动有响应"这一判据（D5）无法成立。
- *把字段塞进 `atmos_baseline` 之类的既有键*：否决。白名单语义会变得含糊，
  且 `atmos_baseline` 属于被刻意剥离的 baseline 家族。
