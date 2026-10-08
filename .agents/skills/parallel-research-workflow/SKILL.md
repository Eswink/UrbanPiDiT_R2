---
name: parallel-research-workflow
description: 当要在一条长期 campaign 里同时推进多个互不依赖的研究线、把资料/CPU 准备与真实实验重叠，或用 workflow 脚本编排多个子代理时使用；覆盖并行准入判断、编排形态选择、隔离与安全边界、科学完整性、证据登记与降级路径。
---

# 并行科研与 workflow 编排

## 何时使用

- 一条 campaign 里有 **≥2 个互不依赖、可单独证伪**的假设要同时推进；
- **资料检索、CPU 准备、只读证据核对**可以与真实实验重叠进行；
- 需要把多个子代理的产出**汇总成一个决策**（候选设计、审阅矩阵、文献对照）。

**不适用**：

- 单步修改，或顺序依赖的假设链（后一步要用前一步结论）；
- 共享同一可变状态的实验（同一 store / checkpoint / cursor / optimizer 写）；
- 科学判据的制定 —— 判据只**引用**已冻结文档，不由并行讨论产生。

## 前置条件

0. **本技能属前瞻立 SOP**：`docs/skills/README.md:9` 要求能力「已重复 ≥2 次且步骤固定」，而本能力
   目前只有「并行已被计划 ≥3 次」的先例、**尚无 ≥2 次实际并行执行**。依据是用户明确要求支持
   workflow 与并行科研（决策 0042）。**首次真实并行轮必须留痕验证**并把结论写进 goal §8 / 计划；
   若实践显示步骤不固定或并行不可行，应修订或撤下本技能，不得长期停留在未验证状态。
1. 读过 `docs/rules/gpu-resources.md`（R-054 共驻）、`docs/rules/reproducibility.md`
   （R-006 协议先冻结）、`docs/rules/ci-and-verification.md`（R-028 离线禁网）。
2. 本轮能写出 `planned_seconds` 与 `hard_cap_seconds`（决策 0030：软预算 + 宽松硬上限）。
3. 每个并行臂都有自己的假设、停止出口和证据落点。

## 步骤

### 1. 并行准入判断（先想能不能并行，再并行）

逐条回答下表。**任一为「否」⇒ 该臂串行**，并把结论与理由写进 goal §8 或计划正文。

| 检查 | 通过条件 |
| --- | --- |
| 假设独立 | 每臂可单独证伪，结论互不依赖 |
| 产物隔离 | 各臂写**各自排他** `outputs/` 路径，不共享任何文件 |
| 计算资源 | 不共享 GPU；若共享，须能由统一父调度按共驻余量串行，且各自余量门通过 |
| 数据实例 | 同一冻结实例（`data_identity` 一致），或各自已冻结实例 |
| 可变状态 | 不共享 checkpoint / cursor / optimizer / store 写 |
| 可独立验证 | 每臂有独立 `protocol.json` + digest + 证据页 + index record |
| 多重比较 | 探索性比较**不触碰 test**；`r`/alpha 纪律按科学合同执行 |

### 2. 选择编排形态

| 形态 | 何时用 | 代价 |
| --- | --- | --- |
| **串行（默认）** | 任一 gate 不过 | 最慢但最可控 |
| **普通 Agent 扇出** | 2–4 个独立只读或隔离写子代理，一次汇总 | 需主链汇总，无控制流 |
| **dynamic workflow 脚本** | 确有控制流（循环 / 条件 / 依赖 gate）、阶段多、需脚本内聚合 | 先加载 `dynamic-workflows` 技能；脚本要类型检查并由用户确认 |

先用普通扇出；只有**确实存在控制流或阶段依赖**时才写脚本。workflow 脚本存 `.zcode/workflows/`（决策 0001），
属工具运行态、不进版本控制（R-035）；本仓此前从未实际使用过该通道（E-163/E-164：目录与表 0 行），
**首次真实使用必须留痕并验证**，不得先假定它可用。

### 3. 安全与隔离

- `AGENTS.md:155-156`：工具事件载荷无子智能体身份（E-186），**子会话很可能不跑 hook**（E-187，推测）。
  因此**并行子代理不能依赖 hook 拦截**。
- 默认只读。需要写时只写各自排他 `outputs/`；**禁止**写 `data/raw|interim|processed`、归档快照、
  既有 `outputs/`、`tests/` 下已跟踪内容。
- GPU 只能由统一父调度按共驻余量**串行**（R-054）；禁止对非本实验进程发信号；禁各自抢卡或重复训练。
- 联网：准备期可联网；实验步骤按 R-028 真禁网（monkeypatch `socket.socket.connect` /
  `create_connection`），不靠环境变量冒称禁网。

### 4. 科学完整性

- 每臂开始前写 `protocol.json` 并记 digest（R-006）；判据只看已冻结文档。
- **test 保持封存**；不得并行探索后在 test 上挑赢家；不得因并行而放松 `r`/alpha 累积纪律
  （见 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` §5）。
- 各臂独立证据页与 index record；负面与失败照实保留，不追认。
- **并行 ≠ 共享预算**：各臂各自记账，各自 `planned`/`hard` 与 overrun 分列。

### 5. 登记与回写

并行轮收尾时回写：master `<!-- round-node: X -->`、账本行（每臂各自一行）、index records、
`<!-- campaign-state: {...} -->`；然后跑

```bash
.venv/bin/python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md
```

注意：该校检器只校验**已存在账本行**的算术与指针，**发现不了缺行** —— 并行臂的账本行要人工核对齐全。

### 6. 降级

- workflow 工具或服务端失败 ⇒ 退回**串行 Agent 扇出**，保留原错误，不伪称成功。
- planner 不可用 ⇒ 自规划，过 `tools/check_planner_plan.py` 同一契约并做独立内容审阅。
- 降级不阻塞：资料/CPU 工作继续，GPU 实验按余量排队。

## 检查点

- **步骤 1 后**：七条 gate 能逐条写出结论吗？写不出就别并行。
- **步骤 2 后**：真的需要脚本控制流吗，还是普通扇出就够？
- **步骤 3 后**：有没有任何并行臂会写共享路径？有 ⇒ 改只读或改串行。
- **收尾前**：每臂都有独立 protocol/digest/record 吗？账本行齐全吗？

## 常见失败

- **把「能同时跑」当「该并行」**：共享 store/checkpoint 的臂并行会互相污染，结论不可归因。
- **靠 hook 兜底安全**：子会话可能不跑 hook，安全必须靠只读/隔离写约定 + Stop hook + CI 兜底。
- **并行探索后看 test 挑赢家**：违反封存与 `r`/alpha 纪律，等于把 test 变成 val。
- **把 workflow 脚本当通用加速**：脚本只解决控制流，不解决资源冲突与多重比较。
- **只列子代理名而不实际调度**：触发条件命中就要真委派/真加载。
- **并行当共享预算**：把两臂的时长记成一轮，掩盖真实成本。

## 完成判据

- 准入判断结论与依据已写入文档；
- 每臂有独立 protocol + digest + 证据页 + index record；
- 收尾 `check_campaign_state.py` 仍退出 0，且账本行与 index 条数一致（人工核缺行）；
- 未使用 test、未放宽判据、失败与负面已保留。

## 明确不覆盖

- **不做科学判定**：阈值、判据、DONE 定义在 `docs/rules/` 与已冻结证据文档里；
- 不替代 `goal-loop`（目标推进）、`planner-delegation`（方案设计）、`bounded-study-run`（实验执行）、
  `result-freeze`（结果定稿）；
- **不授权**数据下载（`real-data-acquisition`）或 GPU 独占；
- 不承诺并行一定更快：资源冲突或共享状态会让并行得不偿失。
