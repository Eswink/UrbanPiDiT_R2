# 目标：M1 已知时空输入贯通 + M2 RW-A 位置化 process 读写（#71/#72 第一轮）

本文件是 goal 模式（`/goal <objective>`）的**长文目标源**：objective 只有 4000 字符，
放不下的细节写在这里。objective 里逐条核对的交付物见 §2，停止条件与预算见 §5。

## 0. objective（可直接粘贴，单段 1373 字符，实测 ≤4000）

> 本轮目标：在 r7/weather-reasoning（起点 263fc62）上完成主模型 V2 的第一轮——把已知时空条件真正接进主模型、把 P.mean 广播换成位置化 process 读写（#71 + #72 的 RW-A）。代码 + 测试 + 一轮有界实验 + 证据文档，全部在已授权范围内，不需要用户再授权。细节与判据见 docs/goals/m1-and-rw-a-iteration.md，本 objective 只放要点。 交付物（逐条核对，未做就写未做）： D1 init 时刻、年内/日内相位、地理坐标进入 NativeAtmosForecaster.forward；相位只由 init+lead 计算，禁用未来观测与服务器时钟，缺字段显式失败。 D2 白名单 forecast_inputs 与固定/截断/流式/自适应/rollout 各路径读到同一组字段，并有测试钉住键集合一致。 D3 RW-A：每个输出位置用各自的 query 读 process token（N×M），替换 model/process_forecast_r7.py:216 的 mean 广播；仍是加法式更新，不做门控、不做局部状态。 D4 新通路默认关；关闭时在固定种子固定输入下输出与重构前逐位相同（等价性测试）；不改 digest 比对逻辑，不复用旧 checkpoint。 D5 因果探针：改未被读取的字段→输出不变；改相位或 lead→输出改变；扰动单个 process token→各输出位置响应不均匀（否则判定机制未生效并停止）。 D6 一轮有界对照实验：同一代码修订内的两臂（新通路关/开）、种子与 updates 固定、协议在第一次 optimizer.step() 前冻结、只读 val、不读 test；报告实测 GPU-h 与参数/FLOPs 增量，负面结果照写。 D7 证据文档 docs/R7_71_72_M1_AND_RWA.md：带 scientific_claim: false、limitations、未做的事、下一项。 D8 提交（治理层进版本控制）并核对 CI；skipped 的实验 workflow 是 commit-message 标签门控，不算失败。 纪律：不用 test 做方法选择（test 已被读过）；不放宽任何已冻结判据，不新增阈值；skip、取消、排队的运行不算通过；结论区分已确认与推测。 预算与停止：本轮 ≤1.5 GPU-h（第二批授权 ≤24 GPU-h，已用 1.204），单次实验 ≤30 min；超出即停并如实记录。达到预算、等价性或扰动测试失败、check_conventions 阻断违规、或必须依赖旧 checkpoint 分数才能完成对照时，停止并报告。需要新数据下载、租 GPU、合并 main、force push、写 issue 时，立即停止该分支并说明「需要用户授权」——这些不在本轮范围。 不做：RW-B 的门控与局部状态；M3/M4/M5；新数据下载；基线家族扩展；为让报告好看而改测试或放宽判据。 结束时报告：启动 SHA、改动面逐文件、实跑过的验证与结果、实验协议 digest 与实测 GPU-h、未做的事、下一项的第一个具体动作。不要自行宣布目标完成。

- 起点：分支 `r7/weather-reasoning`，HEAD `263fc62a99797acc5ab7cb86a276a3b4d3d2285f`
- 权威任务状态：GitHub `#70`（总计划）与 `#71`（M1）、`#72`（M2 核心）；
  本环境**不能**写 issue（`POST /issues` 401、无 `gh`，决策 0011 已豁免），
  所以推进以本文件与产物为准，不依赖任何 issue 状态变化。
- 来源：本目标由一次 **planner 委派**（工作态草稿，JSON 通过
  `tools/check_planner_plan.py`，`fence_stripped: true`）转写而来；
  §6 列出我**改掉**的 planner 事实错误，科学判据与预算上限均按仓库既有纪律写定。

## 1. 现状（已核实的事实，实施前请再核一遍）

| 事实 | 位置 |
| --- | --- |
| 主模型只吃 `coarse_history` 与 `lead_time_hours`；`context = tokens + lead` | `model/weather_forecaster_r7.py:36-49` |
| 编码器无任何位置编码（patch conv → window attention → norm） | `model/coarse_encoder.py:9-40` |
| 草稿反馈的瓶颈是 P 的平均：`summary = process_to_context(process.mean(dim=1))` | `model/process_forecast_r7.py:216` |
| solver 条件化是 `context + summary[:,None,:]`（广播），可选再加 draft token | `model/recursive_weather_r7.py:13-31` |
| 自适应/增益路径的输入白名单只有两个字段 | `model/r7_halting.py:22-27` |
| dataset 已给出 `latitude`/`longitude`/`grid_spacing_deg`/`lead_time_hours`，**但没有 init 时刻** | `data/r7_zarr_dataset.py:56-63` |
| 任何 `model/*.py` 改动都会改变 `model_code_sha256`，旧 checkpoint 在新代码下**拒绝加载** | `training/r7_experiment.py:31-40`、`:124-134` |
| 实验 harness（arms 硬编码、协议 digest、种子并行） | `scripts/study_r7_b2_multiseed.py:115-131`、`:958-980` |

数据与环境：M2 双月段 store 在 `outputs/r7_m2_segment/store/manifests`
（2016-01-01..2016-02-29，17 通道 65×65，8 个 `(month,hour)` 桶，val 22 / test 26 窗口）；
两张 RTX 3090（无 NVLink，**不追求 DDP**）；`.venv/bin/python`。

## 2. 交付物清单（objective 逐条核对用）

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | **M1 输入贯通**：init 时刻 / 年内·日内相位 / 地理坐标真正进入主模型 forward，且只在初始化时刻可得的信息上计算（`init + lead`，不用未来观测、不用服务器时钟；缺字段显式失败） | 改动 + 测试名与结果 |
| D2 | **M1 路径一致**：白名单（`forecast_inputs`）、固定/截断/流式训练、自适应推理、rollout 各路径读到同一组字段 | 测试名与结果 |
| D3 | **RW-A 位置化读写**：每个输出位置用各自的 query 读 process token（N×M，Perceiver-IO 式），替换 P.mean 广播；grid 侧写回同样位置化；**仍是加法式更新**，无门控、无局部状态 Z | 改动 + 形状/FLOPs 记录 |
| D4 | **旧路径逐位不变**：新通路由显式开关控制、默认关；开关关闭时在固定种子/固定输入下输出与重构前**逐位相同**；旧 checkpoint 身份不被静默兼容 | 等价性测试名与结果 |
| D5 | **新通路真的改变了输出**：字段级因果探针（改未读取字段→输出不变；改相位/lead→输出改变）与 process token 位置扰动（各输出位置响应不均匀） | 测试名与结果 |
| D6 | **一轮有界对照实验**：同一代码修订内的两臂（新通路关 vs 开），协议在每个 seed 第一步前冻结、val only、test 不读；报告 GPU-h、参数/FLOPs、逐变量每时效结果与负面结果 | 产物目录 + `protocol.json` digest + 实测 GPU-h |
| D7 | **证据文档**：`docs/R7_71_72_M1_AND_RWA.md`，含 `scientific_claim: false`、`limitations`、未做的事、下一项 | 文件路径 |
| D8 | **提交与 CI**：worktree 干净（除忽略项）、CI 结果按 `r7/weather-*` 的 run 核对；skipped 的实验 workflow 是设计行为 | commit SHA + run id |

## 3. 判据与证据来源（verifier 不能调工具，所以判据必须能在对话里核对）

- **不得由 planner 定义判据**；本轮的判据引用以下**已冻结**文档，不新增阈值：
  - `docs/R7_B2_MULTISEED.md`：协议 digest 在第一步前冻结、seed-paired 同向才算赢、
    不一致记 unresolved（本仓既有裁决方式）。
  - `docs/R7_67_PUBLICATION_PROTOCOL.md`：主表成员、指标约定、**test 已被读过**的声明纪律。
  - `docs/R7_69_BUCKET_EXPANSION.md`：8 桶段、val 22 / test 26、单位（物理 vs 归一化）
    不得再互除的教训（决策 0010）。
  - `docs/rules/testing.md`、`docs/rules/ci-and-verification.md`：测试与 CI 判定纪律。
- **本轮不得用 test 做方法选择**（M2 阶段 test 已读过，`read_count: 1`）；选型只读 val。
- 「算通过」的最小定义：D1–D4 的测试在 CPU 上实跑通过、D5 的探针在实跑中确有非零响应、
  D6 的实验有冻结协议与实测 GPU-h。**skip 不算通过，被取消/排队的 run 不算通过。**
- 结论必须区分「已确认」与「推测」，并写明可复现等级；`process` 与 `generic` 的差别
  在既有证据里不可分离（#65/M2 均为 unresolved），本轮不得把它当作已证机制。

## 4. 实施顺序（依赖顺序，不可跳步）

1. **recon**：记录 HEAD SHA、环境、上面 §1 的 8 个位置；确认 M2 store 可达。
2. **M1 schema**：`data/r7_zarr_dataset.py` 从 store 自带的 `time_ns` 导出 init 时刻字段
   （`init_utc_hour`、`init_day_of_year`；可选 `init_year`），加进 sample 字典；
   缺字段时抛错，禁止用服务器当前时间兜底。
3. **M1 forward**：`NativeAtmosForecaster` 读取这些字段；**per-token** 位置项由
   经纬度网格（`latitude`/`longitude`，已有）映射到 token 网格后加入，**不要**把
   lat/lon 压成标量 mean（那会丢掉空间结构）；相位用 `init + lead` 计算（跨日/跨年取模）。
4. **M1 白名单**：`forecast_inputs` 同步新增字段；写断言钉住「白名单键集合 == 主 forward
   实际读取的字段」，防止两条路径再次漂移。
5. **M1 因果探针**：改未被读取的字段→输出逐位不变；改 init 时刻/lead→输出改变（阳性对照）。
6. **RW-A 层**：新增位置化读取（每输出位置一个 query，cross-attention 读 process token），
   与旧 `process_to_context` 并列；`solver_conditioning` 增加形状分支（[B,N,D] 直接相加，
   不再 `[:,None,:]` 广播）。
7. **RW-A 开关**：新通路默认关；开关关闭时**逐位**等于重构前（用固定种子固定输入断言）。
8. **RW-A 扰动测试**：扰动单个 process token 后各输出位置的响应不均匀；否则视为「只是重排了
   mean」，机制未生效，停止并回到设计。
9. **参数/FLOPs**：用 `training/r7_budget_audit.py:47` 的 `count_forward_flops` 记录两臂的
   参数与 FLOPs 增量（新通路必然更贵，必须如实报告而不是说成等成本）。
10. **协议冻结**：两臂（新通路关/开）、固定种子、固定 updates、val only；
    协议 digest 在第一次 `optimizer.step()` 前写入每个 seed 的 `protocol.json`。
11. **跑实验**：一卡一进程（3090 无 NVLink），跑完 finalize 汇总；记录墙钟与实测 GPU-h。
12. **写证据文档**、**提交**、**核 CI**；报告按 AGENTS.md 的九项。

## 5. 停止条件与预算

| 项 | 值 | 依据 |
| --- | --- | --- |
| 本轮 GPU 上限（自设） | **≤1.5 GPU-h** | M2 实测 800 updates ≈ 0.08 GPU-h/臂-seed；本轮 2 臂×2 seed×400 updates ≈ 0.16 GPU-h，留余量 |
| 第二批授权上限 | ≤24 GPU-h，已用 1.204（M2 训练） | `docs/goals/full-auto-campaign.md`（用户 2026-09-27 授权），**不得自行扩大** |
| 单次实验墙钟 | ≤30 min，超出自行拆批 | 同上 |
| 新产物 / decoded | ≤100 GiB / ≤256 GiB（本阶段） | 同上 |

停止条件（任一触发即停并如实记录）：

1. 本轮 GPU 用量达到 1.5 GPU-h，或第二批累计接近 24 GPU-h；
2. 单次实验超过 30 min；
3. D4（旧路径逐位不变）失败——说明重构破坏了既有行为；
4. D5（位置扰动非均匀）失败——说明新通路未真正生效；
5. `python tools/check_conventions.py` 出现阻断违规；
6. 发现无法在同一代码修订内完成对照（例如需要动用旧 checkpoint 的分数）；
7. 需要任何**未授权**动作（新数据下载、GPU 租赁、合并 main、force push、写 issue）时，
   立即停止该分支并在报告里写「需要用户授权」，不自行扩大。

## 6. 与 planner 草案的差异（我改了什么，为什么）

1. **`data/r7_dataloaders.py` 不存在**（planner 的 open_question 引用了它）→ 实际构造 sample
   文件是 `data/r7_zarr_dataset.py:56-63`，且 `latitude`/`longitude`/`grid_spacing_deg`
   **已经在 sample 里**，缺的只是 init 时刻；训练 harness 注入点需按实际调用链确认。
2. **预算数字自相矛盾**：草案同时写「单 seed ~2.5 GPU-h」与「0.08 GPU-h/seed」，并据此给出
   「总预算 ~10 GPU-h」。按 M2 实测（1.204 GPU-h / 15 臂-seed @800 updates）重算为
   ≈0.08 GPU-h/臂-seed@800u，本轮约 0.16 GPU-h。
3. **lat/lon 不能压成标量 mean**：草案让 `Linear(2,dim)` 吃全格 mean。M1 的目的正是把
   「已知时空条件」变成**位置相关**的条件，压成全局标量会退化成另一个 lead-time embedding；
   改为 per-token 位置项（经纬度已在 sample 中）。
4. **补 FLOPs/参数表**：草案的 verification 只有 pytest 与 conventions，没有量化新通路的
   计算代价；本仓已有 `count_forward_flops`，必须报告（MD §6 要求「额外计算需实测并披露」）。
5. **补「默认值不可静默改变」**：新增通路必须是显式开关、默认关，且老 checkpoint 身份不被
   静默兼容（MD §5 M1 的要求），否则等价性无从证明、旧分数会与新代码混用。
6. **删除草案里「预估总预算 10 GPU-h」这类推算**，改为「按 M2 实测成本估算 + 实测值写报告」。
7. 草案给出的 `slug`/步骤结构保留（已过校验器）；`open_questions` 里关于
   「2 个 seed 符号不一致如何解读」按 §3 的 B2 规则处理：**不一致即 unresolved**，
   不因种子少而放宽。

## 7. 明确不做

- 不做 RW-B（门控 + 局部状态 Z）、不做 M3/M4/M5 的任何部分；
- 不用 test 做方法选择，不重称 test 为未见；
- 不新增数据下载、不扩基线家族（冻结 U-Net/AFNO）、不重跑历史 workflow；
- 不合并 main、不 force push、不租 GPU、不写/关 issue；
- 不为了「有产出」而加投机模块；不接受「参数对齐 = 机制被证明」的表述；
- 不修改 digest 比对逻辑来复用旧 checkpoint。
