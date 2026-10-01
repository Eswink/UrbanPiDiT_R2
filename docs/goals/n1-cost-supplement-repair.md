# n1-cost-supplement-repair：N1 独立 evaluation 成本补测的 v2 重测与 GPU 共驻政策落地

<!-- round-node: N1 -->

**状态：prepared（2026-10-01 扩围），尚未执行；任何 GPU 步骤都需按决策 0021 重新取具名授权。N1 保持 paused，
不因补齐成本工程项改写 48/72h 的 `cannot-distinguish`，也不宣告目标完成。**

本文件是主计划 `docs/goals/main-model-v2-campaign.md` 的节点 N1 的目标长文（N1 的第三个轮次：转向审计 →
补测失败 → 本轮修复与共驻扩围）。节点、账本、对表清单都在主计划里；本文件只写「这一轮怎么做」。开工第一件事
是按主计划 §3 对表（跑 `tools/check_campaign_state.py` 必须退出 0），再按 §4 执行。

## §0 Objective（可粘贴；实测 1024 字符）

> 本轮目标：执行 docs/goals/n1-cost-supplement-repair.md 的扩围轮——先在本机 GPU 上落地「默认共驻、用空闲显存跑实验」的取卡政策，再跑完 N1 的剩余成本验收。0 GPU-h 部分：D0 政策三件套（决策 0026 + docs/rules/ 细则 + AGENTS.md 一行：默认共驻、启动前只读余量门槛、禁止对非本实验进程发送任何信号、不实现冻结或终止自动化、独占须先取具名授权，并同步决策/规则索引、CHANGELOG 与本轮 E 条目）；D2 v2 计量实现（每次评估一个全新子进程、守卫由「他人进程即拒」改为「余量门槛」、拒绝时先落 guard_refusal.json、30 行契约与精确重放不变，先过定向测试与工程 CI）；D1 原因探针（纯 torch 最小复现，记录 allocated/reserved 字节与 memory_snapshot，≤60s GPU；纯 torch 无残渣则加一次复用进程的字节记录 P2）。GPU 部分：D3 执行那一刻按决策 0021 取具名授权（范围＝探针 + 原 6 checkpoint × 5 val 时效共 30 次评估；≤0.25 GPU-h、整轮 ≤1200s、共驻、失败即停全额计费不重试）；D4 执行并登记（attempt/result/四张终态成本表/cost_views、证据页 docs/R7_N1_COST_SUPPLEMENT_V2.md + E 条目 + 索引记录 outcome_class: audit + 账本行 + 本长文与主计划回写 + CI 绑定）；D5 只提议：成本补齐不改写原 48/72h 的 cannot-distinguish 与 N2d 提议。判据与边界见 docs/goals/n1-cost-supplement-repair.md §3–§6，节点与账本见主计划。禁止：改已冻结判据或零基线要求、重跑或覆盖 v1 失败目录、新增训练更新/臂/seed、读 test、下载数据、租 GPU、改 main、force push、关闭 #70–#75、自动推进节点、对非本实验进程发任何信号。预算 ≤0.25 GPU-h（账本结余 19.9238 之内）；停止条件为两卡余量始终不足、探针无法归因、预算用尽、需要新判据或新数据或租 GPU 或合并 main，或任一评估失败。不要自行宣布目标完成。

## §1 现状（2026-10-01）

- 起点：主计划 §8 `current_node=N1`、`status=paused`；账本已用 4.0762 / 余 19.9238 GPU-h（开工时重读）。

### §1.1 v1 失败的 0 GPU-h 审阅（已完成，结论照旧有效）

- 上一轮（一次具名授权、单次、无重试）在 30 项里完成 1 项后失败：第二项评估调用**之前**，清零守卫
  （`training/r7_n1_cost_replay.py:268-273`）在 `gc.collect()+synchronize+empty_cache` 之后仍读到非零
  allocator 基线并抛出 `RuntimeError`（worker 日志
  `outputs/r7_n1_eval_cost_supplement/worker_seed41_process_spacetime_rwa.log:11-15`；12h evaluator 未被调用）。
  失败证据与独立只读复核：`docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`、
  `outputs/r7_n1_cost_failed_independent_review.json`；两者本轮只读，不重跑、不覆盖。
- **本次 0 GPU-h 审阅已确认**（命令见 §1.3，全部只读）：
  1. 失败发生在第二项**未开始**时——守卫是该项第一个触碰 CUDA 的步骤（之前只有 CPU 哈希与设备属性查询），
     因此非零基线只可能是第一项遗留，不是评估代码抛错；
  2. 跑过的评测器与归档 `code.zip` 逐字节相同（`training/r7_evaluate.py` sha256 `95fff1be…`）；
  3. checkpoint 以 `map_location='cpu'` 载入（`training/r7_experiment.py:125`），不产生 CUDA 常驻；
  4. 本路径仓内唯一 `lru_cache` 只返回 CPU 张量（`model/sparse_process_graph.py:6-18`，键 `(h,w,diagonal)`，
     值由 `torch.tensor(...)` 构造，无 device）；对 model/training/data 的 `lru_cache`/`global`/`register_hook`
     扫描未发现会持有 CUDA 张量的模块级状态；
  5. 环境为 torch 2.11.0+cu128（与归档 protocol 记的原版本一致），存在 `torch._C._cuda_clearCublasWorkspaces`。
- **仍未确认（禁止写成结论）**：失败基线的具体字节与残渣持有者。**推测**（与第 1 条形态一致、且无代码级持有
  证据）：torch 进程级工作区（cuBLAS/cuBLASLt 族）在进程生命周期内持有、gc/empty_cache 释放不掉；但未经探针
  验证。守卫拒绝时未记录字节属**实现缺陷**，v2 修（D2）。
- **由审阅得到的修复设计**：零基线由**构造**保证——**每次评估一个全新进程**（30 个子进程），不依赖任何释放 API。
  该拓扑有存在性证明：失败尝试的第一项就是一个全新进程里完成零基线计量与精确重放。备选方案（在
  `reset_measurement` 里调用私有释放 API）本轮**不用**：它依赖残渣恰好是该池且引入私有 API；记录为已考虑、未采用。
- 时间基准（供预算）：原归档 30 个 cell 的评测循环合计 102.2 s（2.3–4.9 s/cell，取自
  `outputs/r7_72_frozen_z/seed*/evaluation/*/provenance.json` 的 `elapsed_seconds`，不含装载/气候态）；
  失败尝试中同一 cell 全新进程整调用 11.99 s ⇒ 每进程固定成本 ≈9 s。

### §1.2 共驻扩围前提（2026-10-01 用户决定；只读观测）

- 用户在本轮准备期决定：本机 GPU 实验改为**默认共驻**——不等待整卡空闲、不要求腾卡、不干扰他人作业，
  显存余量够就开始；该政策作为 **D0** 落地为决策 0026、`docs/rules/` 细则与 `AGENTS.md` 一行约束。
- 只读观测（2026-10-01 17:46 CST，`nvidia-smi --query-gpu=index,memory.free,memory.total`）：
  GPU0 free 18922 MiB / 24576（外部 PID 3650861 占 5156 MiB，同账号另一作业）；
  GPU1 free 13561 MiB / 24576（外部 PID 4131217 占 10538 MiB）。两卡空闲余量都比本补测的峰值
  （首项实测 allocated 39 590 400 B ≈ 37.8 MiB / reserved 46 137 344 B ≈ 44.0 MiB）高两个数量级。
- 计量口径：v2 的 allocated/reserved 峰值是**本进程** allocator 计数，邻居进程不进入该口径；
  共驻对 wall time 的可能影响如实记入 `limitations`，不作判据、不剔除。
- **旧的取卡表述自本节起废止**：本长文此前出现的「一卡独占 / 等待空闲 / 用户手动腾卡」一律不再使用，
  替代为「余量门槛共驻」（§5）。判据、零基线要求与任何数值容差均不放宽（§3.2）。

### §1.3 审阅用到的只读命令（可复核）

```bash
sha256sum docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md          # 4b350357…（与索引记录一致）
unzip -p outputs/r7_72_frozen_z/code.zip training/r7_evaluate.py | sha256sum   # 95fff1be… == HEAD
grep -n "map_location" training/r7_experiment.py         # 125: map_location='cpu'
grep -rn "lru_cache" --include="*.py" model/ training/ data/ scripts/
.venv/bin/python -c "import torch; print(torch.__version__, hasattr(torch._C,'_cuda_clearCublasWorkspaces'))"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader   # §1.2 观测
```

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D0 | GPU 共驻政策落地（0 GPU-h，**先于执行**） | 决策 `docs/decisions/0026-shared-gpu-coresidency-policy.md`（四段齐全含负面后果：默认共驻、启动前只读余量门槛、禁止对非本实验进程发送任何信号、禁止实现冻结/终止自动化、独占需求须先取具名授权）；`docs/rules/` 一条细则（按仓库惯例标注级别/依据/现状）；`AGENTS.md` 硬约束区一行；同步 `docs/decisions/README.md`、`docs/rules/README.md`、`docs/rules/CHANGELOG.md` 与本轮 E 条目 |
| D1 | 原因探针 P1（条件 P2） | P1：纯 torch 最小复现（初始化 → matmul → del → gc/sync/empty_cache → 读 allocated/reserved → `_cuda_clearCublasWorkspaces()` → 再读数），输出 `probe_residue.json`（字节 + `memory_snapshot` 摘要）到 v2 输出目录；≤60s GPU。三态读法见 §3.1。P2（仅当 P1 纯 torch 残渣=0）：同进程两次原评估 + 字节记录，≤120s GPU |
| D2 | v2 计量实现（就地修订 v1 的四个计量文件）+ 测试与 CI（**执行授权之前**） | 每次评估一个全新子进程（30 子进程）；守卫由「他人进程即拒」改为**余量门槛**（保留 UUID 绑定与读时点，仍不发任何信号、不终止任何进程）；守卫拒绝时先落 `guard_refusal.json`（字节 + snapshot 摘要）再抛错；30 行契约、精确重放、四张终态表、`cost_views.json` 不变；v1 的 `measurement_code.zip`（`b7f20e7b…`）与失败目录只读。先过 v2 语义的定向测试 + 干净 `git clone` 全量，CI run 绑定准备提交的 SHA |
| D3 | 具名授权（决策 0021） | 执行那一刻 AskUserQuestion：范围＝探针 + 30 次原 val 评估；建议 ≤0.25 GPU-h / 整轮 ≤1200s / **共驻（余量门槛）** / 失败即停全额计费不重试；回执 JSON 冻结后才执行；回执声明设备政策与 v1 的差异（不再含「competing GPU process」与「no exclusive GPU」停止条件） |
| D4 | 执行与登记 | attempt/result/四成本表/cost_views；证据页 `docs/R7_N1_COST_SUPPLEMENT_V2.md` + E 条目 + 索引记录（`outcome_class: audit`）+ 账本行 + brief 与主计划回写 + CI 绑定 |
| D5 | 只提议 | 成本验收补齐后仍只提议 N2d；不改写原 `cannot-distinguish`，不推进节点 |

## §3 判据与预声明读法

本轮判据全部引自冻结文档：主计划 `docs/goals/main-model-v2-campaign.md` §2/§3/§5/§7、
失败契约 `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`、`docs/rules/ci-and-verification.md`、
决策 0021/0025/0026（汇总见 §3.3）；不新增阈值、不比 v1 放宽（v1 冻结契约整体继承，仅修改零基线的
**实现方式**与设备**占用政策**）。

### §3.1 探针三态（预声明，不改判据）

- ① 纯 torch 残渣 >0 且被 `_cuda_clearCublasWorkspaces()` 清到 0/0 → 归因「torch 进程级可释放工作区族」；
- ② 纯 torch 残渣 >0 且清不掉 → 归因「torch 进程级不可释放残渣」（记录 snapshot 摘要）；
- ③ 纯 torch 残渣 =0 → 触发 P2，归因「项目路径持有」；作为对冻结评测器的**缺陷发现**记录（不在归档里热修）。

三种情形下 v2 拓扑（逐评估新进程）都不变；探针只写归因，不改任何判据或容差。

### §3.2 v2 接受判据（与 v1 冻结契约逐条一致，不新增容差）

30 行齐套（seed×arm×lead 各一）；逐行 `baseline = 0/0`；`0 < peak_allocated ≤ peak_reserved`；
逐 cell RMSE/provenance 精确重放；四张终态表 + `cost_views.json` 产物齐；GPU 计费 ≤ 授权上限；
test 未读、0 训练更新。任一评估失败 ⇒ 即停、全额计费、保留部分、**不重试、不放宽零基线或任何判据**。

共驻只替换设备占用政策（余量门槛替代独占要求）：`peak_allocated/peak_reserved` 与 RMSE 重放均为本进程
确定性口径，不因邻居进程改变；`elapsed_seconds` 受共驻影响的可能如实记录并写进 `limitations`——
它是测量值，不是判据，既不被剔除也不构成失败。

### §3.3 证据来源

主计划 §2/§3/§5/§7；`docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`（失败契约与失败语义）；
`docs/rules/ci-and-verification.md`（运行资格与门禁）；决策 0021（实验授权通道）、0025（主计划与每轮对表）、
0026（GPU 共驻政策，D0 产出）。

## §4 实施顺序（不跳步）

1. **对表**：重跑 `tools/check_campaign_state.py`（退出 0）、重读账本、核 v1 失败页/复核 JSON 的哈希未变。
2. **D0（0 GPU-h）**：写决策 0026、规则细则与 `AGENTS.md` 一行；同步三处索引与 E 条目；门禁 + 定向测试通过。
3. **D2（0 GPU-h）**：就地修订计量四文件与探针；定向测试通过；工程 CI 绿；干净 clone 全量核数。
4. **D3 取授权**：按 §3.2 的范围与预算提问（含共驻设备政策）；**未获授权即停在准备态并报告**，不以存量余量代替许可。
5. **D1 + D4 执行**：同一授权内按序：探针子步骤 → 30 次评估（每次全新进程）→ 四表；任一失败即停。
6. **D4 登记**：证据页、E 条目、索引记录、账本行、brief 与主计划 §8 回写、CI 绑定；**D5 只提议**。

## §5 预算与停止条件

- 探针 ≤60s GPU（条件 P2 ≤120s）；D0/D2（政策落地 + 计量实现 + 测试 + CI）为 0 GPU-h。
- 执行建议值：**≤0.25 GPU-h / 整轮 ≤1200s**（以授权回答为准；主计划单次实验 ≤30 min；账本余 19.9238）；
  cap 常量按执行授权值同步修改。
- **设备政策＝余量门槛共驻**（D0 的工程参数，不是科学判据）：启动与每次子进程 spawn 前只读查询该卡
  `memory.free`，任一卡 ≥2048 MiB 即可用；外部 PID 列表只作记录，**不作为拒绝依据、不发任何信号**；
  两卡均低于门槛时只读重试（间隔 ≤60s、总时长 ≤600s），仍不足则停止并报告。
- **停止条件**（满足任一即停并向用户报告，不自行扩大范围）：两卡余量始终不足；探针无法归因；
  预算用尽或授权被拒；需要新判据/新数据/租 GPU/合并 main；任一评估失败（即停、全额计费、不重试）。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不重跑、不覆盖、不改写 v1 失败目录（含 `measurement_code.zip`）与任何归档产物/历史证据页（`outputs/` 一律只读）。
- 不改零基线要求；不改已冻结判据、阈值、端点或案例集；不新增训练更新、臂或 seed。
- **不对任何非本实验进程发送信号（含 SIGSTOP/SIGCONT/SIGTERM）；不实现冻结/终止/抢占他人作业的自动化；
  不新增常驻轮询守护或自动调度**（共驻政策只读观测，不做任何写动作）。
- 不读封存 test；不下载数据；不租 GPU；不动 main；不 force push；不合并；不关闭 #70–#75。
- 不把「零基线由构造保证」的工程修复写成科学结论；不改写 48/72h 的 `cannot-distinguish` 与 N2d 提议。
- 不自动推进下一节点；不建立定时任务或后台续跑。

## §7 进度块

- **状态**：`prepared`（2026-10-01 扩围）；未执行任何 GPU 步骤；账本未增行。
- **准备期对表（2026-10-01）**：`check_campaign_state.py` `failures=0 notes=6`；账本逐行合计 4.0762/余 19.9238 复核通过；
  v1 失败页/复核 JSON/三个提交/CI `36737308495`（`9d8d2b6`，九步全绿）均在 git 与匿名 API 上核过，见主计划 §8。
- **扩围（2026-10-01）**：用户决定「默认共驻、用空闲显存」并授权把它写成本轮 D0；旧「一卡独占/等腾卡」表述
  废止（§1.2）；§0/§2/§4/§5/§6 同步扩围；主计划 §8 加扩围说明，状态块与账本不动；仍为 prepared，
  未执行任何 GPU 步骤。扩围提交 `d8dc6fd` 的 CI `36845897605` 九主步骤全绿（2026-10-01 匿名 API 只读核对；
  实验 workflow 全 skipped 是标签门控的设计行为）。
- **本轮由此被写死的前提**（退出条件照抄主计划 §5）：不一致即停并问用户；执行前重跑对表。
- **下一动作**：N1 保持 paused/current_node=N1，等用户触发；触发后从 §4.1 开始，先重跑对表，再依次落
  D0 → D2 → D3；执行那一刻按决策 0021 取具名授权（共驻）。不改写 48/72h 的 cannot-distinguish 与 N2d 提议，
  不自宣完成或推进节点。
