# s2-climatology-mechanism-screening：主模型超气候态方向 S2 单因素机制筛选（#77 → #78 → #79）

<!-- round-node: S2 -->

**状态：active（2026-10-05）。本文件是 S2 节点的目标长文（逐机制筛选轮）。执行者先跑
`tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md` 对表，
再按 §2 的顺序推进；本轮的产物是**可证伪的单因素机制筛选结论 + 证据登记**，不是新的科学确认。**

## §0 Objective（可粘贴；持续生效）

> 在 docs/goals/main-model-climatology-campaign.md 的 S2 节点内，按 #76 的 P0→#77→#78 次序完成主模型超气候态方向的单因素机制筛选：每个机制一条冻结协议、一次一因子、三 seed（预声明 41/42/43）、同参数量/FLOPs/初始化对照，val-only、test 全程封存；#77 位置编码频带已登记为 screening-negative（主格 unresolved），随后做 #78 R-A 的 train-only 连续 6h 变化尺度重参数化解码（`Y = X_t + (d_c/s_c)*r_c`，loss 不变），R-B 的差值尺度 loss 另轮；#79 类型诊断臂并行设计、先 aux=0。每轮产物写进 docs/ 证据页并登记进 docs/R7_EVIDENCE_INDEX.jsonl，账本记实测 GPU-h。判据见 docs/goals/s2-climatology-mechanism-screening.md §3，权威计划 docs/goals/main-model-climatology-campaign.md 与 docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md。禁止：改已冻结判据/端点/案例集、读封存 test、跳过测试或弱化断言、伪造 PASS、用合成数据替代失败下载、付费/租卡/独占、main 合并、force/mirror、常驻/cron。预算为软件预算：S2 各轮合计软预算 2.0 GPU-h（超软预算继续并记 overrun，宽松硬上限 6.0 GPU-h 截断），下载沿用 S1 已冻结的每批字节/磁盘门槛。停止条件：任一实验的真实读数已登记且回主线动作明确，或硬上限截断，或需要保留授权（付费/独占/main）。

## §1 承接与现状

- 主计划：`docs/goals/main-model-climatology-campaign.md`（§2 D2、§7 账本、§8 进度块）。
- 科学合同：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（D1–D6 判据全文；正式确认门在 S4）。
- 已完成：S0（身份复核/差距审计，`docs/R7_S0_INCUMBENT_GAP_AUDIT.md`）、S1（四季 2017
  dev store，`docs/R7_S1_FOUR_SEASON_ACQUISITION.md`）、S2/#77（频带筛选，
  `docs/R7_S2_77_PE_BAND.md`，主格 unresolved → screening-negative）。
- 待做：#78 R-A（变化尺度重参数化解码，data 侧 `data/preprocess/r7_change_scale.py` 已就绪、
  sidecar 已发布；model 侧接线待做）、#78 R-B（差值尺度 loss，另轮）、#79（类型诊断三臂设计）。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | #78 R-A 冻结协议 + 双轮（v1 缺陷轮如有）/注册轮 paired comparison | `outputs/r7_78_*_pilot_v2/paired_comparison.json` + 协议 SHA |
| D2 | #78 R-A 证据页（注册读数、成本、确认/推测/未做、回主线动作） | `docs/R7_S2_78_CHANGE_SCALE.md` |
| D3 | change-scale sidecar（train-only d_c、独立身份、BUILD_COMPLETE） | `outputs/r7_s1_seasons_2017/change_scale/` |
| D4 | 证据索引记录 + canonical brief 重渲染 | `docs/R7_EVIDENCE_INDEX.jsonl`、`docs/R7_CANDIDATE_BRIEF.md` |
| D5 | #79 三臂设计（aux=0 首轮、现役路径验证、可证伪机制） | `docs/R7_S2_79_TYPED_DIAGNOSTICS_DESIGN.md` |
| D6 | 主计划 §7 账本与 §8 状态同步 | `docs/goals/main-model-climatology-campaign.md` |

## §3 判据与证据来源

> 每轮筛选的判据是**该轮冻结协议**里预注册的主格（变量/lead/配对/方向规则），决定文本在
> 第一步之前写进 protocol.json 并逐字执行；判定只认每 seed 增量同号，符号不一致即 unresolved
> 且不许求均值。全变量全 lead 如实报告、不并入判定、不事后改选主格。参考
> `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 的 D2/D4/D6 与
> `docs/goals/main-model-climatology-campaign.md` §2；异步确认门（D1 新门）只对 S4 的未见年份
> 实例生效，S2 任何读数不构成"超过气候态"的主张。

**#78 R-A 的本轮预注册（写作时冻结）**：对照开关是解码参数化
（`incumbent` 即旧行为 `Y = X_t + r_c` 的 normalized 空间；候选 `change_scale`
即 `Y = X_t + (d_c/s_c) * r_c`），loss 保持纬度加权 MSE 不变；读 t2m val RMSE 在 6h/12h 的
seed 配对增量；supported 需两 lead 同号且候选更低。若某 lead 数据缺失、或同号但方向为正，
如实写 worsened/unsupported。

## §4 执行顺序（每步先跑校检器对表）

1. `#78 R-A`：model 侧加入可选 decode 尺度（惰性开关，默认旧行为逐位不变）→ 单因素研究脚本
   （两臂、三 seed、400 更新、val-only，沿用 #77 的预算/门/合并链）→ CPU 反证测试 → GPU 两轮
   （v1 缺陷轮如实计费保留、v2 注册轮）→ finalize。
2. `#79` 设计文档（可与 #78 并行，CPU-only）：先验证初始/自生成草稿诊断真实进入 forward，
   再定三臂（现模型 / 同信息无类型融合 / 类型路由），首轮 aux=0。
3. #78 R-B 只在 R-A 完成后另立轮次；不得与 R-A 混合为一次改动。

## §5 预算与停止条件

| 项 | 值 |
| --- | --- |
| 本轮上限（软） | 2.0 GPU-h（超软预算继续并记 overrun） |
| 硬上限 | 6.0 GPU-h（到此截断，不科学通过） |
| 每 seed deadline | 4500 s（冻结于协议） |
| 停止条件 | 读数已登记且回主线动作明确；或硬上限截断；或需要保留授权 |

## §6 不做清单

不读封存 test（S2 全程 val-only）；不改已冻结判据/端点/案例集；不删弱化测试；不覆盖既有
outputs；不合成数据；不付费/租卡/独占；不 main 合并/force/--mirror；不建常驻/cron。

## §7 进度块

- **状态**：`active`
- **已完成**：#77（频带筛选，screening-negative）、#78 R-A（变化尺度重参数化，**worsened 双格**
  → screening-negative；data 侧模块+16 测试+sidecar、model 侧开关+9 测试、两轮 GPU 全部登记）、
  #79 设计冻结 + **实现与注册 GPU 轮**（三臂、aux=0；容量匹配归因配对 B→C 的 t2m 6h/12h
  三 seed 同号为负 **supported**，相对 incumbent 两主格 unresolved；接线逐 seed 可核验）。
- **#78 R-B 已完成并登记（2026-10-05，0.3533 GPU-h）**：与 R-A 的机制差异已在协议内写定
  （解码保持 identity，只把训练目标换成 `w_c=(1/runtime_ratio)^2` 归一化的逐通道权重，即 issue
  公式的 store 空间形式）；接线探针三 seed 相对误差 0.0；预注册主格 t2m 6h/12h 三 seed 同号为
  正 → **worsened**（+0.3101 / +0.4806），按冻结文本证伪。全 85 cell 如实报告（improved 21 /
  worsened 14 / unresolved 50）；test 未读；证据页 `docs/R7_S2_78_RB_LOSS.md`。
- **未做**：R-C（训练配方延长，另轮，现属 S3 轮次目标）；S4 未见年份确认门。
- **下一动作**：进入 **S3** 节点（`docs/goals/s3-confirmation-baselines-and-candidate.md`）：
  batch-2 四季 2022/2023 已获取、v2 确认实例已发布；#79 已将 typed evidence 送入下一确认实例的
  设计冻结候选（须新协议 + 预注册门）；S3 先做同数据气候态/incumbent 基线，再按 R-C 预注册筛选
  候选并冻结 S4 包，不因单个机制正结果跳过数据与基线前提。
- **成本**：S2 累计 **2.0919/6.0 GPU-h**（#77 两轮 0.6498 + #78 R-A 两轮 0.3759 + #79 0.7129
  + #78 R-B 0.3533）。**软预算 2.0 已超 0.0919 GPU-h：按冻结规则记为 overrun 并继续**（单个
  attempt 内 overrun 0；截断只由 6.0 GPU-h 硬上限触发，未到）。
