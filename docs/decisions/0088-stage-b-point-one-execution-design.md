# 0088 — Stage B 第一点（0.25 epoch）的执行设计：定标曲线的一个纵坐标，不是命令

- **日期**：2026-11-04
- **状态**：accepted
- **范围**：`scripts/study_r7_s3_throughput_physics_b.py`、
  `scripts/r7_s3_throughput_physics_b_protocol.py`、`tests/test_r7_s3_throughput_physics_b.py`，
  以及执行源 `docs/goals/s3-throughput-physics-b-p0025.md`
- **依据**：决策 0084（六 Stage 与判据拆分）、0085/0086/0087（Stage A 的三项工程改动）、
  `docs/goals/s3-throughput-and-physics-campaign.md` §3（饱和判据原文）、
  `docs/plans/0028-throughput-physics-campaign.md` §2 Stage B（设计与成本口径）

## Context

**① Stage A 的出口未达到，Stage B 仍必须跑——但理由要写清，不能含糊过去。**
Stage A 的出口是 batch 4 下 `< 1.0 s/update`，实测四次均值 **1.1279**（范围 1.0848–1.1812），
四次全部 > 1.0。按计划「达不到即停下报告、不改判据」，Stage A **停在这里**，
没有把 batch 6 的 0.9814 冒充成出口。目标长文 §7 因此把下一动作定为一个**判定点**：
接受 1.1279 直接进 Stage B，或先补一个计算侧前置轮。本 ADR 取**前者**，理由是可核查的：

- Stage A 修完之后，一个 update 里 **97.8% 是 fwd+bwd**、数据构造只剩 **1.2%**。
  继续改读法在数学上救不回那 8% 的差距，它只能靠改计算语义（K、bf16、activation
  checkpointing）来补，而每一条都要各自 ADR 且会改变本轮冻结的算力口径。
- Stage B 付得起这个价格：第一点 2,008 updates × 1.1279 s ≈ **2,265 s 训练**，
  四点全跑的估计从 43 GPU-h 上调到约 **57 GPU-h**（决策 0084 负面后果 1 已记）。
  57 GPU-h 不是「不可付」，只是比原估计贵 34%。
- 因此**不**新增一个计划外的「计算前置轮」。那会新增一个不属于计划 0028 的 Stage，
  并推迟本轮真正要回答的问题（72 h skill 随训练量走不走）。

**② 一个点不能判定饱和，这一点必须先说清，否则读数会被误用。**
目标长文 §3 冻结的判据是 `Δ = skill_72h(16 epoch) − skill_72h(4 epoch)`，
`Δ > 0.01` 未饱和、`Δ < 0.005` 且低于上一档饱和、其余只记录。这是一个**两点一阶差分**。
本点只有 0.25 epoch，`Δ` 在它上面**不可计算**。因此本轮的预声明出口必须是工程性的；
若把「这个点跑完了」写成科学结论，正是本轮要避免的那个错误（目标长文 §5）。

**③ batch 固定 4 的理由已经是实测的，不是偏好。**
R1 实测 updates 2,400 → 16,071 让 72 h skill 从 +0.0258 走到 +0.1527（数据不变）；
A6 重扫实测 batch 12–14 的 samples/s 只到 1.512（相对 batch 8 仅 +5%）而峰值涨 74%。
定标期改 batch 会同时改变 updates 与显存两条轴，污染「训练量 → skill」这条曲线本身。
batch 只在 Stage F 作为成本选项出现，且必须按 updates 折算。

**④ 一个 seed 是这一步的**上限**，不是节省。**
R2 用三 seed 是因为它在做**配对比较**（两臂同算力，比配对 Δ）。
Stage B 不比较任何两臂：它测的是**同一条配方在不同训练量下的纵坐标**。
加 seed 会把「曲线的形状」问题悄悄变成「某一点是否显著」问题，而后者在四个点上
本来就不成立（四个点各自 n=1）。因此本点 n=1，且 `limitations` 明写不给区间。

**⑤ 十七个变量必须逐字取，不能按印象写。**
本轮初稿的变量清单是按记忆写的（含 `msl`/`sp`/`tcc`/`tp`/`q700`/`t700`/`w700`/`r500`），
与真实 store 的 `channels` **不符**——真实清单是
`t2m,u10,v10,mslp,z850,t850,q850,u850,v850,z500,t500,q500,u500,v500,z250,u250,v250`。
这是本轮实际发生的一次错误，已改正为**从 store 的 `channels` 属性逐字复制**，
并由测试断言两者相等。凭证：`tuple(root.attrs['channels']) == SCORED_VARIABLES` 为 True。

**⑥ 只有 t2m 有注册气候态分母。**
D2 reference 只拟合 t2m 的 48 桶 climatology RMSE。其余 16 个变量**没有**注册分母，
因此报原始 RMSE、`skill` 为 `None`——**不发明分母**。把 16 个变量也写成 skill
会让读数看起来更完整，但那是自造判据。

## Decision

**1. 本点跑 `2,008 updates × batch 4`（0.25 epoch）、单 seed 41、单臂。**
`POINTS` 把四个点冻结成阶梯 `(p0025, 2008) / (p1000, 8033) / (p4000, 32132) / (p1600, 128528)`，
`POINT = "p0025"`，`UPDATES` 由 `dict(POINTS)[POINT]` 取得，`validate_protocol` 断言
协议里的点与本驱动的常量一致。另三个点是**各自独立的 attempt**，`_seed` 不接受任何
「从上一点续」的路径。

**2. 预声明出口是工程性的 (a)–(d)**（brief §4 原文）：
(a) `updates_this_run == 2008` 且 `batch_size == 4`；(b) 五个 lead × 17 变量全部有限、
cohort 与 `VAL_COHORTS` 逐字相符；(c) 72 h t2m skill 有限且分母是注册 D2 值；
(d) 无判据被改（`protocol_sha256` 自校验、`test_read is False`、17 变量清单未变）。
任一假 ⇒ 该点作废、从头重跑。

**3. 饱和规则逐字进协议，`saturation_verdict` 恒为 `null`。**
`SATURATION_RULE` 从目标长文 §3 原文抄录；`readings.json` 的
`saturation_note` 明写「一个点不可判，规则只在阶梯完整时求值」。
`evaluate_exit` **不**产出任何饱和判定。

**4. 桥 (e) 逐 seed 断言父身份。** `load_checkpoint` 的 `model_code_sha256` 检查 +
`_state_digest(model.state_dict()) == MIGRATED_PARENT_STATE_DIGEST[seed]`（155 张量）。
失败 `raise`，不进入 (a)–(d)。本轮**没有**零 query 聚合那类逐位判据，因此
**不涉及**决策 0082 的 float32 界问题（该坑是 R4 特有的）。

**5. checkpoint 与同步守卫按 0086/0087 生效。** `checkpoint_every=5000`，
本点只有端点一个 checkpoint；全模型有限性每 100 步 + 每 checkpoint + 端点。
`retained_checkpoints` 写进 receipt，与 0087 的预注册保留清单一致。

**6. 训练走 Stage A 的读法。** `fine_tune_long_rollout(..., loader=(4, 4))`：
4 worker、prefetch 4。这是**唯一**与历史臂不同的运行时设置，因此本点的
`training_code_sha256` 与任何历史轮都不同——这一点写进 `limitations`，
不冒充「与 R1/R2 同代码身份」。

## Consequences

**负面后果（代价，必须如实列出）**

1. **不新增计算侧前置轮是有代价的。** 每 update 仍比出口慢 12.8%（1.1279 vs 1.0），
   四点全跑因此从 43 GPU-h 变成约 57 GPU-h（**估计**，非实测）。若后来证明阶段 A 的
   1.0 出口本当达到，这 14 小时的差额就是「先跑 Stage B」的代价。
2. **n=1 意味着这条曲线的任何一点都不能给区间。** 若 16 epoch 点与 4 epoch 点恰好卡在
   `Δ` 的 0.005–0.01 之间，本轮只能「只记录、不外推」，不能靠加 seed 把它变成可判定——
   加 seed 是改判据的另一种形式。这条写进 brief 的停止条款。
3. **`training_code_sha256` 与历史轮不同**，因为 Stage A 改了读法与有限性检查频率。
   本点的读数**不能**与 R1/R2 的臂作代码身份层面的对照；能对照的只有「同配方、同数据、
   同父、不同 updates」这一层。引用时必须同时读这条。
4. **只有 t2m 有 skill 数。** 另外 16 个变量报原始 RMSE。若下游要 17 变量的 skill，
   必须先**注册**它们的气候态分母，那是另一份工作，不在本 ADR 授权内。
5. **四点是四次独立重跑。** 第三/四点（32,132 / 128,528 updates）分别约 10 / 40 小时训练，
   中途失败即全部作废、从头重跑。这是「无断点续跑」的直接代价（训练器
   `"no failed-attempt resume is supported"`）。
6. **本轮的出口全真也只是一个纵坐标。** 不得据此自宣 Stage B 完成、不得据本点判饱和、
   不得据此关闭或重开任何支。

**正面后果**：本点把曲线的第一个坐标钉在一个可核查的协议上——17 变量清单从 store 逐字取
并由测试断言、父身份逐 seed 断言 155 张量 digest、饱和规则逐字入协议并显式标为不可判。
父在 `59340710…` 修订下**实测可加载**（`model_code_digest()` 与钉值相等、
三个 seed 的 `parent_update_0001600.pt` 全部存在），因此本点不需要新迁移。

**否决**：把 16 个变量也编一个 skill 分母（自造判据）；
加 seed 到 3 以「更稳」（把曲线问题偷换成显著性问题和掩盖 n=1 的事实）；
把 `saturation_verdict` 在单点上填成 `false`（一个点不可判，填任何一个值都是假的）；
跳过 Stage A 的读法以减少代码身份差异（会白白多付 3× 时间）；
先补计算前置轮再跑 B（新增计划外 Stage，推迟本轮真正的问题，见 Context ①）。
