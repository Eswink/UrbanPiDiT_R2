# S3：吞吐前置、学习曲线定标、物理约束与有界重开的组合 campaign —— 65×65 多年实例

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-throughput-physics-campaign", "driver": "scripts/study_r7_s3_throughput_physics.py", "output_stem": "r7_s3_throughput_physics_20261104", "control_record": "s3-anchor-feedback-65x65-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_throughput_physics.py"], "planned_seconds": 259200.0, "hard_cap_seconds": 518400.0, "code_archive": "outputs/r7_s3_throughput_physics_code.zip"} -->

本文件是 的**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0028-throughput-physics-campaign.md`（全部实测数字、成本表与负面后果在那里）。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：prepared（2026-11-04 建立，未启动）。** `scientific_claim: false`、`test_read: false`、`r = 0`。
本文件不改任何已冻结判据、不改 `model/**` 之外的既有语义、不重建 store、不读 test。

---

## §0 Objective（单段，实测 3,999 字符）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-throughput-and-physics-campaign.md（先完整读它，判据与相位以它为准；设计与全部实测数字见 docs/plans/0028-throughput-physics-campaign.md）。起点：HEAD 407d20e、campaign-state used_gpu_h 401.377595、index 98 条、账本 79 行、current_round_goal 原指 R4；R4 停在 1440/16071 无读数、**不自动续跑**（无断点续跑，重开约 77 GPU-h，须具名授权）；R2 四出口全真但 needs-review、未进 S4；FY candidate 仍等用户验收 —— 这三件处置权在用户与复核者。转向依据三条：本模型仅 3,286,037 参数、单次前向 13.9 GFLOP，一个 update 是 12 次前向加全 BPTT 即约 500 GFLOP、GPU 时间不足 0.1 s，而实测 2.42–2.63 s/update，说明 95% 时间不在计算而在单线程 zarr 逐帧读（每样本读 47.5 MiB 只用 3.84 MiB，12.4 倍放大，chunk (16,17,64,64) 与逐时刻读错配）、每 20 步 fsync 加 reload、每步 155 张量强制同步；R2 证明同一机制在算力匹配后由 2/3 绝对门翻成四出口全真（72h skill +0.264772/+0.263564/+0.284917），R1 控制臂单靠 updates 2400→16071 就让 72h skill 从 +0.0258 到 +0.1527，故 22 条 negative 与 23 条 mixed 有一批是在约 1 epoch 下判死的；而主目标里的物理信息这一半从未被实验过——全仓唯一物理正则化是空间一阶差分惩罚且 gradient_lambda 默认 0.0，守恒、平衡、连续方程、谱损失、CRPS 零命中，legacy 有完整 physical_consistency.py 但当前不 import。按六个 Stage 推进：A 吞吐前置（A1 批量读窗口、A2 DataLoader 并行取数、A3 同步守卫、A4 checkpoint 策略、A5 四档实测、A6 batch 上限扫描；A1/A2 硬验收是 window_sha256 与 usable_windows 逐字节不变、固定 seed 前 200 步 loss 逐位相同，否则回退不合并；出口 batch 4 后小于 1.0 s/update，达不到即停下报告、不改判据；不重建 store，只改读法；A6 真跑完成（每档独立进程、数据构造计时：8/10/12/14 跑通，峰值 12918/15494/19270/22404 MiB，16 与 32 OOM，上限 14、共驻 10，12 至 14 吞吐饱和），A1/A2 后重扫；确需更大 batch 先试既有 activation_checkpointing，且大 batch 会减少每 epoch 的 update 数，故 B 一律以 updates 而非 epoch 为主轴）；B 学习曲线（单臂单 seed、treatment 清单、batch 固定 4 不随 A6 上限变动，因为 updates 而非 samples 才是 skill 主因、定标期改 batch 会污染该曲线，batch 只在 F 作成本选项且须按 updates 折算，四点 0.25/1/4/16 epoch 即 2008/8033/32132/128528 updates，每点全 17 变量 5 lead 评分，预声明判据：16 epoch 相对 4 epoch 的 72h skill 一阶差分大于 0.01 判定未饱和、最终 run 取 16 至 64 epoch，小于 0.005 且低于上一档判定饱和、取饱和点，介于两者只记录不外推；四点必须各自独立跑完，无断点续跑）；C 物理约束两轮（C1 无目标约束束=散度质量一致、比湿非负、露点不高于 T，对齐 legacy 的 w_wind_div 0.1 与 w_dew_leq_t 0.5 但在新模块重写不 import legacy；C2 导数域监督=惩罚预测与目标的散度涡度差，复用 model/r7_process_tensor_diagnostics.py 的 spherical_divergence 与 spherical_vorticity 算子；两轮 ADR 都必须论证与已关闭的空间梯度损失不是同一族，因为那条惩罚预测自身梯度是无目标平滑先验而 C1 是无目标约束、C2 是导数域目标匹配，论证不成立则不得开轮；若 C1 赢必须预注册后续分离束内成分的实验，不得把束当结论）；D 静态协变量接线与白盒化（把 static_identity 绑进 protocol_sha256 而非 data_identity，打通 static_conditioning 到 training 与协议构建器，修「接线报告称四出口已通过但 ADR 0069 仍 proposed」的漂移；这会改 model_code_sha256 故必须重做父迁移，照 0075 与 0083 先例；白盒化是 CPU-only 0 GPU-h，测 P 态 8 个 anchored 槽位与 8 个诊断目标的对应关系、Z 态逐位置求解进度与地形海陆静态场的相关性）；E 有界重开、以 B 的饱和判定为门（E1 R5 训练信号族只取 dense 12 步与 final-only 草稿监督；E2 容量轴须新 ADR 明确取代决策 0048 并取得用户具名授权，0048 无重开条款且其先例自我限制不得沿用；E3 H-C 只能按决策 0077 唯一允许的形态即同网格改操纵强度、低秩对全 N×N、算力与参数量分别声明、不得称 H-C 的重估；B 判定饱和则整段 E 跳过并把理由写进进度块）；F 最终配置菜单留用户手选，执行者只交付按实测吞吐重算的成本表，不代为选定。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑），从 .zcode/workflow-drafts/r7-campaign-round.dwf.ts 派生 .zcode/workflow-drafts/r7-throughput-physics-round.dwf.ts，maxRounds 自设硬上限、stage 超时按本轮实测抬到装得下整轮、world.run 的命令名保持编译期字面量且运行值落盘、自跑 tools/r7_round.py commit-guard（提交前）与 verify（登记后）不依赖 hook 兜底、只推工作分支 r7/weather-reasoning；它是有界可恢复的批量编排器不是守护进程，被打断按 ResumeWorkflowRun 续。每轮写新 ADR（四段齐全、Consequences 含负面后果、R-033）与新 brief（带预声明出口 (a)–(d) 与桥 (e)、任一假即永久关闭该支），过七 gate，先冻 protocol.json 再跑，跑完做四件套登记（index record、账本行、campaign §8、campaign-state）。三个实测的坑必须避开：桥 (e) 不得要求零 query 聚合逐位等于 mean(dim=1)（实测差 2.98e-08，会中止整轮，照决策 0082 改写）；code.zip 必须带 commit 注释否则 archive 拒绝启动；跑 stage 前先跑 tools/r7_round.py state 核对 current_round_goal，否则冻结错轮次。预算：A 约 2 GPU-h、B 约 43 GPU-h（76 h 硬上限）、C 每轮约 64 GPU-h、D 约 1 GPU-h、E 每轮约 64 GPU-h 上限三轮；无总 GPU-h 上限，cap 与 remaining 仅会计；时长按软预算加宽松硬上限，超软继续并记 overrun，只有硬上限因时长截断。禁止：读 2023 test、重建 store、新建 129×129 源、动 data/raw|interim|processed 与归档快照、合成数据、改冻结判据、事后放宽出口、换名重开、删弱测试、绕 checkpoint 与数据身份校验、重开 SAM 与 zero-base decoder 与 plain EMA 与长 lead 权重与 rollout 剂量这些方向明确的负结果、paid 与 rental 与 exclusive 与 cron 与守护与会话外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣 R2 通过独立复核、自宣 FY candidate 进 S4、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在 §6 写「为何停 + 恢复第一条命令」。每步给可核查证据（命令 + 输出 + digest），估计值不得写成实测值。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下一项。

---

## §1 起点与 podium

- 工作分支 `r7/weather-reasoning`；起点 HEAD `407d20e`（与 origin 同步，工作树干净）
- 节点 S3；`used_gpu_h 401.377595`、`remaining −381.377595`（仅会计，无总上限）
- 起点证据：`docs/R7_S3_POSITION_RETENTION_HANDOFF_20261103.md`（R4 停在 1,440/16,071、无读数）、
  `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`（R2 四出口全真、`needs-review`）、
  `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md`（R1 二次关闭年份轴，及其 `:401-404` 的参数量更正）
- **既有漂移，本轮修**：`docs/goals/main-model-climatology-campaign.md:389` 的合计行仍写
  `247.356641`，与机算 `401.377595` 不符；只改这一个字符串，不回写任何历史行

## §2 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 方向扩围 ADR | `docs/decisions/0084-<slug>.md` + `docs/decisions/README.md` 索引行 |
| D2 | 吞吐 ADR ×2 | A3 同步守卫、A4 checkpoint 策略各一份 |
| D3 | Stage A 证据页 | 四档 `seconds_per_update` + A1/A2 的逐位相同证明 + 是否回退的决策 |
| D4 | Stage B 学习曲线 | 四点 `readings.json` + 饱和判定（由驱动的 `evaluate_exit` 算，不手填） |
| D5 | Stage C 两轮 | 各一份 ADR + brief + 冻结协议 + 逐 seed receipt + 结果页 |
| D6 | Stage D 接线与白盒探针 | `static_identity` 绑进 `protocol_sha256` 的协议 + 可解释性证据页 |
| D7 | Stage E 各轮 | 各一份 ADR（E2 须显式取代 0048）+ brief + 结果页 |
| D8 | 四件套登记 | `docs/R7_EVIDENCE_INDEX.jsonl` record / 账本行 / campaign §8 / `campaign-state` |

## §3 判据与设计决定（本轮冻结；科学门只引 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与既有冻结页）

**本轮不发明新科学门。** 每个实验轮的出口在各自 brief 里预声明，格式照
`docs/goals/s3-year-coverage.md:68-79`（(a)–(d) + 桥 (e)；全真 → 只提请独立复核；任一假 → 永久关闭该支）。

本轮**自己新增**的判据只有 Stage A 与 Stage B 两条工程/定标判据：

- **A1/A2（工程，逐位）**：`window_sha256`、`usable_windows`、`excluded_sample_ids`
  必须逐字节不变；固定 seed 的前 200 步 `loss` 序列必须逐位相同。任一不满足 ⇒ 回退该优化，不合并。
- **A 出口**：batch 4 后 `seconds_per_update < 1.0`。达不到 ⇒ 停下报告实际倍数，**不改判据**。
- **B 饱和判定**：`Δ = skill_72h(16 epoch) − skill_72h(4 epoch)`。
  `Δ > 0.01` ⇒ 未饱和；`Δ < 0.005` **且** `skill_72h(16) < skill_72h(4)` ⇒ 饱和；否则只记录不外推。
  三种结果都必须原样登记，包括「饱和」这一与原始设想相反的结果。

### §3.1 并行准入七 gate

每个实验轮开轮前逐条核（`AGENTS.md:26`）：假设独立 / 产物隔离 / 计算资源 / 数据实例 /
可变状态 / 可独立验证 / 多重比较。任一不满足 ⇒ 串行，不并行。

## §4 数字预算与停止

| Stage | GPU-h（估） | 硬上限 |
| --- | --- | --- |
| A | ~2 | 24 h |
| B | ~43 | 76 h |
| C（每轮） | ~64 | 144 h |
| D | ~1 | 48 h |
| E（每轮） | ~64 | 144 h，上限三轮 |
| F | 见 `docs/plans/0028-*.md` §2 菜单 | 用户选定后另立 |

软预算超了继续并记 `soft_overrun`；只有硬上限截断 attempt（记 `budget_limited`、全额记账、不算科学通过）。
**不在运行中改硬限。**

停止三条：触保留授权边界；达已冻结科学停止条件；确无可安全独立推进的工作。

## §5 明确不做

- 不重建 65×65 store（会改 `data_identity`，断掉与全部 98 条已登记记录的可比性）
- 不新建 129×129 多年源（用户已选 65×65）
- 不动 `data/raw|interim|processed`、`data/legacy_v531/`、`model/legacy_v531*`、`legacy_v6/`
- 不读 2023 test（`r` 保持 0、`test_read: false`）；S4 未启动不变
- 不重开方向明确的负结果：SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量
- **不自动续跑 R4**（无断点续跑，重开是重跑约 77 GPU-h，须用户具名授权）
- 不代替用户验收 FY candidate 进 S4；不自宣 R2 通过独立复核
- 不改已冻结判据、不用换名重开、不为报告好看删改测试
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑；不对邻居进程发任何信号

## §6 进度与交接

**（执行者在此追加每个 Stage 的进度块；停时必须写「为何停 + 恢复第一条命令」。）**

- 2026-11-04 建立：状态 `prepared`，未启动，0 GPU-h。

## §7 下一动作

第一条命令（只读，先确认指针与起点）：

```
.venv/bin/python tools/r7_round.py state
```

确认 `campaign_state.current_round_goal` 已指向 `docs/goals/s3-throughput-and-physics-campaign.md`
（若仍指向 `s3-position-retention-reeval.md`，先改指针再冻协议；R4 期间真实发生过一次冻结错轮次）。
随后按 `docs/plans/0028-throughput-physics-campaign.md` §2 从 Stage A 开始。
