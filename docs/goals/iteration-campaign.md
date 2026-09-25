# R7 迭代战役：先修验收链，再验证天气过程递归的有效性（#59–#68）

本文件是 goal 模式目标的长文源（R-034）。objective 有 4000 字符硬上限，细节在此。

**状态：核心交付达成（2026-09-26）。** S0 三项（#60 `13ade3a`、#61 `92a2591`、
#62 `4901a42`）与 #63 read-plan（`091b2dd`）均已关闭，四个提交各自 CI 绿；
#63 的新年份获取 BLOCKED 于下载授权（成本选项已冻结在
`docs/R7_ERA5_V2_READ_PLAN.md`）。#64–#68 未启动，属后续轮次。在线权威来源是
GitHub #59（EPIC）及各子 issue；本文件是执行摘要，与 #59 冲突时以 GitHub 为准。
便携审阅副本：`01-R7_review_and_next_plan.md`（审阅提交 `6748b6f`，远程 CI run
`36130059293` = 887 passed/11 skipped/2 warnings/64.75s）。

---

## 0. 首要规则

1. 先读 GitHub 实时状态：#59 及子 issue、PR #12（已 merged）、最新 Actions；
   与本文件冲突以 GitHub 为准并记录差异。
2. **#59 明文：本计划不自动授权长训练、付费 GPU 或大规模下载。** 预算见 §2，超限先报
   成本与范围选项。
3. 历史 8 个 issue（#1/#5/#6/#7/#8/#9/#13/#20）已关闭，**不要重复宣称本轮关闭它们**；
   其中有界负实验保持归档，Issue 关闭不等于科研假设正向通过。
4. 真实数据纪律沿用 `.agents/skills/real-data-acquisition/SKILL.md` 与 R-004/R-008/
   R-016；工程政策沿用 AGENTS.md「GitHub 通道」（ff 推送 main 已放行、合并需授权、
   实验 workflow 不触发、issue 评论 401 → 证据写 docs + Closes keywords 经 main ff）。

---

## 1. 依赖顺序（#59 的执行表，不要跳）

```
S0-A #60（P0）──┐
S0-B #61（P0）──┤ 并行
S0-C #62（P1）──┘ 整理立即；重聚合等 #60
        ↓
S1 #63（数据 v2）→ S2 #64（可学习性→基线→多种子）→ S3 #65（定向消融）
        → S4 #66（自适应；仅当有效 fixed-K 前沿存在）→ S5 #67（论文协议）
#68（下载层传输边界）独立并行；#67 的指标解析测试可提前开展。
```

---

## 2. 预算与停止条件（冻结；来源 #59「建议预算与停止条件」）

- 第一阶段：本地新产物 ≤16 GiB、decoded source 读取 ≤64 GiB；初始训练每 arm ≤30
  分钟、总计 ≤4 GPU-hours。取得真实 step 成本与学习曲线后，下一批 ≤24 GPU-hours。
- **超预算先报成本与范围选项，不静默扩量**；缩小年份/变量/区域要给成本表请确认。
- NaN/OOM、数据或案例错配、train-only 拟合失败 → **停止该实验，保留失败并修原因**；
  不无限续训到赢。只有 GPU 步骤算本地运行，不把 CPU 修复/计划标成外部阻塞。

---

## 3. 各 issue 的验收判据（Acceptance 为准；全文以 GitHub issue 为准）

### #60（P0）比较器身份配对
`training/r7_coreasoning_compare.py` 的 summarize/compare 丢 seed ID 后按 list 位置
zip；隔离复现：仅重排记录即把 direction=improved 变 unresolved（均值不变）。修复：
1. 以完整 `arm/depth/variable/unit/lead/seed` 键连接；保留 dataset、model-code、
   更新预算与精确初始化/valid-time 清单身份。
2. 缺/重 seed、重复表项、变量/单位/lead 缺失、案例集不一致、非有限/负 RMSE 一律
   fail closed；不得靠相同 n_initializations 推断案例相同。
3. "种子均值 RMSE" 与 "跨案例 pooled RMSE" 分别命名，不混用、异单位不直接相加。
4. `sign_consistent` 只是描述统计；科研 gate 单独读取**实验前冻结的判据**——大量
   unresolved 时不得因剩余一项 improved 就宣布 gate 通过。
5. 历史结果用原始 checkpoint/result CSV 重聚合出新版审计差分；原始不覆盖；原始
   artifact 取不到 → 该项 BLOCKED，不补造 seed/case ID。

验收：随机重排输入、比较输出不变；全部缺项明确失败；targeted tests + 全 CPU CI +
conventions；**不动预报权重、不重新训练**。参考 WeatherBench-X 的显式维度/充分统计量
聚合与 #58 的 exact/common-case 审计契约（不建第三套比较器）。

### #61（P0）DDP 验证 K 与更新次数
`scripts/bench_r7_ddp_smoke.py` 把 `args.steps`（optimizer 更新数）传成了 reasoning
depth——10 更新/K4 的 smoke 实际验证 K10。顺序 A→E：
A 最小正确性（eval K 只来自 reasoning 参数；forward-hook 记录实际 K；负面 resume
契约测试；保留原 smoke 格式）；B 真实 manifest 短训（复用现有 local runner 契约，
单/双卡读同一全局样本序列）；C 尾批/累积/评估（padding 显式记录、每样本恰一次或
正确 mask、按全局有效样本数聚合、覆盖 length 31/33 与非完整累积组）；D streamed 与
DDP（多次 backward、encoder 边界梯度、no_sync、unused params 专项核验；未支持前明确
拒绝，不静默退回 full）；E 性能（同全局 batch 单卡 vs 双卡、真实区域、足够步数；SYS
拓扑下 DDP 更慢则两卡各跑独立 seed 作为有效结果）。

验收：CPU spy/AST + 动态假模型证明 `steps=10, reasoning_steps=4` 只验证 K4；resume
契约变化明确失败、同契约 resume 与不中断参考对齐（记录容差与随机态）；2 GPU 短测对
齐 rank 样本 ID/全局损失梯度/更新/rank0 唯一存盘/val 去重；full_bptt、
retained_truncated、streamed_truncated 分别标记，不互称等价。真实 CUDA 验收另列，
无 GPU 时仅 CUDA 部分 BLOCKED。参考 PyTorch DDP/AMP/DistributedSampler 官方文档与
本仓 `training/r7_streaming.py`、`r7_local_runner.py`、#21 梯度所有权测试。

### #62（P1）GPU 证据包 + 表文纠错
文档与表格不一致（不改原数值）：t500 48/72h 文字优劣与表相反；t2m 48h 改善漏记；
v250 的 feedback/no-feedback 臂说明相反；`use_forecast_feedback` 与
`spatial_solver_feedback` 不是同一消融，不得互证。三 seed 方向一致只是描述性稳定，
不是 "established"；GPU 计时须固定另一开关状态，3 seed×2 checkpoint ≠ 6 个独立 seed；
warmup1/measure2 只能称短微基准，不外推长训吞吐；SM 数与 compute capability 不混写
（8.6 不是 "82 个 SM"）。ACC 自校正：本库 ACC 是同一 climatology 的未再中心化 pooled
anomaly dot product——负 ACC 确实意味着相对该 climatology 的 MSE 不优，**不是 bug**；
新增显式 climatology RMSE/MSE skill 基线与一致性测试并记录前提。

交付：小型可下载审计包（code SHA、环境/设备、命令、protocol、逐 cell/seed/变量/
case 统计量、hash 清单、失败/跳过记录；大文件分包不进 Git 历史；不含令牌/隐私）+
CPU 脚本从审计包独立重生成内存/计时/三 seed 表并校验 metadata。原 1 warmup/2
measure 只能称短微基准。本地取不到的产物明确 BLOCKED 并列具体文件，不捏造；**不需要
再跑 17 条 tag-gated 旧实验**。

### #63（P1）连续区域 ERA5 v2
元数据/read-plan 立即可做；正式发布与比较依赖 #60。**不是再做 48 时次 smoke，也不是
默认批准大下载。** 范围冻结：原 0.25°、65×65（27–43N/107–123E）与现有 17 通道顺序
（从已发布 metadata 读取，不凭提示词重写）。候选 train 2016–2018、val 2019；
**2020 已有开发访问不能再当未见 test**；test 候选 2021 须先审计访问/训练重叠再封存。
D1 先取候选 train 连续 30 天做构窗/单位/IO 验证（不得报告期刊技巧）；D2 预算确认后
扩多季节/全年，每段留足 history 与 +72h 目标、不跨缺测跨 split 拼窗。显式保存初始化、
四 UTC 起报时刻与季节覆盖；评估不再截取"最前 8 个"；最终 test 封存访问。halo 仅来自
t 及其历史，**禁止未来 ERA5 作边界强迫**；比较模型信息预算相同。

读取成本：元数据先行，对 Earthmover temporal tiles 与 ARCO global-per-time 布局计算
真正触及的 chunk/level/variable 并集、解码字节、裁剪字节、磁盘峰值、网络上下限——
不为数十 MB 目标读数十 GiB 全球 slab 而不报告。上限：第一阶段新产物 ≤16 GiB、
decoded ≤64 GiB；不满足先给缩小成本表请确认；实际 HTTP 量未测到就填 null。只读
snapshot、原 pin 保留、新来源/时间/变量须独立 identity；核对 9 raw 变量/13 层与 17
通道对应，不混用 z 与 geopotential height。mean/std/气候态/过程 normalization 只拟合
train；审计 eps=1e-6 是否把 ~1e-8 水汽标签压成零（保留原值/方差/截断比例）。缓存按
时序连续 chunk；输出 BUILD_COMPLETE/schema/清单/checksum/许可证/receipt。lat/lon/
day-of-year/UTC-solar-time/lead 是可选已知输入，不得偷加未来天气；如加入，公平基线
同用并记录 ablation。缺下载授权只阻塞获取，不阻塞 read-plan 与离线测试。

### #64（P1）可学习性 → 强基线 → 多种子
正式比较依赖 #60/#62/#63；单卡独立 seed 不等 DDP 全部优化；既有 18M 显存 smoke 不重做。
B0 train-only 可学习性：固定真实 train 1/8/32 窗口，禁随机正则；最小 native/window/
generic 必须显著降低固定样本误差；查每通道 loss、反归一化、history 索引、lead 对应、
梯度是否抵达 encoder/solver；拟合不动先查实现/归一化/优化器，不跑长训。
B1 强基线：persistence、train-only climatology、U-Net、window/Swin-like、AFNO-small、
generic recursion——复用现有 adapter，先核对名实一致；1–5M 档、参数 ±5% 预注册、
记录实际 FLOPs/wall time/例数；"同更新"≠"同算力"；非神经基线不硬凑参数。
B2 受控多种子：先 2 个固定 seed 探索后冻结协议；确认性比较 ≥3 个**预先写定** seed；
native/generic 过 persistence/climatology 技巧检查（判据预先声明）；共享结构尽量对齐
公共初始权重；schedule/warmup/最大更新/验证频率/checkpoint 规则/早停规则训练前固定，
早停只读 validation。先 +6h 训练与 6–72h rollout；1→2→4 时刻的 curriculum 是另设的有
界实验，与内部 K 混淆。
B3 扩模：B0/B1 稳定后才考虑 15–20M，不默认冲 30M；两卡先各跑独立 seed/arm，实测 DDP
吞吐有利才用 DDP。负结果与 NaN/OOM 完整保留；"强基线可用" ≠ "ours 胜出"。
参考 AFNO/FourCastNet/Swin/TRM：核对实现、记 revision/license；小型区域重训叫
FourCastNet-style/AFNO-small，不冒称原版全球结果。

### #65（P1）定向诊断与最小消融
依赖 #60/#63/#64。对旧 checkpoint 的只读诊断可提前，但不得持续优化反复看过的 8/24
个 validation 案例。两个时间轴分开：内部 K 不推进物理时间；+6h/+12h 是 rollout 轴；
t 时刻 process proxy 不是未来真值。
预诊断先于堆模块：#53 更新几何（逐变量误差变化、修正范数、相邻修正 cosine；oracle
damping 只离线）；train-only 记录 forecast/process loss 对共享参数的梯度范数与夹角
（"夹角负"不是因果证据）；逐个 process proxy 的真实 std/eps 截断/缺值/标准化范围
（尤其水汽，防数值缩放支配）；同 model/data 签名记录 K0–K4 轨迹，trace 可审计但不
破坏 streamed 内存优势。
分阶段消融（不做全笛卡尔积）：C1 过程监督（固定 use_forecast_feedback=True、
spatial_solver_feedback=False、Ktrain=4；Generic16 free vs Process8+8 aux=0/0.01/0.1；
公共初始权重/数据序/更新预算对齐；权重候选运行前冻结）；C2 草稿反馈（分别切换
reasoner 的 use_forecast_feedback 与 solver 的 spatial_solver_feedback；Generic 获同样
spatial 机制；默认 False 保持到新协议通过）；C3 深度（同 Ktrain=4 checkpoint 测
K=1/2/4 + 独立训练的强 K1 对照；K6/8 是超训练深度不称免费 scaling；加等 FLOPs/延迟
非共享加深基线）；C4 条件改进（诊断+C1 显示辅助梯度冲突与退化相关才提 PCGrad 等
独立 issue）。判据：协议预先声明核心变量/时效/非劣容差/成本预算；完整报告 17 变量
（截断处见 GitHub 原文），负面与正向同表。

### #66（P2）自适应（WAITING_ON_RESEARCH_GATE）
仅在 fixed K=1/2/4 已形成验证集精度—成本前沿后启动（含独立训练的强 K1 模型，不是只
比退化 Kmax）；等待时可写离线校准/计时测试，不为填时间盲目再训停止器。历史 #7 负
结果保留：深 K 非自然精度上界、阈值曾回退 full depth。协议：冻结父模型/数据/代码/
归一化；gain controller 仍是最简实现，train 拟合 E_k−E_(k+1)，仅对真实存在的下一轮
给监督、最后一轮不伪造 target，future target 只进损失不进推理特征；阈值/容差只在独立
validation 上选择并在看结果前冻结（候选：RMSE 非劣 1% + batch1 延迟降 10%，需按任务
目标确认，非自动门槛）；比较强制 K1/强制 Kmax/validation-fixed policy/adaptive，报告
整个前沿；同 GPU/batch/缓存/混精、预热后足够重复，分记 batch1 延迟与批量吞吐、平均
与 P95 K、peak memory、控制器额外成本、端到端与纯 forward 两口径；per-sample active
subset 真执行；复杂天气分组来自 train 阈值，K 与涡度/梯度相关只是描述。
失败出口：无可重复的非劣精度/真实成本收益 → 结论 negative/inconclusive，停止加大
控制器复杂度；工程完成可关闭实验，但不得标"adaptive 核心创新已成功"。

### #67（P2）期刊评估协议与封存
指标/统计解析测试可现在做；正式 test 依赖 #60/#62/#63/#64/#65，adaptive 主张另依
赖 #66。无正向证据仍如实产出负结果，不为稿件改 test。协议：有限区域 0.25° 原网格
17 通道、history(t−6h,t) 预测未来；主表=同数据/同初始化/同预算的 persistence/
climatology/U-Net/window/AFNO-small/generic/process（+条件通过后的 adaptive）；外部
预训练 GraphCast/Pangu 只列单独 reference 表并披露全球上下文/训练年份/变量/初始化
差异，不同表声称公平超越。固定 6/12/24/48/72h free rollout，对齐 init/valid-time，
不注入未来 ERA5/未来过程标签/未披露 NWP 边界。从原始充分统计量生成逐变量 weighted
RMSE、pooled ACC 与相对 persistence/训练气候态的 MSE skill；注明本库 ACC 定义；正
ACC ≠ 正 MSE skill；禁止平均不同定义的 ACC 或直接平均各 batch RMSE。headline 变量/
时效/可接受退化/多重比较处理在查看 test 前写入 protocol；17 变量附录保留。不确定
性：同一 weather case 成对比较；相邻起报不是独立样本；≥3 个固定训练 seed 分别呈现；
按天/周 paired block 重采样（block 规则 validation 阶段预选）；seed 与天气采样不确
定性分开；四季/起报时刻/full-interior-boundary 分组；极端阈值只来自 train；无降水
目标不写降水 skills；第二区域是额外外推验证需新预算/协议，不在主实验失败后临时挑。
计算与创新证据：参数/counted ops/更新/样本/GPU-hours/单卡延迟吞吐/K 分布/peak
memory；full/retained-truncated/streamed 语义不同在方法中明确。交付：冻结配置签名
后一次性 test 报告、只读保存并登记访问；代码版本/环境锁定/数据 recipe+许可/
checkpoint-metrics 索引 digest/图表脚本；README 分栏小 smoke、negative study 与
publication benchmark。

### #68（P2，独立并行）下载层传输边界
现状：`reject_non_public_host` 先 getaddrinfo 校验，urllib transport 仍按原 hostname
连接——**校验地址未绑定到实际连接**（TOCTOU/DNS rebinding 窗口）；默认代理行为未入
威胁模型。静态告警减少不证明实际 socket 目的地已受限；这是防御性 follow-up，不是
已发生入侵；固定官方来源+本地 CLI 降低暴露面，应明确区分。
计划：先决定 allowlist（固定可信气象/代码主机）还是任意公网 URL；离线 mock/本地隔
离测试模拟"首次解析公网、连接时重解析非公网"，证明实际连接用已验证地址或在连接层
拒绝非公网 peer；每跳 scheme/host/port 检查，userinfo/空解析/IPv4/IPv6/重定向降级
行为明确；代理显式禁用或指定受信策略；HTTPS 保持原 hostname 的 SNI 与证书校验，
禁止 verify=False 或固定失效 IP；超时/redirect cap/字节上限/失败 receipt 保持；测试
不访问真实私网/云 metadata/外部目标。文档区分"预解析检查"与"连接地址约束"，只做前
者不得宣称 DNS rebinding 免疫。参考 OWASP SSRF Cheat Sheet 与 CPython http.client/
urllib.request 源码。

---

## 4. 工程政策（沿用 AGENTS.md「GitHub 通道」与决策 0003）

- ff 推送到 main 已放行（脚本级，即时生效）；**合并（git merge 涉及 main /
  gh pr merge）需用户明确授权后由用户执行**；删除默认分支、--mirror、force 拒绝。
- **不触发 17 条实验 workflow**（用户决定）；常规 push 的 ci.yml 自跑、不等待不作门槛。
- issue 评论 API 写 401：证据写 docs，关闭靠收尾提交的 `Closes #N` 经 main ff；
  评论正文缺口的说明保留在 `docs/R7_ISSUE_COMMENTS.md`。
- hook 是跨行文本匹配器：命令文本里 "git merge…" 与后文 " main" 共存会误判
  （`git merge-base` 与 "origin/main" 同理）——命令分条或用 SHA。
- 外部源码借用前记录 revision/tag、license、具体文件与改动表；官方上游只读，
  不自动开上游 Issue。
- 新文件命名：snake_case；禁词 final/old/new/tmp/temp/copy/backup/draft/deprecated/
  misc；杂物桶目录名禁用；治理文件 `git add`。

## 5. 每轮固定动作

1. `git rev-parse HEAD` 记录起点；2. 选依赖顺序下**最早 READY** 的一项；3. 实现→
测试→自查→修→再测；4. 全量 `pytest -q` + `check_conventions.py`（34 条 0 违规）；
5. commit（`type(r7): summary (#N)`，不携带实验标签）+ push 分支；6. ci.yml 自跑、
记录 run id 不等待不作门槛；7. 结果写入 `docs/R7_*.md` 与 `docs/R7_TASK_QUEUE.md`
绑定精确 SHA；8. 判定 DONE（Acceptance 满足）或 BLOCKED（写出真实依赖）；9. 阶段
完成后按既定策略 ff 更新 main（可选）；10. 下一轮。

## 6. 停止条件（本 goal 收尾）

满足任一即收尾并如实报告：(a) S0 三项（#60/#61/#62）DONE 且 #63 的 read-plan/
metadata 与离线测试完成——这是本 goal 的**核心交付**；(b) 预算到界（GPU-hours 或
数据上限）且剩余项全部 BLOCKED 写出真实依赖；(c) 出现必须用户决策的分叉（数据获取
授权、超预算扩量、合并）且已给出成本/范围选项。科研 gate（#64–#67 的正向结论）
**不是**本 goal 的完成条件——负结果与 BLOCKED 是合法终态。不得为推进而降低判据、
用 test 找正结果、静默扩量或把 CPU 修复说成 GPU 验证。

## 7. 环境事实（2026-09-25 核实；执行时复核）

- HEAD=main=分支=`6748b6fad5f9b0630fbeba246f7ff2088ced1047`；PR #12 merged；树干净
  （仅未跟踪 `.zcodeignore`，ZCode 生成的镜像文件，不入库）。
- 本地测试 895 passed/3 skipped；远程 CI run `36130059293` = 887/11/2 warnings
  （11=6 CUDA+2 本地真实区域+3 旧 fixture；两口径不可互换）。
- 2×RTX 3090 24GB、SYS/NUMA、无 active NVLink；torch 2.11.0+cu128、Python 3.12.3。
- `data/download/http_public.py` 已有 SSRF 防护（决策 0003 同期）；#68 在其上加
  连接层绑定。
- GPU 逐 cell 原始产物在本地 `outputs/`（未推送）；#62 的审计包从这里取。
- 依赖 `zarr/gcsfs/xarray/icechunk` 均已安装；真实 ARCO/Earthmover 源匿名可读
  （wb13-6h 比 full_37 快约 8 倍；ROI 放大几乎不增成本——chunk 为整张全球场）。
