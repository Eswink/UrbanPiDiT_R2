# n2a-m3-process-supervision：M3（#73）过程监督的尺度修复与「三类时刻」语义

<!-- round-node: N2a -->

**状态：active（2026-10-02）。用户本轮具名触发仅 M3/N2a；实现与 CPU 验证正在推进，GPU 尚未启动。
本轮范围、≤1.0 GPU-h 与禁止项以 §0 当前 objective 为准，不沿用计划的总括授权自动进入 N3/N4 或关闭 issue。
执行那一刻按决策 0021 记录具名范围，先跑 `tools/check_campaign_state.py` 对表，再按 §4 执行。**

本文件是主计划 `docs/goals/main-model-v2-campaign.md` 的节点 **N2a** 的目标长文。节点定义、账本、
对表清单都在主计划里；本文件只写「这一轮怎么做」。开工第一件事是按主计划 §3 对表（退出必须为 0），
再按 §4 执行。实现顺序 A→B→C（见 `docs/plans/0009-r7-v2-completion-and-closeout.md`）。

## §0 Objective（可粘贴；实测 1380 字符）

> 本轮目标：执行 docs/goals/n2a-m3-process-supervision.md 的 M3 轮——把 #73「让 Process State 服务于预报」需要的尺度修复与三类时刻语义做实，再跑一轮有界三臂对照。0 GPU-h 部分：P-A 尺度修复（按 proxy 的物理单位在 train split 上定义有量纲缩放，再在无量纲值上做 train-only 标准化与逐通道相对下限；真实零方差通道标 degenerate 并显式 mask，禁止用 1/floor 放大噪声；参数、有效 mask、原 std/floor、缩放后 std 写成版本化 metadata，产出 outputs/ 下的派生 sidecar，其 identity 计入新训练 contract，不原地修改旧 store 或 checkpoint）；P-B 三类时刻语义（在 training/r7_process_forecast_losses.py 落实 input_diagnostics / future_diagnostic_targets（仅训练监督，来自训练未来真实标签）/ draft_diagnostics（由模型自己的草稿 Y_k 经固定 train 反归一化与地理 metric 计算，推理时可得）三个分离字段名；跨内部 K 是同一 valid-time 重估、不加 lead；推理时禁止读取真实 future 诊断或真实 error；旧 loss 开关与新任务开关分离）；P-C 定向测试（analytic 涡度/散度/平流与纬度 metric；zero variance/NaN/shape/名称不匹配显式拒绝；开启辅助任务时禁止 min(width) 静默截断；poisoned future label 只改 loss、不改 forward 输出或 halting；无 target 的推理 batch 可完整执行）。GPU 部分：tri-arm 有界对照（aux off / 修正尺度的 input aux / 修正尺度的 future+draft aux），固定 #71 后的 data/feature/主干/训练协议与 Ktrain=4，每臂 2 个预固定 seed、400 updates，只改监督因素，报全 17 变量 × 6/12/24/48/72h 并保留全部坏变量；结果可 negative/mixed，结构正确即按 #73 验收。判据见 docs/goals/n2a-m3-process-supervision.md §3，冻结来源 docs/R7_MAIN_MODEL_V2_DESIGN.md §6 与 docs/R7_65_PREDIAGNOSTIC.md。禁止：改已冻结判据/阈值/端点/案例集、重跑或覆盖 outputs/ 归档、读封存 test、下载数据、租 GPU、动 main、force push、合并、关闭 #70–#75（关闭是收尾轮 v2-issue-closeout 的事）、自动推进节点。预算 ≤1.0 GPU-h（主计划 §5 第二批 ≤24 GPU-h 之内，开工时重读账本）；停止条件为主计划 §5 任一条、两卡余量始终不足、需要新判据或新数据、或任一评估失败。不要自行宣布目标完成。

## §1 现状（2026-10-02 只读核对）

- 起点：主计划 §8 `current_node=N1`、`status=paused`、`next_node_proposal=N2d`；账本已用 4.2045 /
  余 19.7955 GPU-h。本轮由用户授权**一次性**进入 N2a（不再经过 N1 审阅放行）。
- 数据：`outputs/r7_m2_segment/store/manifests`：train 186 / val 22 / test 26（**test 封存，本轮不读**）。
- **尺度缺陷（已记录）**：`DEFAULT_EPS=1e-6` 的 floor 命中 8 个 proxy 中的 **2 个**——
  `moisture_advection_850_mean` 原始 std ≈ 9.1e-9、`moisture_convergence_850_mean` ≈ 1.7e-8，
  归一化后 std 只有 0.0076 / 0.0118（`docs/R7_65_PREDIAGNOSTIC.md`）。它影响**监督数值权重**，
  不是让输入变成常量，也未证明是所有退化的原因。
- **旧语义缺陷（已记录）**：旧 loss 把**同一 input-time process target** 广播到每一个内部 K
  （`training/r7_process_forecast_losses.py:34-50`，`process_weight` 默认 0.1 在 `:23`）；P 标量读出
  并不直接进入 forecast solver。
- **已有可复用件**：尺度审计 `training/r7_process_diagnostic.py:51-131`（`DEFAULT_EPS` `:45`、
  `UNUSABLE_NORMALIZED_STD=0.05` `:48`）；诊断公式 `data/preprocess/process_diagnostics.py:56-135`；
  store 写入 `data/preprocess/r7_era5_zarr.py:203`（`process_normalization_std`）；
  只读驱动 `scripts/diagnose_r7_process_reasoning.py:96`。
- **尚不存在（grep 已确认）**：三个字段名 `input_diagnostics` / `future_diagnostic_targets` /
  `draft_diagnostics`、`degenerate` mask、`sidecar`、`scale_metadata` 在 `model/ training/ data/ scripts/`
  里都没有（唯一出现是 `tests/test_r7_shared_step_paths.py:168` 的 poison 输入）。
- **C1 负结果的边界**：提高旧任务权重得到负面结果，**不能**据此否定所有 process-space 学习；本轮不是
  重扫 C1 权重，而是改变过程辅助任务与 forecast 任务的**关联方式**。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | P-A 尺度 sidecar 构建器（0 GPU-h） | `data/preprocess/` 下的构建器 + 版本化 metadata（物理单位、原 std/floor、缩放后 std、`degenerate` mask）；输出 `outputs/r7_m3_scale_sidecar/`；**不原地改旧 store/checkpoint**；构建器的只读 preflight（R-028 风格）与 `--write` 授权 |
| D2 | P-A 尺度审计扩展（0 GPU-h） | `training/r7_process_diagnostic.py` 输出 `degenerate` mask；对 8 个 proxy 逐一给出旧尺度 vs 新尺度对照（**只比尺度，不把所有后续改动归因到修尺度**） |
| D3 | P-B 三类时刻语义（0 GPU-h） | `training/r7_process_forecast_losses.py` 的 `input_diagnostics` / `future_diagnostic_targets` / `draft_diagnostics` 分离字段与 timestamp 契约；`draft_diagnostics` 由模型自己的 `Y_k` + 固定反归一化 + 地理 metric 生成；旧开关与新开关分离 |
| D4 | P-C 定向测试（0 GPU-h） | `tests/test_r7_process_supervision_moments.py`（§3 清单，含反证） |
| D5 | tri-arm 有界实验（GPU ≤1.0 GPU-h） | `outputs/r7_73_process_supervision/`：protocol（先冻结 digest）、逐 seed 结果、合并结果、paired comparison、四张表；`scientific_claim:false` + `limitations` |
| D6 | 证据页与登记 | `docs/R7_73_PROCESS_SUPERVISION.md` + E 条目 + `docs/R7_EVIDENCE_INDEX.jsonl` 记录（`outcome_class` 按结果）+ 账本行 + 主计划 §7/§8 回写 + CI 绑定 |

## §3 判据与预声明读法

判据全部引自冻结文档，**不新增阈值、不放宽任何既有判据**：

- **尺度修复**（`docs/R7_MAIN_MODEL_V2_DESIGN.md:145-151`）：先修尺度再解释；用 train split 物理单位
  缩放 + train-only 标准化 + 逐通道相对下限；零方差通道标 `degenerate` 并 mask；**禁止**用 1/floor
  放大噪声；metadata 版本化，不改旧 store/checkpoint identity，派生 sidecar 的 identity 入新 contract。
- **三类时刻**（同文件 `:125-152`）：`input_diagnostics` = 初始时刻最后历史状态（训练与推理都可得，辅助
  监督）；`future_diagnostic_targets` = 训练未来真实标签（**仅训练**，监督目标）；`draft_diagnostics` =
  模型自己的草稿经固定反归一化与地理 metric（推理可得）。**两个来源不同名**；跨 K 是同一 valid-time
  重估、**不加 lead**；推理时禁止读真实 future 诊断或真实 error；advection 符号不得硬编码成温度趋势的
  因果规则。
- **验收**（issue #73 正文，匿名 API 已核）：实际分割 train 拟合全部统计量；改 val/test 不改变 label
  归一化；物理单位变换后还原一致；analytic 涡度/散度/平流 + 纬度 metric + 边界处理；zero variance/NaN/
  shape/名称不匹配**显式拒绝**；开启辅助任务时**禁止 `min(width)` 静默截断**；poisoned future label
  只影响 loss；无 target 的推理 batch 可完整执行；future 时刻标签与 draft 对应，多 K 不得错加 lead；
  FP32/BF16 敏感差分在 FP32 计算；各分支梯度所有权/范数记录；工程与**一个有界真实 train/val 实验**完成，
  所有变量结果保留（不为了 aux 看起来有效而删坏变量）。
- **MetPy 只作离线 oracle**（`docs/R7_MAIN_MODEL_V2_DESIGN.md:152`）：前向用可微 torch tensor，
  不在每轮 GPU 推理里经 CPU/pint 往返。
- **三态读法**（预声明）：若修尺度后辅助信号正常但 forecast 无改善 → 如实写 mixed/negative，不把修尺度
  包装成方法成功；若信号仍不正常 → 先修尺度再判，不得拿未修尺度的 aux 结论做推论（plan 0004:313）。

## §4 实施顺序（不跳步）

1. **对表并推进节点**：把主计划 §8 状态块推进为 `current_node=N2a` / `previous_node=N1` /
   `current_round_goal=docs/goals/n2a-m3-process-supervision.md`，并在**上一轮长文**
   `docs/goals/n1-cost-supplement-repair.md` **文末的「下一动作」条**补一句指向 N2a（含用户 2026-10-02 的
   授权事实）——C-04 会检查上一轮文末最后一条「下一动作」是否点名新节点，不做这一步
   `tools/check_campaign_state.py` 会硬漂移。另：工作区可能已有一组**已 staged 未提交**的上一轮改动
   （决策 0027 验收边界包，6 文件），先把它作为独立提交处理再动本轮。随后重跑对表（退出 0）、重读账本、
   核 `docs/R7_65_PREDIAGNOSTIC.md` 与旧 loss 指针未变。
2. **D1/D2（0 GPU-h）**：尺度 sidecar 构建器 + 审计扩展；只读 preflight 报告后按需 `--write` 授权。
3. **D3（0 GPU-h）**：三类时刻字段与旧开关分离；旧 checkpoint/模型 digest 不覆盖。
4. **D4（0 GPU-h）**：定向测试全绿（含反证）；工程 CI 绿。
5. **D5（GPU）**：执行那一刻按决策 0021 记录具名范围（范围 = 3 臂 × 2 seed × 400 updates；预算 ≤1.0
   GPU-h；共驻余量门槛；失败即停全额计费不重试）；先冻结 `protocol.json` digest 再训练；只读 val。
6. **D6 登记**：证据页、E 条目、索引记录、账本行、主计划 §7/§8 回写、CI 绑定；**只提议下一节点**。

## §5 预算与停止条件

- 预算 **≤1.0 GPU-h**（主计划 §5 第二批 ≤24 GPU-h 之内；开工时重读账本，当时余 19.7955）；单次实验
  ≤30 min（共驻余量门槛）。
- 配额记录：本轮 + 后续 B/C 与应急的**合计**目标 ≤8.0 GPU-h（见 plan 0009）；若将超出自设额度，
  **先通知用户再继续**。
- **停止条件**（满足任一即停并向用户报告，不自行扩大范围）：预算用尽或账本不足；需要新判据或新数据；
  结论为「不可分辨/不能归因」；需要租 GPU/合并 main/破坏性操作；任一评估失败（即停、全额计费、不重试）。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不改已冻结判据、阈值、端点或案例集；不为了让 aux 看起来有效而删坏变量或弱化测试。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 一律只读）；不原地改旧 store/checkpoint。
- 不读封存 test（M2 test 已被看过，仅作开发材料，不新增 read_count）；不下载数据；不租 GPU。
- 不动 main；不 force push；不合并；**本轮不关闭任何 issue**（关闭是收尾轮 `v2-issue-closeout` 的事）。
- 不新增 solver 部件；不把本轮 12/24h 的小改善写成机制声明；不建立定时任务或后台续跑。

## §7 进度块

- **状态**：`active`（2026-10-02）：用户具名触发本文件 §0；本轮只推进 N2a，不关闭 issue、不自动进入 N3。
- **开工 recheck**：起点 `5fe2ff4ffba76f791296069bf9c72a216b0dbbe0`；campaign 0 failures / 4 历史 ledger notes。
  账本已用 4.2045 / 余 19.7955 GPU-h，四条旧账本没有索引支撑如实保留；本轮 cap ≤1.0 GPU-h。
  上一轮登记 commit 可达，冻结 design/prediagnostic 无改动；既有 rules diff 仅 ADR0027/E-222 的已记录裁定。
  上一轮 staged 六文件包独立提交 `e7755ae`，相关 CPU 228 passed / 0.93s；首次测试命令误写不存在的路径、未执行，已修正实跑。
  train/val manifests 只读核到 186/22；未读取封存 test 清单或数组，两个新输出目标均不存在。
  本轮用户预先书面授权三臂×两预固定 seed×400 updates；实际启动前另冻结具名协议/预算/失败规则回执。
- **规划与校验路线**：planner 草案收到后已人工收敛：去除新增接受阈值、workflow、自宣 complete 和无来源时间估计；
  `/tmp/r7_m3_planner_verified_plan.json` 校验 `verified=true`，抽查旧 loss min-width 与球面公式代码指针。
  实施不改 model 主干，仅训练监督、sidecar 契约与测试；默认新权重全零，保留 legacy 默认 0.1。
  当前 provider 最近 goal 完成校验为 `error`（600002ms），仅作客户端路线诊断，不作为目标失败或成功判据。
  Mimosa commit/push hook 报 `scanner_enobufs`，没有完整安全结论；不声称项目安全通过。
- **D1/D2（0 GPU-h）**：真实只读 preflight `189c33e97e24384081adf2c6aba2e293f937e669627fb9b237feaff705fe07f3`，
  实际 train 186 窗口/188 帧；8 proxy 全 active。水汽两通道 rawstd 9.9155e-9/1.4293e-8，旧 normalized std
  0.0099155/0.0142934，新 std≈1。按本轮已书面授权的派生发布范围，新输出 `outputs/r7_m3_scale_sidecar/`
  排他发布；sidecar identity `4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d`，
  metadata SHA256 `d7c2837ea0083b5ce98f98cd02da729c1f12147387838dccad420393b7295cec`，
  completion SHA256 `a553781a5b68c3e19bb4f8921e819c856c3d7b4062cead535d637dbfe77047b9`。
  publisher 复用 fresh_outputs/BUILD_COMPLETE，不改旧 store；mask、不除 floor、单位变换与 train-only 反证已实跑。
- **当前 CPU 实跑**：尺度+streamed新任务+sidecar contract 111 passed/32.39s；旧路径相关 82 passed/17.49s，
  R-051/R-052及全部37阻断0违规。初期新测试2失败是独立浮点oracle误要求逐位相等与把FP32不可表示尺度误当可运行；
  改为元数据自带幅度精确floor核验与显式underflow拒绝反证，算法与冻结判据未改。streamed测试5失败是读出头
  Sequential 误访问 `.weight`，改取实际 linear `[1].weight`；全参数梯度比较保持且复跑通过。
  GPU 尚未启动；真实发布及 CPU 工程通过不代表预报改善或目标完成。
- **D3/D4 集成与独立复核**：三类字段/显式 UTC ns 时间、训练 future-only 和可微 own-draft 诊断已落地；
  新 aux 贯穿 scheduled→update_group→streamed，各 K 对同一 valid time；full/truncated/streamed 梯度定向一致。
  补强后集合168 passed/9.43s，含 analytic/numpy/FP32-BF16/零场有限梯度、poison forward+halting、无 target 推理。
  独立只读审阅指出初版 fixed atmos inverse 未实际入签名，现已修：sidecar 与实际 train data/store 强核，固定
  FP32 mean/std/有序 channel_names 的 snapshot+SHA 入新 signature；新 checkpoint 评估校验真实 held-out root。
  静态复核确认原三项身份缺口封闭；另补纯内存实际 helper/root 反证，不以 mock-only 绿代替覆盖。
  model_code_sha256 `11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476` 未改，旧模型不覆盖。
- **授权边界**：计划 0009 曾覆盖后续阶段，但本轮 §0 用户指示明确只执行 M3/N2a，并禁止自动推进、
  关闭 issue 与动 main；执行仅使用本轮具名书面授权，不把旧计划的总括授权作为扩围许可。
- **运行器前置核验**：三臂驱动完成 51 项 CPU/fake 定向反证（6.11s）；独立审阅推动修复原异常被
  cleanup/receipt 覆盖、failed-unreaped 状态与 fresh CUDA allocator 初始化。另发现 secondary stderr
  写失败仍可遮原异常，已加受限保护与反证；GPU 尚未执行。完整 CPU 首轮 2219 passed / 1 failed /
  9 skipped（213.03s），唯一失败为规模标记 R-020 的 45/46 漂移；同步实测标记，不弱化检查或阈值。
- **最终本地工程验证（GPU前）**：完整CPU 2272 passed / 9 skipped / 2 warnings（216.37s）；六项
  CUDA测试与三项可选真实fixture跳过，不计通过。运行器最后两窄修后52 passed/5.93s；治理反证348
  passed/23.56s；37阻断、goal、campaign、index、compile与whitespace均exit0（campaign四历史notes）。
  全套上一轮失败分别是规模标记与 MIGRATION 的新增基线引用未同步，均修文档、保留原检查。显式CUDA
  init后初次CPU fake缺新增mock导致1fail/51pass，补同一模拟路径后52pass；未用GPU绕开CPU门禁。
  入跟踪后R-016查到training直接socket import，移至scripts/r7_m3_offline.py，不豁免禁网或检查器。
  R-009现在1176/2975（新增111函数/284断言）；尺寸机器标记0/245/46/41/28/26，硬上限/例外不变。
  实际train尺度审计 `outputs/r7_m3_acceptance/actual_train_scale_audit.json` SHA256
  `6588b7bc9b36b2b487a2baeb17bc01f3910af7ad853a3ba5c63dcff4a1ee34a0`，8通道全报告，不改旧store。
- **下一动作**：绑定本轮精确工程 SHA CI；随后 CPU prepare 冻结协议与代码，再按已有具名
  书面授权执行一次 D5，失败即停且不重试。

## §8 下一动作（仅提议）

M3 完成后**只提议**进入 N3（M4 两步可微自回归，`docs/goals/n3-m4-autoregressive-rollout.md`），
不自行宣告 M3 成功或自动跳步；若 M3 结论为 negative/mixed，按 #73 完成条如实登记并继续（#73 不阻塞
M4 的 loss 接口）。