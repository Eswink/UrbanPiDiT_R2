# 0009 — R7 主模型 V2 收尾：把 #71–#75 的实验做完，然后全部关闭

**状态：已批准（2026-10-02）。本文件是执行前的计划归档（R-032），不是执行记录。**
本次会话按用户要求**只产出文档与 goal 提示词**：计划文件、四个 goal 长文（含可粘贴 objective）、
两处索引。实验、提交、推送、GPU 均留待用户指定的执行窗口。

## 授权与前置（2026-10-02 用户指示）

- 用户明确授权：「可以将所有的 issue 实验全部做完然后全部关闭」「明确显示授权允许将实验授权下放给你，
  你可以自行计划实验内容，实验任务，以及下一项实验，若 GPU 时间不够，可以自行增加，只需要通知我即可」。
- 由此**一次性**解除两处「只提议不执行」的停顿：N1 的 `next_node_proposal=N2d`（停止线）与主计划 §2
  「不自动进入下一节点」。用户已否决停止线，选择把 M3/M4/M5 做完。
- 该授权覆盖**实验范围、顺序与下一项的自定**，以及**预算不足时的自行追加（通知制）**；它不豁免
  R-006/R-028、协议冻结（`protocol.json` + digest）、`scientific_claim:false`、false/negative 如实记录、
  以及主计划 §5 的单次实验 ≤30 min。
- 关闭通道：本项目只用 git/SSH（API 写 401）；`Closes #N` 必须在**默认分支 main** 的提交里才生效；
  main 是本分支严格祖先，**非 force ff 推送**由决策 0003 放行。**合并仍被 hook 拒绝且本计划不需要**。

## 起点事实（2026-10-02 只读核对）

| 项 | 值 | 来源 |
| --- | --- | --- |
| 工作分支 / HEAD | `r7/weather-reasoning` / `5fe2ff4` | `git rev-parse HEAD` |
| `origin/main` | `dafd22e`（HEAD..main=0，main..HEAD=84） | `git rev-list --count` |
| 开放 issue | **#70–#75 全 open**（#70 有 1 条评论，其余 0 条；无 closing keyword） | 匿名 GitHub API |
| 主计划节点 | `current_node=N1`、`status=paused`、`next_node_proposal=N2d` | `docs/goals/main-model-v2-campaign.md:143` |
| GPU 账本 | 已用 **4.2045** / 余 **19.7955** GPU-h（cap 24） | 同上 |
| M2 数据 | `outputs/r7_m2_segment/store`：train 186 / val 22 / test 26（**test 封存**） | manifests |
| M4 父 checkpoint | `outputs/r7_72_rw_b_subtraction/seed{41,42}/training/process_spacetime_rwa/update_0000400.pt` | `ls` |
| 近期 `docs(r7)` 提交 | 只写「close the round's bookkeeping」，**不是 issue 关闭** | `git log` |
| 工作区 | 一组**已 staged 未提交**的 6 文件改动（决策 0027 验收边界包，170 insertions），未跑新 CI | `git diff --cached --stat` |

代码现状：M3/M4 **完全未实现**；M5 的**前置 matched-Generic 缺失**（`GenericRecursiveWeatherForecaster`
挂不上 `local_solver_state`，`model/recursive_weather_r7.py:123-203`）；M1/RW-A/RW-B 的实现与实现证据都在。

## 阶段与预算

| 阶段 | 节点 / issue | 上限 | 内容 | goal 长文 |
| --- | --- | --- | --- | --- |
| A | N2a / #73 | ≤1.0 GPU-h | 尺度修复 + 三类时刻语义 + 三臂有界实验 | `docs/goals/n2a-m3-process-supervision.md` |
| B | N3 / #74 | ≤1.0 GPU-h | 2 步可微自回归 + matched-Generic V2 接线 | `docs/goals/n3-m4-autoregressive-rollout.md` |
| C | N4 / #75 | ≤2.5 GPU-h | ≥3 seed 三臂确认 + adaptive 决定 | `docs/goals/n4-m5-confirmation.md` |
| D | #71/#72 | 0 GPU-h | 完成判据缺口补齐（纯 CPU 测试/证据页） | `docs/goals/v2-issue-closeout.md` |
| E | #70 + 全部 | 0 GPU-h | 判定登记 + 解除禁令 + `Closes` ff 落 main | `docs/goals/v2-issue-closeout.md` |
| — | 应急 | ≤3.5 GPU-h | 失败重跑 / seed 补齐 / λ 敏感性 | — |
| **合计** | | **≤8.0 GPU-h** | 余量 19.7955 覆盖；超自设额度即通知用户 | — |

每阶段开工先跑 `tools/check_campaign_state.py` 对表（退出 0）；执行那一刻按决策 0021 记录具名范围
（写进 protocol，含范围/预算/产物与证据/失败与 skip 处理）。

## 每阶段交付物与门禁

- **A（M3）**：`data/preprocess/` 尺度 sidecar 构建器（物理单位 → train-only 标准化 → 逐通道相对下限；
  零方差通道标 `degenerate` 并 mask，**禁止 1/floor 放大**；版本化 metadata，不原地改 store/checkpoint）；
  `input_diagnostics` / `future_diagnostic_targets` / `draft_diagnostics` 三**分离字段**与同一 valid-time
  语义；定向测试（analytic 涡度/散度/平流、纬度 metric、零方差/NaN/shape 显式拒绝、poisoned future label
  只改 loss 不改 forward/halting、无 target 推理可执行）；三臂有界实验（aux off / 修尺度 input aux /
  修尺度 future+draft aux），出 protocol + 四张表 + paired comparison。
- **B（M4）**：可微 2 步物理 rollout（**不用** `model/r7_rollout.py:72` 的 `@torch.no_grad()` 评估路径），
  warm-start 自同一 +6h 父 checkpoint，`L6 + λ·L12`（λ=0.5 执行前冻结）；字段区分 `reasoning_steps` 与
  `rollout_steps`；**matched-Generic V2 接线**（复用既有组件，不新增 solver 部件）+ 参数差契约测试；
  验收测试（第二步输入 hash = 第一步预测、改真实第二步 target 只改 loss、每步正确更新时间、内部 K 不改
  时间、缺 t+12 窗口显式拒绝、梯度从第二步回到第一步、resume/finite/BF16）；两臂有界实验并**如实记录
  算力不相等**。
- **C（M5）**：三臂 = 旧 Ours / matched Generic V2 / Process V2，**≥3 预声明 seed**；primary 端点运行前
  冻结（建议 t2m +6/+12h 并写明容许退化），全 17 变量 × 6/12/24/48/72h 照报；单位正确 RMSE /
  climatology skill / ACC；K1/2/4 同 checkpoint 探针标「非独立训练」；四张成本表；三 seed **只作一致性、
  不作显著性**；adaptive 默认**不启动**并如实记录（除非出现固定 K 有效前沿）。
- **D（#71/#72 证据）**：补 `tests/` 反证——`test_the_output_follows_the_declared_fields_and_only_those`
  仅断言输出随字段变、离 direct 的隐含断言 0.05 单位（`tests/test_r7_spacetime_inputs.py:114-141`）；
  补**末端块（闰年/年末/跨日/negative-longitude）**、odd grid+padding、**内部 K 保持同一 valid_time** 的
  直接断言；#72 的 bean 补**位置化反馈到达相应 latent** 与 SWA `process_reader` 非零梯度。
- **E（关闭）**：判定登记进 `docs/R7_ISSUE_COMMENTS.md` + `docs/R7_TASK_QUEUE.md`；解除主计划 §2
  （`main-model-v2-campaign.md:71`）与各长文的关闭禁令并新立决策；工作分支写 `Closes #70`…`Closes #75`
  的默认分支提交，**非 force ff 推送 main**；确认六 issue 变 closed；停 `docs/goals/open-issue-resolution.md`
  为 complete。

## 验证（每阶段收尾统一）

`tools/check_conventions.py`（37 条 0 违规）、`tools/check_campaign_state.py`（退出 0）、
`tools/check_goal_brief.py`、`pytest -q`（含新测试与反证）、`git show --check` + 空白检查；推送工作分支后
**等 `ci.yml` 绿**（`R7 CPU CI` 九步），实验轮按需带 `[tag]` 触发对应 `r7-*.yml`（skipped 是设计行为）。

## 风险与明确不做

- **风险**：M3/M4/M5 可能负向——各 issue 完成条明文允许 negative/mixed（#72/#73/#74 结构正确即可验收）；
  #70 EPIC 若拿不出「实际改进」则按 negative/mixed 如实写；matched-Generic 需改公共模型类 → 产生新
  `model_code_sha256`，按 `[model-digest-change]` 显式处理，不绕过旧 checkpoint digest。
- **不做**：不改已冻结判据/阈值/端点/案例集；不重跑或覆盖 `outputs/` 归档；不读封存 test；不下载数据、
  不租 GPU；不 force push、不 `--mirror`、不删默认分支；不建立定时任务或后台续跑；不把三 seed 写成显著性。
- **权限边界**：可在授权内跑实验、登记、提交、ff 推 main 关闭 issue；**合并（`git merge` / PR merge）仍
  不能在会话内执行**——本计划**不需要**合并。

## 交接注意事项（执行窗口必读，否则 `check_campaign_state.py` 会硬漂移）

四份 goal 长文已按主计划的机器契约写好（`<!-- round-node: X -->` + 文末「下一动作」点名下一节点），
执行时按下列顺序推进，**每一步都要重跑对表**：

0. **先处理已 staged 的上一轮改动**：工作区里已有一组**已 `git add`、未提交**的 6 文件改动
   （决策 `docs/decisions/0027-n1-cost-acceptance-boundaries.md` + `docs/decisions/README.md` +
   `docs/goals/main-model-v2-campaign.md` 的**进度尾** + `docs/goals/n1-cost-supplement-repair.md` §7 +
   `docs/rules/CHANGELOG.md` + `docs/rules/EVIDENCE.md`，共 170 insertions）。它属上一轮、**未跑新 CI**；
   应**先**把它做成一个独立提交并绑 CI，再开始阶段 A，避免与后续改动混在一起（该组不改 §8 状态块、不改
   §2 禁令，仅追加进度尾；`git diff --cached` 可复核）。
1. **推 N2a 时**：主计划 §8 改为 `current_node=N2a` / `previous_node=N1` /
   `current_round_goal=docs/goals/n2a-m3-process-supervision.md`；**并且在 `n1-cost-supplement-repair.md`
   文末「下一动作」条补一句点名 N2a**（C-04 读的是上一轮文末最后一条「下一动作」是否含新节点名；不补
   会 CI 红）。同一提交里登记 `previous_round_evidence`（N1 的证据页）。
2. **推 N3 / N4 / N5 时**：同理改 §8 三键，并保证**上一轮**尾部点名新节点——`n2a` 尾已含 `N3`、
   `n3` 尾已含 `N4`、`n4` 尾已含 `N5`（本轮已预置）。
3. **每轮的证据页**（`docs/R7_73_*` / `docs/R7_74_*` / `docs/R7_75_*` / 关闭轮登记）必须先进
   `docs/R7_EVIDENCE_INDEX.jsonl` 且有可达 `evidence_commit`，否则 C-06 硬失败。
4. **账本行**：每行要有 `docs/...md` 指针；能引 `record:<id>` 的就引（C-03 会比对 `gpu_hours`，容差
   1e-4），没有索引支撑的行会被如实列为 note，不要伪装成机器核过。
5. 新 goal 长文的 objective 均为**单行**且自指本文件，符合 `tools/check_goal_brief.py`（G-01…G-08）。

## 实际结果

- **完成情况**：**未执行**。本次会话按用户要求只交付文档：本计划文件、四个 goal 长文
  （`n2a-m3-process-supervision.md` / `n3-m4-autoregressive-rollout.md` / `n4-m5-confirmation.md` /
  `v2-issue-closeout.md`）与其可粘贴 objective，并更新 `docs/plans/README.md`、`docs/goals/README.md`
  索引。未跑实验、未改代码、未提交、未推送、未碰 GPU；**未自宣任何阶段完成**。
- **与计划的差异**：计划原本设想在同一会话直接执行（阶段 A–E）；用户改为「只出文件、另开窗口执行」，
  故执行部分整体后移，本文件转为交接件。计划内容未因该决定改变。
- **验证**：`tools/check_goal_brief.py --brief docs/goals --quiet` 对新四条报 0 失败（objective 单段、
  ≤4000 字符、自指本文件）；`git show --check` 与空白检查通过。CI 未触发（仅推送时运行）。
- **遗留**：四份 goal 长文尚未进 `docs/R7_EVIDENCE_INDEX.jsonl`、尚未 commit；执行窗口需按其 §4 顺序
  做对表、实现、实验、登记与关闭。