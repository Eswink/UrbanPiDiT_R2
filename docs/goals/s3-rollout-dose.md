# S3：rollout 监督剂量响应 200 → 800（单因素）

<!-- round-node: S3 -->

状态 **已完成**（2026-10-09），登记为 mixed / needs-review。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；承接 `docs/goals/s3-climatology-anomaly-anchor.md`
"本实例结束、下一假设另行预声明独立新协议"的下一动作，不重跑或扩展已停止的 anchor/ordering 实例，
不进入 S4，不自行宣布最终 goal 完成。

## §0 Objective（单段，实测 1183 字符）

> 在 docs/goals/s3-rollout-dose.md 接续S3/index63/23.3140GPU-h、2023test未评分r0。唯一问题：注册的S3 long-rollout screen在200更新时余弦日程已退到0.1×下限，因此只界定一个剂量点；把更新数改成800、父/12步权重/LR2e-5/warmup10/FP32/K4/batch1/clip1/控制/判定形式全不变，测该杠杆是否饱和。父checkpoint由修订66836d2训练、其model_code_sha256 3ddab46b与当前d3fb58db不同，受审计的load_checkpoint按设计拒绝；不改守卫，做可审计model-only迁移：在归档修订下用其自身loader导出state，再在新代码下复核源SHA pin/导出state digest/张量key-shape-dtype/模块语义digest并写新r7-local-v1+归档provenance，不带optimizer/cursor/RNG，并以固定合成batch在两修订下逐位相同的FP32 forward为惰性证据。三seed 41/42/43、v3实例train2017-2021/val2022、原cohort472/468/460/444/428、控制按SHA复用注册v3-D3 400更新读数不重训、test不读。冻结函数只认primary（t2m/full 6h与12h三seed对控制RMSE增量同号全负）、守门（u10/v10/mslp五lead三seed相对MSE变化≤0）、advance决定，不做同时区间、不宣称收敛或S4就绪。整轮soft9000/hard18000/perseed4200先冻，soft超继续记overrun、hard或真错停attempt并全额留失败；probe另冻1800/3600。独立子代理用自有脚本从原始产物复算全部关键数字，不得引用生产者工具。全部失败、负面与偏差如实登记；不paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置，不自行宣布最终goal完成。

## §1 起点、证据与实质差异

- 起点工作分支 `r7/weather-reasoning`，HEAD `99e94a3`（其前 index63 已登记、累计 23.3140）。
- 注册的 200 更新 screen（`docs/R7_S3_LONG_ROLLOUT_SCREEN.md`）primary supported、控制守门 0/45、
  决定 advance-to-S4-freeze，但 t2m 48/72h 气候态 skill 全负。其实测 LR 日程在 update 200 已达
  0.1× 下限，故该轮只界定一个剂量。
- 与已停止实例的实质差异：**只改更新数**（200→800），父、12 步物理权重、LR、warmup、精度、
  K、batch、clip、控制与判定形式全部不变；不复用 anchor/ordering 的任何失败链。
- 不重跑已否定假设；不无限加剂量——本轮目的正是判断该杠杆是否仍在付。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 单因素剂量协议与新排他 outputs | `outputs/r7_s3_rollout_dose_20261009_attempt02/protocol.json`（canonical `655fec89…`） |
| D2 | model-only 迁移的可审计工具与回执 | `scripts/export_r7_parent_state.py`、`scripts/migrate_r7_parent_state.py`、迁移 receipt `23daf794…`、两修订 forward 逐位相同 |
| D3 | 三 seed 800 更新与 val 全 cohort 评分 | `seed{41,42,43}_receipt.json`、每 lead CSV、provenance（`split=val`、472/468/460/444/428） |
| D4 | 冻结形式判读与只读复算 | `outputs/r7_s3_rollout_dose_readings_20261009_attempt01/readings.json`、`docs/R7_S3_ROLLOUT_DOSE.md` |
| D5 | 独立复核、失败保留与成本登记 | 独立代理 14/14 CONFIRMED、`failure.json`、index64/brief/ledger、精确 CI |

## §3 判据与开发读法

判据只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与注册 screen 的冻结决定文本；本轮只做
**开发筛选点估计**：primary = t2m/full 6h/12h 每 seed 对注册 v3-D3 控制的配对 RMSE 增量符号；
守门 = u10/v10/mslp 五 lead/三 seed 相对 MSE 变化 ≤ 0.0，容忍 0；两者合取才 advance。
不做同时区间、不读 test、不把三 seed 当显著性。

## §4 数字预算与停止

planned 9000 / hard 18000 / per-seed 4200 s（probe 1800/3600）；整轮含准备/归档/三 seed/判读/清理。
soft 超继续并记 overrun，hard 或真错停该 attempt 并全额留失败；不运行中改硬限、不重试至偶然通过。
GPU0 默认共驻，启动/spawn 前只读余量门，不 signal 邻居。

## §5 明确不做

不重训控制、不重跑 anchor/ordering、不加 seed/剂量以外的因素、不做完整 val/全年/同时区间、不读 test、
不改共享判定代码（只读复算工具另立并记录偏离）、不改科学合同与冻结证据、不 main 合并/发版/关 issue。

## §6 进度与交接

- 已完成：probe attempt01（父 digest 未迁移，失败保留）→ model-only 迁移 → probe attempt02 成功
  → 三 seed 800 更新 screen（五阶段成功）→ 冻结 reading 阶段失败（共享收集器 parent-pin 缺陷）
  → 只读复算得 primary supported / 守门 0/45 / advance-to-S4-freeze → 独立复核 14/14。
- 成本：0.4808 + 0.4706 + 3.3251 = **4.2765 GPU-h**，累计 **27.5905**（cap20/remaining −7.5905 仅会计）。
- 下一动作（S3）：修复共享 `_references` 对 model-only 迁移父的表达（或把迁移注册为可 pin 的父记录），
  然后以"提高 pattern 相关"为下一杠杆；最可证伪的一支是扩大区域/引入域外大尺度上下文，并重建
  同数据气候态与 incumbent 后再比较。test 未读、r=0、S4 未启动不变。
