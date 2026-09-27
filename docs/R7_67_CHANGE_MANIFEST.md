# #67 交付：改动面逐文件清单

本文件列出 2026-09-27 战役（S3–S5）实际改动与新增的文件，按用途分组。
每个条目给出**为什么存在**，而不是逐行复述 diff。

代码 SHA 范围：`f4a552b` → 本轮末次提交。

---

## 1. 诊断与消融代码

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `training/r7_process_diagnostic.py` | 新增 | #65 预诊断的三项测量：proxy 尺度审计、train 模式梯度冲突（按参数分组）、K0..K4 轨迹（复用 #53 的 `correction_terms`，不另写第二套几何） |
| `scripts/diagnose_r7_process_reasoning.py` | 新增 | 在真实 store + 冻结 checkpoint 上运行上述诊断；诊断前后各校验 checkpoint 摘要，使「只读」是被检验的而非被声称的 |
| `scripts/study_r7_65_ablation.py` | 新增 | C1/C2/C3 三条消融轴的预注册 harness；每臂声明自己的训练深度，训练后断言 runner 契约一致（fail closed） |
| `scripts/analyze_r7_65_ablation.py` | 新增 | 把消融结果喂给 #60 比较器读回配对结果；重新推导协议 digest 并与结果比对，不一致即拒绝分析 |
| `scripts/study_r7_64_curriculum.py` | 新增 | 1→2→4 预测时效 curriculum（有界、与内部深度 K 不混淆） |

## 2. #67 封存与统计

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `configs/r7_era5_b2_recut_test_january.yaml` | 新增 | January-only held-out 重切配置；train 块与冻结链逐字节相同 |
| `scripts/freeze_r7_67_test_report.py` | 新增 | 一次性只读 test 报告驱动：发现每臂的 selected checkpoint、校验单一 `data_identity`、对每个 (臂, seed, 时效) 在 test 上评估并写三张汇总表 |
| `scripts/analyze_r7_67_paired_blocks.py` | 新增 | 按天 paired block 重采样（天为抽样单位）+ full/interior/boundary 分层的读取函数 |
| `scripts/analyze_r7_67_sealed_report.py` | 新增 | 把上述两项分析施加到封存报告的**全部** 15 个 arm-seed 单元上（60 paired + 75 region 单元格），只读报告自身文件 |
| `tools/prune_r7_run_checkpoints.py` | 新增 | 证据保全式剪枝：只删非 `selected_checkpoint` 的中间 checkpoint，保留全部评估产物与 endpoint，写入 path/bytes/sha256 receipt |

## 3. #68 传输层

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `data/download/http_pinned.py` | 新增 | 把 socket 连接到**已被校验的 sockaddr**（check 与 connect 共享一次解析）；逐跳校验 scheme/host/port 与解析结果；代理策略显式化；保留 SNI 与证书校验 |
| `data/download/http_public.py` | 改写为再导出 | 保留历史导入路径，使既有下载器无需改动即获得更强行为；消除第二套较弱实现 |
| `docs/R7_SECURITY_SCAN_TRIAGE.md` | 更新 | 把「预解析检查」与「连接地址约束」分开陈述，替换原先的「rebound 窗口风险可接受」表述 |

## 4. 测试

| 文件 | 用例数 | 覆盖的性质 |
| --- | --- | --- |
| `tests/test_r7_process_prediagnostic.py` | 18 | proxy 尺度含反证（常量通道不得读作可用；floor vs 正确 eps 对比）；零权重下 process 梯度为 undefined 而非伪造 0；未知参数组 fail closed |
| `tests/test_r7_process_arm_equivalence.py` | 6 | B2/B3 的「共享结构对」在功能上只差一个投影；含「未对齐时两臂确实不同」的反证 |
| `tests/test_r7_65_ablation_harness.py` | 9 | 每臂声明并使用自己的训练深度；**反证：浅层臂的 forward FLOPs 必须严格更低**（静默 K=4 的原始症状） |
| `tests/test_http_pinned.py` | 37 | 核心反证：首次解析公网、连接时重解析非公网——预解析检查会通过，pinned 连接拒绝且**从不创建 socket**；另有混合公私解析、IPv6 环回、zone id、逐跳校验、代理不继承、无 insecure-TLS 开关的反证 |
| `tests/test_prune_r7_run_checkpoints.py` | 9 | 端点缺失/报告缺字段/无报告/未 `--apply` 四类拒绝；非 checkpoint 产物在 `--apply` 后必须存活 |
| `tests/test_r7_67_paired_blocks.py` | 12 | 抽样单位是**天**；单天拒绝给出区间（而非退回 case 级）；case 集不匹配拒绝出配对统计；固定 seed 可复现 |

## 5. 文档

| 文件 | 用途 |
| --- | --- |
| `docs/R7_65_PREDIAGNOSTIC.md` | #65 四项预诊断结果与归因结论（含对战役原始假设的修正） |
| `docs/R7_65_C1_PROCESS_SUPERVISION.md` | C1 结果 + 四张表 + 完整 epoch 均值曲线 |
| `docs/R7_65_C2_C3_FEEDBACK_AND_DEPTH.md` | C2/C3 结果 + 四张表 + 完整 epoch 均值曲线 + 静默 K=4 缺陷记录 |
| `docs/R7_64_CURRICULUM.md` | 1→2→4 curriculum（negative）+ 四张表 + 逐阶段曲线 |
| `docs/R7_66_GATE_AUDIT.md` | #66 门槛自查：G1/G2 满足、G3 不满足，缺失条件 M1–M4 |
| `docs/R7_67_PUBLICATION_PROTOCOL.md` | 冻结于读 test 之前的期刊协议（任务定义、主表、口径、不确定性、分组、多重比较、交付清单） |
| `docs/R7_67_SEALED_TEST_REPORT.md` | **只读 test 报告**：主表、按天 paired block 区间、区域分层、完整 epoch 曲线、局限 |
| `docs/R7_67_CHANGE_MANIFEST.md` | 本文件 |
| `docs/decisions/0009-campaign-scope-and-narrowing.md` | 六条跨任务决策（含重切与剪枝的理由与放弃选项的代价） |
| `docs/R7_SECURITY_SCAN_TRIAGE.md` | #68 的边界声明更新 |
| `README.md` | 新增 R7 三栏导航（smoke / negative study / publication benchmark），并写明各栏证据等级 |
| `docs/R7_TASK_QUEUE.md` | 逐 issue 的终态、绑定 SHA、CI run id、预算与产物核算 |

## 6. 未改动的部分（避免误解）

以下**刻意未改**，理由见 ADR 0009：

- `model/**`：全部 model digest 保持 `d9fb07f2…f665` 不变——所有已发布 checkpoint 的
  身份校验因此仍然成立。
- `model/r7_halting.py` 与默认策略：C2 的结果不构成改默认的依据（#65 C2 要求新协议）。
- store 归一化（`eps=1e-6` floor）：修改会产生新 store 与新 identity，作废全部 checkpoint。
- `data/raw|interim|processed` 与三个归档目录：硬约束禁止改动，本轮未触碰。
- 已冻结的判据（B2 的 `SKILL_CRITERIA`、各协议的 required cells）：**无任何事后增删**。
