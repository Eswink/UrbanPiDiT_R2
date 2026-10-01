# N1 v2 成本修复与 GPU 共驻政策：工程准备记录

**状态：D0/D2 工程准备已验证；D3 具名授权提问未收到回答，未执行任何 GPU 探针或评估。
独立成本验收仍未补齐，不宣告目标完成。N1 保持 paused/current_node=N1，原 48/72h 的
cannot-distinguish 与 N2d 提议不变。**

## 1. 范围与身份

本轮执行 `docs/goals/n1-cost-supplement-repair.md` 的零 GPU 部分。起点
`7064ff138daddde852cce100767aa3d65505b79c`，工程准备提交
`b227020ca3d4932a765a6cd7a79e26fc659cdeaf`，仅推工作分支 `r7/weather-reasoning`。
不动 main、不合并/force/关闭 #70–#75、不读取封存 test、不下载或租 GPU、不新增训练更新/臂/seed。

只读归档核验：30 项原 val、六 checkpoint、source/manifest/receipt/preflight/BUILD_COMPLETE 身份齐。
原 code.zip SHA256 `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a`；原 protocol
`e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01`；source
`496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21`。只有 CPU 哈希/元数据核对，未执行模型。

v1 三项冻结证据在收尾重验仍未变：

| 文件 | SHA256 |
| --- | --- |
| `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md` | `4b350357af20ce95bc9c2e31b7411f83108fda4dc4366e681b61e0cbecdac9fc` |
| `outputs/r7_n1_cost_failed_independent_review.json` | `9540549b02e12bd0aad36c62f89ecf7d744bc1d8d1564cb13c664ab4894a526d` |
| v1 `measurement_code.zip` | `b7f20e7b8adda117c394deb0751fcd1683b536404d5cfa18e121d61c39b742c0` |

## 2. D0/D2 已实现内容

- 决策 0026、`docs/rules/gpu-resources.md` 的 R-054、AGENTS 一行及决策/规则索引、CHANGELOG、E-216 齐。
  默认共驻，启动前/每 spawn 前只读核 UUID 与 ≥2048 MiB，外部 PID 只记录；禁止非本实验信号和
  冻结/终止自动化，独占另取具名授权。门槛不是显存预留，共驻墙钟影响明确列为代价。
- 原四计量文件就地修订：父调度30个 seed/arm/lead 新进程、单行汇总，逐项零基线与显式 peak 读取；
  非零基线先写字节/snapshot 拒绝记录。原归档评测器/模型与 source/checkpoint pins 保持严格核对，
  30 行/17 RMSE变量/provenance/case/逐case MSE 精确重放不新增容差。
- P1 单个1024² FP32 torch matmul，cleanup前后 allocator 字节与 memory_snapshot；私有clear只用于诊断，
  从不用于接受成本路径。P1无残渣才P2同进程两原val；无法归因即停。**这只是实现，尚无真实原因读数。**
- 具名 v2 scope、共驻、P1≤60s/条件P2≤120s、GPU≤900s/whole≤1200s、失败即停全额记账不重试写入回执
  校验；旧v1回执不可复用。协议冻结绝对截止/输出，父PID+唯一token的单次启动claim在CUDA前核对，
  finalizer绑定30个不同launch ID、owned PID/退出及spawn/调用前后观察。超时只kill自身child，有界reap；
  退出未确认如实记失败/计费上限且不冒称实际预算合规。输出拒绝原归档/数据/v1失败目录的上下层重叠。
- 四终态成本表与 cost_views 的接受契约不变；默认输出改为新目录
  `outputs/r7_n1_eval_cost_supplement_v2`。参数/FLOPs/训练吞吐引用历史，只有新独立eval内存属于待测量。

独立工程复核发现并修复：failed P1可能先触发P2、输出v1子目录、无界reap、启动/括号证据缺失、直接worker
截止/重放、float任务标识及失败诊断遮原异常；对应反证已加入。复核最终建议进入工程验证，不认证CUDA路径。
planner请求失败，按能力降级自规划，草案校验verified=true；草案不是证据，不外委科学判据。

## 3. 实际工程验证

| 检查 | 实跑结果 |
| --- | --- |
| 政策元数据及缺失边界反证 | 2 passed |
| v2计量/资源/探针/政策反证全集 | 193 passed（最终定向合集的子集，非额外计数） |
| 准备轮定向合集 | 359 passed，23.31s |
| 精确 b227020 干净 clone 全量 | 1988 passed / 14 skipped / 2 warnings，173.22s |
| 阻断/goal/campaign/索引/空白 | 37阻断0违规；brief0失败；campaign0失败/4notes；索引14条通过；空白0 |
| 工程 CI | run 36855190840，head_sha=b227020完整SHA，completed/success，九主步骤成功 |

干净 clone `/tmp/n1_v2_exact_clone_8cktlsir`，CUDA隐藏，没有真实数据产物；skip分别是6项CUDA与8项可选
真实产物缺失，不计为通过。两个warning是既有Lightning未挂Trainer的self.log，未改测试压制。
远端CI日志计数未取得，不把本地1988当远端计数。17条实验workflow未带触发标签，不作为实验通过。

原日志与匿名API响应归档在 `outputs/r7_n1_cost_v2_preparation/`；准备验证回执
`preparation_verification.json` SHA256 `362c8686d1a87ece440122a8fc2a902fa2c9fd9079262f237044cf3f6ff4cc8f`。
CI一手[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840)、
[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840/jobs?per_page=100)，
访问2026-10-01（匿名curl只读，未用gh/PAT）；逐响应摘要在回执。工程CI不代表科学结论或GPU授权。

测试基线1021/2569→1065/2691（新增44测试函数/122断言）；规则总数55，37阻断不变，R-054未机械化。
尺寸标记201/45/38/23/23，600/200硬上限与冻结例外不扩大。初期fixture缺字段/动态尺寸标记失败如实修复，
未删除测试/断言、未弱化科学或零基线契约。

## 4. D3 授权与未执行状态

2026-10-01完成上述工程门禁后，按决策0021在执行那一刻用 AskUserQuestion 提问具名范围
`n1-evaluation-cost-supplement-v2-with-residue-probe`：P1/条件P2+原30val、共驻、≤0.25GPU-h/≤1200s、
新输出/证据、失败全额计费即停不重试、不改原科学读法。**工具返回用户未提供回答。**
这不是拒绝，也不是许可；没有授权回执，不凭存量预算或原v1授权执行。

截至本轮收尾：`outputs/r7_n1_cost_v2_authorization.json` 和 v2执行目录均不存在；没有probe/attempt/result、
30成本行、四终态表或cost_views，没有原因归因或成本数字；D1/D3/D4仍未完成。未创建
`docs/R7_N1_COST_SUPPLEMENT_V2.md` 来冒充已执行证据。本页只登记零GPU工程准备。

## 5. 影响、限制与出口

本轮GPU实耗 **0 GPU-h**，账本不增行，已用4.0762/余19.9238。数据/归档/模型digest/科学判据不变，
原48/72h cannot-distinguish与N2d仅提议保持，节点不推进。无新增依赖/凭据/安全配置变更；接口仅v2回执/
协议、单lead子请求、新输出及审计字段。v1历史产物要使用其归档measurement_code.zip，不能用v2回执读取器冒充。

Mimosa commit/push前扫描均scanner_enobufs，无安全结论；未执行另一次完整项目安全审计。
CPU/fakeCUDA、CI与只读余量观测不能证明真实allocator、计量可行性或预算；资源查询有竞争窗口，禁网为
Python socket层防意外而非沙箱。未宣称新的bit-reproducibility，精确重放仍待实际执行证据。

下一项的第一个具体动作是执行前取得上述具名授权回答，重读账本/余量/代码与input pins并冻结新回执后，
才允许一次探针+30val。未获授权就维持准备态；不自行重试、扩大范围、放宽零基线、推进节点或宣布目标完成。
