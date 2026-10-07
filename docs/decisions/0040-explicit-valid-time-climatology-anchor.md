# 0040 显式 valid-time 气候态解码锚与异常反馈

- **日期**：2026-10-07
- **状态**：accepted

## Context

主模型长时效尚未达到气候态科学门。原单病例目标响应、无更新目标—最终评分诊断和等曝光顺序试验分别已登记；最后一项开发支持为 false，不是遗忘或容量原因的证明。继续增加该实例剂量没有前瞻依据。

现有 `NativeAtmosForecaster` 默认解码 `X_t + tendency`。Process 的普通 correction 更新绝对 draft，而 RW-B proposal 又从物理 `X_t` 解码。`atmos_baseline` 只属于裸 forward 的历史接口，不在正式训练/评分 input whitelist，不能借它偷塞气候态。长期方向内普通架构与工程决定由决策 0030/0038 下放；科学门仍仅来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。

## Decision

采用 default-off 的显式 `climatology_anchor_spec` 与 `anomaly_feedback` 接口。新模式将 train-only valid-time 固定气候态 `C_valid` 与物理当前状态 `X_t` 分开：初始 forecast 为 `C_valid + A0(history)`；RW-B 绝对 proposal 也从 `C_valid` 解码；开启异常反馈时 draft encoder 消费 `Y_k - C_valid`。普通 correction 仍更新绝对 draft，最终 `Y_K` 仍是 forecast，物理自回归 history 仍写入绝对 forecast。typed evidence 和物理语义继续使用 `X_t`，不能将其改名成气候态。

气候态由已有 train-only month/hour/grid estimator 在模型外拟合一次，表与 source/train/norm/grid/channel/unit/桶覆盖及 counts 身份写进完整 JSON 模型配置和实验契约，normalized FP32 表为 checkpoint persistent buffer。spec-only 构造器仅分配 persistent 表占位与 `ready=false`，不读取文件；只有一次显式安装，或工厂 `make_model(contract['kind'], contract['model'])` 后普通顶层 `model.load_state_dict` 的递归严格恢复，先核完整表字节/hash/shape/FP32/ready 才能成为可用状态。未安装、缺失/篡改表或未 ready 时拒绝 forward；不能仅在子模块 `load_state_dict` 覆盖里校验而让普通顶层恢复绕过。只在安装和恢复时校验表字节身份；forward 不拟合、不读取 store/文件/未来标签、不改表。缺桶、单位/归一化/网格/身份不符或非有限直接拒绝，无告警跳过、补零或 held-out fallback。既有 16 桶不假装全年覆盖。

每次物理 transition 从已知 Gregorian 初始化年月日及 lead 查 valid-time；同一次 transition 的所有内部 K 使用同一个 `C_valid`，只有下一物理 transition 推进日历。气候态日历需求独立于空间时间 positional embedding 开关，正式 input whitelist 不扩为任意 baseline。气候态模式与 `atmos_baseline` 同时出现时拒绝。

关闭模式不得增加 state_dict keys、参数、RNG 消耗或浮点运算；旧默认路径保持逐操作兼容。开启模式不增加训练参数，但新增固定表存储与 query 运算需要分列成本。固定、streamed、adaptive 及其 calibration/oracle 共用过程 step 并显式传递 `C_valid`，不在模型上缓存可变的每调用 anchor。

活跃模型源码改变会改变全局 digest。旧 checkpoint 继续只用它的归档 `code.zip` 和原普通 loader 资格；新实例采用独立、逐 key/shape/dtype/byte 核实的 model-only 权重导入和新 code/完整契约/普通 checkpoint，不放宽 digest。新候选 update0 forecast 已变化，必须重新评分。已完成且同病例/recipe 的 interleaved 控制0/80分数合法核 pins 后复用，不重训或恢复其 optimizer/cursor/RNG。

## Consequences

**变容易的：** 输出先验、反馈表示和物理状态有显式独立接口；控制默认路径可复核，模型恢复不依赖运行时读 store；负结果不会复活旧顺序实例。

**负面后果 / 代价：** checkpoint 与新配置更大；新增日历/表身份失败面；需要贯通多条固定、streamed 与 adaptive 入口。稀疏桶不能预测缺覆盖日期；新模型不能用旧 digest 直接加载。相同 trainable 参数数量和更新数不等于相同信息、存储或 FLOPs。锚和异常反馈组成 package 性能假设，不是单独机制归因，可能使近时效或守门变量变差。该工程选择没有获益证据，也不保证科学门可达。

**备选与否决理由：** 不采用 whitelist 外 `atmos_baseline` 注入、在 forward 读/拟合气候态、单位不符只告警、伪造全年桶、以 `X_t-C_valid` 精确补偿将旧模型仅改坐标名字，或只改初始锚而保留 RW-B proposal 的旧物理锚。它们分别绕正式输入/身份、引入泄漏或不对应所声明的模型表示。保持旧模式供历史重放；未取代既有科学合同、输入白名单或原失败停止线。
