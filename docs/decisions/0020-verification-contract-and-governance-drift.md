# 0020 验证回执身份契约与治理漂移防护

- **日期**：2026-09-29
- **状态**：accepted

## Context

0019 先在 `r7-cpu-study` 试点了机器可读验证回执，但试点生成器仍固定使用单一产物布局，且回执中的 `source_sha256` 主要是对协议字段的转录。仅验证 artifact 文件自身没有被改写，不能证明该文件就是 protocol 和 source receipt 所声明的 source。与此同时，历史证据索引已经可以生成候选简报，但 index 更新后没有机器检查已提交简报是否同步；CI 分类规则、skill 文档和 workflow 实际行为也存在再次漂移的可能。

## Decision

1. 保留 `urbanpidit-verification-receipt-v1` 的读取语义和六种运行状态；新增严格的 `urbanpidit-verification-receipt-v1.1` profile 契约。v1.1 的成功回执必须把 source bytes、protocol source digest、source receipt digest、结果身份和代码身份交叉绑定。
2. 用显式、只读的 artifact-layout profile 描述 protocol、primary result、source、source receipt、代码归档、指标和结果字段指针。builder 不从目录内容猜实验类型，不运行实验、不下载数据、不改写真实产物。
3. profile 只描述工程身份和文件完整性，不计算 forecast skill、不改变科学判据，也不把候选状态转换为训练授权。replay 缺少 source bytes 时只能声明受限的 delegated scope，不能冒充完整本地 source 验证。
4. evidence index 的规范渲染结果作为 candidate brief 的唯一来源；新增只读同步检查，index 或 brief 不一致即失败，但检查器不自动修改 brief。
5. workflow 分类继续以实际 YAML 执行行为为准。先用同步测试和 checker 覆盖修复 rules 文档、`ci-workflow-triage` skill 与 17 个 `r7-*.yml` 的分类、标签、timeout、artifact pin 和实验步骤禁网事实；暂不引入第三份手工 workflow manifest。
6. goal/planner checker 继续保持现有非阻断边界；本决定不扩大到自动训练、GPU、数据下载、main 合并、push 或 force push。

## Consequences

**变容易的：** 回执可以复用于不同产物布局；source 身份链的缺口会在离线校验中暴露；提交的候选简报不会静默落后于 evidence index；workflow 分类和禁网检查能覆盖不含 `study/control` 文件名的 artifact-consuming workflow。

**变难 / 代价：** 回执 profile 需要维护精确路径和字段指针；legacy `signature`/`limits` 结果不能自动获得与新结果相同的验证强度；CI 增加 brief 同步检查；严格 source-binding 会拒绝过去“文件 digest 正确但声明身份不一致”的 fixture。当前没有实际归档的 `verification_receipt.json`，因此真实 workflow receipt 仍需在获得运行授权后单独验证。

**明确不做：** 不一次性改造全部实验 workflow；不从 candidate brief 自动创建 protocol 或启动运行；不放宽 frozen scientific gates；不迁移旧 goal 长文、不处理 Q-009–Q-013，也不引入新的静态分析依赖。
