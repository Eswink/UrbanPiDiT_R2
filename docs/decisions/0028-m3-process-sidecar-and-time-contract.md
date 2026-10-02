# 0028 M3 派生尺度 sidecar 与三类时刻训练契约

- **日期**：2026-10-02
- **状态**：accepted
- **依据**：本轮用户 M3/#73 objective；`docs/R7_MAIN_MODEL_V2_DESIGN.md` §6；
  `docs/goals/n2a-m3-process-supervision.md` §3；工程提交 `d6c98cf1c33eca5885772c473805af3ef0ad62ba`。

## Context

旧 process normalization 用统一的物理单位下限 1e-6，使两条水汽 proxy 的 train 标准差被压缩；
旧 loss 把同一 input-time target 广播到内部 K。M3 要求修复尺度与训练时刻关联，同时保持旧数据、
checkpoint 与模型代码身份不变。新辅助任务不能成为推理读取未来标签的入口。

## Decision

1. 在新 `outputs/` 目录排他发布 `r7-process-scale-v1` sidecar，不修改旧 store。使用实际 train
   manifest 的 history/target 帧并集，按物理单位的 train RMS 定有量纲 scale，再在无量纲值上拟合
   train-only mean/std。逐通道相对 floor 只判退化、不作除数；退化通道显式 mask 且输出严格为零。
   metadata 记录原 std/floor、单位、缩放后 std、有效 mask、源/分割/发布身份；使用原发布标记契约。
2. 三个来源使用不同字段：`input_diagnostics`、训练专用 `future_diagnostic_targets` 与可微
   `draft_diagnostics`。最后历史状态对应 init time，未来训练标签与所有内部 K 的 own-draft 对应
   同一个 valid time。要求显式 UTC ns timestamp，不用 K 推进物理时间或猜测缺失 timestamp。
3. 新 input/future/draft loss 权重默认全零，保留 legacy `process_weight` 默认值；同时开启旧、新
   任务显式拒绝。辅助任务要求精确名称/顺序/宽度与有效通道，不允许 min-width 截断。
4. 只在明确传入新上下文与六字段契约时，scheduled trainer 增加 `process_supervision` 契约。
   绑定 sidecar identity、protocol digest、三权重，以及实际 train 的 FP32 mean/std/有序17通道
   snapshot 与 canonical digest；实际外层 data identity、train manifest/store 必须对上。
   新 checkpoint 评估必须显式提供对应 sidecar，并核真实 held-out store/root 的固定 inverse。
   不传新契约的 legacy signature 保持原结构；不绕过 model/source/data/checkpoint 校验。
5. 诊断前向用 Torch 球面 metric 与固定 train inverse；敏感差分/归一化在 FP32，零场范数梯度有限。
   显式拒绝无效元数据/shape/名称/NaN，推理上下文不读取 future 标签或标签时间。

## Consequences

**变容易的：** 尺度修复与旧产物身份分离，可以追溯、拒绝错误 sidecar；辅助 loss 与真实 streamed
训练使用同一实现；未来调用者可区分可用于推理的 own-draft 与训练真实 future target。

**变难 / 代价：** 新 checkpoint 依赖额外 sidecar 和固定 inverse 身份，不能只拷一个 checkpoint
就评估；旧 batch 未带显式时间信息时必须通过适配器；更多拒绝分支增加维护/测试成本。
FP32差分不恢复 BF16 输入量化丢掉的信息，统计尺度修复也不保证 forecast 改善。物理代理不是
完整天气真值；小段两seed实验没有显著性或过程结构归因能力。

**备选方案与否决理由：** 原地改 store/checkpoint 会破坏历史身份；继续用 1/floor 放大退化噪声
违背冻结判据；只记录 sidecar 而不绑定实际 atmospheric inverse 可让训练签名遗漏关键输入；
把 future 与 draft 同名会模糊监督边界。这些方案不采用。

本决策记录已实现的工程契约，不修改科学阈值、实验案例、预算、GPU政策或授权通道，不宣告
本轮目标完成，也不授权补跑、下一节点、main 或 issue 关闭。
