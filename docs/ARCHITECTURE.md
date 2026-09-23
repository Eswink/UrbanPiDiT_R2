# UrbanPiDiT-R² V6 架构规范

## 核心计算图

```text
coarse_history ──> CoarseEncoder ───────────────┐
                                                │
urban_static ──> cheap StaticPreview ───────────┤
                                                v
                                       Process State Bank
                                                │
                                  shared recurrent Reasoner
                                                │
                                      STOP / ZOOM Router
                                      │             │
                                      │             └─> Urban Expert
                                      │                    │
urban_baseline ──────────────────────┴────────────── + residual
                                                           │
                                                        Verifier
                                                           │
                                                 PASS / MORE_REASONING
```

## 为什么不再使用固定 0.25° 城市网格

0.25°适合作为大尺度气象背景，但与建筑、街谷、粗糙度、热储存等城市过程存在明显尺度错配。V6 将 coarse atmospheric context 与 urban evidence 解耦：动态天气主 target 第一阶段约 1 km，形态证据可来自 10–100 m 数据；未来 micro expert 再接 4–20 m simulation teacher。

## 4090D-first

- Urban 128×128 tile 使用 patch=4，变为 32×32=1024 tokens；
- attention 仅在 8×8 window 内计算；
- sparse graph 仅连接 4/8 邻域，O(Nk)；
- Process State 仅 24 tokens；
- reasoner 参数共享，增加 reasoning steps 只增加 test-time compute，不增加参数；
- 默认 batch=1 + gradient accumulation；
- 正式训练使用 BF16 + activation checkpointing（后续 GPU 训练脚本可进一步为每个 block 包装 checkpoint）。

## Reasoning 定义

V6 不把自然语言 CoT 作为模型核心。模型内部推理对象为结构化 `Process State`，默认包含 12 anchored states 与 12 free latent states。Verifier/Router 对其进行监督和消费。未来 LLM 只读取结构化证据，作为外部解释与工具层。
