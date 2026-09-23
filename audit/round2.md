# 自审 Round 2 — 路由与资源修正后

**门槛：90/100**  
**评分：86/100 — FAIL，继续打回**

| 项目 | 分值 | 得分 | 结论 |
|---|---:|---:|---|
| 技术路线一致性 | 30 | 27 | 逐样本路由、shared recurrent reasoner、selected-only Urban Expert 已落实 |
| 可运行性与测试 | 30 | 27 | 8/8 单测通过，Lightning fit→checkpoint→test 通过 |
| 4090D 资源约束 | 20 | 17 | 13.39M 参数、window attention、O(Nk) sparse graph、checkpointing 已落实 |
| 数据/工程可维护性 | 20 | 15 | 多尺度 contract 与 manifest loader 已有，正式数据尚未落盘 |

## 不通过原因

发现关键科学闭环缺口：`Verifier` 只有输出，没有监督损失。未经训练的 verifier 不能可靠控制 `MORE_REASONING/STOP`，因此“Verifier-guided reasoning”仍不成立。

## 打回项

1. 为 verifier 增加显式训练目标与 loss；
2. 验证 verifier 梯度信号；
3. 重新跑正式尺寸前向/反向与最终集成测试；
4. 完善旧版源码隔离、租卡 benchmark 与资源文档。
