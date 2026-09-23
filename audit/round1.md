# 自审 Round 1 — 初始重构

**门槛：90/100**  
**评分：58/100 — FAIL，打回重做**

| 项目 | 分值 | 得分 | 结论 |
|---|---:|---:|---|
| 技术路线一致性 | 30 | 21 | 已有 Coarse/Process/Router/Urban/Verifier 骨架，但 hard route 为 batch-global |
| 可运行性与测试 | 30 | 8 | `scripts/` 与 pytest 均出现根目录 import 失败 |
| 4090D 资源约束 | 20 | 13 | 已采用 window/sparse 思路，但 activation checkpointing 只写在设计中未落实 |
| 数据/工程可维护性 | 20 | 16 | data/model 已拆分，但入口边界不足 |

## 打回项

1. 修复直接执行脚本与 pytest import；
2. Router 改为逐样本 STOP/ZOOM，不能按 batch 平均值；
3. 把 activation checkpointing 变成真实代码开关；
4. 增加正式尺寸资源检查。
