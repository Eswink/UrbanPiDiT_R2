# 0034 补齐显式局部草稿 query，保留旧读取模式与父资格

- **日期**：2026-10-03
- **状态**：accepted

## Context

设计 `docs/R7_MAIN_MODEL_V2_DESIGN.md` §3与issue#72明确要求逐位置query由C_i、E(Y_k)_i与pos_i构造。
活跃 `PositionalProcessReadout` 实际仅对context作LayerNorm加位置；草稿只经递推更新P后间接影响
reader的K/V，或经局部solver的独立输入影响Z。D4已证明现役P/read→Z的因果边及必要梯度，但不能
把它替代直接draft→query的实现。独立只读审查再次确认了这个差异，旧preflight已如实披露。

用户0030/0032下放方向内架构选择及暴露缺陷的活跃修复，不允许回改旧负结果、归档或判据。
B需要复用合法aux_off RW-A/K4已学父，不能悄悄改变它的结构。C仍需相同信息/完整结构Generic对照。

## Decision

1. 新增显式布尔 `draft_query_feedback`，默认False。开启须同时有positional reader与forecast
   feedback，缺任一依赖必须拒绝，不能默默忽略。
2. 开启时同位置C与已存在的E(Y_k)相加，再经既有query_norm，最后加固定position encoding：
   `Q_i = LayerNorm(C_i + E(Y_k)_i) + pos_i`。不新增参数、token数、全图attention或外部依赖。
   pooled-query容量控制仍在位置注入前pool，保留其无位置差异的定义。
3. Process与matched Generic必须完整同步该开关与readout接口；固定/streamed/adaptive通过各自既有
   共享step透传同一个draft encoding，不能再复制更新公式。关闭路径保持原表达式、初始化流与state keys。
4. 增加固定P/C/pos而只变局部draft的直接query反证，隔离间接P和直接solver输入；阻断/丢弃query
   feedback须使同一判据失败。并核默认位等价、全K/stream/adaptive、梯度、poison及FP32/BF16。
5. B声明False，合法父导入不准靠此API修改架构；旧loader继续拒绝新digest。修正候选是否纳入C须
   在独立C协议/model-spec/初始化mapping中预先声明，不能拿旧对照或工程PASS证明预报收益。

## Consequences

**变容易的：** 直接实现已写定的局部草稿寻址依赖，而不是用间接P反馈混淆query语义；新增开关无
参数增量，既有父模式与Generic公平对照均能明确声明并审计。

**变难 / 代价（如实列出）：** 更多构造组合与元数据需要维护，模型源码digest再次变化，所有新实验
须绑定最终source；旧分数不能与开启模式混用。即便direct query工程成立，可能无收益或更差。若C
同时使用其他新结构，只能确认整体包，不能声称单独query贡献；精确因果与独立预报对照是不同证据。

**备选方案与否决理由：** 把间接P/read→Z等同严格Q(C,E(Y),pos)或改设计删去E(Y)会掩盖缺口，不采用。
默认打开会悄悄改变合法父与旧路径，不采用。扩大query通道并加新投影会增加容量因素，当前不采用。
重开已负面的spatial_solver_feedback直接加solver不是本次query修正，仍保持其原对照与负面。
