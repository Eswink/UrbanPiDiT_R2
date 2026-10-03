# 0035 补齐原known-context与source-key位置契约，B不回改

- **日期**：2026-10-03
- **状态**：accepted

## Context

原issue#71明确要求由UTC和经度纯推导本地日相位、显式history offsets，以及内容固定位置置换/
内容位置共同重排反证；#72要求P更新读取位置与来源角色标记的C/E(Y)。616b029真实原文对表与
两次只读审查确认：旧8feature网络仅独立geo+UTC相位，历史slot/cadence虽存在但没有模型offset
接口；recurrent_key仅共享role未显式绑定draft位置。现direct query已0034修复，但这是不同边。

正在执行的B protocol `9f76e6e3e0eb0d7e27d1aaaa8d59ff7616241ff50e13c30b26be7f13f5e62b02`
使用合法aux_off母体模式。任何运行源修改会破坏冻结身份；普通修复按0030/0032自主，但不能回写
B/M3或拿旧分数当新功能效果。C为独立三臂整体包与完整matched Generic确认，不是每feature归因。

## Decision

1. B结束与身份封印前不改active runtime/source或HEAD。B结果/旧checkpoint保持其code.zip身份。
2. 后续新增严格布尔 `known_context_inputs=False`：开时require spacetime、真实已知calendar year、
   `history_offsets_hours`及既有UTC/坐标；缺失/invalid必拒绝，不默默降级到UTC或取系统clock。
3. offset为历史UTC timestamp减init的有符号小时，sample[T]/batch[B,T]，长度与history_steps一致、
   finite、严格递增、末项0，原producer从已验证store/manifest在线导出，不改任何原始store。
   序号与固定history slot相同，不mean-pool掉历史顺序。物理history推进后相对offset按同cadence
   保持，内部K不推进offset。负offset的日历使用既有Gregorian arithmetic。
4. local phase采用地方平太阳时 `sin/cos(2π*UTC_hour/24 + longitude_radians)`，不是民用时区。
   先原生网格算harmonics，再同patch padding/pooling，避免直接平均跨日界线的经度角。
   显式历史每slot年内/local phase与offset幅度、valid-time local phase进小附加context residual。
   保留旧8feature网络的形状/参数；新projection最后于isolated_stream下构造，关路径不调用且逐位旧表达式。
5. 后续新增 `source_position_markers=False`，开须位置reader；用同既有fixed token position basis，
   在P更新的context/draft source-key各加一次后再加来源role，不累加到carry，不新增encoder/attention。
   Generic与Process共享helper/相同生效语义；无draft时保留context标记，role/assertion不弱化。
6. fixed/streamed/adaptive/physical-rollout及真实producer/whitelist/strict语义contract同步验证；
   direct query仍0034的LN(C+E(Y))+pos而非重复位置加draft编码。共同重排要求完整token等变或
   latent集合不变，固定同参数/role/dropout；不以pooled moments接近替代。位置错配使同判据失败。
7. source变更后独立fullsuite/反证/精确CI再冻结C source/spec/完整same-anchor映射与新digest。
   C旧pooled对新known-context/reader只归于整体包，Process对完整Generic才是同信息结构对照。
   不改变已预声明primary、容忍、B选择或adaptive科学gate；不能声称补齐必然提高天气技巧。

## Consequences

**变容易的：** 原始验收scope的明确缺口变成可拒绝、可反证、可审计的已知输入契约；避免用可能
学习的MLP交互或间接反馈代替显式local phase/位置绑定；新Generic与Process仍公平配对。

**变难 / 代价（如实列出）：** 新字段与projection增加接口、参数和成本；旧model digest改变，
B必须先封印后才修改，历史分数不可混用。C包级对照无法单独归因新增特征，若与Generic相同不能
声称过程语义有效；新增phase可能无益或伤害。default-off兼容需真实位等价及全路径验证。

**备选方案与否决理由：** 静默把缺字段降级、把history offsets限制非负、pool mean/std证明joint
permutation或省略Generic对应结构均不采用。直接扩旧8feature Linear改变旧weightshape不采用。
仅写limitations然后假关原缺口不采用；再加solver家族或无限seed/unroll直到获胜也不采用。
