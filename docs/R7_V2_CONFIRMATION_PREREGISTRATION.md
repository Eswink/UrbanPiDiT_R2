# V2 三臂确认的前瞻判定与 adaptive 门禁

日期：2026-10-03。状态：**运行前预声明，尚无 C 预报结果**；`scientific_claim: false`。
本页不更改 M3/RW-B/N1 的旧判据、暂停或负结果。执行权限来自0030/0032；科学读法来自
`docs/plans/0004-r7-main-model-v2.md` Evaluation protocol / Scientific gates、ADR0023及issue#75正文。
本页中的明确字段将在独立 C `protocol.json` 以 digest 冻结，不能在运行后调整。

## 1. 假设、对照与选择

- H-C-package：在相同 source、训练统计、窗口、seed、初始化映射及训练配方下，新的确定性已知
  信息和局部/位置读取整体可能改善旧 pooled Ours。旧 Ours 与新模型的信息预算不同，改善只可归于
  整体包，不能拆称某一个 query、reader 或过程语义的作用。
- H-C-process：相同结构、相同已知输入与对应初始权重的 Process 和 matched Generic 是否在预报
  目标下分离。无辅助过程监督时，它们可能函数及梯度等价；完全负向或 cannot-distinguish 是必要
  对照的合法结果，不机械增加 seed、训练长度或模块直到出现分离。
- 三臂固定为 `old_ours`、`process`、`matched_generic`；seed固定41/42/43，各400updates、Ktrain4。
  所有臂使用同一新建优化器与 full internal-K 梯度配方，初始/all-draft深监督保持。
- 物理训练配方仅由 B 的预声明选择规则决定：两步臂须在两个 primary lead、每个 B seed 都严格
  优于200-update L6，且对400-update L6控制不恶化，否则 C 保留 L6。不能依长lead或某个幸运seed改选。
- 最终模型规格与完整 target←anchor映射在 C 运行前独立冻结；对应的工程缺陷修正必须有默认关闭
  兼容与直接反证。未经对照不能声称修正带来独立预报收益。不得复制旧分数代替新三臂运行。

## 2. 数据与端点

- 只消费既有 M2 train/val；test不读取、不重封存。三个臂共用预声明185个精确t+12可用训练窗口。
  唯一 split 边界排除随 protocol/window digest固定，不允许临时skip或跨split填补。
- Primary固定为 `t2m` 的6h与12h，单位K，允许退化容忍 **0.0**；比较同seed、同lead、同K、同region。
  不对不同物理单位取平均；不跨lead平均不同病例集合。
- 完整端点为17个已冻结通道×6/12/24/48/72h×full/interior/edge_2×K1/2/4；每lead保留全部
  22/21/19/15/11个初始化，边界固定2格。K1/K2来自同K4 checkpoint，不是独立训练K1/K2。
- 报RMSE、物理单位climatology MSE skill与真实 pooled ACC，保留零能量undefined、负skill、坏变量
  与逐病例统计。另报零训练persistence与train-only climatology，基线不隐去。
- 总表使用既有#60逐seed同号三态读法；primary两lead须每seed改善才称本小协议的重复改善。
  混合/平局/未决全部保留，三seed只提供一致性，不表示显著性或独立样本数充足。

## 3. 前瞻 adaptive gate（schema `r7-v2-adaptive-gate-v1`）

固定候选为C的 `process`，不在结果后挑较有利模型、变量、region或seed。
下列条件**全部**成立才允许另立 train-only拟合/val-only标定的控制器协议；门禁计算本身不训练控制器。

1. **整体预报改善**：full/K4的t2m6h与12h，在41/42/43各seed的RMSE均严格低于old_ours。
   此条件只证明包级开发信号；matched Generic结果仍决定过程归因，二者不混称。
2. **固定K的准确率—成本取舍**：对两个primary lead的每个seed，K4 RMSE严格低于K1，同时实测
   isolated forward median latency满足K1<K4；完整报告K2，不能用reasoning-step数量代替实际latency。
   任一成本缺失、非正或不一致则gate未满足，不用理论FLOPs代填。
3. **病例级异质性**：同一seed/lead内逐病例t2m MSE在K1/2/4选择最小值，形成纯描述oracle。
   其pool后的RMSE须严格低于该seed/lead最好的固定K；唯一最优K必须至少覆盖两种深度，且各有
   非零病例数。并列最优不计为深度偏好，不靠选最贵K后宣称异质性。
4. **完整工程证据**：全库存、精确配对、17变量与坏case、单位、输入/产物/代码身份、实际latency
   和allocated/reserved计量全部接受；未决的机械缺项不能当作科学gate通过。

门禁输出 `evaluated`、`conditions`、逐seed/lead证据、`start_training` 与
`oracle_deployable:false`。oracle使用未来误差，仅是此门的描述性上界，**绝不进入模型或部署策略**。
不满足任一条时输出 **not-started / start_training:false**，是合法交付而非skip通过；旧#66负面保留。
缺schema或证据时输出 refused/unevaluated，而不是默认通过。即使gate成立，新控制器仍需自己的冻结
协议、预算、train/val信息边界与相对强K1/Kmax实测对照，不自动获得科学支持。

## 4. 层级、失败与限制

- ENGINEERING PASS由实际forward/backward/shape/默认兼容/poison/resume/FP32/BF16及身份验证支撑。
- EXPERIMENTAL SUPPORT仅是此局部协议、同seed主要端点的重复改善。Process相对matched Generic
  未分离时不能把包级改善写成过程结构或过程语义获支持。
- SCIENTIFIC SUPPORT在本轮**不宣称**：冬季单区域、小样本、三seed、已开发数据且缺独立冻结最终
  test/代表性和不确定度，不满足发表级门禁；正ACC不等同正climatology skill。
- 完整失败/硬截断/缺worker/错误身份须停止对应attempt并全额计费，保留失败；合法修复另立新
  协议/输出，不复活旧attempt、不事后放宽端点、容忍或case。
- C整轮planned10800秒/hard21600秒；包含最早CPU准备、imports、归档、prepare-run间隔、训练、评估、
  汇总及owned清理。软超继续记录overrun；continuous GPU记首spawn至末owned reap，包括间隔与失败。
- 结论必须绑定实际 protocol、代码commit/zip、输入与产物digest、工程CI和独立复核。本页自身不是
  实验进展、科学结论或关闭issue的充分证据。
