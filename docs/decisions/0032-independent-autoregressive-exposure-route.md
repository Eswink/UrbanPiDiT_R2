# 0032 M3 辅助监督终结，后续采用独立自回归暴露主线

- **日期**：2026-10-03
- **状态**：accepted

## Context

起点 `edcc33539178f95646983eca3084f7c16c241692`，工作分支 `r7/weather-reasoning`。
N2a规定交付已经完成：原一次attempt失败、独立补测23项及跨来源30项完整覆盖、正式登记与精确CI
均存在。`docs/R7_73_VALIDATION_COMPLEMENT.md` 的any-unresolved/paused与advance=false是该辅助
监督实验的冻结出口，不能回写成成功。此前主计划将这个出口解释成N3–N5总前置。

用户本次明确选择：结束M3辅助监督假设，保留其冻结出口；以aux_off或合法既有RW-A母体转入独立
autoregressive-exposure主线，不再将已交付辅助监督假设的暂停当作所有后续实验的总前置。
这是用户对总体路线的前瞻选择，不是执行者以普通推进权取消旧科学停止线。

## Decision

1. 后续采用独立自回归暴露路线，承接主campaign的B/N3、C/N4、D/E/N5。N2a的失败、负面、未解决
   cell、paused、advance=false、协议、结果及证据页保持原样；不再补23项、不加辅助权重或seed
   使M3闯关。新goal为 `docs/goals/v2-remaining-stages-exploration.md`，同步主计划与next-action。
2. N3先核父模型实际身份与深度。M3 aux_off候选父为K4；旧process_spacetime_rwa候选父实际是
   RW-A/K3，不能由路径或旧计划误称RW-B。复用必须强核归档code/model/source/data/sidecar身份；
   当前代码不兼容时采用有逐参数映射、旧身份接受和新身份明确记录的显式导入，或另立有界重训，
   不修改旧加载器的digest比对，不伪造旧contract。
3. matched-Generic V2复用同款既有组件、同结构与同已知输入，不赋予过程监督语义。先实现真实可微
   两步物理rollout，同seed同父比较继续L6与L6+0.5L12；两个物理步与内部K分轴报告。未来真值
   只进监督，第二步history来自第一步预测。两臂两预声明seed及可负担equal-compute对照先于确认。
4. C执行旧Ours/matched-Generic V2/Process V2三臂至少三预声明seed，候选不成立也交付公平负向
   对照。新协议运行前指定primary、退化容忍、案例/seed/单位/选择规则，引用既有判据而不回改
   冻结端点。全17变量、五lead、全域/边界与K1/2/4、四成本照报；adaptive仅独立预登记门满足
   才启动，否则“不启动”。三个seed不是显著性。
5. 普通故障、负面或不可分辨只终结相应attempt/假设，独立任务继续。下一探索须先核证据、写可
   反驳假设、核一手来源、做最小探针，再独立预登记对照并登记取舍；不机械增加算力、seed或unroll。
   D/E补足#71/#72现役机制反证，逐issue以实际证据给工程、实验及科学不同verdict。
6. 完整继承0030的方向内实验/普通工程/方法/预算/节点自主权和保留权限。逐轮planned/hard先冻结，
   整轮含前置至清理，软超继续记overrun、硬截止或真实错误停止该attempt，全额记账、新协议新输出
   方可修复重试。无总GPU-h许可闸门，无cron/守护或会话后自建续跑。
7. #70–#75仍仅在各自证据充分、verdict受支持且精确工作分支SHA主CI必要步骤全部success后，按0030
   写Closes并非force ff推main、匿名核终态。#70没有实际改进不标DONE-positive；充分负面结题
   明列重开条件。main分歧不强推或擅自合并。关闭不是科学成功或最终goal完成。
8. 联网准备期用户明确授权clone必要公开官方参考源码，隔离在
   `outputs/reference_sources/<project>/<commit>/`，固定URL/访问日/commit/license/文件hash及借鉴范围，
   不运行上游安装/训练脚本、不import归档。实验离线；不下载天气数据或发布--write。

## Consequences

**变容易的：** 已交付辅助假设不再成为独立物理rollout与公平归因的总前置；能回答不同机制的问题，
并完成即使负向也有价值的确认与逐issue交付。方向改变与执行细节授权都有可检索出处。

**变难 / 代价（如实列出）：** 新主线需要新的训练契约、父身份导入证明、可微两步内存与算力测量，
不能直接把历史分数拼成当前确认。新增实验可能仍无收益；既有两个月区域/22val窗口限制仍在，
无法据此给泛化或显著性结论。新模型digest使历史评估必须继续使用其归档代码，证据管理成本增加。

**备选方案与否决理由：** 重跑M3、增加辅助权重/seed或改any-unresolved出口没有新机制且违反冻结
纪律，不采用。把暂停改成旧成功、把CI绿当研究进展、用当前代码弱化旧digest均否决。停止整个目标
等待普通方法选择不采用，用户已明确路线并下放执行。扩大外部baseline竞赛或新数据范围不在本路线。

本决策仅前瞻限定0031第5条出口的适用实验范围，不取代或回改0031及历史证据、判据和失败事实。
