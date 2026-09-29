# 0019 机器可读验证回执先行试点

- **日期**：2026-09-29
- **状态**：accepted

## Context

当前 R7 运行已经分别记录 protocol、数据来源、模型代码、checkpoint、结果 JSON、CI run 和限制，但这些信息分散在运行目录、GitHub artifact 与人工冻结文档中。工程验证因此能确认很多局部条件，却缺少一个统一的机器可读状态来区分成功、部分完成、失败、取消、排队和按标签跳过。另一方面，项目明确禁止把工程通过当作科学结论，也不能把取消、排队或跳过当作通过。

## Decision

我们采用一个最小的 `verification_receipt.json` 试点，先接入 `r7-cpu-study` 的 `always()` 收尾步骤。生成器只扫描已有文件并记录 run/workflow、完整 commit、protocol/source/data identity、必需产物及其 SHA256、状态、限制和失败原因；独立的只读校验器只检查结构、文件完整性和身份绑定。只有 `status: success` 且完整性校验通过才是工程上的接受状态。回执不计算科学指标、不放宽冻结判据、不选择模型、不启动训练，也不替代 source replay、result-freeze 或人工科学判断。

## Consequences

**变容易的：** CI 失败、取消和部分完成都有统一的机器可读落点；成功运行的关键身份和产物可以用同一校验器核对；后续只读证据索引可以消费稳定字段，减少人工抄录 run id、digest 和限制时的遗漏。

**变难 / 代价：** CPU study 收尾多一个生成与校验步骤；成功状态要求列出的关键文件实际存在，原先只上传目录的缺失会更早暴露；GitHub artifact 的保留期和 job-level skipped 仍是外部边界，回执不能替代长期归档；非确定 GPU 运行仍不能因此声称 bit-reproducible。

**备选方案与否决理由：** 一次性改造全部 workflow 被否决，因为会扩大触发、产物和失败语义的变更面；把回执直接并入现有 protocol 被否决，因为 protocol 是实验前冻结的研究协议，而回执描述运行后的工程状态；让回执自动作科学判定被否决，因为判据和结论必须留在本仓冻结流程与人工复核中。
