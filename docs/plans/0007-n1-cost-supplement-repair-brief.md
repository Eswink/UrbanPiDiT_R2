# 0007 N1 成本补测失败审阅与修复轮长文（prepared）

**日期**：2026-10-01 · **状态**：已完成（本归档与修复轮长文同批提交）

## 背景与计划

用户报告：一次具名授权的 N1 独立 evaluation 成本补测在完成 1/30 项后失败（第二项零基线守卫拒绝），
已停止、无重试；完整成本验收仍 BLOCKED；要求给出下一轮提示词（承接既有的「主计划 + 每轮对表」机制）。

本会话的计划（对表后执行）：

1. 对表（主计划 §8、账本、`check_campaign_state.py`）并独立核验失败轮的登记声明（哈希、提交、CI）；
2. 0 GPU-h 只读审阅：把「第二项为何基线非零」查到能查的边界，**并如实标注查不到的**；
3. 写修复轮长文 `docs/goals/n1-cost-supplement-repair.md`（prepared，含可粘贴 objective、探针设计、
   v2 计量拓扑、预算与停止条件）；**执行那一刻**才按决策 0021 取具名授权；
4. 回写主计划（状态块、§1、§8）与 `docs/goals/README.md` 索引；门禁 + 本地全量 + 推送 + CI。

## 实际结果

- **完成情况**：1–4 全部完成；**未执行任何 GPU 步骤**（修复轮保持 prepared，等待用户触发与授权）。
- **审阅结论（确认）**：失败发生在第二项评估**调用之前**（清零守卫是该子步骤第一个 CUDA 触点，
  之前只有 CPU 哈希与设备属性查询）；跑过的评测器与归档 `code.zip` 逐字节相同
  （`training/r7_evaluate.py` sha256 `95fff1be…`）；checkpoint 以 `map_location='cpu'` 载入；
  本路径仓内唯一 `lru_cache`（`model/sparse_process_graph.py:6-18`）只产生 CPU 张量；
  扫描未发现持有 CUDA 张量的模块级状态；环境 torch 2.11.0+cu128（与归档 protocol 一致）。
- **审阅结论（未确认，禁止当结论）**：失败基线的具体字节与持有者。**推测**＝进程级 allocator 残渣
  （cuBLAS/cuBLASLt 族工作区），与「第一项完成后即非零」的形态一致，但守卫未记字节、未取证。
- **修复设计**：零基线由**构造**保证——每次评估一个全新进程（30 子进程）；已考虑但未采用
  「调用私有释放 API」方案（依赖残渣归属且引入私有 API）。探针三态读法已预声明（长文 §3.1）。
- **登记核验（确认）**：失败页 sha256 `4b350357…`、复核 JSON 在盘；三提交
  `1e03f82/33d57d6/9d8d2b6` 均在 git；CI `36737308495` 对 `9d8d2b6` `completed/success` 且九步全绿
  （匿名 API 只读核对）；账本 4.0762/19.9238 未变（本轮 0 GPU-h，不加行）。

## 与计划的差异

- 计划把探针放在「执行授权」之后直接跑；落文时改为**先实现、先过测试与 CI，再取授权跑探针 + 30 项**，
  避免把未过测试的代码带到授权窗口里。
- 立即可得的只读证据不足以把原因钉死（守卫未记字节是 v1 的实现缺陷）——审阅只做到「范围收窄 + 推测标注」，
  并把取证动作移进 v2 的 D1 探针；这比原计划更诚实也更有边界。
- `docs/goals/README.md` 的索引缺口（`main-model-v2-*` 全部缺行）是审阅时顺带发现的，一并补上（索引非证据页，可回填）。
- 按仓库登记纪律补了 **E-215**（附机械重数）与 CHANGELOG 一条——原计划草稿只写了主计划与长文，
  但历轮惯例是 E 条目 + CHANGELOG 同期记录，否则证据台账会漏掉这一遍审阅。
- 机器对表暴露一处口径：审计轮长文自带 `<!-- round-node: N1 -->`，因此主计划的
  `previous_node` 应为 N1（不是 N0），且 `current_round_goal` 改成修复轮后旧的
  `current_round_record` 键不再指向本轮，已按工具契约同步（C-04 先红后绿，见验证）。

## 验证

- `python tools/check_campaign_state.py`：`failures=0`（节点 N1；账本算术；上一轮指向；本轮 brief 结构）。
  过程中先出现两条失败（G-07 判据节首个 `docs/...md` 指针缺失；C-04 previous_node=N0 与审计轮
  的 round-node=N1 冲突），均按工具提示修正后归零——没有放宽工具或判据。
- `python tools/check_goal_brief.py --brief docs/goals/n1-cost-supplement-repair.md`：0 失败
  （objective 907 码点、单段、点名自身路径）。
- `python tools/check_conventions.py`：37 条阻断 0 违规；`pytest -q`：本机 **1872 passed / 3 skipped**；
  定向（goal brief + campaign state）32 passed。
- `python tools/verify_r7_evidence_index.py --index docs/R7_EVIDENCE_INDEX.jsonl --root .`：PASS（14 records）。
- 推送后 CI run 绑定本提交 SHA（工作分支，普通 push，只触发工程 CI）。

## 遗留

- 修复轮执行（探针 + 30 项）未发生：需要用户触发并按决策 0021 取新的具名授权；
  预算余量 19.9238 GPU-h 只是算术余量，不是重试许可。
- v1 失败基线的字节与持有者仍无取证；v2 的守卫将记录字节与 snapshot 摘要。
- N1 的 48/72h 仍 `cannot-distinguish`、N2d 仅为提议；成本补齐不改变科学读法（长文 §3.2、§6）。
