# 0008 N1 成本轮的共驻扩围与 goal 交接（prepared）

**日期**：2026-10-01 · **状态**：已完成（本轮只产出目标提示词与承载文档；执行留给新会话的 goal 轮）

## 背景与计划

用户报告：两张 GPU 都被同一账号下的其它作业长时间占用（spine_xray 训练与 fives_snr_pilot），
询问「有几张卡空闲」并要求：需要跑实验时**直接冻结他的所有进程、跑完再恢复**；随后依次改为
「直接杀掉他的进程（对方已同意）」与「充分利用 GPU 显存，写成项目规则」。用户最终把范围收窄为：
**只给出 goal 目标提示词与相关文档文件，后续由用户在新窗口以 goal 形式执行。**

本会话的计划（经只读侦察后收敛）：

1. 不做任何冻结/终止实现（`SIGSTOP` 不释放显存，且 GPU 独占校验会拒绝；终止会让不可续跑的邻居作业
   损失数小时进度），改为把用户的意图落成**默认共驻**政策；
2. 就地扩围 `docs/goals/n1-cost-supplement-repair.md`（保持 `<!-- round-node: N1 -->`，不新建第二份
   N1 轮次文档）；新增 D0 政策交付物，把 D2 守卫改为余量门槛、D3 授权改为共驻；
3. 同步 `docs/goals/README.md` 索引行与主计划 §8 说明（状态块、账本、节点不动）；
4. 写本质计划归档；不改任何代码、不跑 GPU。

## 实际结果

- **完成情况**：1–4 全部完成。产出五件：扩围后的 N1 轮长文（§0 objective 实测 1024 字符、§1.2 共驻前提、
  §2 D0 行、§4 顺序、§5 余量门槛与停止条件、§6 禁信号与禁自动化）、`docs/goals/README.md` 索引行、
  主计划 §8 扩围说明、本归档、以及聊天消息里可直接粘贴的 objective 全文。
- **未做**：未写决策 0026 / 规则细则 / `AGENTS.md` 一行的正文（定为目标内的 D0，随测试与 CI 一起落地，
  避免半成品治理层）；未改代码；未跑 GPU；未读 test；未碰 `data/` 与归档；未新建第二份 N1 轮次文档。
- **与计划的差异**：
  - 计划里写的「objective 实测 1078 字符」是估算，落文后用校检器实测为 **1024 字符**，已按实测值改正；
  - 侦察发现工作树已前进三个提交（`c56cbe3`/`afe9a8f`/`9c814dd`，后者刚把交付物编号对齐为
    D1…D5），计划中的「D0 + D2/D3/D4」命名据实调整为「在 D1…D5 之前插入 D0」，未重排既有编号；
  - 政策载体从计划里的「决策 0026 + 规则 + AGENTS.md 由本轮写入」收敛为「本轮只写规格、正文由目标 D0 落」，
    因为用户明确要求本轮只交付提示词与文档。
- **验证**（实跑）：
  - `.venv/bin/python tools/check_goal_brief.py --brief docs/goals/n1-cost-supplement-repair.md`
    → `briefs=1 failures=0 advisories=0`，退出 0；
  - `.venv/bin/python tools/check_campaign_state.py` → `node=N1 failures=0 notes=4`，退出 0
    （4 条 note 是历史账本行无索引支撑的既有提示，非本轮引入）；
  - `.venv/bin/python tools/check_conventions.py` → `blocking rules=37 failing=0`，退出 0；
  - `.venv/bin/python tools/check_goal_brief.py --brief docs/goals` → `failures=15 advisories=6`，
    全部落在 `docs/goals/README.md:32-34` 明确登记为「早于约定、报告而非待办」的六份旧长文上，
    本轮文件 0 失败；
  - 本地全量 `pytest -q` → **1872 passed / 3 skipped**（170.96s）；3 个 skip 是
    「干净 checkout 不含真实数据 fixture」的既有跳过，skip 不算通过。
- **遗留**：共驻政策的正式文本（决策 0026、规则细则、AGENTS.md 一行）与 v2 计量改造仍待执行轮的
  D0/D2；N1 成本验收仍 BLOCKED；本扩围不改写 48/72h 的 `cannot-distinguish` 与 N2d 提议。
