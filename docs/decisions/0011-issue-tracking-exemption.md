# 0011 —— 会话内无 GitHub 写通道，issue 追踪按环境限制豁免

- **日期**：2026-09-28
- **状态**：accepted
- **范围**：所有要求「开 issue / 评论 / 关闭 issue」的流程（objective ⑤ 等）
- **代码 SHA 范围**：`0e09800` → `ee616b8`
- **依据证据**：`docs/R7_69_BUCKET_EXPANSION.md` §12.1；AGENTS.md「GitHub 通道」节

## Context（背景）

objective ⑤ 要求「新发现的问题自行开 issue，并在 DONE 后经 `Closes #N` + main ff 关闭」。
ZCode 会话内需要判定：这条要求**能否在本环境执行**。此前 AGENTS.md 记载「API 写
（评论/关闭 issue）**401**」，但那是既有笔记；本决策把它改成**本次实测**，并穷尽所有
可能通道与凭据来源，避免把「能力不可用」误记成「执行者跳过」。

## Decision（决定）

**在 ZCode 会话内，issue 追踪按环境限制显式豁免；科学交付物不依赖任何 issue 状态。**

穷尽探测结果（2026-09-28，全部实测，未打印任何秘密值）：

| 通道 | 结果 |
| --- | --- |
| `POST /repos/Eswink/UrbanPiDiT_R2/issues` | **401** |
| `POST /repos/Eswink/UrbanPiDiT_R2/issues/69/comments` | **401** |
| `PATCH /repos/Eswink/UrbanPiDiT_R2/issues/69`（关闭） | **401** |
| `POST /graphql` | **403** |
| `GET /user`（认证成功才 200） | **401** ⇒ 会话未认证 |
| `GET /issues/69` | **404** |
| `gh` CLI | 未安装（项目亦不用 `gh`） |
| git / SSH | 可用，但 git 协议**没有 issue 动词** |

凭据来源核查：`GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_PAT` / `GITHUB_API_TOKEN` / `GH_PAT`
全部未设置；`credential.helper = store --file=.git/github-credentials` 所指文件**不存在**；
`~/.netrc` 仅含 `api.wandb.ai` 与 `wandb.r6siege.cn`，无任何 github 条目。
即：**本机不存在可用的 GitHub 写凭据。**

`Closes #N` 生效需同时满足两个条件——issue **已存在**，且提交落到**默认分支**。
前者需要那条 401 的写通道，故二者不可得兼。git 通道只能推送提交，无法替代。

豁免的**边界**（防止被读成放宽判据）：

- 三个提交的 message **不含**任何 `Closes`/`Fixes`/`Resolves` 关键字（grep 核实 0 命中），
  因此不存在意外关闭或假装关闭；`#69` 仅作 subject 标签，指向不存在的编号；
- 结论以 `docs/R7_69_BUCKET_EXPANSION.md` 与 `docs/decisions/0010-*` 为准，**均在版本控制内**
  （R-037），不依赖 issue 状态；
- **豁免范围仅限 issue 追踪**。34 条阻断规则、全量 pytest、协议先冻结并 digest 校验、
  #60 比较器唯一口径、`failed-no-fallback` 留痕等**一律未放宽**；
- 需要 issue 追踪时，请用户在 ZCode 之外的终端用已认证通道创建并挂 `Closes #N`。

## Consequences（后果）

- **代价**：本阶段在 GitHub issue 面板上**没有**对应条目，仅存在于版本控制的文档中。
  对以 issue 为入口的外部读者，追踪入口缺失；需靠 `docs/R7_69_BUCKET_EXPANSION.md` §12 索引。
- **放弃的选项**：不使用任何未授权的凭据来源（例如从别处寻找 PAT 或改写远端配置）——
  那会越过「凭据只从环境变量或密钥服务读取」的约束，且属于本文件范围外的资源获取。
- **复核触发**：当环境中出现 `GITHUB_TOKEN` 等已认证通道、或用户安装并登录 `gh` 时，
  本条豁免自动失效，届时应补开 issue 并挂 closing keyword。
- 本决策**不**声称项目安全，也**不**改变任何科学结论；`docs/R7_69_BUCKET_EXPANSION.md`
  §12.1 与本文档互为引用。
