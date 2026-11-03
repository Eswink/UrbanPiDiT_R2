# 0082 — R4 的桥 (e)④ 改写：零 query 的聚合**不可能逐位等于** `value.mean(dim=1)`，判据改为「权重精确均匀 + 聚合在 float32 累加界内」

- **日期**：2026-11-03
- **状态**：accepted
- **范围**：`docs/decisions/0080-...` §Decision 4 的桥 (e) 第 ④ 条与
  `docs/goals/s3-position-retention-reeval.md` §3 的对应文字；**替换一条判据的表述与它的通过条件**，
  0080 的路线②/侧车/`position_scale` 字段、0081 的常数 8000、以及 (a)–(d) 四个出口**全部不变**
- **依据**：本会话实测（下 §Context ②）；`model/typed_evidence_r7.py:246-257`
  的 `_position_attention` 与它自己的 docstring；本轮**尚未冻结任何协议**（`outputs/` 下只有父迁移目录，
  无 `preparation_protocol.json` / `protocol.json`）

## Context

**① 要改的是哪一句。** 0080 §Decision 4 与 brief §3 把桥 (e) 的第四条写成：

> C 的 query 清零后聚合**逐位等于** B 的均值（`zero_query_max_abs == 0`）

**② 这句话在数学上不可满足——不是「可能偏一点」，是必然非零。**
两条路径算的不是同一个 float32 程序：

- `_position_attention` 是 `sum_i w_i * v_i`，`w_i = softmax(0)_i`（精确等于 `1/P` 的 float32 邻居）；
- `value.mean(dim=1)` 是 `sum_i v_i / P`。

数学上相等，浮点上「先乘再加」与「先加再除」不是同一串运算。实测（`P=1089`，`dim=192`，
`torch.randn` 输入的 64 个不同种子 × batch 2）：

| 量 | 实测 |
| --- | --- |
| `softmax(zeros)` 是否逐位等于 `full(1/P)` | **是**（权重本身精确均匀，`1/P` 的 float32 值完全一致） |
| `max abs(attn − mean)` | **2.9802e-08**（64 seed 最坏值） |
| 该值 / `(eps_f32 · max|v|)` | `5.2e-02` |
| 该值 / `(2·P·eps_f32 · max|v|)` | `2.4e-05` |
| `value.mean(1)` 是否逐位等于 `value.sum(1)/P` | **是**（所以差异来自乘 `w` 与除以 `P`，不是来自 `mean` 本身） |

**③ 原判据会造成什么后果。** 驱动里它写成 `if difference != 0.0: raise RuntimeError(...)`，
并且在 `_seed` 里**每个 seed 训练前**都跑一次。也就是说：这个 bug 不是「读数有微小偏差」，
而是**开跑后第一个 seed 立刻 `raise`**，整轮 0 产出。本轮**尚未冻结、尚未开跑**，
因此在冻结前更正，不构成「事后放宽已冻结判据」。

**④ 为什么这不是在弱化判据。** 原来那条断言其实**什么都没测到**：它在测一个不可达的等式，
唯一的可能结果是永远失败。真正有内容的两个断言是分开的、都可精确判定：

- **权重**：`|q|=0` ⇒ 每个 score 精确为 0 ⇒ `softmax` **逐位**等于 `1/P`。这是精确的，已实测。
- **聚合**：在上述精确均匀权重下，`attn` 与 `mean` 的差落在 float32 累加界内。界取
  `2·P·eps·max|v|`，实测占用 ≤ `2.4e-05`，余量 4 个数量级。

## Decision

1. **桥 (e)④ 改写为两条，都保留在驱动里、都逐 seed 断言**：
   (i) `uniform_weights_exact`：`|q|=0` 时 softmax 权重**逐位**等于 `1/P`（精确，`raise` 若不符）；
   (ii) `zero_query_is_pooled`：`max|attn − mean| ≤ 2·P·eps_f32·max|v|`（`raise` 若越界），
   并把**实测的** `zero_query_max_abs`、所比对的 `zero_query_bound`、以及占用比例
   `zero_query_bound_ratio` 一并写进 receipt 与 `readings.json`。
2. **`zero_query_max_abs` 从「必须为 0 的判据」降为「记录下来的测量值」**；通过条件是上面的两条。
   任何后续引用都必须同时写实测值与界，不得单独引用 `max_abs` 当作「偏差」。
3. **桥仍然不是科学判定**（0080 §Decision 4 不变）：它失败只记 `identity/engineering unresolved`，
   不进入 (a)–(d)。
4. **`REQUIRE_ZERO_EXACT_ABS` 这个语义不再存在**；本轮不设「必须为 0」的隐藏阈值。
5. **不做**：不因为这个 bug 改动 `model/`（`_position_attention` 的实现是对的，它的 docstring
   早就写明「up to float summation order」——是 0080 的**判据文字**写错了，不是实现错了）；
   不动 0081 的常数；不动 (a)–(d)。

## Consequences

**变容易的：**
- 桥 (e)④ 现在**真的可跑、可判**，而且比原来更严：原来只有一条不可达的等式，现在多了一条
  **精确**的权重断言（原来那条隐含地假设了权重均匀但从未断言它）。
- 驱动里 `${...}_zero_query_is_pooled` 与 receipt 字段名保持不变，历史可比性不破。

**变难 / 代价（如实列出）：**
- **`zero_query_max_abs` 不再是 0**，读报告的人必须知道这是 float32 累加的下界效应而不是模型差异。
  本条 ADR 就是那个说明；receipt 里同时存 `bound` 与 `ratio`，让读者不必相信文字。
- **界是我选的**（`2·P·eps·max|v|`），不是从某篇文献推的。选它的依据是实测占用 ≤ `2.4e-05`
  且形式可复算；若将来有人要更紧的界，那是另一个判据，须另立 ADR。
- **这只修了「不可达」这一种缺陷**：桥 (e) 的另外三条（零起点、`pathway_init_sha256` 相同、
  `position_scale` 相同）本来就是精确等式，不受影响，也没有被本次改动放松。

**备选方案与否决理由：**
- **备选 A：保留 `== 0`，把 `raise` 改成警告。** 否决：那等于把一条**永远失败**的判据
  降级成噪音，桥就变成不可判定的；而且「零起点 ⇒ 等于均值」这个真正的恒等关系就没人验了。
- **备选 B：把 `_position_attention` 改成在 `|q|=0` 时走 `value.mean(dim=1)` 分支。**
  否决：那会为了迁就一句判据文字而改 `model/` 的前向路径，代价是 `model_code_sha256` 再动一次、
  父钉再重绑一次，而且会在实现里留一个「只有标量恰好为零才生效」的分支——正是 0079 反对的
  隐藏自由度的形状。
- **备选 C：只放宽到 `torch.allclose` 默认容差。** 否决：默认容差是相对量、随输入尺度漂，
  而这里的量纲是「绝对误差 ≤ 累加下界」，用相对容差会把大值处的越界当成通过。
