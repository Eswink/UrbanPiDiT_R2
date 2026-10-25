# 0059 — S4 首次确认的 2016 实例：store 契约无法同时表达「2016 目标帧」与「2017–2021 归一化」

- **日期**：2026-10-26
- **状态**：accepted
- **范围**：`docs/goals/s3-matched-exposure-reeval.md` §4.1（Track 0）的实例构造机制；
  不改任何科学门、不改 store 契约本身、不改已冻结判据

## Context

目标长文 §4.1 把 Track 0 的实例写成：**新建排他目录**，放 ① 只含 2016 帧的小 store
（**装入 FY candidate 的 2017–2021 `normalization_mean/std`**）；② **逐字节复制** FY 该臂的
`train.jsonl` 作同级文件（理由是身份门 `training/r7_evaluate.py:175` 比的是**同级 `train.jsonl`**
的 `dataset_identity`）；③ 指向小 store 的 `test.jsonl`（2016）。两臂各一份。

逐条核对本仓契约后发现**这三样不能同时成立**：

1. **`dataset_identity` 绑的不只是清单字节，还有 store 的内容指纹。**
   `training/r7_experiment.py:89-103`：`manifest_sha256`（清单文件字节）之外，还对**每个
   `store_path`** 记 `attributes`（全 attrs 字典）、`state` 形状、以及 `latitude` / `longitude` /
   `time_ns` / `normalization_mean` / `normalization_std` 的**字节 SHA256**。
   ⇒ 逐字节复制 `train.jsonl` 只锁住 `manifest_sha256`；它引用的 store 若换成「2016 帧」的 store，
   `time_ns` 与 `state` 形状必然不同 ⇒ `dataset_identity` 必然不等于 checkpoint 记录的
   `data_identity`，身份门**照旧会拒**。§4.1 括号里「逐字节复制才恰好等于」这一步**不充分**。

2. **store 契约把「归一化拟合年份」与「训练年份」绑死。**
   `data/r7_store.py:145-146`：`set(attrs['normalization_years']) != splits['train']` ⇒ `raise`。
   而 `data/preprocess/contracts.py:63-64` 要求 `max(train) < min(val) < min(test)`。
   若要让 2016 帧在某个 held-out split 下被选中，2016 必须落在 `val` 或 `test`
   （`data/r7_evaluation.py:26-27` 只接受 val/test），于是所有 train/val 年份都必须 **< 2016**；
   而 FY 归一化（`normalization_years=[2017..2021]`，实测 `outputs/r7_s3_fy_annual_20261021_attempt01/store/cache.zarr`
   的 attrs）要求 train = 2017–2021。两者**在同一份 attrs 里不可同时为真**。
   ⇒ 「2016 目标帧 + 2017–2021 归一化」这个组合，在冻结的 store 契约里**没有诚实表示**。

3. **窗口归属校验同样按年份。**
   `data/r7_store.py:179-183`（year 模式）对 `record['split']` 的声明年份做归属校验；
   `data/r7_evaluation.py:48-51` 的 `permits` 也按 `times[position].year in split_years[split]`。
   2016 的记录在任何 FY 声明的 split 下都过不了。

**因此 §4.1 的字面机制要么必须伪造 `normalization_years`（把「2017–2021 拟合」的数组标成别的年份），
要么必须改 store 契约。两条都被硬约束禁止**（「禁止绕过 checkpoint / 数据身份校验」、
「禁止事后放宽已冻结判据」）。这不是预算或算力问题，是**契约表达力**问题。

**补充实测（同日）：**「小 store」也**不能由标准构建器产出**。`data/preprocess/r7_era5_zarr.py:166-167`
要求 `any(not rows for rows in records.values())` 为假，即 **train/val/test 三个 split 各自都要有
≥1 个完整精确时间窗**；只含 2016 帧的源配上任何「2016 是 held-out」的 `split_years`，val/test
必然为空 ⇒ 构建器直接 `raise`。要造「2016 是 held-out」的 store 只能**手工装配**（绕过构建器的
全部校验与 `BUILD_COMPLETE` 语义），这正是要避免的。

## Decision

1. **不新建 store，不伪造任何 store 属性，不改 store 契约。** Track 0 的 2016 实例改用
   **「显式 init-time 集合 + 只读复用已冻结 store」**：目标帧直接取自**多年 store**
   `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`（`[34084,17,65,65]`，2016 完整在位、
   身份已冻结），由清单显式列出 2016 的 init 时间；**不复制、不切片、不重建**任何 store。
   这样 2016 帧由多年 store 的身份（`state` 形状 + `time_ns` + attrs 摘要）钉住，
   与「新建一个只含 2016 帧的 store」在**帧集合**上等价，但少一个可能漂移的派生物。

2. **身份门照真校验，不绕过。** Track 0 的评分驱动**显式**断言每个 checkpoint 的
   `contract['data_identity']` 等于 `dataset_identity(<FY 该臂 train.jsonl>)`——即在 FY 目录里、
   store 仍是 FY store 时算出的那个值（treatment `c17494ab63…`、control `0469312978…`）。
   这条断言是**真**校验，比 §4.1 的「同级复制」更强，因为它同时绑住了 store 的
   `time_ns` / 形状 / 归一化数组。

3. **窗口选择不经过 split 归属。** 评分驱动按 **2016 的显式 `init_time` 集合**（由清单给出）
   取窗，并**保留 `validate_record` 除「跨 split 年份」外的全部完整性校验**（索引在界、
   时间戳与 store 逐位一致、`init_time` 等于最后一个历史步、lead 与时间戳一致、等间隔）。
   被跳过的**只有**那一条无法表达的年份归属子句，且必须在协议里逐字声明。

4. **保留 `evaluate_local` 的其余一切**：模型加载、`make_model`、rollout、`RolloutRMSEAccumulator`
   / `RolloutClimatologySkillAccumulator`、物理单位、气候态分母（FY 的 48 桶 `reference.json`，
   实测 sha256 `4c89c70937cd848682fdfefd216fd77e1d648cfb0292c416e87e2d5d3aeb2135`）。
   评分驱动只替换**实例装配**，不替换统计或指标定义。

## Consequences

**变容易的：** Track 0 有了一条**诚实且可核查**的路径——所有 attrs 对各自的数组为真，
身份门是真校验，被跳过的唯一子句被逐字点名；同时把「§4.1 字面机制不可行」这件事**留痕**，
避免后续轮次（或 workflow 的侦察/实现子代理）照着一个不成立的机制去写驱动、白烧预算。

**变难 / 代价（如实列出）：**
- **偏离了目标长文 §4.1 的字面形状**：不再有「2016 小 store」与「同级 `train.jsonl` 让身份门自然放行」
  这两步，取而代之的是（i）2016 帧取自多年 store、由显式 init-time 集合钉住；（ii）驱动里一条
  **显式身份断言**。§4.1 需要一处更正（已记入 §8），后人读到两份文档时必须知道以本条为准。
- **D1 的验收形态变了**：不再是「小 store + `BUILD_COMPLETE` + 两臂 `dataset_identity` 等于 checkpoint 的
  `data_identity`」（那是 §4.1 的形态），而是「实例清单（2016 init 时间集）+ 显式身份断言通过 +
  反证测试证明身份不符时确实 `raise`」。**没有** `BUILD_COMPLETE` 这一层保护，改由清单与 store 的
  身份摘要共同承担。
- **多年 store 的 `split_years` 不描述 2016 是 held-out**（它声明 train 2000–2021 / val 2022 / test 2023），
  所以**不能**用 `ZarrRolloutDataset` / `EvaluationPreparation` 的 split 门去选 2016 窗
  （`permits` 会把 2016 全判为不属 test）。窗口选择必须在驱动里按显式 init-time 集合做——
  这既是限制，也是上面「跳过唯一一条无法表达的年份归属子句」的具体落点。

**备选方案与否决理由：**
- **伪造 `normalization_years`**（把 FY 数组标成 2016 拟合）：被「禁止绕过/伪造」直接禁止，否决。
- **改 store 契约**（放宽 `normalization_years == train years` 或改 split 排序）：会动到
  已冻结的共享契约与全部既有 store，且正是「事后放宽判据」，否决。
- **把 2016 帧的 `time_ns` 改成 2023**：伪造数据，否决。
- **不做 2016 实例，改用 2023 四季块 test**：达不到协议
  `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111`「至少覆盖一个完整未见年份」的最终正门，
  目标长文已明确不接受，否决。
