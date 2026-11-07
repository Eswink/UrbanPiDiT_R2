# 0098 — C2 `attempt04` 主动终止：读数阶段三处必然失败，继续跑只能得到零读数

- **日期**：2026-11-08
- **状态**：accepted
- **范围**：Stage C2（`docs/goals/s3-throughput-physics-c2.md`）的 `attempt04`；
  只涉及**读数/评估路径**，不涉及训练循环、判据或任何冻结量
- **依据**：`AGENTS.md`「禁止未实跑就写 PASS」「skip 不算通过」、
  目标长文「每步给可核查证据」「估计值不得写成实测值」；
  2026-11-08 第三次 CI 红的教训（决策 0096 的 brief 里已记「先动手、再验证」的失误）

## Context

**① 发现经过。** `attempt04` 训练健康（两卡各一 seed 的 `off` 臂跑到 **32%**，
1.136/1.147 s/update，loss 0.63/0.99）。在等待期间我**只读**审读
`scripts/study_r7_s3_derivative_matching_c2.py` 的 `readings()` 与 `evaluate_exit()`，
准备登记时按这些函数实际会做什么逐条推演，发现**三处独立缺陷**，
它们都在**十小时训练之后**才会触发。

**② 三处缺陷，全部实跑复现，不是读代码猜的。**

| # | 缺陷 | 复现方式 | 实测结果 |
| --- | --- | --- | --- |
| 1 | `readings()` 与 `_train_arm()` 都写 `curve.EVALUATION_LEADS`，而该名字**只存在于** `study_r7_s3_wide_single_factor`；`r7_s3_curve_point_large_protocol` 把它 import 成 `wide`，**从不重新绑定** | 直接取属性 | `AttributeError: module 'scripts.r7_s3_curve_point_large_protocol' has no attribute 'EVALUATION_LEADS'` |
| 2 | `_train_arm` 的**第 138 行**同样写 `curve.EVALUATION_LEADS`——即在**训练跑完之后、评分之前** | 同上 | 同上（同一名字，两处调用点） |
| 3 | 填充 `entry[arm]["rmse"]/["skill"]` 的四行被**缩进在 `if arm == CANDIDATE:` 内部**，于是 `off` 臂的 `rmse`/`skill` 永远是空字典 | 用合成的两臂结果调 `evaluate_exit` | `KeyError: '72'` |

**③ 为什么这三处必须停机而不是「跑完再修」。**
`attempt04` 的 `execution_files_sha256` 把**驱动与协议层**都钉住了（94 项）。三处缺陷
全在这两个文件里，所以它们在跑的这一份代码上**不可能通过补丁修好**——
`validate_protocol` 会逐文件校验，改一个字节就让整轮身份失效。**剩下的选择只有两个**：

- 让两卡各跑完两个臂（`off` 剩约 7 h、`on` 再约 10 h），然后在 `reading` 阶段
  拿到 `AttributeError`，**零读数、全额记账约 61 GPU-h**；
- 现在停下（已用 **6.945 GPU-h**），修好后重开。

**④ 已用算力如实记账。** 从 freeze 到终止共 **3.472 h 墙钟 × 2 卡 = 6.945 GPU-h**。
这 6.945 GPU-h **不产生任何读数**，按 `AGENTS.md` 全额记入账本，
且**不得**记成 `attempt04` 的「部分成功」。两个 `off` 臂的 `progress.jsonl` 与
checkpoint 保留在原地作为留痕，**不作为任何结论的证据**。

**⑤ 这三处为什么没有被任何测试拦住——这一点比缺陷本身更重要。**
现有 21 项测试里，`readings` / `evaluate_exit` **一项覆盖都没有**。
已有一条 `test_every_driver_global_resolves_before_a_gpu_stage_sees_it` 能抓
**全局名**（它正是为 `arm_spec()` 那个 `NameError` 加的），但
`curve.EVALUATION_LEADS` 是**属性访问**，反汇编看到的是 `LOAD_GLOBAL curve` +
`LOAD_ATTR EVALUATION_LEADS`——全局名存在，属性不存在，于是那条测试**通过**。
第 3 处是**缩进**，更不可能被「名字都能解析」这类检查发现。
⇒ 已有的那道网覆盖的是「名字」，缺陷落在**属性**与**控制流**两层，那是网的孔隙。

## Decision

1. **终止 `attempt04`，记 `attempt04` 作废、**6.945 GPU-h 全额记账、零读数**。**
   已做：向本实验自己的两个 `seed` worker 发送 `SIGTERM`（PID 1607930/1607931，
   先在 `/proc/<pid>/cmdline` 里逐个确认命令行含 `attempt04` 才发信号），
   随后确认 0 GPU 占用、`stage-seeds` 父进程同时退出。
   **未对任何非本实验进程发信号。**

2. **修复形态：只改读数路径，不动任何训练/判据/冻结量。**
   - `LEADS = wide.EVALUATION_LEADS` 成为驱动里的**单一具名常量**，
     替换两处 `curve.EVALUATION_LEADS`；
   - `curve.check_provenance` → `curve_driver.check_provenance`
     （该函数定义在 `r7_s3_curve_point_large_driver.py:96`，是**第三个**同类缺陷，
     同样只在评分时触发）；
   - `rmse`/`skill` 的填充移出 `if arm == CANDIDATE:`，回到**每臂都会执行**的位置；
   - 新增一条**断言**：每臂的 lead 集合必须**恰好等于**冻结的五个 lead
     （`sorted(int(k) ...) != sorted(LEADS)` 即 `RuntimeError`），
     使「某一档 lead 悄悄消失」在此处失败而不是把比较面悄悄变窄。

3. **测试必须先失败、后通过（反证），否则不算修好。**
   新增两项：
   - `test_reading_attributes_resolve_and_the_exit_consumes_both_arms`：
     在**合成的两 seed × 两臂 receipt 树**上真跑 `readings()`，断言两臂都拿到五个 lead
     的 `rmse`/`skill`、`off` 臂**不再为空**、`exit_conditions.all` 为真。
   - `test_the_fill_lives_in_the_lead_loop_not_the_candidate_branch`：
     结构性反证——填充的两个赋值必须**出现在 per-lead 循环里**，
     且 `if arm == CANDIDATE` 子树里**不得**含填充。
   **实测反证**：把驱动 `git stash` 回缺陷版本后，这两项**都失败**；
   恢复修复后 **23 项全过**。
   **第一版结构测试是假阴性**（它只断言候选分支下「没有」填充，而缺陷版本里
   整段嵌套块**就是**填充，于是它在缺陷代码上**通过**）——已改为断言**正面性质**，
   并如实记下这次假阴性。

4. **顺带把 R-021 计数从 136 还原到 135。** 决策 0096 的 brief 里记着
   「该文件在 C2 跑完后应缩回 ≤400 行、计数回到 135」。`attempt04` 终止后，
   「缩回会让在跑轮次身份失效」这条理由**消失**，于是
   `r7_s3_derivative_matching_c2_protocol.py` **404→400**
   （只压缩 docstring 与一处三行注释，**零逻辑改动**），
   新测试文件**恰好 399 行**。阈值/级别/冻结例外**一字未改**。
   `docs/rules/size-thresholds.md` 的 marker 与实测值现**一致**为 135。

5. **`attempt05` 必须重新冻结，不能复用 `attempt04` 的协议。**
   修复改了 `scripts/study_r7_s3_derivative_matching_c2.py` 的字节，
   `execution_files_sha256` 与 `training_code_sha256` 都会变；
   `attempt04` 的协议**永久作废**，不作为 `attempt05` 的任何输入。
   **`attempt05` 的 `prepare` 必须重跑 preflight**（它是冻结前置，不是可继承的测量）。

6. **本轮不宣布任何科学结论。** 三处缺陷都是**工程**缺陷；修好它们**不**意味着
   C2 的 (a)–(d) 会通过，也**不**意味着导数域匹配有效或无效。
   `scientific_claim: false` 不变。

## Consequences

**负面后果（代价，必须如实列出）**

1. **6.945 GPU-h 零读数，全额记账。** 这是本轮的实际代价。它买到的是一个
   「读数阶段根本跑不通」的事实——但这个事实**本来可以在 0 GPU-h 时**通过
   一次 `readings()` 的单元测试得到。**这笔钱是我该在冻结前花的，没有花。**
2. **`attempt04` 的两个 checkpoint 与 `progress.jsonl` 成为死产物。**
   它们保留在原地作为留痕，占据磁盘，且**不得**被当作任何结论的证据；
   它们也不会被 `attempt05` 复用（`reuse-plan` 只复用**完整**的 seed receipt）。
3. **第一个结构测试版本是假阴性，我误以为已覆盖。** 它是被「回退到缺陷代码再跑」
   这一步抓出来的——**如果只跑修复后的版本，这个假阴性会带着「已测试」的标签留下去**。
   已如实登记，并把「先让测试失败」写成第 3 条的形式要求。
4. **这类缺陷仍可能重现。** `curve.<attr>` 这种「模块 A 的常量经模块 B 转手」的写法
   在本仓多处存在；修复只处理了 C2 驱动里的三处，**没有**建通用守卫
   （例如一个「驱动用到的每个 `module.attr` 都必须存在」的静态检查）。
   这条留作未决项。
5. **R-021 回到 135 是「因为轮次被终止」而不是「因为工程变好」。**
   下一次 `attempt05` 冻结时会**重新**把 400 行的协议钉进 `execution_files_sha256`，
   同一个坑会**随每轮重开重现**。这不是一次性债务。
6. **终止本身有一个判断风险，必须写明**：我是在**推演**「读数会失败」之后停的，
   没有让一轮完整跑完去**实证** `AttributeError`。停机的依据是
   ① 两个 `AttributeError` 在**当前字节**上直接复现（不是推测），
   ② 第 3 处用合成结果复现 `KeyError`，③ 三处都在这两个被钉住的文件里，
   在当前身份下无法修补。**但「整轮跑到 `reading` 会怎样」本身没有被实测**——
   这一点如实记录，不把它说成已证实的运行期观测。

**正面后果**：在花掉约 61 GPU-h 之前停下，代价 6.945；读数路径现在有**两项**
能真正抓住缺陷的测试（已用「回退到缺陷版本必失败」验证过，不是自证）；
R-021 计数与实测重新一致；第三个同类缺陷（`check_provenance`）被一并发现并修掉，
而它原本也会在同一时刻炸。

**否决**：

- **让它跑完再修。** 否决：`validate_protocol` 逐文件校验 94 项，三处缺陷都在被钉文件里，
  跑完只会得到 `AttributeError` + 零读数 + 约 61 GPU-h 账单。
- **改 `execution_files_sha256` 去「修」在跑轮次的身份。** 否决：那是绕过身份校验，
  `AGENTS.md` 明令禁止（与 2026-11-08 那次「先改被钉文件、后验证」的失误同类）。
- **用 `hasattr` 兜底或 `try/except AttributeError` 让读数「不炸就行」。** 否决：
  那会把缺陷变成静默的**错误读数**（例如少的 lead 被跳过），比崩溃更糟。
- **只修第 1、2 处（`AttributeError`）而不修缩进。** 否决：那会让读数阶段
  过了属性检查、然后在 `evaluate_exit` 上以 `KeyError('72')` 失败——
  **同一笔钱再花一次**。
- **把这次终止记成 `attempt04` 的 `budget_limited`。** 否决：它不是时长截断，
  是主动作废；记账类别必须如实（作废 + 全额记账 + 零读数）。
- **复用 `attempt04` 的 `prepare` 结果给 `attempt05`。** 否决：preflight 是冻结前置，
  且代码字节已变，`execution_files_sha256` 必然不同。

## 未决（留给下一轮）

- 是否值得建一条通用静态守卫：解析驱动里每个 `module.attr`，断言属性存在
  （`test_every_driver_global_resolves...` 的**属性级**版本）。它能同时抓第 1/2/3 类中的两类。
- `attempt05` 的 `per_seed_seconds` 是否仍取 115,200 s（本轮实测 1.136/1.147 s/update，
  比 brief 的 1.1096 慢约 3%）；重算后若包络仍 > 2×估计则沿用。
- `attempt05` 是否需要把 `readings()` 的合成树测试**扩展到 b 项为假**的分支
  （当前只测了全真路径），以证明 (b) 真的会因为「未记录 constraint 项」而失败。
