# 目标：S3 城市静态协变量接线轮（通路 + 身份绑定，不出科学结论）

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-static-covariates-wiring", "driver": "scripts/study_r7_static_covariates_wiring.py", "output_stem": "r7_s3_static_covariates_wiring_20261029", "control_record": "s3-urban-static-covariates-acquisition", "seeds": [41], "device_mode": "dual", "test_path": ["tests/test_r7_static_covariates_wiring.py"], "planned_seconds": 7200.0, "hard_cap_seconds": 14400.0, "code_archive": "outputs/r7_s3_static_covariates_wiring_code.zip", "commit_paths": ["scripts/study_r7_static_covariates_wiring.py", "scripts/r7_static_wiring_reference.py", "scripts/r7_static_wiring_support.py", "model/static_conditioning_r7.py", "model/weather_forecaster_r7.py", "model/process_forecast_r7.py", "tests/test_r7_static_covariates_wiring.py", "tests/test_r7_static_wiring_reference.py", "docs/goals/s3-urban-static-wiring.md", "docs/goals/s3-urban-static-covariates.md", "docs/goals/main-model-climatology-campaign.md", "docs/decisions/0069-static-covariate-wiring.md", "docs/decisions/README.md"], "commit_message": "feat(r7): wire the urban static covariates into the trunk behind an off-by-default switch", "register_paths": ["docs/R7_STATIC_COVARIATES_WIRING.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-urban-static-wiring.md"], "register_message": "docs(r7): register the static-covariate wiring round as an engineering evidence record"} -->

本文件是 goal 模式的目标长文源。objective 见 §0；交付物见 §2；判据见 §3；
实施顺序见 §4；预算与停止见 §5；明确不做见 §6；进度块见 §7。
**机器可读的 round spec 就是本文件顶部的 `round-spec` 块**，
`campaign-state.current_round_goal` 指向本文件。

**状态：prepared（2026-10-29 建立）。** 授权见
`docs/decisions/0069-static-covariate-wiring.md`；数据来源是已发布并登记的
`outputs/r7_urban_static_covariates_attempt01/published/`
（`static_identity = 8671ed7c…`，证据页 `docs/R7_URBAN_STATIC_COVARIATES.md`，
record `s3-urban-static-covariates-acquisition`）。`scientific_claim: false`、`test_read: false`。

**这一轮只买"通路存在且可审计"，不买"静态量有用"。** 任何 `on` vs `off` 的读数只作为
下一轮预注册的输入。

---

## 0. objective（可直接粘贴，单段，实测 1357 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning执行目标长文docs/goals/s3-urban-static-wiring.md（先完整读它，判据与出口以它为准；授权见决策0069）。任务：把已发布的城市静态协变量接进主模型，只做通路与身份绑定，不做科学结论。四个预声明出口全为工程断言：(W1)把static_identity改一个字符，protocol_sha256必须变化，否则身份绑定是假的；(W2)model/**改动前后在同一seed同一固定输入上必须得到同一组指纹——初始参数摘要、一个update后的loss与参数摘要三者逐位相等，参照读数必须在改model之前落盘（用scripts/r7_static_wiring_reference.py，已在outputs/r7_static_wiring_20261029/pre_change_reference.json）；(W3)on臂从seed初始化跑有界updates并在val上产出5个lead的boundary_rmse.csv，行数与off臂逐行相同；(W4)抽样65×65格上与已发布static_fields.npz逐格float32位相等，证明读进来的确实是发布的那份。实现=model/static_conditioning_r7.py新增模块+weather_forecaster_r7/process_forecast_r7两个开关参数，默认off且off路径不构造任何模块；static_identity必须绑进protocol_sha256而不是data_identity（决策0063第3条），data_identity保持不变。同时执行文件清单必须具名（决策0067，用tools/r7_explicit_inventory.py的import闭包），并含"新增无关.py不改变清单"的反证。预算：记录本轮实际GPU-h、网络0、磁盘、墙钟与overrun；planned7200s/hard14400s（1:2），软超继续并记overrun，只有硬上限因时长截断，不在运行中改硬限。单seed×两臂是通路验证，不得当作种子一致性证据，也不得用on/off差异下科学结论。禁止：改冻结判据、事后放宽出口、伪造PASS、删弱测试、绕身份、把static_identity写进data_identity或store attrs、把静态量做成第18个state通道、重建store、重跑数据轮、改已发布产物、读2023test、动data/raw|interim|processed与归档快照、main合并/force/merge/关issue、cron/守护/会话外续跑。GPU默认共驻，spawn前只读核余量，不对邻居进程发任何信号；不paid/rental/exclusive。只在三种情况停：触保留授权边界、达已冻结停止条件、确无可安全独立推进的工作；停时必须在§7写"为何停+恢复第一条命令"。每步给可核查证据（命令+输出+digest），估计值不得写成实测值；结束按AGENTS.md汇报改动/验证/接口/数据影响/兼容性/依赖/未做/下一项。不要自行宣布目标完成。

---

## 1. 现状与前置

- **节点**：S3。上一轮是数据相位（`docs/goals/s3-urban-static-covariates.md`，0 GPU-h，
  六个子命令已跑完并发布 `attempt01`）。本轮是它的**模型侧**后续，决策 0063 第 3 条
  明确要求"接线需自己的 ADR 与预声明出口"，即决策 0069。
- **数据已就绪且身份可核查（实测）**：
  ```
  outputs/r7_urban_static_covariates_attempt01/published/manifest/BUILD_COMPLETE.json
    {"schema_version": 1, "build_complete": true}
  published/manifest/manifest.json → identity_components
    static_identity: 8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03
    output_sha256:   1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5
    protocol_sha256: 8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904
    recheck_max_relative_delta: 5.1535351518462713e-08
  ```
  `published/fields/static_fields.npz` 36 个数组、全部 `(65,65)` float32（除
  `built_surface_frac`/`pop_density_km2` 为 `(5,65,65)` 多期与两个坐标轴）。
- **为什么不能走"第 18 个通道"**：写进 store `attrs` 会改 `data_identity`，把本轮与
  R1–R5 已登记队列**脱钩**；不写进 `attrs` 则两份不同静态场会得到同一个 `data_identity`
  ——那是审计洞。⇒ 独立产物 + `static_identity` 绑 `protocol_sha256`。
- **一个必须先说的代价（实测依据）**：本轮**必然**改动 `model/**` 字节
  （`model_code_sha256` 现为 `60de3b2d…`，`training/r7_experiment.py:31-39` 对
  `model/*.py` 做哈希），而 `training/r7_experiment.py:124-134` 的 `load_checkpoint`
  按设计拒绝外来实现 ⇒ **所有现存迁移父（`60de3b2d…` 下产出）在本轮代码下不可加载**。
  受影响的后续轮次：R2–R5（都经 `scripts/r7_s3_fy_annual_parent.pinned_parent`）。
  解法是重跑一次 `scripts/migrate_r7_parent_state.py --out <新目录>`（model-only 安装，
  六个源 pin 与 `state_digest` 双向核对不变）；那是**下一次身份操作**，不属本轮，
  但必须在本轮登记里写明，否则 R2 会在开轮时才发现阻塞。
- **R1 的 `check_model_revision` 会因此失败**，这是预期的：R1 已完成并登记，
  它的证据页与 `readings.json` 是冻结的历史，不重跑。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | `model/static_conditioning_r7.py` 与两个开关参数 | 默认 off、off 路径不构造模块；`parameter_delta` 显式申报 |
| D2 | 静态场读取与身份校验 | 读 `.npz` 前核 `static_identity`/`manifest_sha256`/`BUILD_COMPLETE`，不符即抛 |
| D3 | `protocol.json` 的 `static` 段 | `static_identity` 进 `canonical_digest`，`data_identity` 不变 |
| D4 | W1–W4 四个出口的实测读数 | 逐条命令 + 输出 + digest |
| D5 | 有界两臂运行产物 | 每臂 5 lead 的 `boundary_rmse.csv`，行数逐行相同 |
| D6 | 证据页 `docs/R7_STATIC_COVARIATES_WIRING.md` | 出口表、参数增量、代价与限制 |
| D7 | 四件套登记 | 主计划 §8 + §7 账本行 + index record + `campaign-state` |

## 3. 判据与证据来源

**本轮的科学判据为空。** 唯一判据是决策 0069 的四个机关出口，逐条写在本文件 §4.2。
引用来源：

- 授权与边界：`docs/decisions/0069-static-covariate-wiring.md`、决策 0063 第 3 条；
- 数据身份：`outputs/r7_urban_static_covariates_attempt01/published/manifest/manifest.json`、
  `docs/R7_URBAN_STATIC_COVARIATES.md`；
- 身份纪律：`AGENTS.md`「禁止绕过 checkpoint / 数据身份校验」、
  `training/r7_experiment.py`（`model_code_digest`/`load_checkpoint`）、
  `training/r7_autoregressive_runner.py`（`_module_semantics`）；
- 执行文件清单：决策 0067、`tools/r7_explicit_inventory.py`；
- 机械门：`tools/check_conventions.py`（37 条阻断）、`tools/check_goal_brief.py`、
  `tools/check_campaign_state.py`、`tools/verify_r7_evidence_index.py`、
  `tools/r7_round.py round-guard|verify`。

**不算通过的**：skip / cancelled / queued / 硬上限截断；`report()` 与 CI 绿都不是结论；
"看起来一样"不是逐位相同；单 seed 不是种子一致性证据。

## 4. 实施顺序

### 4.1 顺序（哪一步先做，本身是证据）

1. **先落参照读数**（W2 的前提），已做：
   `scripts/r7_static_wiring_reference.py --out outputs/r7_static_wiring_20261029/pre_change_reference.json`
   → `model_code_sha256 60de3b2d…`、`initial 511da521b55c8194`、`loss 0.01961401104927063`、
   文件 sha256 `4d0dc228699ae374…`；同命令跑第二次 `values` 完全相同（确定性已验）。
   **必须在改 `model/**` 之前完成**，事后补的参照不算。
2. **改 `model/**`**：新增 `model/static_conditioning_r7.py`；在
   `model/weather_forecaster_r7.py` 与 `model/process_forecast_r7.py` 各加一个
   `static_conditioning: bool = False` 透传参数并纳入 `_validate_switches` 的布尔校验。
3. **改完立刻跑 W2 比对**：`--compare` 必须报 `differences: []`。
4. **驱动**：`scripts/study_r7_static_covariates_wiring.py`，
   冻结协议（`static` 段进 `protocol_sha256`）→ 有界两臂运行 → 5 lead 评分 → 出口读数。
5. **提交前** `tools/r7_round.py commit-guard`；**登记后** `tools/r7_round.py verify`。

### 4.2 预声明出口（本轮不看读数就已冻结）

| 出口 | 判据 | 性质 |
| --- | --- | --- |
| **(W1)** | 把 `static_identity` 改 1 个字符 ⇒ `protocol_sha256` 必变；且 `data_identity` **不变** | 身份绑定为真 |
| **(W2)** | 改 `model/**` 前后同 seed 同固定输入的 `initial_parameter_digest`、`first_update_loss`、`post_update_parameter_digest` 三者**逐位相等** | 无副作用 |
| **(W3)** | `on` 臂从 seed 初始化跑有界 updates，val 上产出 5 lead 的 `boundary_rmse.csv`，与 `off` 臂**行数逐行相同**、全部有限 | 通路可用 |
| **(W4)** | 抽样格上读进的静态场与 `published/fields/static_fields.npz` **float32 位相等**（容差 0） | 读的是发布的那份 |

**任一为假 ⇒ 本轮记 `negative`，不推进到任何 on/off 比较，也不重试到偶然通过。**
四个全真 ⇒ 记 `engineering-positive`，**仍不宣称静态协变量有用**。

### 4.3 参数增量申报（必须实测，不得估算冒充）

`on` 路径的注入形态与 `parameter_delta` 在实现时确定并写进协议；若为 0 必须说明
是"调制"而非"额外层"。tendency 轮的先例（`TENDENCY_PARAMETERS = 13056`，
`channels*dim*patch_size^2`）说明本项目要求**公式 + 实测**两处一致。

## 5. 预算与停止条件

- **GPU**：两臂 × 1 seed，有界 updates。名义 **planned 7,200 s（2 h）/ hard 14,400 s（4 h）**
  （1:2）。软超继续并记 `soft_overrun_seconds`；只有硬上限因时长截断，记
  `budget_limited` 并全额记账。**不在运行中改硬限。**
- **网络**：0（静态场已在本地，不下载）。
- **磁盘**：本轮产物只有协议、两臂 checkpoint、CSV 与读数；不新增数据集。
- **GPU 共驻**：spawn 前只读核余量（R-054），不对邻居进程发任何信号。
- **停止条件（仅三种，决策 0043）**：触保留授权边界；达已冻结停止条件（任一出口为假即停）；
  确无可安全独立推进的工作。停下必须在 §7 写「为何停 + 恢复第一条命令」。

## 6. 明确不做

- **不宣称**静态协变量改善天气、不自宣 SOTA、不关闭任何轴、不把这轮当 on/off 科学比较。
- **不**把 `static_identity` 写进 `data_identity` 或 store `attrs`；**不**加第 18 个 state 通道；
  **不**重建 store；**不**重跑数据轮；**不**改已发布产物。
- **不**重跑父迁移（`migrate_r7_parent_state.py`）：那是下一次独立的身份操作，
  本轮只在登记里写明它是 R2–R5 的解阻塞动作。
- **不**读 2023 test；**不**动 `data/raw|interim|processed` 与归档快照。
- **不**改任何冻结判据、**不**事后放宽出口、**不**删弱测试、**不**绕身份。
- **不** main 合并 / force / merge / 关 issue；**不** cron / 守护 / 会话外续跑。

## 7. 进度块（状态机）

- **状态**：**已执行并登记**（2026-10-29；`scientific_claim: false`、0.9020 GPU-h）。
  最终 goal 状态不由本文件宣布。
- **已完成（本轮实测）**：
  - 决策 `0069-static-covariate-wiring.md`（proposed）；本长文的 `round-spec` 已解析、
    `campaign-state` 已指向本文件、`previous_round_*` 指向数据轮与其证据页。
  - **W2 参照读数先于改动落盘**：`outputs/r7_static_wiring_20261029/pre_change_reference.json`
    （`model_code_sha256 60de3b2d…`、`first_update_loss 0.01961401104927063`、
    `model_dirty: false`、`head 5e4adf9f6ad9`、`sha256 4d0dc228…`）；同命令第二次 `values`
    完全相同 ⇒ 参照本身可复现。
  - **实现**：`model/static_conditioning_r7.py` + 两处透传；R-052 靠**抽取**
    `require_read_field_mode` 守住（未放宽阈值）；执行清单 **94 个具名文件**（决策 0067）。
  - **attempt01（失败，0 GPU-h）**：prepare 507.58 s，W1/W2 通过、W4 因探针缺陷必假
    （float64 重建 vs float32 输出，差 1 ulp＝7.15e-07）；保留 `FAILED_ATTEMPT.json`。
  - **attempt02（成功）**：prepare 512.26 s；seed 阶段 3,247.23 s；
    `protocol_sha256 48c09621e2fb…`；**W1/W2/W3/W4 全真**，`verdict: engineering-positive`。
  - **证据页**：`docs/R7_STATIC_COVARIATES_WIRING.md`；index record
    `s3-urban-static-covariates-wiring`；账本 `0.9020` GPU-h（累计 `323.940495`）。
  - 提交：实现 `896eabf`、探针修正 `67ebf14`、登记 `3e14eec`。
- **未做**：
  - **没有**做 on/off 的科学比较（单 seed、120 updates、10 样例/lead）；§3 的 Δ 只记录。
  - **没有**重跑父迁移。本轮把 `model_code_sha256` 从 `60de3b2d…` 改成 `a38be16c…`，
    所以**所有已登记迁移父在本代码下不可加载**；R2–R5 的**前置**就是这一次迁移，
    它不属本轮，也没有被本轮跳过而不记账——它是 §8 的下一动作。
  - **没有**用时变静态场（多期数组留在产物里）；**没有**引入可学习静态编码器
    （会引入容量混杂，需要自己的 ADR）。
  - **没有**读 2023 test；`r` 仍为 0。
- **下一动作（节点 S3；可执行的第一条命令）**：
  `.venv/bin/python tools/r7_round.py state`
  —— 核对指针；随后是 **R2–R5 的父迁移解阻塞**（见主计划 §8 2026-10-29 块）：
  `.venv/bin/python scripts/migrate_r7_parent_state.py --export-root outputs/r7_s3_rollout_dose_parent_export_20261009_attempt02 --out outputs/r7_s3_parent_migrated_20261029_attempt01`
