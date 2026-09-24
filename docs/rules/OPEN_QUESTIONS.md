# 待决问题

需要人拍板的问题。每条给出：问题、现状证据、选项、各选项影响、我的建议。

**状态说明**：Q-001 – Q-008 已由项目所有人于 2026-09-24 决定并落地（见同目录
`CHANGELOG.md` 第二遍记录）。下面保留原始分析并标注决定结果，不删除推理过程。
Q-009 起为仍未决或本轮新识别的问题。

## Q-001 ✅ 已决：归档到根 `legacy_v6/`

**决定**：选项 A —— 新建根目录 `legacy_v6/` 并移入，沿用 `legacy_v531` 的 README + `pyproject.toml`
exclude + 只读约定。

**落地**：新增 `legacy_v6/README.md`；`training/legacy_physical_consistency.py` 移出（833 行内容
逐行比对一致，仅补结尾换行）；`pyproject.toml` exclude 增加 `legacy_v6*`；
`tools/check_conventions.py` 的 `ARCHIVAL_PREFIXES` 增加 `legacy_v6/`；`AGENTS.md` 硬约束同步。
R-021 命中 1 → 0。

<details>
<summary>原始分析</summary>

- **现状**：833 行（活跃区最大文件，R-021 的唯一违反），**未被任何模块导入**（E-084），
  依赖 `pytorch_lightning`（本机未装）。它是 V6 时期的物理一致性损失。
- **冲突**：它让 R-021 无法达标；但实验项目里"看起来死了"的代码常是方法学记录，
  而本项目已有 `legacy_v531_full/` 这一归档惯例。
- **选项 A**：移入 `legacy_v531_full/` 或新建 `legacy_v6/`，保留可查。
- **选项 B**：删除（git 历史仍在）。
- **选项 C**：保持原位，接受 R-021 长期为 C 类。
- **影响**：A 最符合本项目既有的归档习惯（有 `legacy_v531` 先例），成本低；
  B 会让 V6 对照材料离开工作树；C 让一条目标态指标长期挂红。
- **我的建议**：A。依据是本项目已经用 `data/legacy_v531/` + `model/legacy_v531/` + README 说明
  的方式处理过同类问题，沿用既有惯例比新建第三套做法好。

</details>

## Q-002 ✅ 已决：3 个保留在根目录 + 3 个移入 `scripts/`

**决定**：分组处理（选项 A 的变体）。关键是移动边界由**调用方式**决定，不由"是否被 git 引用"决定。

- **保留在根目录**：`diagnose_r7_gain.py`、`profile_r7_inference.py`、`tune_r7_halting.py`
  —— 它们被 3 个测试以**裸文件名 + `cwd=repo`** 调用（E-153），且 `profile_r7_inference.py`
  还被 `training/r7_baseline_study.py:83-84` 按路径调用，移走即破坏。已补进 `py-modules`。
- **移入 `scripts/`**：`train_r7.py`、`train_r7_process.py`、`train_r7_recursive.py`
  —— 无任何测试或脚本调用它们；移入后各补 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
  （沿用 `scripts/study_r7_cpu.py:7` 惯例，E-154），已用「从 `/tmp` 运行」验证路径修正生效。
- **保留**：`train.py`（README 引用它）。
- `AGENTS.md` 新增**入口点表**，写明哪些进 wheel、哪些须在仓库根运行。

**注意**：`train_r7{,_process,_recursive}.py` 移入 `scripts/` 后，其默认 `--config`
（`configs/...`）是相对 cwd 解析的，因此仍需**从仓库根**调用：
`python scripts/train_r7.py --config configs/...`。这与其它 `scripts/` 用法一致。

<details>
<summary>原始分析</summary>

- **现状**：`train.py`、`train_r7.py`、`train_r7_process.py`、`train_r7_recursive.py`、
  `diagnose_r7_gain.py`、`profile_r7_inference.py`、`tune_r7_halting.py` 不在
  `pyproject.toml` 的 `py-modules` 中，也不是 console script（E-012）。
  其中 `train_r7_process.py`、`train_r7_recursive.py`、`train_r7.py` 无任何文档/测试引用（E-014）；
  另 3 个被 docs 与 tests 引用（E-014 的对照扫描）。
- **冲突**：它们能被 `python xxx.py` 直接运行，所以不是废代码；但打包后不在 wheel 内，
  `test_r7_installed_package.py` 也不检查它们。
- **选项 A**：给被引用的 3 个补 `py-modules` 声明；无引用的 3 个移入 `scripts/` 或归档。
- **选项 B**：全部补声明。
- **选项 C**：全部保持现状。
- **影响**：A 让打包边界与文档引用一致；B 会把无引用的实验入口带进发布物。
- **我的建议**：A。

</details>


## Q-003 ✅ 已决：启用 `--strict-markers`

**决定**：选项 A。`pytest.ini` 加入 `--strict-markers` 与 `markers` 段，注册 `gpu`、`network`、`slow`。
**落地**：已核验当前仅使用内置 `parametrize`/`skipif`，开启后无任何测试受影响（收集数量不变）。

<details>
<summary>原始分析</summary>

- **现状**：`pytest.ini` 无 `markers` 段、无 `--strict-markers`（E-025）。
  当前只用 `parametrize`（76 处）与 `skipif`（2 处），没有自定义 marker，因此**现在没有风险**。
- **触发条件**：一旦引入"需要 GPU"或"需要网络"的 marker（`docs/R7_TASK_QUEUE.md:20`
  记录 #20 GPU 验收仍被阻塞），未注册的 marker 会被静默忽略，测试会误跑或误跳。
- **选项 A**：现在就加 `--strict-markers` 与空 `markers` 段（成本 1 行，立即生效）。
- **选项 B**：等真需要 marker 时再加。
- **影响**：A 让拼错的 marker 立刻报错；B 保持现状但保留隐患。
- **我的建议**：A。这是一行改动、零风险，且与项目"失败要显式"的整体风格一致。

</details>

## Q-004 ✅ 已决：补 `deterministic` 字段（不引入假 seed）

**决定**：选项 B 的轻量版本。
**落地**：`training/r7_evaluate.py` 的 provenance 增加 `'deterministic': True` 与
`'determinism_scope'` 说明字段。已核验无测试比较整字典或对 provenance 做哈希
（`r7_policy_selection.py:105` 的 `report_sha256` 会对整字典取 digest，因此**新产生的**
`selection.json` 哈希会变化——这是记录事实，不是缺陷）。

<details>
<summary>原始分析</summary>

- **现状**：`r7_evaluate.py` 与 `r7_inference_profile.py` 的 provenance 不含 seed；
  训练记录（`r7_local_runner.py:74`）与协议（`r7_cpu_study.py:17`）都含 seed（E-069）。
- **理由**：推理按构造是确定性的，且有 checkpoint digest 校验（`r7_inference_profile.py:127-128`）。
- **冲突**：`r7_paired_comparison.py:35` 的 block bootstrap 有随机性并记录 seed，
  但同一条流水线里的评估记录没有 seed 字段，字段语义不一致。
- **选项 A**：保持现状（推理无需 seed，bootstrap 单独记录）。
- **选项 B**：给评估 provenance 补 `inference_seed: null` 或显式 `deterministic: true` 标记。
- **影响**：A 成本 0；B 让"为什么没有 seed"成为记录事实，避免读者误以为遗漏。
- **我的建议**：B 的轻量版本——加 `deterministic: true` 字段即可，不必引入假 seed。

</details>

## Q-005 ✅ 已决：`.mimosa/` 与 `.zcode/` 均进 `.gitignore`

**决定**：选项 A，并扩展到本轮新发现的 `.zcode/`（ZCode 计划/会话状态，同为工具产物）。
**落地**：`.gitignore` 重写为分组带注释形式，加入 `.mimosa/`、`.zcode/`、6 个 CI 产物暂存目录
（`/input_pilot/`、`/input_seasonal/`、`/input_study/`、`/input_continuous/`、`/input54/`、`/input56/`）、
构建与静态分析缓存、虚拟环境与 `.env`。
**已核验**：用 `git ls-files` 逐条比对，**没有任何已跟踪文件**会因新增模式变成被忽略；
`data/manifests/real_smoke/*.jsonl` 与所有 `.gitkeep` 仍被跟踪。

<details>
<summary>原始分析</summary>

- **现状**：`.mimosa/` 是代理工具（Mimosa 安全扫描）的运行目录，会随会话增长，
  当前显示为未跟踪（E-003），且不在 `.gitignore` 中。
- **影响**：不处理的话，`git add -A` 一类操作可能把它带进提交；
  而它的内容是工具状态，不属于项目产物。
- **选项 A**：在 `.gitignore` 加一行 `.mimosa/`。
- **选项 B**：由代理工具侧配置到仓库之外。
- **我的建议**：A。这是 1 行改动，属于仓库卫生，且与该文件已有的"什么算产物"定位一致。

</details>

## Q-006 ✅ 已决：改造为检查最近提交

**决定**：选项 B（不删除）。
**落地**：`.github/workflows/ci.yml` 的 `git diff --check` 改为 `git show --check --format= HEAD`，
并加注释说明原因。已核验该命令在干净 checkout 上返回 0 且真正检查内容。

<details>
<summary>原始分析</summary>

- **现状**：`ci.yml:38` 在干净 checkout 上执行 `git diff --check`，此时 diff 为空，
  该步骤实际不产生任何检查作用（E-099）。
- **冲突**：删掉它减少噪音；但它可能在同 job 的后续步骤（如 `pip install` 改写文件）后有用。
  实测当前步骤顺序中，它之前只有 checkout 与 pip install，均不改动被跟踪文件。
- **选项 A**：删除该步骤。
- **选项 B**：改为 `git diff --check HEAD~1` 或对 `git show --check` 检查最新提交。
- **选项 C**：保留（无害，只是无效）。
- **我的建议**：B 或 C。B 能让它真的起作用；C 成本为零。**不建议 A**。

</details>

## Q-007 ✅ 已决：README 顶部加主线指针

**决定**：选项 A。
**落地**：`README.md` 在副标题后插入 4 行引用块，指向 `AGENTS.md` 与 `docs/R7_TASK_QUEUE.md`，
并说明 V6 各节仍然有效但不再是开发重点。

<details>
<summary>原始分析</summary>

- **现状**：根 `README.md` 通篇描述 V6 架构与 V6 smoke 流程（E-142），
  而当前活跃开发主线是 R7（`docs/R7_*.md` 39 篇），`README.md` 未提及 R7。
- **冲突**：新读者按 README 操作会进入 V6 路径，而 R7 才是当前分支的工作内容。
  这不是错误——V6 是已交付的 MVP，README 记录它是对的——但缺一个入口指向 R7。
- **选项 A**：在 `README.md` 顶部加一段"当前活跃主线见 docs/R7_TASK_QUEUE.md"，3 行。
- **选项 B**：保持现状，靠 `AGENTS.md` 的路由表引导。
- **影响**：A 成本极低，且对人类读者同样有效；B 只对代理有效。
- **我的建议**：A。

</details>

## Q-008 ✅ 已决：三份清单各补头部注释（不合并）

**决定**：选项 B。保留三份清单，用注释把"隐性不一致"变成"已声明的分工"。
**落地**：三份 `requirements*.txt` 各加头部说明：适用场景、是否被 CI 安装、与 `pyproject.toml`
optional 组的对应关系。特别标注 `requirements-data.txt`（rasterio/Pillow）**当前无 CI workflow 安装**。
未改动任何依赖条目、未改动 CI 安装命令。

<details>
<summary>原始分析</summary>

- **现状**：`requirements.txt`（5 项）、`requirements-data.txt`（7 项）、
  `requirements-r7-data.txt`（7 项）与 `pyproject.toml` 的 optional 组内容互不一致
  （E-019、E-020）。CI 安装 `requirements.txt` + `requirements-r7-data.txt`。
- **冲突**：`requirements-data.txt` 里的 rasterio/Pillow 未被任何 CI workflow 安装，
  但它们对应的 `worldcover_cog.py` 被测试导入。
- **选项 A**：合并为 `requirements.txt` + `pyproject.toml` optional 两组，删除第三份。
- **选项 B**：保留三份，但补注释说明各自适用场景与是否被 CI 使用。
- **影响**：A 更干净但需要动 CI；B 成本低。
- **我的建议**：B 先做，A 在有稳定 CI 反馈窗口时再做。

</details>

## Q-009 ⏳ 未决：三条重放 workflow 的 HEAD-vs-归档脆弱性

- **现状**：`r7-restored-diagnostic.yml`、`r7-correction-audit.yml`、`r7-extended-control.yml`
  用**当前 HEAD 的代码**加载**早期 commit 产生的 checkpoint**。`load_checkpoint` 会比对
  `model_code_sha256`，因此任何 `model/` 字节变化都会让这三条路径抛 `ValueError`。
- **本轮已发生**：长行清理改动了 `model/` 下 5 个文件，digest 从 `20196c64...` 变为
  `d9fb07f2...`。**没有绕过校验**，而是在三篇对应文档中记录了该约束与新 pin。
- **冲突**：这三条 workflow 的价值是"用固定产物做跨时间核对"，但它们与"允许模型代码演进"
  天然矛盾。当前处置是"用归档 code.zip 重放"，但那需要人工把 zip 解开并切代码，不是 CI 能自动做的。
- **选项 A**：保持现状 + 文档约定（当前做法）。每次改 `model/` 后需人工用归档代码重放。
- **选项 B**：让这三条 workflow 显式 checkout 产物对应的 commit
  （例如从 `code_commit.txt` 读取后 `git checkout`），使流程自证其代码身份。
- **选项 C**：冻结 `model/` 不再演进，直到这些核对任务全部完成。
- **影响**：A 成本低但依赖人的纪律；B 更可靠，但需要改 workflow 的 checkout 逻辑，
  且要在 CI 中处理"归档 commit 可能不在当前分支历史里"的情况；C 会阻塞正常开发。
- **我的建议**：B 是正确方向，但**需要单独一轮授权**（涉及改动 3 条实验 workflow 的取码逻辑）。
  在此之前按 A 执行，并把"这些 workflow 只用归档代码重放"写进各自的文档（已完成）。

## Q-010 ⏳ 未决：`r7_cpu_study.py` 与 `r7_baseline_study.py` 缺内部墙钟截止

- **现状**：R-030 报告这 2 个模块在函数体内跑有界训练但没有墙钟检查（已登记为
  `DEADLINE_EXCEPTIONS`）；另外 3 个控制模块有（`r7_continuous_control.py`、
  `r7_extended_control.py`、`r7_spatial_study.py`，均为 1080 秒）。
- **影响**：它们依赖 workflow 的 `timeout-minutes` 兜底，意味着失败前会耗尽整个 CI 预算，
  而不是尽早失败并保留中间证据。
- **选项 A**：给这 2 个模块补同样的 1080 秒检查，并从例外清单移除。
- **选项 B**：保持例外，接受"靠 CI 超时兜底"。
- **影响**：A 是机械改动（各 2 行），但会改动正在被引用的实验模块；
  B 零风险但把 R-030 长期留在 C 类。
- **我的建议**：A，但放在下一轮——本轮已改动 `training/` 下的评估与季节研究模块，
  不宜同时改动实验编排的主路径。

## Q-011 ⏳ 未决：把环境缺口依赖并入 requirements

- **现状**：`icechunk`、`pcodec`、`numcodecs`、`pip` 被代码或测试使用，但三份 `requirements*.txt`
  与 `pyproject.toml` optional 组**都不声明**；CI 是在单条 workflow 里临时 pin
  （`icechunk==2.2.2` / `pcodec>=0.3`）。完整清单见 `environment.md`。
- **冲突**：CI 的两处 pin 本身也不一致——3 条 pilot workflow 用 `icechunk==2.2.2`，
  而 `r7-earthmover-probe.yml` 用 `icechunk>=1.1,<3`。
- **影响**：任何按 requirements 建的新环境，跑到 ARCO/icechunk 下载路径时会 `ModuleNotFoundError`；
  用 `uv venv` 建的环境没有 `pip`，会让 `test_r7_installed_package.py` 失败。
- **选项 A**：把 4 个包并入 `requirements-r7-data.txt`，并把 CI 的 workflow 级 pin 收敛为一处。
- **选项 B**：保持现状，仅保留 `environment.md` 这份记录（当前做法）。
- **我的建议**：A 是正确方向，但属于依赖面变更且会改动 4 条正在跑的 workflow，
  应单独一轮授权后处理。在此之前按 B 执行，新环境按 `environment.md` 的复现命令搭建。

## Q-012 ⏳ 未决：`--device cuda` 在真实 GPU 上长期不可用（已修，但需确认影响面）

- **现状**：`training/r7_experiment.py` 的 `select_device` 把裸 `torch.device('cuda')`
  传给 `torch.cuda.set_device()`，而 torch>=2.8 要求整数索引或带索引的 device，
  否则抛 `ValueError`。这意味着**所有** `--device cuda` 入口在真有 GPU 的机器上都会失败。
- **为何长期未发现**：唯一相关的测试 `test_cuda_unavailable_never_falls_back` 用 monkeypatch
  把 CUDA 设为不可用，因此在到达 `set_device` 之前就抛错了——**CUDA 可用路径从未被测试覆盖**。
  CI 是 CPU-only，所以也没暴露。
- **本轮处置**：已修为「无索引时回退到 `torch.cuda.current_device()`」（保留
  `CUDA_VISIBLE_DEVICES` 语义），并新增 3 个测试覆盖真实 GPU 路径（无 GPU 时自动 skip）。
  已在 2×RTX 3090 上实测 `--device cuda`、`cuda:0`、`--bf16` 三条路径通过。
- **需要确认**：是否有历史实验产物/文档记录了"在 GPU 上跑过"却实际走了别名路径。
  如有，那些结论需要用修复后的代码重跑核对。**这需要你确认，我无法从代码判断。**
- **选项 A**：确认没有受影响的结论，本次修复即为最终处置。
- **选项 B**：审计历史产物中的 device 字段（`contract`/`provenance` 记录了 device），
  找出声称 cuda 的运行并复核。
- **我的建议**：B 的低成本版本——搜索历史产物 JSON 里的 `"device": "cuda"` 字样即可判断是否存在。



- **现状**：R-030 报告这 2 个模块在函数体内跑有界训练但没有墙钟检查（已登记为
  `DEADLINE_EXCEPTIONS`）；另外 3 个控制模块有（`r7_continuous_control.py`、
  `r7_extended_control.py`、`r7_spatial_study.py`，均为 1080 秒）。
- **影响**：它们依赖 workflow 的 `timeout-minutes` 兜底，意味着失败前会耗尽整个 CI 预算，
  而不是尽早失败并保留中间证据。
- **选项 A**：给这 2 个模块补同样的 1080 秒检查，并从例外清单移除。
- **选项 B**：保持例外，接受"靠 CI 超时兜底"。
- **影响**：A 是机械改动（各 2 行），但会改动正在被引用的实验模块；
  B 零风险但把 R-030 长期留在 C 类。
- **我的建议**：A，但放在下一轮——本轮已改动 `training/` 下的评估与季节研究模块，
  不宜同时改动实验编排的主路径。

