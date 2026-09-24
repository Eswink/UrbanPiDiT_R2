# 命名

范围：活跃区的文件、模块、包、类、函数、常量、测试、文档、配置。

**本页所有规则都基于实测**（对活跃区 204 个 `.py` 做 AST 全量测量，归档目录与 `.venv` 除外）。
每条「现状」栏给出的是**当时测到的比例**，不是声明；依据指向 `EVIDENCE.md` 的编号。

**归档目录不在本页范围内**：`legacy_v531_full/`、`data/legacy_v531/`、`model/legacy_v531/`、
`legacy_v6/` 是只读方法学记录，它们的命名（含 `utils/` 等杂物桶）保持原样。
改造归档等于篡改证据，见 R-002。

## R-038 模块与包文件名 snake_case

- **级别**：必须
- **范围**：活跃区所有 `.py` 文件名与包目录名
- **陈述**：模块与包名用 `snake_case`，匹配 `^[a-z][a-z0-9_]*$`；`__init__.py` 除外。
- **依据**：E-169（实测 204/204 命中，0 例外）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-038`，阻断）
- **例外**：`__init__.py`（Python 包协议名）
- **引入日期**：2026-09-24
- **复核触发**：当引入命名空间包或自动生成模块时

## R-039 类名 PascalCase

- **级别**：必须
- **范围**：活跃区所有类定义
- **陈述**：类名用 PascalCase，匹配 `^_?[A-Z][A-Za-z0-9]*$`（允许一个前导 `_` 表示私有）。
- **依据**：E-170（实测 89/89；其中 5 个带前导 `_` 的是私有类，
  `model/r7_baselines.py:22 _Base`、`data/download/arco_tiny_bounded.py:14 _NoRedirect` 等）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-039`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入 `TypeVar`/`Protocol` 等类型别名类时（它们通常是单字母，应显式立项）

## R-040 函数与方法名 snake_case

- **级别**：必须
- **范围**：活跃区所有函数与方法定义
- **陈述**：函数名用 `snake_case`；私有可加前导 `_`；`__x__` 形式的 dunder 不受限。
- **依据**：E-171（实测 936/936：dunder 83 + 私有 115 + 公开 738，0 个驼峰）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-040`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当需要实现与第三方库签名同名的回调时（如某些框架要求驼峰钩子）

## R-041 模块级常量 UPPER_SNAKE

- **级别**：必须
- **范围**：活跃区模块顶层的常量赋值（值为字面量/容器，或名字本已全大写）
- **陈述**：模块级常量名用 UPPER_SNAKE，匹配 `^_?[A-Z][A-Z0-9_]*$`。
- **依据**：E-172（实测 172/172；`__all__` 是 Python 模块协议名，不算常量）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-041`，阻断）
- **例外**：`__all__`。另有两个模块级小写赋值 `requires_real_fixture`/`requires_dynamic_fixture`
  （`tests/test_real_data_pipeline.py:18,25`）是 `pytest.mark.skipif` 标记而非常量，
  且不在模块顶层之外——判定按"值为字面量/容器"识别，不按名字大小写猜
- **引入日期**：2026-09-24
- **复核触发**：当引入需要在模块级保存可变单例时（那不是常量，不应套用本规则）

## R-042 标识符纯 ASCII

- **级别**：必须
- **范围**：活跃区的类名、函数名、参数名、变量名、属性名
- **陈述**：**标识符**必须为纯 ASCII（无 `ord(c) > 127` 的字符）。
  **注释、docstring、字符串字面量不受本规则约束** —— 本项目大量使用中文注释，这是刻意的。
- **依据**：E-173（实测 0 命中；同时实测 **43/204** 个文件含非 ASCII 的注释/docstring，
  密度最高的是 `tools/check_conventions.py` 与 `data/preprocess/real_station_smoke.py`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-042`，阻断）。**实现只遍历 AST 标识符节点，不做整行正则** ——
  否则 43 个含中文注释的文件会全线误报
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当项目需要面向非英语母语维护者改用拼音标识符时（需显式评审，不建议）

## R-043 代码路径禁用词与杂物桶

- **级别**：禁止
- **范围**：`data/`、`model/`、`training/`、`scripts/`、`tests/`、`tools/`、`configs/`、
  `docs/`（`docs/` 的子目录见 R-047）、`.github/workflows/`、根目录文件
- **陈述**：文件名与目录名不得含 `final`/`old`/`new`/`tmp`/`temp`/`copy`/`backup`/`bak`/
  `draft`/`deprecated`/`misc` 等词（按 `_`/`-`/`.` 定界），不得以 `_vN` 结尾；
  不得出现 `utils`/`misc`/`common`/`helpers`/`manager`/`base`/`shared`/`lib`/`core` 这类杂物桶目录名。
- **依据**：E-174（实测代码路径 0 命中；`_vN` 结尾 0 命中；杂物桶目录 0 命中）。
  唯一命中的 9 个 `final` 全在 `audit/**`，是冻结的证据文件名
- **现状**：A 类 —— 代码路径 0 处违反
- **执行方式**：脚本（`--rule R-043`，阻断）
- **例外**：`audit/**`（证据目录，文件名如 `final_scorecard.md`、
  `real_data_logs/final_pytest.log` 是历史记录，改名会断引用）；
  另 `configs/r2_v6_*.yaml` 中的 `v6` 是**版本标识**而非 `_vN` 后缀，不受 `_vN` 规则影响
- **引入日期**：2026-09-24
- **复核触发**：当审计目录需要轮转或改名时（应保留旧名并加索引，而不是重命名）

## R-044 测试命名（范围限 `tests/`）

- **级别**：必须
- **范围**：`tests/**`
- **陈述**：测试文件名 `test_*.py`（`conftest.py` 例外）；测试函数 `test_*()`，`snake_case`。
- **依据**：E-175（实测 315/315 函数合规；68/69 文件合规，唯一例外是 pytest 规定的 `conftest.py`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-044`，阻断）
- **例外**：`tests/conftest.py`（pytest 规定的文件名）
- **引入日期**：2026-09-24
- **复核触发**：当引入 `pytest` 插件要求的其它约定文件名时

### 为什么范围必须限定在 `tests/`

实测发现 `tests/` 之外有 **6 个** PyTorch-Lightning 钩子以 `test` 开头：

`training/lit_module.py:22 test_step`、`training/r7_lit_module.py:45 test_step`、
`training/r7_process_forecast_lit_module.py:85 test_step`、
`training/r7_recursive_lit_module.py:55 test_step`、
`data/multiscale_dataset.py:72 test_dataloader`、`data/r7_dataset.py:93 test_dataloader`

它们是 Lightning 的协议方法名，**不可改名**。规则若写成"以 test 开头的函数"会全部误报。

同理，`data/manifests/real_smoke/test.jsonl` 的 stem 是 `test`，
所以判定必须按**路径段**而非裸词匹配。

## R-045 `r7` 主题标记出现在定界位置

- **级别**：默认
- **范围**：活跃区模块文件名
- **陈述**：属于 R7 主线的模块，`r7` 必须出现在**定界位置**（`r7_` 前缀 或 `_r7` 后缀），
  形如 `^r7_[a-z0-9_]*\.py$` 或 `^[a-z0-9_]+_r7[a-z0-9_]*\.py$`。
  **不强制前缀** —— 两种形态都是本项目既有写法。
- **依据**：E-176（实测 129 个含 `r7` 的模块**全部**匹配上式的两种形态之一，0 例外）。
  其中 42 个用 `r7_` 前缀、5 个用 `_r7` 后缀、82 个为中缀（`test_r7_*`、`scripts/*_r7_*`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-045`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当 R7 主线结束、模块需要去掉版本标记时

### 一处必须如实记录的更正

本规则的原假设是「`model/` 层规范做法是 `r7_` 前缀，其余写法不规范」。
**实测否定了这个假设**：`model/` 层内部本身就混用三种方案 ——

| 方案 | 数量 | 例子 |
| --- | --- | --- |
| `r7_` 前缀 | 3 | `model/r7_baselines.py`、`r7_halting.py`、`r7_rollout.py` |
| `_r7` 后缀 | 3 | `model/process_forecast_r7.py`、`recursive_weather_r7.py`、`weather_forecaster_r7.py` |
| 无标记 | 14 | `model/process_reasoner.py`、`urban_pidit_r2.py` 等 |

所以「统一为前缀」这条规则**与现有 5 个文件冲突**，不能立。证据支持的措辞是
「定界位置」，129/129 成立。**本轮不重命名 `model/` 下任何文件** ——
改 `model/` 字节会使 `model_code_sha256` 变化，令 Q-009 记录的 3 条重放 workflow
无法加载归档 checkpoint，代价不值得。

## R-046 配置与 workflow 命名

- **级别**：必须
- **范围**：`configs/*.yaml`、`.github/workflows/*.yml`
- **陈述**：配置文件 `snake_case.yaml`（匹配 `^[a-z][a-z0-9_]*\.yaml$`）；
  workflow 用 `kebab-case.yml`（匹配 `^[a-z0-9-]+\.yml$`）。
- **依据**：E-177（配置 10/10；workflow 18/18）。配置两族：`r2_v6_*` 5 个、`r7_*` 5 个
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-046`，阻断）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当引入其它配置格式（`.toml`/`.json`）时，应显式规定其命名

## R-047 文档命名（按目录分层）

- **级别**：必须
- **范围**：`docs/**/*.md`
- **陈述**：文档命名按目录分层 ——
  - `docs/*.md`（顶层）：`UPPER_SNAKE.md`
  - `docs/rules/*.md`：**两种并存**，按性质分。**台账类**（`CHANGELOG`、`EVIDENCE`、
    `MIGRATION`、`OPEN_QUESTIONS`）用 `UPPER_SNAKE.md`；**分类细则类**用 `kebab-case.md`
  - `docs/plans/`、`docs/decisions/`：`NNNN-kebab-case.md`
  - 任意目录的 `README.md` 例外
- **依据**：E-178（实测：顶层 45/45 为 UPPER_SNAKE；`docs/rules/` 为 4 个 UPPER_SNAKE 台账文件
  + 10 个 kebab-case 分类文件；`docs/plans/`、`docs/decisions/` 为 `NNNN-kebab-case`）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-047`，阻断）
- **例外**：任意目录下的 `README.md`
- **引入日期**：2026-09-24
- **复核触发**：当 `docs/rules/` 新增台账类文件时（须沿用 UPPER_SNAKE）或
  新增分类文件时（须用 kebab-case）

### 为什么是分层而不是"一律 UPPER_SNAKE"

实测 66 个 `.md` 中，单套规则无法覆盖：

| 位置 | 模式 | 计数 |
| --- | --- | --- |
| `docs/*.md` | `UPPER_SNAKE.md` | 46/46 |
| `docs/rules/` 台账类 | `UPPER_SNAKE.md` | 4/4 |
| `docs/rules/` 分类类 | `kebab-case.md` | 10/10 |
| `docs/plans/`、`docs/decisions/` | `NNNN-kebab-case.md` | 3/3 |

`docs/rules/` 内部两种并存是**有语义的**：台账文件（变更史、证据、迁移、待决问题）是
项目级账目，沿用顶层的大写惯例；分类细则文件是可读的规则正文，用 kebab-case。
**这条区分是本规则的初版写错后才发现的** —— 初版断言"子目录一律 kebab-case"，
被自己的门禁当场报出 4 例（CHANGELOG/EVIDENCE/MIGRATION/OPEN_QUESTIONS），
核对后确认是规则写错而非文件不合规。详见 CHANGELOG 第七遍的更正记录。

## R-048 参数名缩写（报告型）

- **级别**：默认
- **范围**：活跃区函数与 lambda 的参数名
- **陈述**：只报告**单个无信息 token** 形式的参数名 ——
  长度 ≥4、不含下划线、不是已知领域缩写、且全是辅音串（如 `bldg`）。
  多词名（`mlp_ratio`、`model_cfg`）与领域缩写一律放行。
- **依据**：E-179（实测 1803 个参数中 103 个 ≤ 2 字符，其中 44 个不在常规集合内，
  但高频者均为领域惯用：`ds` 10、`hw` 7、`u`/`q`（气象学）、`B`/`D`/`H`/`W`（张量维度）、
  `lr`、`kv`、`kw`、`fn`、`p0`、`wp`）。按本判据实测当前 **0 处命中**
- **现状**：C 类 —— 报告型，当前 0 命中。**不阻断**：短参数名在本项目是常态而非缺陷
- **执行方式**：脚本（`--rule R-048`，报告型）
- **例外**：领域缩写（`ds`/`da`/`lr`/`kv`/`kw`/`fn`/`hw`/`hp`/`wp`/`p0`、单字母维度名）
- **引入日期**：2026-09-24
- **复核触发**：当出现真正难懂的缩写导致评审困难时

### 判据为什么收得这么窄（一处已撤回的实现）

本规则的初版判据是"名字里出现 3 个以上连续辅音即报告"。实测下来它命中 **113 处**，
其中包含 `mlp_ratio`、`model_cfg`、`training_std` 这类**完全清晰**的名字 ——
因为它把 `mlp`、`cfg`、`std` 这些公认缩写当成了问题，还把多词名里的每一段都单独检查。

**一份噪音率这么高的报告会被直接忽略**，比没有报告更糟。因此该判据被撤回，
收窄为"单个无信息 token"，实测命中 0（即：现有代码没有这类问题）。
`tests/test_check_conventions.py::test_r048_stays_quiet_on_clear_names`
用 `mlp_ratio`/`model_cfg`/`bldg_ht` 锁住这个边界，防止判据再被放宽回去。

### 为什么不把短参数名做成阻断规则

44 个"非常规缩写"全部是 2 字符的领域惯用词。把它们阻断意味着要求把 `ds` 写成 `dataset`、
`hw` 写成 `height_width` —— 这与项目既有的紧凑数值风格冲突，且不会减少任何真实缺陷。
**规则强加于项目的代价大于收益**，故只报告。
