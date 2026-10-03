# R7 深度安全扫描分诊（Mimosa，2026-09-25）

本文档记录两次封存的 Mimosa 深度扫描（`static_only_no_runtime_execution` 边界）
的结果与处置。背景：2026-09-25 的 goal 收尾把「补上完整安全审计」列为未完成事项
（此前多次 push 时扫描器报 `scanner_enobufs`、未取得完整结论）。

## 扫描（修复前）

- scanId：`scan-2026-09-25T10-46-34.545Z-e148e037d7f7`
- seal：`sha256:7f1f89e5a280bd059421618d9cd94111848aca00d5a56f6a617812ab389e91ba`
- depth：deep；依赖扫描 102 个包（2 个匹配、5 条 advisory）
- **findings：29**（high 24 / medium 4 / low 1）

| 位置 | 条数 | 类别 |
| --- | --- | --- |
| `legacy_v531_full/**`（只读归档） | 27 | 不安全反序列化 18、路径穿越 6、SQL 拼接 2、不安全随机数 1 |
| `data/download/uci_beijing.py:28` | 1 | **SSRF（CWE-918，high）** |
| `data/download/worldcover_smoke.py:47` | 1 | **SSRF（CWE-918，high）** |

## 处置

### 1. `data/download/` 的 2 条 SSRF —— 已修复（活跃代码）

两个下载器用 `urllib.request.urlopen` 直接打开动态构造的 URL。修复：新增
`data/download/http_public.py`——

- `reject_non_public_host(url)`：仅允许 http/https；`socket.getaddrinfo` 解析后
  逐地址校验 `ipaddress.ip_address(...).is_global`，拒绝 localhost/环回/私网/
  链路本地/保留地址（含 169.254 云元数据）；
- `_ValidatedRedirectHandler`：每一跳重定向都重新过上述校验（限制重定向 +
  收窄 DNS rebinding 窗口）；
- `PUBLIC_OPENER` / `open_public(...)`：统一的校验式 opener。

`uci_beijing.py` 与 `worldcover_smoke.py` 改为经 `open_public` 下载，
**文件内不再有 `urlopen` 调用**。两个下载器的目标 URL 均为固定 https 常量
（archive.ics.uci.edu / raw.githubusercontent.com / titiler.terrascope.be），
校验不改变其行为；失败路径（`failed-no-fallback`，R-008）保持不变。

新增 `tests/test_http_public.py`（10 用例，离线安全：IP 字面量与 localhost
均本地解析）：9 类拒绝（环回/私网三段/云元数据/localhost/非 http/无 scheme）
+ 1 类公网放行。全量测试 885 → **895 passed, 3 skipped, 0 failed**。

### 2. `legacy_v531_full/` 的 27 条 —— 接受（只读归档边界）

归档快照按 R-002/R-031 是**只读**的：活跃代码不得 import 它、不得写入它，
`tools/check_conventions.py` 有两条阻断规则持续保证这一边界。其中的
pickle/yaml.load、路径拼接与 SQL 字符串拼接是 V5.3.1 时代的既有代码，
按「不擅改历史、只控边界」的原则**记录并接受**：

- 活跃代码（`data/ model/ training/ scripts/ tests/`）与其零 import（R-031 PASS）；
- 归档目录无任何数据写入（R-002 PASS）；
- 运行时不可达：无 workflow、无入口脚本执行归档内模块。

若未来某条归档基线需要复活，应先把它迁入活跃区并通过门禁，而不是在归档内打补丁。

## 复扫（修复后）

- scanId：`scan-2026-09-25T11-26-09.525Z-21c50bd8c942`
- seal：`sha256:aa8b34f793c63b8b99012512eecfdd03f796efe91e7b676e0865f258646ccdf8`
- **findings：27**（high 22 / medium 4 / low 1）——`data/download/` 的 2 条 SSRF
  **已消失**；剩余 27 条全部位于 `legacy_v531_full/**`，与上面的接受处置一致。

## 残留与如实声明

- 扫描器为静态分析。**2026-09-27（#68）起，解析-校验-连接之间的 DNS rebinding
  窗口已被关闭**：原先的 `reject_non_public_host` 只是**预解析检查**（解析后丢弃
  结果），实际连接仍由 urllib 按原 hostname 二次解析，check 与 use 之间存在窗口。
  现改为 `data/download/http_pinned.py`：**只解析一次**、逐地址要求 `is_global`、
  并把 socket 连接到**已被校验的那批 `sockaddr`**（不再把 hostname 交给 socket 层
  二次解析）。重定向逐跳校验 scheme / **host** / **port** 与解析结果；代理策略显式化
  （不再静默继承 `*_proxy`）；HTTPS 用 URL hostname 作 SNI 并保留证书与主机名校验。
  `http_public.py` 保留历史导入路径并**再导出**该实现，既有调用方无需改动即获得
  更强行为。离线回归见 `tests/test_http_pinned.py`（33 例），其中
  「首次解析公网、连接时重解析非公网」的脚本化反证是本项的核心证据。
  **边界如实声明**：本实现提供**连接地址约束**（非公网地址不会被拨号），
  但**不**提供 allowlist、不检查响应内容、也不保证一个公网 origin 提供什么。
  「预解析检查」与「连接地址约束」的区别已写入模块 docstring——只做前者不得
  宣称 DNS rebinding 免疫。
- 归档内 27 条不修复：边界受两条阻断级规则持续守护，修复它们等于改写冻结的
  历史基线，代价大于收益。

## 复扫（2026-09-28，#71/#72 第一轮收尾）

起因：本轮 `git commit`/`git push` 时 hook 再次报 `scanner_enobufs`（未取得完整结论），
故按提示主动跑一次全仓深度扫描。结果如下（**未**据此宣称任何安全性）。

- scanId：`scan-2026-09-27T20-23-50.106Z-f2810810d5e8`
- seal：`sha256:399496deefef72173350d80f01f2765f2ed20791d7e56b8c05a19a6b936f21ea`
- depth：deep；依赖扫描 102 个包（2 个匹配、5 条 advisory、5 个 unknown）
- **run status：`inconclusive`**，覆盖缺口写明为「调用图部分不完整：部分调用为动态派发
  或超出分析规模，跨文件可达性可能不完整」；`verdictEffect: none`
- **findings：27**（high 22 / medium 4 / low 1）——与 2026-09-25 复扫是**同一集合**，
  且**全部**位于 `legacy_v531_full/**`（只读归档，处置见上一节）
- **本轮改动集（model / data / training / scripts / tests）内 0 条命中**：按本轮新增与
  修改的文件名检索该报告，无任何条目指向 `model/spacetime_conditioning_r7.py`、
  `model/process_readout_r7.py`、`model/r7_rollout.py`、`data/r7_store.py`、
  `tests/test_r7_switched_path_equivalence.py` 或 `scripts/study_r7_71_72_spacetime_rwa.py`。
- 附带说明（本轮唯一的"扫描发现→修复"闭环，发生在代码入库**之前**）：早期草稿里
  两个测试辅助写法被写入前的扫描器拦下——`subprocess.run` 配非字面量 argv
  （命令注入模式）、以及 `tarfile.extractall` 的无校验回退（路径穿越）。最终提交的
  `tests/test_r7_switched_path_equivalence.py` 只用**全字面量 argv** 调 `git archive`，
  并自行实现带边界校验的归档解包（逐成员 `is_relative_to` 检查，**不**调用 `extractall`）。

## M3 证据提交阻塞分诊（2026-10-03；不授予安全放行）

### 原始事件与证据边界

本轮按已批准计划 0012，仅本地有限静态分诊和原记录读取，没有运行 scanner、密封深扫、GLM、
归档代码或新实验。HEAD 保持 `011ab4cdc3b4fa9f6671a22962a2cf3227998c66`；新 M3 证据页及原
七文档已暂存未提交，canonical index 仍 17 条。科学 `N2a/paused` 与安全提交阻塞是独立边界。

从指定会话 SQLite 的 `part` 表只读投影原记录（`mode=ro`、`query_only=ON`），只保存错误、
警告及 hook 风险位置，不保存完整工具命令、prompt 或输出。会话为
`sess_5f5a3598-8be3-41bf-845b-9f456f8ff4c8`；以下时间为 UTC：

| 事件 | 原记录身份 | 已确认的结果 |
| --- | --- | --- |
| 工程 commit | rowid 61410，call_LDQqwI9HVnuuKza3yzlmgoh7；2026-10-02 16:51:23.174–16:51:37.875 | `scanner_enobufs`，兼容放行；没有完整安全结论 |
| 工程 push | rowid 61413，call_Itg9WS4nuaFJtzF71uLg45Xj；16:52:12.307–16:52:12.663 | 同类警告；工具 exit0 不等于扫描通过 |
| 证据 commit | rowid 61950，call_iZAllAK7RKouEca7HbRGX3Wk；17:35:36.100 | 12 high/1 low，high 强制拒绝，整条工具调用未执行；覆盖不完整 |
| Stop 增量复查 | run 20261002T175316239Z-3181628-8e459f852ab6；17:53:16.239 | required2/scanned0/failed2，`ETIMEDOUT`；partial/inconclusive，零发现不是 PASS |

证据 commit 的原 error 为「Mimosa L3 在 commit 前发现 12 个高危、1 个低危，最高等级 high」。
hook 原附加文本说明「高危已强制拦截，请修复并重新扫描。本次覆盖不完整，不能把未发现更多问题解释为项目安全」。
附加文本 UTF-8 SHA256 `7990c81b53533124a4dbde696b0c4149f807be67f029bdc594131624fcd03b34`。
low 的位置没有提供；没有该次 L3 完整独立报告或覆盖缺口原因。不能把 Stop 报告当成 L3 报告。
`scanner_enobufs` 的具体异常栈和缓冲参数尚未定位，不把错误名当作已证实根因。

### 十二处 high 的有限静态复核

下表位置均相对 `legacy_v531_full/`，按物理一基行号核对；九份源文件 SHA256、逐处调用链与限制
保存到本轮独立诊断包。**观察到路径/SQL 写法，不等于确认可利用漏洞；常量调用反证也不自动等于误报。**

| 位置 | 已观察输入和调用条件 | 未决边界 |
| --- | --- | --- |
| baselines/external_sources/FourCastNet-master/train.py:611 | CLI config/run_num 与 YAML exp_dir 拼接到固定 hyperparams.yaml；未见 traced chain 包含校验 | CLI/YAML 信任、权限与初始化条件 |
| train_suite.py:234 | suite YAML alias 拼在 CLI 输出根后，写 config.yaml；resolve/重复名检查不限制 alias | suite 作者权限、允许根及运行暴露 |
| train_suite.py:399 | CLI 输出根下固定 summary.csv；alias/metric_keys 不参与此 sink 的路径 | 输出根授权、汇总是否到达 |
| training/trainer.py:165 | logging 目录/run/stage 拼接写 wandb_id.txt；W&B 开启且可用才到达，resume 可提前返回 | 归档 UI 提供目录/名称字段，但部署和鉴权未核 |
| tests/run_static_feature_ablation.py:226 | CLI out_dir 下固定 results.json | 调用权限与评估完成条件 |
| tests/run_static_feature_ablation.py:301 | 同 out_dir 下固定 report.csv，独立写入处 | 调用权限与前序报告构造 |
| streamlit_app/utils/db.py:118 | PRAGMA 标识符插值；可见 init_db 唯一具名调用传入字面量 | 动态/别名调用与实际部署未覆盖 |
| streamlit_app/utils/db.py:121 | ALTER 插值三个参数；可见调用全为常量，DB 列名只控制分支 | helper 无标识符白名单，其他实参来源未证明不存在 |
| tests/compact_v4_metrics.py:53 | 固定绝对 COMPACT 常量；输入只影响内容 | 文件系统控制、权限与 symlink 未核 |
| tests/extract_v4_metrics.py:227 | 固定绝对 OUTPUT 常量；枚举结果不选择输出路径 | 文件系统控制和执行条件未核 |
| baselines/common.py:246 | save_json 接受调用者路径；CLI/YAML alias，另有归档 UI 目录输入链 | UI 部署/鉴权及其他调用者未完整核 |
| baselines/run_all.py:76 | CLI out_dir 下固定 summary.json，前序三 baseline 完成后写 | 调用授权和前序完成未核 |

其中归档 DB 的可见调用为 `db.py:94`，参数 `experiment_metrics`、`model_name` 与固定 TEXT 定义。
归档 UI 到 logging/baseline 写入的静态链确实存在；未执行它们，不把「存档」推断成天然安全。
本次源 hash 只绑定当前字节，原 hook 未提供源码快照 hash，不能宣称逐字节还原其扫描输入。

### 修正历史边界解释与接口核查

上文 2026-09-25 的「记录并接受归档风险」「运行时不可达」保留为当时处置记录；本轮明确补充：
它们不是已接入当前 Mimosa 的 finding 豁免，也不是完整调用图或 OS 隔离证明。R-031 是有限正则检查，
R-002 检查活跃侧归档写入，wheel 配置排除归档；只支持有限声明入口/源码搜索未见活跃连接，
不覆盖动态 import、路径注入、显式外部启动或部署。旧密封扫描的 partial/inconclusive 不作全仓 PASS。

当前安装的 Mimosa 1.0.3 随包 `payload/README.md`：第14/29行说明 commit/push 项目级 L3；
第80/91行分别定义 failure mode 与 high 拒绝策略；第138/142/148行说明 Edit/Stop 增量与历史展示
不跳过扫描/deny。公开文档有全局关闭和模式控制，但本任务不使用；**未文档化可核实的 L3 按 finding/
归档接受历史债务、exclude 或 changed-files 接口**，不能把另一 profile 的 diff-only 直接套到 L3。
README SHA256 `fee93794b7639b461e63859265008b53b47b3ae6437b47b491297f6f780b63fa`。
公开代码止于受保护加载器；未解密、修补或重签载荷。`.zcodeignore`/`.gitignore` 是否被 L3 采用未知，
两文件原本均刻意保留归档；不盲加 ignore，不降低文件上限来“修复”未知缓冲错误。

### 本轮出口与复核回执

**未证实官方非降级处置接口，按批准的硬停点维持 BLOCKED。** 没有再尝试 commit（含 dry-run）、
push、供应商上传、安装更新或安全豁免。若只能改变安全范围，需另有明确政策授权及正式产品接口；
若供应商纠正规则/可靠性，仍须真实有效复核和门禁放行，不能用文字风险接受、测试绿或旧工程 CI 替代。

独立输出：`outputs/r7_m3_registration_diagnostic_20261003T024214Z/`；工具态不提交。

| 回执 | SHA256 |
| --- | --- |
| starting_identity.json | `9d062228ef842ffda11f66448c253f6f9b7f9a2ee0ec93d40100d27c62a8a1f2` |
| original_gate_events.json | `f3f569a8cce5892f7f4b16a7699f61d8e4926bc38b052da1e62b7ea9b434f12a` |
| stop_review_projection.json | `b7986ecc9adcd9514f67da9d911b49673b6750580f2d384267b677f283db456b` |
| finding_static_triage.json | `59b60eee0a64926403c1f2c054dc74de4354440c34378dfe8550977a378e6726` |
| supported_contract_v2.json（一基引用修正） | `b788b11e71e6d4e1e3db365595494de61ce302067e95727ff77ef4ebf1b084d8` |

独立只读报告复核确认核心结论与回执一致；唯一必修为四处README引用采用了零基偏移，已生成
`supported_contract_v2.json` 修正为物理一基，原回执及hash留存、不覆盖。引用修正不是规则或安全政策变化。

本轮重核 366 个已跟踪归档文件、原109/新276 seal文件集合及43科学 pins，无 mismatch；仅 opaque hash，
不读取数组或反序列化 checkpoint。相关 CPU 532 项测试实跑通过（38.34s，0 skip）；首次临时目录
放在仓库中，使 conventions 反证 fixture 被 Git 识别为未跟踪内容，24 failed/508 passed/40.80s，
失败日志保留；只改运行临时目录到仓库外，未改断言/测试/规则后同集合通过。
没有新模型、数据、结果、依赖、凭据或安全配置变化。N2a/paused、N3/N4/N5 未做、issue 未关闭与
最终 goal 未完成保持；本次诊断不是完整安全审计、漏洞 runtime verification 或科学推进许可。

## 用户关闭工具后的 D6 登记接续（2026-10-03）

上述 BLOCKED 是此前分诊的真实出口，不回改为安全通过。用户随后先回复「我对其进行明确授权」，
再说明「我已经关闭了Mimosa，可以继续进行」；只读核用户及仓库本地配置的同名插件开关为 false。
执行者没有修改安全配置，用户的 `.zcode/config.json` 未暂存差异及两个无关未跟踪文件不进入登记提交。
这一接续属于用户明确关闭工具后的有限工程恢复，不是官方 finding 裁定、L3 扫描通过或漏洞修复；
未运行新深扫、升级插件、豁免 finding、发送支持工单或解密/修补载荷。历史12high/1low、覆盖缺口、
scanner_enobufs/ETIMEDOUT与静态分诊限制继续保留；正常调用若仍被拒绝，停止而不换路。

证据页已通过正常 git 命令单独提交 `1de3a672d859b42efa7a5fac3293c2832df4a6ee`，blob SHA256
`17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`；正式索引单增 audit/blocked
记录，不把成功提交当安全扫描结果。恢复前原109/新276/366跟踪归档/43科学pins不变；新完整CPU实跑
2395passed/9skipped/2warnings、230.96s，跳过不计通过。执行者只继续D6证据/index/账本/工作分支CI，
新GPU-h=0，模型/数据/依赖/凭据不变，N2a/paused、advance=false、N3/N4/N5与issue/main边界保持。
新回执在仓外 `/tmp/r7_m3_registration_resume_u8t215bd/`，登记的精确CI只在活进度绑定，不伪造未来结果。

实际仍有增量安全链：Write仓外核验草稿被Mimosa报告动态subprocess调用的命令注入模式并拒绝，
草稿未写入。按该工具反馈去掉候选中所有子进程/动态命令，纯只读脚本重新正常Write成功并实跑；
机械检查继续分别直接调用既有检查器，不改任何安全设置/阈值或换提交通道。用户关闭开关false不
证明全部安全链停用；不把一次成功Write/commit或CPU绿说成L3通过，实际commit若拒绝仍停。
