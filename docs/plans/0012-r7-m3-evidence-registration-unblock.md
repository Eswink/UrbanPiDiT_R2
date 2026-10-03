# 0012 — M3 证据登记的安全闸门分诊与条件化收尾

**日期：2026-10-03。状态：原分诊已完成并保留 BLOCKED 出口；用户随后关闭 Mimosa 并明确允许继续，D6 有限登记恢复，科学暂停不变。**
本计划源自一次 planner 委派（工作态草稿）及主链事实复核。草稿把证据路径、hash 长度和科学状态写错，
未采用；主链按用户批准范围修正 JSON，`tools/check_planner_plan.py` 实跑 `verified=true`、0 失败。
规划输出不是实验、安全或科学证据。本轮批准不包含安全策略豁免、外部发布或科学方向变更。

## §1 起点与不变式

- 分支 `r7/weather-reasoning`，HEAD `011ab4cdc3b4fa9f6671a22962a2cf3227998c66`。
- 原七份文档已暂存未提交，无已跟踪文件的未暂存改动；两个无关未跟踪文件不纳入。
- M3 独立补测 23/23 已完成，旧失败不改；新结果完整但科学 `paused`，不是缺项实验重新执行。
- canonical index 仍 17 条，新记录仅 pending，evidence/registration commit 与登记 CI 不存在。
- 工程 CI 37036869969 只绑定上述 HEAD，不覆盖未提交文档。N3/N4/N5 与 issue 关闭未执行。
- 原始数据、四历史归档、原运行目录、code.zip/checkpoint、现有 acceptance/seal 全部只读；
  不删除、不迁移、不 untrack、不运行归档、不改身份比对或冻结科学出口。

科学依据只引用 `docs/decisions/0031-m3-cross-attempt-validation-provenance.md`、
`docs/goals/n2a-m3-validation-complement.md`、`docs/goals/v2-autonomous-completion-and-closeout.md`
及 `docs/R7_73_PROCESS_SUPERVISION.md`。0030 的普通推进权不解除已触发的 any-unresolved 暂停。

## §2 交付物与边界

| # | 交付物 | 验收形态 |
| --- | --- | --- |
| D1 | 计划校验与起点身份 | 修正版 JSON、校验回执、HEAD/staged 集合、归档/旧产物 hashes |
| D2 | 脱敏安全阻塞分诊 | 分开的事件时间与原始拒绝引用、12 个 high 的有限静态输入/调用链核查、未知 low 与覆盖限制 |
| D3 | 支持接口核查与停止决定 | 官方随包文档位置/身份；无合规接口则停止提交并交付本地诊断包 |
| D4 | 有条件证据 A、登记 B、CI 尾 C | 只有安全前置满足才执行；真实 SHA、实际页 blob、canonical brief、精确登记 CI |
| D5 | 版本化治理草稿与实际结果 | 本计划、安全分诊追加、E/CHANGELOG；真实测试结果及未提交/未推进说明 |

不默认安装/升级插件、调用密封深扫或 GLM、向供应商上传源码/日志，不公开反馈内容，不改配置。
只读归档和有限静态 import/wheel 检查不是 OS 沙箱；hash 一致不是无漏洞证明。

## §3 实施顺序与真实硬停点

1. 校验修正版计划，在全新排他输出保存起点，核 109 原文件、276 新文件及 43 个科学 pins。
2. 分别记录工程 commit/push 的 `scanner_enobufs`、证据 commit 的 high 拒绝及 Stop 的 `ETIMEDOUT`。
   12 个 high 逐项核输入和有限调用链，不把固定参数等上下文自动判成 scanner 误报，不执行旧代码。
3. 核官方安装包的公开接口。当前未证实 L3 归档例外、baseline 或 changed-files 接口；
   **若仍无官方非降级处置，立即保持 BLOCKED 并交付诊断，不重试无变化的提交。**
   不盲加 ignore、不关闭/放宽 hook、不解密或修补受保护载荷、不换提交通道。
   历史风险例外既要另有明确政策授权，也要正式产品接口；文字 ADR 不能代替机器放行。
4. 仅安全前置真实满足时：A 提交包含新证据页的文档包，实测页 blob SHA256（64 位）和 commit（40 位）；
   B 仅增加 `m3-validation-complement-complete-paused` 一条 audit/blocked/scientific_claim=false 记录，
   index 17→18、brief canonical 同步、新成本加入 record 引用，四历史 note 保留（预期 5→4）。
   正常 SSH 推工作分支，核 B 精确 SHA 的主 CI 九步骤 completed/success；不使用旧工程 CI 代替。
5. C 只追加活进度与计划实际结果，不追写冻结证据/index；核自身 CI，回执和最终报告保存其验证，
   不递归尾提交。整个工程过程仍停在 N2a/paused，不做 N3/N4/N5、Closes/main 或 final goal complete。

## §4 验证与范围

实施阶段跑 conventions、campaign、两个当前 goal、index/brief、whitespace、相关 hooks/conventions/
planner/campaign/goal/index/M3 complement 定向测试；只有合法进入登记阶段才要求再跑完整 CPU pytest。
用仓库 `.venv`，禁 GPU 可见性、不启用 GPU 测试；临时文件在新诊断目录，`-B`/禁 pycache。
不对受保护归档运行 compileall，不重跑会向旧目录排他创建 receipt 的 seal 脚本。
核既有产物前后 hash，不新增真实数据或数组读取，不反序列化 checkpoint、不启动训练/评估。
0 新增 GPU-h。测试绿不等于安全、科学或 goal 完成；未实跑的验证不得标 PASS。

## 实际结果

- **D1 已完成**：2026-10-03 实跑 planner 合同 verified=true/0 failures；起点七暂存路径与 HEAD 核齐。
  366 个已跟踪归档文件已 opaque hash，原 109/新 276 文件逐项对 seal 且集合无差异；43 个科学 pins 无 mismatch。
  649 个非归档已跟踪文件起点另保存；不声称包含非跟踪数据全盘或 OS 可达性证明。
- **D2/D3 已完成诊断并命中停止条件**：SQLite part表rowid61410/61413/61950与callID只读核齐，保存
  原警告/拒绝及hook文本；三个事件区分，低危详情与L3完整报告仍缺。12处high有限静态复核及九源hash
  对清单一致，SQL常量调用/固定路径仅反证，不认定全误报；归档UI目录输入链也保留。官方文档明确
  有全局控制但无已证实归档例外/非降级可靠性纠正接口；不使用放宽开关，按硬停点终止提交。
- **D4 未执行**：没有新 evidence/registration commit、推送或登记 CI，index 保持 17/五notes；
  不重复同样commit或以dry-run冒充放行，不上传供应商或升级插件。安全政策改变不是本轮已获授权。
- **定向验证已实跑**：首次仓内tmp使conventions反证被Git识别为未跟踪而容忍，24failed/508passed/
  40.80s；保存失败日志，只改仓外tmp后相同集合 **532 passed in 38.34s，0 skipped**，未改代码/测试/
  断言。未进入登记，不再跑完整CPU/新CI；旧2395/9仍仅历史记录。
- **治理验证实际通过**：conventions37条阻断0违规、campaign0fail/5notes、两当前goal0fail、
  index/canonical brief17条一致、staged/unstaged whitespace均exit0。日志与机械回执在独立诊断目录，
  不是安全扫描或真实登记CI。新计划及本轮治理追加只待版本化，不绕门禁强行提交。
- **D5 本地治理完成**：分诊正文追加、plans索引、E-231–E-233/CHANGELOG和三活进度记录真实阻塞，
  原证据页/科学state/账本/index不改；**10份治理文档已暂存、未提交，跟踪文件无unstaged差异**。
  收尾重核366归档/385旧新实验/50原acceptance/43科学pins与插件公开文件/配置均不变；
  `final_identity.json` SHA256 `c39bfbe0b556495d3d0d24db36208e5df3dfb05b75690140a1d8df5c81d25e45`、
  `diagnostic_manifest.json` SHA256 `258824d8440c82655312519863454f516ed4aeecee78d0fbf61d05115095aecc`；
  这些是有限文件身份核验，不是全OS隔离/扫描证明。本条尾追加后的暂存身份另在交付回执核对，不递归修改原回执。
- **独立文档复核**：核心阻塞/科学边界与回执一致，唯一必修是四处README引用零基偏移；已用新
  `supported_contract_v2.json` 修正一基引用并同步正文hash，原回执保留。该复核不是安全放行或goal校验。
- **与计划的差异**：修正 planner 的不存在路径、SHA256 长度及错误科学状态；不采用 commit dry-run 放行测试。
  诊断命中预设硬停点，所以没有进入条件化登记；首次test tmp失误及复跑均如实保留。
- **工具态**：`outputs/r7_m3_registration_diagnostic_20261003T024214Z/`，不提交。
  起点回执 `starting_identity.json` SHA256 `9d062228ef842ffda11f66448c253f6f9b7f9a2ee0ec93d40100d27c62a8a1f2`；
  `repository_baseline.json` SHA256 `75469782b08bef957dce87f25bd94035c86236a86ec5b18ca083a8fd2b6044d0`。
- **不变状态**：N2a/paused、科学 advance=false，N3/N4/N5 未做；issue 未关闭、goal 未完成。

## 后续用户授权与登记接续（2026-10-03）

- **与原计划的差异**：D3 硬停点此前实际执行，没有找到官方非降级处置。用户随后回复「我对其进行明确授权」，
  再说明「我已经关闭了Mimosa，可以继续进行」。本次据用户关闭工具恢复D6，不声称已满足原来的官方处置前置，
  不将自动通知或原未答回执当许可。执行者未修改插件/安全配置，用户本地配置差异不进入登记提交。
  原正文及 D4 未执行记录保留为其时点事实；没有官方裁定、L3 clearance、新深扫、升级或支持工单外发。
- **恢复范围**：仅证据 A、index/brief/账本 B 与真实 CI 尾 C，正常 git/SSH 工作分支通道；实际拒绝仍立即停止，
  不换路。原109/新276/366跟踪归档/43科学pins已再次只读核齐，0新增GPU，不推进N2a、不操作main/关闭issue。
- **A 已真实完成**：单独证据提交 `1de3a672d859b42efa7a5fac3293c2832df4a6ee`，仅一文件，实际 blob SHA256
  `17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`。不是扫描通过；此后不回写证据页。
- **B 本地登记已构造**：仅 append `m3-validation-complement-complete-paused`，index17→18，旧17条原文不变，
  canonical brief18条/1历史候选；audit/blocked/scientific_claim=false，科学暂停不改。历史pending草稿不覆盖，
  新账本行仅增加record引用、不重复GPU成本；精确 B 提交/推送/CI尚待实际执行与本节尾绑定。
- **本次完整CPU实际结果**：2395passed/9skipped/2warnings、230.96s，exit0；六CUDA条件与三真实fixture跳过
  不算通过，两Lightning无Trainer警告保留。命令使用 `.venv/bin/python -B -m pytest -q -p no:cacheprovider`，
  CUDA不可见/禁pycache/仓外tmp；日志 SHA256 `2ec0741610d13c681dc27c9ff68e818975ee1ca9d56f379aabfb7eff0c534dab`。
  本会话此前420定向/53.87s/0skip与3399只读检查核齐，但不累加为独立科学样本；MetPy/新GPU/真实17通道
  CPU端到端未执行。工具态另存 `/tmp/r7_m3_registration_resume_u8t215bd/`，不提交旧/新运行产物。
- **当前出口**：科学 N2a/paused 与 any-unresolved、原失败/全额账本不变，正式精确登记CI尚未取得；
  N3/N4/N5、main、issue关闭与最终goal完成均未执行或自行裁定。
- **B提交前实跑**：治理/hooks/conventions/planner/campaign/goal/index/M3补测定向532passed/37.67s、0skip；
  37阻断0违规、campaign0fail4历史notes、两goal0失败、index/brief18和两侧空白通过。纯只读18项身份/
  E统计/359活跃Python AST核查全过，回执SHA256 `d29d610b276b9f6189749fb5f17a6c55e7aed6bb1df3d86711b40ff5377c1d0c`；
  E统计实际235条/全体155/抽样21/单点59/已确认234/推测1。用户配置/两无关文件与全部385实验文件不变。
- **增量扫描仍在的事实**：Write仓外核验草稿 `verify_registration.py` 被Mimosa报告动态子进程调用的
  命令注入模式并拒绝，整条Write未执行；按工具反馈删除候选中所有subprocess/动态命令后，纯只读
  `verify_registration_readonly.py` 经正常Write成功，实跑上述18检查。未修改安全设置或绕过扫描，
  也未判旧拒绝为误报；用户开关false不证明全部扫描链停用，正常commit若被拒绝仍立即停止。
