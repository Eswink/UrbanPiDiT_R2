# V2 已知历史与源位置契约补齐：工程接受进度

日期2026-10-03，决策0035/计划0015，`scientific_claim:false`。本页不改旧M3/RW-B/N1负面，不把
B的旧模型分数当新功能效果。B已原样failed封印、源码运行期间未变后，主链才合法集成新实现。

## 1. 实际接口与边界

- `known_context_inputs=False` 严格bool，开启须spacetime、真实Gregorian year/day/hour、坐标、lead及
  `history_offsets_hours`。sample[T]/batch[B,T] signed UTC差，从已验证实际history timestamps在线导出；
  regular正cadence、末项0，保留slot顺序，Native历史间隔不绑定lead，物理rollout开启时须cadence==step。
- 原八feature网络/参数不变；新Linear residual最后isolated_stream构造，T=2时12features/13*dim参数。
  有序每slot annual sin/cos、local mean-solar sin/cos、offset/24，再valid local-solar sin/cos。
  地方平太阳时来自UTC+经度，原生网格harmonic先算再同padding/pooling，不先平均日界线经度角。
- 关闭模型路径不读取/验证offset，不构造residual或加零替代；所有实际开路径缺metadata拒绝，不取
  旧init_year或系统clock。shuffled日期/offset共同滚动，lead原位；constant先严格验证再丢信息。
- `source_position_markers=False` 严格bool；开只须位置reader，独立于roles/known/draft-query开关。
  P/Generic latent临时source key为(C+fixedpos)+role与(E(Y)+fixedpos)+role，无draft时仍context标记。
  不原地/不累加到carry，不新增encoder、attention或参数；query仍LN(C+E(Y))+pos，不重复encode。
- producer、schema、forecast白名单、fixed/streamed/adaptive/physical training/evaluation同步；
  内部K不推进calendar/offset，第二物理history为prediction，relative offsets保持[-6,0]。

## 2. 已实跑与原失败保留

| 实际范围 | 结果 | 限制 |
| --- | --- | --- |
| 临时known helper/Native/producer/控制/日期/梯度/off poison/BF16/state restore | 36passed，5.57s | fixture工程，非GPU天气结果 |
| 临时source完整key/P/Q/read joint等式、错配反证、Generic映射/F32-BF16/odd/K/streamed/adaptive | 20passed，1.79s | 同固定参数/role/position，非pooled moments |
| source新测试及整个旧shared-step角色绑定修复 | 30passed，2.79s | source identity不等同空间信息 |
| 主链active集成五文件完整定向 | 220passed，50.18s | 真.venv入口，无skip；仍非fullsuite或新CI |
| B失败登记治理合集 | 409passed，31.13s | conventions/hooks/campaign/goal/planner/index反证，不作科学证据 |

四默认关闭case的参数/state/RNG/全output/梯度/streamed/adaptive前后JSON逐字节相同，SHA256
7c4bd4badbe453497e25d864de023d82a6d3c71393f57350793319de249d24c2；这是同recipe单机工程范围。

原失败全部保存：known attempt01实现耗时超过1800硬上限，pytest未spawn；attempt02=29pass/3fail
为dateline test误用historical slot cosine索引，新轮仅修当前00Z slot1的精确索引，不放宽oracle。
attempt04=134pass/2fail、source组合189pass/3fail来自旧fixture缺新增可选offset；主链为真实fixture
补offset、开启known causality路径及白名单集合，不删弱时间/K/poison/梯度断言。
主链第一隔离验证把.venv Python symlink resolve成conda解释器，2collection error/0pass；原日志保留，
不安装依赖或改环境，纠正为.venv实际入口后220pass。

旧role反证每次重采role造成假差，现固定一组role：同半重排仍set不变，跨source内容交换改变绑定；
原两assert保留，另补同半invariance/unmarked raw-set invariance与equal-role故障同判据失败。
它不证明位置关联，完整token位置共同重排/错配由独立source tests验证。

## 3. 身份与产物

临时活跃工程复制根 `/tmp/r7_known_context_active_km7rhsvo`，不是legacy/冻结快照的活跃import。
原B封印后才复制18新/修改源与fixture，主链集成回执
`outputs/r7_v2_remaining_acceptance_20261003/known_context_active_integration.json` SHA256
b28fba84dc380392f44f2d7023f5e2e755332e576aa84cbb51779eb9ea4b3fb2。
角色namedtest另外单独强hash复制2aa52a66…f444a5；不改原B protocol/zip/checkpoint/CSV。

owner全部原协议/通过/失败stdout/result/source ZIP保留于
`outputs/r7_v2_remaining_acceptance_20261003/known_context_engineering/`；known final05源码zip SHA256
4bb88bbb3ab7e350128c7f7855fbe5ea6f798a6a40fe321d46ffbade98d1f622，source角色回执SHA256
ff85b870f67771a53868734139dee47899b7220519a060c172b959f0021df885。原/tmp receipt绝对路径不回写，
只复制原字节并另列映射。source文件≤407行/函数≤152，未增加600/200例外或放宽任何阈值。

## 4. 独立反证发现的未通过项

独立CPU审阅不接受当前工程整包：初轮34项直接探针31通过、3失败；新定向文件20+34通过不能
抵消失败。原HEAD616b活跃Git blob与新默认关闭的state/输出/loss/全部参数及history梯度/RNG精确
一致，耦合Process/Generic FP32/BF16两步与L12新projection/role/reader梯度也通过。

确认两处真实实现缺陷：`r7_halting.py`活跃batch从B3缩为样本[0,2]后传入未切片的B3 anchor，
local solver提案拒绝；known offset检查把合法20分钟regular int64 UTC timestamps派生的
[-1,-2/3,-1/3,0]浮点差值要求严格相等，误拒合法Native cadence。两缺陷已活跃修复：anchor仅增加[selected]切片，history regularity仅允许原dtype半ULP传播的
表示误差，不使用宽rtol/atol。owner分别16定向+3repeat、50定向通过（两checkpoint病例未在该owner
范围执行，不算通过）；原source及故障注入的失败原样保留。新独立窄复验46项通过，仅接受两修复和有限FP32范围，不升级原整体FAIL；不改运行前科学容忍、不删除失败。原审阅日志/协议保存于
`/tmp/r7_independent_known_source_review_20261003_eavOSK/`，源码前后hash一致。

第三项BF16 fixed-vs-streamed proposal decoder梯度差异在旧RW-B flags与新包均出现，FP32全部
梯度按原3e-4/3e-6通过；BF16新权重/偏置最大绝对差6.10e-5/9.16e-5。两种loss表达式均保留差异，
forecast与draft误差精确一致，尚不能将其归因于新输入或证明streamed BF16梯度等价。
原容差不放宽；诊断`gradient_diagnostic_results.json`保留。进一步只改变诊断的fixed autocast
`cache_enabled=False`后，旧与新包全部参数梯度在同一3e-4/3e-6下通过，forecast精确不变；这将
差异限定于共享BF16权重cast缓存/分步adjoint累积舍入，而不是loss代数或新projection/source。
`autocast_cache_results.json`与原失败并存，此诊断不修改已冻B或probe、不等同原BF16 assertion通过。
另33项固定初始化role/完整joint key/position错配/L12 detach/控制非法输入/default-off反证全部通过。
C仍按既定FP32 full BPTT执行。

## 5. 尚未接受

修复身份：known helper`1eeac07ef5f0a833ee3064ccb4605e4f6937273e1e8f246fe4e0b4fac6afeac1`，
known test`be8c3b2891e883d702cf0e41160077d1ebc5866bf667541ded623d9b4c24ed11`；halting
`8fb98144b728418ad852e32c3e320badec0277b128a2cdc06999c6073bf83eb3`，新active-anchor test
`d98faece4984b2de224693e5d1b0d4d6967478ff6e543f646a374ea23ab86775`。anchor原版与unsliced故障在
同一个真实positive谓词均失败，样本[0,2]继续、最终深度[4,1,4]与同state固定K逐样本forecast差均0；
force-full bit-equal。独立owner整轮1317.82秒/软超417.82秒、硬1800未截断，回执SHA
`e15bff8004b1e5f074f5786d8b63642df3c9def2df7a165f179147f9e7cccc97`；只新CPU工程，无新GPU。

独立active修复窄复验已完成：27独立+16 cadence-only+3 active-anchor共46passed、0fail/error/skip；另外36明确deselected不算通过，protocol body `2a02cf7c241aea250c893db8503682ea0b2eb06d83a3b75960cbe4b82db73815`，closeout SHA `c3a1a45899b538d969f3ccfa76a88d7d6c7dadadf043fd1565d30a27c46a82b4`，整轮351.819秒/软超0。原两失败predicate原样通过，6个FP32参数case/14成对K检查state/output/loss/全部参数及history梯度/RNG逐位相同；源与旧产物无漂移。主链七文件完整定向249passed/51.03秒、整轮52.933秒，包含两checkpoint序列化case，protocol `648cc75500c8cedcd2d7b0d9768043459711723fbcc840a35cdc59dd5ea427a4`，stdout SHA `22d750fe654af3c3cde8f654296cb9795e21c22930508932be2b5a8c7e04bb90`。

稳定fullsuite/installed-wheel已实际完成3467passed/9skipped/3warnings、903.06秒，含完整reader/latent反馈梯度反证文件及显式known module/七CLI安装检查。整轮905.674715秒、1800软/3600硬、overrun0、source_changed[]；protocol `3703be5ccf2eaf7788742f7e0c108117793060b1510194301cca3be1fd31388b`、stdout `f0d2e99bdfedfb9ebcea6dbe48fa2988a3ede837fa19ebf2514d3055b8e0fc86`。六CUDA与三optional真实fixture跳过不算通过，原5fail/3412pass全量失败保留。最终model digest `551261c4a501a6f9727fef2a712b40ab67ec51adaa460298a6b5b59bf8ae5dc1`；精确新CI尚未完成。新C完整规格/target←anchor mapping已CPU核三seed/三target完整覆盖（115/155/151 state keys）。B统计attempt01漏worker.log的完整库存资格FAIL保留；新attempt02独立受限接受全部113文件/111 pins及数值，按冻结primary选择L6，审计SHA `0f965f2e7fd8dff509637835866737f3961003eb56b434704abf515676bf4e8b`，只接受元数据统计，原B仍failed/finalizedfalse，见R7_74_STATISTICS_COMPLEMENT。真实FP32/BF16新包probe、C9训练/135验证及adaptive实裁尚未做，不先写PASS或收益。
旧B/precision checkpoint须用各自归档，不拿新model digest加载旧权重。

新known projection增加容量/成本，C旧pooled vs新包只解释整体包，Process vs完整Generic才是同信息
对照；无辅助监督可能函数/梯度等价，不能隐藏负面或据此机械改设计。单冬季开发数据与三seed不作
显著性、科学泛化或SOTA。无data/raw/interim/processed/store改写、真实下载/发布、依赖安装、
安全/凭据变化、付费/独占/邻居信号；现新增工作0GPU-h，最终goal不自行判完成。
