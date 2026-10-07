# S3 开发诊断与后续假设：公开来源核对

访问日期2026-10-07；`scientific_claim: false`。本页记录提案及官方机制参照，
不是天气实验结果或科学接受门。唯一科学合同为 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`。
当前实际诊断见 `docs/goals/s3-objective-forecast-score-diagnostic.md`。

## 1. issue 实际正文、评论与状态

web-researcher 主通道实际失败：`upstream stream idle for 3m0s`，duration1001795ms。
备选通道实际读取了部分一手正文/评论，但完整#76正文未返回且若干页面超时；
WebSearch错误 `Provider API kind openai does not encode provider-native WebSearch`，备选改WebFetch。
主链按合法降级用匿名curl取原始JSON，未使用gh/PAT、未写GitHub/关闭issue，
全部9次请求HTTP200，无本轮curl重试；原子摘要不是原文证据。

原字节归档：`outputs/web_research/r7_s3_issues_20261007_attempt01/`，包含排他protocol、
逐请求URL/UTC/HTTP/bytes/hash、完整正文、全部comments和state=all最近20项。
网络body259290 bytes，整轮11.347789s，planned600/hard1800/reserve90、overrun0、0GPU。
receipt SHA `ea1267ccd305ff43251f77258142b5d729cdaf91754efe7ee23d7b9395dac5eb`。
这是API正文/评论获取口径，不包含子代理WebFetch未计量网络字节。

| issue | 一手URL | 状态 / updated | 评论覆盖 | 原JSON SHA256 |
| --- | --- | --- | --- | --- |
| 76 | https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/76 | open / 2026-10-04T17:09:09Z | 1/1 | `140c4322e3739a9e627b1c9cd97872cbc28cc7fd0c1115c331e433f84c9cb9e2` |
| 77 | https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/77 | open / 2026-10-04T17:07:17Z | 0/0 | `e0fd175af4c86c5b0bf105c2bd277f0079167d56d19a817b2dbec1e2d4f3a3b4` |
| 78 | https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/78 | open / 2026-10-04T17:08:07Z | 1/1 | `2d7650038cda89f79ebdab5c5662881ade23e4fd7b6edbf46737ee202cf72ee6` |
| 79 | https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues/79 | open / 2026-10-04T17:08:52Z | 0/0 | `b413697c845d1d5d1dc3ffb2326b3ad2d87a9f86a688b03b324af0d86ae7a364` |

四issue作者API均Eswink，GPT6PRO来源是用户说明，不作为事实权威。
主链实读完整body字符数3593/2037/2245/2810，非搜索摘要；评论条数与declared comments一致。

- **#76原文与当前作用：**“matched_generic_v2_equivalence单独作为ablation列，不能混成‘必须战胜自己的等价副本’”；
  “先current configuration/learning curves/预测误差bias与anomaly decomposition定位主要差距，再定有界pilot”；
  “不以Issue数量或测试数量判模型成功；也不为证明科学谨慎而永久停留在baseline审计”。
  本页采用其性能/归因分离与诊断优先建议，不用2026-10-04旧TODO重做S0–S2或替代当前科学合同。
- **#77原文与停止边界：**“不能静默修改旧checkpoint语义”；“若无提升，关闭为负实验/保留legacy默认；不无限枚举频带到某个val提升”。
  已登记PE负/未决证据由campaign索引权威，不因issue仍open而重跑；本窗口没有新关闭授权。
- **#78原文与停止边界：**R-A、R-B、训练配方分开；“这是需要实测的假设，不是已证明的主要错误”；
  不机械重跑#74负two-step。旧scale/loss/剂量结果保持。单位评论原文：
  “直接用 normalized 张量时必须写为 `(s_c/d_c) * (Yhat_norm-Ytarget_norm)`。禁止把 normalized 误差除以物理单位 d_c”。
  来源 https://github.com/Eswink/UrbanPiDiT_R2/issues/78#issuecomment-5982409045 ，comment JSON SHA
  `0327191c195a023fa6ae0e8e9258e92efd18c339ac34fea97d18d4a7e80410f2`。
- **#79原文与当前作用：**“同容量/同诊断信息但不作类型约束的通用融合”；“B→C才考察typed归纳偏置”；
  “不读取真实未来Y*”；“负面时撤回该候选的科学主张，保留原V2性能路线”。
  支持旧控制纪律，不推typed必须提高当前候选性能、不重做aux扫参。
- #76评论来源 https://github.com/Eswink/UrbanPiDiT_R2/issues/76#issuecomment-5982417254 ，
  仍称三项“有界假设”，不是永久matrix或全部叠加。comment JSON SHA
  `9d5bb242fd5e21a78330725caac4e26481c7b20cd14468e6008be6af0de7e834`。

近期列表一手URL：
https://api.github.com/repos/Eswink/UrbanPiDiT_R2/issues?state=all&sort=created&direction=desc&per_page=20 。
2026-10-07T13:19:22Z实读为#79至#60，全20项非PR；该响应中没有#80+。
这里只声明该访问时点窗口，不把“未出现”说成永久不存在。#70–#75仍closed，不重开旧路线。

## 2. 官方机制参照：事实与本仓推测分开

检索由web-researcher-backup实际完成；主研究通道此前不可用，不改provider配置。
一手原文经第二次定向WebFetch回核；未核到的源码行为不补记忆。

### GraphCast：6h自由递归与直接长序列目标

一手论文正文 https://ar5iv.labs.arxiv.org/html/2212.12794 ，
版本记录 https://arxiv.org/abs/2212.12794 （v2，2023-08-04，CC BY4.0），访问2026-10-07。
第二次取原文确认：

- §2.2：“the temporal resolution of data and forecasts was always Δd=6 hours”。
- §3.1：“we can apply GraphCast iteratively to produce a forecast”。
- §4.2：“GraphCast was trained to minimize an objective function over 12-step forecasts (3 days)”。
- §4.2：“We applied uniform averages across time and batch.”

**确认范围：**论文提供6h自由递归和多步训练参照，不证明本区域/主模型同样有效，
也不将全球外部模型替换为主线。源码raw请求ECONNRESET、API对应路径404，
未确认当前源码commit或实现license，本页仅引用论文事实。

### WRF：区域模式侧边界信息

官方指南
https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/dynamics.html#lateral-boundary-conditions-for-real-data-cases ，
访问2026-10-07，页面©2021，未标WRF版本。第二次回核原文：

- “The first row and column are specified with external model values”。
- “The rows and columns in ‘relax_zone’ have values blended from an external model and WRF.”

**确认范围：**数值区域模式使用外部侧边界与relax blending；未核边界更新频率。
这不证明本65×65神经模型长lead失败由区域外信息不足造成，更不授权未来真值进forward。
可研究方向仅合法initial历史halo/更大inputcontext、同信息受控路由，不采用检索员建议的future-boundary oracle。

### TRM：只作梯度机制参照，不冒称本仓实现等同

固定一手源码
https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/c01103738605ba39d1430519b1ee0c62f4c707f8/models/recursive_reasoning/trm.py ，
commit `c01103738605ba39d1430519b1ee0c62f4c707f8`；仓库MIT声明由官方API一次读取。
访问2026-10-07，首次源码抓取成功、第二次固定页重访60s超时，后者保留未重核限制。
原文函数 `TinyRecursiveReasoningModel_ACTV1_Inner.forward`：前置递归用
`with torch.no_grad():`，末递归有梯度，返回carry `z_H=z_H.detach(), z_L=z_L.detach()`。

**确认范围：**上游递归内/段间截断不等于本仓fullphysical/internalBPTT；
不因此推荐无依据截断或把其QA性能当天气证据。既有本仓差异见 `docs/R7_TRM_DIFFERENCE_TABLE.md`。
本窗口未clone/import/运行上游，不搬代码或声明候选通过。

## 3. 可证伪问题、限制与未做

以下仍是**推测**，不是已确定机制或冻结实验；必须先等当前目标—最终评分真实诊断，
再另写实质差异/正负控制/数据及数字预算protocol，不并行抢GPU或先看test：

1. 如果原目标下降主要在earlydraft而final未改善，监督位置/聚合与最终预报对齐值得独立干预。
2. 如果只同Jan病例改善，研究跨病例梯度干扰/数据制度或合法输入上下文，不能直接认定容量不足。
3. 若四开发例同向支持才扩大完整val；不把singlecase memory当泛化，不默认加updates。

新资料不重启旧PE/scale/typed/aux/two-step，不能替代同数据incumbent/climatology公平控制。
时长/GPU-h无总许可上限，但真实失败/negative止相关假设，软硬/全部成本/独立审阅和S4正式门不变。

未做新架构/训练/数据下载/clone/正式test/独立确认；上述公开事实不产生天气胜负数字。
子代理网络字节与完整wall调用口径未被工具独立计量，仅curl原文留实际bytes和11.347789秒。
科学接受、最终goal完成与新issue关闭均未裁定。
