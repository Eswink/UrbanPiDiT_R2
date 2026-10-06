# S3 rollout 训练文献核对（2026-10-06）

本页是方法假设的来源台账，不是本模型的实验证据。初次 web-researcher 检索后，
又对以下三个一手正文做了一次只取原文的独立核对；没有移植上游代码、安装或训练上游模型。
访问日期均为 **2026-10-06**。ar5iv HTML 的精确 arXiv revision 未从版本页面再次确认，
因此不声称这些页面属于特定 v2/v3；下列编号和章节为实际核对对象。

## 核对到的配方

| 一手来源 | 正文位置 | 核对事实 |
| --- | --- | --- |
| [GraphCast, arXiv 2212.12794](https://ar5iv.labs.arxiv.org/html/2212.12794) | Supplements §4.5, Curriculum training schedule | 单步阶段 1000 + 299,000 更新；随后 11,000 更新展开长度从 2 到 12，每 1000 更新增加一步，固定学习率 3e-7 |
| [Stormer, arXiv 2312.03876](https://ar5iv.labs.arxiv.org/html/2312.03876) | Appendix B.2.2、B.2.4；§3 | 三阶段 K=1/4/8；100 epoch 单步、后两阶段各 20 epoch，学习率分别 5e-6/5e-7，接续前一阶段 checkpoint |
| [Keisler 2022, arXiv 2202.07575](https://ar5iv.labs.arxiv.org/html/2202.07575) | §3.3.2, Multi-step loss | 4→8→12 步课程；对照“全程 4 步损失”仅略差；不是先纯单步训练的证据 |

短原文核对片段：

- GraphCast：“The third phase consisted of 11,000 gradient descent updates”；
  “the number of autoregressive steps increased from 2 to 12”；“a fixed learning rate of 3e-7”。
- Stormer：“the number of rollout steps K is equal to 1, 4, and 8”；
  “20 epochs with a learning rate of 5e-6 and 5e-7, respectively”。
- Keisler：“4, 8, and 12-step losses”；“only slightly worse results when using a 4-step loss throughout”。

## 对初次检索转述的更正

1. “rollout 微调占更新预算 <5%”只能由 GraphCast 的更新数支持，不能泛化到所有论文。
   Stormer 的微调为 40/140 epoch；且 epoch、更新、样本量与 FLOPs 不是同一个分母。
2. Keisler 的全程 4 步是对照，其主要配方是渐进 4/8/12 步；没有核对到纯单步预训练阶段。
3. 这三篇没有提供本仓需要的 **3.1M 参数、2340 训练窗口、等 FLOPs 的 warm-start vs
   from-scratch** 对照。文献不能证明本模型的 200 更新两步微调会改善 48–72h，也不能证明
   1600 更新已收敛。
4. 未核对来源的版本号、其它模型配方、噪声/stop-gradient/scheduled sampling 细节不作为本轮依据。

## 本仓使用边界

仅借鉴 GraphCast/Stormer 的“单步阶段后再多步微调”组织形态。使用本仓既有
`training/r7_autoregressive_rollout.py` 的两步全 BPTT 和全部草稿深监督，
没有搬运模型结构、上游代码或其数值超参数；200 更新、LR 2e-5 和 lambda12=0.5
是本仓独立冻结的假设剂量，不叫论文复现。初始权重、控制、时效、判据与限制以
每次 `protocol.json` 为准；失败或负结果均保留，不能从这些引用推导 PASS。
