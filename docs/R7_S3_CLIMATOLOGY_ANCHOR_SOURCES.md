# S3 气候态锚/异常反馈：一手方法资料与未确认范围

`scientific_claim: false`。这是新假设的准备资料，不是运行结果、气候态技巧证据或科学门。访问日期 **2026-10-07**。

## 已核实的一手源码

- 官方历史 GraphCast normalization wrapper：<https://api.github.com/repos/google-deepmind/weathernext/contents/graphcast/normalization.py?ref=08cf73625c9d12bd9aaa038868bcb2fe488f2a22>，commit `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`，blob `1733670b2abbac8b5f07f3a3fb312c0135ad07f7`。该仓库 API 当前将旧 `google-deepmind/graphcast` 路径解析到 `weathernext`；这里引用具名历史 revision，不冒称当前 main 仍有同一路径。
- 主链在备选研究员取到官方代码后再次实际 curl 原API，HTTP200，1.378531秒，未用论文摘要补 Methods；保存源 SHA256 `5b5f95bcf830be1236244fd09cc418caa28c1c970793130e93bccbc98e0e00e7`、API JSON SHA `bf489f0227ed4736e268dbeaacfb9e30e1da1434b60fa9e393c90da3d11442c9`。原文和回执在 `outputs/web-research/r7_climatology_anchor_20261007_attempt01/`，只作只读参考，不 import/执行上游。

实际源码的 `InputsAndResiduals._subtract_input_and_normalize_target` 对名字出现在 inputs 的 target 使用：

```python
last_input = inputs[target.name].isel(time=-1)
target_residual = target_residual - last_input
return normalize(target_residual, self._residual_scales, self._residual_locations)
```

预测恢复时，对同类变量先反变换 residual，再执行 `prediction = prediction + last_input`。名字不在 inputs 的 target/prediction 则直接按普通 scale/location 标准化或反变换。因此这份历史源码是**输入已有变量的时间差分残差参数化**，不是“GraphCast仅输出绝对状态、无输出残差”。最终交付绝对状态与网络预测残差可以同时成立。

本次检查的该文件没有 `climatology` 字面量；这仅是**单文件局部观察**，不能证明整个项目或所有天气模型不采用气候态锚。源码 header 为 Apache-2.0。这里只引极短操作片段作出处核对，没有搬入上游实现或运行其安装/训练。

## 首轮检索失败与未确认

首轮 web-researcher 将摘要/无法取得的 PDF 方法归纳为“三个模型均仅绝对状态”，又未访问官方源码；该摘要**不接受为事实**。备选通道官方代码核对及主链二次HTTP200已明确纠正 GraphCast 部分。Pangu/FourCastNet 的实际 output parameterization 本窗口未完成代码级资格，不能据首轮摘要作全局结论。

备选通道原生 WebSearch 不支持（`Provider API kind openai does not encode provider-native WebSearch`），转官方已知URL WebFetch；raw URL 有 ECONNRESET/timeout，使用 Contents/blob API。没有修改provider、用户配置或安全门；没有天气数据获取。上述失败不被新API成功追认。

## 本仓假设与区别

决策0040研究 `Y0=C_valid+A0(history)`、RW-B proposal 亦从 `C_valid` 解码，并让 feedback 编码 `Y_k-C_valid`；`C_valid` 是原 train-only 月/小时/网格统计。它与输入最后状态的时间差分残差、per-channel norm、旧 `d_c/s_c` 变化量缩放均不同。此处一手参考只帮助区分表示，**不证明该新 package 有效、稳定或优于气候态**，也不替代本仓被冻结的开发停止线或最终科学合同。

未能确认：外部是否已有完全同构的 valid-time climate anchor + internal-K anomaly feedback 方案；本候选实际收益、近时效与守门代价、完整年度泛化及统计确认。以上只能由本仓新的合法有界实例取得证据。
