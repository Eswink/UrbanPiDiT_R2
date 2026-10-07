# S3：原 deep-K 目标与最终预报评分配对诊断

2026-10-07；`scientific_claim: false`、`scientific_pass: false`。
本轮是已曝光 train/development 病例上的描述性诊断，不是候选确认或泛化证据。
科学合同只引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，不修改其中门槛。
当前 goal 为 `docs/goals/s3-objective-forecast-score-diagnostic.md`；关联 #76，未关闭 issue。

## 1. 实际执行与范围

首次完整执行 `outputs/r7_s3_objective_forecast_20261007_attempt01/`：
update0/20/80 × 原2021四季 train 病例与2022四季 val/development 病例，共24对。
每对从同一 checkpoint/case 出发，分别运行原12步 train/enable_grad deep-K 目标与
普通12步 eval/no_grad 最终预测，共576次物理 transition。K4、FP32、原物理权重和
initial+allK监督不变；未来真值只用于监督/评分，未进 forward 输入。

- 没有 optimizer、backward、梯度更新或 checkpoint 续训；旧80更新和精确反算不重复。
- 新作用域 val_read=true；不回写旧 fixed-case 的 val_read=false。
- 2023 test 未科学评分，确认 r=0；未进入 S4，也未获得科学接受。
- 五 lead 为6/12/24/48/72h，全部17变量及 full/interior_1/edge_1 保留。
- 逐变量物理 MSE/RMSE/skill/ACC 完整1530行导出为
  `outputs/r7_s3_objective_forecast_analysis_20261007_attempt01/all_17_by_5_by_3.csv`，
  SHA `7be8f7e5e9a30577d1c5a94134503b24f6d2d99684bd030c7dd0007b0e1dc7b2`。
  不平均不同物理单位，先平均MSE再开方，ACC先汇总充分统计再除。

## 2. 已确认读数与解释边界

update80相对0的原 normalized deep-K 目标变化：

| 原病例 | update0 | update80 | 相对变化 |
| --- | ---: | ---: | ---: |
| train2021 Jan（拟合例） | 1.2963442802 | 0.5330567360 | −58.8800% |
| train2021 Apr | 1.2238487005 | 1.9121105671 | +56.2375% |
| train2021 Jul | 0.6454952955 | 0.7370198369 | +14.1790% |
| train2021 Oct | 1.1550655365 | 1.3659410477 | +18.2566% |
| val2022 Jan | 0.8380196095 | 1.1590720415 | +38.3108% |
| val2022 Apr | 0.7385417223 | 1.4291691780 | +93.5123% |
| val2022 Jul | 0.9373079538 | 1.0208476782 | +8.9127% |
| val2022 Oct | 0.8843467832 | 1.5424988270 | +74.4224% |

拟合Jan例的最终full预报85个变量×lead cell在20/80端点分别改善72/83、变差13/2；
t2m五lead在两个端点均改善。但其余三train的update80最终full预报仅73/255改善、182变差；
四val仅99/340改善、241变差，t2m为2/20改善、18变差。这些相关cell计数只是描述，
不是独立样本数、显著性或科学判据。

### t2m/full：四val等初始化权重 pooled 分数

| lead h | RMSE update0 K | RMSE update20 K | RMSE update80 K | skill update0 | skill update80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 1.937445 | 1.967875 | 1.982635 | +0.388881 | +0.360040 |
| 12 | 2.041813 | 2.165353 | 2.096060 | +0.240510 | +0.199618 |
| 24 | 2.678474 | 3.315476 | 3.331548 | +0.437961 | +0.130471 |
| 48 | 3.744069 | 4.401595 | 5.567815 | −0.335360 | −1.953115 |
| 72 | 4.411828 | 4.851880 | 7.367494 | −1.208763 | −5.159587 |

四train pooled update0→80的t2m RMSE为
1.828124→1.824975、2.576722→2.619575、3.952309→3.696168、
5.801766→6.163532、4.932374→6.692067 K；好拟合例可以掩盖其余病例退化，不能只报pooled。
原train-only climatology全部均值身份匹配，所有报告cell无skill undefined；
climatology自身ACC仍为零forecast anomaly的null，不用epsilon替代。

### 目标、draft 与普通评分路径

24对的 train-final 与 eval-final 最大 normalized绝对差均为0，final tensor SHA逐对完全相同。
没有独立终端head；原最后draft就是Y_K并进入下一物理步。拟合Jan例的final-draft-only
加权 normalized误差1.3000736852→0.5299885112，initial-draft-only
1.2927227397→0.5579911257：下降不只是早期draft的平均伪影，最终预测也学会了这个病例。
FP64分解与native FP32不是逐位等价；全部24×12 residual最大绝对值
`9.053387040047767e-08`，原native loss/total及分解差均原样保存。

**已确认：**这次单病例更新有同病例目标与最终预测响应，但其他原train/val病例多数退化；
四val的全部原目标均上升，不能据此扩完整val或把update80晋为候选。
**推测：**病例记忆/跨病例干扰与训练数据制度是下一可证伪问题；本诊断没有区分容量不足、
季节覆盖、梯度干扰和优化机制的因果，也没有证明任何新方法有效。
不默认加Jan剂量、seed、clip或开关，不重启已注册negative路线。

## 3. 代码、完整contract与数据身份

| 身份 | SHA256 / commit |
| --- | --- |
| 兼容code.zip | `72953e44fa814db1e1da4d7f5c7505bcb878dc2a9765bc9f8875022a822b6ea5` |
| archive commit | `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99` |
| model digest | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217` |
| training digest | `24dde71ba380312f2c201c8f5fd2cabb9c884585bb0ce904fdee768fc5f675b9` |
| final thin bundle manifest | `6ca336511c8eda4d2d04f8bd34806387db1af0cc3dd0546b487ca5f649714254` |
| 新external pins | `31b7d5d8a4afc2c24cc4728c65391e8c91022e1156914ebba441027544a70959` |
| complete new80 contract | `954cef09cae2a81b6933340dbe8b2e8a4aa987719dc1d14b9bceb6e6e59b6ab6` |
| bound contract file | `9fcf23198a91bc2538f9d6cb05790f0fda6406d09f1e37ef5a9682de971ab93a` |
| full source540856239B | `bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8` |
| source preflight | `1b914a216bb3a79e044c316749bfad45a2a27bcd5df8bd07cd5cbf9ce4bd58e8` |
| train identity | `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac` |
| val identity | `0c34a887216b02b41716e7837f5be4d515ad2027e7b4a4728cbef15accd02d5a` |
| climatology mean identity | `c633df585b1894704d942e24305047a74ff7af2460cbc24ca2ebe7be1948a205` |

普通r7-local-v1 loader按完整new80 body/签名与model digest核三个标准pt，不使用旧固定1600/200_restore。
归档692 Python entries执行，不用当前checkout替换。optimizer/RNG/cursor是opaque读取资格，
未恢复运行时optimizer/RNG。每pair state与RNG前后相同、`.grad None`、backward_calls0。
原2021四train下中位病例manifest1941/2059/2177/2295，2022四val53/171/289/407；
原全部12 targets/history/calendar/grid与full-window inventory逐项核，统计仅fit原train。

## 4. 协议、结果与实际进程/成本

| 文件/身份 | SHA256 |
| --- | --- |
| preparation protocol canonical | `fa93763b9dbffa6190db02749e8fbeb311fdf78cc97534e19218ef301977fdc1` |
| final protocol canonical | `26a17743b50c159b4b5e1e70703c89c3a9dd0ee56aaf4efebe4e687663d2a59d` |
| final protocol bytes | `ebad8a894dd5d42947d39d43359f1706f977059bec954a9e9b9af1e446cb756a` |
| prepare receipt | `2954fb34745ebb89e319188a802ef064eee1d3442ee28ba56b02898ff88999dd` |
| result | `ca8919139c6999d9a7c2532ee91e2e2eae6dd37fb830f424ab5d5a12999e9d9a` |
| attempt | `84e3a756ee9a1b021f1ddc92e319887cde2625013700341a2f8acbf8f676c289` |
| reading receipt | `5bdb162152b736beba1436e377df2a7a3acc94ceee1862c2b1979330ef5922f4` |
| artifact inventory | `ddbb95947afa4078f97731dc08108e47539fc8082414bf48ed5d3db7a3094115` |
| endhash receipt | `f4484678afe9cee27fb64f9b6b2b5c650edb868d648b830926cdd02df763fc9e` |
| analysis JSON | `9dd785444870ac0225a85669d4c499be8d8a537b6051cd309788f6eeb1984221` |

实际整轮planned3600/hard10800/reserve180先冻。CPUprepare671.480996秒、
diagnostic1205.413004秒、CPUreading6.282190秒；三worker均exit0/reaped/signals=[]。
inner attempt1888.921116秒，父supervisor含最终inventory和reaping整轮1891.377279秒，
soft/hard超0。按ceil1892秒/3600、向上四位记 **0.5256GPU-h**，不是只算训练或forward。

startup与每spawn即时GPU1 UUID `GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced`，
free25280118784B≥known2491416576+2GiB=4638900224B；新reservedpeak2409627648B小于known，
下一门仍带更大known值。allocatedpeak2268912128B仅归本进程，不干预邻居。
父回执在 `/tmp/r7_s3_objective_launch_control_20261007_attempt01/receipt.json`；
网络请求0、新下载/派生发布0、optimizer/backward/update0。结果最高configuration-only。

## 5. CPU资格、失败与准备成本

- producer最终完整CPU19/19、原指标88/88实际通过，0skip；初次独立55pass/33FALSE_ACCEPT原样保留。
  required expected units/std、exact5×17、typed counts、原病例outer unique24、region complement与
  ACC Cauchy domain最小修复；pool只能证计数，不伪称其验证uniqueID。
- 初始原指标snapshot `0565af9d…`，33失败独立真实核验，不回写为绿。CPU attempt02两fullarea错误、
  metrics revision01全部setup错误、错误archive member查询与所有过程日志保留。
- 最终独立prelaunch receipt `600a7abf8b1e98f05a7b19ffc8e4a73e88cf2c6b9360ec5187122261a4999aff`：
  19CPU、38rows通过；原92项91pass/1 reviewer TMP接线fail/0skip，失败保留，单环境补测+三gate4/4。
  总153calls=152pass+1保留fail，不冒称单all-pass套件；33producer错误接受全部拒绝。
- 原准备whole7178.884820秒、soft3600超3578.884820/hard10800超0/reserve180，0GPU。
  producer4766.362815秒、soft超1166.362815/hard7200超0；独立prelaunch1515.378506秒、
  soft超915.378506/hard1800超0/reserve90。这些是重叠准备墙钟，不能求和冒称额外GPU。
- 准备证据zip `43183ba86490d06d03dee9e49cf2530fde8705b82c85c4674559e04b96eaeca3`，
  独立prelaunch zip `fd71dafbddb7fb5ccbc23bec1959f67c0fcba4eebc601d86502028460033e2ba`；
  位于 `outputs/r7_s3_objective_preparation_20261007_attempt01/` 与
  `outputs/r7_s3_objective_prelaunch_20261007_attempt01/`，归档本身0GPU。
- 精确准备文档提交3c354e642682b3767f4f94771104ea7c7aa937b8主CI37629370467必要steps成功。
  首匿名API403 inspector failed-incomplete保留，后exact run/jobs200验证闭环，不是测试失败。
  此CI尚不覆盖本页与最终登记提交；工程CI不表示天气胜出。

## 6. 独立终态与登记状态

独立NumPy/标准库saved-facts helper原准备47/47真实合成测试通过，0skip；
首46/47的helper字段假设错误及服务overloaded中断保留，修自身schema后完整重跑。
helper `b0533ee6cb2d41afe2892408b2dd6e16a52f97668dc62138db0d50ddd13fe2cd`，
准备1661.467504秒、soft600超1061.467504/hard1800剩138.532496/reserve90，0GPU。
真实终态已用同字节helper单次执行，exit0、allfailed_mismatches=[]；新阶段600/1800/reserve90、whole194.56秒、soft/hard超0、时钟不reset。回执 `/tmp/r7_s3_objective_terminal_helper_20261007_attempt01/results/terminal01_stage/receipt.json` SHA `c60add04a8cd1952b637524fb8b0383b0183dd6a8dfd07c3adec2ec7f83eff0e`，独立结果 SHA `62df597aab9c1c5bf7db19fd51bc34251c433d92533cd08a392f851b79ab2f4d`。754 allowed artifact/parent文件首末SHA相同，原六case JSON与外锚全部同；24pairs/288原生loss/24480draftcell/每path1530聚合metriccell、native round32总数、FP64分解/物理std²/三区域补集/ACC先pool/undefined全部核齐。独立重算最大分解残差9.053387051149997e−8，最大相对1.459731433580313e−7，不作逐位native等价；producer保存差的最末位不同属独立算术顺序描述，未改科学/旧exact门。父whole1891.377279秒及保守0.5256GPU-h独立核齐，父/三workerexit0reaped/signals=[]，三即时GPU门与maxpeakcarry齐。本审阅仅保存事实，不follow真实source/store/checkpoint/weather，也未新forward/GPU/联网；runtime source资格仍由实际prepare与fieldworker报告，不伪称独立重读。最高configuration-only，非科学或最终goal接受。

独立准备/终态helper、原病例快照、父cost/全部后续协调证据已归档
`outputs/r7_s3_objective_terminal_20261007_attempt01/terminal_evidence.zip`，
SHA `c0a90f893e0c3184b8ae6c8a92f919e8fed81a044178ed094333ede8c194b281`；
60files/28288074输入bytes、归档0.654072秒、0GPU。复制证据不追加天气或科学通过。

## 7. 影响、限制与接续

没有活跃model/training接口、旧数据/产物、科学门、依赖或凭据变更。
新TEMP API必传expected identity，归档兼容性仍受强hash保护。
共驻/离线/owned guards是合作式意外访问防护，不是OS沙箱；记录signals空不是全系统证明。
八已曝光病例、单seed41、四季30日块不是完整未见年度；cell相关、无同时置信区间，
不能由本轮宣布泛化、clip/容量/季节因果或最终goal完成。

下一项是在S3按实际跨病例负响应研究数据制度/信息路由的不同可证伪干预，保留同数据气候态和incumbent控制；
本单病例80端点不作为候选，不默认提高剂量，不据四开发例退化扩完整val，也不重复旧exact replay。
