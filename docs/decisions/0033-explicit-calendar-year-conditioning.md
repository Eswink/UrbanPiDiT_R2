# 0033 显式已知日历年修正跨年条件，保留历史时间路径

- **日期**：2026-10-03
- **状态**：accepted

## Context

#71要求年末、闰年、UTC跨日和batch多日期的直接验证。活跃时空条件仅有经纬度、init小时与
init_day_of_year，按365.25天取年相位；缺calendar year无法区分平/闰年及精确wrap。
CPU解析探针发现Dec31 18Z+6h与次年Jan1 00Z相位不同，不能把历史平均年周期误称Gregorian正确。
原tests特意将init_year当未读字段，不能为了新验证弱化这个信息白名单反证或回改旧训练身份。
用户本次明确要求暴露缺陷后自主修活跃实现；这属0030/0032方向内工程选择。

## Decision

1. 引入可选的显式已知字段 `init_calendar_year`，来自样本UTC初始化日期，不从系统clock推断。
   `init_year`仍保留旧metadata/未读语义。旧四项 `SPACETIME_INPUT_FIELDS`维持必需集合，另立
   `CALENDAR_INPUT_FIELDS`可选集合，forecast白名单与producer/rollout同步。
2. 未提供新字段时保留旧365.25运算逐位路径，不改变历史父checkpoint的函数或伪造其身份。
   提供时严格核year/day/hour匹配及finite/整年/合法日期，采用精确Gregorian年份长度计算valid
   日期和annual/diurnal相位。字段存在但非法必须拒绝，不回退猜测；不增加未来观测输入。
3. 纯torch日期算术推进，物理step更新year/day/hour且每步lead=6；内部K不推进日期。
   年末/闰日/跨日/东西经/batch/padding与故障注入测试先保留缺陷FAIL，再修活跃实现复验。
   新实验明确声明calendar route；旧归档评估仍用原code.zip，不把旧路径称已修。
4. 新model digest显式记录，父导入先接受旧归档身份再映射到新模型，不修改旧加载器比对逻辑。
   本修正是时间接口正确性，不是预报改善证据；其预测影响属于新冻结实验而非历史结果修订。

## Consequences

**变容易的：** 能直接验证真实Gregorian年末和闰年，并让每次物理rollout准确更新已知日期；
旧模型输入语义与逐位回归仍保留，未来标签与系统时间继续不得进入前向。

**变难 / 代价（如实列出）：** 并存显式日历与历史平均年周期两种路径，producer/白名单/rollout
必须一致声明；新增模型源码改变digest，不能直接加载旧归档。即使修正时间，预报可能无收益，
新实验的paired结果不能冒称历史同函数复现。需要更多非法日期/跨年/batch反证与身份导入记录。

**备选方案与否决理由：** 用现有init_year悄悄成为模型输入会改变旧白名单测试与旧行为，不采用。
忽略年末差或放宽连续性断言是掩盖真实缺陷，不采用。用系统clock填年份/未来valid truth倒推，
或按366固定年长，会分别泄漏/违反已知信息或误处理平年，不采用。
