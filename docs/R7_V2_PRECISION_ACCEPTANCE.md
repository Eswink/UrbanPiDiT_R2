# V2 可微物理训练的真实GPU精度接受

日期2026-10-03。范围：N3工程前置，`scientific_claim:false`。不是B/C天气对照、收敛、泛化或SOTA。
本页不改M3原failed/完整补测paused、不重复其23项评估；执行/预算与共驻按0030/0032。

## 1. 冻结身份与实际运行

- 工程commit `616b029dce569b92bd08295737981512180a1ad3`，工作分支；模型digest
  `0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e`。
- 精确主CI [37133487340](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/37133487340)：
  匿名Jobs API于2026-10-03核九principal及三post/complete步骤全部completed/success。
  回执 `outputs/r7_v2_remaining_acceptance_20261003/engineering_ci_acceptance.json`；
  未下载CI测试日志，17条标签实验未请求/skipped不当通过。
- 实际目录 `outputs/r7_v2_precision_probe_20261003_attempt01/`；CLI `tools/r7_v2_precision_probe.py all`，
  本仓.venv、-I/-B；只消费合法M3 aux_off seed41 update400父、原protocol/code.zip、scale sidecar及M2 trainmanifest。
- 冻结canonical protocol `0829da0251449a840d256878ff5d2687bfcb7681db39c585df6a2f202255f4ba`；
  最早CLI计时，planned900/hard1800秒；85成员源码归档及Git身份，用户三无关dirty文件不进代码闭包。
- 既有source/data/preflight/BUILD_COMPLETE/原父身份均强核，只用前两个train窗口
  `era5z_train_2016010106_p006h`、`era5z_train_2016010112_p006h`。17通道、65×65、batch2、K4，
  不读取封存test，不发布数据或下载；真实train数组不是合成fixture。

## 2. 实际结果（原产物不回写）

| 精度 | 实际优化更新 | 独立分支 | allocator baseline | allocated峰值MiB | reserved峰值MiB | worker秒 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| FP32 | 4 | uninterrupted2 + intentional_resume2 | 0/0 bytes | 987.6782 | 1134 | 84.0124 |
| BF16 | 4 | uninterrupted2 + intentional_resume2 | 0/0 bytes | 631.8799 | 674 | 79.1489 |

两precision各自在同seed同端点的weights/optimizer/RNG/losses严格torch.equal，四acceptance true；
跨precision不要求相等。保存1/恢复2为checkpoint-body实际接受，不冒称fine_tune中断编排；后者另有
CPU实际subprocess/同端点resume定向反证。没有恢复failed/completed attempt或延长旧端点。

实际调用既有training_one_step/training_two_step，保留initial/all5draft normalized权重与final_weight2；
L6+.5L12，internal K与两物理步均full graph。L12-only backward中第一forecast、encoder、全部4read/
internal-K激活及encoder/reader参数norm均finite>0。每物理步+6h，internal K同valid time；
实际calendar/conditioning feature与独立Gregorian oracle相符（冻结phase atol2e-6不是resume容忍）。
future_target poison改变loss但forecast严格不变，L6控制与两步第一forecast严格一致。

两fresh CUDA worker各自成功退出，run/finalized success、budget_limitedfalse；每spawn只读UUID/余量，
设备 `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b` 共驻，不signal邻居、不清私有缓存、不切CPU fallback。
worker冻结deterministic algorithms、TF32false与CUBLAS workspace=:4096:8，仅适用于此小探针。

## 3. 全额成本与可追溯性

- 实际attempt whole242.43521373253316秒；closeout242.43781951908022秒，后者包含attempt发布，
  closeout自己的序列化发生在快照之后（如实保留亚秒范围）。连续首spawn至末owned reap
  169.61740489676595秒 = **0.04711594580465721 GPU-h**，包含启动/两worker间隔/清理。
- soft_overrun_seconds0；未达软/硬上限。日志/协议/检查点及失败规则原样保留，0failed worker。
- attempt文件SHA256 `8ecd15ea505f43235544d576b214ddc648e2a76c143ed6928334bd9fad094ada`；
  closeout文件SHA256 `4db58b6a6b8070f9468154e7432e2b3cea67277872dacd8fc3fa346895a4c29e`。
- 独立只读CPU核验完成：两precision各两final checkpoint weights_only/CPU递归比131 model tensors、
  123 optimizer states及RNG严格相等；loss/L6/L12/grad_norm来自分支receipt严格相等，checkpoint
  不含loss不冒称从checkpoint复算。29产物before/after SHA/bytes不变；85member/source/model/HEAD/
  opaque父与source、首二train元数据/calendar oracle、0/0/headroom/owned退出/continuous账全部核。
  证据原目录 `/tmp/r7_v2_precision_independent_20261003_z8cWG6/`，原字节映射至
  `outputs/r7_v2_remaining_acceptance_20261003/precision_independent_review/`；最终acceptance文件SHA256
  `bd167cb754eb179cc457f08f60841b336f46edbaca684fe8bda9319c8099aa86`，29pins digest
  `062d27579b98cf50231a429c21ec66ab7f68fa7c8cdb567f639b11eb9c9f09e8`。
- 独立首轮checker自加current git status==frozen status约束，因新三份证据/goal文档失败（不是probe
  实际身份失败），原failed-audit stdout/receipt保留。identity-only新核按actual契约验证原status文件
  hash与所有runtime/source/archive未改，明确仅doc delta；不回写失败或降低原experiment身份比较。
- L12梯度norm/poison属于实际worker执行并记录的检查，原poison两forecast张量及grad数组未另存，
  独立复核核对应归档执行代码与receipt，不声称再计算了GPUforward或梯度。timing记录已退出清理
  string，不含原始exit code/末reap时间；归档driver只accept exit0并于owned reap后取last快照。

## 4. 边界、负面与未做

工程首轮完整CPU10failed/2614passed/9skipped、并发源变动身份拒绝原日志保留；修活跃缺陷而非放宽
断言，稳定完整3107passed/9skipped/2warnings，767.02s，最终160定向passed。六CUDA与三optional
fixture跳过不算通过，此GPU探针也不替代这些全部旧测试。两个Lightning warning保留。

两窗口/两updates不是天气比较，不产出validation技巧/任何科学收益；同设备软件栈checkpoint-body
bitwise resume可核，但B/C普通GPU训练仍最多代码/数据/配置绑定数值可复现，不声称逐位训练。
父默认桥接另外四合成case位等价，不扩为calendar-enabled旧实验或跨设备保证。

未做B6训练30val、C9训练135val/真实adaptive判定、六issue关闭、main推进、科学收益或最终goal完成。
无新依赖/接口源修改/天气下载/数据发布/凭据或用户安全配置变化；此probe实际GPU成本须记账。
下一同节点实验为已预声明B同父L6/两步/equal-compute，独立输出与5400soft/10800hard协议。
