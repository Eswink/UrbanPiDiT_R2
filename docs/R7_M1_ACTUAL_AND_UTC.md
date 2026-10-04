# R7-V2-M1：单输入因素补全（actual M1）与 M1-only UTC 统计

- 登记日期：2026-10-04；`scientific_claim: false`、`actual_pass: false`、`test_read: false`。
- **已确认**：actual M1 实跑完成 24/24 job（4 fresh scratch 训练 + 20 K4 评估，合法复用 C 的 old_ours 41/42 十个
  K4 评分且不导入任何权重/优化器）；CLI verify（`accept_own_output`）通过，返回 protocol_sha256
  `c5df8cc63c8860e21f8feb6d2ec658d6218e82e2bb87923b8cc5b93974ca2d58`。
- **已确认**：M1-only UTC 标量统计链（prepare→run→verify）完成 400 groups / 20,400 cells，
  独立 oracle 复算 receipt `635cd90c9e74958b567e6aefbe5687fd1eadac872c168416cd3b9b8c2eee9d41`。
- **未接受**：不是科学增益或 SOTA；M1 单因素读数仍为 unresolved（见 §3），#71 的「实际改善」条件未证明。

## 1. 工程资格（前置，冻结）

M1 运行前需要 revision06 的工程资格：unique 修复（revision05→06 仅改 `independent_test.py` 的 test_05
stale fixture 键，guard/断言未动，`revision06_changes.json` SHA256 `cdc4a764b282bed2c032dd2c231e7ada51efb8a91b57d091534372e69ac8e44f`）
后第 5 轮实跑**56/56 ok、0 fail、0 error、0 skip**（`FINAL_SEALED.json` SHA256
`a70254d8af438cabb4516c8b66e1a4231109fd667714b40f51d8ef1c543848d4`，status `qualified`；
whole 1236.59 s，软超 0）。
revision06 归档 `revision06_code.zip` SHA256 `908592566cce70445cfc8e50ed539b12ab259fa473d7da1814ca287aa459e4b0`（111 成员）；
实际运行使用其 7 个 RUNTIME_FILES 的逐字节一致副本（staged 于 `/tmp/r7_m1_actual_runtime_20261004/revision06/`）。
round-4 曾因运算符先放收据冲突 sealed as failed（原因如实记录，非测试失败），round 5 从新根完成。

## 2. actual M1 身份与终态

| 项 | 值 |
| --- | --- |
| 输出根 | `outputs/r7_v2_comparison_20261004_attempt02` |
| protocol 文件 SHA256 | `b6093a349301317690f35455a2e77fa95c9b899ff47b1801b74ef710d407aba8` |
| 内部 `protocol_sha256` | `c5df8cc63c8860e21f8feb6d2ec658d6218e82e2bb87923b8cc5b93974ca2d58` |
| base commit | `562e526afc5fdbdc99053a25ab20036b66a2a8bc` |
| code.zip SHA256 | `e00a3196c298cfe78f50dc07136cf4fd03f28abd545b00ec7d066434348cf909` |
| complete_source_sha256 | `5865f435ed38e7ba731ff69be4062c559b8a8031fcf120ebedbeb1fa325bd0c2` |
| C 复用引用 | C binder pin `89f36ff02d46fa095b496d72c43a6a077e01262ca1ff32a58b9b52695acca990` + C acceptance `11af848a7d1ecca768c32e9a4081865cf3b7741c9e375cef68fa1c60972f2a99` |
| 预算 | planned 5400 s / hard 10800 s |
| 终态 | `success`，`finalized: true`，partial/budget_limited/owned_unreaped 全 false，failure_reason null |
| jobs | 24/24（4 train + 20 eval） |
| whole clock | 3543.913658151403 s（soft_overrun 0.0） |
| GPU 连续计费 | `0.8757688270136714` GPU-h（首 spawn→末 owned reap） |
| manifest | `results-sealed`，277 文件 pin |

复用契约（已核）：C root 的 old_ours seed41/42 十个 K4 评估（510 records）以原件 receipt/provenance/CSV
identity pin 引用，C 侧独立接受为前置；M1 自身 4 个训练均为 **fresh scratch**（`resume: false`、
`parent_optimizer_imported: false`、`updates_run == selected_update == 400`），未从 selected400 权重续训。

## 3. M1 端点读数（描述）

Primary 细胞（t2m、K4、full、6/12h，同 seed/lead）：

| seed | lead | old_ours （复用 C） | m1_process | m1_generic | process−old_ours | generic−old_ours | process−generic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 6h | 3.9089 | 2.3684 | 2.3684 | −1.5405 | −1.5405 | −4.02e−6 |
| 41 | 12h | 4.8160 | 2.9692 | 2.9692 | −1.8468 | −1.8468 | −4.27e−6 |
| 42 | 6h | 3.7350 | 2.5264 | 2.5264 | −1.2086 | −1.2086 | −3.03e−6 |
| 42 | 12h | 4.3664 | 3.2131 | 3.2131 | −1.1532 | −1.1532 | −4.17e−6 |

**已确认**：M1-only 版（已知上下文输入 + 保留旧 pooled reader/solver）在 primary 4/4 cell 上相对 old_ours 严格改善，
且改善主要由 known-context 输入贡献（process 与 generic 差在 4e−6 K，`improved` 三态但在数值噪声量级）。
**已确认**：M1 process−generic 全表 67 improved / 141 unresolved / 47 worsened——**单输入因素读数为 unresolved**：
现有证据不能证明「过程语义读写」相对同结构 Generic 的独立改善；#71 的「实际改善」验收条因此**不能判 DONE-positive**。

全表三态（255 cell/对）：

| 对 | improved | unresolved | worsened |
| --- | ---: | ---: | ---: |
| m1_process − old_ours | 100 | 67 | 88 |
| m1_generic − old_ours | 100 | 67 | 88 |
| m1_process − m1_generic | 67 | 141 | 47 |

## 4. M1-only UTC 统计链

| 层 | 产物 | SHA256 |
| --- | --- | --- |
| claims 报告（M1 root 元数据复核） | `/tmp/r7_m1_claims_review_20261004_attempt01/claims_report.json` | `15c0e436b463ec978f11360798eab22b55e0eab800adf1676a13631bae6281aa` |
| 外部 binder | `/tmp/r7_m1_external_binder_20261004_attempt01.json` | `d583aa8d654f975688fb966c8b06b9fb9239df483e31f6ed3cb6d6fba7bd5632` |
| 端 protocol | `/tmp/r7_m1_utc_actual_20261004_attempt01/protocol.json` | `cd66048d1a2e354b80085230aae77fec673cfe4aa948eb6d6ef5a8dc59e89e61` |
| 端 attempt | 同目录 `attempt.json` | `10c205c160070c80f8cd8b09d4108224c1ebc0831951f270fd544f769f29da83` |
| 终端 seal | `/tmp/r7_m1_utc_terminal_20261004_attempt01.json` | `0f6a37e7b7be05a36ae091eabcae7fde049ab42b98d4e57ef9b0e9b8b99f40b5` |
| oracle verify | `/tmp/r7_m1_utc_verify_20261004_attempt01.json` | `635cd90c9e74958b567e6aefbe5687fd1eadac872c168416cd3b9b8c2eee9d41` |

端到端：C 外部 binder（`89f36ff0…`）+ M1 外部 binder（`d583aa8d…`）共同 bind（C 10 records + M1 20 records）；
期望 400 groups / 20,400 cells：model entities 30、baseline entities 20，全部命中；whole 236.96603804454207 s，
软预算 900 s 内，无 overrun，无 GPU。独立 oracle 以自身公式重算全部 400 group identity 与 20,400 cell
（CSV 标量、精确 group/行 digest、单位、baseline 伪臂拒绝），`verified: true`。

C 侧对应 UTC 链（已完成并登记于本页交叉引用）：`/tmp/r7_v2_c_utc_actual_20261004_attempt01/`（protocol
文件 `121105c3…`）→ official CLI `actual-statistics-complete`，67,320 rows / 1,320 groups，独立审阅
`review-complete-C-metadata-only`（receipt `59a6d0a8e174e78002d25b3ad921beb508441f401abbc602cd3b3172a7616a11`），
`all_group_and_row_hashes_exact: true`、`original_c_final_inventory_before_after_identical: true`、
历史 old whole-owner log 失败保留不回修（`FAIL-preserved-no-retroactive-repair`）。

## 5. 限制（如实）

- 两个种子、单冬季段；M1 复用 C 的十个评估（合法复用条件为 C 已独立接受，已满足），未新增第三 seed。
- M1 相对旧 ours 的改善主要来自 known-context 输入容量；过程语义独立贡献 unresolved。
- UTC 统计为标量 metadata 重算（独立公式），不是天气场/tensor oracle，也不改任何 GPU 数值身份。
- round-4 失败因 (operator collision) 如实保留于轮次记录；不隐藏、不回写。
- 可复现等级：config/numerical identity bound，非 GPU 逐位。

## 6. 下一动作（仅提议）

- #71 若要追认过程语义贡献，需要能分辨「同输入信息量」的新协议；当前不追加。
- UTC 统计若需更细的 strata 判定，另立协议；本页不自动触发。
