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
