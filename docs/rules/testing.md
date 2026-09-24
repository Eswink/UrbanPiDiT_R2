# 测试与验收

范围：`tests/**`、`pytest.ini`。

现状基线（实测，2026-09-24 第二遍）：67 个测试文件、**280 个测试函数、560 个断言**、
78 处 `parametrize`、2 处 `skipif`、0 处 `xfail`、0 处被注释掉的断言、0 处 TODO/FIXME。
这两项计数被 R-009 作为报告基线；**有意增删测试时应同步更新 `tools/check_conventions.py`
的 `TEST_FUNCTION_BASELINE` / `ASSERT_BASELINE`**，并在 CHANGELOG 说明原因。

全量可运行测试：改动前 451 passed → 改动后 492 passed，两侧同为 29 failed / 3 skipped / 18 errors，
失败**全部**为缺失可选依赖（`xarray`/`h5netcdf`/`pytorch_lightning`），非代码缺陷（E-157、E-158）。

## R-024 测试只写 tmp_path

- **级别**：必须
- **范围**：`tests/**`
- **陈述**：测试不得向仓库树内的路径写入；所有临时产物写到 `tmp_path`（或等价临时目录）。
- **依据**：E-087（AST 扫描：活跃测试中 0 处写入 `data/`、`outputs/`、`logs/`、`results/` 前缀路径；
  27 个文件使用 `tmp_path`，仅 2 处 `@pytest.fixture`）、E-088（`tests/conftest.py` 只做 `sys.path` 注入，
  不创建或清理仓库内的文件）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-024`，AST 判定写入调用的字符串操作数）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当需要跑需要固定路径的集成测试时（应改为参数化 tmp_path 并显式文档化）

## R-025 测试禁止导入真实网络客户端

- **级别**：禁止
- **范围**：`tests/**`
- **陈述**：测试不得导入 `requests`/`urllib.request`/`fsspec`/`s3fs`/`gcsfs`/`http.client`。
  允许 `socket`，但仅用于**禁用**出网（见 `tests/test_r7_pinned_real_fixture.py:6`）。
- **依据**：E-089（精确 AST 扫描：测试中唯一出网相关导入是 `socket`，用于网络封禁）、
  E-090（`tests/test_r7_pinned_real_fixture.py:16-19` 安装 socket 拒绝后断言真实 fixture 哈希，
  用于抓住"仿造真实数据"的行为）
- **现状**：A 类 —— 0 处违反
- **执行方式**：脚本（`--rule R-025`，`socket` 与 `urllib.parse` 已从禁止集豁免）
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：当需要测试真实下载器时（应走 CI 的显式 fetching workflow，不是单元测试）

## R-026 每个判据必须有反证

- **级别**：必须
- **范围**：`tools/**`、`tests/**`
- **陈述**：任何"检查/门禁/守卫"必须有一个证明它**能失败**的用例。
- **依据**：E-091（`tests/test_r7_chunk_budget.py:42-58`：删除 chunk 预算的 raise 会让
  "拒绝全球压力层"用例失败）、E-092（`tests/test_r7_bounded_download.py:27-32`：删除读前字节上限检查会让
  `test_wire_cap_checked_before_any_body_read` 失败，因为断言 `read_calls == 0`）、
  E-093（`tests/test_r7_storage_safety.py:45-97` 覆盖不完整 store 与篡改拒绝）
- **现状**：A 类 —— 现有守卫均有对应反证；本次新增的 `tools/check_conventions.py` 也配套 27 个自测
  （其中 19 个是"故意违规必须报错"的反证）
- **执行方式**：人工自觉 + `pytest tests/test_check_conventions.py -q`
- **例外**：无
- **引入日期**：2026-09-24
- **复核触发**：新增任何守卫时

### 观察（不立规）

- `pytest.ini` 没有 `markers` 段，也没有 `--strict-markers`。当前只用 `parametrize`（76 处）
  与 `skipif`（2 处），未注册自定义 marker，所以现在没有实际风险。
  一旦引入需要 GPU/网络的 marker，应先注册并开启 `--strict-markers`。见 OPEN_QUESTIONS Q-003。
- 2 处 `skipif` 依赖 `data/raw/real_smoke/` 下的可选 fixture，理由写得很明确
  （"clean checkout 无 fixture，禁止合成回退"）。这符合 R-008 的精神，**不应**视为债务。
- 测试直接依赖 `torch`（27 个文件）、`pandas`（16）、`xarray`（7）等，这些是 `requirements.txt`
  与 `requirements-r7-data.txt` 中的依赖，CI 会安装，因此离线可跑。
