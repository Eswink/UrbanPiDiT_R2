# 证据台账

引导阶段（2026-09-24）的只读侦察记录。每条只写**观察到的事实**，判断单独标注。
「依据」栏中的 E-xxx 编号被 `docs/rules/*.md` 引用。

覆盖度：全体 = 全仓扫描；抽样 N 处；单点 = 仅一处观察。
置信度：已确认 = 可指到具体位置；推测 = 间接迹象；未知。

## 阶段 0：环境与授权

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-001 | 仓库根为 `/data/esw/UrbanPiDiT_R2`，是 git 仓库，分支 `r7/weather-reasoning`，有远端 `origin` | `git status -b`、目录标志物 | 单点 | 已确认 |
| E-002 | 平台为 Linux x86_64，shell 为 bash；解释器 `python3` 3.12.3（miniconda） | `uname -a`、`which python3` | 单点 | 已确认 |
| E-003 | 工作树在引导开始时干净，唯一未跟踪项是 `.mimosa/`（代理工具运行目录） | `git status --porcelain` | 全体 | 已确认 |
| E-004 | 项目**没有**根 `AGENTS.md`、`CLAUDE.md`、`.cursor/`、`.claude/`、`CONTRIBUTING.md`、`Makefile`、`tox.ini`、`noxfile.py` | 顶层 `ls` 与定向查找 | 全体 | 已确认 |
| E-005 | 项目**已有**一份明确的协作纪律文档 `docs/R7_MANUAL_ITERATION.md`，含"不得 force push / 不得后台继续 / 取消的运行不算通过"等要求 | `docs/R7_MANUAL_ITERATION.md:1-40` | 单点 | 已确认 |
| E-006 | 规模基线：活跃区 205 个 `.py`（20,494 行），归档区 202 个 `.py`（46,890 行）；测试 67 个文件；notebook 3 个（全在归档区） | 脚本统计 + `os.walk` | 全体 | 已确认 |
| E-007 | 归档体积占主体：`legacy_v531_full/` 9.5MB / 全仓 16MB；归档 Python 行数占全仓 70% | `du -sh`、行数统计 | 全体 | 已确认 |
| E-008 | 项目处于科研进行中状态：有 7 个进行中/阻塞的研究关卡，Draft PR #12 未合并 | `docs/R7_TASK_QUEUE.md:45-56` | 单点 | 已确认 |
| E-009 | 活跃区只有 2 处 `os.environ` 使用（均为子进程传线程数），无 `.env` 文件 | AST/正则扫描 | 全体 | 已确认 |

## 阶段 1.1–1.2：布局与入口

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-010 | 不是 src-layout：`data/`、`model/`、`training/` 三个平级顶层包 | 顶层目录 | 全体 | 已确认 |
| E-011 | `pyproject.toml` 声明 4 个 console script（`urbanpidit-r7-train/evaluate/prepare/calibrate`）与 4 个 `py-modules` | `pyproject.toml:17-23` | 全体 | 已确认 |
| E-012 | 根目录另有 7 个未在 `pyproject.toml` 声明的 `.py` 入口：`train.py`、`train_r7.py`、`train_r7_process.py`、`train_r7_recursive.py`、`diagnose_r7_gain.py`、`profile_r7_inference.py`、`tune_r7_halting.py` | 顶层 `ls` 对比 `pyproject.toml` | 全体 | 已确认 |
| E-013 | 11 个根入口全部有 `main()` 与 `if __name__ == "__main__"` 保护 | AST 扫描 | 全体 | 已确认 |
| E-014 | `train_r7_process.py`、`train_r7_recursive.py`、`train_r7.py` 无任何文档或测试引用（孤儿入口） | `git grep` 于 docs/tests/scripts/.github | 全体 | 已确认 |
| E-015 | `scripts/` 下有 25 个脚本，其中 5 个是真实数据 pilot（`real_r7_*.py`），3 个是 probe | `ls scripts/` | 全体 | 已确认 |
| E-016 | 无 Makefile / justfile / tox / nox；任务入口是"根脚本 + CI workflow" | 顶层查找 | 全体 | 已确认 |

## 阶段 1.3：依赖

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-017 | 依赖全部用 `>=` 范围而非精确 pin：`torch>=2.4`、`numpy>=1.26`、`pytorch-lightning>=2.2` | `pyproject.toml:10`、`requirements.txt` | 全体 | 已确认 |
| E-018 | 存在 3 份依赖清单：`requirements.txt`（5 项）、`requirements-data.txt`（7 项，含 rasterio/Pillow）、`requirements-r7-data.txt`（7 项，含 h5netcdf/h5py/s3fs） | 三个文件 | 全体 | 已确认 |
| E-019 | `requirements.txt` 与 `pyproject.toml` 的 `[project] dependencies` 不一致：前者多出 `pytest>=8.0` | `requirements.txt:5` vs `pyproject.toml:10` | 全体 | 已确认 |
| E-020 | `pyproject.toml` 的 optional `data` 组与 `requirements-r7-data.txt` 不一致：前者无 h5netcdf/h5py/s3fs，后者无 rasterio/Pillow | 对比两份清单 | 全体 | 已确认 |
| E-021 | 无 lockfile（无 `uv.lock`/`poetry.lock`/`Pipfile.lock`）、无 `setup.cfg`、无 `environment.yml`、无 Dockerfile | 顶层查找 | 全体 | 已确认 |
| E-022 | CI 中 `torch` 绝大多数用 `"torch>=2.4"`，但有 3 个 workflow 用 `'torch==2.14.0+cpu'` 精确 pin | `.github/workflows/r7-{continuous-control,extended-control,spatial-solver}.yml:24` | 抽样 18 处 | 已确认 |
| E-023 | `requires-python = ">=3.10"`，CI 固定使用 3.11（主 CI）或 3.12（实验 workflow） | `pyproject.toml:8`、`ci.yml:22` | 全体 | 已确认 |
| E-024 | 本机已装 torch/numpy/pandas/yaml/pytest/h5py，**未装** xarray/zarr/h5netcdf/gcsfs/s3fs/rasterio/pytorch_lightning | 本机 import 探测 | 单点 | 已确认 |

## 阶段 1.4：测试与验收

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-025 | `pytest.ini` 只有 `testpaths=tests`、`python_files=test_*.py`、`addopts=-ra`；无 markers 段、无 `--strict-markers`、无 `--strict-config` | `pytest.ini`（3 行） | 全体 | 已确认 |
| E-026 | 67 个测试文件、262 个测试函数、539 个断言、76 处 `parametrize`、14 个测试类 | AST 扫描 | 全体 | 已确认 |
| E-027 | 仅 2 处 `skipif`、0 处 `xfail`、0 处被注释掉的断言、0 处 TODO/FIXME | AST/正则扫描 | 全体 | 已确认 |
| E-028 | 2 处 `skipif` 依赖 `data/raw/real_smoke/` 下的可选 fixture，理由是"clean checkout 无 fixture，禁止合成回退" | `tests/test_real_data_pipeline.py:18-31` | 全体 | 已确认 |
| E-029 | `tests/conftest.py` 只做 `sys.path` 注入，无共享 fixture；仅 2 个文件声明 `@pytest.fixture` | `tests/conftest.py`（5 行）、AST 扫描 | 全体 | 已确认 |
| E-030 | 27 个测试文件使用 `tmp_path`；17 处使用 `subprocess` 测试真实 CLI 入口 | AST 扫描 | 全体 | 已确认 |
| E-031 | 测试直接导入真实模块（`training.r7_experiment` 12 次、`data.download.earthmover_pilot` 10 次等），不是测试替身 | import 统计 | 全体 | 已确认 |
| E-032 | 上次完整 CI 记录：578 passed, 3 old-fixture-skipped, 2 Lightning warnings，用时 49.25s | `docs/R7_TASK_QUEUE.md:18` | 单点 | 已确认（文档声明，未实跑复现） |
| E-033 | 有一个测试专门验证 wheel 内容与从干净目录导入（`test_r7_installed_package.py`），并断言 wheel 内不含 legacy/tests | `tests/test_r7_installed_package.py:10-31` | 单点 | 已确认 |

## 阶段 1.5：静态门禁

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-080 | 全仓**无** ruff/flake8/mypy/black/pylint 配置：`pyproject.toml` 无 `[tool.*]` 段，无 `.ruff.toml`/`setup.cfg`/`mypy.ini`/`.flake8`/`.pre-commit-config.yaml` | 定向查找 + `pyproject.toml` 全文 | 全体 | 已确认 |
| E-098 | 唯一语法门禁是 `ci.yml:37` 的 `compileall`，且**不覆盖 `scripts/` 与 `tests/`** | `ci.yml:35-38` | 全体 | 已确认 |
| E-099 | `ci.yml:38` 的 `git diff --check` 在干净 checkout 上 diff 为空，实质不产生门禁作用 | `ci.yml:35-38` + checkout 语义 | 单点 | 已确认 |
| E-100 | 无 git hook（`.git/hooks/` 只有 sample 文件）、无代理生命周期钩子配置 | `ls .git/hooks/` | 全体 | 已确认 |

## 阶段 1.6：数据、配置与产物

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-034 | `.gitignore` 忽略 `__pycache__`、`*.pyc`、`.pytest_cache`、`outputs/`、`logs/`、`*.ckpt`，以及 `data/{raw,interim,processed}/*`（保留 `.gitkeep`） | `.gitignore`（13 行） | 全体 | 已确认 |
| E-035 | `data/manifests/real_smoke/` 下的 3 个 `.jsonl` 与 1 个 `provenance.json` **被跟踪**（约 37KB），因为它们不被 ignore | `git ls-files data/manifests/` | 全体 | 已确认 |
| E-036 | `data/raw`、`data/interim`、`data/processed` 在版本控制中只有 `.gitkeep` | `git ls-files data/` | 全体 | 已确认 |
| E-037 | `outputs/` 当前不存在（未运行过本地实验） | 文件系统检查 | 全体 | 已确认 |
| E-038 | 活跃区无被跟踪的二进制产物（无 `.ckpt`/`.npy`/`.npz`/`.png`）；归档区有 6 个大文件（最大 2.1MB 的 gif） | `git ls-files` + 体积统计 | 全体 | 已确认 |
| E-039 | 活跃区有 22 个被跟踪的 `.log` 文件（共 31KB，在 `audit/real_data_logs/`） | `git ls-files`、体积统计 | 全体 | 已确认 |
| E-040 | 无 DVC / git-lfs：无 `.dvc`、`.lfsconfig`、`.gitattributes` | 顶层查找 | 全体 | 已确认 |
| E-101 | `configs/` 下 10 个 YAML 分为两代：V6 的 5 个用 `data/model/loss/optim/train` 结构，R7 的 5 个用各自的结构（`kind/synthetic/train`、`split_years/...`） | YAML 顶层键扫描 | 全体 | 已确认 |
| E-102 | 无 hydra/omegaconf/pydantic-settings；配置由 `yaml.safe_load` 直接读入 dict | 依赖清单 + 代码扫描 | 全体 | 已确认 |

## 阶段 1.7：可复现性设施

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-068 | `seed_everything` 覆盖 random/numpy/torch/torch.cuda；RNG 状态被保存进 checkpoint 并可恢复，CUDA 拓扑不匹配时抛错 | `training/r7_experiment.py:56-78` | 全体 | 已确认 |
| E-069 | 训练 contract 记录 seed、batch/accum、lr、device、torch 版本、数据集长度；resume 等价性有测试断言 | `training/r7_local_runner.py:73-78,111,141`、`tests/test_r7_local_runner.py:36-56` | 全体 | 已确认 |
| E-103 | 23 个文件含种子/确定性相关代码；训练主路径全部显式传 seed | 正则扫描 | 全体 | 已确认 |
| E-104 | **未**设置 `torch.use_deterministic_algorithms`、`cudnn.deterministic` 或 TF32 开关；`r7_inference_profile.py:136` 只报告这些状态 | 全仓 grep + `r7_inference_profile.py:136` | 全体 | 已确认 |
| E-105 | 运行记录字段丰富：`protocol.json`（8 个模块）、`provenance.json`（评估与数据集）、`receipt.json`（下载）、`code_commit.txt`/`code.zip`（CI）、`environment.txt`（仅 4 个 workflow） | 各写入点 | 全体 | 已确认 |
| E-106 | 数据集溯源记录 `source_sha256`、`split_policy`、`normalization`、`limitations` | `data/manifests/real_smoke/provenance.json` | 单点 | 已确认 |
| E-107 | 无实验跟踪服务（无 wandb/mlflow/tensorboard 依赖或调用） | 依赖清单 + 全仓扫描 | 全体 | 已确认 |

## 阶段 1.8：命名与风格

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-108 | 活跃区 205 个模块全部 `snake_case`（`__init__.py` 除外）；无拼音、无空格、无非 ASCII | 正则分类 | 全体 | 已确认 |
| E-109 | 测试命名统一 `test_<subject>.py`；`tests/test_check_conventions.py` 是本次新增 | 文件名扫描 | 全体 | 已确认 |
| E-110 | 活跃区无杂物桶目录（无 `utils/`、`misc/`、`common/`、`helpers/`、`manager/`） | 目录名扫描 | 全体 | 已确认 |
| E-111 | 文档命名有强模式：44 个 `.md` 中 39 个以 `R7_` 前缀 + 大写下划线 | `ls docs/` | 全体 | 已确认 |
| E-112 | 活跃区文件名中无 `final`/`old`/`tmp`/`copy`/`_vN` 类标记；命中的是 `configs/r2_v6_v100.yaml`（硬件型号，非版本后缀）与 `audit/*final*`（归档轮次命名） | 正则扫描 | 全体 | 已确认 |

## 阶段 1.9：Git 历史

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-094 | 134 个提交中 133 个符合 Conventional Commits；类型分布 feat 28/test 8/fix 6/data 6/ci 5/docs 4/experiment 2/build 1；scope 全为 `r7`；140 个提交中带 `(#N)` 引用者 133 个，带方括号标签者 34 个 | `git log --format=%s` | 全体 | 已确认 |
| E-113 | 无 merge commit（线性历史）；5 个分支（含 2 个远端）；0 个 tag | `git log --merges`、`git branch -a`、`git tag` | 全体 | 已确认 |
| E-114 | 提交粒度小：最近 60 个提交平均改动 3.5 个文件，多数为 1–6 个文件 | 统计 | 抽样 60 | 已确认 |
| E-115 | 单一贡献者身份（`Eswlnk|1525566427@qq.com` 133 次，`Eswink` 1 次——同一邮箱的大小写差异） | `git log --format=%an|%ae` | 全体 | 已确认 |
| E-116 | 全部 134 个提交发生在 2026-09-23 至 2026-09-24 两天内 | 首个与最后提交日期 | 全体 | 已确认 |
| E-117 | 无 revert/rollback 提交；失败处理方式是"向前修复 + 保留产物" | `git log --grep=revert`（空） | 全体 | 已确认 |
| E-118 | 高频共同改动：`evaluate_r7_local.py` ↔ `training/r7_evaluate.py`（4 次）、`pyproject.toml` ↔ `tests/test_r7_installed_package.py`（3 次） | `git log --name-only` 共现统计 | 抽样 60 | 已确认 |
| E-119 | 无超过 1 年未改动的 Python 文件（仓库存在仅 2 天） | `git log -1 --format=%at` | 全体 | 已确认 |

## 阶段 1.10–1.11：环境耦合与安全

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-167 | 见 E-009：仅 2 处环境变量使用，无 `.env`（原编号 E-009 被复用，第三遍改为唯一编号） | 扫描 | 全体 | 已确认 |
| E-120 | 疑似凭据扫描：password/api_key/aws_key/private_key/bearer 模式**全部 0 命中** | 5 类正则全仓扫描 | 全体 | 已确认 |
| E-121 | 无数据外传代码：无 `.post(`/`.put(`/upload 调用；唯一的长网络导入集中在 `data/download/` | AST 扫描 | 全体 | 已确认 |
| E-122 | 无自动下载大模型权重的代码；下载目标均为小型 ERA5 子集与 WorldCover ROI，且有字节预算 | `data/download/**` 扫描 + 预算常量 | 全体 | 已确认 |
| E-123 | 不处理人类被试/敏感个人数据；使用的是 ERA5 再分析、WorldCover、UCI 北京气象等公开数据 | 数据文档与下载器 | 全体 | 已确认 |
| E-124 | 出网客户端仅出现在 4 个文件的导入语句中（`arco_tiny_bounded`、`ncar_public_probe`、`uci_beijing`、`worldcover_smoke`），全部在 `data/download/` | 精确 AST 扫描 | 全体 | 已确认 |
| E-125 | 测试中唯一出网相关导入是 `socket`，用于**禁用**网络 | `tests/test_r7_pinned_real_fixture.py:6,16-19` | 全体 | 已确认 |

## 阶段 1.12：能力面与执行面

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-168 | 无任何能力目录（`.cursor/skills/`、`.claude/skills/`、`prompts/`、`playbooks/`、`runbooks/` 均不存在）；44 个 `docs/*.md` 是平铺的实验记录（原编号 E-004 被复用，第三遍改为唯一编号） | 定向查找 | 全体 | 已确认 |
| E-065 | 11 个 workflow 同时归档 `code_commit.txt` 与 `code.zip`；7 个不归档 | workflow 扫描 | 全体 | 已确认 |
| E-077 | 10 个 workflow 内联安装 socket 拒绝逻辑 | workflow 扫描 | 全体 | 已确认 |
| E-096 | 18 个 workflow 全部设置 `timeout-minutes`（10–30 分钟） | workflow 扫描 | 全体 | 已确认 |
| E-126 | 仅 3 个 workflow 设置 `concurrency`（`ci.yml`、`r7-public-data.yml`、`r7-surface-pilot.yml`），其余 15 个没有 | workflow 扫描 | 全体 | 已确认 |
| E-127 | 两个文件的 tags 与文件名不一致：`r7-earthmover-probe.yml` 的触发标签是 `[era5-temporal-probe]`；`r7-real-smoke.yml` 是 `[real-smoke]` 而文件名不同 | workflow 扫描 | 全体 | 已确认 |
| E-128 | 无 pre-commit、无 git hook、无 CI 中的 lint/type 步骤 | 定向查找 + workflow 扫描 | 全体 | 已确认 |

## 阶段 3–4：约定与冲突（实证扫描）

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-041 | 41 处排他模式（`'x'`/`'xb'`）写入，分布在 30 个文件 | 正则扫描 | 全体 | 已确认 |
| E-042 | 发布契约用 `open('x')` 创建 `.r7-build.lock` 与 `BUILD_COMPLETE.json` | `data/preprocess/contracts.py:37,43` | 单点 | 已确认 |
| E-043 | NetCDF 与大文件用 `'xb'` 二进制排他写 | `data/preprocess/r7_era5.py:165`、`data/download/pressure_pilot_replay.py:96` | 抽样 2 处 | 已确认 |
| E-044 | 归档 README 明确声明不属于 V6 import 主路径 | `model/legacy_v531/README.md`、`data/legacy_v531/README.md` | 全体 | 已确认 |
| E-045 | 全仓 0 处活跃代码 import `legacy_v531` | 正则 + AST 扫描 | 全体 | 已确认 |
| E-046 | 打包时 exclude `*.legacy*` 与 `legacy_v531_full*` | `pyproject.toml:29` | 单点 | 已确认 |
| E-047 | 文档声明旧的无版本缓存必须在新目标重建，不得靠加标记改装 | `docs/R7_DATA_PUBLICATION.md:7` | 单点 | 已确认 |
| E-048 | 数据集溯源记录 source_sha256/split_policy/normalization/limitations | `data/manifests/real_smoke/provenance.json` | 单点 | 已确认 |
| E-049 | 三个数据目录在版本控制中仅有 `.gitkeep` | `git ls-files data/` | 全体 | 已确认 |
| E-050 | `.gitignore:8-13` 显式忽略三个数据目录的内容 | `.gitignore` | 全体 | 已确认 |
| E-051 | 文档声明"输入文件永不被修改" | `docs/R7_DATA_PUBLICATION.md:1`、`docs/R7_REAL_DATA_PREFLIGHT.md:18` | 单点 | 已确认 |
| E-052 | `fresh_outputs` 拒绝已存在/符号链接/嵌套的输出路径，成功后写 BUILD_COMPLETE | `data/preprocess/contracts.py:10-50` | 单点 | 已确认 |
| E-053 | 读取方拒绝不完整/无版本 store | `data/r7_store.py:12-19,31-35` | 单点 | 已确认 |
| E-054 | `build_complete` 起始 False，末尾才翻转 | `data/preprocess/r7_era5_zarr.py:93,147` | 单点 | 已确认 |
| E-055 | 6 个 study/control 模块的协议写入行均早于首个训练调用（含显式 "BEFORE" 注释） | `r7_cpu_study.py:184`、`r7_continuous_control.py:80`、`r7_extended_control.py:99`、`r7_spatial_study.py:93`、`r7_seasonal_study.py:102`、`r7_baseline_study.py:127` | 全体 | 已确认 |
| E-056 | 协议 digest 被 3 个测试断言 | `tests/test_r7_continuous_control.py:94`、`test_r7_cpu_study.py:20`、`test_r7_spatial_study.py:17` | 全体 | 已确认 |
| E-057 | 22 个结果写入模块中 18 个带 `scientific_claim`，另 4 个用等价字段（`scientific_forecast_claim`/`not_claimed`） | 扫描 + `scripts/real_r7_bounded_smoke.py:33` | 全体 | 已确认 |
| E-058 | 5 个测试直接断言非科学声明标志为 False | `test_r7_inference_profile.py:37`、`test_r7_policy_selection.py:27`、`test_r7_pinned_real_fixture.py:25,45`、`test_r7_preflight.py:46` | 全体 | 已确认 |
| E-059 | 下载失败写出 `status='failed-no-fallback'`、`synthetic_fallback=False` 后重新抛出 | `data/download/arco_tiny_bounded.py:152-159` | 单点 | 已确认 |
| E-060 | 交付文档声明禁止合成静默回退 | `DELIVERY.md:22` | 单点 | 已确认 |
| E-061 | 测试断言无回退记录存在 | `tests/test_r7_bounded_download.py:54-62` | 单点 | 已确认 |
| E-062 | 纪律文档声明"取消或排队的运行不算通过" | `docs/R7_MANUAL_ITERATION.md:17` | 单点 | 已确认 |
| E-063 | 纪律文档声明"修复真实失败，不弱化科学/测试要求" | `docs/R7_MANUAL_ITERATION.md:15` | 单点 | 已确认 |
| E-064 | 262 个测试函数仅 2 处 skipif，0 处 xfail，0 处注释掉的断言 | AST 扫描 | 全体 | 已确认 |
| E-066 | 文档记录"旧产物需用其归档的 code.zip/commit 评估" | `docs/R7_CPU_REFINEMENT_RESULTS.md:163` | 单点 | 已确认 |
| E-067 | 跨配置比较需显式案例审计 | `docs/R7_CASE_ALIGNMENT.md`、`docs/R7_TASK_QUEUE.md:52` | 单点 | 已确认 |
| E-070 | 活跃区无产物类被跟踪文件；22 个 `.log` 是审计记录（31KB） | `git ls-files` + 体积 | 全体 | 已确认 |
| E-071 | 0 处文本 open 缺 encoding（AST 精确判定） | AST 扫描 | 全体 | 已确认 |
| E-072 | 全部活跃源文件可 UTF-8 解码，0 个非 UTF-8 | 逐文件解码 | 全体 | 已确认 |
| E-073 | 0 处硬编码宿主绝对路径（覆盖 .py/.yaml/.yml/.json/.md/.cfg/.ini/.toml） | 正则扫描 | 全体 | 已确认 |
| E-074 | 路径以 `Path(__file__).resolve().parents[1]` 相对定位 | `scripts/real_r7_seasonal_pilot.py:9` 等 | 抽样 5 处 | 已确认 |
| E-075 | 0 处裸 except | AST 扫描 | 全体 | 已确认 |
| E-076 | 出网模块导入仅 4 个文件，全在 `data/download/` | 精确 AST 扫描 | 全体 | 已确认 |
| E-078 | `from __future__ import annotations` 覆盖：data 33/35、model 22/24、training 36/37、scripts 35/36；例外 6 个（5 个空 `__init__.py` + `train.py`） | AST/文本扫描 | 全体 | 已确认 |
| E-079 | `Path` 出现在 93 个活跃文件；`os.path.*` 仅 4 处，全为 `relpath` | AST 扫描 | 全体 | 已确认 |
| E-081 | 行长分布：中位 40、p95 96、p98 110、p99 120、最大 570；>120 共 159 行（73 个文件），>200 共 18 行 | 逐行统计 | 全体 | 已确认 |
| E-082 | 966 个函数：62 个 >50 行、16 个 >80、12 个 >100；最长 445 行（归档区） | AST 统计 | 全体 | 已确认 |
| E-083 | 205 个活跃文件：中位 64 行、14 个 >200 行、5 个 >300、1 个 >400（`legacy_physical_consistency.py` 833 行） | AST 统计 | 全体 | 已确认 |
| E-084 | `training/legacy_physical_consistency.py` 未被任何模块导入 | import 图 | 全体 | 已确认 |
| E-085 | 非测试代码最深嵌套 5（1 个函数）；测试最深 9 | AST 统计 | 全体 | 已确认 |
| E-086 | 966 个函数中 11 个非 `__init__` 函数参数 >8，最多 27 个 | AST 统计 | 全体 | 已确认 |
| E-087 | 0 处测试写入仓库树路径；27 个文件用 `tmp_path` | AST 扫描 | 全体 | 已确认 |
| E-088 | `tests/conftest.py` 只注入 `sys.path`，不创建/清理仓库文件 | `tests/conftest.py` | 全体 | 已确认 |
| E-089 | 测试中出网导入仅 `socket`（用于禁网） | 精确 AST 扫描 | 全体 | 已确认 |
| E-090 | 网络封禁测试断言真实 fixture 哈希，用于抓"仿造真实数据" | `tests/test_r7_pinned_real_fixture.py:16-26` | 单点 | 已确认 |
| E-091 | chunk 预算守卫有反证：删除 raise 会让"拒绝全球压力层"用例失败 | `tests/test_r7_chunk_budget.py:42-58` | 单点 | 已确认 |
| E-092 | 读前字节上限有反证：断言 `read_calls == 0` | `tests/test_r7_bounded_download.py:27-32` | 单点 | 已确认 |
| E-093 | store 安全有不完整/篡改拒绝用例 | `tests/test_r7_storage_safety.py:45-97` | 单点 | 已确认 |
| E-095 | 子进程内同样禁网并检查返回码 | `training/r7_baseline_study.py:85-91,96-98` | 单点 | 已确认 |
| E-097 | 3 个控制模块在循环内检查 1080 秒截止时间并抛错 | `r7_continuous_control.py:92-93`、`r7_extended_control.py:103-104`、`r7_spatial_study.py:103-104` | 全体 | 已确认 |
| E-129 | 3 个 replay 模块各有硬编码 SHA256 pin，源不匹配即抛 ValueError | `pressure_pilot_replay.py:10,35-36`、`seasonal_pilot_replay.py:5`、`continuous_pilot_replay.py:9` | 全体 | 已确认 |
| E-130 | 恢复流程把归档元数据与冻结期望值逐字段比对，不符即拒绝，成功才写 `RESTORATION_ACCEPTED.json` | `data/restore_pilot_cache.py:20-29,55-58,78-95` | 单点 | 已确认 |
| E-131 | `r7_seasonal_study.py` 无内部墙钟截止检查（只测量 elapsed） | `training/r7_seasonal_study.py:103,136` | 单点 | 已确认 |
| E-132 | 活跃区有 19 处 `print`，全部在 `main()` 内或输出 JSON；归档区有 35 处 | AST 位置扫描 | 全体 | 已确认 |
| E-133 | 活跃区 **0 处 logging**（无 `import logging`） | 全仓扫描 | 全体 | 已确认 |
| E-134 | 活跃区 0 处模块级副作用（无模块级 I/O 调用） | AST 扫描 | 全体 | 已确认 |
| E-135 | 41 个活跃模块未被任何其他活跃模块 import；其中多数是 CLI 入口与 smoke 脚本（由测试或 CI 调用而非 import） | import 图 | 全体 | 已确认 |
| E-136 | 有 22 处函数内 import，理由包括可选依赖延迟导入（`zarr`、`rasterio`、`icechunk`）与避免循环导入 | AST 扫描 | 全体 | 已确认 |

## 阶段 4：冲突清单（记录，不裁定）

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-137 | 依赖清单三份并存且内容不一致（见 E-018/E-019/E-020） | 三份清单对比 | 全体 | 已确认 |
| E-138 | torch 版本策略不统一：15 个 workflow 用 `>=2.4`，3 个用 `==2.14.0+cpu` | workflow 扫描 | 全体 | 已确认 |
| E-139 | Python 版本策略不统一：`requires-python>=3.10`、主 CI 3.11、实验 workflow 3.12、本机 3.12.3 | 配置对比 | 全体 | 已确认 |
| E-140 | 配置 schema 两代并存：V6 的 `data/model/loss/optim/train` 与 R7 的各自结构 | E-101 | 全体 | 已确认 |
| E-141 | 中文与英文注释/文档混用：`data/legacy_v531/loader.py`、`worldcover_smoke.py` 等用中文；`training/`、`model/` 用英文 | 抽样阅读 | 抽样 20 处 | 已确认 |
| E-142 | 文档与实现的一处张力：`README.md` 描述 V6 架构（`scripts/smoke_forward.py`、`train.py`），而当前活跃主线是 R7（`train_r7_local.py` 等）；R7 主线的入口在 `docs/R7_*.md` 里 | `README.md` vs `docs/R7_*` | 全体 | 已确认 |
| E-143 | `README.md` 的"快速 smoke test"段落指向 `configs/r2_v6_smoke.yaml` 与 `scripts/smoke_forward.py`，不涉及 R7 主线 | `README.md:39-48` | 单点 | 已确认 |

## 第二遍（2026-09-24）：决定落地与验证证据

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-144 | `tools/check_conventions.py` 曾有两个模块级函数被**重复定义**（`r_002_archival_readonly` L208/L609、`r_018_os_path` L465/L728），Python 取后定义，导致 AST 版 R-002 从未生效 | `ast` 解析统计 `FunctionDef` 名（**grep 检测不到重名**） | 全体 | 已确认 |
| E-145 | 归档移动后 833 行内容与原文逐行一致（仅补结尾换行；原文件末尾无换行符） | 与 `git show HEAD:training/legacy_physical_consistency.py` 逐行比对 | 单点 | 已确认 |
| E-146 | `model_code_digest()` 只覆盖 `model/` 下非 legacy 的 `.py`；`data/`、`training/`、`scripts/`、根脚本不在其内 | `training/r7_experiment.py:31-40`（`directory.rglob('*.py')` 且跳过含 `legacy` 的路径） | 全体 | 已确认 |
| E-147 | `load_checkpoint` 在不匹配 `model_code_sha256` 时抛 `ValueError`，并提示"使用记录在案的代码版本" | `training/r7_experiment.py:120-125` | 单点 | 已确认 |
| E-148 | 3 条 workflow 用当前 HEAD 代码加载归档 checkpoint：`r7-restored-diagnostic.yml`（run 35886241381）、`r7-correction-audit.yml`、`r7-extended-control.yml` | `restore_pilot_cache` → `load_checkpoint` 调用链；`r7_extended_control.py:91`、`r7_restored_diagnostic.yml:29-31` | 抽样 3 处 | 已确认 |
| E-149 | `training/r7_profile_comparison.py` 的 `_model_sources` 读的是**归档 code.zip 内**的 model 源码，不受工作树改动影响 | `r7_profile_comparison.py:111-116` | 单点 | 已确认 |
| E-150 | 18 处 >200 字符的行分布在 13 个文件；重构后逐一用 `ast.dump` 结构哈希比对，**全部 MATCH**（13/13） | 重构前后 `hashlib.sha256(ast.dump(ast.parse(src)))` 相等 | 全体 | 已确认 |
| E-151 | `.gitignore` 新增模式后，**没有任何已跟踪文件**变为被忽略；`data/manifests/real_smoke/*.jsonl` 与 4 个 `.gitkeep` 仍被跟踪 | `git ls-files` 逐条 `git check-ignore` 比对 | 全体 | 已确认 |
| E-152 | 6 个 CI 产物暂存目录名：`input_pilot`、`input_seasonal`、`input_study`、`input_continuous`、`input54`、`input56`（`input_study` 被 3 条 workflow 使用，最多） | workflow `download-artifact` 的 `path:` 汇总 | 全体 | 已确认 |
| E-153 | 3 个入口脚本被**裸文件名 + `cwd=repo`** 调用，移走即破坏：`diagnose_r7_gain.py`、`profile_r7_inference.py`、`tune_r7_halting.py`；后者之一还被 `training/r7_baseline_study.py` 按路径调用 | `tests/test_r7_gain_oracle_local.py:47`、`tests/test_r7_inference_profile_local.py:52`、`tests/test_r7_policy_search_e2e.py:54`、`training/r7_baseline_study.py:83-84` | 抽样 4 处 | 已确认 |
| E-154 | `scripts/` 的导入惯例是 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`，而根目录脚本直接 `from training.X import ...`（依赖 cwd） | `scripts/study_r7_cpu.py:7`、`scripts/smoke_forward.py:4-6` | 抽样 2 处 | 已确认 |
| E-155 | 测试套件实测 280 个测试函数 / 560 个断言（第二遍重构后），原记录 262/539 | AST 统计 `tests/**/test_*.py` | 全体 | 已确认 |
| E-156 | `--strict-markers` 开启后收集不受影响：全仓仅使用内置 `parametrize`（78 处）与 `skipif`（2 处），无自定义 marker | 正则统计 `pytest.mark.*` + 收集运行 | 全体 | 已确认 |
| E-157 | 全量可运行测试：改动前 451 passed，改动后 **492 passed**，两侧同为 29 failed / 3 skipped / 18 errors，失败与收集错误**全部**是缺失可选依赖（`xarray` 14、`h5netcdf` 2、`pytorch_lightning` 1、`numcodecs` 1） | 在 `git archive HEAD` 的纯净副本与工作树上分别运行 `pytest -q --continue-on-collection-errors` | 全体 | 已确认 |
| E-158 | 本机未安装 `xarray`/`zarr`/`h5netcdf`/`gcsfs`/`s3fs`/`rasterio`/`pytorch_lightning`，因此 18 个测试模块无法收集；这不是代码缺陷 | 本机 import 探测 + 收集错误原因分组 | 全体 | 已确认 |

## 第三遍（2026-09-24）：成品存放与治理层跟踪

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-159 | **治理层全部未被 git 跟踪**：`AGENTS.md`、`docs/rules/*`（13）、`.agents/skills/*`（7）、`tools/check_conventions.py` 与 `tools/agent_hooks/*`、对应测试全部 `git ls-files` 为空；而 `ci.yml` 已在运行 `python tools/check_conventions.py` ⇒ 干净克隆上 CI 必然失败 | `git ls-files` 逐路径核对 + `.github/workflows/ci.yml` | 全体 | 已确认 |
| E-160 | 计划由客户端写入 `.zcode/plans/plan-<session-id>.md`（命名来自客户端源码：`plan-` + 会话 ID 消毒），而 `.zcode/*` 被忽略 ⇒ 计划从不进版本控制 | `.zcode/plans/` 实际文件 + `.gitignore:64` | 全体 | 已确认 |
| E-161 | 决策散在四处且无统一形式：`docs/rules/CHANGELOG.md`（按轮次）、`OPEN_QUESTIONS.md`（按 Q 号）、`docs/R7_PUBLIC_DATA.md`（自称 decision log）、`docs/RESOURCE_BUDGET.md`（租卡决策）；全仓 0 处 ADR / decision record | 全仓 grep + 文件阅读 | 全体 | 已确认 |
| E-162 | 本机 agent memory 存放在 `~/.zcode/cli/memories/projects/<project-id>/memory/`（`MEMORY.md` 索引 + 主题文件），**机器本地、不随仓库共享** | 目录列举 + 客户端 bundle 中的 schema | 全体 | 已确认 |
| E-163 | `.zcode/*` + `!.zcode/workflows/` 能放行子目录：先写 `.zcode/*` 排除内容，再对子目录用 `!` **可行**（与"文件级放行需 `dir/*`"的教训不冲突）；用隔离临时仓库同时验证 `check-ignore` 与真实 `git add` 行为一致 | `git check-ignore -q` × 4 路径 + 临时仓库 `git add` 实测 | 全体 | 已确认 |
| E-164 | `.zcode/workflows/` 与 `~/.zcode/workflows` 当前均不存在，数据库 `workflow_definition`/`workflow_run` 表 **0 行** ⇒ agent workflow 在本机从未被创建过 | 目录查找 + sqlite 表计数 | 全体 | 已确认 |
| E-165 | `docs/rules/EVIDENCE.md` 自身有两处**编号复用**：`E-004` 与 `E-009` 各被用于两个不同发现（阶段 0 与阶段 1.12 / 1.10）。已在第三遍改为唯一编号（新条目 **E-167、E-168**），并确认无其它文件引用这两个编号的复用含义 | 全仓 `grep 'E-004\|E-009'` + 逐行核对 | 全体 | 已确认 |
| E-166 | `docs/rules/repository-layout.md` 的「观察（不立规）」曾称 `.mimosa/` 未在 `.gitignore` 中，实测已被 `.gitignore:63` 忽略 ⇒ 该陈述过时 | `git check-ignore -v .mimosa/` + 文件比对 | 单点 | 已确认 |

## 第四遍（2026-09-24）：命名约定测量

对活跃区 **204 个 `.py`** 做 AST 全量测量（归档目录与 `.venv` 除外，0 个解析失败）。
每条给出实测计数与例外位置，供 `docs/rules/naming.md` 引用。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-169 | 模块/包文件名：**204/204** 为 `snake_case` 或 `__init__.py`，0 例外 | AST + 路径扫描 | 全体 | 已确认 |
| E-170 | 类名：89 个中 84 个 PascalCase；其余 5 个是**带前导 `_` 的私有类**（`model/r7_baselines.py:22 _Base`、`data/download/arco_tiny_bounded.py:14 _NoRedirect`、`tests/test_agent_hooks.py:61 _Stdin`、`tests/test_forward.py:57 _SplitVerifier`、`training/r7_memory.py:7 _Packed`）。允许前导 `_` 后为 **89/89** | AST 扫描 | 全体 | 已确认 |
| E-171 | 函数/方法名：**936/936** 为 snake_case（dunder 83 + 私有 115 + 公开 738），0 个驼峰 | AST 扫描 | 全体 | 已确认 |
| E-172 | 模块级常量：172 个候选中 169 个 UPPER_SNAKE；3 个例外全是 `__all__`（模块协议名）。豁免后 **172/172** | AST 扫描（按"值为字面量/容器"识别，不按名字大小写猜） | 全体 | 已确认 |
| E-173 | **标识符非 ASCII：0 命中**。同时 **43/204** 个文件含非 ASCII 的**注释/docstring**（中文），密度最高者是 `tools/check_conventions.py`（1344 字节）与 `data/preprocess/real_station_smoke.py`（1346）—— 规则必须只查标识符，不得误伤注释 | AST 标识符遍历 + 逐文件字节统计 | 全体 | 已确认 |
| E-174 | 禁用词与杂物桶：代码路径 **0 命中**、`_vN` 结尾 **0 命中**、杂物桶目录 **0 命中**。唯一命中的 9 个 `final` 全在 `audit/**`（`final_scorecard.md`、`real_data_logs/final_pytest.log` 等），是冻结的证据文件名 | 正则扫描全活跃树 | 全体 | 已确认 |
| E-175 | 测试命名：函数 **315/315**；文件 68/69（唯一例外是 pytest 规定的 `conftest.py`）。**关键**：`tests/` 之外有 **6 个** Lightning 协议方法以 `test` 开头（`training/lit_module.py:22 test_step`、`training/r7_lit_module.py:45`、`training/r7_process_forecast_lit_module.py:85`、`training/r7_recursive_lit_module.py:55`、`data/multiscale_dataset.py:72 test_dataloader`、`data/r7_dataset.py:93`）⇒ 规则范围必须限定 `tests/` | AST 扫描 | 全体 | 已确认 |
| E-176 | `r7` 主题标记：129 个含 `r7` 的模块**全部**匹配"定界位置"（42 个 `r7_` 前缀 + 5 个 `_r7` 后缀 + 82 个中缀），0 例外。**注意**：`model/` 内部混用三种方案（前缀 3 / 后缀 3 / 无标记 14），所以"统一为前缀"这条规则会被现有 5 个文件否决 | AST + 路径分类 | 全体 | 已确认 |
| E-177 | 配置 10/10 为 `snake_case.yaml`（两族：`r2_v6_*` 5 个、`r7_*` 5 个）；workflow 18/18 为 `kebab-case.yml`（17 个带 `r7-` 前缀，`ci.yml` 为通用入口） | 文件名扫描 | 全体 | 已确认 |
| E-178 | 文档命名分四层且各自 100%：`docs/*.md` 46/46 UPPER_SNAKE；`docs/rules/` 台账类 4/4 UPPER_SNAKE（`CHANGELOG`/`EVIDENCE`/`MIGRATION`/`OPEN_QUESTIONS`）+ 分类类 10/10 kebab-case；`docs/plans/`、`docs/decisions/` 3/3 `NNNN-kebab-case` | 文件名扫描 | 全体 | 已确认 |
| E-179 | 参数名：1803 个参数中 103 个 ≤2 字符，其中 44 个不在常规集合内，但高频者均为领域惯用（`ds` 10、`hw` 7、`u`/`q` 气象学、`B`/`D`/`H`/`W` 张量维度、`lr`、`kv`、`kw`、`fn`、`p0`、`wp`）。**结论：短名是本项目常态，不应阻断** | AST 参数扫描 | 全体 | 已确认 |
| E-180 | 命名规则初版把 `docs/rules/` 子目录一律判为 kebab-case，**被自己的门禁报出 4 例**（`CHANGELOG.md`/`EVIDENCE.md`/`MIGRATION.md`/`OPEN_QUESTIONS.md`）；核对后确认是**规则写错**——该目录台账类文件刻意沿用 UPPER_SNAKE。规则已改为分层判定 | 门禁自报 + 逐文件核对 | 全体 | 已确认 |
| E-181 | R-048 初版判据（"名字含 3 个以上连续辅音即报告"）实测命中 **113 处**，含 `mlp_ratio`、`model_cfg`、`training_std` 等完全清晰的名字。判定为噪音过高后**撤回该判据**，收窄为"单个无信息 token"，实测 0 命中 | 门禁实测（113 → 0） | 全体 | 已确认 |

## 第五遍（2026-09-28）：外部检索路由与引用

为 R-049/R-050 取证。前六条是仓库与客户端侧的测量；最后一条是对外可达性的**实测**
（8 个 HEAD 请求，`--max-time 5`，只记状态码与耗时，未落任何数据文件）。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-182 | `WebSearch`/`WebFetch` 门禁缺口：`tools/`、`tests/`、`.github/`、`scripts/`、`training/`、`model/` 的 py/yml/yaml/json/sh 内 **0 命中**；全仓命中只在 5 份 docs 与两个 agent 定义（`.zcode/agents/*.md`） | `grep -rn` 全活跃树 | 全体 | 已确认 |
| E-183 | agent 定义清单：`.zcode/agents/` 两个文件——`planner.md`（**已跟踪**）工具集 `Read/Grep/Glob/WebFetch/WebSearch`（`:7-12`）；`web-researcher.md`（**未跟踪**，`??`）工具集 `WebSearch/WebFetch`（`:7-9`）⇒ 引入前「唯一出口」不成立 | 文件 + `git ls-files` / `git status` | 全体 | 已确认 |
| E-184 | 「摘要≠证据」有真实先例：`cn.bing.com` 的 WebFetch 搜索摘要对 HRCLDAS/SMBFD 返回**完全无关**结果（浏览器游戏页面），文档明言「a search summary must not be treated as evidence」，且每条「可达」都以实测端点为准；另有可用性纪律「搜索得到的每个数字都必须回到一手来源核对」 | `docs/R7_URBAN_EXTENSION_BLOCKED.md:37,39-41`、`docs/goals/open-issue-resolution.md:97-103` | 抽样 2 处 | 已确认 |
| E-185 | `docs/*.md` 共 **26 行**含 URL：**14 行外部引用无访问日期**（arco-era5 3、arXiv 3、pytorch docs 2、ECMWF 1、xarray 1、UCI 1、GitHub 上游镜像 1、上游代码仓库 1、AWS 开放数据注册表 1）、6 行为本仓 CI run 链接、2 行为 curl 命令模板 | `grep -rn "https\?://" docs/*.md` + 逐行分类 | 全体 | 已确认 |
| E-186 | 工具事件的 hook 载荷**没有**子智能体身份：载荷构造 `{...e, agent_type:e.agentName, hook_event_name:…, permission_mode:…, session_id:…}`，而 `PreToolUse`/`PostToolUse`/`PermissionRequest`/`PostToolUseFailure` 的调用点（`foo`/`hoo`/`moo`/`goo`）只传 `cwd/hookEventName/mode/sessionId/…`，**不传 `agentName`**（该键只在 `SessionStart`/`UserPromptSubmit`/`Stop` 传入）⇒ 闸门无法按身份区分，只能按 `session_id` | 客户端运行时源码 `~/.zcode/server/agents/glm/zcode.cjs`（载荷构造与四个调用点窗口） | 单点 | 已确认 |
| E-187 | 子会话很可能**不运行**本仓 hook：运行器仅在 hooks 配置开启或存在 workspace hook 快照时创建，子会话的依赖里两者皆缺。**未实测**——安全探针需要一次会被拒绝的写操作，无法在不触碰受保护路径的前提下构造 | 同上游源码 `hookRunner` 构造与守卫窗口（代码推断） | 单点 | 推测 |
| E-188 | 本机可达性实测（2026-09-28，8 个 HEAD，`--max-time 5`）：`api.github.com` 200(2.4s)、`docs.python.org` 200(1.3s)、`pytorch.org` 200(1.5s)、`arxiv.org` 200(0.8s)、`cn.bing.com` 200(0.2s)；**`raw.githubusercontent.com` 超时（000，5.0s）**、`www.google.com` 000(0.1s)、`duckduckgo.com` 超时（000，5.0s）⇒ 抓 GitHub 原始文件要走 `api.github.com` 的 contents 接口，不能假设 raw 域名可用 | `curl -sS -I -L -m 5 -o /dev/null -w '%{http_code} %{time_total}'`，8 个 URL | 单点 | 已确认 |

**E-187 的处置**：该推测不阻塞决策——`guard_web_research_route.py` 已按 `sess_subagent_`
前缀放行，无论子会话是否运行 hook 都不会自锁；待验证事项登记为 `OPEN_QUESTIONS.md` Q-013。

## 第六遍（2026-09-29）：机器可读回执的首次真实运行

为决策 0019/0020 的严格 v1.1 回执取证：一次经用户授权的真实 CI 运行（决策 0021 通道）。
两条只读取自 GitHub Actions runs/jobs API（匿名读，未下载日志或产物）。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-189 | 首次真实 v1.1 回执在 CI 上生成并**通过严格校验**：`R7 real offline multiseed CPU study` run `36549248954`（commit `b967fc1`）第 8 步 "Build and validate verification receipt" 成功——builder 写盘前自校验与 `--require-success` 校验器同时通过（source bytes、protocol 源摘要、source receipt digest/字节数、data/code identity 全部一致）。产物含 `verification_receipt.json`，留存 30 天；未下载核对（匿名 API 无 artifact 下载权限） | `/repos/Eswink/UrbanPiDiT_R2/actions/runs/36549248954/jobs`，访问日期 2026-09-29 | 单点 | 已确认 |
| E-190 | 同一提交的主门禁 `R7 CPU CI` run `36549248794` 在第 7 步失败：`git show --check` 报 `docs/R7_CANDIDATE_BRIEF.md:120: new blank line at EOF`（生成器规范文本已以换行结尾，导出又追加一个）；修复提交 `bf96ea8` 的 run `36549905342` 八步全绿。**教训**：`git diff --check` 看不到未跟踪文件，且本地提交前 guard 不检查空白——新文件提交前应跑 `git diff --cached --check` | `/repos/Eswink/UrbanPiDiT_R2/actions/runs/36549248794/jobs` 与 `…/36549905342/jobs`，访问日期 2026-09-29 | 单点 | 已确认 |

## 第七遍（2026-09-29）：规模与命名的全面测量、范围缺口与文档漂移

为决策 0022 取证：在给文件行数、函数长度与命名加硬上限之前，先量分布、再查范围、再核对文档。
三条都是只读测量（`tools/check_conventions.py --report` 与 AST 复算）。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-191 | 规模分布实测（尺寸族范围 `SIZE_SCOPES` = `data/ model/ training/ scripts/ tests/ tools/` 加仓库根级模块；295 个 `.py`、49,148 行）：**文件** p50=101 / p90=343 / p95=580 / p99=946 / 最大 1,834，>400: 24、>600: 13、>800: 5、>1000: 3；**函数**（2,127 个）p50=10 / p95=58 / p99=128 / 最大 245，>100: 39、>150: 14、>200: 5；**行长**（49,148 行）p50=46 / p95=89 / p99=106 / 最大 198，>120: 169、>200: 0；**嵌套** p99=5 / 最大 9，>5: 18；**参数**（2,052 个非构造器函数）p99=8 / 最大 24，>8: 15。R-051 取 600（p95 与 p99 之间）、R-052 取 200（p99 与最大值之间） | AST 全量复算 + `--report`，口径见 `docs/rules/size-thresholds.md` | 全体扫描 | 已确认 |
| E-192 | 命名范围缺口实测：**(a)** `NAMING_PY_SCOPES` 不含仓库根级 8 个模块（`train.py`、`calibrate_r7_local.py`、`diagnose_r7_gain.py`、`evaluate_r7_local.py`、`prepare_r7_local.py`、`profile_r7_inference.py`、`train_r7_local.py`、`tune_r7_halting.py`），纳入后 295/295 仍 0 违规；**(b)** 目录名此前只被 R-043 的杂物桶词表扫到，实测 16 个代码目录 + 11 个 skill 目录全部合规（R-053 零改名落地）；**(c)** R-043 文档把范围写到 `configs/`、`docs/`、workflow 与根目录文件，实现只扫活跃 `.py`——扩到非 `.py` 会命中 3 个**版本名**（`configs/r2_v6_v100.yaml`、`docs/UPGRADE_FROM_V531.md`、根级 `CHANGELOG_V6.md`），故选择把文档收窄到实现；**(d)** R-047 的顶层 `README.md` 豁免"文档有、实现无"（该文件今天不存在，属预防性修复）；**(e)** R-048 文档写 0 命中，实测 1（`tests/test_r7_multiseed_comparison.py:29` 的 `ckpt`） | AST 复算 + `git ls-files` + 逐目录核对 + 词表试跑 | 全体扫描 | 已确认 |
| E-193 | 规模文档漂移：`size-thresholds.md` 曾写 R-021 命中 **0**（实测 21，范围补齐后 24）、R-020 写 **6**（实测 36→39）、R-022 写 10（实测 13→18）、R-023 写 11（实测 15）、R-019b 写 157（实测 169）。**两个根因**：一半是"这些数字此后没人核对"，一半是尺寸族范围不含 `tools/` 与根级模块——全仓最大的文件（1,741 行）对 R-021 不可见，"0 违规"里有一半是范围造成的。处置：范围补齐为 `SIZE_SCOPES`、全部数字重算、页内加 `<!-- measured: ... -->` 机器核对标记（`test_size_report_counts_match_the_checker` 重算比对） | 文档原文 vs `--report` 实测逐条对照 | 全体扫描 | 已确认 |

| E-194 | 本轮变更的 CI 验证与一处被冒烟测出的错拒：推送 `ca2f202` 后主门禁 `R7 CPU CI` run `36555351319` **八步全绿**（含 `Check repository conventions`＝新的 37 条阻断规则、`Check evidence index and candidate brief`、`Compile active modules and check whitespace`、全量 pytest），18 条 workflow 中 17 条实验 workflow 按设计 **skipped**（commit message 无标签）。另：用真实仓库对提交闸门做冒烟测时发现 **R-037 会拒绝"引入新治理资产"的那次提交**——`git add X && git commit` 的 hook 在 `git add` 之前运行，于是 X 尚未入索引，而 CI 检出提交树时它已被跟踪（本地红、CI 不红的假阳性）。已修为"被声明为提交内容的路径计为已跟踪"并加反证 | `/repos/Eswink/UrbanPiDiT_R2/actions/runs/36555351319/jobs`，访问日期 2026-09-29；闸门冒烟测：`guard_conventions_before_commit.py` 读 stdin 载荷（修复前 exit=2，修复后 exit=0） | 单点 | 已确认 |

## 第八遍（2026-09-29）：主模型 V2 归档前的只读结构核对

为 `docs/R7_MAIN_MODEL_V2_DESIGN.md` §3.2 取证：核对主模型递推步里「模型实际能看到什么」。
只读代码，不训练、不改动任何文件。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-195 | 主模型递给递推 cell 的键**没有来源角色标记**：`torch.cat([context, draft_tokens], dim=1)` 把 C 与 E(Y_k) 拼成 `[B,2N,D]` 后直接作 key/value，cell 无法区分哪一半是上下文、哪一半是草稿，拼接处也没有 mask。该写法在**三处镜像实现**里一致存在（`model/process_forecast_r7.py` 的 forward、`model/r7_halting.py` 的 `reasoning_step`、`training/r7_streaming.py` 的 `_recursive_step`）。#70 的 CPU 结构探针（只反转 draft token 顺序 → 默认 Process 反馈预测 allclose，max 差 1.19e-7）与该缺口相容，但**该探针本轮未重跑**：本条目只确认代码事实，不确认「这就是探针结果的原因」这一因果解释 | HEAD `e1d5a7d` 的三个实现文件原文（`model/process_forecast_r7.py` 中 `recurrent_context=torch.cat([context,draft_tokens],dim=1)`，无 role/mask） | 单点 | 已确认 |

## 第九遍（2026-09-29）：E0 两个预注册诊断的执行

执行二轮 §11 与三轮 §13 各自留下的「下一项第一个具体动作」。只读 val、只用已归档 checkpoint、
0 GPU-h；完整记录见 `docs/R7_E0_DIAGNOSTICS.md`。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-196 | 加一条 process 读取通路会**一致放大** solver 的修正幅度，但**不会**让修正与误差更反相关。四臂 × 三 seed × 22 val 窗口的第 3 轮修正：C−B 的 `update_energy`（全变量）逐 seed 差 +0.000178/+0.000046/+0.000307（三 seed 同号为正，池化比值 C/B = 1.31），D−B 同形；而误差—修正余弦 `cos(e,d)` 的差是 +0.0092/+0.0097/+0.0355，方向与「读取把 context 拉偏」的预测**相反**（反相关应当使余弦下降）。C 与 D 的数字几乎逐 seed 相同（`update_energy` 相对差 <0.1%），而 D 的 query 按位置池化、读取与位置无关 ⇒ 放大来自「这条通路存在」，不是「读是位置化的」。T2m 单变量上幅度差不一致同号（seed42 为负），故幅度结论只在跨变量索引上成立 | `outputs/r7_e0_diagnostic/e0_correction_replay.json`（12 个 checkpoint 的 sha256、逐 seed 数字、`model_code_digest`＝`8d9262d1…`）；回放代码为 `d8aff68` 的 `git archive`；驱动 `scripts/study_r7_e0_correction_replay.py`；判读的预注册原文见 `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md:329-340` | 抽样（12 checkpoint × 22 窗口 × 1 段） | 已确认 |
| E-197 | 三轮 §13 的「E−A 在 t2m 72h 三 seed 同号」**为假**：逐 seed 是 −0.1629 / **+0.2378** / −0.9492，seed42 反号，其均值 −0.2915 是反号三值的算术平均；比较器自己已把该格标为 `unresolved`。三轮文档**自己的** §7 表也把 72h 标成「四对全部否」。而 48h 那一格三 seed 同号且跨两轮复现：C−B −0.2254、D−B −0.2235、E−A −0.2468 K（三次独立配对、各三 seed 同号，彼此相差 11% 以内）。因此三轮 §13 的「若不稳定」分支被触发（72h 的 seed42 异号在两轮都出现） | `outputs/r7_e0_diagnostic/e0_paired_stability.json`（两个输入 JSON 的 sha256 + 逐格 `mean_reproduced`/`sign_claim_reproduced`）；被推翻的原句 `docs/R7_71_72_ROUND_THREE.md:463-472`；自身矛盾处 `docs/R7_71_72_ROUND_THREE.md:228` | 抽样（2 轮 × 3 对 × 2 时效） | 已确认 |

## 第十遍（2026-09-29）：RW-B 的一轮有界真实对照（E1/D5）

首次执行 #72 M2-B 的有界实验（决策 0021 的会话内授权，2 seed × 4 臂 × 400 updates，只读 val）。
记录见 `docs/R7_72_RW_B_PILOT.md`。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-198 | RW-B 相对 RW-A 在登记主端点上**未获支持**：t2m 逐 seed 同号的三态是 supported×2（12h −0.079、24h −0.075 K）/ worsened×3（6h +0.118、48h +1.065、72h +1.580 K），模态读数为 worsened，改善比恶化小一个数量级。登记文字里的两条证伪条件**均未触发**，故严格读法是「模态不支持且长时效受损」，不是「被证伪」 | `outputs/r7_72_rw_b_pilot/paired_comparison.json` 的 `primary`（`protocol_sha256` `6f488742…`、`model_code_sha256` `9ddd2660…`、8 个 run 同 digest） | 抽样（2 seed × 4 臂 × 85 格） | 已确认 |
| E-199 | 独立于结果的两条实测：(a) RW-B 的第一步修正幅度约为 RW-A 的 **1.8 倍**（能量约 3.4 倍），且第 2–3 步误差—修正余弦**转正**（+0.070/+0.057）而 RW-A 全程为负，第 3 步恶化比例 0.61/0.57 超过一半；(b) K=1/2/4 探针上两个 seed 都不单调改善，6h 处 K=4 一致最差且 RW-B 在每个 K 上都差于 RW-A。这两条与 E-196、#66/S4 同向，但本轮**没有**能做归因的消融臂 | `outputs/r7_72_rw_b_pilot/seed*/seed_result.json` 的 `probes.correction` / `probes.depth`；`depth_probe_table.csv` | 抽样（2 seed × 8 窗口 / 2 seed × 2 臂 × 3 K） | 已确认 |
| E-200 | 一轮**实现缺陷**与它的代价：首次尝试在训练与 40 次评估**全部完成后**，于新写的修正探针抛 `KeyError: 'atmos_target'`（`ZarrRolloutDataset` 的字段名是 `rollout_targets`）。该尝试的产物目录**保留未删**（`outputs/r7_72_rw_b_pilot_attempt1_probe_defect`），其 0.4203 GPU-h 照记，数字**未采用**；修字段映射+深度守卫、补 reader 回归钉住测试后重跑一个干净尝试（0.4154 GPU-h），本轮合计 **0.8356 GPU-h**（自设上限 1.0，未超）。另一条同轮发现的工具缺陷：`tools/check_conventions.py` 的 R-006 里 `tolerated` 是在路径循环内才赋值的变量，无 bounded-study 函数的 study 作用域模块会**继承上一个文件的 tolerated 值**从而被静默放行（本轮把 `training/r7_arm_study.py` 改名出 `training/r7_*study.py` glob 以回避，未改 checker） | 失败日志 `logs/r7_72_rw_b_seed41.log` 的 traceback；两个产物目录的 `elapsed_seconds` 求和；`tools/check_conventions.py:535-561` 与 `tests/test_check_conventions.py::test_repository_study_exceptions_are_exactly_three` | 单点 | 已确认 |

## 第十一遍（2026-09-30）：goal 完成校验的两种失败形态（客户端只读取证）

为决策 0024 取证：用户报告 goal 模式下 UI 停在「目标校验中」不推进。全部证据来自本机客户端日志与
`db.sqlite` 的只读查询；不改动客户端状态，也不涉及本仓代码或数据。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-201 | goal 完成校验调用在客户端**没有任何超时**（只有 `abortSignal`），provider 不返回时会永久停在 `goal.status="verifying"`：UI 显示「第 N 次迭代 · 目标校验中」，该会话新输入被排队。2026-09-30 的实例悬挂 **1773.0 s** 后由用户暂停中止（`model_usage`：`status='cancelled'`、`cancelled_by_user=1`、`input_tokens=0`）。全量 26 次校验调用按 provider 分布：`new-provider`（deepseek-v4.1-flash）**14/14 completed**（11.6–47.4 s，最大输入 532,730 token）、`account:zai-start-plan`（GLM-5.3-Flash）**6/6 completed**、`new-provider-4`（cline-pass 路由）**0/6**（`error`×4 各 17.1–46.9 s 且 `input_tokens=0`；`cancelled`×2 各 1436.6 s / 1773.0 s）⇒ 失败与请求体积无关、与路由相关。已排除：objective 长度（990 码点）、`check_goal_brief.py`（0 失败）、仓库产物与工作区、本仓 hooks（该会话项目 hooks 处于 pending workspace trust 且 blocked）。代码依据：非流式校验调用（`Jka`）不传 `timeout`，主流式回合的 10 分钟 idle 看门狗不覆盖它。**另发现台账自身的漂移**：本文件「统计」块的覆盖度/置信度数字与按行重数不符，本次一并改正 | `~/.zcode/cli/log/zcode-2026-09-30.jsonl`（span `e6c36daf-e5a8-4a` 的起始与失败三连）；`db.sqlite` 的 `session_entry 7d7ce89b` / `08141ff0`、`model_usage` 该 `query_source` 全 26 行、`session_target.status='paused'`；客户端 `~/.zcode/server/agents/glm/zcode.cjs` 偏移 ≈12862744（失败语义）/≈12863500（调用点）/≈4022247（executor）；整理见 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2 | 抽样（26 次校验调用 + 2 次会话事件） | 已确认 |

## 第十二遍（2026-09-30）：RW-B 的减法归因（两个子开关、两条逐位等价与一个退化的负控制）

证据文档 `docs/R7_72_RW_B_SUBTRACTION.md`；产物 `outputs/r7_rw_b_subtraction_probe/`（D1，0 GPU-h）
与 `outputs/r7_72_rw_b_subtraction/`（D4，0.4128 GPU-h）。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-202 | **推理期留一消融把「×1.8 第 1 步修正幅度」定位到门控+锚定提案这条通路**：在归档的 RW-B checkpoint 上把门控+提案移除后，第 1 步修正幅度塌到 RW-A 的 **0.20/0.24 倍**（0.0070；两 seed 同向）；把 `solver_cell` 换成恒等（Z 停在 `solver_init`、提案与门控仍施加）后幅度是 RW-A 的 **1.48/3.04 倍**（两 seed 不稳）而**误差—修正余弦从第 1 步就转正**（+0.010/+0.021，RW-A 与 RW-B 第 1 步均为负）。⇒ 幅度需要 (a) 通路存在；余弦转正出现在任何让 Z 停止跨步递推的配置里，**两者不是同一件带来的**。同一探针测出：该臂训练时 `correction_head` **从未收到梯度**（137 个移动过的张量里 0 个 correction_head 张量），所以「移除 (a)」这一行是未被训练的修正头 + RW-B 其余权重，**不能**当成训练期结论 | `outputs/r7_rw_b_subtraction_probe/probe.json`（sha256 `d56cb8b4…`，含全部 32 个 checkpoint 的 sha256 与 `running_model_code_sha256` 核对）；驱动 `scripts/study_r7_rw_b_subtraction_probe.py` | 抽样（2 seed × 8 val 窗口 × 1 lead） | 已确认 |
| E-203 | **拆出的两个子开关在两端都逐位等价，且新增零参数**：`local_solver_state=False` ≡ 起点修订 `e6085bc` 的实现；`local_solver_state=True` 且两子开关默认 ≡ 起点修订的 RW-B——两条都用冻结修订树 + 双包名导入、逐字节零容差比较，并有「扰动冻结树必须打破等价」的反证。组合矩阵按冻结语义保留**一处必须发生的坍缩**：`(递推开, 门控关)` 与 `(递推关, 门控关)` 输出**逐位相同**而前者仍返回状态 | `tests/test_r7_rw_b_subtraction.py`（15 项）；`model/process_step_r7.py`；`pytest -q` 本机 1499 passed / 3 skipped | 全体（该文件内两条端到端等价 + 反证） | 已确认 |
| E-204 | **预注册的负控制 `RW-B−(a)` 在构造上退化为 RW-A 本身，因此它的「同号恶化」是浮点噪声**：同一份权重分别装进 RW-A 与 `solver_gate_proposal=False` 两个构造，前向、逐步 drafts 与 adaptive 三路**逐位相同**（最大绝对差 `0.0`，唯一区别是多返回一个状态）；同一份权重下一步 streamed backward 的总损失**逐位相同**、131 个共享张量梯度**全部逐位相同**（最大相对差 `0.000e+00`），solver 侧**0 个非零梯度**。训练后两臂的共享权重相对差 **1.6e-4**（同比较下 RW-B 对 RW-A 是 **1.74**，相差四个数量级），修正几何与 RW-A **逐位相同**。⇒ 该臂区间 −0.000011…+0.000078 K 的「delta」是 GPU 非确定性的轨迹差，不是混杂。**按冻结分支规则本轮仍不做归因**（规则照写触发：`stop-confounded-control`），并如实记为**本轮控制臂设计的缺陷** | `outputs/r7_72_rw_b_subtraction/paired_comparison.json` 的 `primary.branch` 与三对逐 seed delta；本机权重/梯度逐位比较（同一权重双构造）；`outputs/r7_72_rw_b_subtraction/arm_table.csv`、`memory_table.csv`、`training_table.csv` | 全体（逐位比较覆盖全部共享张量） | 已确认 |
| E-205 | **上一轮登记的 `RW-B − RW-A` 在改过 `model_code_sha256` 与 `protocol_sha256` 之后几乎逐位复现**：t2m 五时效逐 seed delta 与 `docs/R7_72_RW_B_PILOT.md` 的登记值对照，最大偏差 ~1.8e-4 K（6h 0.198648 vs 0.198633；48h 1.061699 vs 1.061712；72h 1.817077 vs 1.816831）。⇒ 该负结果不是一次性噪声；且新增的两个子开关**没有移动默认路径**（与 E-203 的两条等价一致）。两轮 digest 不同，**不得相加或并列** | `docs/R7_72_RW_B_PILOT.md` §3；`outputs/r7_72_rw_b_subtraction/paired_comparison.json` 的 `round_reference` | 抽样（2 seed × 5 时效 × 1 变量） | 已确认 |
| E-206 | **同一「去掉递推」干预在推理期与训练期差 35–60 倍**：D1 在冻结门控的归档 checkpoint 上移除递推，第 1 步幅度是 RW-A 的 1.48/3.04 倍；而从零训练的 `RW-B−(b)` 第 1 步幅度只有 **0.0015**（RW-A 的 0.04/0.05 倍，约 23 倍更小），余弦全程为正。⇒ **门控在「Z 不递推」的配置里学会几乎关闭**，这是门控作为稳定器起作用的直接证据，也说明推理期消融读数不能直接当成训练期结论 | `outputs/r7_rw_b_subtraction_probe/probe.json` 的 `recurrence_removed`；`outputs/r7_72_rw_b_subtraction/seed*/seed_result.json` 的 `probes.correction` | 抽样（2 seed × 8 窗口） | 已确认 |

## 第十三遍（2026-09-30）：N1 四块零GPU审计与冻结随机Z的不可分辨停止

证据页 `docs/R7_N1_PIVOT_AUDIT.md`；零GPU复算 `outputs/r7_n1_audit/`，一次具名授权的D2
`outputs/r7_72_frozen_z/`。下列事实不认证科学机制，也不把工程成功当目标完成。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-207 | **训练目标权重轴是内部K，而非物理时效**：K=3、final_weight=2时四草稿权重精确为1/6、2/9、5/18、1/3，全部对同一+6h target。真实train186/val22的manifest lead全为6；process_weight=0；+6h归一化MSE选择checkpoint。自由48/72h评估却分别含8/12次+6h预测写回，内部K间detach不覆盖物理自由轨迹反传。这确认目标/暴露的结构失配，不单独归因RW-B相对RW-A的损害 | `training/r7_streaming.py:123-160`、`training/r7_scheduled_runner.py:367-392`、`model/r7_rollout.py:72-88,114-139`；`outputs/r7_n1_audit/audit.json` SHA256 `d895c96dd4936ebb75adbf9bd23bb8799185bd00b1c41482c6d8182c7a2c4a07`；证据页§2/§4 | 单点（一个冻结协议与实现） | 已确认 |
| E-208 | **data regime的量化是描述，不新增噪声阈值**：val五lead窗口22/21/19/15/11；旧减法轮RW-B−RW-A在48/72h平均delta为1.065106/1.580487 K，除以RW-B两seed max−min为1.599049/0.757468；除以RW-A spread为1.615576/0.979625。48h既有同号恶化不能用spread抹去，72h同量级幅度限制归因与外推，不是显著性或新判据。标准库只读六个具名输入，复算stdout digest与保存JSON一致，未读test/checkpoint/cache数组 | `tools/recompute_r7_n1_audit.py`（SHA256 `0acb81c18e5ee6944ad10e4034177ed2732ced8ace00186b0e36e6a88c690dba`）；`outputs/r7_n1_audit/audit.json`及`input_pins.json`；证据页§3/§6.1 | 抽样（2seed×2长lead，旧归档算术） | 已确认 |
| E-209 | **冻结随机Z的构造不新增solver或模型开关**：private CPU RNG、seed+1000003、FP32 normal×0.02、shape[1,1089,192]，同seed跨样本/内部K/物理转移复用。原solver_cell前向仍算但输出被替换，只冻结solver_init/cell，门控/提案仍学习；同seed151个RW-B原参数安装前相同，RW-A转入131共享张量/ignored0。strict加载前安装、加载后buffer/spec校验；CPU测试显示solver权重未动而proposal/gate有梯度和学习。总参数不减但trainable及反向FLOPs下降，不能声称容量/算力完美匹配 | `training/r7_frozen_z_intervention.py`、`tests/test_r7_frozen_z_control.py`（89passed）、`tests/test_r7_frozen_z_runner.py`；D2协议digest `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01`与seed_result.arm_pairing；证据页§5.1/§5.3 | 全体（本轮声明的构造与CPU反证） | 已确认 |
| E-210 | **主问句不可分辨并触发停止**：冻结Z−RW-A的t2m48h两seed为+0.063121651/−0.193231282 K，72h为+0.325881772/−0.151426589，均unresolved/mean=null；分支cannot-distinguish、stop_required=true、N2d仅提议。同轮RW-B−RW-A两长lead仍worsened；冻结Z−RW-B虽同号改善但属于单独载体对比，不替换主问句。六训练各400更新、30val评估、510RMSE cell/255depth0行/3×85cell完整，共同case22/21/19/15/11；按归档code.zip纯元数据重算paired JSON逐字节相同，未再执行模型 | `outputs/r7_72_frozen_z/paired_comparison.json` SHA256 `d7a345c17923cd49d1b03f2cbea767c95579e5d18a043965bf29fb697c222a64`；merged/30provenance/CSV；`outputs/r7_n1_audit/metadata_replay.json`；证据页§5.2/§6.1 | 抽样（2seed×400updates×一个冬季段） | 已确认 |
| E-211 | **实际预算合规，但四成本视图（参数/FLOPs/吞吐/内存）有实质缺口**：attempt success，GPU1292.544625543058s=0.3590401737619605h（≤0.45），whole1297.8983452636749s（≤1800）。validation内部不检查deadline、future finalizer未强制全集合；本次终态全集合已外部只读核齐，未超预算。两seed的30个eval峰值均255424000B，继承末臂training峰值且未reset，不是独立eval测量，内存视图不得称验收齐全。merged的whole字段实际是GPU区间；whole取attempt。代码/产物保留不热改、不补测 | `attempt.json` SHA256 `a4d8da66f8d25f68c89b5ccbf680f498b000fb82e6884e0099b6b9514074f226`、四成本/案例CSV；`training/r7_scheduled_runner.py:108-128,367-377`、`scripts/study_r7_72_frozen_z.py:219-243,289-313`、`training/r7_arm_harness.py:374-397`；证据页§5.3/§5.4 | 单点（一次运行及其计量/守卫） | 已确认 |
| E-212 | **代码身份与工程CI有精确绑定，失败照记**：实验commit c4e7e83，code.zip SHA256 `5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a`的953路径/字节与commit核齐；model digest不变。初次CI36690064571的pytest因R-021 marker34/actual35失败，本地重现，文档标记修复efa410b的CI36691526554 completed/success、九主步骤成功；精确efa410b的干净clone1795passed/14skipped/2warnings，skip不算通过，远端日志计数未取得。Mimosa报scanner_enobufs，安全扫描无结论；不能把CI成功或扫描fail-open解释为机制/安全通过 | `code_commit.txt`、`code.zip`、git blob比对；[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36691526554)、[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36691526554/jobs?per_page=100)，访问日期2026-09-30；`outputs/r7_n1_audit/ci/`与本地pytest日志；证据页§6.2 | 单点（本轮代码及两次CI） | 已确认 |

## 第十四遍（2026-09-30）：N1 一次具名独立evaluation成本补测失败

证据页`docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`；原证据页与归档保持不变，failed也计费，不自动重试。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-213 | **一次成本补测只完成1/30，零基线守卫如实拒绝，成本验收仍阻塞**：执行前AskUserQuestion具名授权30原val/0训练/≤324s/失败即停；计量代码1e03f82、原code.zip不改。首项seed41/RW-A/+6h从allocated/reserved均0开始，peak为39590400/46137344B，22case/17变量RMSE及逐caseMSE精确重放；第二lead12h在evaluate调用前因allocated或reserved基线非零失败。没有result、四成本CSV或其余29行，没有自动重试。原guard没记录具体非零字节数，原因未取证；不能断言内存泄漏或外部负载 | `outputs/r7_n1_eval_cost_supplement/attempt.json` SHA256 `662dd5ab2862bd4b0a1cfc585366bfe4b29a0e57edffd6f7f03715f405a35479`、worker日志、首项cost_measurement SHA256 `b654d17d812d3353e887b392505248f9bbdf5d8796c397b2dccb51ab85dcf270`；新protocol canonical `1ce9222321bfe6d799b0f86d7bc0ff4de127d451edaa0e5e8a45ca5a4a3ffc22`；证据页§1–3 | 单点（一次失败尝试及首项计量） | 已确认 |
| E-214 | **失败全额计入账本，不扩大授权或改科学读法**：GPU保守区间21.443860329687595s＝0.005956627869357666h，whole22.372659532353282s；原N1加失败共0.3649968016313182h≤0.45，算术余0.08500319836868184h不构成重试许可。campaign精确已用4.076196801631319/余19.923803198368685，显示4.0762/19.9238。原主48/72h仍unresolved/不可分辨，N1/paused；GPU1由用户手动腾出。另授单个旧PID的SIGTERM两次检查均未发信号（API缺失/旧PID消失），不干预替代任务；计量子进程失败后退出，没有新训练/节点推进 | 原attempt与`outputs/r7_n1_cost_authorization.json`、两份`r7_n1_cost_gpu_release*_result.json`；只读budget算术、GPU状态；证据页§1/§4–5，主计划账本 | 单点（一次失败成本与停止处理） | 已确认 |

## 第十五遍（2026-10-01）：补测失败原因的 0 GPU-h 审阅与修复轮准备

无新实验；只读审阅 `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md` 所记录失败的原因边界，修复轮长文
`docs/goals/n1-cost-supplement-repair.md`（prepared，未执行）；失败记录与归档在本遍未被改写。

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-215 | **失败原因审阅把范围收窄到进程级 allocator 残渣（推测、未取证），修复设计＝零基线由构造保证（每次评估一个全新进程）**：确认（a）失败发生在第二项评估调用之前，清零守卫（`training/r7_n1_cost_replay.py:268-273`）是该子步骤第一个分配相关的 CUDA 触点，之前只有 CPU 哈希/子进程查询/设备属性；（b）跑过的评测器与归档 code.zip 逐字节相同（`training/r7_evaluate.py` sha256 `95fff1be…`）；（c）checkpoint 以 `map_location='cpu'` 载入（`training/r7_experiment.py:125`）；（d）对活跃代码表面的扫描未发现会持有 CUDA 张量的模块级状态——唯一 `lru_cache` 只产生 CPU 张量（`model/sparse_process_graph.py:6-18`），无模块级 torch 张量，两处 `register_forward_hook` 分属冻结 Z 干预与 DDP smoke、均不在 RW-A 评估路径；（e）环境 torch 2.11.0+cu128 与归档 protocol 一致，且存在 `torch._C._cuda_clearCublasWorkspaces`。**推测（未取证）**：torch 进程级工作区族（cuBLAS/cuBLASLt）在第一项完成后常驻、`gc/empty_cache` 释放不掉；v1 守卫拒绝时未记录字节，持有者与字节数均未取证，不能写成已确认。修复＝**每次评估一个全新进程**（30 子进程，不调用释放/私有 API；已考虑未采用私有 API 方案），探针三态读法已预声明（修复轮长文 §3.1）。登记核验：失败页 sha256 `4b350357…`、三提交 `1e03f82/33d57d6/9d8d2b6` 均在 git、CI `36737308495` 对 `9d8d2b6` completed/success 九主步骤全绿（匿名 API，访问 2026-10-01）。本遍 0 GPU-h，账本未增行 | `outputs/r7_n1_eval_cost_supplement/worker_seed41_process_spacetime_rwa.log:11-15`、`scripts/measure_r7_n1_eval_worker.py:28-42`、`training/r7_n1_cost_replay.py:268-273`、`model/sparse_process_graph.py:6-18`、`training/r7_experiment.py:125`；只读命令见 `docs/goals/n1-cost-supplement-repair.md` §1.1；[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36737308495) | 全体（排除性扫描）+ 单点（一次失败尝试） | 已确认 |

## 第十六遍（2026-10-01）：N1 修复扩围的 GPU 共驻政策

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-216 | **用户决定本机 GPU 默认使用空闲显存共驻，不干预邻居**：修复轮明确预声明启动/每次 spawn 前只读余量门槛 ≥2048 MiB，PID 只记录，禁止非本实验信号与冻结/终止自动化，独占须另取具名授权。政策三件套已写为决策0026、R-054与AGENTS一行；读数不是显存预留，共驻墙钟影响列为代价。起点7064ff1，对表failures=0 notes=4；v1失败页sha4b350357…、review sha9540549b…、计量归档shab7f20e7b…只读核齐。此条只证明政策决定与落地，不证明GPU运行或成本验收；本步骤0 GPU-h | `docs/goals/n1-cost-supplement-repair.md` §1.2/§2/§5与本轮用户objective；`docs/decisions/0026-shared-gpu-coresidency-policy.md`；`docs/rules/gpu-resources.md`；`AGENTS.md`；2026-10-01开工git/hash/对表输出 | 单点（一次用户政策决定与本轮落地） | 已确认 |

| E-217 | **v2工程修复按逐评估新进程与余量门槛实现，CPU证据不冒称CUDA验收**：父调度30个seed/arm/lead子进程、单行summary；UUID固定单可见卡，邻居只记录，拒绝先写字节/snapshot。P1单matmul+私有clear仅诊断，零残渣才P2两原val同进程，不能归因即停。30行/17RMSE/source/checkpoint/归档code.zip精确重放契约保持。独立工程复核发现failed P1可能先触发P2、输出可能写v1子目录，均加严格父报告身份/状态/分支验证与冻结v1 ancestry拒绝及反证。定稿初步329passed、复核新增后175核心/资源/探针/政策passed；GPU尚未授权/执行，账本不增加 | `training/r7_n1_cost_replay.py`、`scripts/measure_r7_n1_eval_cost.py`、`scripts/measure_r7_n1_eval_worker.py`、`scripts/probe_r7_n1_allocator.py`及三个计量/探针反证测试；`/tmp/n1-v2-preparation-targeted.log`，最终完整回归/clone/CI另记轮次进度 | 全体（本轮声明的实现与CPU反证） | 已确认 |

| E-218 | **v2工程准备门禁通过，但具名GPU授权未答，仍无实际成本验收**：准备b227020完整SHA的CI36855190840 completed/success九主步骤绿；精确clone1988passed/14skipped/2warnings173.22s，定向359passed。原30val/六checkpoint/source只读身份与v1三个冻结hash核齐。执行那一刻AskUserQuestion具名探针+30val/共驻/≤0.25GPU-h/≤1200s/失败全额计费即停不重试未收到回答；不生成授权回执、不执行GPU，probe/attempt/result/成本四表/cost_views未产。实耗0，账本不增，N1paused及cannot-distinguish/N2d提议不变；Mimosa scanner_enobufs无安全结论 | `docs/R7_N1_COST_V2_PREPARATION.md`、`outputs/r7_n1_cost_v2_preparation/preparation_verification.json` SHA256 `362c8686…`；[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840)与[jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36855190840/jobs?per_page=100)，匿名curl访问2026-10-01；本轮AskUserQuestion返回“用户未提供回答” | 单点（本轮工程准备与授权阻塞） | 已确认 |

## 第十七遍（2026-10-01）：N1 v2具名一次成本补测成功（不改变科学读法）

| 编号 | 发现 | 证据 | 覆盖度 | 置信度 |
| --- | --- | --- | --- | --- |
| E-219 | **P1纯torch复现进程级可释放工作区族，条件P2不触发**：用户随后明确答「明确授权，刚刚没有看到」，确认先前AskUserQuestion的一次具名探针+30val共驻≤0.25h/≤1200s范围；回执SHA0770b4f8…先冻。1024² FP32 matmul删除+gc/sync/empty_cache后allocated/reserved8519680/20971520B、1snapshot段，private clear诊断后0/0。按预声明三态①、P1内0.9619s/含启动至退出4.1747s≤60、requires_p2=false；P2不触发不是skip。只确认工作区族，不回填v1当时字节或具体对象，不在30成本路径用私有clear | `outputs/r7_n1_eval_cost_supplement_v2/probe_residue.json` SHA256 `b94aa9383e453635bce5a67edaa38696e816a630ef6dda23f8a09b7474f134b9`、probe parent/claim/exit、具名授权回执；`docs/R7_N1_COST_SUPPLEMENT_V2.md` §1–2 | 单点（一次P1原因探针） | 已确认 |
| E-220 | **30独立eval成本齐并全额记账，原cannot-distinguish不变**：原六checkpoint×五val，30不同PID/launchID、30零baseline、正allocated≤reserved、528案例/510RMSE格；30rmse.csv文件hash等于原归档，固定provenance/case/MSE精确重放。参数/FLOPs/train吞吐引用原值，四表3/3/6/36行与cost_views digest齐；attempt/result success，无失败/重试/未确认退出。GPU461.74807197228074s=0.12826335332563354h≤0.25，whole463.4213050529361s≤1200；账本显示4.2045/余19.7955。原N1+v1+v2共0.49326015495695175h，v2另授权成本审计，不冒称总≤原0.45h。共驻门槛minfree23713MiB且无外部PID观测，不等于有邻居负载性能实测；N1仍paused/N2d仅提议，0训练/test未读，v1原hash未变 | v2 `attempt.json` SHA256 `e172da59a39ceb9b456e2e910b0e616e67876b8a1e51df34e2d530e1f1cae9e0`、`result.json` SHA256 `7b758e2d428f9a256e8b60fb97535c479b9114b9098aa20d0d3aa34b5b076b0c`、四表/cost_views/启动前后记录；protocol canonical `877d0cab…`；`docs/R7_N1_COST_SUPPLEMENT_V2.md` §3–5 | 全体（本轮30项与终态成本契约） | 已确认 |

## 统计

- 台账条目：**220** 条（E-001 – E-220；第一遍143 + 第二遍15 + 第三遍10 + 第四遍13 + 第五遍7 + 第六遍2 + 第七遍4 + 第八遍1 + 第九遍2 + 第十遍3 + 第十一遍1 + 第十二遍5 + 第十三遍6 + 第十四遍2 + 第十五遍1 + 第十六遍3 + 第十七遍2）。
- 按覆盖度（2026-10-01按行重数）：全体/全体扫描 **148** 条、抽样 **20** 条、单点 **52** 条。
  保留此前漂移处置：旧值「140/21/39」与当时行数不符，已经逐行改正；本次再按新增条目累加核对。
- 按置信度（2026-10-01按行重数）：已确认 **219** 条、推测1条、未知0条；旧值「已确认197」是历史漂移。
  不确定者写入`OPEN_QUESTIONS.md`，不编造答案；推测E-187已登记Q-013；E-215 记明其推测成分（失败字节未取证）。
- 未列入凭据类条目：历史5类凭据模式全部0命中，不是本轮重新全仓安全扫描。
