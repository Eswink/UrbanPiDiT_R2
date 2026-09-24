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

## 统计

- 台账条目：**181** 条（E-001 – E-181；第一遍 143 + 第二遍 15 + 第三遍 10 + 第四遍 13）。
- 按覆盖度：全体扫描 134 条，抽样 16 条，单点 31 条。
- 按置信度：已确认 181 条，推测 0 条，未知 0 条。
  （凡不确定者均写入 `OPEN_QUESTIONS.md`，不在此处填一个看起来合理的答案。）
- 未列入凭据类条目：5 类凭据模式全部 0 命中，故无"疑似凭据点位"可报告。
