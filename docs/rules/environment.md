# 完整数据依赖（三份 requirements 未覆盖的部分）

三份 `requirements*.txt` 与 `pyproject.toml` 的 optional 组并集**不等于**项目实际运行所需的全部依赖。
以下包被代码 import，但任何 requirements 文件都未声明——CI 是在**单个 workflow 里临时 pin** 的。

由 2026-09-24 的虚拟环境搭建（`.venv`）发现：装上三份 requirements 后仍有 1 个测试失败
（`test_r7_installed_package` 需要的 `pip` 不在锁内）与若干运行时 `ModuleNotFoundError` 风险。

## 缺口清单

| 包 | 谁 import | 三份 requirements | CI 中的处理 |
| --- | --- | --- | --- |
| `icechunk` | `data/download/earthmover_pilot.py:265`（函数内延迟导入） | ❌ 未声明 | `r7-continuous-pilot.yml` 等 3 条 pin `icechunk==2.2.2`；`r7-earthmover-probe.yml` 用 `icechunk>=1.1,<3` |
| `pcodec` | `icechunk` 的运行时编解码依赖 | ❌ 未声明 | 随 `icechunk` 一起 pin `pcodec>=0.3` |
| `numcodecs` | `data/download/arco_tiny_bounded.py:63`（函数内延迟导入） | ❌ 未声明 | 作为 `zarr` 的依赖被间接带入 |
| `pip` | `tests/test_r7_installed_package.py` 调用 `python -m pip wheel` | ❌ 未声明 | `ci.yml` 显式 `pip install --upgrade pip` |

**为什么要写这份清单**：用 `uv venv` 或 `python -m venv --without-pip` 建的环境默认没有 `pip`，
会让 `test_r7_installed_package.py` 失败；而 `icechunk`/`numcodecs` 是**延迟导入**，
只在跑到对应下载路径时才炸——静态检查 `requirements` 不会发现。

## 本项目虚拟环境的精确复现

```bash
cd /data/esw/UrbanPiDiT_R2
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "torch>=2.4" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-data.txt -r requirements-r7-data.txt
uv pip install --python .venv/bin/python "icechunk==2.2.2" "pcodec>=0.3" pip setuptools wheel
```

本机实测结果（2026-09-24）：`622 passed, 3 skipped`，GPU 为 2×RTX 3090，
`torch==2.11.0+cu128`，CUDA 12.8。

## 说明与建议

- 这份清单是**记录**，不是新的依赖契约。把它并入 requirements 属于依赖面变更，
  会影响正在跑的科研流程，因此未擅自合并；建议单独一轮处理（见 `OPEN_QUESTIONS.md` Q-011）。
- CI 的工作流级 pin（`icechunk==2.2.2`）与 `r7-earthmover-probe.yml` 的范围 pin
  （`icechunk>=1.1,<3`）**不一致**。本地环境采用前者，因为它是三条 pilot workflow 的实测 pin。
- 若将来把 `torch` 装成 CPU-only 版本，本项目大量测试仍可通过，但
  `--device cuda` 相关路径与 `test_bare_cuda_device_is_accepted_on_real_gpu` 会被 skip。

## conda 共存（本服务器特有）

本机有 miniconda，且 `auto_activate_base=True`，所以终端默认落在 conda base。
这会让提示符在项目内显示成 `(.venv) (base)`。

**功能上不冲突**（已实测）：`.venv/pyvenv.cfg` 的 `include-system-site-packages = false`，
conda 的 `site-packages` 不在 `sys.path`，`torch`/`numpy`/`xarray` 全部解析在 `.venv` 内。

**已做的处理**：`.bashrc` 的 `project venv auto-activation` 块在**受管项目目录内**
临时退出 conda base，离开时自动恢复。行为边界（均已实测）：

| 场景 | conda 状态 | python |
| --- | --- | --- |
| 家目录 / `/tmp` / 其它项目 | `base`（不变） | miniconda |
| 项目内（含子目录） | 退出 base | `.venv` |
| 离开项目 | 自动恢复 `base` | miniconda |
| 用户先前手动退出过 base | 保持退出（不强行激活） | 按用户选择 |

设计要点：只在 `CONDA_DEFAULT_ENV=base` 且 `CONDA_SHLVL=1`（即 base 是被自动激活的）时才隐藏；
只恢复本机制自己隐藏过的 base；`conda` 命令在项目内仍可用，随时 `conda activate <env>` 可手动切走。
开关：把 `__PROJECT_HIDE_CONDA_BASE` 设为 `0` 即关闭隐藏行为。
