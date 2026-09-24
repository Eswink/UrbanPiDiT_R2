---
name: environment-rebuild
description: 当需要重建运行环境时使用——换机器或容器、ModuleNotFoundError、CUDA 不可用、测试因缺依赖失败。含未声明依赖清单与虚拟环境搭建顺序。
---

# 重建运行环境

## 何时使用

- 换机器 / 新容器 / 磁盘重置后要恢复可运行环境；
- 出现 `ModuleNotFoundError`（尤其是 `xarray`、`h5netcdf`、`pytorch_lightning`、`icechunk`、`numcodecs`）；
- `torch.cuda.is_available()` 为 False，或 `--device cuda` 报错；
- 测试出现**大批** collection error 或 `Failed` 且错误都是缺模块。

**不适用**：
- 只是要升级某个依赖版本（那是依赖变更，需单独授权）；
- CI 环境（CI 有自己的安装步骤，见 `.github/workflows/ci.yml`）。

## 前置条件

- 知道目标 Python 版本：要求 `>=3.10`，本项目实测 3.12（见 `docs/rules/environment.md`）。
- 知道有无 GPU：有则装 CUDA 构建的 torch，无则 CPU 构建（见步骤 3）。
- 确认有网络访问 PyPI 与 `download.pytorch.org`。

## 步骤

1. **先读环境记录。** `docs/rules/environment.md` 记录了本项目的虚拟环境、依赖缺口与
   精确复现命令。**不要**只照着 `requirements*.txt` 装——三份清单的并集不是全部依赖。

2. **建虚拟环境。**
   ```bash
   cd /data/esw/UrbanPiDiT_R2
   uv venv .venv --python 3.12        # 或 python -m venv .venv
   ```

3. **装 torch（选对构建）。**
   ```bash
   # 有 GPU（本机是 2×RTX 3090，驱动支持 CUDA 13，实测 cu128 可用）：
   uv pip install --python .venv/bin/python "torch>=2.4" \
       --index-url https://download.pytorch.org/whl/cu128
   # 无 GPU：把 index-url 换成 .../whl/cpu
   ```

4. **装三份 requirements。**
   ```bash
   uv pip install --python .venv/bin/python \
       -r requirements.txt -r requirements-data.txt -r requirements-r7-data.txt
   ```

5. **补未声明的依赖（关键，最易漏）。** 三份 requirements 都**没有**声明这些，但代码/测试需要：
   ```bash
   uv pip install --python .venv/bin/python \
       "icechunk==2.2.2" "pcodec>=0.3" pip setuptools wheel
   ```
   - `icechunk`/`numcodecs` 是**函数内延迟导入**，不装不会立刻报错，跑到下载路径才炸。
   - `pip` 必须有：`tests/test_r7_installed_package.py` 会调 `python -m pip wheel`；
     `uv venv` 建的环境默认没有 pip。

6. **验证环境真的可用**（不要只看安装成功）。
   ```bash
   .venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   .venv/bin/python -m pytest -q          # 期望 622 passed, 3 skipped, 0 failed
   .venv/bin/python tools/check_conventions.py
   ```

## 检查点

- **步骤 3 后**：`torch.cuda.is_available()` 必须为 True（有 GPU 时）。
  若为 False，先查驱动与 CUDA 构建是否匹配，**不要**直接改用 CPU 版本绕过。
- **步骤 6 后**：全量测试的 skip 数应为 **3**（都是 clean checkout 缺真实数据 fixture，
  属预期）。跳过数变大说明有 fixture 或依赖没到位。
- **步骤 6 后**：若出现 `Failed` 而错误文本是 `ModuleNotFoundError`，回到步骤 5 补包，
  而不是去改测试。

## 常见失败

- **只装 requirements 就以为装完了**：会漏 `icechunk`/`pcodec`/`numcodecs`/`pip`。
  这三个包在 CI 里是**单条 workflow 临时 pin** 的，不在 requirements 中（见 Q-011）。
- **`uv venv` 后没有 pip** → `test_r7_installed_package` 失败，错误是
  `No module named pip`。装 `pip setuptools wheel` 即可。
- **装了 CPU 版 torch**：大量测试仍能过，但 `--device cuda` 路径与
  `test_bare_cuda_device_is_accepted_on_real_gpu` 会被 skip —— 这在**有 GPU 的机器上**
  是配置错误，不是正常状态。
- **`--device cuda` 报 `Expected a torch.device with a specified index`**：
  这是 torch≥2.8 的接口收紧，本项目已修复（`select_device` 在无索引时回退到
  `torch.cuda.current_device()`）。若在新环境又出现，说明改到了 `training/r7_experiment.py`
  的 `select_device`，不要去改测试。
- **忘记虚拟环境自动激活**：进入项目目录应看到 `(.venv)`。若没有，看
  `~/.bashrc` 的 `project venv auto-activation` 块；修好后在当前终端 `source ~/.bashrc`。

## 完成判据

- `.venv/bin/python` 能 import 全部依赖（含 `icechunk`、`numcodecs`、`pcodec`、`pytorch_lightning`）；
- 有 GPU 时 `torch.cuda.is_available()` 为 True，且真机做一次 matmul 成功；
- `pytest -q` 结果为 **622 passed, 3 skipped, 0 failed**（或与当时基线一致）；
- `tools/check_conventions.py` 阻断规则 0 违规。

## 明确不覆盖

- 不覆盖依赖**版本升级/合并**（`requirements*` 的收敛见 `docs/rules/OPEN_QUESTIONS.md` Q-011，
  属依赖面变更，需单独授权）；
- 不覆盖 CI 环境的搭建（CI 用 `ci.yml` 自己的安装步骤，且是 CPU-only）；
- 不覆盖数据集的获取（真实数据见 `real-data-acquisition`，且需要显式授权）；
- 不覆盖 GPU 驱动/CUDA toolkit 的安装（那是系统层，超出仓库范围）。
