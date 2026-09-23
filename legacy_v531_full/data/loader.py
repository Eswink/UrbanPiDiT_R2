"""数据加载与归一化（v3）

本模块提供：
- 训练集归一化统计（仅使用 TRAIN split，避免 val/test 泄露）
- BeijingWeatherDataset：支持 npz 与 npy 两种数据格式
- MetroWeatherDataModule：Lightning DataModule

v3 重点修复：
- 修复 npy 格式下动态变量通道堆叠顺序与 npz 不一致的问题。
  旧版(np y)：按“时间优先”堆叠，会导致模型 last_indices 取到错误通道；
  v3：统一为“变量优先”堆叠，确保与模型假设一致。
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

try:
    import pytorch_lightning as pl
except Exception:
    pl = None  # type: ignore

from torch.utils.data import DataLoader, Dataset

# 通道映射：当数据以 .npy([C,H,W]) 存储时，用于定位变量所在通道
CHANNEL_MAPPING = {
    "d2m": 0,
    "sp": 1,
    "t2m": 2,
    "tcc": 3,
    "tp": 4,
    "u10": 5,
    "v10": 6,
    "landcover": 7,
    "building_surface": 8,
    "buildings": 9,
    "building_volume": 10,
    "population": 11,
}

DEFAULT_CATEGORICAL_STATIC_VARS = ("landcover",)
_TIMESTAMP_HOUR_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})\.npy$")


def parse_hour_from_filename(path: str | Path) -> Optional[int]:
    """从 `YYYY_MM_DD_HH.npy` 文件名解析小时。"""

    match = _TIMESTAMP_HOUR_RE.match(Path(path).name)
    if match is None:
        return None
    hour = int(match.group(4))
    return hour if 0 <= hour <= 23 else None


def parse_static_schema(
    static_vars: Iterable[str],
    static_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """解析静态变量 schema，保留 static_vars 中的变量顺序。

    Args:
        static_vars: 配置中的静态变量顺序。
        static_schema: 可选 schema，支持 categorical/continuous 列表，也支持逐变量 type 描述。

    Returns:
        包含 categorical、continuous、categorical_cardinality 的标准化 schema。
    """

    ordered_vars = [str(v) for v in static_vars]
    schema = dict(static_schema or {})
    known = set(ordered_vars)

    categorical_cfg = schema.get("categorical", None)
    continuous_cfg = schema.get("continuous", None)

    if categorical_cfg is None:
        categorical = []
        for name in ordered_vars:
            spec = schema.get(name, None)
            if isinstance(spec, dict) and str(spec.get("type", "")).lower() in {"cat", "categorical", "category"}:
                categorical.append(name)
        if not categorical:
            categorical = [name for name in ordered_vars if name in DEFAULT_CATEGORICAL_STATIC_VARS]
    else:
        categorical = [str(v) for v in categorical_cfg if str(v) in known]

    if continuous_cfg is None:
        categorical_set = set(categorical)
        continuous = []
        for name in ordered_vars:
            spec = schema.get(name, None)
            if isinstance(spec, dict) and str(spec.get("type", "")).lower() in {"cont", "continuous", "numeric"}:
                continuous.append(name)
            elif name not in categorical_set:
                continuous.append(name)
    else:
        continuous = [str(v) for v in continuous_cfg if str(v) in known]

    categorical_set = set(categorical)
    continuous = [name for name in ordered_vars if name in set(continuous) and name not in categorical_set]
    categorical = [name for name in ordered_vars if name in categorical_set]

    raw_cardinality = schema.get("categorical_cardinality", schema.get("num_categories", schema.get("cardinality", {})))
    cardinality = dict(raw_cardinality) if isinstance(raw_cardinality, dict) else {}
    for name in categorical:
        spec = schema.get(name, None)
        if isinstance(spec, dict):
            if "num_classes" in spec:
                cardinality[name] = int(spec["num_classes"])
            elif "num_categories" in spec:
                cardinality[name] = int(spec["num_categories"])
            elif "cardinality" in spec:
                cardinality[name] = int(spec["cardinality"])
        cardinality.setdefault(name, 20 if name == "landcover" else 256)

    return {
        "categorical": categorical,
        "continuous": continuous,
        "categorical_cardinality": {name: int(cardinality[name]) for name in categorical if name in cardinality},
    }


def _merge_mean_var(count, mean, m2, batch_count, batch_mean, batch_var):
    """合并在线均值/方差统计（Welford-like 合并公式）。

    Args:
        count, mean, m2: 当前累计统计
        batch_count, batch_mean, batch_var: 新批次统计（batch_var 为总体方差 ddof=0）

    Returns:
        更新后的 (count, mean, m2)
    """

    if batch_count <= 0:
        return count, mean, m2
    if count == 0:
        return batch_count, batch_mean, batch_var * batch_count

    delta = batch_mean - mean
    tot = count + batch_count
    new_mean = mean + delta * (batch_count / tot)
    new_m2 = m2 + batch_var * batch_count + (delta**2) * (count * batch_count / tot)
    return tot, new_mean, new_m2


def compute_train_normalization_stats(
    data_root,
    dynamic_vars,
    static_vars,
    include_static=True,
    static_schema: Optional[Dict[str, Any]] = None,
):
    """仅使用 TRAIN split 计算每个变量的均值/标准差。

    重要：
    - 归一化统计只从 train/ 目录中计算，避免 val/test 数据泄露。

    Returns:
        dyn_mean, dyn_std, static_mean, static_std（均为 dict: name -> float）
    """

    data_root = Path(data_root)
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"train 目录不存在: {train_dir}")

    npz_files = sorted(train_dir.glob("*.npz"))
    npy_files = sorted(train_dir.glob("*.npy"))

    dyn = {v: {"count": 0, "mean": 0.0, "m2": 0.0} for v in dynamic_vars}
    parsed_schema = parse_static_schema(static_vars, static_schema)
    continuous_static_vars = parsed_schema["continuous"]
    stat = {s: {"count": 0, "mean": 0.0, "m2": 0.0} for s in continuous_static_vars} if include_static else {}

    def update(stats_dict, name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return
        bc = int(arr.size)
        bm = float(arr.mean())
        bv = float(arr.var(ddof=0))
        st = stats_dict[name]
        c, mu, m2 = _merge_mean_var(st["count"], st["mean"], st["m2"], bc, bm, bv)
        st["count"], st["mean"], st["m2"] = c, mu, m2

    if npz_files:
        for fp in npz_files:
            data = np.load(fp)
            for v in dynamic_vars:
                if v in data:
                    update(dyn, v, data[v])
            if include_static:
                for s in continuous_static_vars:
                    if s in data:
                        update(stat, s, data[s])
    elif npy_files:
        for fp in npy_files:
            arr = np.load(fp)
            for v in dynamic_vars:
                ch = CHANNEL_MAPPING.get(v, None)
                if ch is None:
                    continue
                update(dyn, v, arr[ch])
            if include_static:
                for s in continuous_static_vars:
                    chs = CHANNEL_MAPPING.get(s, None)
                    if chs is None:
                        continue
                    update(stat, s, arr[chs])
    else:
        raise FileNotFoundError(f"train 目录下未找到 .npz 或 .npy 文件: {train_dir}")

    dyn_mean = {k: float(v["mean"]) for k, v in dyn.items()}
    dyn_std = {}
    for k, v in dyn.items():
        dyn_std[k] = float(np.sqrt(v["m2"] / v["count"])) if v["count"] > 0 else 1.0

    static_mean = {k: float(v["mean"]) for k, v in stat.items()}
    static_std = {}
    for k, v in stat.items():
        static_std[k] = float(np.sqrt(v["m2"] / v["count"])) if v["count"] > 0 else 1.0

    return dyn_mean, dyn_std, static_mean, static_std


def compute_train_climatology_mean_field(data_root, dynamic_vars):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"train 目录不存在: {train_dir}")

    npz_files = sorted(train_dir.glob("*.npz"))
    npy_files = sorted(train_dir.glob("*.npy"))

    sums = {}
    counts = {}

    def ensure(name, shape):
        if name not in sums:
            sums[name] = np.zeros(shape, dtype=np.float64)
            counts[name] = 0

    if npz_files:
        for fp in npz_files:
            data = np.load(fp)
            for v in dynamic_vars:
                if v not in data:
                    continue
                arr = np.asarray(data[v], dtype=np.float64)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                if arr.ndim != 3:
                    continue
                ensure(v, arr.shape[-2:])
                sums[v] += arr.sum(axis=0)
                counts[v] += int(arr.shape[0])
    elif npy_files:
        for fp in npy_files:
            arr = np.load(fp)
            for v in dynamic_vars:
                ch = CHANNEL_MAPPING.get(v, None)
                if ch is None:
                    continue
                x = np.asarray(arr[ch], dtype=np.float64)
                if x.ndim != 2:
                    continue
                ensure(v, x.shape)
                sums[v] += x
                counts[v] += 1
    else:
        raise FileNotFoundError(f"train 目录下未找到 .npz 或 .npy 文件: {train_dir}")

    out = {}
    for v in dynamic_vars:
        if v not in sums or counts.get(v, 0) <= 0:
            continue
        out[v] = (sums[v] / float(counts[v])).astype(np.float32)
    return out


class BeijingWeatherDataset(Dataset):
    """北京城市微尺度气象数据集。

    支持两种格式：
    - npz：一个文件包含一个时间序列（多个时刻）
    - npy：一个文件对应一个时刻（建议按时间命名）

    输出字典：
        {
          "x_ctx": [ctx_C, H, W],  # 上下文(历史帧 + 静态)
          "x0":    [C, H, W],      # 目标(未来 delta_t)
          "meta":  {lat/lon},
          "norm":  {mean/std},     # 每个动态变量的归一化参数
        }
    """

    def __init__(
        self,
        split_dir,
        k,
        delta_t,
        lead_times,
        dynamic_vars,
        static_vars,
        include_static,
        broadcast_static,
        normalize_root,
        static_perturb=None,
        static_schema: Optional[Dict[str, Any]] = None,
        expose_hour: bool = False,
        dynamic_perturb=None,
    ):
        self.split_dir = Path(split_dir)
        self.k = int(k)
        # -------------------------
        # 预测提前期（lead time）配置
        # -------------------------
        # 说明：
        # - delta_t：单一提前期（旧版/兼容模式），例如 1 表示预测未来 1 个时间步（6h）
        # - lead_times：多提前期列表（新功能），例如 [1,2,3,4] 对应 6/12/18/24h。
        #   当 lead_times 不为 None 时，Dataset 会返回 y=[L,C,H,W] 的多提前期真值。
        self.delta_t = int(delta_t)

        # lead_times 允许为 None / list / tuple
        self.lead_times = None
        if lead_times is not None:
            # 去重但尽量保留用户写入的顺序
            lt_list = [int(x) for x in list(lead_times)]
            seen = set()
            ordered = []
            for x in lt_list:
                if x not in seen:
                    ordered.append(x)
                    seen.add(x)
            if len(ordered) > 0:
                self.lead_times = ordered

        # 构建索引时需要保证“最大提前期”的未来帧存在
        self.max_delta_t = max(self.lead_times) if self.lead_times is not None else self.delta_t
        self.dynamic_vars = list(dynamic_vars)
        self.static_vars = list(static_vars)
        self.static_schema = parse_static_schema(self.static_vars, static_schema)
        self.continuous_static_vars = list(self.static_schema["continuous"])
        self.categorical_static_vars = list(self.static_schema["categorical"])
        self.categorical_cardinality = dict(self.static_schema["categorical_cardinality"])
        self.include_static = bool(include_static)
        self.broadcast_static = bool(broadcast_static)
        self.static_perturb = dict(static_perturb or {})
        self.dynamic_perturb = dict(dynamic_perturb or {})
        self.dynamic_perturb_mode = str(self.dynamic_perturb.get("mode", "none")).lower()
        self.static_perturb_mode = str(self.static_perturb.get("mode", "none")).lower()
        self.static_perturb_seed = int(self.static_perturb.get("seed", 42))
        self.static_perturb_deterministic = bool(self.static_perturb.get("deterministic", True))
        raw_modes = self.static_perturb.get("categorical_modes", {})
        self.static_perturb_categorical_modes = {
            str(name): int(value) for name, value in dict(raw_modes or {}).items()
        }
        self._static_perturb_warned = set()
        self.expose_hour = bool(expose_hour)

        self.files = sorted(list(self.split_dir.glob("*.npz")))
        self.format = None

        # 可选：提供 lat.npy / lon.npy
        self.lat = np.load(self.split_dir.parent / "lat.npy") if (self.split_dir.parent / "lat.npy").exists() else None
        self.lon = np.load(self.split_dir.parent / "lon.npy") if (self.split_dir.parent / "lon.npy").exists() else None

        # -------------------------
        # 读取归一化文件（优先 *_train.npz）
        # -------------------------
        mean_path = Path(normalize_root) / "normalize_mean_train.npz"
        std_path = Path(normalize_root) / "normalize_std_train.npz"
        legacy_mean_path = Path(normalize_root) / "normalize_mean.npz"
        legacy_std_path = Path(normalize_root) / "normalize_std.npz"

        self.mean = None
        self.std = None

        try:
            # 若只存在 legacy 文件，则尝试自动生成 train-only 统计
            if (not mean_path.exists() or not std_path.exists()) and (legacy_mean_path.exists() or legacy_std_path.exists()):
                print("[BeijingWeatherDataset] [Warning] 检测到 legacy normalize_*.npz，正在尝试从 TRAIN split 重新生成 *_train.npz（避免泄露）...")
                try:
                    dyn_mean, dyn_std, stat_mean, stat_std = compute_train_normalization_stats(
                        Path(normalize_root),
                        self.dynamic_vars,
                        self.static_vars,
                        include_static=self.include_static,
                        static_schema=self.static_schema,
                    )
                    np.savez(mean_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_mean.items()})
                    np.savez(std_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_std.items()})
                    if self.include_static:
                        np.savez(Path(normalize_root) / "normalize_static_mean_train.npz", **{k: np.array([v], dtype=np.float32) for k, v in stat_mean.items()})
                        np.savez(Path(normalize_root) / "normalize_static_std_train.npz", **{k: np.array([v], dtype=np.float32) for k, v in stat_std.items()})
                    print("[BeijingWeatherDataset] 已保存 train-only 归一化统计到 *_train.npz")
                except Exception as _e:
                    print(f"[BeijingWeatherDataset] [Warning] 自动生成 train-only 统计失败: {_e}，将回退使用 legacy 文件（可能泄露）")

            if mean_path.exists():
                nm = np.load(mean_path)
                self.mean = {k: nm[k] for k in nm.files}
            elif legacy_mean_path.exists():
                print("[BeijingWeatherDataset] [Warning] 使用 legacy normalize_mean.npz（可能包含 val/test 泄露）")
                nm = np.load(legacy_mean_path)
                self.mean = {k: nm[k] for k in nm.files}

            if std_path.exists():
                ns = np.load(std_path)
                self.std = {k: ns[k] for k in ns.files}
            elif legacy_std_path.exists():
                print("[BeijingWeatherDataset] [Warning] 使用 legacy normalize_std.npz（可能包含 val/test 泄露）")
                ns = np.load(legacy_std_path)
                self.std = {k: ns[k] for k in ns.files}

        except Exception as e:
            print(f"[BeijingWeatherDataset] 归一化统计读取失败: {e}。将使用原始数据训练/评估。")
            self.mean = None
            self.std = None

        # 静态变量归一化
        static_mean_path = Path(normalize_root) / "normalize_static_mean_train.npz"
        static_std_path = Path(normalize_root) / "normalize_static_std_train.npz"
        legacy_static_mean_path = Path(normalize_root) / "normalize_static_mean.npz"
        legacy_static_std_path = Path(normalize_root) / "normalize_static_std.npz"

        self.static_mean = None
        self.static_std = None
        try:
            if static_mean_path.exists():
                sm = np.load(static_mean_path)
                self.static_mean = {k: sm[k] for k in sm.files}
            elif legacy_static_mean_path.exists():
                print("[BeijingWeatherDataset] [Warning] 使用 legacy normalize_static_mean.npz（可能泄露）")
                sm = np.load(legacy_static_mean_path)
                self.static_mean = {k: sm[k] for k in sm.files}

            if static_std_path.exists():
                ss = np.load(static_std_path)
                self.static_std = {k: ss[k] for k in ss.files}
            elif legacy_static_std_path.exists():
                print("[BeijingWeatherDataset] [Warning] 使用 legacy normalize_static_std.npz（可能泄露）")
                ss = np.load(legacy_static_std_path)
                self.static_std = {k: ss[k] for k in ss.files}

        except Exception as e:
            print(f"[BeijingWeatherDataset] 静态特征归一化读取失败: {e}。将使用原始静态特征。")
            self.static_mean = None
            self.static_std = None

        self.clim = None
        clim_path = Path(normalize_root) / "normalize_clim_train.npz"
        try:
            if clim_path.exists():
                clim = np.load(clim_path)
                self.clim = {k: clim[k] for k in clim.files}
        except Exception as e:
            print(f"[BeijingWeatherDataset] climatology 读取失败: {e}。将跳过 ACC climatology。")
            self.clim = None

        # -------------------------
        # 构建索引
        # -------------------------
        self.index = []

        if self.files:
            # ===== npz 模式 =====
            self.format = "npz"
            for fp in self.files:
                data = np.load(fp)
                n = data[self.dynamic_vars[0]].shape[0]
                if n < self.k + self.max_delta_t:
                    continue
                for i in range(self.k - 1, n - self.max_delta_t):
                    self.index.append(("npz", fp, i))
        else:
            # ===== npy 模式 =====
            npy_files = sorted(list(self.split_dir.glob("*.npy")))
            groups = {}
            for f in npy_files:
                m = re.match(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})\.npy$", f.name)
                if not m:
                    continue
                key = f"{m.group(1)}_{m.group(2)}"  # 按月份分组
                groups.setdefault(key, []).append(f)

            for _, flist in groups.items():
                flist = sorted(flist)
                n = len(flist)
                if n < self.k + self.max_delta_t:
                    continue
                for i in range(self.k - 1, n - self.max_delta_t):
                    self.index.append(("npy", flist, i))

            self.format = "npy" if len(self.index) > 0 else "none"

        if self.format == "npz":
            print(f"[BeijingWeatherDataset] 使用 npz 格式: 文件数={len(self.files)} 样本数={len(self.index)} 路径={self.split_dir}")
        elif self.format == "npy":
            print(f"[BeijingWeatherDataset] 使用 npy 格式: 文件数={len(list(self.split_dir.glob('*.npy')))} 样本数={len(self.index)} 路径={self.split_dir}")
        else:
            print(f"[BeijingWeatherDataset] 未找到可用数据文件: 路径={self.split_dir}")

        # 额外提示：当前提前期设置
        if self.lead_times is None:
            print(f"[BeijingWeatherDataset] 单提前期模式: delta_t={self.delta_t}")
        else:
            print(f"[BeijingWeatherDataset] 多提前期模式: lead_times={self.lead_times} (max_delta_t={self.max_delta_t})")

    def __len__(self):
        return len(self.index)

    def _static_rng(self, idx: int) -> np.random.RandomState:
        if self.static_perturb_deterministic:
            return np.random.RandomState(self.static_perturb_seed + int(idx))
        return np.random.RandomState(None)

    def _fill_static_channel_zero(self, out: np.ndarray, channel: int, vname: str) -> None:
        m = 0.0
        s = 1.0
        if self.static_mean is not None and vname in self.static_mean:
            m = float(self.static_mean[vname][0])
        if self.static_std is not None and vname in self.static_std:
            s = float(self.static_std[vname][0])
        out[channel, :, :] = (-m / s) if (s != 0.0) else 0.0

    def _fill_static_channel_mean(self, out: np.ndarray, channel: int, vname: str) -> None:
        if vname in self.categorical_static_vars:
            out[channel, :, :] = float(self.static_perturb_categorical_modes.get(vname, 0))
        else:
            out[channel, :, :] = 0.0

    def _apply_static_perturb(self, S: np.ndarray, idx: int) -> np.ndarray:
        mode = self.static_perturb_mode
        if (not self.include_static) or S.size == 0:
            return S
        if mode in ("", "none", "off", "disable", "disabled"):
            return S

        if mode in ("shuffle_hw", "spatial_shuffle", "shuffle", "spatial", "shuffle_spatial"):
            ds, H, W = S.shape
            rng = self._static_rng(idx)
            perm = rng.permutation(H * W)
            flat = S.transpose(1, 2, 0).reshape(H * W, ds)
            flat = flat[perm]
            return flat.reshape(H, W, ds).transpose(2, 0, 1)

        if mode in ("shuffle_batch", "batch_shuffle"):
            if "shuffle_source" in self.static_perturb:
                source = np.asarray(self.static_perturb["shuffle_source"], dtype=np.float32)
                if source.shape == S.shape:
                    return source.copy()
            if "shuffle_sources" in self.static_perturb:
                sources = np.asarray(self.static_perturb["shuffle_sources"], dtype=np.float32)
                if sources.ndim == 4 and tuple(sources.shape[1:]) == tuple(S.shape):
                    rng = self._static_rng(idx)
                    return sources[int(rng.randint(0, sources.shape[0]))].copy()
            if "shuffle_batch" not in self._static_perturb_warned:
                print("[BeijingWeatherDataset] [Warning] shuffle_batch 缺少外部 batch/source，回退为 shuffle_spatial")
                self._static_perturb_warned.add("shuffle_batch")
            old_mode = self.static_perturb_mode
            self.static_perturb_mode = "shuffle_spatial"
            try:
                return self._apply_static_perturb(S, idx)
            finally:
                self.static_perturb_mode = old_mode

        if mode in ("fill_mean", "mean_out", "mean"):
            out = np.zeros_like(S, dtype=np.float32)
            for i, vname in enumerate(self.static_vars):
                self._fill_static_channel_mean(out, i, vname)
            return out

        if mode in ("fill_zero", "zero_out", "zero"):
            out = np.empty_like(S, dtype=np.float32)
            for i, vname in enumerate(self.static_vars):
                self._fill_static_channel_zero(out, i, vname)
            return out

        if mode in ("permute_landcover", "landcover_permute"):
            out = S.copy().astype(np.float32)
            if "landcover" in self.static_vars:
                channel = self.static_vars.index("landcover")
                cardinality = int(self.categorical_cardinality.get("landcover", 20))
                out[channel] = (out[channel].astype(np.int64) + 1) % max(cardinality, 1)
            return out

        if mode in ("landcover_only", "categorical_only"):
            out = np.zeros_like(S, dtype=np.float32)
            for i, vname in enumerate(self.static_vars):
                if vname in self.categorical_static_vars:
                    out[i] = S[i]
                else:
                    self._fill_static_channel_mean(out, i, vname)
            return out

        if mode in ("continuous_only", "numeric_only"):
            out = np.zeros_like(S, dtype=np.float32)
            for i, vname in enumerate(self.static_vars):
                if vname in self.continuous_static_vars:
                    out[i] = S[i]
                else:
                    self._fill_static_channel_mean(out, i, vname)
            return out

        if mode.startswith("remove_"):
            remove_name = mode[len("remove_") :]
            out = S.copy().astype(np.float32)
            for i, vname in enumerate(self.static_vars):
                if vname == remove_name:
                    self._fill_static_channel_mean(out, i, vname)
            return out

        if mode in ("wrong_city_static", "wrong_city"):
            source_path = self.static_perturb.get("source_path", None)
            if source_path:
                src = Path(str(source_path))
                if src.exists():
                    arr = np.load(src)
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        key = str(self.static_perturb.get("source_key", "static"))
                        if key in arr:
                            value = np.asarray(arr[key], dtype=np.float32)
                            if value.shape == S.shape:
                                return value.copy()
                    else:
                        value = np.asarray(arr, dtype=np.float32)
                        if value.shape == S.shape:
                            return value.copy()
            if "wrong_city_static" not in self._static_perturb_warned:
                print("[BeijingWeatherDataset] [Warning] wrong_city_static 未提供可用静态源，跳过该扰动")
                self._static_perturb_warned.add("wrong_city_static")
            return S

        if mode in ("rot90", "rotate90", "rotate"):
            k = int(self.static_perturb.get("k", 1)) % 4
            if k == 0:
                return S
            return np.rot90(S, k=k, axes=(-2, -1)).copy()

        if mode in ("flip_ud", "flip_up_down", "flip_vertical"):
            return np.flip(S, axis=-2).copy()

        if mode in ("flip_lr", "flip_left_right", "flip_horizontal"):
            return np.flip(S, axis=-1).copy()

        raise ValueError(f"Unknown static_perturb.mode: {self.static_perturb_mode}")

    def _apply_dynamic_perturb(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mode = self.dynamic_perturb_mode
        if mode in ("", "none", "off", "disable", "disabled"):
            return fields
        if mode in ("rotate_wind_90", "rot90_wind", "wind_rot90"):
            out = dict(fields)
            if "u10" in out and "v10" in out:
                u = np.asarray(out["u10"], dtype=np.float32)
                v = np.asarray(out["v10"], dtype=np.float32)
                direction = str(self.dynamic_perturb.get("direction", "ccw")).lower()
                if direction in {"cw", "clockwise"}:
                    out["u10"] = v.copy()
                    out["v10"] = -u.copy()
                else:
                    out["u10"] = -v.copy()
                    out["v10"] = u.copy()
            return out
        raise ValueError(f"Unknown dynamic_perturb.mode: {self.dynamic_perturb_mode}")

    def _apply_context_dynamic_perturb(self, x_ctx: np.ndarray) -> np.ndarray:
        mode = self.dynamic_perturb_mode
        if mode in ("", "none", "off", "disable", "disabled"):
            return x_ctx
        if mode not in ("rotate_wind_90", "rot90_wind", "wind_rot90"):
            raise ValueError(f"Unknown dynamic_perturb.mode: {self.dynamic_perturb_mode}")
        if "u10" not in self.dynamic_vars or "v10" not in self.dynamic_vars:
            return x_ctx

        out = x_ctx.copy().astype(np.float32)
        u_idx = self.dynamic_vars.index("u10")
        v_idx = self.dynamic_vars.index("v10")
        direction = str(self.dynamic_perturb.get("direction", "ccw")).lower()
        for step in range(self.k):
            u_channel = u_idx * self.k + step
            v_channel = v_idx * self.k + step
            if max(u_channel, v_channel) >= out.shape[0]:
                continue
            u = out[u_channel].copy()
            v = out[v_channel].copy()
            if direction in {"cw", "clockwise"}:
                out[u_channel] = v
                out[v_channel] = -u
            else:
                out[u_channel] = -v
                out[v_channel] = u
        return out

    def _split_static_maps(self, S: np.ndarray, static_names: Optional[Iterable[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        if S.size == 0:
            return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0, 0), dtype=np.int64)

        names = list(static_names) if static_names is not None else list(self.static_vars)
        name_to_idx = {name: i for i, name in enumerate(names)}
        cont = [S[name_to_idx[name]].astype(np.float32, copy=False) for name in self.continuous_static_vars if name in name_to_idx]
        cat = [S[name_to_idx[name]].astype(np.int64, copy=False) for name in self.categorical_static_vars if name in name_to_idx]

        H, W = S.shape[-2], S.shape[-1]
        static_cont = np.stack(cont, axis=0).astype(np.float32) if cont else np.empty((0, H, W), dtype=np.float32)
        static_cat = np.stack(cat, axis=0).astype(np.int64) if cat else np.empty((0, H, W), dtype=np.int64)
        return static_cont, static_cat

    def _hour_tensor(self, value: Optional[int]) -> torch.Tensor:
        hour = -1 if value is None else int(value)
        return torch.tensor(hour, dtype=torch.long)

    def _norm(self, vname, arr, is_static=False):
        """按变量名执行归一化。"""

        if is_static:
            if vname in getattr(self, "categorical_static_vars", []):
                return arr
            if (
                self.static_mean is not None
                and vname in self.static_mean
                and self.static_std is not None
                and vname in self.static_std
            ):
                m = float(self.static_mean[vname][0])
                s = float(self.static_std[vname][0])
                if s != 0:
                    arr = (arr - m) / s
            return arr

        if self.mean is not None and vname in self.mean and self.std is not None and vname in self.std:
            m = float(self.mean[vname][0])
            s = float(self.std[vname][0])
            if s != 0:
                arr = (arr - m) / s
        return arr

    def __getitem__(self, idx):
        mode, payload, t_idx = self.index[idx]
        hour_of_day = None

        if mode == "npz":
            data = np.load(payload)
            H = data[self.dynamic_vars[0]].shape[-2]
            W = data[self.dynamic_vars[0]].shape[-1]

            # 1) 上下文：动态变量按“变量优先”堆叠（与模型假设一致）
            ctx = []
            for v in self.dynamic_vars:
                seq = data[v]
                for j in range(self.k):
                    x = seq[t_idx - (self.k - 1) + j]
                    x = np.squeeze(x)
                    x = self._norm(v, x)
                    ctx.append(x)

            # 2) 静态变量：只追加一次
            static_raw = np.empty((0, H, W), dtype=np.float32)
            static_names = []
            if self.include_static:
                stat_maps = []
                for s in self.static_vars:
                    if s in data:
                        x = data[s][t_idx] if self.broadcast_static else data[s][0]
                        x = np.squeeze(x)
                        x = self._norm(s, x, is_static=True)
                        stat_maps.append(x)
                        static_names.append(s)
                if len(stat_maps) > 0:
                    S = np.stack(stat_maps, axis=0).astype(np.float32)
                    S = self._apply_static_perturb(S, idx)
                    static_raw = S
                    for i in range(S.shape[0]):
                        ctx.append(S[i])

            x_ctx = np.stack(ctx, axis=0).astype(np.float32)
            x_ctx = self._apply_context_dynamic_perturb(x_ctx)

            # 3) 目标：未来帧（支持单提前期/多提前期）
            if self.lead_times is None:
                # ===== 单提前期：返回 [C,H,W] =====
                y = []
                for v in self.dynamic_vars:
                    seq = data[v]
                    x0 = seq[t_idx + self.delta_t]
                    x0 = np.squeeze(x0)
                    x0 = self._norm(v, x0)
                    y.append(x0)
                y = np.stack(y, axis=0).astype(np.float32)
            else:
                # ===== 多提前期：返回 [L,C,H,W] =====
                y_futures = []
                for lt in self.lead_times:
                    y_lt = []
                    for v in self.dynamic_vars:
                        seq = data[v]
                        x0 = seq[t_idx + int(lt)]
                        x0 = np.squeeze(x0)
                        x0 = self._norm(v, x0)
                        y_lt.append(x0)
                    y_futures.append(np.stack(y_lt, axis=0))
                y_futures = np.stack(y_futures, axis=0).astype(np.float32)

        else:
            # ===== npy 模式 =====
            flist = payload
            arr0 = np.load(flist[0])
            H, W = arr0.shape[-2], arr0.shape[-1]
            if self.expose_hour:
                hour_of_day = parse_hour_from_filename(flist[t_idx])

            # 1) 读取过去 k 个时刻的数组（一次性读取，避免重复 IO）
            arr_list = []
            for j in range(self.k):
                fctx = flist[t_idx - (self.k - 1) + j]
                arr_list.append(np.load(fctx))

            # 2) 动态变量：按“变量优先”堆叠（v3 修复点）
            # 通道顺序: [var0_t0, var0_t1, ..., var0_t{k-1}, var1_t0, ..., varC-1_t{k-1}]
            ctx = []
            for v in self.dynamic_vars:
                ch = CHANNEL_MAPPING[v]
                for j in range(self.k):
                    x = arr_list[j][ch]
                    x = np.squeeze(x)
                    x = self._norm(v, x)
                    ctx.append(x)

            # 3) 静态变量：只追加一次
            static_raw = np.empty((0, H, W), dtype=np.float32)
            static_names = []
            if self.include_static:
                f_static = flist[t_idx] if self.broadcast_static else flist[0]
                arr_s = np.load(f_static)
                stat_maps = []
                for s in self.static_vars:
                    chs = CHANNEL_MAPPING.get(s, None)
                    if chs is None:
                        continue
                    xs = arr_s[chs]
                    xs = np.squeeze(xs)
                    xs = self._norm(s, xs, is_static=True)
                    stat_maps.append(xs)
                    static_names.append(s)
                if len(stat_maps) > 0:
                    S = np.stack(stat_maps, axis=0).astype(np.float32)
                    S = self._apply_static_perturb(S, idx)
                    static_raw = S
                    for i in range(S.shape[0]):
                        ctx.append(S[i])

            x_ctx = np.stack(ctx, axis=0).astype(np.float32)
            x_ctx = self._apply_context_dynamic_perturb(x_ctx)

            # 4) 目标：未来帧（支持单提前期/多提前期）
            if self.lead_times is None:
                fy = flist[t_idx + self.delta_t]
                arry = np.load(fy)
                y = []
                for v in self.dynamic_vars:
                    ch = CHANNEL_MAPPING[v]
                    x0 = arry[ch]
                    x0 = np.squeeze(x0)
                    x0 = self._norm(v, x0)
                    y.append(x0)
                y = np.stack(y, axis=0).astype(np.float32)
            else:
                y_futures = []
                for lt in self.lead_times:
                    fy = flist[t_idx + int(lt)]
                    arry = np.load(fy)
                    y_lt = []
                    for v in self.dynamic_vars:
                        ch = CHANNEL_MAPPING[v]
                        x0 = arry[ch]
                        x0 = np.squeeze(x0)
                        x0 = self._norm(v, x0)
                        y_lt.append(x0)
                    y_futures.append(np.stack(y_lt, axis=0))
                y_futures = np.stack(y_futures, axis=0).astype(np.float32)

        meta = {
            "lat": self.lat if self.lat is not None else np.linspace(41.0, 39.4, H, dtype=np.float32),
            "lon": self.lon if self.lon is not None else np.linspace(115.2, 117.0, W, dtype=np.float32),
        }

        # 构造每变量的归一化参数（缺省 mean=0, std=1）
        means = []
        stds = []
        for v in self.dynamic_vars:
            if self.mean is not None and v in self.mean and self.std is not None and v in self.std:
                mv = float(self.mean[v][0])
                sv = float(self.std[v][0])
            else:
                mv = 0.0
                sv = 1.0
            means.append(mv)
            stds.append(sv)

        norm = {
            "mean": torch.tensor(means, dtype=torch.float32).view(-1, 1, 1),
            "std": torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1),
        }

        static_cont, static_cat = self._split_static_maps(static_raw, static_names)

        # -------------------------
        # 输出结构
        # -------------------------
        sample = {
            "x_ctx": torch.from_numpy(x_ctx),
            "static_raw": torch.from_numpy(static_raw.astype(np.float32, copy=False)),
            "static_cont": torch.from_numpy(static_cont),
            "static_cat": torch.from_numpy(static_cat),
            "meta": meta,
            "norm": norm,
        }
        if self.expose_hour:
            sample["hour_of_day"] = self._hour_tensor(hour_of_day)
            sample["diurnal_available"] = torch.tensor(hour_of_day is not None, dtype=torch.bool)
        if self.clim is not None:
            clim_list = []
            for v in self.dynamic_vars:
                x = self.clim.get(v, None)
                if x is None:
                    clim_list.append(np.zeros((H, W), dtype=np.float32))
                else:
                    x = np.asarray(x, dtype=np.float32)
                    x = np.squeeze(x)
                    if x.shape != (H, W):
                        x = np.zeros((H, W), dtype=np.float32)
                    clim_list.append(x)
            sample["clim"] = torch.from_numpy(np.stack(clim_list, axis=0))

        if self.lead_times is None:
            # 单提前期输出
            sample["x0"] = torch.from_numpy(y)
            sample["lead_times"] = torch.tensor([self.delta_t], dtype=torch.long)
        else:
            # 多提前期输出：y=[L,C,H,W]
            sample["y"] = torch.from_numpy(y_futures)
            sample["lead_times"] = torch.tensor(self.lead_times, dtype=torch.long)

            # 兼容旧字段：x0 默认取 delta_t 对应的那一帧；若 delta_t 不在列表中，则取第一帧。
            try:
                idx0 = self.lead_times.index(self.delta_t)
            except Exception:
                idx0 = 0
            sample["x0"] = sample["y"][idx0]

        return sample


# Lightning DataModule：若未安装 pl，则仅提供 Dataset
if pl is None:

    class MetroWeatherDataModule:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装 pytorch_lightning，无法使用 MetroWeatherDataModule")

else:

    class MetroWeatherDataModule(pl.LightningDataModule):
        """数据模块：封装 train/val/test DataLoader。"""

        def __init__(
            self,
            data_root,
            k,
            delta_t,
            lead_times,
            batch_size,
            num_workers,
            dynamic_vars,
            static_vars,
            include_static,
            broadcast_static,
            static_perturb=None,
            static_schema: Optional[Dict[str, Any]] = None,
            expose_hour: bool = False,
            dynamic_perturb=None,
        ):
            super().__init__()
            self.data_root = Path(data_root)
            self.k = int(k)
            self.delta_t = int(delta_t)
            self.lead_times = lead_times
            self.batch_size = int(batch_size)
            self.num_workers = int(num_workers)
            self.dynamic_vars = list(dynamic_vars)
            self.static_vars = list(static_vars)
            self.include_static = bool(include_static)
            self.broadcast_static = bool(broadcast_static)
            self.static_perturb = dict(static_perturb or {})
            self.dynamic_perturb = dict(dynamic_perturb or {})
            self.static_schema = parse_static_schema(self.static_vars, static_schema)
            self.expose_hour = bool(expose_hour)

        def prepare_data(self):
            """确保 train-only 归一化文件存在。"""

            mean_path = self.data_root / "normalize_mean_train.npz"
            std_path = self.data_root / "normalize_std_train.npz"
            smean_path = self.data_root / "normalize_static_mean_train.npz"
            sstd_path = self.data_root / "normalize_static_std_train.npz"
            clim_path = self.data_root / "normalize_clim_train.npz"

            need_norm = not (
                mean_path.exists()
                and std_path.exists()
                and (not self.include_static or (smean_path.exists() and sstd_path.exists()))
            )
            need_clim = not clim_path.exists()

            if (not need_norm) and (not need_clim):
                return

            try:
                if need_norm:
                    dyn_mean, dyn_std, stat_mean, stat_std = compute_train_normalization_stats(
                        self.data_root,
                        self.dynamic_vars,
                        self.static_vars,
                        include_static=self.include_static,
                        static_schema=self.static_schema,
                    )
                    np.savez(mean_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_mean.items()})
                    np.savez(std_path, **{k: np.array([v], dtype=np.float32) for k, v in dyn_std.items()})
                    if self.include_static:
                        np.savez(smean_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_mean.items()})
                        np.savez(sstd_path, **{k: np.array([v], dtype=np.float32) for k, v in stat_std.items()})
                    print("[MetroWeatherDataModule] 已保存 train-only 归一化统计到 *_train.npz")

                if need_clim:
                    clim = compute_train_climatology_mean_field(self.data_root, self.dynamic_vars)
                    if len(clim) > 0:
                        np.savez(clim_path, **{k: np.asarray(v, dtype=np.float32) for k, v in clim.items()})
                        print("[MetroWeatherDataModule] 已保存 train-only climatology 到 normalize_clim_train.npz")
            except Exception as e:
                print(f"[MetroWeatherDataModule] 计算 train-only 归一化统计失败: {e}")

        def setup(self, stage=None):
            self.train_ds = BeijingWeatherDataset(
                self.data_root / "train",
                self.k,
                self.delta_t,
                self.lead_times,
                self.dynamic_vars,
                self.static_vars,
                self.include_static,
                self.broadcast_static,
                self.data_root,
                self.static_perturb,
                self.static_schema,
                self.expose_hour,
                self.dynamic_perturb,
            )
            self.val_ds = BeijingWeatherDataset(
                self.data_root / "val",
                self.k,
                self.delta_t,
                self.lead_times,
                self.dynamic_vars,
                self.static_vars,
                self.include_static,
                self.broadcast_static,
                self.data_root,
                self.static_perturb,
                self.static_schema,
                self.expose_hour,
                self.dynamic_perturb,
            )
            self.test_ds = BeijingWeatherDataset(
                self.data_root / "test",
                self.k,
                self.delta_t,
                self.lead_times,
                self.dynamic_vars,
                self.static_vars,
                self.include_static,
                self.broadcast_static,
                self.data_root,
                self.static_perturb,
                self.static_schema,
                self.expose_hour,
                self.dynamic_perturb,
            )

        def train_dataloader(self):
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
