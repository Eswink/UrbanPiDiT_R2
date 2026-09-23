"""可行性与过程代理损失模块。

该模块把预测约束拆成三层代理：
- feasibility：变量取值范围与基本可行性；
- structure：频谱、空间梯度与风场散度结构；
- process_consistency：相对湿度、形态阻力与日周期热响应一致性。

`PhysicalConsistencyLoss` 保留旧入口和旧日志键，新增日志键只作为更清晰的
审稿解释层，不改变默认训练目标语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.fft
import torch.nn.functional as F


DYNAMIC_VAR_ORDER = ("d2m", "sp", "t2m", "tcc", "tp", "u10", "v10")


@dataclass(frozen=True)
class PhysicalConsistencyConfig:
    """物理一致性损失配置。

    Args:
        use_physics_loss: 是否启用旧版 feasibility + wind-divergence 物理项。
        use_feasibility_loss: 是否启用可行性代理项。
        use_structure_loss: 是否启用结构代理诊断。
        use_process_consistency: 是否启用新增过程一致性代理诊断。
        use_fft_loss: 是否启用频谱结构项。
        use_gradient_loss: 是否启用梯度结构项。
        lambda_phys: 旧版物理空间项外部组合权重。
        lambda_fft: 频谱项外部组合权重。
        lambda_grad: 梯度项外部组合权重。
        lambda_process_rh: 相对湿度过程项内部权重。
        lambda_process_drag: 形态阻力过程项内部权重。
        lambda_process_diurnal: 日周期热响应过程项内部权重。
        w_tp: 降水非负约束内部权重。
        w_tcc: 云量范围约束内部权重。
        w_dew: 露点不高于气温约束内部权重。
        w_sp: 气压非负约束内部权重。
        w_div: 风场散度约束内部权重。
    """

    use_physics_loss: bool = True
    use_feasibility_loss: bool = True
    use_structure_loss: bool = True
    use_process_consistency: bool = False
    use_fft_loss: bool = True
    use_gradient_loss: bool = True
    lambda_phys: float = 0.1
    lambda_fft: float = 0.0
    lambda_grad: float = 0.0
    lambda_process_rh: float = 0.0
    lambda_process_drag: float = 0.0
    lambda_process_diurnal: float = 0.0
    lambda_process_proxy: float = 0.0
    lambda_proxy_roughness_wind: float = 0.0
    lambda_proxy_diurnal_building: float = 0.0
    lambda_proxy_impervious_d2m: float = 0.0
    lambda_proxy_wind_tcc: float = 0.0
    w_tp: float = 1.0
    w_tcc: float = 0.5
    w_dew: float = 0.5
    w_sp: float = 0.1
    w_div: float = 0.1

    @classmethod
    def from_mapping(cls, cfg: Optional[Mapping[str, Any]]) -> "PhysicalConsistencyConfig":
        """从 YAML 风格配置构造损失配置。

        Args:
            cfg: physics 配置段或统一 builder 输出。

        Returns:
            PhysicalConsistencyConfig 实例。
        """

        data = dict(cfg or {})
        process_cfg = dict(data.get("process", {}) or {}) if isinstance(data.get("process", {}), Mapping) else {}
        return cls(
            use_physics_loss=bool(data.get("use_physics_loss", True)),
            use_feasibility_loss=bool(data.get("use_feasibility_loss", data.get("use_physics_loss", True))),
            use_structure_loss=bool(data.get("use_structure_loss", True)),
            use_process_consistency=bool(data.get("use_process_consistency", process_cfg.get("enabled", False))),
            use_fft_loss=bool(data.get("use_fft_loss", True)),
            use_gradient_loss=bool(data.get("use_gradient_loss", True)),
            lambda_phys=float(data.get("lambda", 0.1)),
            lambda_fft=float(data.get("lambda_fft", 0.0)),
            lambda_grad=float(data.get("lambda_grad", 0.0)),
            lambda_process_rh=float(data.get("lambda_process_rh", process_cfg.get("lambda_rh", 0.0))),
            lambda_process_drag=float(data.get("lambda_process_drag", process_cfg.get("lambda_drag", 0.0))),
            lambda_process_diurnal=float(data.get("lambda_process_diurnal", process_cfg.get("lambda_diurnal", 0.0))),
            lambda_process_proxy=float(data.get("lambda_process_proxy", process_cfg.get("lambda", 0.0))),
            lambda_proxy_roughness_wind=float(
                data.get("lambda_proxy_roughness_wind", process_cfg.get("lambda_roughness_wind", 0.0))
            ),
            lambda_proxy_diurnal_building=float(
                data.get("lambda_proxy_diurnal_building", process_cfg.get("lambda_diurnal_building", 0.0))
            ),
            lambda_proxy_impervious_d2m=float(
                data.get("lambda_proxy_impervious_d2m", process_cfg.get("lambda_impervious_d2m", 0.0))
            ),
            lambda_proxy_wind_tcc=float(data.get("lambda_proxy_wind_tcc", process_cfg.get("lambda_wind_tcc", 0.0))),
            w_tp=float(data.get("w_tp_nonneg", 1.0)),
            w_tcc=float(data.get("w_tcc_01", 0.5)),
            w_dew=float(data.get("w_dew_leq_t", 0.5)),
            w_sp=float(data.get("w_sp_nonneg", 0.1)),
            w_div=float(data.get("w_wind_div", 0.1)),
        )


def _split_dynamic_channels(pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
    if pred_denorm.shape[1] < len(DYNAMIC_VAR_ORDER):
        raise ValueError("物理一致性损失要求至少 7 个动态变量通道。")
    return {name: pred_denorm[:, idx] for idx, name in enumerate(DYNAMIC_VAR_ORDER)}


def _zero_terms(pred: torch.Tensor, names: tuple[str, ...]) -> Dict[str, torch.Tensor]:
    zero = pred.new_tensor(0.0)
    return {name: zero for name in names}


def _resize_static(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == ref.shape[-2:]:
        return value
    return F.interpolate(value.float(), size=ref.shape[-2:], mode="nearest")


def _normalize_proxy(proxy: torch.Tensor) -> torch.Tensor:
    flat = proxy.flatten(1)
    lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
    hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
    return (proxy - lo) / (hi - lo).clamp_min(1e-6)


def _static_proxy(
    ref: torch.Tensor,
    *,
    static_raw: Optional[torch.Tensor] = None,
    static_cont: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    source = None
    if static_cont is not None and static_cont.numel() > 0:
        source = static_cont[:, :1]
    elif static_raw is not None and static_raw.numel() > 0:
        source = static_raw[:, 1:2] if static_raw.size(1) > 1 else static_raw[:, :1]
    if source is None:
        return None
    source = _resize_static(source.to(device=ref.device, dtype=ref.dtype), ref)
    return _normalize_proxy(source[:, 0])


def _masked_mean(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=field.device, dtype=field.dtype)
    return (field * weights).sum() / weights.sum().clamp_min(1.0)


def _to_celsius(value: torch.Tensor) -> torch.Tensor:
    scale_hint = value.detach().mean()
    if bool((scale_hint > 150.0).item()):
        return value - 273.15
    return value


class FeasibilityProxyLoss:
    """变量物理可行性代理项。"""

    def __init__(self, config: PhysicalConsistencyConfig) -> None:
        self.config = config

    def __call__(self, pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算变量范围与基本关系约束。

        Args:
            pred_denorm: 物理量空间预测场。

        Returns:
            feasibility 子项与加权总量。
        """

        terms = self.terms(pred_denorm)
        total = (
            self.config.w_tp * terms["tp_nonneg"]
            + self.config.w_tcc * terms["tcc_01"]
            + self.config.w_dew * terms["dew_leq_t"]
            + self.config.w_sp * terms["sp_nonneg"]
        )
        return {**terms, "total": total}

    @staticmethod
    def terms(pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        channels = _split_dynamic_channels(pred_denorm)
        return {
            "tp_nonneg": torch.relu(-channels["tp"]).mean(),
            "tcc_01": torch.relu(-channels["tcc"]).mean() + torch.relu(channels["tcc"] - 1.0).mean(),
            "dew_leq_t": torch.relu(channels["d2m"] - channels["t2m"]).mean(),
            "sp_nonneg": torch.relu(-channels["sp"]).mean(),
        }

    @staticmethod
    def zeros(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = _zero_terms(pred, ("tp_nonneg", "tcc_01", "dew_leq_t", "sp_nonneg"))
        out["total"] = pred.new_tensor(0.0)
        return out


class StructureProxyLoss:
    """频谱、梯度与风场散度结构代理项。"""

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
        *,
        fft_channel_weights: torch.Tensor,
        grad_channel_weights: torch.Tensor,
        sobel_x: torch.Tensor,
        sobel_y: torch.Tensor,
    ) -> None:
        self.config = config
        self.fft_channel_weights = fft_channel_weights
        self.grad_channel_weights = grad_channel_weights
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算结构代理项。

        Args:
            pred: 归一化空间预测场。
            target: 归一化空间目标场。
            pred_denorm: 物理量空间预测场。

        Returns:
            structure 子项与诊断总量。
        """

        terms = self.zeros(pred)
        if self.config.use_structure_loss and self.config.use_fft_loss and self.config.lambda_fft > 0.0:
            terms["fft"] = self.spectral(pred, target)
        if self.config.use_structure_loss and self.config.use_gradient_loss and self.config.lambda_grad > 0.0:
            terms["grad"] = self.gradient(pred, target)
        if self.config.use_physics_loss and self.config.lambda_phys > 0.0:
            terms["wind_div"] = self.wind_divergence(pred_denorm)
        terms["legacy_total"] = terms["fft"] + terms["grad"]
        terms["total"] = terms["legacy_total"] + self.config.w_div * terms["wind_div"]
        return terms

    def spectral(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z_pred = torch.fft.rfft2(pred, norm="ortho")
        z_tgt = torch.fft.rfft2(target, norm="ortho")
        loss_amp = torch.abs(z_pred.abs() - z_tgt.abs())
        loss_phase = torch.abs(z_pred.angle() - z_tgt.angle())
        weights = self.fft_channel_weights.to(device=pred.device, dtype=loss_amp.dtype)
        total = (loss_amp + 0.1 * loss_phase) * weights
        return total.mean()

    def gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = pred.shape
        pred_flat = pred.reshape(batch * channels, 1, height, width)
        target_flat = target.reshape(batch * channels, 1, height, width)

        sobel_x = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sobel_y = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
        pred_gx = F.conv2d(pred_flat, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, sobel_y, padding=1)
        target_gx = F.conv2d(target_flat, sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, sobel_y, padding=1)

        weights = self.grad_channel_weights.to(device=pred.device, dtype=pred.dtype)
        loss = torch.abs(pred_gx - target_gx) + torch.abs(pred_gy - target_gy)
        loss = loss.reshape(batch, channels, height, width) * weights
        return loss.mean()

    @staticmethod
    def wind_divergence(pred_denorm: torch.Tensor) -> torch.Tensor:
        channels = _split_dynamic_channels(pred_denorm)
        u10 = channels["u10"]
        v10 = channels["v10"]

        du_dx = u10[:, :, 1:] - u10[:, :, :-1]
        du_dx = F.pad(du_dx, (0, 1, 0, 0))
        dv_dy = v10[:, 1:, :] - v10[:, :-1, :]
        dv_dy = F.pad(dv_dy, (0, 0, 0, 1))
        return (du_dx + dv_dy).abs().mean()

    @staticmethod
    def zeros(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = _zero_terms(pred, ("fft", "grad", "wind_div"))
        out["legacy_total"] = pred.new_tensor(0.0)
        out["total"] = pred.new_tensor(0.0)
        return out


class ProcessConsistencyLoss:
    """相对湿度、形态阻力与日周期热响应过程代理项。"""

    def __init__(self, config: PhysicalConsistencyConfig) -> None:
        self.config = config

    def __call__(
        self,
        pred_denorm: torch.Tensor,
        *,
        latest_state: Optional[torch.Tensor] = None,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算过程一致性代理项。

        Args:
            pred_denorm: 物理量空间预测场。
            latest_state: 当前上下文最后一帧的物理量空间场；缺失时增量过程项跳过。
            static_raw: 可选原始静态字段。
            static_cont: 可选连续静态字段。
            static_cat: 可选类别静态字段，保留接口兼容。
            hour_of_day: 可选当前上下文小时；缺失时日周期项跳过。

        Returns:
            process 子项、加权总量与诊断。
        """

        del static_cat
        if not self.config.use_process_consistency:
            return self.zeros(pred_denorm)

        rh = self.relative_humidity_consistency(pred_denorm)
        drag, drag_diag = self.drag_consistency(
            pred_denorm,
            latest_state=latest_state,
            static_raw=static_raw,
            static_cont=static_cont,
        )
        diurnal, diurnal_diag = self.diurnal_thermal_response(
            pred_denorm,
            latest_state=latest_state,
            static_raw=static_raw,
            static_cont=static_cont,
            hour_of_day=hour_of_day,
        )
        total = (
            self.config.lambda_process_rh * rh
            + self.config.lambda_process_drag * drag
            + self.config.lambda_process_diurnal * diurnal
        )
        return {
            "rh_consistency": rh,
            "drag_consistency": drag,
            "diurnal_thermal": diurnal,
            "diurnal_available": diurnal_diag["diurnal_available"],
            "drag_available": drag_diag["drag_available"],
            "high_drag_delta_speed": drag_diag["high_drag_delta_speed"],
            "low_drag_delta_speed": drag_diag["low_drag_delta_speed"],
            "drag_consistency_gap": drag_diag["drag_consistency_gap"],
            "daytime_heat_response_gap": diurnal_diag["daytime_heat_response_gap"],
            "nighttime_heat_release_gap": diurnal_diag["nighttime_heat_release_gap"],
            "total": total,
        }

    @staticmethod
    def relative_humidity_consistency(pred_denorm: torch.Tensor) -> torch.Tensor:
        channels = _split_dynamic_channels(pred_denorm)
        dew_c = _to_celsius(channels["d2m"])
        temp_c = _to_celsius(channels["t2m"])
        a = 17.625
        b = 243.04
        vapor = torch.exp(a * dew_c / (b + dew_c).clamp_min(1e-6))
        saturation = torch.exp(a * temp_c / (b + temp_c).clamp_min(1e-6))
        rh = vapor / saturation.clamp_min(1e-6)
        return (torch.relu(rh - 1.0).pow(2) + torch.relu(-rh).pow(2)).mean()

    @staticmethod
    def drag_consistency(
        pred_denorm: torch.Tensor,
        *,
        latest_state: Optional[torch.Tensor] = None,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        drag_quantile: float = 0.2,
        drag_margin: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero_diag = {
            "drag_available": pred_denorm.new_tensor(0.0),
            "high_drag_delta_speed": pred_denorm.new_tensor(0.0),
            "low_drag_delta_speed": pred_denorm.new_tensor(0.0),
            "drag_consistency_gap": pred_denorm.new_tensor(0.0),
        }
        if latest_state is None:
            return pred_denorm.new_tensor(0.0), zero_diag
        proxy = _static_proxy(pred_denorm, static_raw=static_raw, static_cont=static_cont)
        if proxy is None:
            return pred_denorm.new_tensor(0.0), zero_diag
        if latest_state.shape[-2:] != pred_denorm.shape[-2:]:
            latest_state = F.interpolate(latest_state, size=pred_denorm.shape[-2:], mode="bilinear", align_corners=False)
        latest_state = latest_state.to(device=pred_denorm.device, dtype=pred_denorm.dtype)
        pred_channels = _split_dynamic_channels(pred_denorm)
        last_channels = _split_dynamic_channels(latest_state)
        speed_pred = torch.sqrt(pred_channels["u10"].pow(2) + pred_channels["v10"].pow(2) + 1e-8)
        speed_last = torch.sqrt(last_channels["u10"].pow(2) + last_channels["v10"].pow(2) + 1e-8)
        delta_speed = speed_pred - speed_last
        flat = proxy.flatten(1)
        q = float(min(max(drag_quantile, 1e-3), 0.49))
        low_q = torch.quantile(flat, q, dim=1).view(-1, 1, 1)
        high_q = torch.quantile(flat, 1.0 - q, dim=1).view(-1, 1, 1)
        low_delta = _masked_mean(delta_speed, proxy <= low_q)
        high_delta = _masked_mean(delta_speed, proxy >= high_q)
        gap = high_delta - low_delta
        loss = torch.relu(gap + float(drag_margin))
        return loss, {
            "drag_available": pred_denorm.new_tensor(1.0),
            "high_drag_delta_speed": high_delta.detach(),
            "low_drag_delta_speed": low_delta.detach(),
            "drag_consistency_gap": gap.detach(),
        }

    @staticmethod
    def diurnal_thermal_response(
        pred_denorm: torch.Tensor,
        *,
        latest_state: Optional[torch.Tensor] = None,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        daytime_hours: tuple[int, int] = (8, 18),
        nighttime_hours: tuple[int, int] = (20, 6),
        margin: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero_diag = {
            "diurnal_available": pred_denorm.new_tensor(0.0),
            "daytime_heat_response_gap": pred_denorm.new_tensor(0.0),
            "nighttime_heat_release_gap": pred_denorm.new_tensor(0.0),
        }
        if hour_of_day is None or latest_state is None:
            return pred_denorm.new_tensor(0.0), zero_diag
        proxy = _static_proxy(pred_denorm, static_raw=static_raw, static_cont=static_cont)
        if proxy is None:
            return pred_denorm.new_tensor(0.0), zero_diag
        if latest_state.shape[-2:] != pred_denorm.shape[-2:]:
            latest_state = F.interpolate(latest_state, size=pred_denorm.shape[-2:], mode="bilinear", align_corners=False)
        latest_state = latest_state.to(device=pred_denorm.device, dtype=pred_denorm.dtype)

        hour = hour_of_day.to(device=pred_denorm.device, dtype=pred_denorm.dtype).reshape(-1)
        if hour.numel() == 1 and pred_denorm.size(0) > 1:
            hour = hour.expand(pred_denorm.size(0))
        if hour.numel() != pred_denorm.size(0):
            raise ValueError(f"hour_of_day 需要 shape=[B]，但得到 {tuple(hour_of_day.shape)}")
        valid = ((hour >= 0.0) & (hour <= 23.0))
        if float(valid.to(dtype=pred_denorm.dtype).sum().detach().cpu()) <= 0.0:
            return pred_denorm.new_tensor(0.0), zero_diag

        pred_channels = _split_dynamic_channels(pred_denorm)
        last_channels = _split_dynamic_channels(latest_state)
        delta_temp = pred_channels["t2m"] - last_channels["t2m"]
        flat = proxy.flatten(1)
        low_q = torch.quantile(flat, 0.2, dim=1).view(-1, 1, 1)
        high_q = torch.quantile(flat, 0.8, dim=1).view(-1, 1, 1)
        low_mask = proxy <= low_q
        high_mask = proxy >= high_q

        day_start, day_end = float(daytime_hours[0]), float(daytime_hours[1])
        night_start, night_end = float(nighttime_hours[0]), float(nighttime_hours[1])
        is_day = (hour >= day_start) & (hour < day_end) & valid
        if night_start <= night_end:
            is_night = (hour >= night_start) & (hour < night_end) & valid
        else:
            is_night = ((hour >= night_start) | (hour < night_end)) & valid

        loss_day = pred_denorm.new_tensor(0.0)
        loss_night = pred_denorm.new_tensor(0.0)
        day_gap = pred_denorm.new_tensor(0.0)
        night_gap = pred_denorm.new_tensor(0.0)
        if bool(is_day.any().item()):
            day_mask = is_day.view(-1, 1, 1)
            high_delta = _masked_mean(delta_temp, high_mask & day_mask)
            low_delta = _masked_mean(delta_temp, low_mask & day_mask)
            day_gap = high_delta - low_delta
            loss_day = torch.relu(low_delta - high_delta + float(margin))
        if bool(is_night.any().item()):
            night_mask = is_night.view(-1, 1, 1)
            high_delta = _masked_mean(delta_temp, high_mask & night_mask)
            low_delta = _masked_mean(delta_temp, low_mask & night_mask)
            night_gap = high_delta - low_delta
            loss_night = torch.relu(low_delta - high_delta + float(margin))

        return loss_day + loss_night, {
            "diurnal_available": valid.to(dtype=pred_denorm.dtype).mean(),
            "daytime_heat_response_gap": day_gap.detach(),
            "nighttime_heat_release_gap": night_gap.detach(),
        }

    @staticmethod
    def zeros(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = _zero_terms(pred, ("rh_consistency", "drag_consistency", "diurnal_thermal"))
        out["diurnal_available"] = pred.new_tensor(0.0)
        out["drag_available"] = pred.new_tensor(0.0)
        out["high_drag_delta_speed"] = pred.new_tensor(0.0)
        out["low_drag_delta_speed"] = pred.new_tensor(0.0)
        out["drag_consistency_gap"] = pred.new_tensor(0.0)
        out["daytime_heat_response_gap"] = pred.new_tensor(0.0)
        out["nighttime_heat_release_gap"] = pred.new_tensor(0.0)
        out["total"] = pred.new_tensor(0.0)
        return out


def _group_variance_gap(field: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
    low_mask, high_mask = _low_high_masks(_normalize_proxy(proxy))
    low_mean = _masked_mean(field, low_mask)
    high_mean = _masked_mean(field, high_mask)
    low_var = _masked_mean((field - low_mean).square(), low_mask)
    high_var = _masked_mean((field - high_mean).square(), high_mask)
    return torch.relu(high_var - low_var)


def _proxy_from_fields(ref: torch.Tensor, proxy_fields: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in proxy_fields:
        return ref.new_zeros(ref.size(0), ref.size(2), ref.size(3))
    proxy = proxy_fields[name].detach().to(device=ref.device, dtype=ref.dtype)
    if proxy.ndim == 4:
        proxy = proxy[:, 0]
    if proxy.shape[-2:] != ref.shape[-2:]:
        proxy = F.interpolate(proxy.unsqueeze(1), size=ref.shape[-2:], mode="nearest")[:, 0]
    return _normalize_proxy(proxy)


def _low_high_masks(proxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = proxy.flatten(1)
    low_q = torch.quantile(flat, 0.2, dim=1).view(-1, 1, 1)
    high_q = torch.quantile(flat, 0.8, dim=1).view(-1, 1, 1)
    return proxy <= low_q, proxy >= high_q


def _safe_spatial_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    left_flat = left_flat - left_flat.mean(dim=1, keepdim=True)
    right_flat = right_flat - right_flat.mean(dim=1, keepdim=True)
    denom = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return ((left_flat * right_flat).sum(dim=1) / denom.clamp_min(1e-6)).mean()


class ProcessProxyConsistencyLoss:
    """基于 process proxy 的软一致性诊断。"""

    def __init__(self, config: PhysicalConsistencyConfig) -> None:
        self.config = config

    def __call__(
        self,
        pred_denorm: torch.Tensor,
        *,
        proxy_fields: Optional[Mapping[str, torch.Tensor]] = None,
        latest_state: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if proxy_fields is None:
            return self.zeros(pred_denorm)
        terms = self.terms(
            pred_denorm,
            proxy_fields=proxy_fields,
            latest_state=latest_state,
            hour_of_day=hour_of_day,
        )
        total = (
            self.config.lambda_proxy_roughness_wind * terms["roughness_wind_decay"]
            + self.config.lambda_proxy_diurnal_building * terms["diurnal_range_by_building"]
            + self.config.lambda_proxy_impervious_d2m * terms["impervious_d2m_corr"]
            + self.config.lambda_proxy_wind_tcc * terms["wind_tcc_spread"]
        )
        return {**terms, "total": total}

    @staticmethod
    def terms(
        pred_denorm: torch.Tensor,
        *,
        proxy_fields: Mapping[str, torch.Tensor],
        latest_state: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del hour_of_day
        channels = _split_dynamic_channels(pred_denorm)
        roughness = _proxy_from_fields(pred_denorm, proxy_fields, "roughness_proxy")
        impervious = _proxy_from_fields(pred_denorm, proxy_fields, "impervious_proxy")
        wind_speed = (channels["u10"].square() + channels["v10"].square()).sqrt()
        low_rough, high_rough = _low_high_masks(roughness)
        roughness_wind = torch.relu(_masked_mean(wind_speed, high_rough) - _masked_mean(wind_speed, low_rough))

        diurnal = pred_denorm.new_tensor(0.0)
        if latest_state is not None:
            latest = latest_state.to(device=pred_denorm.device, dtype=pred_denorm.dtype)
            if latest.shape[-2:] != pred_denorm.shape[-2:]:
                latest = F.interpolate(latest, size=pred_denorm.shape[-2:], mode="bilinear", align_corners=False)
            delta_t = channels["t2m"] - _split_dynamic_channels(latest)["t2m"]
            low_imp, high_imp = _low_high_masks(impervious)
            diurnal = torch.relu(_masked_mean(delta_t, low_imp) - _masked_mean(delta_t, high_imp))

        return {
            "roughness_wind_decay": roughness_wind,
            "diurnal_range_by_building": diurnal,
            "impervious_d2m_corr": torch.relu(_safe_spatial_corr(impervious, channels["d2m"])),
            "wind_tcc_spread": _group_variance_gap(channels["tcc"], wind_speed),
        }

    @staticmethod
    def zeros(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = _zero_terms(
            pred,
            ("roughness_wind_decay", "diurnal_range_by_building", "impervious_d2m_corr", "wind_tcc_spread"),
        )
        out["total"] = pred.new_tensor(0.0)
        return out


class PhysicalConsistencyLoss:
    """兼容旧 LightningModule 的物理一致性 facade。"""

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
        *,
        fft_channel_weights: torch.Tensor,
        grad_channel_weights: torch.Tensor,
        sobel_x: torch.Tensor,
        sobel_y: torch.Tensor,
    ) -> None:
        self.config = config
        self.feasibility_proxy = FeasibilityProxyLoss(config)
        self.structure_proxy = StructureProxyLoss(
            config,
            fft_channel_weights=fft_channel_weights,
            grad_channel_weights=grad_channel_weights,
            sobel_x=sobel_x,
            sobel_y=sobel_y,
        )
        self.process_proxy = ProcessConsistencyLoss(config)
        self.process_proxy_consistency = ProcessProxyConsistencyLoss(config)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        std: torch.Tensor,
        mean: torch.Tensor,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        latest_state: Optional[torch.Tensor] = None,
        proxy_fields: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """计算完整物理一致性损失包。

        Args:
            pred: 归一化空间预测场，形状 [B,C,H,W]。
            target: 归一化空间目标场，形状 [B,C,H,W]。
            std: 训练集标准差，形状可广播到 pred。
            mean: 训练集均值，形状可广播到 pred。
            static_raw: 可选静态字段，仅供新增过程诊断使用。
            static_cont: 可选连续静态字段，仅供新增过程诊断使用。
            static_cat: 可选类别静态字段，保留接口兼容。
            hour_of_day: 可选当前上下文小时；缺失时日周期过程项跳过。

        Returns:
            与旧 LightningModule 兼容的损失字典，并附带新日志别名。
        """

        pred_denorm = pred * std + mean

        if self.config.use_physics_loss and self.config.use_feasibility_loss and self.config.lambda_phys > 0.0:
            feasibility = self.feasibility_proxy(pred_denorm)
        else:
            feasibility = FeasibilityProxyLoss.zeros(pred)

        structure = self.structure_proxy(pred, target, pred_denorm)
        process = self.process_proxy(
            pred_denorm,
            latest_state=latest_state,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
        )
        proxy_process = self.process_proxy_consistency(
            pred_denorm,
            proxy_fields=proxy_fields,
            latest_state=latest_state,
            hour_of_day=hour_of_day,
        )

        legacy_wind_div_loss = self.config.w_div * structure["wind_div"]
        loss_feasibility = feasibility["total"]
        loss_process = legacy_wind_div_loss + process["total"] + proxy_process["total"]
        loss_phys_spatial = loss_feasibility + loss_process
        loss_structure = structure["legacy_total"]

        physical_feasibility_score = torch.reciprocal(1.0 + loss_feasibility.detach())
        structure_score = torch.reciprocal(1.0 + (loss_structure + legacy_wind_div_loss).detach())
        process_consistency_score = torch.reciprocal(1.0 + (process["total"] + proxy_process["total"]).detach())

        logs: Dict[str, Any] = {
            "phys_tp_nonneg": feasibility["tp_nonneg"],
            "phys_tcc_01": feasibility["tcc_01"],
            "phys_dew_leq_t": feasibility["dew_leq_t"],
            "phys_sp_nonneg": feasibility["sp_nonneg"],
            "phys_wind_div": structure["wind_div"],
            "phys_feasibility": loss_feasibility,
            "phys_process_consistency": loss_process,
            "phys_structure": loss_structure,
            "phys_spatial": loss_phys_spatial,
            "fft": structure["fft"],
            "grad": structure["grad"],
            "loss/feasibility_tp_nonneg": feasibility["tp_nonneg"],
            "loss/feasibility_tcc_01": feasibility["tcc_01"],
            "loss/feasibility_dew_leq_t": feasibility["dew_leq_t"],
            "loss/feasibility_sp_nonneg": feasibility["sp_nonneg"],
            "loss/feasibility_total": loss_feasibility,
            "loss/structure_fft": structure["fft"],
            "loss/structure_grad": structure["grad"],
            "loss/structure_wind_div": legacy_wind_div_loss,
            "loss/structure_total": structure["total"],
            "loss/process_rh": process["rh_consistency"],
            "loss/process_drag": process["drag_consistency"],
            "loss/process_diurnal": process["diurnal_thermal"],
            "loss/process_proxy_roughness_wind_decay": proxy_process["roughness_wind_decay"],
            "loss/process_proxy_diurnal_range_by_building": proxy_process["diurnal_range_by_building"],
            "loss/process_proxy_impervious_d2m_corr": proxy_process["impervious_d2m_corr"],
            "loss/process_proxy_wind_tcc_spread": proxy_process["wind_tcc_spread"],
            "loss/process_proxy_total": proxy_process["total"],
            "loss/process_total": process["total"] + proxy_process["total"],
            "feasibility/tp_nonneg": feasibility["tp_nonneg"],
            "feasibility/tcc_01": feasibility["tcc_01"],
            "feasibility/dew_leq_t": feasibility["dew_leq_t"],
            "feasibility/sp_nonneg": feasibility["sp_nonneg"],
            "feasibility/total": loss_feasibility,
            "structure/fft": structure["fft"],
            "structure/grad": structure["grad"],
            "structure/wind_div": legacy_wind_div_loss,
            "structure/total": structure["total"],
            "process_proxy/rh_consistency": process["rh_consistency"],
            "process_proxy/drag_consistency": process["drag_consistency"],
            "process_proxy/diurnal_thermal": process["diurnal_thermal"],
            "process_proxy/roughness_wind_decay": proxy_process["roughness_wind_decay"],
            "process_proxy/diurnal_range_by_building": proxy_process["diurnal_range_by_building"],
            "process_proxy/impervious_d2m_corr": proxy_process["impervious_d2m_corr"],
            "process_proxy/wind_tcc_spread": proxy_process["wind_tcc_spread"],
            "process_proxy/total": process["total"] + proxy_process["total"],
            "diag/drag_available": process["drag_available"],
            "diag/high_drag_delta_speed": process["high_drag_delta_speed"],
            "diag/low_drag_delta_speed": process["low_drag_delta_speed"],
            "diag/drag_consistency_gap": process["drag_consistency_gap"],
            "diag/diurnal_available": process["diurnal_available"],
            "diag/daytime_heat_response_gap": process["daytime_heat_response_gap"],
            "diag/nighttime_heat_release_gap": process["nighttime_heat_release_gap"],
            "physical_feasibility_score": physical_feasibility_score,
            "structure_score": structure_score,
            "process_consistency_score": process_consistency_score,
        }

        return {
            "x0_pred_denorm": pred_denorm,
            "loss_phys_spatial": loss_phys_spatial,
            "loss_feasibility": loss_feasibility,
            "loss_process_consistency": loss_process,
            "loss_structure": loss_structure,
            "loss_fft": structure["fft"],
            "loss_grad": structure["grad"],
            "logs": logs,
            "feasibility": feasibility,
            "structure": structure,
            "process_consistency": process,
            "process_proxy_consistency": proxy_process,
        }

    def feasibility(self, pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧调用的 feasibility 入口。"""

        return FeasibilityProxyLoss.terms(pred_denorm)

    def process_consistency(self, pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧调用的风场散度入口。"""

        return {"wind_div": StructureProxyLoss.wind_divergence(pred_denorm)}

    def spectral_structure(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """兼容旧调用的频谱结构入口。"""

        return self.structure_proxy.spectral(pred, target)

    def gradient_structure(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """兼容旧调用的梯度结构入口。"""

        return self.structure_proxy.gradient(pred, target)

    @staticmethod
    def _split_dynamic_channels(pred_denorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧静态方法。"""

        return _split_dynamic_channels(pred_denorm)

    @staticmethod
    def _zero_feasibility(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧静态方法。"""

        return {k: v for k, v in FeasibilityProxyLoss.zeros(pred).items() if k != "total"}

    @staticmethod
    def _zero_process_consistency(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧静态方法。"""

        return {"wind_div": pred.new_tensor(0.0)}

    @staticmethod
    def _zero_structure(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """兼容旧静态方法。"""

        return {"fft": pred.new_tensor(0.0), "grad": pred.new_tensor(0.0)}


__all__ = [
    "DYNAMIC_VAR_ORDER",
    "FeasibilityProxyLoss",
    "PhysicalConsistencyConfig",
    "PhysicalConsistencyLoss",
    "ProcessConsistencyLoss",
    "ProcessProxyConsistencyLoss",
    "StructureProxyLoss",
]