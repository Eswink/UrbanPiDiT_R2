r"""
模型复杂度测量工具。

该模块提供项目内统一的参数量与 FLOPs 估计逻辑。当前 FLOPs 口径与
``baselines.forecast_runner`` 原实现保持一致：只统计 ``Conv2d`` 与 ``Linear``
前向计算，排除 FFT、归一化、激活函数和张量重排等操作。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import torch


ForwardFn = Callable[[], Any]


FLOPS_NOTE = (
    "conv2d_linear_forward_estimate_excludes_fft_norm_activation; "
    "rollout_gflops_is_accumulated_forward_estimate"
)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    r"""
    统计模型参数量。

    Parameters
    ----
    model : torch.nn.Module
        待统计的模型。

    Returns
    ----
    Dict[str, int]
        ``num_parameters`` 与 ``num_trainable``。
    """

    return {
        "num_parameters": int(sum(param.numel() for param in model.parameters())),
        "num_trainable": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }


def batch_size_from_batch(batch: Optional[Dict[str, Any]]) -> int:
    r"""
    从嵌套 batch 中提取 batch size。

    Parameters
    ----
    batch : Dict[str, Any], optional
        DataLoader 输出的 batch。

    Returns
    ----
    int
        批大小。无法判断时返回 0。
    """

    if batch is None:
        return 0
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, dict):
            size = batch_size_from_batch(value)
            if size > 0:
                return size
    return 0


def compute_conv_linear_gflops(model: torch.nn.Module, forward_fn: ForwardFn) -> float:
    r"""
    使用 forward hook 估算 ``Conv2d`` 与 ``Linear`` 的前向 GFLOPs。

    Parameters
    ----
    model : torch.nn.Module
        待统计模型。
    forward_fn : Callable[[], Any]
        不带参数的前向闭包。闭包内部应执行一次需要计数的推理路径。

    Returns
    ----
    float
        估算 GFLOPs。
    """

    flops = 0
    hooks = []

    def conv_hook(module: torch.nn.Conv2d, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal flops
        if not isinstance(output, torch.Tensor):
            return
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        flops += int(output.numel() * kernel_ops * 2)
        if module.bias is not None:
            flops += int(output.numel())

    def linear_hook(module: torch.nn.Linear, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal flops
        if not isinstance(output, torch.Tensor):
            return
        flops += int(output.numel() * module.in_features * 2)
        if module.bias is not None:
            flops += int(output.numel())

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            forward_fn()
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return float(flops) / 1e9


def forecast_model_complexity_summary(
    model: torch.nn.Module,
    batch: Optional[Dict[str, Any]] = None,
    lead_times: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    r"""
    生成 ForecastModelBase 风格模型的复杂度摘要。

    Parameters
    ----
    model : torch.nn.Module
        具备 ``predict(batch, lead_times=...)`` 接口的预测模型。
    batch : Dict[str, Any], optional
        用于触发前向路径的样例 batch。
    lead_times : Sequence[int], optional
        需要评估的提前期步数。

    Returns
    ----
    Dict[str, Any]
        与基线结果兼容的复杂度摘要。
    """

    protocol = str(getattr(model, "forecast_protocol", "direct"))
    one_step = int(getattr(model, "one_step_lead", 1))
    requested_leads = [int(x) for x in list(lead_times or getattr(model, "lead_times", None) or [one_step])]
    rollout_steps = max(requested_leads) if protocol == "official_rollout" and requested_leads else len(requested_leads)
    diffusion_steps = int(getattr(model, "num_noise_levels", 0) or 0)
    summary: Dict[str, Any] = {
        **count_parameters(model),
        "protocol": protocol,
        "one_step_lead": one_step,
        "rollout_steps": int(rollout_steps),
        "num_diffusion_steps": diffusion_steps,
        "estimated_gflops": 0.0,
        "estimated_gflops_per_sample": 0.0,
        "one_step_gflops": 0.0,
        "one_step_gflops_per_sample": 0.0,
        "rollout_gflops": 0.0,
        "rollout_gflops_per_sample": 0.0,
        "flops_note": FLOPS_NOTE,
    }
    if batch is None:
        return summary

    batch_gflops = compute_conv_linear_gflops(
        model,
        lambda: model.predict(batch, lead_times=requested_leads),
    )
    one_step_gflops = compute_conv_linear_gflops(
        model,
        lambda: model.predict(batch, lead_times=[one_step]),
    )
    batch_size = max(batch_size_from_batch(batch), 1)
    summary["estimated_gflops"] = batch_gflops
    summary["estimated_gflops_per_sample"] = batch_gflops / batch_size
    summary["one_step_gflops"] = one_step_gflops
    summary["one_step_gflops_per_sample"] = one_step_gflops / batch_size
    summary["rollout_gflops"] = batch_gflops if protocol == "official_rollout" else 0.0
    summary["rollout_gflops_per_sample"] = summary["rollout_gflops"] / batch_size
    return summary


def _default_urban_pidit_steps(lit_module: torch.nn.Module) -> tuple[int, float, float]:
    diff_cfg = dict(getattr(lit_module, "hparams", {}).get("diffusion_cfg", {}) or {})
    return (
        int(diff_cfg.get("sample_steps", 8)),
        float(diff_cfg.get("t_start", 0.0)),
        float(diff_cfg.get("t_end", 1.0)),
    )


def _urban_static_kwargs(lit_module: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
    if hasattr(lit_module, "_move_static_batch") and hasattr(lit_module, "_model_static_kwargs"):
        static_batch = lit_module._move_static_batch(batch)
        return dict(lit_module._model_static_kwargs(static_batch))
    return {"static_raw": None, "static_cont": None, "static_cat": None, "hour_of_day": None}


def _urban_lead_time(lit_module: torch.nn.Module, x_ctx: torch.Tensor, lead_steps: int) -> Optional[torch.Tensor]:
    if not bool(getattr(lit_module, "use_lead_time_conditioning", False)):
        return None
    lead = torch.full((x_ctx.size(0),), float(int(lead_steps)), device=x_ctx.device, dtype=torch.float32)
    if hasattr(lit_module, "_normalize_lead_time"):
        return lit_module._normalize_lead_time(lead)
    denom = float(max(int(getattr(lit_module, "max_lead_time_steps", 1)), 1))
    return lead / denom


def urban_pidit_complexity_summary(
    lit_module: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    lead_times: Optional[Sequence[int]] = None,
    steps: Optional[int] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Dict[str, Any]:
    r"""
    生成 UrbanPiDiT 的复杂度摘要。

    Parameters
    ----
    lit_module : torch.nn.Module
        ``UrbanPiDiTLitModule`` 实例。
    batch : Dict[str, Any]
        用于触发真实采样路径的样例 batch。
    lead_times : Sequence[int], optional
        需要评估的提前期步数。
    steps : int, optional
        扩散采样步数。None 时读取配置。
    t_start : float, optional
        扩散起始时间。
    t_end : float, optional
        扩散结束时间。

    Returns
    ----
    Dict[str, Any]
        与基线复杂度摘要兼容的字典。
    """

    default_steps, default_t_start, default_t_end = _default_urban_pidit_steps(lit_module)
    steps = int(default_steps if steps is None else steps)
    t_start = float(default_t_start if t_start is None else t_start)
    t_end = float(default_t_end if t_end is None else t_end)
    requested_leads = [int(x) for x in list(lead_times or getattr(lit_module, "eval_lead_times", None) or [1])]
    protocol = str(getattr(lit_module, "multi_horizon_inference", "direct")).lower()
    if protocol == "direct" and not bool(getattr(lit_module, "use_lead_time_conditioning", False)):
        protocol = "autoregressive"
    rollout_steps = max(requested_leads) if protocol == "autoregressive" and requested_leads else len(requested_leads)

    net = getattr(lit_module, "net", lit_module)
    x_ctx = batch["x_ctx"].to(next(lit_module.parameters()).device)
    static_kwargs = _urban_static_kwargs(lit_module, batch)

    def one_step_forward() -> None:
        lead_time = _urban_lead_time(lit_module, x_ctx, 1)
        lit_module.predict_one(
            x_ctx,
            steps=steps,
            t_start=t_start,
            t_end=t_end,
            seed=None,
            lead_time=lead_time,
            **static_kwargs,
        )

    def requested_forward() -> None:
        if protocol == "autoregressive":
            ctx_cur = x_ctx
            for _ in range(1, int(max(requested_leads)) + 1):
                lead_time = _urban_lead_time(lit_module, ctx_cur, 1)
                x_pred = lit_module.predict_one(
                    ctx_cur,
                    steps=steps,
                    t_start=t_start,
                    t_end=t_end,
                    seed=None,
                    lead_time=lead_time,
                    **static_kwargs,
                )
                ctx_cur = lit_module._update_ctx_autoregressive(ctx_cur, x_pred)
            return
        for lead_steps in requested_leads:
            lead_time = _urban_lead_time(lit_module, x_ctx, int(lead_steps))
            lit_module.predict_one(
                x_ctx,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                seed=None,
                lead_time=lead_time,
                **static_kwargs,
            )

    one_step_gflops = compute_conv_linear_gflops(net, one_step_forward)
    estimated_gflops = compute_conv_linear_gflops(net, requested_forward)
    batch_size = max(batch_size_from_batch(batch), 1)
    summary: Dict[str, Any] = {
        **count_parameters(net),
        "protocol": protocol,
        "one_step_lead": 1,
        "rollout_steps": int(rollout_steps),
        "num_diffusion_steps": int(steps),
        "estimated_gflops": estimated_gflops,
        "estimated_gflops_per_sample": estimated_gflops / batch_size,
        "one_step_gflops": one_step_gflops,
        "one_step_gflops_per_sample": one_step_gflops / batch_size,
        "rollout_gflops": estimated_gflops if protocol == "autoregressive" else 0.0,
        "rollout_gflops_per_sample": (estimated_gflops / batch_size) if protocol == "autoregressive" else 0.0,
        "flops_note": FLOPS_NOTE,
    }
    return summary