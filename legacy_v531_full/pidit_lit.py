"""UrbanPiDiT LightningModule（V5.3.1 兼容训练封装）。

该文件封装了：
- 扩散式预测的训练、验证和测试逻辑，并保持验证/测试阶段不使用目标场构造模型输入。
- 可行性、结构一致性和 morphology-derived process proxy 一致性辅助损失的可控组合。
- 消融开关、提前期训练、rollout 微调、自适应权重、对抗训练和知识蒸馏等历史训练能力。

说明：
- 文件内保留的 `phys_*` 日志键、函数名和部分训练术语属于历史兼容接口。
- V5.3.1 主叙述采用 Morphology-conditioned Process Diffusion Transformer 口径，不把 proxy 表述为真实物理参数。
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except Exception as _e:
    # 允许在未安装 Lightning 的环境下做静态检查/导入（实际训练仍需安装 pytorch_lightning）
    pl = None  # type: ignore

from torch.optim.lr_scheduler import CosineAnnealingLR

# 兼容“作为包导入”和“直接运行脚本”两种方式
try:
    from .models.model_registry import build_model_from_config
    from .models.urban_pidit import UrbanPiDiT, AdaptiveWeightNet
    from .metrics.acc import acc as _acc
    from .metrics.crps import crps_ensemble as _crps_ensemble
    from .losses import PhysicalConsistencyConfig, PhysicalConsistencyLoss
    from .utils.diagnostics import collect_model_diagnostics
except Exception:
    from models.model_registry import build_model_from_config
    from models.urban_pidit import UrbanPiDiT, AdaptiveWeightNet
    from metrics.acc import acc as _acc
    from metrics.crps import crps_ensemble as _crps_ensemble
    from losses import PhysicalConsistencyConfig, PhysicalConsistencyLoss
    from utils.diagnostics import collect_model_diagnostics


class Discriminator(nn.Module):
    """判别器网络（用于对抗训练）。

    说明：
    - 判别器学习区分真实场 x0 与生成场 x0_pred。
    - 通过对抗训练提升生成场的“真实感/纹理”。
    """

    def __init__(self, in_channels: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # [B, C, H, W] -> [B, hidden, H/2, W/2]
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # -> [B, 2*hidden, H/4, W/4]
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B, C, H, W]

        Returns:
        logits: [B, 1, H', W']（越大表示越“真实”）
        """

        return self.net(x)


# Lightning 依赖：若未安装 pytorch_lightning，则 UrbanPiDiTLitModule 不可用
if pl is None:
    UrbanPiDiTLitModule = object  # type: ignore
else:

    class UrbanPiDiTLitModule(pl.LightningModule):
        """UrbanPiDiT 的 Lightning 封装。"""

        def __init__(
            self,
            model_cfg: Dict[str, Any],
            optim_cfg: Dict[str, Any],
            diffusion_cfg: Dict[str, Any],
            physics_cfg: Dict[str, Any],
            metrics_cfg: Dict[str, Any],
            inference_cfg: Optional[Dict[str, Any]] = None,
            forecast_cfg: Optional[Dict[str, Any]] = None,
        ):
            super().__init__()
            self.save_hyperparameters()

            # =========================
            # 1) 读取消融/开关配置
            # =========================
            self.use_cross_attention = bool(model_cfg.get("use_cross_attention", True))
            self.use_position_encoding = bool(model_cfg.get("use_position_encoding", True))
            self.use_timestep_conditioning = bool(model_cfg.get("use_timestep_conditioning", True))
            # 是否启用“预测提前期(lead time)”条件（多提前期训练/评估的关键开关）
            self.use_lead_time_conditioning = bool(model_cfg.get("use_lead_time_conditioning", False))
            self.use_hybrid_attention = bool(model_cfg.get("use_hybrid_attention", False))
            self.use_multi_scale = bool(model_cfg.get("use_multi_scale", False))
            self.use_variable_graph = bool(model_cfg.get("use_variable_graph", False))
            self.use_dynamic_vg = bool(model_cfg.get("use_dynamic_vg", self.use_variable_graph))
            self.use_static_vg = bool(model_cfg.get("use_static_vg", self.use_variable_graph))
            self.use_proxy_conditioned_vg = bool(model_cfg.get("use_proxy_conditioned_vg", False))
            self.vg_num_heads = int(model_cfg.get("vg_num_heads", 4))
            self.use_static_morphology_encoder = bool(model_cfg.get("use_static_morphology_encoder", False))
            self.use_morphology_graph = bool(model_cfg.get("use_morphology_graph", False))
            self.use_wind_aware_graph = bool(model_cfg.get("use_wind_aware_graph", False))
            self.use_urban_canopy_coupling = bool(model_cfg.get("use_urban_canopy_coupling", model_cfg.get("use_urban_canopy", False)))
            self.use_micromet_coupling = bool(model_cfg.get("use_micromet_coupling", False))
            self.use_process_proxy_encoder = bool(model_cfg.get("use_process_proxy_encoder", False))
            self.use_process_adaln = bool(model_cfg.get("use_process_adaln", False))
            self.use_urban_control_branch = bool(model_cfg.get("use_urban_control_branch", False))
            self.use_anisotropic_process_graph = bool(model_cfg.get("use_anisotropic_process_graph", False))
            self.use_morphology_residual_head = bool(model_cfg.get("use_morphology_residual_head", False))
            self.use_urban_graph = bool(model_cfg.get("use_urban_graph", False))

            # =========================
            # 1.5) 多提前期（multi-horizon）配置
            # =========================
            # 注意：time_step_hours=6 表示 1 个时间步=6小时（符合 ERA5 6h 数据）。
            self.forecast_cfg = forecast_cfg or {}
            self.time_step_hours = float(self.forecast_cfg.get("time_step_hours", 6.0))

            # 训练/评估时要覆盖的提前期（单位：步），例如 [1,2,3,4] 对应 6/12/18/24h
            # - 如果用户未设置，则默认只评估 1 步（兼容旧版）
            self.eval_lead_times = [int(x) for x in self.forecast_cfg.get("eval_lead_times", [1])]

            # 训练阶段的 lead time 采样策略（仅当 use_lead_time_conditioning=True 且 batch 提供 y 时生效）
            # 可选：uniform / curriculum
            self.train_lead_time_sampling = str(self.forecast_cfg.get("train_lead_time_sampling", "uniform"))
            self.curriculum_epochs = int(self.forecast_cfg.get("curriculum_epochs", 0))

            # lead time 的归一化尺度（用于网络条件输入）；若不提供则自动用 eval_lead_times 的最大值
            self.max_lead_time_steps = int(self.forecast_cfg.get("max_lead_time_steps", max(self.eval_lead_times)))

            # 多提前期推理策略：
            # - direct：直接预测指定 lead（需要 use_lead_time_conditioning=True）
            # - autoregressive：用 1-step 模型滚动预测（行业常用；GraphCast/FourCastNet 等多使用 6h autoregressive rollout）
            self.multi_horizon_inference = str(
                self.forecast_cfg.get(
                    "multi_horizon_inference",
                    "direct" if self.use_lead_time_conditioning else "autoregressive",
                )
            )

            # 是否在“同一批次/同一成员”下复用初始噪声以获得时间一致的多提前期轨迹
            # 该做法在扩散概率预报里很常见，可参考 Continuous Ensemble Forecasting 的“相关噪声/一致轨迹”思想。
            self.correlate_noise_across_leads = bool(self.forecast_cfg.get("correlate_noise_across_leads", True))

            # -------------------------
            # 训练阶段：多提前期 loss 组合策略（可控开关，便于消融）
            # -------------------------
            # 说明：
            # - 当 use_lead_time_conditioning=true 且 Dataset 返回 batch['y'] 时，训练会同时面对多个 lead 的监督信号。
            # - 如果每个 batch 只随机采 1 个 lead，且对 lead 不做权重处理，则长提前期(误差更大)会在梯度上占主导，
            #   这在工程上常见的副作用是：长提前期更好，但 6h(最短 lead) 会变差（你们反馈的现象正符合这一点）。
            #
            # train_objective：
            # - 'single_lead'     : 每个 batch 只训练 1 个随机 lead（计算最省，作为 baseline）
            # - 'lead1_plus_k'    : 每个 batch 固定包含最短 lead(通常 1 step=6h)，再额外随机抽 k 个更长 lead
            # - 'weighted_sum_all': 对允许的所有 lead 都计算 loss，再按 lead 权重加权求和
            self.train_objective = str(self.forecast_cfg.get('train_objective', 'lead1_plus_k')).lower()
            self.extra_leads_per_batch = int(self.forecast_cfg.get('extra_leads_per_batch', 1))
            self.force_include_short_lead = bool(self.forecast_cfg.get('force_include_short_lead', True))

            # lead loss 权重策略（用于避免长提前期误差更大导致梯度主导，从而拖累 6h）
            # - 'uniform'      : 全 1
            # - 'inverse'      : 1/lead
            # - 'inverse_sqrt' : 1/sqrt(lead)
            # - 'exp_decay'    : exp(-beta*(lead-min_lead))
            # - 'custom'       : 使用 lead_loss_weights 列表（与 lead_times 顺序对齐）
            self.lead_loss_weighting = str(self.forecast_cfg.get('lead_loss_weighting', 'inverse')).lower()
            self.lead_weight_beta = float(self.forecast_cfg.get('lead_weight_beta', 0.5))
            self.lead_loss_weights = self.forecast_cfg.get('lead_loss_weights', None)
            self.normalize_lead_loss = bool(self.forecast_cfg.get('normalize_lead_loss', True))

            # 是否把物理/FFT/梯度等“辅助损失”也施加到所有 lead
            # 默认 False：只对最短 lead 施加（更不容易伤 6h，也更省算力）；需要时可打开做消融
            self.apply_aux_losses_to_all_leads = bool(self.forecast_cfg.get('apply_aux_losses_to_all_leads', False))

            # -------------------------
            # v4 新增：lead loss 权重退火（Annealing）
            # -------------------------
            # 目的：
            # - 训练早期更偏短 lead（先把 6h 做稳），训练后期再逐步提高长 lead 权重（提升 18h/24h）。
            # - 这是一种“课程学习”的变体，在天气预报与序列建模里很常见。
            self.lead_weight_anneal_cfg = dict(self.forecast_cfg.get("lead_weight_anneal", {}) or {})
            self.lead_weight_anneal_enabled = bool(self.lead_weight_anneal_cfg.get("enabled", False))
            self.lead_weight_anneal_start = str(
                self.lead_weight_anneal_cfg.get("start_weighting", self.lead_loss_weighting)
            ).lower()
            self.lead_weight_anneal_end = str(self.lead_weight_anneal_cfg.get("end_weighting", "uniform")).lower()
            self.lead_weight_anneal_warmup_epochs = int(self.lead_weight_anneal_cfg.get("warmup_epochs", 0))
            self.lead_weight_anneal_epochs = int(self.lead_weight_anneal_cfg.get("anneal_epochs", 0))
            self.lead_weight_anneal_schedule = str(self.lead_weight_anneal_cfg.get("schedule", "linear")).lower()
            self.lead_weight_anneal_start_beta = float(self.lead_weight_anneal_cfg.get("start_beta", self.lead_weight_beta))
            self.lead_weight_anneal_end_beta = float(self.lead_weight_anneal_cfg.get("end_beta", 0.0))
            # 当 start/end_weighting=custom 时，可分别提供权重列表（与 lead_times 对齐）
            self.lead_weight_anneal_start_weights = self.lead_weight_anneal_cfg.get("start_weights", None)
            self.lead_weight_anneal_end_weights = self.lead_weight_anneal_cfg.get("end_weights", None)

            # -------------------------
            # v4 新增：多步 rollout 微调（multistep fine-tuning）
            # -------------------------
            # 说明：
            # - 业界很多天气模型（GraphCast/FourCastNet 等）是 1-step 预测 + autoregressive rollout。
            # - 单步训练在长 rollout 时会出现误差累积/漂移；因此常见做法是追加“多步微调”，
            #   在训练时显式考虑多步 rollout 的误差（可配合 scheduled sampling）。
            # - 这里实现为可选开关：只要 Dataset 提供 batch['y'](多提前期 GT)，即可启用。
            self.rollout_finetune_cfg = dict(
                self.forecast_cfg.get("rollout_finetune", None)
                or self.forecast_cfg.get("rollout_train", None)
                or self.forecast_cfg.get("multistep_finetune", None)
                or {}
            )
            self.rollout_finetune_enabled = bool(self.rollout_finetune_cfg.get("enabled", False))
            # 最终要 roll 到的最大步数（例如 4 => 24h）
            self.rollout_max_steps = int(self.rollout_finetune_cfg.get("max_steps", max(self.eval_lead_times)))
            # rollout 步数的课程学习：从 1 逐步增长到 rollout_max_steps
            self.rollout_curriculum_epochs = int(self.rollout_finetune_cfg.get("curriculum_epochs", 0))
            self.rollout_schedule = str(self.rollout_finetune_cfg.get("schedule", "linear")).lower()
            # loss 计算方式：
            # - all        : 对 1..K 的每一步都计算 loss（最严格）
            # - final_only : 只对最后一步(K)计算 loss（更省算力）
            self.rollout_loss_on = str(self.rollout_finetune_cfg.get("loss_on", "all")).lower()
            # 即便 loss_on=final_only，也建议始终保留 step-1(6h) 的 loss，避免 6h 回退
            self.rollout_always_include_step1 = bool(self.rollout_finetune_cfg.get("always_include_step1", True))
            # 更新上下文时是否 detach 预测（默认 detach：更省显存，避免多步反传导致不稳定）
            self.rollout_detach_context = bool(self.rollout_finetune_cfg.get("detach_context", True))

            # scheduled sampling（teacher forcing 比例）配置：从 start_ratio 逐步降到 end_ratio
            tf_cfg = dict(self.rollout_finetune_cfg.get("teacher_forcing", {}) or {})
            self.teacher_forcing_start = float(tf_cfg.get("start_ratio", 1.0))
            self.teacher_forcing_end = float(tf_cfg.get("end_ratio", 0.0))
            self.teacher_forcing_decay_epochs = int(tf_cfg.get("decay_epochs", self.rollout_curriculum_epochs))
            self.teacher_forcing_schedule = str(tf_cfg.get("schedule", "linear")).lower()

            # 损失相关开关
            self.use_channel_weights = bool(optim_cfg.get("use_channel_weights", True))
            self.use_adaptive_weights = bool(optim_cfg.get("use_adaptive_weights", False))

            # 物理相关开关
            self.use_physics_loss = bool(physics_cfg.get("use_physics_loss", True))
            self.use_fft_loss = bool(physics_cfg.get("use_fft_loss", True))
            self.use_gradient_loss = bool(physics_cfg.get("use_gradient_loss", True))
            self.use_hard_physics = bool(physics_cfg.get("use_hard_physics", False))
            self.use_adversarial = bool(physics_cfg.get("use_adversarial", False))
            self.use_distillation = bool(physics_cfg.get("use_distillation", False))

            # =========================
            # 2) 模型初始化
            # =========================
            H = int(model_cfg.get("H", 8))
            W = int(model_cfg.get("W", 8))
            D = int(model_cfg.get("D", 512))
            depth = int(model_cfg.get("depth", 12))
            heads = int(model_cfg.get("heads", 8))
            mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
            dropout = float(model_cfg.get("dropout", model_cfg.get("drop", 0.0)))

            in_channels = int(model_cfg.get("in_channels", 7))
            ctx_channels = int(model_cfg.get("ctx_channels", 28))
            out_channels = int(model_cfg.get("out_channels", 7))
            static_channels = int(model_cfg.get("static_channels", 0))

            # DropPath（可能被 ablation.use_drop_path 覆盖为 0）
            drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))

            model_init_cfg = dict(model_cfg)
            model_init_cfg.update(
                {
                    "H": H,
                    "W": W,
                    "D": D,
                    "depth": depth,
                    "heads": heads,
                    "mlp_ratio": mlp_ratio,
                    "dropout": dropout,
                    "in_channels": in_channels,
                    "ctx_channels": ctx_channels,
                    "out_channels": out_channels,
                    "static_channels": static_channels,
                    "drop_path_rate": drop_path_rate,
                    "use_hybrid_attention": self.use_hybrid_attention,
                    "use_multi_scale": self.use_multi_scale,
                    "use_cross_attention": self.use_cross_attention,
                    "use_position_encoding": self.use_position_encoding,
                    "use_timestep_conditioning": self.use_timestep_conditioning,
                    "use_lead_time_conditioning": self.use_lead_time_conditioning,
                    "use_variable_graph": self.use_variable_graph,
                    "use_dynamic_vg": self.use_dynamic_vg,
                    "use_static_vg": self.use_static_vg,
                    "use_proxy_conditioned_vg": self.use_proxy_conditioned_vg,
                    "vg_num_heads": self.vg_num_heads,
                    "use_static_morphology_encoder": self.use_static_morphology_encoder,
                    "use_morphology_graph": self.use_morphology_graph,
                    "use_wind_aware_graph": self.use_wind_aware_graph,
                    "use_urban_canopy_coupling": self.use_urban_canopy_coupling,
                    "use_urban_graph": self.use_urban_graph,
                }
            )
            self.net = build_model_from_config(model_init_cfg)

            # =========================
            # 3) 优化器/训练超参
            # =========================
            self.lr = float(optim_cfg.get("lr", 5e-4))
            self.max_epochs = int(optim_cfg.get("max_epochs", 200))
            self.weight_decay = float(optim_cfg.get("weight_decay", 1e-4))

            # 通道加权损失
            weights_list = optim_cfg.get("loss_weights", [1.0] * out_channels)
            if len(weights_list) != out_channels:
                print(
                    f"[Warning] loss_weights 长度({len(weights_list)}) != out_channels({out_channels})，将使用全 1 权重。"
                )
                weights_list = [1.0] * out_channels
            self.register_buffer("loss_weights", torch.tensor(weights_list, dtype=torch.float32).view(1, -1, 1, 1))

            # =========================
            # 4) 扩散参数
            # =========================
            self.P_mean = float(diffusion_cfg.get("P_mean", -0.8))
            self.P_std = float(diffusion_cfg.get("P_std", 0.8))

            # =========================
            # 5) 物理/频域/梯度损失权重
            # =========================
            # v3 约定：三类损失权重彼此独立（不再把 lambda_fft/lambda_grad 再乘到 lambda_phys 里）
            self.lambda_phys = float(physics_cfg.get("lambda", 0.1))
            self.lambda_fft = float(physics_cfg.get("lambda_fft", 0.0))
            self.lambda_grad = float(physics_cfg.get("lambda_grad", 0.0))
            self.lambda_adv = float(physics_cfg.get("lambda_adv", physics_cfg.get("lambda_adversarial", 0.0)))

            # 物理约束子项权重
            self.w_tp = float(physics_cfg.get("w_tp_nonneg", 1.0))
            self.w_tcc = float(physics_cfg.get("w_tcc_01", 0.5))
            self.w_dew = float(physics_cfg.get("w_dew_leq_t", 0.5))
            self.w_sp = float(physics_cfg.get("w_sp_nonneg", 0.1))
            self.w_div = float(physics_cfg.get("w_wind_div", 0.1))

            # FFT/梯度损失通道权重
            fft_list = physics_cfg.get("fft_channel_weights", [1.0] * out_channels)
            if len(fft_list) != out_channels:
                print(
                    f"[Warning] fft_channel_weights 长度({len(fft_list)}) != out_channels({out_channels})，将使用全 1 权重。"
                )
                fft_list = [1.0] * out_channels
            self.register_buffer(
                "fft_channel_weights",
                torch.tensor(fft_list, dtype=torch.float32).view(1, -1, 1, 1),
                persistent=False,
            )

            grad_list = physics_cfg.get("grad_channel_weights", [1.0] * out_channels)
            if len(grad_list) != out_channels:
                print(
                    f"[Warning] grad_channel_weights 长度({len(grad_list)}) != out_channels({out_channels})，将使用全 1 权重。"
                )
                grad_list = [1.0] * out_channels
            self.register_buffer(
                "grad_channel_weights",
                torch.tensor(grad_list, dtype=torch.float32).view(1, -1, 1, 1),
                persistent=False,
            )

            # 评估用（不加权）MSE
            self.mse = nn.MSELoss()
            self.var_names = metrics_cfg.get("var_names", [str(i) for i in range(out_channels)])
            self.acc_cfg = dict(metrics_cfg.get("acc", {}) or {})
            self.acc_enabled = bool(self.acc_cfg.get("enabled", False))
            self.acc_climatology = str(self.acc_cfg.get("climatology", "train_mean_field")).lower()
            self.acc_weighting = str(self.acc_cfg.get("weighting", "coslat")).lower()
            self.acc_center = bool(self.acc_cfg.get("center", True))
            self.acc_log_per_channel = bool(self.acc_cfg.get("log_per_channel", True))

            # Sobel 算子（用于梯度损失）
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            self.physical_consistency_cfg = PhysicalConsistencyConfig.from_mapping(physics_cfg)

            # =========================
            # 6) 可选模块：自适应权重
            # =========================
            if self.use_adaptive_weights:
                # 4 项：recon / phys_spatial / fft / grad
                self.adaptive_weight_net = AdaptiveWeightNet(num_weights=4, hidden_dim=64)
                print("✅ 启用自适应权重学习 (recon/phys/fft/grad)")

            # =========================
            # 7) 可选模块：对抗训练
            # =========================
            self.discriminator: Optional[Discriminator] = None
            self.d_optimizer: Optional[torch.optim.Optimizer] = None
            if self.use_adversarial:
                self.discriminator = Discriminator(in_channels=out_channels, hidden_dim=64)
                if self.lambda_adv <= 0.0:
                    # 若用户仅开关打开但未给权重，给一个温和默认值避免“开了但没效果”
                    self.lambda_adv = 0.05
                print(f"✅ 启用对抗训练 (lambda_adv={self.lambda_adv})")

            # =========================
            # 8) 可选模块：知识蒸馏
            # =========================
            self.teacher_model: Optional[UrbanPiDiT] = None
            if self.use_distillation:
                self.distill_temperature = float(physics_cfg.get("distill_temperature", 1.0))
                self.distill_alpha = float(physics_cfg.get("distill_alpha", 0.5))
                teacher_ckpt = physics_cfg.get("teacher_ckpt", physics_cfg.get("teacher_checkpoint", None))
                if teacher_ckpt is None:
                    raise ValueError(
                        "已开启 use_distillation，但未在 physics 配置中提供 teacher_ckpt/teacher_checkpoint。"
                    )
                self.teacher_model = self._load_teacher_model(
                    teacher_ckpt,
                    teacher_model_cfg=physics_cfg.get("teacher_model_cfg", None),
                    fallback_model_cfg=model_cfg,
                )
                print(f"✅ 启用知识蒸馏: teacher_ckpt={teacher_ckpt}")

            # =========================
            # 9) 可选模块：硬物理投影
            # =========================
            self.projection_steps = int(physics_cfg.get("projection_steps", 5))
            self.projection_lr = float(physics_cfg.get("projection_lr", 0.1))
            if self.use_hard_physics:
                print(f"✅ 启用硬物理约束投影: steps={self.projection_steps}, lr={self.projection_lr}")

        def _move_static_batch(self, batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
            needs_static = bool(
                getattr(self, "use_static_morphology_encoder", False)
                or getattr(self, "use_morphology_graph", False)
                or getattr(self, "use_wind_aware_graph", False)
                or getattr(self, "use_urban_canopy_coupling", False)
                or getattr(self, "use_micromet_coupling", False)
                or getattr(self, "use_process_proxy_encoder", False)
                or getattr(self, "use_process_adaln", False)
                or getattr(self, "use_urban_control_branch", False)
                or getattr(self, "use_anisotropic_process_graph", False)
                or getattr(self, "use_morphology_residual_head", False)
                or getattr(getattr(self, "physical_consistency_cfg", None), "use_process_consistency", False)
            )
            if not needs_static:
                return {"static_raw": None, "static_cont": None, "static_cat": None, "hour_of_day": None, "latest_state": None}

            out: Dict[str, Optional[torch.Tensor]] = {}
            for key in ("static_raw", "static_cont", "static_cat", "hour_of_day"):
                value = batch.get(key, None)
                out[key] = value.to(self.device) if isinstance(value, torch.Tensor) else None
            out["latest_state"] = None
            return out

        @staticmethod
        def _model_static_kwargs(static_batch: Optional[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Optional[torch.Tensor]]:
            """只保留模型 forward 支持的静态字段。"""

            static_batch = static_batch or {}
            return {key: static_batch.get(key) for key in ("static_raw", "static_cont", "static_cat", "hour_of_day")}

        def _attach_latest_state(
            self,
            static_batch: Optional[Dict[str, Optional[torch.Tensor]]],
            x_ctx: torch.Tensor,
            *,
            mean: torch.Tensor,
            std: torch.Tensor,
        ) -> Dict[str, Optional[torch.Tensor]]:
            """把历史最后一帧反归一化后加入过程一致性上下文。"""

            out = dict(static_batch or {})
            C = int(getattr(self.net, "in_channels", self.hparams["model_cfg"].get("in_channels", 1)))
            k = int(getattr(self.net, "real_k", max(1, (x_ctx.size(1) // max(C, 1)))))
            indices = [i * k + (k - 1) for i in range(C)]
            if indices and max(indices) < x_ctx.size(1):
                idx = torch.tensor(indices, device=x_ctx.device, dtype=torch.long)
                latest_norm = x_ctx.index_select(1, idx)
                out["latest_state"] = latest_norm * std + mean
            else:
                out["latest_state"] = None
            return out

        def _log_forward_diagnostics(
            self,
            diagnostics: Optional[Dict[str, Any]],
            *,
            split: str,
            on_step: bool,
            on_epoch: bool,
        ) -> None:
            """把模型 forward 诊断中的稳定标量写入 Lightning 日志。"""

            if not diagnostics:
                return
            logs = collect_model_diagnostics(
                diagnostics,
                log_prefix=f"{split}/diagnostics",
                device=self.device,
            )
            for key, value in logs.items():
                self.log(
                    key,
                    value,
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )

        # -------------------------
        # 推理配置
        # -------------------------
        def _get_infer_cfg(self) -> Dict[str, Any]:
            cfg = dict(self.hparams.get("inference_cfg") or {})
            if "mode" not in cfg:
                cfg["mode"] = cfg.get("test_mode", "deterministic")
            return cfg

        # -------------------------
        # 时间采样
        # -------------------------
        def sample_t(self, n: int) -> torch.Tensor:
            """采样 t ~ sigmoid(N(P_mean, P_std^2))。"""

            z = torch.randn(n, device=self.device) * self.P_std + self.P_mean
            return torch.sigmoid(z)

        # -------------------------
        # 推理：单样本/集成
        # -------------------------
        @torch.no_grad()
        def predict_one(
            self,
            x_ctx: torch.Tensor,
            *,
            steps: int,
            t_start: float,
            t_end: float,
            seed: Optional[int] = None,
            lead_time: Optional[torch.Tensor] = None,
            static_raw: Optional[torch.Tensor] = None,
            static_cont: Optional[torch.Tensor] = None,
            static_cat: Optional[torch.Tensor] = None,
            hour_of_day: Optional[torch.Tensor] = None,
            return_diagnostics: bool = False,
        ) -> torch.Tensor:
            """单样本预测。

            Args:
                x_ctx: [B, ctx_C, H, W]
                steps: 采样步数
                t_start/t_end: 采样时间范围
                seed: 若提供，则固定初始噪声随机种子（保证可复现）
            """

            gen = None
            if seed is not None:
                gen = torch.Generator(device=x_ctx.device)
                gen.manual_seed(int(seed))
            return self.generate_x0(
                x_ctx,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                generator=gen,
                lead_time=lead_time,
                static_raw=static_raw,
                static_cont=static_cont,
                static_cat=static_cat,
                hour_of_day=hour_of_day,
                return_diagnostics=return_diagnostics,
            )

        @torch.no_grad()
        def predict_ensemble(
            self,
            x_ctx: torch.Tensor,
            *,
            ensemble_size: int,
            base_seed: int,
            steps: int,
            t_start: float,
            t_end: float,
            seed_offset: int = 0,
            lead_time: Optional[torch.Tensor] = None,
            static_raw: Optional[torch.Tensor] = None,
            static_cont: Optional[torch.Tensor] = None,
            static_cat: Optional[torch.Tensor] = None,
            hour_of_day: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """集成预测。

            Returns:
                ensemble: [B, M, C, H, W]
            """

            M = int(ensemble_size)
            preds = []
            for m in range(M):
                seed = int(base_seed) + int(seed_offset) + m
                gen = torch.Generator(device=x_ctx.device)
                gen.manual_seed(seed)
                preds.append(
                    self.generate_x0(
                        x_ctx,
                        steps=steps,
                        t_start=t_start,
                        t_end=t_end,
                        generator=gen,
                        lead_time=lead_time,
                        static_raw=static_raw,
                        static_cont=static_cont,
                        static_cat=static_cat,
                        hour_of_day=hour_of_day,
                    )
                )
            return torch.stack(preds, dim=1)

        @torch.no_grad()
        def generate_x0(
            self,
            x_ctx: torch.Tensor,
            steps: int = 8,
            t_start: float = 0.0,
            t_end: float = 1.0,
            generator: Optional[torch.Generator] = None,
            lead_time: Optional[torch.Tensor] = None,
            static_raw: Optional[torch.Tensor] = None,
            static_cont: Optional[torch.Tensor] = None,
            static_cat: Optional[torch.Tensor] = None,
            hour_of_day: Optional[torch.Tensor] = None,
            return_diagnostics: bool = False,
        ) -> torch.Tensor:
            """扩散式生成未来场 x0（推理/预测）。

            关键点：
            - 推理阶段**不使用** GT 的 x0 构造输入，避免评估泄露(label leakage)。
            - 使用与训练一致的插值噪声形式：x_t = t*x0 + (1-t)*eps。
            - 从纯噪声开始，采用 DDIM 风格确定性更新逐步推进到 t_end≈1。
            """

            device = x_ctx.device
            B, _, H, W = x_ctx.shape
            C = int(getattr(self.net, "out_channels", self.hparams["model_cfg"].get("out_channels", 1)))

            steps = max(int(steps), 1)
            ts = torch.linspace(float(t_start), float(t_end), steps + 1, device=device, dtype=torch.float32)

            # 初始噪声
            if generator is None:
                x_t = torch.randn((B, C, H, W), device=device, dtype=x_ctx.dtype)
            else:
                x_t = torch.randn((B, C, H, W), device=device, dtype=x_ctx.dtype, generator=generator)

            for i in range(steps):
                t_cur = ts[i].expand(B)
                # 注意：lead_time 是“预测提前期”条件，不同于扩散时间步 t_cur。
                x0_pred = self.net(
                    x_t,
                    x_ctx,
                    t_cur,
                    lead_time=lead_time,
                    static_raw=static_raw,
                    static_cont=static_cont,
                    static_cat=static_cat,
                    hour_of_day=hour_of_day,
                    return_diagnostics=bool(return_diagnostics and i == steps - 1),
                )
                if isinstance(x0_pred, tuple):
                    x0_pred, _ = x0_pred

                # eps_hat = (x_t - t*x0) / (1-t)
                denom = (1.0 - ts[i]).clamp(min=1e-5)
                eps_hat = (x_t - ts[i].view(1, 1, 1, 1) * x0_pred) / denom.view(1, 1, 1, 1)

                t_next = ts[i + 1]
                x_t = t_next.view(1, 1, 1, 1) * x0_pred + (1.0 - t_next).view(1, 1, 1, 1) * eps_hat

            # 当 t_end=1 时，x_t 就是对 x0 的预测
            return x_t

        # -------------------------
        # 多提前期（multi-horizon）辅助函数
        # -------------------------
        def _lead_steps_to_hours(self, lead_steps: int) -> float:
            """将提前期的“步数”转换为小时数。

            说明：数据为 ERA5 6h，因此默认 1 step=6h；但这里做成可配置，便于未来扩展。
            """

            return float(lead_steps) * float(self.time_step_hours)

        def _lead_tag(self, lead_steps: int) -> str:
            """用于日志键的 lead 标记，例如：6h/12h/24h。"""

            hours = self._lead_steps_to_hours(int(lead_steps))
            if abs(hours - round(hours)) < 1e-6:
                return f"{int(round(hours))}h"
            return f"{hours:.1f}h"

        def _normalize_lead_time(self, lead_steps: torch.Tensor) -> torch.Tensor:
            """将 lead_steps 归一化到 [0,1]（用于 lead-time conditioning）。"""

            denom = float(max(int(self.max_lead_time_steps), 1))
            return lead_steps.float() / denom

        def _anneal_alpha(self, *, warmup_epochs: int, anneal_epochs: int, schedule: str = "linear") -> float:
            """计算一个 [0,1] 的退火系数 alpha。

            说明：
            - alpha=0 表示处于“退火前/起始阶段”，alpha=1 表示退火到“目标阶段”。
            - 常用于：lead 权重退火 / rollout 步数课程学习 / teacher forcing 比例退火。
            """

            warmup_epochs = max(int(warmup_epochs), 0)
            anneal_epochs = int(anneal_epochs)

            # anneal_epochs<=0 视为“直接到终态”
            if anneal_epochs <= 0:
                return 1.0

            e = max(0, int(self.current_epoch) - warmup_epochs)
            p = min(1.0, float(e) / float(max(anneal_epochs, 1)))

            schedule = str(schedule).lower()
            if schedule in ("cos", "cosine"):
                # 0 -> 1 的 cosine ease-in
                return float(0.5 - 0.5 * math.cos(math.pi * p))
            return float(p)

        def _lead_loss_weight_base(
            self,
            lead_steps: int,
            lead_list: Optional[list],
            *,
            weighting: str,
            beta: float,
            custom_weights: Optional[Any] = None,
        ) -> float:
            """lead loss 权重的基础实现（不含退火）。"""

            lead_step = max(int(lead_steps), 1)
            weighting = str(weighting).lower()

            # 1) 自定义权重：与 lead_list 对齐
            if weighting == "custom" and custom_weights is not None and lead_list is not None:
                try:
                    weights = list(custom_weights)
                    if len(weights) == len(lead_list) and lead_step in lead_list:
                        return float(weights[int(lead_list.index(lead_step))])
                except Exception:
                    # 任何异常都回退到内置策略
                    pass

            # 2) 内置策略
            if weighting == "uniform":
                return 1.0
            if weighting == "inverse":
                return 1.0 / float(lead_step)
            if weighting == "inverse_sqrt":
                return 1.0 / float(lead_step) ** 0.5
            if weighting == "exp_decay":
                # 以最短 lead 为 1.0，其余按指数衰减
                min_lead_step = 1
                if lead_list is not None and len(lead_list) > 0:
                    try:
                        min_lead_step = int(min(int(x) for x in lead_list))
                    except Exception:
                        min_lead_step = 1
                beta = float(beta)
                return float(math.exp(-beta * float(lead_step - min_lead_step)))

            # 默认回退：uniform
            return 1.0

        def _lead_loss_weight(self, lead_steps: int, lead_list: Optional[list]) -> float:
            """返回某个 lead 的 loss 权重（float）。

            目的：
            - 多提前期训练时，长提前期的误差通常更大，如果不做权重，梯度会被长提前期主导，
              常见副作用是 6h 变差而 24h 变好。
            - 通过对长 lead 降权，可以显著缓解 6h 退化。

            Args:
                lead_steps: 提前期（单位：步）
                lead_list: 当前 batch 的 lead 列表（单位：步），用于 custom 权重对齐
            """

            # v4：可选的“权重退火”
            if not bool(getattr(self, "lead_weight_anneal_enabled", False)):
                return float(
                    self._lead_loss_weight_base(
                        lead_steps,
                        lead_list,
                        weighting=self.lead_loss_weighting,
                        beta=self.lead_weight_beta,
                        custom_weights=self.lead_loss_weights,
                    )
                )

            alpha = self._anneal_alpha(
                warmup_epochs=int(getattr(self, "lead_weight_anneal_warmup_epochs", 0)),
                anneal_epochs=int(getattr(self, "lead_weight_anneal_epochs", 0)),
                schedule=str(getattr(self, "lead_weight_anneal_schedule", "linear")),
            )

            start_w = str(getattr(self, "lead_weight_anneal_start", self.lead_loss_weighting)).lower()
            end_w = str(getattr(self, "lead_weight_anneal_end", "uniform")).lower()

            start_beta = float(getattr(self, "lead_weight_anneal_start_beta", self.lead_weight_beta))
            end_beta = float(getattr(self, "lead_weight_anneal_end_beta", 0.0))

            start_custom = (
                getattr(self, "lead_weight_anneal_start_weights", None)
                if start_w == "custom"
                else self.lead_loss_weights
            )
            end_custom = (
                getattr(self, "lead_weight_anneal_end_weights", None)
                if end_w == "custom"
                else self.lead_loss_weights
            )

            w0 = self._lead_loss_weight_base(
                lead_steps,
                lead_list,
                weighting=start_w,
                beta=start_beta,
                custom_weights=start_custom,
            )
            w1 = self._lead_loss_weight_base(
                lead_steps,
                lead_list,
                weighting=end_w,
                beta=end_beta,
                custom_weights=end_custom,
            )

            return float((1.0 - alpha) * float(w0) + alpha * float(w1))

        @staticmethod
        def _extract_lead_list(batch: Dict[str, Any]) -> Optional[list]:
            """从 batch 中提取 lead_times 列表（单位：步）。"""

            lt = batch.get("lead_times", None)
            if lt is None:
                return None
            if isinstance(lt, torch.Tensor):
                if lt.ndim == 2:
                    lt = lt[0]
                return [int(x) for x in lt.detach().cpu().tolist()]
            # list/tuple
            try:
                return [int(x) for x in list(lt)]
            except Exception:
                return None

        def _teacher_forcing_ratio(self) -> float:
            """scheduled sampling 中 teacher forcing 的比例。

            - 比例越高：越多使用 GT 更新上下文（更稳定，但与推理分布差异更大）；
            - 比例越低：越多使用模型预测更新上下文（更贴近推理，但训练更难）。
            """

            start = float(getattr(self, "teacher_forcing_start", 1.0))
            end = float(getattr(self, "teacher_forcing_end", 0.0))
            decay_epochs = int(getattr(self, "teacher_forcing_decay_epochs", 0))

            # decay_epochs<=0：不做退火，固定使用 start
            if decay_epochs <= 0:
                return float(start)

            alpha = self._anneal_alpha(warmup_epochs=0, anneal_epochs=decay_epochs, schedule=str(getattr(self, "teacher_forcing_schedule", "linear")))
            ratio = start + (end - start) * float(alpha)
            return float(max(0.0, min(1.0, ratio)))

        def _current_rollout_steps(self, lead_list: Optional[list]) -> int:
            """根据 epoch 与配置，决定当前 batch 的 rollout 步数 K。

            说明：
            - 若未启用课程学习，K=rollout_max_steps；
            - 若启用 curriculum_epochs，则 K 从 1 逐步增长到 rollout_max_steps。
            """

            max_steps_cfg = max(int(getattr(self, "rollout_max_steps", 1)), 1)
            if lead_list is not None and len(lead_list) > 0:
                # 数据里未必提供到 max_steps_cfg，这里做下界约束
                max_steps_cfg = min(max_steps_cfg, int(max(int(x) for x in lead_list)))

            cur_epochs = int(getattr(self, "rollout_curriculum_epochs", 0))
            if cur_epochs <= 0:
                return int(max_steps_cfg)

            # 课程学习：K 从 1 -> max_steps_cfg
            alpha = self._anneal_alpha(warmup_epochs=0, anneal_epochs=cur_epochs, schedule=str(getattr(self, "rollout_schedule", "linear")))
            k = int(round(1 + float(alpha) * float(max_steps_cfg - 1)))
            return int(max(1, min(max_steps_cfg, k)))

        def _update_ctx_autoregressive(self, x_ctx: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
            """autoregressive rollout 时更新上下文。

            约定：x_ctx 的动态部分通道顺序为“变量优先”：
              [var0_t0, ..., var0_t{k-1}, var1_t0, ..., var{C-1}_t{k-1}, (static...)]

            Args:
                x_ctx: [B, ctx_C, H, W]
                x_next: 下一时刻预测 [B, C, H, W]（归一化空间）

            Returns:
                x_ctx_next: [B, ctx_C, H, W]
            """

            B, ctx_C, H, W = x_ctx.shape
            C = int(getattr(self.net, "in_channels", self.hparams["model_cfg"].get("in_channels", 1)))
            k = int(getattr(self.net, "real_k", max(1, (ctx_C // max(C, 1)))))

            dyn_C = C * k
            dyn = x_ctx[:, :dyn_C, :, :].reshape(B, C, k, H, W)
            stat = x_ctx[:, dyn_C:, :, :]  # 可能为空

            # shift：丢掉最早一帧，追加最新预测
            dyn = torch.cat([dyn[:, :, 1:, :, :], x_next.unsqueeze(2)], dim=2)
            dyn = dyn.reshape(B, dyn_C, H, W)

            if stat.numel() == 0:
                return dyn
            return torch.cat([dyn, stat], dim=1)

        def _log_deterministic_metrics(
            self,
            *,
            split: str,
            lead_steps: int,
            pred_denorm: torch.Tensor,
            gt_denorm: torch.Tensor,
            on_step: bool,
            on_epoch: bool,
            prog_bar: bool = False,
        ) -> None:
            """记录确定性指标（整体 + 分变量）到 logger。"""

            tag = self._lead_tag(int(lead_steps))
            mse_all = torch.mean((pred_denorm - gt_denorm) ** 2)
            rmse_all = torch.sqrt(mse_all)
            mae_all = torch.mean(torch.abs(pred_denorm - gt_denorm))
            bias_all = torch.mean(pred_denorm - gt_denorm)

            self.log(
                f"{split}/RMSE@{tag}",
                rmse_all,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
            )
            self.log(
                f"{split}/MAE@{tag}",
                mae_all,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{split}/Bias@{tag}",
                bias_all,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{split}/CRPS@{tag}",
                mae_all,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch,
            )

            var_rmse = torch.sqrt(torch.mean((pred_denorm - gt_denorm) ** 2, dim=(2, 3)))  # [B,C]
            var_mae = torch.mean(torch.abs(pred_denorm - gt_denorm), dim=(2, 3))  # [B,C]
            var_bias = torch.mean(pred_denorm - gt_denorm, dim=(2, 3))  # [B,C]
            C = pred_denorm.size(1)
            for i in range(C):
                name = self.var_names[i] if i < len(self.var_names) else str(i)
                self.log(
                    f"{split}/RMSE_{name}@{tag}",
                    var_rmse[:, i].mean(),
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{split}/MAE_{name}@{tag}",
                    var_mae[:, i].mean(),
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{split}/Bias_{name}@{tag}",
                    var_bias[:, i].mean(),
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{split}/CRPS_{name}@{tag}",
                    var_mae[:, i].mean(),
                    sync_dist=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )

        def _log_mae_rmse(
            self,
            *,
            split: str,
            lead_steps: int,
            pred_denorm: torch.Tensor,
            gt_denorm: torch.Tensor,
            on_step: bool,
            on_epoch: bool,
            prog_bar: bool = False,
        ) -> None:
            """兼容旧入口：记录确定性评估指标。"""

            self._log_deterministic_metrics(
                split=split,
                lead_steps=lead_steps,
                pred_denorm=pred_denorm,
                gt_denorm=gt_denorm,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
            )

        def _log_acc(
            self,
            *,
            split: str,
            lead_steps: int,
            pred_denorm: torch.Tensor,
            gt_denorm: torch.Tensor,
            clim_denorm: torch.Tensor,
            lat: Optional[torch.Tensor],
            on_step: bool,
            on_epoch: bool,
            prog_bar: bool = False,
        ) -> torch.Tensor:
            tag = self._lead_tag(int(lead_steps))
            acc_all = _acc(
                pred_denorm,
                gt_denorm,
                clim_denorm,
                lat=lat,
                center=bool(self.acc_center),
                reduction="mean",
            )
            self.log(
                f"{split}/ACC@{tag}",
                acc_all,
                sync_dist=True,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
            )

            if bool(self.acc_log_per_channel):
                acc_ch = _acc(
                    pred_denorm,
                    gt_denorm,
                    clim_denorm,
                    lat=lat,
                    center=bool(self.acc_center),
                    reduction="channel_mean",
                )
                C = pred_denorm.size(1)
                for i in range(C):
                    name = self.var_names[i] if i < len(self.var_names) else str(i)
                    self.log(
                        f"{split}/ACC_{name}@{tag}",
                        acc_ch[i],
                        sync_dist=True,
                        on_step=on_step,
                        on_epoch=on_epoch,
                    )

            return acc_all

        # -------------------------
        # 损失：重建/物理一致性
        # -------------------------
        def _calculate_weighted_recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """通道加权 MSE：mean( w * (pred-target)^2 )。"""

            diff2 = (pred - target) ** 2
            return (diff2 * self.loss_weights).mean()

        def _zero_physics_logs(self, x: torch.Tensor) -> Dict[str, Any]:
            """构造关闭辅助损失时的日志占位。"""

            zero = x.new_tensor(0.0)
            return {
                "phys_tp_nonneg": zero,
                "phys_tcc_01": zero,
                "phys_dew_leq_t": zero,
                "phys_sp_nonneg": zero,
                "phys_wind_div": zero,
                "phys_feasibility": zero,
                "phys_process_consistency": zero,
                "phys_structure": zero,
                "phys_spatial": zero,
                "fft": zero,
                "grad": zero,
                "loss/feasibility_tp_nonneg": zero,
                "loss/feasibility_tcc_01": zero,
                "loss/feasibility_dew_leq_t": zero,
                "loss/feasibility_sp_nonneg": zero,
                "loss/feasibility_total": zero,
                "loss/structure_fft": zero,
                "loss/structure_grad": zero,
                "loss/structure_wind_div": zero,
                "loss/structure_total": zero,
                "loss/process_rh": zero,
                "loss/process_drag": zero,
                "loss/process_diurnal": zero,
                "loss/process_proxy_roughness_wind_decay": zero,
                "loss/process_proxy_diurnal_range_by_building": zero,
                "loss/process_proxy_impervious_d2m_corr": zero,
                "loss/process_proxy_wind_tcc_spread": zero,
                "loss/process_proxy_total": zero,
                "loss/process_total": zero,
                "diag/drag_available": zero,
                "diag/high_drag_delta_speed": zero,
                "diag/low_drag_delta_speed": zero,
                "diag/drag_consistency_gap": zero,
                "diag/diurnal_available": zero,
                "diag/daytime_heat_response_gap": zero,
                "diag/nighttime_heat_release_gap": zero,
                "physical_feasibility_score": zero,
                "structure_score": zero,
                "process_consistency_score": zero,
            }

        def _calculate_physics_losses(
            self,
            x0_pred: torch.Tensor,
            x0_gt: torch.Tensor,
            std: torch.Tensor,
            mean: torch.Tensor,
            static_batch: Optional[Dict[str, Optional[torch.Tensor]]] = None,
            x_ctx: Optional[torch.Tensor] = None,
        ) -> Dict[str, Any]:
            """计算物理一致性损失包。"""

            calculator = PhysicalConsistencyLoss(
                self.physical_consistency_cfg,
                fft_channel_weights=self.fft_channel_weights,
                grad_channel_weights=self.grad_channel_weights,
                sobel_x=self.sobel_x,
                sobel_y=self.sobel_y,
            )
            static_batch = static_batch or {}
            if x_ctx is not None:
                static_batch = self._attach_latest_state(static_batch, x_ctx, mean=mean, std=std)
            proxy_fields = getattr(self.net, "last_process_proxy_fields", None)
            return calculator(
                x0_pred,
                x0_gt,
                std=std,
                mean=mean,
                static_raw=static_batch.get("static_raw"),
                static_cont=static_batch.get("static_cont"),
                static_cat=static_batch.get("static_cat"),
                hour_of_day=static_batch.get("hour_of_day"),
                latest_state=static_batch.get("latest_state"),
                proxy_fields=proxy_fields,
            )

        # -------------------------
        # 硬物理约束投影
        # -------------------------


        def _apply_hard_physics_constraints(
            self,
            x0_pred_denorm: torch.Tensor,
            num_steps: int = 5
        ) -> torch.Tensor:
            """对预测做硬物理约束投影（推荐：闭式投影 + STE，不断梯度）。

            Args:
                x0_pred_denorm: [B, C, H, W]，已经是反归一化（物理量空间）的预测
                num_steps: 保留接口兼容（不再用于 PGD 内循环）。如果你们后续确实要做 PGD，可再扩展。

            Returns:
                x_hard: [B, C, H, W]
                    - 前向值：等于投影后的结果（严格满足硬约束）
                    - 反向梯度：等价于 identity（梯度从原 x0_pred_denorm 回传，不会断图）
            """

            x = x0_pred_denorm  # 原预测（需要保留计算图）
            proj = x.clone()    # 投影后的结果（用于前向输出）

            # -------------------------
            # 通道定义（按你们的顺序）
            # dynamic_vars: [d2m, sp, t2m, tcc, tp, u10, v10]
            # idx:            0    1   2    3    4   5    6
            # -------------------------

            # 1) tcc ∈ [0,1]：强烈建议硬约束（物理意义稳定，且不易伤 RMSE）
            proj[:, 3] = proj[:, 3].clamp(0.0, 1.0)

            # 2) d2m ≤ t2m：强烈建议硬约束（物理关系清晰）
            #    注意：torch.minimum 是分段可导的；用 STE 后反向更稳。
            proj[:, 0] = torch.minimum(proj[:, 0], proj[:, 2])

            # 3) tp ≥ 0：谨慎使用（仅当 tp 在“真实降水量空间”时才建议硬约束）
            #    如果你们 tp 反归一化后确实是 mm/h 之类的物理量，relu 是合理的。
            #    如果 tp 仍处于 log 空间/标准化空间，直接 relu 可能引入偏差。
            enable_tp_hard = True
            # 如果你们希望通过配置控制，可改成：
            # enable_tp_hard = bool(self.cfg.get("ablation", {}).get("hard_physics_tp", False))

            if enable_tp_hard and proj.shape[1] > 4:
                proj[:, 4] = F.relu(proj[:, 4])

            # 4) sp ≥ 0：通常不推荐做硬约束（气压本身总为正，且若 sp 标准化会出错）
            #    默认关闭，除非你们确认 sp_denorm 是真实气压值且确实会预测成负数。
            enable_sp_hard = False
            # enable_sp_hard = bool(self.cfg.get("ablation", {}).get("hard_physics_sp", False))

            if enable_sp_hard and proj.shape[1] > 1:
                proj[:, 1] = F.relu(proj[:, 1])

            # -------------------------
            # STE：Straight-Through Estimator（关键）
            # 前向：输出 proj（硬约束结果）
            # 反向：梯度从 x 回传（避免断图 / loss 不需要 grad）
            # -------------------------
            x_hard = x + (proj - x).detach()
            return x_hard



        # -------------------------
        # 教师模型加载（蒸馏）
        # -------------------------
        def _load_teacher_model(
            self,
            ckpt_path: str,
            *,
            teacher_model_cfg: Optional[Dict[str, Any]] = None,
            fallback_model_cfg: Optional[Dict[str, Any]] = None,
        ) -> UrbanPiDiT:
            """从 checkpoint 加载教师 UrbanPiDiT，并冻结其参数。"""

            ckpt_path = os.fspath(ckpt_path)
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"teacher_ckpt 不存在: {ckpt_path}")

            cfg = dict(teacher_model_cfg or fallback_model_cfg or {})
            if not cfg:
                raise ValueError("无法构建教师模型：teacher_model_cfg 与 fallback_model_cfg 均为空")

            teacher = UrbanPiDiT(
                H=int(cfg.get("H", 8)),
                W=int(cfg.get("W", 8)),
                in_channels=int(cfg.get("in_channels", 7)),
                ctx_channels=int(cfg.get("ctx_channels", 28)),
                out_channels=int(cfg.get("out_channels", 7)),
                D=int(cfg.get("D", 512)),
                depth=int(cfg.get("depth", 12)),
                heads=int(cfg.get("heads", 8)),
                mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
                static_channels=int(cfg.get("static_channels", 0)),
                drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
                dropout=float(cfg.get("dropout", 0.0)),
                use_hybrid_attention=bool(cfg.get("use_hybrid_attention", False)),
                use_multi_scale=bool(cfg.get("use_multi_scale", False)),
                use_cross_attention=bool(cfg.get("use_cross_attention", True)),
                use_position_encoding=bool(cfg.get("use_position_encoding", True)),
                use_timestep_conditioning=bool(cfg.get("use_timestep_conditioning", True)),
                # 教师模型也可能启用了 lead time conditioning；若未启用则保持默认 False
                use_lead_time_conditioning=bool(cfg.get("use_lead_time_conditioning", False)),
                use_variable_graph=bool(cfg.get("use_variable_graph", False)),
                use_dynamic_vg=bool(cfg.get("use_dynamic_vg", cfg.get("use_variable_graph", False))),
                use_static_vg=bool(cfg.get("use_static_vg", cfg.get("use_variable_graph", False))),
                use_proxy_conditioned_vg=bool(cfg.get("use_proxy_conditioned_vg", False)),
                vg_num_heads=int(cfg.get("vg_num_heads", 4)),
                use_static_morphology_encoder=bool(cfg.get("use_static_morphology_encoder", False)),
                use_urban_graph=bool(cfg.get("use_urban_graph", False)),
                urban_graph_k=int(cfg.get("urban_graph_k", 8)),
                urban_graph_sigma=float(cfg.get("urban_graph_sigma", 1.0)),
                static_vars=tuple(cfg.get("static_vars", []) or []),
                static_schema=dict(cfg.get("static_schema", {}) or {}),
            )

            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            elif isinstance(state, dict):
                sd = state
            else:
                raise ValueError("teacher_ckpt 不是可识别的 state_dict/checkpoint 格式")

            # Lightning ckpt 常见前缀为 "net."，这里做兼容处理
            if any(k.startswith("net.") for k in sd.keys()):
                sd_net = {k[len("net.") :]: v for k, v in sd.items() if k.startswith("net.")}
            else:
                sd_net = sd

            missing, unexpected = teacher.load_state_dict(sd_net, strict=False)
            if missing:
                print(f"[Teacher] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if unexpected:
                print(f"[Teacher] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            return teacher

        # -------------------------
        # 训练/验证/测试 step
        # -------------------------
        def training_step(self, batch: Dict[str, Any], batch_idx: int):
            """训练 step。

            v4 改进要点（针对你们反馈的现象）：
            - 支持多提前期训练的多种 objective（single_lead / lead1_plus_k / weighted_sum_all）；
            - 支持 lead loss 权重（uniform / inverse / exp_decay / custom），避免长 lead 梯度主导拖累 6h；
            - 默认只对最短 lead 施加物理/FFT/梯度等辅助损失（可配置开关扩展到所有 lead）。

            说明：
            - 当 use_lead_time_conditioning=true 且 batch 含 y=[B,L,C,H,W] 时启用多提前期训练；
            - 否则退化为原版单提前期训练（完全兼容）。
            """

            x_ctx = batch["x_ctx"].to(self.device)
            static_batch = self._move_static_batch(batch)
            norm = batch["norm"]
            mean = norm["mean"].to(self.device)
            std = norm["std"].to(self.device)
            static_batch = self._attach_latest_state(static_batch, x_ctx, mean=mean, std=std)

            # -------------------------
            # 1) 解析多提前期 GT
            # -------------------------
            lead_list = self._extract_lead_list(batch)  # e.g. [1,2,3,4]

            # v4：
            # - direct multi-lead 训练需要 use_lead_time_conditioning=true
            # - rollout fine-tuning 不强制 lead conditioning，只要 batch 提供 y(多提前期 GT) 即可
            multi_horizon_train = ("y" in batch) and (self.rollout_finetune_enabled or bool(self.use_lead_time_conditioning))
            if multi_horizon_train:
                # y_futures: [B, L, C, H, W]
                y_futures = batch["y"].to(self.device)
                B, L = int(y_futures.shape[0]), int(y_futures.shape[1])

                if lead_list is None:
                    lead_list = list(range(1, L + 1))

                # -------------------------
                # v4：多步 rollout 微调（优先级最高）
                # -------------------------
                if bool(getattr(self, "rollout_finetune_enabled", False)):
                    return self._training_step_rollout_finetune(
                        x_ctx=x_ctx,
                        y_futures=y_futures,
                        lead_list=lead_list,
                        mean=mean,
                        std=std,
                        static_batch=static_batch,
                    )

                # curriculum：先学短 lead，再逐步放开长 lead（可选）
                allowed = list(range(L))
                if self.train_lead_time_sampling == "curriculum" and self.curriculum_epochs > 0 and len(lead_list) > 0:
                    min_l, max_l = int(min(lead_list)), int(max(lead_list))
                    progress = min(1.0, float(self.current_epoch) / float(max(self.curriculum_epochs, 1)))
                    threshold = int(round(min_l + progress * (max_l - min_l)))
                    allowed = [i for i, lt in enumerate(lead_list) if int(lt) <= threshold]
                    if len(allowed) == 0:
                        allowed = [0]

                # 最短 lead（通常是 1）
                short_lead = int(min(int(x) for x in lead_list))
                short_idx = int(lead_list.index(short_lead)) if short_lead in lead_list else 0

                # -------------------------
                # 2) 选择本 batch 参与训练的 lead 集合
                # -------------------------
                obj = str(self.train_objective).lower()

                if obj == "weighted_sum_all":
                    selected = list(allowed)
                elif obj in ("lead1_plus_k", "lead1_plus", "lead1_plus_random"):
                    selected = []
                    if self.force_include_short_lead and (short_idx in range(L)):
                        selected.append(short_idx)
                    # 从允许集合里采样额外 lead（排除 short_idx）
                    candidates = [i for i in allowed if i != short_idx]
                    k_extra = max(int(self.extra_leads_per_batch), 0)
                    if k_extra > 0 and len(candidates) > 0:
                        # 不放回采样
                        perm = torch.randperm(len(candidates), device=self.device).tolist()
                        pick = [candidates[i] for i in perm[: min(k_extra, len(candidates))]]
                        selected.extend(pick)
                    if len(selected) == 0:
                        selected = [short_idx]
                else:
                    # 默认：single_lead
                    if len(allowed) == 0:
                        allowed = [0]
                    pick = int(torch.randint(0, len(allowed), (1,), device=self.device).item())
                    selected = [int(allowed[pick])]

                # 去重并保持顺序
                seen = set()
                selected = [x for x in selected if not (x in seen or seen.add(x))]

                # -------------------------
                # 3) 扩散训练：对 selected leads 做加权求和
                # -------------------------
                # 同一个 batch 共享 t 与 eps（减少额外随机性，训练更稳定）
                t = self.sample_t(B)
                eps = torch.randn_like(y_futures[:, 0])

                loss_sum = torch.tensor(0.0, device=self.device)
                w_sum = 0.0

                # 仅用于日志：我们把“主 lead”定义为 short_lead（最短提前期）
                main_lead_steps = int(short_lead)
                main_pred_denorm: Optional[torch.Tensor] = None
                main_gt_denorm: Optional[torch.Tensor] = None
                main_phys_logs: Optional[Dict[str, Any]] = None
                main_loss_recon = torch.tensor(0.0, device=self.device)
                main_loss_phys = torch.tensor(0.0, device=self.device)
                main_loss_feasibility = torch.tensor(0.0, device=self.device)
                main_loss_process_consistency = torch.tensor(0.0, device=self.device)
                main_loss_structure = torch.tensor(0.0, device=self.device)
                main_loss_fft = torch.tensor(0.0, device=self.device)
                main_loss_grad = torch.tensor(0.0, device=self.device)
                main_forward_diagnostics: Optional[Dict[str, Any]] = None

                for lead_idx in selected:
                    lead_idx = int(lead_idx)
                    lead_steps = int(lead_list[lead_idx])

                    x0 = y_futures[:, lead_idx]
                    x_t = t.view(-1, 1, 1, 1) * x0 + (1.0 - t.view(-1, 1, 1, 1)) * eps

                    # lead_time conditioning：传入归一化后的 [B]
                    lt_steps_t = torch.full((B,), float(lead_steps), device=self.device, dtype=torch.float32)
                    lead_cond = self._normalize_lead_time(lt_steps_t)

                    x0_pred = self.net(
                        x_t,
                        x_ctx,
                        t,
                        lead_time=lead_cond,
                        return_diagnostics=bool(lead_steps == main_lead_steps),
                        **self._model_static_kwargs(static_batch),
                    )
                    if isinstance(x0_pred, tuple):
                        x0_pred, forward_diagnostics = x0_pred
                    else:
                        forward_diagnostics = None

                    # 是否对该 lead 施加辅助损失（物理/FFT/梯度/硬约束）
                    apply_aux = self.apply_aux_losses_to_all_leads or (lead_steps == main_lead_steps)

                    # ---- 硬物理约束投影（可选） ----
                    if apply_aux and self.use_hard_physics:
                        x0_pred_denorm_tmp = x0_pred * std + mean
                        x0_pred_denorm_tmp = self._apply_hard_physics_constraints(
                            x0_pred_denorm_tmp, num_steps=self.projection_steps
                        )
                        x0_pred = (x0_pred_denorm_tmp - mean) / std

                    # ---- 重建损失 ----
                    if self.use_channel_weights:
                        loss_recon = self._calculate_weighted_recon_loss(x0_pred, x0)
                    else:
                        loss_recon = F.mse_loss(x0_pred, x0)

                    # ---- 物理/频域/梯度损失（可选） ----
                    if apply_aux:
                        phys = self._calculate_physics_losses(x0_pred, x0, std, mean, static_batch, x_ctx=x_ctx)
                        x0_pred_denorm = phys["x0_pred_denorm"]
                        loss_phys_spatial = phys["loss_phys_spatial"]
                        loss_feasibility = phys["loss_feasibility"]
                        loss_process_consistency = phys["loss_process_consistency"]
                        loss_structure = phys["loss_structure"]
                        loss_fft = phys["loss_fft"]
                        loss_grad = phys["loss_grad"]
                        phys_logs = phys["logs"]
                    else:
                        x0_pred_denorm = x0_pred * std + mean
                        loss_phys_spatial = torch.tensor(0.0, device=self.device)
                        loss_feasibility = torch.tensor(0.0, device=self.device)
                        loss_process_consistency = torch.tensor(0.0, device=self.device)
                        loss_structure = torch.tensor(0.0, device=self.device)
                        loss_fft = torch.tensor(0.0, device=self.device)
                        loss_grad = torch.tensor(0.0, device=self.device)
                        phys_logs = self._zero_physics_logs(x0_pred)

                    # ---- 组合总损失（两种模式） ----
                    if self.use_adaptive_weights and hasattr(self, "adaptive_weight_net") and apply_aux:
                        term_recon = loss_recon
                        term_phys = (
                            self.lambda_phys * loss_phys_spatial
                            if (self.use_physics_loss and self.lambda_phys > 0.0)
                            else torch.tensor(0.0, device=self.device)
                        )
                        term_fft = (
                            self.lambda_fft * loss_fft
                            if (self.use_fft_loss and self.lambda_fft > 0.0)
                            else torch.tensor(0.0, device=self.device)
                        )
                        term_grad = (
                            self.lambda_grad * loss_grad
                            if (self.use_gradient_loss and self.lambda_grad > 0.0)
                            else torch.tensor(0.0, device=self.device)
                        )

                        terms = torch.stack([term_recon, term_phys, term_fft, term_grad])
                        loss_vec = terms.detach().clamp(min=0.0)
                        adaptive_w = self.adaptive_weight_net(loss_vec)

                        mask = torch.tensor(
                            [
                                1.0,
                                1.0 if (self.use_physics_loss and self.lambda_phys > 0.0) else 0.0,
                                1.0 if (self.use_fft_loss and self.lambda_fft > 0.0) else 0.0,
                                1.0 if (self.use_gradient_loss and self.lambda_grad > 0.0) else 0.0,
                            ],
                            device=self.device,
                            dtype=adaptive_w.dtype,
                        )
                        adaptive_w = adaptive_w * mask
                        adaptive_w = adaptive_w / (adaptive_w.sum() + 1e-8)

                        loss_total = torch.sum(adaptive_w * terms)

                        # 自适应权重只记录在主 lead 上，避免日志刷屏
                        if lead_steps == main_lead_steps:
                            self.log("train/adapt_w_recon", adaptive_w[0], sync_dist=True)
                            self.log("train/adapt_w_phys", adaptive_w[1], sync_dist=True)
                            self.log("train/adapt_w_fft", adaptive_w[2], sync_dist=True)
                            self.log("train/adapt_w_grad", adaptive_w[3], sync_dist=True)
                    else:
                        loss_total = loss_recon
                        if apply_aux and self.use_physics_loss and self.lambda_phys > 0.0:
                            loss_total = loss_total + self.lambda_phys * loss_phys_spatial
                        if apply_aux and self.use_fft_loss and self.lambda_fft > 0.0:
                            loss_total = loss_total + self.lambda_fft * loss_fft
                        if apply_aux and self.use_gradient_loss and self.lambda_grad > 0.0:
                            loss_total = loss_total + self.lambda_grad * loss_grad

                    # ---- lead 权重（关键） ----
                    w = float(self._lead_loss_weight(lead_steps, lead_list))
                    loss_sum = loss_sum + loss_total * w
                    w_sum += w

                    # ---- 保存主 lead 日志用的中间量 ----
                    if lead_steps == main_lead_steps:
                        main_pred_denorm = x0_pred_denorm
                        main_gt_denorm = x0 * std + mean
                        main_phys_logs = phys_logs
                        main_loss_recon = loss_recon
                        main_loss_phys = loss_phys_spatial
                        main_loss_feasibility = loss_feasibility
                        main_loss_process_consistency = loss_process_consistency
                        main_loss_structure = loss_structure
                        main_loss_fft = loss_fft
                        main_loss_grad = loss_grad
                        main_forward_diagnostics = forward_diagnostics

                # 归一化（避免 selected 数量改变导致 loss 标度变化）
                if self.normalize_lead_loss and (w_sum > 0.0):
                    loss = loss_sum / float(w_sum)
                else:
                    loss = loss_sum

                # ---- 对抗训练与蒸馏（仅对主 lead 执行，避免额外算力与不稳定） ----
                # 说明：对抗/蒸馏本身就是额外正则项，默认只作用在 6h 主任务上更稳。
                # 如果你们确实希望对所有 lead 生效，可以后续再扩展为开关。

                # 对抗训练（可选）
                if self.use_adversarial and (self.discriminator is not None) and (self.d_optimizer is not None):
                    # 主 lead 的 pred/gt 必须存在
                    if main_pred_denorm is not None:
                        # (1) 更新判别器
                        self.d_optimizer.zero_grad()

                        # 判别器输入使用“归一化空间”还是“物理空间”都可以，但需保持一致。
                        # 这里沿用原实现：使用归一化空间的张量。
                        # main_gt_norm / main_pred_norm
                        # 为了取到 main_pred_norm，我们重新用 (main_pred_denorm-mean)/std 得到。
                        main_pred_norm = (main_pred_denorm - mean) / std
                        main_gt_norm = (main_gt_denorm - mean) / std

                        d_real = self.discriminator(main_gt_norm)
                        d_fake = self.discriminator(main_pred_norm.detach())

                        d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(
                            d_fake, torch.zeros_like(d_fake)
                        )
                        d_loss.backward()
                        self.d_optimizer.step()

                        # (2) 生成器对抗损失
                        for p in self.discriminator.parameters():
                            p.requires_grad_(False)
                        d_fake_for_g = self.discriminator(main_pred_norm)
                        g_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
                        for p in self.discriminator.parameters():
                            p.requires_grad_(True)

                        loss = loss + self.lambda_adv * g_adv
                        self.log("train/d_loss", d_loss, sync_dist=True)
                        self.log("train/g_adv", g_adv, sync_dist=True)

                # 知识蒸馏（可选）
                if self.use_distillation and (self.teacher_model is not None):
                    # 只对主 lead 做蒸馏
                    # 重新构造主 lead 的 x_t 以避免存缓存（简单明了）
                    if lead_list is not None and (main_lead_steps in lead_list):
                        main_idx = int(lead_list.index(main_lead_steps))
                        x0_main = y_futures[:, main_idx]
                        x_t_main = t.view(-1, 1, 1, 1) * x0_main + (1.0 - t.view(-1, 1, 1, 1)) * eps

                        lt_steps_t = torch.full((B,), float(main_lead_steps), device=self.device, dtype=torch.float32)
                        lead_cond_main = self._normalize_lead_time(lt_steps_t)

                        with torch.no_grad():
                            try:
                                teacher_pred = self.teacher_model(
                                    x_t_main,
                                    x_ctx,
                                    t,
                                    lead_time=lead_cond_main,
                                    **self._model_static_kwargs(static_batch),
                                )
                            except TypeError:
                                teacher_pred = self.teacher_model(x_t_main, x_ctx, t)

                        # 学生预测
                        student_pred = self.net(x_t_main, x_ctx, t, lead_time=lead_cond_main, **self._model_static_kwargs(static_batch))

                        T = float(getattr(self, "distill_temperature", 1.0))
                        distill_loss = F.mse_loss(student_pred / T, teacher_pred / T) * (T ** 2)
                        loss = self.distill_alpha * loss + (1.0 - self.distill_alpha) * distill_loss

                        self.log("train/distill_loss", distill_loss, sync_dist=True)

                # ---- 日志（主 lead 为准，避免刷屏） ----
                self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
                self.log("train/loss_recon", main_loss_recon, sync_dist=True, on_step=True, on_epoch=True)
                self.log("train/loss_phys_spatial", main_loss_phys, sync_dist=True, on_step=True, on_epoch=True)
                self.log("train/loss_feasibility", main_loss_feasibility, sync_dist=True, on_step=True, on_epoch=True)
                self.log(
                    "train/loss_process_consistency",
                    main_loss_process_consistency,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                )
                self.log("train/loss_structure", main_loss_structure, sync_dist=True, on_step=True, on_epoch=True)
                self.log("train/loss_fft", main_loss_fft, sync_dist=True, on_step=True, on_epoch=True)
                self.log("train/loss_grad", main_loss_grad, sync_dist=True, on_step=True, on_epoch=True)

                if main_forward_diagnostics is not None:
                    self._log_forward_diagnostics(
                        main_forward_diagnostics,
                        split="train",
                        on_step=True,
                        on_epoch=True,
                    )

                # 记录训练阶段主 lead 的 MAE/RMSE（整体 + 分变量）
                if (main_pred_denorm is not None) and (main_gt_denorm is not None):
                    self._log_mae_rmse(
                        split="train",
                        lead_steps=int(main_lead_steps),
                        pred_denorm=main_pred_denorm,
                        gt_denorm=main_gt_denorm,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                    )

                # 物理子项日志（主 lead）
                if main_phys_logs is not None:
                    for k, v in main_phys_logs.items():
                        self.log(f"train/{k}", v, sync_dist=True, on_step=True, on_epoch=True)

                return loss

            # -------------------------
            # 单提前期训练（完全兼容旧版）
            # -------------------------
            x0 = batch["x0"].to(self.device)
            B = x0.size(0)

            # ---- 标准扩散训练：用 GT 的 x0 构造噪声输入 x_t ----
            t = self.sample_t(B)
            eps = torch.randn_like(x0)
            x_t = t.view(-1, 1, 1, 1) * x0 + (1.0 - t.view(-1, 1, 1, 1)) * eps

            x0_pred = self.net(
                x_t,
                x_ctx,
                t,
                lead_time=None,
                return_diagnostics=True,
                **self._model_static_kwargs(static_batch),
            )
            if isinstance(x0_pred, tuple):
                x0_pred, forward_diagnostics = x0_pred
            else:
                forward_diagnostics = None

            # ---- 硬物理约束投影（可选） ----
            if self.use_hard_physics:
                x0_pred_denorm = x0_pred * std + mean
                x0_pred_denorm = self._apply_hard_physics_constraints(
                    x0_pred_denorm, num_steps=self.projection_steps
                )
                x0_pred = (x0_pred_denorm - mean) / std

            # ---- 重建损失 ----
            if self.use_channel_weights:
                loss_recon = self._calculate_weighted_recon_loss(x0_pred, x0)
            else:
                loss_recon = F.mse_loss(x0_pred, x0)

            # ---- 物理/频域/梯度损失 ----
            phys = self._calculate_physics_losses(x0_pred, x0, std, mean, static_batch, x_ctx=x_ctx)
            x0_pred_denorm = phys["x0_pred_denorm"]
            loss_phys_spatial = phys["loss_phys_spatial"]
            loss_feasibility = phys["loss_feasibility"]
            loss_process_consistency = phys["loss_process_consistency"]
            loss_structure = phys["loss_structure"]
            loss_fft = phys["loss_fft"]
            loss_grad = phys["loss_grad"]
            phys_logs = phys["logs"]

            # ---- 组合总损失 ----
            if self.use_adaptive_weights and hasattr(self, "adaptive_weight_net"):
                term_recon = loss_recon
                term_phys = (
                    self.lambda_phys * loss_phys_spatial
                    if (self.use_physics_loss and self.lambda_phys > 0.0)
                    else torch.tensor(0.0, device=self.device)
                )
                term_fft = (
                    self.lambda_fft * loss_fft
                    if (self.use_fft_loss and self.lambda_fft > 0.0)
                    else torch.tensor(0.0, device=self.device)
                )
                term_grad = (
                    self.lambda_grad * loss_grad
                    if (self.use_gradient_loss and self.lambda_grad > 0.0)
                    else torch.tensor(0.0, device=self.device)
                )

                terms = torch.stack([term_recon, term_phys, term_fft, term_grad])
                loss_vec = terms.detach().clamp(min=0.0)
                adaptive_w = self.adaptive_weight_net(loss_vec)

                mask = torch.tensor(
                    [
                        1.0,
                        1.0 if (self.use_physics_loss and self.lambda_phys > 0.0) else 0.0,
                        1.0 if (self.use_fft_loss and self.lambda_fft > 0.0) else 0.0,
                        1.0 if (self.use_gradient_loss and self.lambda_grad > 0.0) else 0.0,
                    ],
                    device=self.device,
                    dtype=adaptive_w.dtype,
                )
                adaptive_w = adaptive_w * mask
                adaptive_w = adaptive_w / (adaptive_w.sum() + 1e-8)

                loss = torch.sum(adaptive_w * terms)

                self.log("train/adapt_w_recon", adaptive_w[0], sync_dist=True)
                self.log("train/adapt_w_phys", adaptive_w[1], sync_dist=True)
                self.log("train/adapt_w_fft", adaptive_w[2], sync_dist=True)
                self.log("train/adapt_w_grad", adaptive_w[3], sync_dist=True)
            else:
                loss = loss_recon
                if self.use_physics_loss and self.lambda_phys > 0.0:
                    loss = loss + self.lambda_phys * loss_phys_spatial
                if self.use_fft_loss and self.lambda_fft > 0.0:
                    loss = loss + self.lambda_fft * loss_fft
                if self.use_gradient_loss and self.lambda_grad > 0.0:
                    loss = loss + self.lambda_grad * loss_grad

            # ---- 对抗训练（可选） ----
            if self.use_adversarial and (self.discriminator is not None) and (self.d_optimizer is not None):
                self.d_optimizer.zero_grad()

                d_real = self.discriminator(x0)
                d_fake = self.discriminator(x0_pred.detach())

                d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake)
                )
                d_loss.backward()
                self.d_optimizer.step()

                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                d_fake_for_g = self.discriminator(x0_pred)
                g_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)

                loss = loss + self.lambda_adv * g_adv

                self.log("train/d_loss", d_loss, sync_dist=True)
                self.log("train/g_adv", g_adv, sync_dist=True)

            # ---- 知识蒸馏（可选） ----
            if self.use_distillation and (self.teacher_model is not None):
                with torch.no_grad():
                    try:
                        teacher_pred = self.teacher_model(x_t, x_ctx, t, **self._model_static_kwargs(static_batch))
                    except TypeError:
                        teacher_pred = self.teacher_model(x_t, x_ctx, t)

                T = float(getattr(self, "distill_temperature", 1.0))
                distill_loss = F.mse_loss(x0_pred / T, teacher_pred / T) * (T ** 2)
                loss = self.distill_alpha * loss + (1.0 - self.distill_alpha) * distill_loss

                self.log("train/distill_loss", distill_loss, sync_dist=True)

            # ---- 日志 ----
            self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_recon", loss_recon, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_phys_spatial", loss_phys_spatial, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_feasibility", loss_feasibility, sync_dist=True, on_step=True, on_epoch=True)
            self.log(
                "train/loss_process_consistency",
                loss_process_consistency,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
            self.log("train/loss_structure", loss_structure, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_fft", loss_fft, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_grad", loss_grad, sync_dist=True, on_step=True, on_epoch=True)

            with torch.no_grad():
                x0_denorm = x0 * std + mean
                self._log_mae_rmse(
                    split="train",
                    lead_steps=1,
                    pred_denorm=x0_pred_denorm,
                    gt_denorm=x0_denorm,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

                if forward_diagnostics is not None:
                    self._log_forward_diagnostics(
                        forward_diagnostics,
                        split="train",
                        on_step=True,
                        on_epoch=True,
                    )

                for k, v in phys_logs.items():
                    self.log(f"train/{k}", v, sync_dist=True, on_step=True, on_epoch=True)

            return loss

        def _training_step_rollout_finetune(
            self,
            *,
            x_ctx: torch.Tensor,
            y_futures: torch.Tensor,
            lead_list: list,
            mean: torch.Tensor,
            std: torch.Tensor,
            static_batch: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        ) -> torch.Tensor:
            """v4：多步 rollout 微调训练。

            设计目标：
            - Stage-1：先把 6h(1-step) 做好；
            - Stage-2：在训练中显式加入多步 rollout 的误差（可配合 scheduled sampling），
              用来缓解长提前期(18h/24h)的误差累积与漂移。

            说明：
            - 该实现不要求 use_lead_time_conditioning=true。
              如果启用了 lead conditioning，则在 rollout 训练里固定使用 lead=1（因为每一步都是 next-step）。
            - 为了节省显存与提高稳定性，默认在更新上下文时 detach 预测结果（可通过配置关闭）。
            """

            device = x_ctx.device
            B = int(y_futures.shape[0])
            L = int(y_futures.shape[1])

            # 当前 epoch 的 rollout 步数（例如 1->4 的课程学习）
            K = int(self._current_rollout_steps(lead_list))
            K = max(1, min(K, L))

            # teacher forcing 比例（scheduled sampling）
            tf_ratio = float(self._teacher_forcing_ratio())

            # rollout 期间哪些 step 参与 loss
            loss_steps = list(range(1, K + 1))
            if str(getattr(self, "rollout_loss_on", "all")).lower() in ("final", "final_only", "last"):
                loss_steps = [K]
                if bool(getattr(self, "rollout_always_include_step1", True)):
                    if 1 not in loss_steps:
                        loss_steps = [1] + loss_steps

            # 日志：把关键的 schedule 输出，便于你们排查训练曲线
            self.log("train/rollout_K", float(K), sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/teacher_forcing_ratio", float(tf_ratio), sync_dist=True, on_step=True, on_epoch=True)

            # 共享 t 与 eps（更稳；也更接近 direct multi-lead 训练的做法）
            t = self.sample_t(B)
            eps = torch.randn_like(y_futures[:, 0])  # [B,C,H,W]

            ctx_cur = x_ctx
            physics_ctx_cur = x_ctx

            loss_sum = torch.tensor(0.0, device=device)
            w_sum = 0.0

            # 仅用于日志：主任务仍然以 1-step(6h) 为准
            main_pred_denorm: Optional[torch.Tensor] = None
            main_gt_denorm: Optional[torch.Tensor] = None
            main_phys_logs: Optional[Dict[str, Any]] = None
            main_loss_recon = torch.tensor(0.0, device=device)
            main_loss_phys = torch.tensor(0.0, device=device)
            main_loss_feasibility = torch.tensor(0.0, device=device)
            main_loss_process_consistency = torch.tensor(0.0, device=device)
            main_loss_structure = torch.tensor(0.0, device=device)
            main_loss_fft = torch.tensor(0.0, device=device)
            main_loss_grad = torch.tensor(0.0, device=device)
            main_forward_diagnostics: Optional[Dict[str, Any]] = None

            # 同时记录最后一步（例如 24h）的误差，方便观察“长提前期是否真的被拉起来”
            final_pred_denorm: Optional[torch.Tensor] = None
            final_gt_denorm: Optional[torch.Tensor] = None

            static_batch = static_batch or {"static_raw": None, "static_cont": None, "static_cat": None}

            # rollout 主循环
            for step in range(1, K + 1):
                # lead_list 与 y_futures 对齐：取出对应 step 的 GT
                if step in lead_list:
                    idx = int(lead_list.index(step))
                else:
                    # 数据不包含该 step，提前结束
                    break

                x0 = y_futures[:, idx]  # [B,C,H,W]
                x_t = t.view(-1, 1, 1, 1) * x0 + (1.0 - t.view(-1, 1, 1, 1)) * eps

                # rollout 训练里每一步都是 next-step，因此 lead_cond 固定=1
                if bool(self.use_lead_time_conditioning):
                    lt_steps_t = torch.ones((B,), device=device, dtype=torch.float32)
                    lead_cond = self._normalize_lead_time(lt_steps_t)
                else:
                    lead_cond = None

                x0_pred = self.net(
                    x_t,
                    ctx_cur,
                    t,
                    lead_time=lead_cond,
                    return_diagnostics=bool(step == 1),
                    **self._model_static_kwargs(static_batch),
                )
                if isinstance(x0_pred, tuple):
                    x0_pred, forward_diagnostics = x0_pred
                else:
                    forward_diagnostics = None

                # 是否对该 step 施加辅助损失（物理/FFT/梯度/硬约束）
                apply_aux = bool(self.apply_aux_losses_to_all_leads) or (step == 1)

                # ---- 硬物理约束投影（可选） ----
                if apply_aux and self.use_hard_physics:
                    x0_pred_denorm_tmp = x0_pred * std + mean
                    x0_pred_denorm_tmp = self._apply_hard_physics_constraints(
                        x0_pred_denorm_tmp, num_steps=self.projection_steps
                    )
                    x0_pred = (x0_pred_denorm_tmp - mean) / std

                # ---- 重建损失 ----
                if self.use_channel_weights:
                    loss_recon = self._calculate_weighted_recon_loss(x0_pred, x0)
                else:
                    loss_recon = F.mse_loss(x0_pred, x0)

                # ---- 物理/频域/梯度损失（可选） ----
                if apply_aux:
                    phys = self._calculate_physics_losses(x0_pred, x0, std, mean, static_batch, x_ctx=physics_ctx_cur)
                    x0_pred_denorm = phys["x0_pred_denorm"]
                    loss_phys_spatial = phys["loss_phys_spatial"]
                    loss_feasibility = phys["loss_feasibility"]
                    loss_process_consistency = phys["loss_process_consistency"]
                    loss_structure = phys["loss_structure"]
                    loss_fft = phys["loss_fft"]
                    loss_grad = phys["loss_grad"]
                    phys_logs = phys["logs"]
                else:
                    x0_pred_denorm = x0_pred * std + mean
                    loss_phys_spatial = torch.tensor(0.0, device=device)
                    loss_feasibility = torch.tensor(0.0, device=device)
                    loss_process_consistency = torch.tensor(0.0, device=device)
                    loss_structure = torch.tensor(0.0, device=device)
                    loss_fft = torch.tensor(0.0, device=device)
                    loss_grad = torch.tensor(0.0, device=device)
                    phys_logs = self._zero_physics_logs(x0_pred)

                # ---- 组合总损失 ----
                if self.use_adaptive_weights and hasattr(self, "adaptive_weight_net") and apply_aux:
                    term_recon = loss_recon
                    term_phys = (
                        self.lambda_phys * loss_phys_spatial
                        if (self.use_physics_loss and self.lambda_phys > 0.0)
                        else torch.tensor(0.0, device=device)
                    )
                    term_fft = (
                        self.lambda_fft * loss_fft
                        if (self.use_fft_loss and self.lambda_fft > 0.0)
                        else torch.tensor(0.0, device=device)
                    )
                    term_grad = (
                        self.lambda_grad * loss_grad
                        if (self.use_gradient_loss and self.lambda_grad > 0.0)
                        else torch.tensor(0.0, device=device)
                    )

                    terms = torch.stack([term_recon, term_phys, term_fft, term_grad])
                    loss_vec = terms.detach().clamp(min=0.0)
                    adaptive_w = self.adaptive_weight_net(loss_vec)

                    mask = torch.tensor(
                        [
                            1.0,
                            1.0 if (self.use_physics_loss and self.lambda_phys > 0.0) else 0.0,
                            1.0 if (self.use_fft_loss and self.lambda_fft > 0.0) else 0.0,
                            1.0 if (self.use_gradient_loss and self.lambda_grad > 0.0) else 0.0,
                        ],
                        device=device,
                        dtype=adaptive_w.dtype,
                    )
                    adaptive_w = adaptive_w * mask
                    adaptive_w = adaptive_w / (adaptive_w.sum() + 1e-8)

                    loss_total = torch.sum(adaptive_w * terms)
                    if step == 1:
                        self.log("train/adapt_w_recon", adaptive_w[0], sync_dist=True)
                        self.log("train/adapt_w_phys", adaptive_w[1], sync_dist=True)
                        self.log("train/adapt_w_fft", adaptive_w[2], sync_dist=True)
                        self.log("train/adapt_w_grad", adaptive_w[3], sync_dist=True)
                else:
                    loss_total = loss_recon
                    if apply_aux and self.use_physics_loss and self.lambda_phys > 0.0:
                        loss_total = loss_total + self.lambda_phys * loss_phys_spatial
                    if apply_aux and self.use_fft_loss and self.lambda_fft > 0.0:
                        loss_total = loss_total + self.lambda_fft * loss_fft
                    if apply_aux and self.use_gradient_loss and self.lambda_grad > 0.0:
                        loss_total = loss_total + self.lambda_grad * loss_grad

                # ---- 仅对 loss_steps 里的 step 计入梯度 ----
                if step in loss_steps:
                    w = float(self._lead_loss_weight(step, lead_list))
                    loss_sum = loss_sum + loss_total * w
                    w_sum += w

                # ---- 保存 step-1 与 final 的日志 ----
                if step == 1:
                    main_pred_denorm = x0_pred_denorm
                    main_gt_denorm = x0 * std + mean
                    main_phys_logs = phys_logs
                    main_loss_recon = loss_recon
                    main_loss_phys = loss_phys_spatial
                    main_loss_feasibility = loss_feasibility
                    main_loss_process_consistency = loss_process_consistency
                    main_loss_structure = loss_structure
                    main_loss_fft = loss_fft
                    main_loss_grad = loss_grad
                    main_forward_diagnostics = forward_diagnostics

                if step == K:
                    final_pred_denorm = x0_pred_denorm
                    final_gt_denorm = x0 * std + mean

                # ---- 更新上下文，进入下一步 ----
                if step < K:
                    # scheduled sampling：按 tf_ratio 决定是否使用 GT
                    if tf_ratio >= 1.0:
                        x_next = x0
                    elif tf_ratio <= 0.0:
                        x_next = x0_pred.detach() if bool(getattr(self, "rollout_detach_context", True)) else x0_pred
                    else:
                        pred_for_ctx = x0_pred.detach() if bool(getattr(self, "rollout_detach_context", True)) else x0_pred
                        # mask: [B,1,1,1]
                        mask = (torch.rand((B, 1, 1, 1), device=device) < tf_ratio).float()
                        x_next = mask * x0 + (1.0 - mask) * pred_for_ctx

                    ctx_cur = self._update_ctx_autoregressive(ctx_cur, x_next)
                    physics_next = x0_pred.detach() if bool(getattr(self, "rollout_detach_context", True)) else x0_pred
                    physics_ctx_cur = self._update_ctx_autoregressive(physics_ctx_cur, physics_next)

            # 归一化（避免 K 或 loss_steps 改变导致 loss 标度变化）
            if self.normalize_lead_loss and (w_sum > 0.0):
                loss = loss_sum / float(w_sum)
            else:
                loss = loss_sum

            # ---- 对抗训练与蒸馏（仅对 step-1 执行，稳定优先） ----
            if self.use_adversarial and (self.discriminator is not None) and (self.d_optimizer is not None):
                if main_pred_denorm is not None and main_gt_denorm is not None:
                    self.d_optimizer.zero_grad()

                    main_pred_norm = (main_pred_denorm - mean) / std
                    main_gt_norm = (main_gt_denorm - mean) / std

                    d_real = self.discriminator(main_gt_norm)
                    d_fake = self.discriminator(main_pred_norm.detach())

                    d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(
                        d_fake, torch.zeros_like(d_fake)
                    )
                    d_loss.backward()
                    self.d_optimizer.step()

                    for p in self.discriminator.parameters():
                        p.requires_grad_(False)
                    d_fake_for_g = self.discriminator(main_pred_norm)
                    g_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
                    for p in self.discriminator.parameters():
                        p.requires_grad_(True)

                    loss = loss + self.lambda_adv * g_adv
                    self.log("train/d_loss", d_loss, sync_dist=True)
                    self.log("train/g_adv", g_adv, sync_dist=True)

            if self.use_distillation and (self.teacher_model is not None):
                # 只蒸馏 step-1：
                x0_main = y_futures[:, int(lead_list.index(1))] if 1 in lead_list else y_futures[:, 0]
                x_t_main = t.view(-1, 1, 1, 1) * x0_main + (1.0 - t.view(-1, 1, 1, 1)) * eps

                with torch.no_grad():
                    try:
                        teacher_pred = self.teacher_model(x_t_main, x_ctx, t, **self._model_static_kwargs(static_batch))
                    except TypeError:
                        teacher_pred = self.teacher_model(x_t_main, x_ctx, t)

                student_pred = self.net(x_t_main, x_ctx, t, lead_time=None, **self._model_static_kwargs(static_batch))

                T = float(getattr(self, "distill_temperature", 1.0))
                distill_loss = F.mse_loss(student_pred / T, teacher_pred / T) * (T ** 2)
                loss = self.distill_alpha * loss + (1.0 - self.distill_alpha) * distill_loss
                self.log("train/distill_loss", distill_loss, sync_dist=True)

            # ---- 日志：主 step + final step ----
            self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_recon", main_loss_recon, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_phys_spatial", main_loss_phys, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_feasibility", main_loss_feasibility, sync_dist=True, on_step=True, on_epoch=True)
            self.log(
                "train/loss_process_consistency",
                main_loss_process_consistency,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
            self.log("train/loss_structure", main_loss_structure, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_fft", main_loss_fft, sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/loss_grad", main_loss_grad, sync_dist=True, on_step=True, on_epoch=True)

            if main_forward_diagnostics is not None:
                self._log_forward_diagnostics(
                    main_forward_diagnostics,
                    split="train",
                    on_step=True,
                    on_epoch=True,
                )

            if (main_pred_denorm is not None) and (main_gt_denorm is not None):
                self._log_mae_rmse(
                    split="train",
                    lead_steps=1,
                    pred_denorm=main_pred_denorm,
                    gt_denorm=main_gt_denorm,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

            # final step 指标（不放到 prog_bar，避免干扰主指标）
            if (final_pred_denorm is not None) and (final_gt_denorm is not None) and (K > 1):
                self._log_mae_rmse(
                    split="train",
                    lead_steps=int(K),
                    pred_denorm=final_pred_denorm,
                    gt_denorm=final_gt_denorm,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )

            if main_phys_logs is not None:
                for k, v in main_phys_logs.items():
                    self.log(f"train/{k}", v, sync_dist=True, on_step=True, on_epoch=True)

            return loss
        # -------------------------
        # 多提前期验证/测试（统一实现）
        # -------------------------
        @torch.no_grad()
        def _shared_eval_step(self, batch: Dict[str, Any], batch_idx: int, *, split: str):
            """验证/测试阶段的统一 step。

            功能：
            1) 支持多提前期（multi-horizon）指标记录：对 eval_lead_times 中每个 lead 记录 MAE/RMSE（整体 + 分变量）；
            2) 支持两类推理策略：
               - direct：直接预测指定 lead（需要 use_lead_time_conditioning=True）
               - autoregressive：用 1-step 逐步滚动到目标 lead（业界常用）
            3) 支持 deterministic / ensemble 两种模式；ensemble 下可选计算 CRPS。
            """

            x_ctx = batch["x_ctx"].to(self.device)
            static_batch = self._move_static_batch(batch)
            norm = batch["norm"]
            mean = norm["mean"].to(self.device)
            std = norm["std"].to(self.device)
            meta = batch.get("meta", None)
            lat = None
            if isinstance(meta, dict) and ("lat" in meta):
                lat = meta["lat"].to(self.device)
            if str(getattr(self, "acc_weighting", "coslat")).lower() != "coslat":
                lat = None

            clim_denorm = None
            if bool(getattr(self, "acc_enabled", False)) and str(getattr(self, "acc_climatology", "train_mean_field")).lower() == "train_mean_field":
                clim_denorm = batch.get("clim", None)
                if clim_denorm is not None:
                    clim_denorm = clim_denorm.to(self.device)

            # 目标：单提前期 x0（兼容旧版）
            x0_single = batch["x0"].to(self.device)

            # 目标：多提前期 y_futures（可选）
            y_futures = batch.get("y", None)
            if y_futures is not None:
                y_futures = y_futures.to(self.device)  # [B,L,C,H,W]

            lead_list = self._extract_lead_list(batch)  # 例如 [1,2,3,4]
            if lead_list is None and y_futures is not None:
                lead_list = list(range(1, int(y_futures.shape[1]) + 1))

            # 需要评估的 lead（过滤掉数据里没有的）
            leads_to_eval = [int(x) for x in (self.eval_lead_times or [1])]
            if lead_list is not None and len(lead_list) > 0:
                leads_to_eval = [lead_step for lead_step in leads_to_eval if lead_step in lead_list]
                if len(leads_to_eval) == 0:
                    leads_to_eval = [int(lead_list[0])]

            leads_to_eval = sorted(list(dict.fromkeys(leads_to_eval)))  # 去重保序

            # 推理配置
            diff_cfg = self.hparams.get("diffusion_cfg", {})
            steps = int(diff_cfg.get("sample_steps", 8))
            t_start = float(diff_cfg.get("t_start", 0.0))
            t_end = float(diff_cfg.get("t_end", 1.0))

            infer = self._get_infer_cfg()
            if split == "val":
                mode = str(infer.get("val_mode", infer.get("mode", "deterministic"))).lower()
            else:
                mode = str(infer.get("test_mode", infer.get("mode", "deterministic"))).lower()

            is_ens = mode in ("ensemble", "ens", "probabilistic")
            M = int(infer.get("ensemble_size", 20)) if is_ens else 1

            fixed_seed = int(infer.get("fixed_seed", 42))
            base_seed = int(infer.get("base_seed", fixed_seed))
            seed_offset_per_batch = int(infer.get("seed_offset_per_batch", 100000))
            seed_offset_batch = seed_offset_per_batch * int(batch_idx)

            # 若用户选择 direct，但模型未启用 lead_time conditioning，则自动回退到 autoregressive
            mh_infer = str(self.multi_horizon_inference).lower()
            if mh_infer == "direct" and (not self.use_lead_time_conditioning):
                mh_infer = "autoregressive"

            preds_by_lead: Dict[int, torch.Tensor] = {}
            ens_by_lead: Dict[int, torch.Tensor] = {}
            main_forward_diagnostics: Optional[Dict[str, Any]] = None

            # -------------------------
            # 1) 生成预测
            # -------------------------
            if mh_infer == "autoregressive":
                # 一次 rollout 到最大的 lead，并缓存中间结果（避免重复采样）
                max_lead = int(max(leads_to_eval))
                ctx_cur = x_ctx

                for s in range(1, max_lead + 1):
                    # 每一步都是 1-step 预测（6h），因此 lead_time 取 1 step
                    lt_cond = None
                    if self.use_lead_time_conditioning:
                        lt_steps_t = torch.full((x_ctx.size(0),), 1.0, device=self.device, dtype=torch.float32)
                        lt_cond = self._normalize_lead_time(lt_steps_t)

                    if is_ens:
                        # 为不同 step 引入 seed 偏移，避免每一步都用同一份噪声
                        ens = self.predict_ensemble(
                            ctx_cur,
                            ensemble_size=M,
                            base_seed=base_seed,
                            steps=steps,
                            t_start=t_start,
                            t_end=t_end,
                            seed_offset=seed_offset_batch + s * 1000,
                            lead_time=lt_cond,
                            **self._model_static_kwargs(static_batch),
                        )
                        x_pred = ens.mean(dim=1)
                    else:
                        x_pred = self.predict_one(
                            ctx_cur,
                            steps=steps,
                            t_start=t_start,
                            t_end=t_end,
                            seed=fixed_seed + seed_offset_batch + s * 1000,
                            lead_time=lt_cond,
                            return_diagnostics=bool(s in leads_to_eval and s == int(leads_to_eval[0])),
                            **self._model_static_kwargs(static_batch),
                        )
                        if s in leads_to_eval and s == int(leads_to_eval[0]):
                            main_forward_diagnostics = getattr(self.net, "last_forward_diagnostics", {}) or None
                        ens = None

                    if s in leads_to_eval:
                        preds_by_lead[s] = x_pred
                        if ens is not None:
                            ens_by_lead[s] = ens

                    # 更新上下文，进入下一步
                    ctx_cur = self._update_ctx_autoregressive(ctx_cur, x_pred)
            else:
                # direct：对每个 lead 单独采样一次（需要模型支持 lead_time conditioning）
                for lead_step in leads_to_eval:
                    lt_cond = None
                    if self.use_lead_time_conditioning:
                        lt_steps_t = torch.full((x_ctx.size(0),), float(lead_step), device=self.device, dtype=torch.float32)
                        lt_cond = self._normalize_lead_time(lt_steps_t)

                    # 是否在多提前期下复用同一份噪声（更“连贯”的轨迹）
                    lead_offset = 0 if self.correlate_noise_across_leads else int(lead_step) * 1000

                    if is_ens:
                        ens = self.predict_ensemble(
                            x_ctx,
                            ensemble_size=M,
                            base_seed=base_seed,
                            steps=steps,
                            t_start=t_start,
                            t_end=t_end,
                            seed_offset=seed_offset_batch + lead_offset,
                            lead_time=lt_cond,
                            **self._model_static_kwargs(static_batch),
                        )
                        x_pred = ens.mean(dim=1)
                        ens_by_lead[lead_step] = ens
                    else:
                        x_pred = self.predict_one(
                            x_ctx,
                            steps=steps,
                            t_start=t_start,
                            t_end=t_end,
                            seed=fixed_seed + seed_offset_batch + lead_offset,
                            lead_time=lt_cond,
                            return_diagnostics=bool(int(lead_step) == int(leads_to_eval[0])),
                            **self._model_static_kwargs(static_batch),
                        )
                        if int(lead_step) == int(leads_to_eval[0]):
                            main_forward_diagnostics = getattr(self.net, "last_forward_diagnostics", {}) or None
                    preds_by_lead[lead_step] = x_pred

            # -------------------------
            # 2) 记录多提前期 MAE/RMSE
            # -------------------------
            main_lead = int(leads_to_eval[0])
            x0_main_gt = x0_single

            # 额外：记录跨 lead 的平均指标，方便 early-stopping/对比多提前期综合效果
            rmse_list = []
            mae_list = []
            bias_list = []
            crps_list = []
            acc_list = []

            for lead_step, x_pred in preds_by_lead.items():
                # 选择对应 lead 的 GT
                if y_futures is not None and lead_list is not None and (lead_step in lead_list):
                    idx = int(lead_list.index(int(lead_step)))
                    x_gt = y_futures[:, idx]
                else:
                    # 只有单提前期 GT 的情况下，只能评估该提前期
                    x_gt = x0_single

                if int(lead_step) == int(main_lead):
                    x0_main_gt = x_gt

                pred_denorm = x_pred * std + mean
                gt_denorm = x_gt * std + mean

                # 本 lead 的整体确定性指标（用于跨 lead 的平均）
                rmse_list.append(torch.sqrt(torch.mean((pred_denorm - gt_denorm) ** 2)))
                mae_val = torch.mean(torch.abs(pred_denorm - gt_denorm))
                mae_list.append(mae_val)
                bias_list.append(torch.mean(pred_denorm - gt_denorm))
                crps_list.append(mae_val)

                self._log_mae_rmse(
                    split=split,
                    lead_steps=int(lead_step),
                    pred_denorm=pred_denorm,
                    gt_denorm=gt_denorm,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(split == "val" and int(lead_step) == int(main_lead)),
                )
                if clim_denorm is not None:
                    acc_val = self._log_acc(
                        split=split,
                        lead_steps=int(lead_step),
                        pred_denorm=pred_denorm,
                        gt_denorm=gt_denorm,
                        clim_denorm=clim_denorm,
                        lat=lat,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    acc_list.append(acc_val)

                # 概率指标：CRPS（仅 ensemble 且用户开启）
                if is_ens and (lead_step in ens_by_lead) and bool(infer.get("compute_crps", True)):
                    ens = ens_by_lead[int(lead_step)]  # [B,M,C,H,W]
                    mean_e = mean.unsqueeze(1)
                    std_e = std.unsqueeze(1)
                    ens_denorm = ens * std_e + mean_e

                    tag = self._lead_tag(int(lead_step))
                    crps = _crps_ensemble(ens_denorm, gt_denorm, reduction="mean")
                    crps_list[-1] = crps
                    self.log(f"{split}/CRPS@{tag}", crps, sync_dist=True)

                    crps_ch = _crps_ensemble(ens_denorm, gt_denorm, reduction="channel_mean")
                    for i in range(gt_denorm.size(1)):
                        name = self.var_names[i] if i < len(self.var_names) else str(i)
                        self.log(f"{split}/CRPS_{name}@{tag}", crps_ch[i], sync_dist=True)

                    spread = ens_denorm.std(dim=1).mean()
                    self.log(f"{split}/spread@{tag}", spread, sync_dist=True)

            # -------------------------
            # 2.5) 记录跨 lead 的平均指标（可作为 early stopping 的 monitor）
            # -------------------------
            if len(rmse_list) > 0:
                rmse_mean = torch.stack(rmse_list).mean()
                mae_mean = torch.stack(mae_list).mean()
                bias_mean = torch.stack(bias_list).mean()
                crps_mean = torch.stack(crps_list).mean()
                self.log(
                    f"{split}/RMSE_mean_leads",
                    rmse_mean,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(split == 'val'),
                )
                self.log(
                    f"{split}/MAE_mean_leads",
                    mae_mean,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{split}/Bias_mean_leads",
                    bias_mean,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{split}/CRPS_mean_leads",
                    crps_mean,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
            if len(acc_list) > 0:
                acc_stack = torch.stack(acc_list)
                acc_finite = acc_stack[torch.isfinite(acc_stack)]
                acc_mean = acc_finite.mean() if acc_finite.numel() > 0 else torch.tensor(float("nan"), device=acc_stack.device)
                self.log(
                    f"{split}/ACC_mean_leads",
                    acc_mean,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

            # -------------------------
            # 3) 兼容旧版：保留不带 @tag 的 loss / RMSE / MAE（默认对应 main_lead）
            # -------------------------
            x0_main_pred = preds_by_lead[main_lead]

            if self.use_channel_weights:
                loss_recon = self._calculate_weighted_recon_loss(x0_main_pred, x0_main_gt)
            else:
                loss_recon = F.mse_loss(x0_main_pred, x0_main_gt)

            phys = self._calculate_physics_losses(x0_main_pred, x0_main_gt, std, mean, static_batch, x_ctx=x_ctx)
            x0_main_pred_denorm = phys["x0_pred_denorm"]
            loss_phys_spatial = phys["loss_phys_spatial"]
            loss_fft = phys["loss_fft"]
            loss_grad = phys["loss_grad"]
            phys_logs = phys["logs"]

            loss = loss_recon
            if self.use_physics_loss and self.lambda_phys > 0.0:
                loss = loss + self.lambda_phys * loss_phys_spatial
            if self.use_fft_loss and self.lambda_fft > 0.0:
                loss = loss + self.lambda_fft * loss_fft
            if self.use_gradient_loss and self.lambda_grad > 0.0:
                loss = loss + self.lambda_grad * loss_grad

            x0_main_denorm = x0_main_gt * std + mean
            rmse_main = torch.sqrt(torch.mean((x0_main_pred_denorm - x0_main_denorm) ** 2))
            mae_main = torch.mean(torch.abs(x0_main_pred_denorm - x0_main_denorm))
            bias_main = torch.mean(x0_main_pred_denorm - x0_main_denorm)

            # 兼容旧版：不带 @tag 的分变量 RMSE/MAE（默认对应 main_lead）
            var_rmse_main = torch.sqrt(torch.mean((x0_main_pred_denorm - x0_main_denorm) ** 2, dim=(2, 3)))
            var_mae_main = torch.mean(torch.abs(x0_main_pred_denorm - x0_main_denorm), dim=(2, 3))
            var_bias_main = torch.mean(x0_main_pred_denorm - x0_main_denorm, dim=(2, 3))
            C = x0_main_denorm.size(1)
            for i in range(C):
                name = self.var_names[i] if i < len(self.var_names) else str(i)
                self.log(f"{split}/RMSE_{name}", var_rmse_main[:, i].mean(), sync_dist=True)
                self.log(f"{split}/MAE_{name}", var_mae_main[:, i].mean(), sync_dist=True)
                self.log(f"{split}/Bias_{name}", var_bias_main[:, i].mean(), sync_dist=True)
                self.log(f"{split}/CRPS_{name}", var_mae_main[:, i].mean(), sync_dist=True)

            self.log(f"{split}/loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{split}/loss_recon", loss_recon, sync_dist=True)
            self.log(f"{split}/RMSE", rmse_main, prog_bar=True, sync_dist=True)
            self.log(f"{split}/MAE", mae_main, sync_dist=True)
            self.log(f"{split}/Bias", bias_main, sync_dist=True)
            self.log(f"{split}/CRPS", mae_main, sync_dist=True)
            self.log(f"{split}/main_lead_steps", float(main_lead), sync_dist=True)

            if main_forward_diagnostics is not None:
                self._log_forward_diagnostics(
                    main_forward_diagnostics,
                    split=split,
                    on_step=False,
                    on_epoch=True,
                )

            for k, v in phys_logs.items():
                self.log(f"{split}/{k}", v, sync_dist=True)

            return {f"{split}_loss": loss}

        def validation_step(self, batch: Dict[str, Any], batch_idx: int):
            return self._shared_eval_step(batch, batch_idx, split="val")

        def test_step(self, batch: Dict[str, Any], batch_idx: int):
            return self._shared_eval_step(batch, batch_idx, split="test")


        # -------------------------
        # checkpoint：保存/恢复判别器优化器状态（可选）
        # -------------------------
        def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            if self.use_adversarial and self.d_optimizer is not None:
                checkpoint["d_optimizer_state"] = self.d_optimizer.state_dict()

        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            if self.use_adversarial and self.d_optimizer is not None:
                st = checkpoint.get("d_optimizer_state", None)
                if st is not None:
                    try:
                        self.d_optimizer.load_state_dict(st)
                        print("[Checkpoint] 已恢复判别器优化器状态")
                    except Exception as e:
                        print(f"[Checkpoint] 判别器优化器状态恢复失败: {e}")

        # -------------------------
        # 优化器配置
        # -------------------------
        def configure_optimizers(self):
            # 生成器优化器：只优化生成网络(以及可选的自适应权重网络)，避免误更新判别器
            params = list(self.net.parameters())
            if self.use_adaptive_weights and hasattr(self, "adaptive_weight_net"):
                params += list(self.adaptive_weight_net.parameters())

            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

            # 判别器优化器（手动 step）
            if self.use_adversarial and (self.discriminator is not None):
                self.d_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.lr * 0.1,
                    weight_decay=self.weight_decay,
                )
                print("✅ 已创建判别器优化器（将于 training_step 内手动更新）")

            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
