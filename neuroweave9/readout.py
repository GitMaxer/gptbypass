"""LaplaceContextReadout — декодер v9 «Cortex» (порт из Spiker v6.0).

Декодер поверх объединённого состояния коры [s_A, s_B, s_C] (спайки L2/3
ламинарных колонок), мультимасштабного спектра LaplaceTimingBank и
межполушарного VSA-фрейма синхронизации. Обучается Next-Token CE; гейт
``alpha_zone_a`` учится балансировать контекстный канал и прямые логиты
колонки A. Память O(1) по времени.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LaplaceContextReadout(nn.Module):
    def __init__(
        self,
        cortical_dim: int,
        laplace_dim: int,
        vocab_size: int,
        sync_dim: int = 0,
        lr: float = 0.02,
        log_prior: Optional[torch.Tensor] = None,
        prior_weight: float = 0.0,
        feature_rms: Optional[float] = None,
        hidden: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cortical_dim + laplace_dim + sync_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )
        self.alpha_zone_a = nn.Parameter(torch.tensor(0.75))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.prior_weight = float(prior_weight)
        self.feature_rms = float(feature_rms) if feature_rms is not None else None
        if log_prior is not None:
            self.register_buffer("log_prior", log_prior.reshape(1, -1).float())
        else:
            self.log_prior = None

    def forward(
        self,
        cortical_spikes: torch.Tensor,
        laplace_flat: torch.Tensor,
        zone_a_logits: torch.Tensor,
        sync_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [cortical_spikes.detach(), laplace_flat.detach()]
        if sync_frame is not None:
            parts.append(sync_frame.detach())
        features = torch.cat(parts, dim=-1)
        if self.feature_rms is not None:
            rms = features.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            features = features * (self.feature_rms / rms)
        ctx_logits = self.net(features)
        gate = torch.sigmoid(self.alpha_zone_a)
        logits = gate * zone_a_logits.detach() + (1.0 - gate) * ctx_logits
        if self.log_prior is not None and self.prior_weight != 0.0:
            logits = logits + self.prior_weight * self.log_prior
        return logits
