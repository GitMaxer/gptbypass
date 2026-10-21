"""Лапласова память / Спектральный банк временных констант — порт из Spiker v6.0.

Механизм «клеток времени» (Shankar & Howard, 2012): геометрический спектр
экспоненциальных интеграторов tau_k = tau_min * c^k. Состояние F(s_k, t) —
дискретное вещественное преобразование Лапласа входного потока; память строго
O(K * D), не растёт с длиной последовательности. Профиль по шкалам кодирует,
на каком шаге назад появился каждый токен (разностный фильтр Поста-Виддера).

В v9 «Cortex» это — плотный мультимасштабный контекстный канал декодера:
спайковая кора даёт параметро-эффективный разреженный код, а спектр Лапласа
добавляет покрытие длинного контекста (в Spiker: кора 25.1% + спектр 28.3%
-> 29.9% top-1 на пробе признаков).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class LaplaceTimingBank(nn.Module):
    """Масштабно-инвариантный банк Лапласовой памяти (порт Spiker v6.0)."""

    def __init__(
        self,
        feature_dim: int,
        num_scales: int = 8,
        tau_min: float = 4.0,
        tau_max: float = 512.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_scales = max(2, int(num_scales))
        self.tau_min = max(1.0, float(tau_min))
        self.tau_max = max(self.tau_min * 2.0, float(tau_max))
        self.dt = float(dt)

        log_taus = torch.linspace(
            math.log(self.tau_min), math.log(self.tau_max), steps=self.num_scales)
        taus = torch.exp(log_taus)
        alphas = torch.exp(-self.dt / taus).view(self.num_scales, 1, 1)

        self.register_buffer("taus", taus)
        self.register_buffer("alphas", alphas)
        self.laplace_state: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu"),
                    dtype: torch.dtype = torch.float32) -> None:
        self.laplace_state = torch.zeros(
            (self.num_scales, batch_size, self.feature_dim),
            device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, x_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x_input.shape[0]
        if (self.laplace_state is None
                or self.laplace_state.shape[1] != batch_size
                or self.laplace_state.device != x_input.device):
            self.reset_state(batch_size, x_input.device, x_input.dtype)

        x_exp = x_input.detach().unsqueeze(0)
        self.laplace_state = self.alphas * self.laplace_state + x_exp

        left_scale = torch.roll(self.laplace_state, shifts=1, dims=0)
        right_scale = torch.roll(self.laplace_state, shifts=-1, dims=0)
        time_cells = torch.relu(self.laplace_state - 0.5 * (left_scale + right_scale))
        context_summary = self.laplace_state.mean(dim=0)

        return {
            "laplace_state": self.laplace_state,
            "time_cells": time_cells,
            "context_summary": context_summary,
        }

    @torch.no_grad()
    def estimate_elapsed_time(self, feature_idx: int, batch_idx: int = 0) -> float:
        if self.laplace_state is None:
            return 0.0
        col = self.laplace_state[:, batch_idx, feature_idx].clamp_min(1e-8)
        k_fast, k_slow = 1, self.num_scales - 1
        inv_diff = (1.0 / float(self.taus[k_fast])) - (1.0 / float(self.taus[k_slow]))
        if abs(inv_diff) < 1e-9:
            return 0.0
        ratio = float((col[k_slow] / col[k_fast]).item())
        if ratio <= 1.0:
            return 0.0
        return max(0.0, math.log(ratio) / inv_diff)
