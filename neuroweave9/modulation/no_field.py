"""NO-поле (оксид азота) — диффузионный сигнал объёма — порт из Spiker v6.0.

NO — ретроградный газовый посредник, свободно диффундирующий через мембраны
(не требует синаптической структуры): пространственное размытие локального
спайкового паттерна создаёт объёмный (volume) сигнал, сообщающий соседям
о локальной активности без весов. В v9 привязан к спайкам L2/3 колонки C —
поле размывает правополушарный код и экспортируется как мягкий контекст.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NitricOxideField(nn.Module):
    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,
        tau_ms: float = 30.0,
        diffusion_coeff: float = 0.25,
        kernel_radius: int = 8,
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.tau_ms = max(1.0, float(tau_ms))
        self.diffusion_coeff = float(diffusion_coeff)
        self.field: Optional[torch.Tensor] = None
        self.register_buffer("decay", torch.exp(torch.tensor(-dt / self.tau_ms)))

        k_size = 2 * kernel_radius + 1
        sigma = max(1.0, kernel_radius / 2.0)
        pos = torch.arange(k_size, dtype=torch.float32) - kernel_radius
        kernel = torch.exp(-0.5 * (pos / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer("diffusion_kernel", kernel.view(1, 1, k_size))
        self.kernel_radius = kernel_radius

    def reset_state(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu"),
                    dtype: torch.dtype = torch.float32) -> None:
        self.field = torch.zeros(
            (batch_size, self.num_neurons), device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        if (self.field is None or self.field.shape != spikes.shape
                or self.field.device != spikes.device):
            self.reset_state(spikes.shape[0], spikes.device, spikes.dtype)

        field = self.decay * self.field + spikes.detach()
        padded = F.pad(field.unsqueeze(1), (self.kernel_radius, self.kernel_radius),
                       mode="replicate")
        diffused = F.conv1d(padded, self.diffusion_kernel.to(field.dtype)).squeeze(1)
        self.field = torch.clamp(
            (1.0 - self.diffusion_coeff) * field + self.diffusion_coeff * diffused,
            min=0.0, max=4.0)
        return {"field": self.field, "field_energy": self.field.pow(2).mean()}
