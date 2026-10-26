"""Трёхсторонний синапс и астроцитарная кальциевая сеть — порт из Spiker v5.0.

Третья шкала памяти между w_fast (мс) и w_slow (консолидация):
  * астроциты интегрируют спайковую и ACh/NE-активность микродомена в
    медленную Ca2+-динамику с tau_ca ~ 2500 мс;
  * латеральная щелевая диффузия (gap junctions) даёт пространственную
    кальциевую волну;
  * глиотрансмиттер мультипликативно модулирует пресинаптическое
    высвобождение (STP-фасилитация), удерживая контекст активного ансамбля
    строго O(N) по времени и памяти.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class TripartiteAstrocyte(nn.Module):
    """Астроцитарный контур рабочей памяти большого горизонта (tau_ca ~ 2500 мс).

    Уравнение кальциевой волны:
        tau_ca * dCa/dt = -Ca + alpha_s*s + alpha_ne*NE(t)*s + D_gap*Lap(Ca)
    Глиотрансмиттерный гейт:
        g_astro(t) = 1 + gamma_gli * tanh(Ca / theta_ca)
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,
        tau_ca: float = 2500.0,
        alpha_spike: float = 0.35,
        alpha_ne_boost: float = 1.5,
        gamma_gli: float = 0.65,
        theta_ca: float = 0.20,
        diffusion_rate: float = 0.05,
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.tau_ca = float(tau_ca)
        self.alpha_spike = float(alpha_spike)
        self.alpha_ne_boost = float(alpha_ne_boost)
        self.gamma_gli = float(gamma_gli)
        self.theta_ca = float(theta_ca)
        self.diffusion_rate = float(diffusion_rate)

        self.register_buffer(
            "decay_ca", torch.exp(torch.tensor(-dt / tau_ca)))
        self.ca_level: Optional[torch.Tensor] = None
        self.gliotransmitter: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32) -> None:
        shape = (batch_size, self.num_neurons)
        self.ca_level = torch.zeros(shape, device=device, dtype=dtype)
        self.gliotransmitter = torch.zeros(shape, device=device, dtype=dtype)

    def detach_state(self) -> None:
        if self.ca_level is not None:
            self.ca_level = self.ca_level.detach()
            self.gliotransmitter = self.gliotransmitter.detach()

    @torch.no_grad()
    def step(
        self,
        zone_spikes: torch.Tensor,
        ne_level: Optional[torch.Tensor] = None,
        ach_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Один такт dt интеграции астроцитарного кальция.

        zone_spikes: [B, N] спайки микродомена; ne_level/ach_level усиливают
        захват событий через Gq-рецепторы (IP3 -> Ca2+).
        Возвращает ca_level, gliotransmitter и stp_gain [B, N].
        """
        if self.ca_level is None or self.ca_level.shape != zone_spikes.shape:
            self.reset_state(zone_spikes.shape[0], zone_spikes.device,
                             zone_spikes.dtype)

        s = zone_spikes.detach()

        mod_boost = 1.0
        if ne_level is not None:
            mod_boost = mod_boost + self.alpha_ne_boost * torch.clamp(
                ne_level.detach(), min=0.0)
        if ach_level is not None:
            mod_boost = mod_boost + 0.5 * torch.clamp(
                ach_level.detach(), min=0.0)

        drive = self.alpha_spike * mod_boost * s

        # Латеральная диффузия через щелевые контакты между соседними астроцитами
        if self.num_neurons > 2 and self.diffusion_rate > 0.0:
            left_nb = torch.roll(self.ca_level, shifts=1, dims=-1)
            right_nb = torch.roll(self.ca_level, shifts=-1, dims=-1)
            laplacian = 0.5 * (left_nb + right_nb) - self.ca_level
        else:
            laplacian = 0.0

        # Точное экспоненциальное интегрирование медленной кальциевой волны
        self.ca_level = (
            self.decay_ca * (self.ca_level + self.diffusion_rate * laplacian)
            + (1.0 - self.decay_ca) * drive
            * (self.tau_ca / max(1.0, self.dt * 20.0))
        )
        self.ca_level = torch.clamp(self.ca_level, min=0.0, max=5.0)

        # Высвобождение глиотрансмиттера (D-серин + астроцитарный глутамат)
        self.gliotransmitter = torch.tanh(
            self.ca_level / max(1e-6, self.theta_ca))

        # Мультипликативный гейт кратковременной фасилитации (STP WM)
        stp_gain = 1.0 + self.gamma_gli * self.gliotransmitter

        return {
            "ca_level": self.ca_level,
            "gliotransmitter": self.gliotransmitter,
            "stp_gain": stp_gain,
        }
