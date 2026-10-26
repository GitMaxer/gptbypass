"""Таламическое реле с тормозной оболочкой TRN и сонными веретёнами — порт из Spiker.

Бодрствование: TRN оценивает вход и решает, пропустить ли его в кору —
селективный прожектор внимания (surprise/ACh/NE снимают торможение).
Сон: TRN полностью блокирует сенсорный поток (изоляция коры) и генерирует
ритмические веретёна (~14 Гц), синхронизирующие SWR-реплей.
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Optional


class Thalamus(nn.Module):
    """Таламическое реле с тормозной оболочкой TRN и генератором веретён."""

    def __init__(
        self,
        input_dim: int,
        dt: float = 1.0,
        spindle_period_ms: float = 70.0,
        baseline_trn_inhibition: float = 0.15,
        surprise_relief_gain: float = 1.5,
        ach_relief_gain: float = 0.25,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dt = float(dt)
        self.spindle_period_ms = max(5.0, float(spindle_period_ms))
        self.baseline_trn_inhibition = float(baseline_trn_inhibition)
        self.surprise_relief_gain = float(surprise_relief_gain)
        self.ach_relief_gain = float(ach_relief_gain)

        self.is_sleeping: bool = False
        self.register_buffer("spindle_phase", torch.tensor(0.0))
        self.trn_gate: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spindle_phase = torch.zeros((), device=device, dtype=dtype)
        self.trn_gate = torch.ones((batch_size, 1), device=device, dtype=dtype)
        self.is_sleeping = False

    def enter_sleep(self) -> None:
        """Сенсорная блокада TRN + режим генерации сонных веретён."""
        self.is_sleeping = True

    def wake(self) -> None:
        """Снимает блокаду сна, возвращая таламус в режим сенсорного реле."""
        self.is_sleeping = False

    def forward(
        self,
        sensory_input: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        ach_level: Optional[torch.Tensor] = None,
        ne_level: Optional[torch.Tensor] = None,
        oscillator_gain: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Пропускает сенсорный вход [B, input_dim] через шлюз TRN."""
        batch_size = sensory_input.shape[0]
        device = sensory_input.device
        dtype = sensory_input.dtype

        if (self.trn_gate is None or self.trn_gate.shape[0] != batch_size
                or self.trn_gate.device != device):
            self.trn_gate = torch.ones((batch_size, 1), device=device, dtype=dtype)
            self.spindle_phase = self.spindle_phase.to(device=device, dtype=dtype)

        two_pi = 2.0 * math.pi
        self.spindle_phase = torch.remainder(
            self.spindle_phase + two_pi * (self.dt / self.spindle_period_ms),
            two_pi)
        spindle_wave = 0.5 * (1.0 + torch.cos(self.spindle_phase))

        if self.is_sleeping:
            # Во сне TRN полностью блокирует сенсорный поток, пропуская в кору
            # только ритм веретён для синхронизации SWR-реплея.
            trn_inhibition = torch.ones((batch_size, 1), device=device, dtype=dtype)
            relay_gain = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            self.trn_gate = relay_gain
            return {
                "relayed_input": torch.zeros_like(sensory_input),
                "trn_inhibition": trn_inhibition,
                "relay_gain": relay_gain,
                "spindle_wave": spindle_wave,
            }

        # Бодрствование: торможение TRN и его снятие при surprise / ACh / NE
        relief = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if surprise is not None:
            s_val = surprise.detach()
            if s_val.dim() == 1:
                s_val = s_val.unsqueeze(-1)
            relief = relief + self.surprise_relief_gain * s_val
        if ach_level is not None:
            ach_mean = ach_level.detach().mean(dim=-1, keepdim=True)
            relief = relief + self.ach_relief_gain * torch.clamp(
                ach_mean, min=-1.0, max=1.0)
        if ne_level is not None:
            ne_mean = ne_level.detach().mean(dim=-1, keepdim=True)
            relief = relief + 0.35 * torch.clamp(ne_mean, min=0.0, max=1.0)

        trn_inhibition = torch.clamp(
            self.baseline_trn_inhibition - relief, min=0.0, max=0.60)
        attention_boost = torch.clamp(
            relief - self.baseline_trn_inhibition, min=0.0, max=0.15)
        relay_gain = (1.0 - trn_inhibition + attention_boost).clamp(0.40, 1.15)

        if oscillator_gain is not None:
            osc_factor = 0.85 + 0.15 * oscillator_gain.to(
                device=device, dtype=dtype)
            relay_gain = relay_gain * osc_factor

        self.trn_gate = relay_gain.detach()
        relayed = sensory_input * relay_gain

        return {
            "relayed_input": relayed,
            "trn_inhibition": trn_inhibition,
            "relay_gain": relay_gain,
            "spindle_wave": spindle_wave,
        }
