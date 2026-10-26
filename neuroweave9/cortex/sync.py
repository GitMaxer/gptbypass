"""Межполушарная синхронизация v9: θ-γ биндер + фазовая VSA поверх Callosum.

Это точка «The Great Merge», которой нет ни в одном из предков:

  * Из Spiker взят измеренный фазовый протокол: θ-осциллятор открывает
    окно Corpus Callosum (callosum_phase_gate) — пакеты идут на пике θ.
  * Из NEUROWEAVE 8 взята phase.py: ThetaGammaBinder (слоты рабочей
    памяти = γ-подциклы) и PhaseVSA (связывание через сложение фаз).

Каждый такт оба полушария пишут свои коды в ОБЩИЙ биндер на соседние
γ-слоты — θ-цикл становится общей шкалой времени обоих полушарий.
Поверх них PhaseVSA строит межполушарный фрейм

    frame = bind(L, role_L) + bundle bind(R, role_R)

— связанную репрезентацию «левый аналитический код И правый контекстный
код» одновременно. Это и есть синхронизация не по сигналу, а по
СТРУКТУРЕ: два полушария становятся двумя ролями одного фрейма.

Экспортируемые признаки (канал синхронизации декодера):
  * frame.real — сплющенный межполушарный вектор (реляционный канал);
  * coherence — косинусная близость проекций L/R (семантический градус
    согласия полушарий в этот такт);
  * occupancy — заполнение слотов биндера (телеметрия WM).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroweave8.config import PhaseCfg
from neuroweave8.core.phase import PhaseVSA, ThetaGammaBinder


class HemisphereSync(nn.Module):
    """Общий θ-γ биндер + VSA-фрейм для кодов левого и правого полушарий."""

    def __init__(
        self,
        code_dim: int,
        vsa_dim: int = 512,
        n_slots: int = 7,
        gamma_ticks: int = 4,
        seed: int = 909,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.vsa_dim = int(vsa_dim)
        self.binder = ThetaGammaBinder(
            code_dim, PhaseCfg(n_slots=n_slots, gamma_ticks=gamma_ticks))
        gen = torch.Generator().manual_seed(seed)
        self.vsa = PhaseVSA(
            PhaseCfg(vsa_dim=vsa_dim, vsa_codebook=256), generator=gen)
        # Проекции спайкового кода в пространство фазового паттерна.
        self.proj_l = nn.Linear(code_dim, vsa_dim)
        self.proj_r = nn.Linear(code_dim, vsa_dim)
        with torch.no_grad():
            for p in (self.proj_l, self.proj_r):
                p.weight.mul_(2.0 / math.sqrt(code_dim))
                p.bias.zero_()
        # Фазовый сдвиг полушарий: правое кодирует в отставшей фазе θ-цикла
        # (половина γ-слота ~ задержка проведения Callosum). Связывание
        # учитывает его через расширенный ключ role_R * phase_R.
        self.hemi_phase_r = math.pi / max(1, n_slots)
        self.register_buffer(
            "phase_r", torch.polar(torch.ones(1),
                                   torch.tensor([self.hemi_phase_r])))
        self.frame: Optional[torch.Tensor] = None
        self.coherence = 0.0

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self.binder.reset_state(batch, device, dtype)
        self.frame = None
        self.coherence = 0.0

    def _ensure(self, ref: torch.Tensor) -> None:
        if (self.binder.buf is None or self.binder.buf.shape[0] != ref.shape[0]
                or self.binder.buf.device != ref.device):
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    @torch.no_grad()
    def step(
        self,
        left_code: torch.Tensor,
        right_code: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Один такт межполушарной синхронизации.

        left_code/right_code: [B, code_dim] — репрезентационный код области
        (спайки L2/3). Оба пишутся в общий биндер на соседние γ-слоты.
        """
        self._ensure(left_code)
        # Общая θ-шкала: L занимает чётный γ-слот, R — следующий нечётный.
        self.binder.write(left_code.detach())
        self.binder.write(right_code.detach())

        l = torch.sigmoid(self.proj_l(left_code.detach()))
        r = torch.sigmoid(self.proj_r(right_code.detach()))
        enc_l = self.vsa.encode_pattern(l.clamp(1e-4, 1 - 1e-4).gt(0.5).float(),
                                        role_idx=0)
        # Правое полушарие: собственная роль + фазовый сдвиг на полслота
        enc_r = (self.vsa.encode_pattern(
            r.clamp(1e-4, 1 - 1e-4).gt(0.5).float(), role_idx=1)
            * self.phase_r)
        self.frame = self.vsa.bundle([enc_l, enc_r])

        # Когерентность между VSA-НАПОЛНИТЕЛЯМИ (содержанием), а не между
        # проекциями: развязываем каждый код его ключом (роль x фаза) и
        # сравниваем чистые фазовые векторы — это структурная, а не
        # сигнальная согласованность полушарий.
        key_r = self.vsa.role(1) * self.phase_r
        fill_l = self.vsa.unbind(enc_l, self.vsa.role(0))
        fill_r = self.vsa.unbind(enc_r, key_r)
        coh = (fill_l * torch.conj(fill_r)).real.mean(dim=-1)
        self.coherence = float(coh.mean())
        strength = float(self.frame.real.pow(2).mean())

        return {
            "frame_real": self.frame.real,
            "coherence": torch.tensor(self.coherence),
            "sync_strength": torch.tensor(strength),
            "occupancy": torch.tensor(self.binder.occupancy()),
        }

    def features(self) -> torch.Tensor:
        """Реляционный канал для декодера: вещественная часть фрейма."""
        if self.frame is None:
            raise RuntimeError("sync.step() ещё не вызывался")
        return self.frame.real
