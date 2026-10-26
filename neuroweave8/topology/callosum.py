"""Corpus Callosum v8: буферизованная асинхронная шина латентов.

Наследие v7 (все патчи канона v4.0.1):
  * кольцевой буфер задержки с ТЕНЗОРНЫМ указателем и чтением через
    index_select (КОПИЯ — патч S7: view-алиасинг уничтожал задержку);
  * разреженный GABA-фильтр (гладкий гейт амплитуды, без ветвлений);
  * асимметричные проекции L->R / R->L (разные матрицы, разные гейны);
  * detach на входе кольца (шина вне градиента Фазы 1).

Отличие v8: сообщения — латенты КОЛОНОК (пакет на область = выход L6 +
surprise-код), а шина общая для всего организма (не «межполушарная»
только) — любой маршрут `routes` с видом 'fb' может идти через шину,
что даёт настраиваемую топологию отложенных контекстных путей.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorpusCallosum(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, delay: int = 7,
                 density: float = 0.25, theta_bus: float = 0.30,
                 k_bus: float = 10.0, gain_fwd: float = 1.0,
                 gain_bwd: float = 0.6, seed: int = 2026):
        super().__init__()
        if not (5 <= delay <= 10):
            raise ValueError("delay вне канонического 5-10")
        self.dim_in, self.dim_out, self.delay = int(dim_in), int(dim_out), int(delay)
        self.theta_bus, self.k_bus = float(theta_bus), float(k_bus)
        gen = torch.Generator().manual_seed(seed)
        bound = 1.0 / math.sqrt(dim_in)
        self.w_fwd = nn.Parameter(torch.empty(dim_out, dim_in).uniform_(-bound, bound, generator=gen))
        self.w_bwd = nn.Parameter(torch.empty(dim_in, dim_out).uniform_(-bound, bound, generator=gen))
        self.register_buffer("mask_fwd", (torch.rand(dim_out, dim_in, generator=gen) < density).float())
        self.register_buffer("mask_bwd", (torch.rand(dim_in, dim_out, generator=gen) < density).float())
        self.register_buffer("gain_fwd", torch.tensor(float(gain_fwd)))
        self.register_buffer("gain_bwd", torch.tensor(float(gain_bwd)))
        self.register_buffer("_ptr", torch.zeros(1, dtype=torch.long))
        self._buf_fwd: Optional[torch.Tensor] = None
        self._buf_bwd: Optional[torch.Tensor] = None
        self._filtered: Optional[torch.Tensor] = None
        # Адаптивная миелинизация (Spiker): обучаемые веса задержек по
        # глубине кольцевого буфера. Инициализация — пик на максимальной
        # задержке (контракт: до обучения лаг = delay тактов).
        init_m = torch.full((self.delay, self.dim_in), -10.0)
        init_m[0, :] = 10.0
        self.register_buffer("myelin_fwd", init_m.clone())
        init_m2 = torch.full((self.delay, self.dim_out), -10.0)
        init_m2[0, :] = 10.0
        self.register_buffer("myelin_bwd", init_m2.clone())
        self.adaptive_myelination: bool = False

    def enable_adaptive_myelination(self, on: bool = True) -> None:
        """Включить мягкое смешивание задержек: задержанный пакет =
        softmax(myelin)-взвешенная сумма слотов кольца (обучаемый лаг)."""
        self.adaptive_myelination = bool(on)

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self._buf_fwd = torch.zeros(self.delay, batch, self.dim_in, device=device, dtype=dtype)
        self._buf_bwd = torch.zeros(self.delay, batch, self.dim_out, device=device, dtype=dtype)
        self._ptr.fill_(0)

    def _ensure(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if self._buf_fwd is None or self._buf_fwd.shape[1] != a.shape[0] \
                or self._buf_fwd.device != a.device:
            self.reset_state(a.shape[0], a.device, a.dtype)

    def _gaba(self, m: torch.Tensor) -> torch.Tensor:
        return m * torch.sigmoid(self.k_bus * (m.abs() - self.theta_bus))

    @torch.no_grad()
    def step(self, m_fwd: torch.Tensor, m_bwd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Один такт: вернуть (доставленное fwd, доставленное bwd)."""
        self._ensure(m_fwd, m_bwd)
        # чтение ДО записи в тот же слот — копия через index_select (патч S7)
        if self.adaptive_myelination and self.delay > 1:
            # Миелинизация: мягкий выбор лага — от самого старого (ptr,
            # лаг=delay) к самому новому (ptr-1, лаг=1).
            idx = [(int(self._ptr.item()) + k) % self.delay
                   for k in range(self.delay)]
            ord_f = self._buf_fwd[idx]            # [D, B, dim_in]
            ord_b = self._buf_bwd[idx]            # [D, B, dim_out]
            w_f = torch.softmax(self.myelin_fwd, dim=0).unsqueeze(1)
            w_b = torch.softmax(self.myelin_bwd, dim=0).unsqueeze(1)
            out_fwd = (ord_f * w_f).sum(dim=0)
            out_bwd = (ord_b * w_b).sum(dim=0)
        else:
            out_fwd = self._buf_fwd.index_select(0, self._ptr)[0]
            out_bwd = self._buf_bwd.index_select(0, self._ptr)[0]
        self._buf_fwd.index_copy_(0, self._ptr, m_fwd.detach().unsqueeze(0))
        self._buf_bwd.index_copy_(0, self._ptr, m_bwd.detach().unsqueeze(0))
        self._ptr.copy_((self._ptr + 1) % self.delay)
        f_fwd = self._gaba(out_fwd)
        f_bwd = self._gaba(out_bwd)
        self._filtered = f_fwd
        to_dst = F.linear(f_fwd, self.w_fwd * self.mask_fwd) * self.gain_fwd
        to_src = F.linear(f_bwd, self.w_bwd * self.mask_bwd) * self.gain_bwd
        return to_dst, to_src

    @torch.no_grad()
    def credit(self, delivered: torch.Tensor, filtered: torch.Tensor,
               da: torch.Tensor, lr: float = 0.02, amp: float = 0.3,
               direction: str = "fwd") -> None:
        """Транзитный 3-факторный кредит проекции (v7 step_bus)."""
        w = self.w_fwd if direction == "fwd" else self.w_bwd
        m = self.mask_fwd if direction == "fwd" else self.mask_bwd
        da_pos = torch.clamp(da.detach().float(), min=0.0)
        delta = lr * da_pos * torch.einsum("bo,bi->oi", delivered.detach(),
                                           filtered.detach()) / max(1, delivered.shape[0])
        w.add_(delta * m)
        w.clamp_(-amp, amp)

    @torch.no_grad()
    def myelin_update(self, insight_da: float, lr: float = 0.01) -> None:
        """Обучение задержек: REM-инсайт сдвигает миелиновые логиты к лагам,
        давшим когерентный мост (простая политика направления — по одному
        шагу к более короткому лагу на единицу инсайта)."""
        g = float(insight_da)
        if g <= 0.0:
            return
        grad = torch.linspace(0.0, 1.0, self.delay).unsqueeze(-1)
        self.myelin_fwd.add_(lr * g * grad)
        self.myelin_bwd.add_(lr * g * grad)

    @property
    def traffic(self) -> float:
        return 0.0 if self._filtered is None else float(self._filtered.abs().mean())
