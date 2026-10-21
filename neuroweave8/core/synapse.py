"""Синапс с состоянием: две шкалы памяти + честные O(N)-следы пригодности.

w_slow  — кристаллизованный базис (Фаза 1, суррогатный градиент; замораживается
          перед сборкой организма).
w_fast  — оперативный буфер (Фаза 2, трёхфакторное правило под DA-гейтом),
          амплитудно ограничен и затухает КАЖДЫЙ такт (иначе буфер «съедает»
          базис — дефект Y1 канона v4.0.1).

Следы пригодности факторизованы: хранятся 4 вектора O(N), а не матрица
O(N^2). Обновление при приходе дофамина — внешнее произведение векторов
(einsum ранга 1), поэтому память строго линейна по числу нейронов.

СТРУКТУРНАЯ ПЛАСТИЧНОСТЬ (новое в v8): маска ``mask`` — не константа, а
живой объект. ``grow``/``prune`` меняют её между эпизодами, реализуя
синаптогенез и прунинг. Маска входит в эффективный вес, поэтому «мёртвый»
синапс не даёт ни тока, ни следа.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PlasticityCfg


class StatefulSynapse(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        cfg: PlasticityCfg,
        density: float = 1.0,
        name: str = "syn",
        trainable_slow: bool = True,
        dt: float = 1.0,
        sign: str = "exc",
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.cfg = cfg
        self.name = name
        self.sign = sign

        gen = generator or torch.Generator().manual_seed(1234 + n_in * 7919 + n_out)
        # Fan-in нормировка: дисперсия драйва инвариантна размеру пула
        scale = 1.0 / math.sqrt(max(1, n_in))
        w0 = torch.randn(n_out, n_in, generator=gen) * scale
        if sign == "exc":
            # Возбуждающий синапс не бывает отрицательным: знак в коре задаётся
            # типом клетки, а не знаком веса. Это и физиология, и защита
            # интегратора от отрицательной проводимости (иначе denom < 0 и
            # точная экспонента расходится в NaN).
            w0 = w0.abs()

        mask = (torch.rand(n_out, n_in, generator=gen) < density).float()
        # Каждой строке — хотя бы один вход (иначе нейрон структурно мёртв)
        mask[torch.rand(n_out, generator=gen) < 0.02] = 1.0
        self.register_buffer("mask", mask)

        w0 = w0 * mask
        self.w_slow = nn.Parameter(w0, requires_grad=trainable_slow)
        self.register_buffer("w_fast", torch.zeros(n_out, n_in))
        self.register_buffer("w_snapshot", w0.clone())
        self.register_buffer("age", torch.zeros(n_out, n_in, dtype=torch.long))
        self.register_buffer("frozen", torch.tensor(0.0))

        self.register_buffer("dec_pre", torch.exp(torch.tensor(-dt / cfg.tau_pre)))
        self.register_buffer("dec_post", torch.exp(torch.tensor(-dt / cfg.tau_post)))
        self.register_buffer("dec_elig", torch.exp(torch.tensor(-dt / cfg.tau_elig)))

        self.z_pre: Optional[torch.Tensor] = None
        self.z_post: Optional[torch.Tensor] = None
        self.e_pre: Optional[torch.Tensor] = None
        self.e_post: Optional[torch.Tensor] = None
        self._last_delta: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self.z_pre = torch.zeros(batch, self.n_in, device=device, dtype=dtype)
        self.z_post = torch.zeros(batch, self.n_out, device=device, dtype=dtype)
        self.e_pre = torch.zeros(batch, self.n_in, device=device, dtype=dtype)
        self.e_post = torch.zeros(batch, self.n_out, device=device, dtype=dtype)

    def _ensure(self, ref: torch.Tensor) -> None:
        if (self.z_pre is None or self.z_pre.shape[0] != ref.shape[0]
                or self.z_pre.device != ref.device or self.z_pre.dtype != ref.dtype):
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    # ------------------------------------------------------------------
    @property
    def w_eff(self) -> torch.Tensor:
        w = (self.w_slow + self.w_fast) * self.mask
        if self.sign == "exc":
            return torch.clamp(w, min=0.0)
        return w

    def forward(self, s_pre: torch.Tensor) -> torch.Tensor:
        """Постсинаптический драйв проводимости: [B, n_in] -> [B, n_out]."""
        return F.linear(s_pre, self.w_eff)

    def drive_sparse(self, s_pre: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Драйв только для подмножества пост-нейронов (событийный путь)."""
        w = self.w_eff.index_select(0, idx)
        return F.linear(s_pre, w)

    # ------------------------------------------------------------------
    def freeze_for_phase2(self) -> None:
        """Контракт Фазы 2: w_slow вне Autograd + неизменяемый снапшот."""
        self.w_slow.requires_grad_(False)
        with torch.no_grad():
            self.w_snapshot.copy_(self.w_slow)
        self.frozen.fill_(1.0)

    def unfreeze(self) -> None:
        self.w_slow.requires_grad_(True)
        self.frozen.fill_(0.0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_traces_and_plasticity(
        self,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
        da: Optional[torch.Tensor] = None,
        lr: Optional[float] = None,
        a_plus: Optional[float] = None,
        a_minus: Optional[float] = None,
        learn: bool = True,
    ) -> torch.Tensor:
        """Трёхфакторное правило: пре-след x пост-спайк x дофамин.

        Следы O(N): z_* — быстрые спайковые, e_* — медленные пригодностные.
        Обновление весов — внешнее произведение (einsum), без матрицы E_ij.
        Затухание буфера выполняется ВСЕГДА (не только при награде).
        """
        self._ensure(s_pre)
        cfg = self.cfg
        lr = cfg.lr_fast if lr is None else lr
        a_plus = cfg.a_plus if a_plus is None else a_plus
        a_minus = cfg.a_minus if a_minus is None else a_minus

        self.z_pre = self.z_pre * self.dec_pre + s_pre
        self.z_post = self.z_post * self.dec_post + s_post
        self.e_pre = self.e_pre * self.dec_elig + self.z_pre
        self.e_post = self.e_post * self.dec_elig + self.z_post

        if da is None:
            da_pos = torch.zeros((), device=s_pre.device)
        else:
            da_pos = torch.clamp(da if da.dim() == 0 else da.mean(), min=0.0)

        ltp = torch.einsum("bo,bi->oi", s_post * da_pos, self.e_pre)
        ltd = torch.einsum("bo,bi->oi", self.e_post * da_pos, s_pre)
        delta = lr * (a_plus * ltp - a_minus * ltd) / float(s_pre.shape[0])

        if learn:
            self.w_fast.mul_(cfg.w_fast_decay).add_(delta * self.mask)
            self.w_fast.clamp_(-cfg.w_fast_max, cfg.w_fast_max)
            self.age.add_(1)
        self._last_delta = delta
        return delta

    # ------------------------------------------------------------------
    @torch.no_grad()
    def consolidate_to_slow(
        self,
        theta_c: Optional[float] = None,
        temp: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> Dict[str, float]:
        """Пороговая консолидация: перенос «подтвердившегося», а не копия.

        g = sigmoid((|w_fast| - theta_c)/T)
        w_slow += eta * g * w_fast ;  w_fast -= w_fast * g

        Уравнение (21) канона в форме Sign(w)|w| вырождено (это просто w) —
        здесь используется мягкий порог доверия, как предписано v4.0.1.
        """
        cfg = self.cfg
        theta_c = cfg.theta_consol * cfg.w_fast_max if theta_c is None else theta_c
        temp = cfg.temp_consol * cfg.w_fast_max if temp is None else temp
        eta = cfg.eta_consol if eta is None else eta

        trust = torch.sigmoid((self.w_fast.abs() - theta_c) / max(temp, 1e-6))
        trust = trust * self.mask
        moved = eta * trust * self.w_fast
        with torch.no_grad():
            self.w_slow.add_(moved)
            self.w_slow.clamp_(-cfg.w_max, cfg.w_max)
            self.w_fast.sub_(trust * self.w_fast)
        return {
            "moved_mass": float(moved.abs().sum()),
            "trust_mean": float(trust.mean()),
            "w_fast_mass": float(self.w_fast.abs().sum()),
        }

    @torch.no_grad()
    def decay_fast(self) -> None:
        self.w_fast.mul_(self.cfg.w_fast_decay).mul_(self.mask)

    # ------------------------------------------------------------------
    # Структурная пластичность
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune(self, threshold: float) -> int:
        """Удалить синапсы, чей вес так и не подтвердился (возраст > grace)."""
        weak = (self.w_slow.abs() + self.w_fast.abs() < threshold) & (self.mask > 0)
        weak = weak & (self.age > 0)
        n = int(weak.sum())
        self.mask.mul_((~weak).float())
        self.w_fast.mul_(self.mask)
        return n

    @torch.no_grad()
    def grow(self, candidates: torch.Tensor, w_init: float = 0.01) -> int:
        """Добавить синапсы по индексам-кандидатам (n_new, 2) = (out, in)."""
        cand = candidates[(self.mask[candidates[:, 0], candidates[:, 1]] == 0)]
        if cand.numel() == 0:
            return 0
        n = cand.shape[0]
        self.mask[cand[:, 0], cand[:, 1]] = 1.0
        self.w_slow.data[cand[:, 0], cand[:, 1]] = w_init
        self.w_fast[cand[:, 0], cand[:, 1]] = 0.0
        self.age[cand[:, 0], cand[:, 1]] = 0
        return n

    def edge_count(self) -> int:
        return int(self.mask.sum())

    def structural_report(self) -> Dict[str, float]:
        return {
            "edges": float(self.mask.sum()),
            "density": float(self.mask.mean()),
            "w_slow_mass": float(self.w_slow.abs().sum()),
            "w_fast_mass": float(self.w_fast.abs().sum()),
            "snapshot_drift": float((self.w_slow - self.w_snapshot).abs().max()),
        }
