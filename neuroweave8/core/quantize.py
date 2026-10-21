"""Квантизация синапсов: int8-режим инференса организма.

Симметричная построчная (per-post-neuron) схема: каждая строка весовой
матрицы получает свой масштаб, поэтому динамический диапазон подстраивается
под конкретный нейрон. Хранится квантизованный снимок ``w_eff``
(= (w_slow + w_fast) * mask с уже применённым знаковым клэмпом) —
квантизованный синапс заморожен по определению: это режим развёртывания
после обучения и консолидации, а не режим обучения.

Эффективный вес де квантизируется НА ЛЕТУ (``w_eff`` -> float32):
проводимостная математика нейрона остаётся в полной точности, экономится
только память хранения (~4x при int8). Для честного событийного пути
поддерживается ``drive_sparse``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .synapse import StatefulSynapse


class QuantizedSynapse(nn.Module):
    """Замороженный int8-снимок StatefulSynapse с интерфейсом оригинала."""

    def __init__(self, syn: StatefulSynapse, bits: int = 8):
        super().__init__()
        self.n_in, self.n_out = syn.n_in, syn.n_out
        self.cfg = syn.cfg
        self.name = syn.name
        self.sign = syn.sign
        self.bits = int(bits)
        if not 2 <= self.bits <= 16:
            raise ValueError("bits должен быть в [2, 16]")

        qmax = float(2 ** (self.bits - 1) - 1)
        w = syn.w_eff.detach().float()
        # Построчный масштаб: строка = все входы одного постсинаптического
        # нейрона. Нулевая строка получает scale=1 (де квантизация даст 0).
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        self.register_buffer("w_scale", scale.clamp(min=1e-8))
        q = torch.round(w / self.w_scale).clamp(-qmax, qmax)
        self.register_buffer("w_q", q.to(torch.int8))
        # Маска бинарна: bool-буфер (1 байт/ребро) — иначе float32-маска
        # съедает всю экономию квантизации.
        self.register_buffer("mask", syn.mask.detach().to(torch.bool))

        # Интерфейс совместимости: состояния следов пустые, но существуют
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

    @property
    def w_eff(self) -> torch.Tensor:
        w = self.w_q.float() * self.w_scale * self.mask.float()
        if self.sign == "exc":
            return torch.clamp(w, min=0.0)
        return w

    def forward(self, s_pre: torch.Tensor) -> torch.Tensor:
        return F.linear(s_pre, self.w_eff)

    def drive_sparse(self, s_pre: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        w = self.w_eff.index_select(0, idx)
        return F.linear(s_pre, w)

    # ------------------------------------------------------------------
    # Пластичность выключена: квантизованный снимок заморожен.
    # Методы сохранены как безопасные no-op, чтобы горячий контур мог
    # работать с learn=False без веток по типу синапса.
    # ------------------------------------------------------------------
    def update_traces_and_plasticity(self, s_pre, s_post, da=None, lr=None,
                                     a_plus=None, a_minus=None, learn=True):
        self._ensure(s_pre)
        self.z_pre = self.z_pre * self.cfg_decay("tau_pre") + s_pre
        self.z_post = self.z_post * self.cfg_decay("tau_post") + s_post
        return torch.zeros(self.n_out, self.n_in, device=s_pre.device)

    def _ensure(self, ref: torch.Tensor) -> None:
        if self.z_pre is None or self.z_pre.shape[0] != ref.shape[0]:
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    def cfg_decay(self, name: str) -> float:
        import math
        return math.exp(-1.0 / getattr(self.cfg, name))

    def consolidate_to_slow(self, **kw) -> Dict[str, float]:
        return {"moved_mass": 0.0, "trust_mean": 0.0,
                "w_fast_mass": 0.0}

    def decay_fast(self) -> None:
        pass

    def prune(self, threshold: float) -> int:
        return 0

    def grow(self, candidates: torch.Tensor, w_init: float = 0.01) -> int:
        return 0

    def freeze_for_phase2(self) -> None:
        pass

    def unfreeze(self) -> None:
        pass

    def edge_count(self) -> int:
        return int(self.mask.sum())

    def memory_bytes(self) -> int:
        return (self.w_q.numel() * self.w_q.element_size()
                + self.w_scale.numel() * 4 + self.mask.numel() * 1)

    def structural_report(self) -> Dict[str, float]:
        return {"edges": float(self.mask.sum()),
                "density": float(self.mask.mean()),
                "quantized_bits": float(self.bits)}


# ---------------------------------------------------------------------------
def quantize_organism(org, bits: int = 8) -> Dict[str, float]:
    """Заменить все StatefulSynapse колонок на int8-снимки.

    Возвращает статистику: байты до/после и максимальную абсолютную
    ошибку эффективного веса. Пластичность выключена — вызывать после
    обучения и сна (инференс-развёртывание).
    """
    before, after, max_err = 0, 0, 0.0
    for col in org.columns.values():
        for key in list(col.syn.keys()):
            syn = col.syn[key]
            if not isinstance(syn, StatefulSynapse):
                continue
            # Полный объём оригинала на ребро: w_slow + w_fast + w_snapshot
            # + mask (4B каждый) + age (8B long) — против 2B у снимка.
            before += syn.w_eff.numel() * 16 + syn.age.numel() * 8
            w_ref = syn.w_eff.detach()
            qs = QuantizedSynapse(syn, bits=bits)
            after += qs.memory_bytes()
            max_err = max(max_err, float((qs.w_eff - w_ref).abs().max()))
            col.syn[key] = qs
    return {"bytes_before": float(before), "bytes_after": float(after),
            "compression": before / max(1, after), "max_abs_err": max_err,
            "bits": float(bits)}
