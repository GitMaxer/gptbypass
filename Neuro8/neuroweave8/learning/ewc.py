"""EWC (диагональный Фишер) — защита консолидированных навыков.

Канон v4.0.1: штраф активен ТОЛЬКО между циклами сна; после каждого сна
rebaseline() переснимает снапшот на новый w_slow (иначе EWC боролся бы
с консолидацией w_fast->w_slow). Фишер нормируется на max — иначе штраф
микроскопичен против CE ~ 1.5.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch


class EWCRegularizer:
    def __init__(self, params: Dict[str, torch.nn.Parameter], lam: float = 50.0,
                 normalize_fisher: bool = True):
        for name, p in params.items():
            if not p.requires_grad:
                raise ValueError(f"EWC-параметр {name!r} без requires_grad")
        self._params = dict(params)
        self.lam = float(lam)
        self.normalize = bool(normalize_fisher)
        self.snapshot = {k: p.detach().clone() for k, p in params.items()}
        self.fisher = {k: torch.zeros_like(p) for k, p in params.items()}
        self._armed = False

    def consolidate_snapshot(self) -> None:
        for k, p in self._params.items():
            self.snapshot[k] = p.detach().clone()
        self._armed = True

    @torch.enable_grad()
    def estimate_fisher(self, loss_fn: Callable[[], torch.Tensor],
                        n_samples: int = 64) -> float:
        acc = {k: torch.zeros_like(p) for k, p in self._params.items()}
        for _ in range(int(n_samples)):
            for p in self._params.values():
                if p.grad is not None:
                    p.grad = None
            loss_fn().backward()
            for k, p in self._params.items():
                if p.grad is not None:
                    acc[k] += p.grad.detach() ** 2
        for k in acc:
            self.fisher[k] = acc[k] / max(1, n_samples)
            if self.normalize:
                m = self.fisher[k].max()
                if float(m) > 0:
                    self.fisher[k] = self.fisher[k] / m
        self._armed = True
        return self.fisher_mass()

    def penalty(self) -> torch.Tensor:
        if not self._armed:
            return torch.zeros(())
        return sum((self.fisher[k] * (p - self.snapshot[k]) ** 2).sum()
                   for k, p in self._params.items()) * self.lam

    def rebaseline(self) -> None:
        self.consolidate_snapshot()

    def fisher_mass(self) -> float:
        return float(sum(f.sum() for f in self.fisher.values()))

    @property
    def is_armed(self) -> bool:
        return self._armed
