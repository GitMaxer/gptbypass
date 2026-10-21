"""Структурная пластичность: синаптогенез, прунинг, критический период.

science_grounding §8 (Hensch; Holtmaat&Svoboda; Caroni):
  * окно пластичности управляется тормозным gain и NE (стресс открывает
    окно заново), а не календарём;
  * turnover-фаза ПЕРЕД консолидацией: рост кандидатов пропорционален
    ПРОГРЕССУ обучения (dL/dt), а не абсолютной ошибке.
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch


class StructuralPlasticity:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = 0
        self.prev_loss: float = 0.0
        self.progress_ema = 0.0

    @property
    def window_open(self) -> bool:
        return self.cfg.enabled and self.t < self.cfg.critical_ticks

    def maybe_reopen(self, ne_level: float) -> bool:
        """NE-стресс вновь открывает критическое окно."""
        if self.cfg.enabled and float(ne_level) > self.cfg.ne_reopen:
            self.t = min(self.t, self.cfg.critical_ticks // 2)
            return True
        return False

    @torch.no_grad()
    def step(self, columns: Iterable, loss: float) -> Dict[str, int]:
        """Один цикл структурного обновления (вызывается медленным контуром)."""
        self.t += 1
        progress = self.prev_loss - float(loss)
        self.prev_loss = float(loss)
        self.progress_ema = 0.9 * self.progress_ema + 0.1 * progress
        grown = 0
        if self.window_open and self.progress_ema > 0.0:
            for col in columns:
                for syn in col.syn.values():
                    if float(syn.mask.mean()) >= self.cfg.max_edges_frac * \
                            syn.mask.numel() / syn.mask.numel():
                        pass
                    n_cand = max(1, int(self.cfg.grow_rate * syn.n_in))
                    cand = torch.randint(0, syn.n_out, (n_cand, 1)).expand(n_cand, 2).clone()
                    cand[:, 1] = torch.randint(0, syn.n_in, (n_cand,))
                    grown += syn.grow(cand, w_init=0.01)
        return {"grown": grown, "progress": self.progress_ema,
                "window_open": float(self.window_open)}
