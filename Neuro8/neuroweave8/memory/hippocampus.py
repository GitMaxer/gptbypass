"""Гиппокампальный эпизодический контур: DG -> CA3 -> CA1 + хранилище эпизодов.

Новое в v8 (science_grounding §§4-5: McClelland/Treves&Rolls/Diekelmann&Born).

    DG   — паттерн-сепарация: проекция входа в расширенное пространство
           (n_dg >> n_in) + жёсткая разреженность k-WTA (dg_sparsity).
           Два похожих эпизода получают почти ортогональные коды.
    CA3  — паттерн-комплишн: рекуррентный аттрактор (ca3_steps итераций)
           восстанавливает целое по частичному ключу.
    CA1  — компаратор: сравнивает восстановленное с текущим входом ->
           новизна = ||x - CA3(DG(x))||². Новизна решает, записывать ли
           эпизод, и задаёт приоритет реплея во сне (SWR-пачки).
    Store — кольцевое хранилище эпизодов (n_slots): вход, DG-код, ошибка,
           возраст. Вытеснение — по наименьшему приоритету.

Это БЫСТРАЯ память организма: одноэпизодная запись без обучения весов
(статические случайные проекции DG/CA3 — как в биологии гиппокампа,
пластичность там нестабильна по дизайну). Системная консолидация
переносит опыт в медленную кору (w_slow) через сон (learning/sleep.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HippoCfg
from ..core.sparse import k_wta


class Hippocampus(nn.Module):
    """Быстрый эпизодический субстрат: сепарация, комплишн, новизна, реплей."""

    def __init__(self, n_in: int, cfg: HippoCfg, seed: int = 4242):
        super().__init__()
        self.cfg = cfg
        self.n_in = int(n_in)
        self.n_dg = int(cfg.n_dg)
        gen = torch.Generator().manual_seed(seed)

        # DG: статическая проекция расширения (нормированная на fan-in)
        w_dg = torch.randn(self.n_dg, n_in, generator=gen) / math.sqrt(max(1, n_in))
        self.register_buffer("w_dg", w_dg)
        self.k_dg = max(2, int(cfg.dg_sparsity * self.n_dg))

        # CA3: разрежённая симметричная автоассоциативная матрица на DG-кодах
        w_ca3 = torch.randn(self.n_dg, self.n_dg, generator=gen)
        keep = torch.rand(self.n_dg, self.n_dg, generator=gen) < cfg.ca3_fan_in
        w_ca3 = w_ca3 * keep
        w_ca3.fill_diagonal_(0.0)
        # спектральный радиус < 1: иначе аттрактор расходится
        self.register_buffer("w_ca3", w_ca3 / (w_ca3.abs().max() + 1e-6) * 0.9)

        # CA1: DG-код -> пространство входа (псевдоинверсия-аппроксимация)
        self.register_buffer("w_ca1", w_dg.t() / math.sqrt(max(1, self.n_dg)))

        # Хранилище эпизодов (кольцо): x, код DG, ошибка-новизна, возраст
        self._slots_x: Optional[torch.Tensor] = None
        self._slots_code: Optional[torch.Tensor] = None
        self._slots_err: Optional[torch.Tensor] = None
        self._slots_age: Optional[torch.Tensor] = None
        self._n_stored = 0
        self._ptr = 0

    # ------------------------------------------------------------------
    def reset_state(self, device, dtype=torch.float32) -> None:
        cfg = self.cfg
        self._slots_x = torch.zeros(cfg.n_slots, self.n_in, device=device, dtype=dtype)
        self._slots_code = torch.zeros(cfg.n_slots, self.n_dg, device=device, dtype=dtype)
        self._slots_err = torch.zeros(cfg.n_slots, device=device, dtype=dtype)
        self._slots_age = torch.zeros(cfg.n_slots, device=device, dtype=dtype)
        self._n_stored = 0
        self._ptr = 0

    def _ensure(self, x: torch.Tensor) -> None:
        if self._slots_x is None or self._slots_x.device != x.device:
            self.reset_state(x.device, x.dtype)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def separate(self, x: torch.Tensor) -> torch.Tensor:
        """DG: разрежённый ортогонализованный код входа [B, n_dg]."""
        proj = F.linear(x, self.w_dg)
        return k_wta(torch.clamp(proj, min=0.0), self.k_dg)

    @torch.no_grad()
    def complete(self, code: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """CA3: аттракторное восстановление полного кода по частичному."""
        steps = self.cfg.ca3_steps if steps is None else int(steps)
        c = code.clone()
        for _ in range(steps):
            c = torch.tanh(self.cfg.ca3_gain * (c + F.linear(c, self.w_ca3)))
            c = k_wta(torch.clamp(c, min=0.0), self.k_dg)
        return c

    @torch.no_grad()
    def reconstruct(self, code: torch.Tensor) -> torch.Tensor:
        """CA1: код -> аппроксимация исходного входа."""
        return F.linear(code, self.w_ca1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Полный проход: сепарация -> комплишн -> новизна -> запись.

        Возвращает {code, novelty [B], written [B]}: эпизод пишется в
        хранилище при новизне выше порога (или перезаписью худшего слота
        при переполнении).
        """
        self._ensure(x)
        code = self.separate(x)
        rec = self.reconstruct(self.complete(code))
        novelty = ((x - rec) ** 2).sum(-1) / max(1, x.shape[-1])
        written = torch.zeros(x.shape[0], device=x.device)
        for b in range(x.shape[0]):
            nov = float(novelty[b])
            if nov >= self.cfg.novelty_threshold or self._n_stored < self.cfg.n_slots:
                if self._n_stored < self.cfg.n_slots:
                    i = self._n_stored
                    self._n_stored += 1
                else:
                    # вытеснение слота с минимальным остаточным приоритетом
                    i = int(torch.argmin(self._slots_err).item())
                self._slots_x[i] = x[b]
                self._slots_code[i] = code[b]
                self._slots_err[i] = nov
                self._slots_age[i] = 0.0
                written[b] = 1.0
        self._slots_age.add_(1.0)
        return {"code": code, "novelty": novelty, "written": written}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def replay_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """SWR-пачка: выборка эпизодов с приоритетом ~ остаточной ошибке.

        Kumaran: replay взвешен goal-dependent статистикой — у нас это
        |err| (что система не объяснила, то и повторяется).
        Возвращает (паттерны входа [n, n_in], их DG-коды [n, n_dg]).
        """
        if self._n_stored == 0:
            return (torch.zeros(0, self.n_in, device=self.w_dg.device),
                    torch.zeros(0, self.n_dg, device=self.w_dg.device))
        k = min(int(n), self._n_stored)
        w = self._slots_err[: self._n_stored].clamp(min=1e-6)
        idx = torch.multinomial(w / w.sum(), k, replacement=True)
        return self._slots_x[idx].clone(), self._slots_code[idx].clone()

    def report(self) -> Dict[str, float]:
        return {
            "stored": float(self._n_stored),
            "err_mean": float(self._slots_err[: max(1, self._n_stored)].mean()),
            "fill": float(self._n_stored) / float(self.cfg.n_slots),
        }
