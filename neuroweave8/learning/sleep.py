"""Протокол сна v8: реплей из гиппокампа + консолидация + прунинг.

Наследие v7 (разд. 7.3): AMPA-подавление, ACh->минимум, энтропия 5-HT вниз,
генеративный реплей, пороговая консолидация w_fast->w_slow (v4.0.1),
прунинг, обнуление буферов.

Отличие v8: реплей идёт НЕ на шуме/снимке входа, а на эпизодах
гиппокампа, выбранных с приоритетом пропорционально остаточной ошибке
(Kumaran: goal-dependent replay). Это системная консолидация
CLS-теории: быстрый гиппокамп учит медленную кору.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


class SleepProtocol:
    def __init__(self, cfg_sleep=None, ach_sleep: float = -0.8,
                 ach_wake: float = 0.3, sleep_input_gain: float = 0.25,
                 replay_5ht_drop: float = 0.5):
        cs = cfg_sleep
        self.ach_sleep = float(ach_sleep)
        self.ach_wake = float(ach_wake)
        self.sleep_input_gain = float(sleep_input_gain)
        self.replay_5ht_drop = float(replay_5ht_drop)
        self.n2 = getattr(cs, "n2_ticks", 16)
        self.n3 = getattr(cs, "n3_ticks", 24)
        self.rem = getattr(cs, "rem_ticks", 12)
        self.ripple_batch = getattr(cs, "ripple_batch", 8)
        self.pr_frac = getattr(cs, "pr_frac", 0.02)
        self._ht_backup: Optional[torch.Tensor] = None

    @torch.no_grad()
    def enter(self, neuro, columns: Iterable) -> None:
        """ACh -> минимум, 5-HT вниз (энтропия), DA/NE обнулены."""
        if self._ht_backup is None:
            self._ht_backup = neuro.ht.clone()
        neuro.ach.fill_(self.ach_sleep)
        neuro.set_ach.fill_(self.ach_sleep)
        neuro.ht.mul_(self.replay_5ht_drop)
        neuro.da.zero_()
        neuro.ne.zero_()

    @torch.no_grad()
    def replay(self, columns: Iterable, hippo, neuro, batch: int = 1) -> Dict[str, float]:
        """Приоритизированный реплей эпизодов гиппокампа через колонки.

        Пластичность работает, но DA = 0 (сон не захватывает случайности —
        гомеостаз только затухает следы и w_fast релаксирует).
        """
        xs, _ = hippo.replay_batch(max(1, self.ripple_batch))
        n_steps = self.n2 + self.n3 + self.rem
        total_spikes = 0.0
        for t in range(n_steps):
            for col in columns:
                if xs.shape[0] > 0:
                    x = xs[t % xs.shape[0]].unsqueeze(0)
                    if x.shape[-1] == col.cfg.n_in:
                        s_in = x.to(next(iter(col.syn.values())).w_slow.device) \
                            * self.sleep_input_gain
                    else:
                        s_in = torch.randn(batch, col.cfg.n_in) * 0.1 * self.sleep_input_gain
                else:
                    s_in = torch.randn(batch, col.cfg.n_in) * 0.1 * self.sleep_input_gain
                out = col.step(s_in, ach=torch.full((batch, 1), self.ach_sleep),
                               sltt=False, learn=True)
                total_spikes += float(out.spikes["l23"].sum() + out.spikes["l5"].sum())
            neuro.step()
        return {"replay_steps": float(n_steps), "replay_spikes": total_spikes,
                "episodes": float(xs.shape[0])}

    @torch.no_grad()
    def consolidate(self, columns: Iterable) -> float:
        """Пороговая консолидация всех синапсов (v4.0.1, ур. 21)."""
        moved = 0.0
        for col in columns:
            for syn in col.syn.values():
                rep = syn.consolidate_to_slow()
                moved += rep["moved_mass"]
        return moved

    @torch.no_grad()
    def prune(self, columns: Iterable, threshold: float = 0.004) -> int:
        """Структурный прунинг неподтверждённых синапсов."""
        n = 0
        for col in columns:
            for syn in col.syn.values():
                n += syn.prune(threshold)
        return n

    @torch.no_grad()
    def wake(self, neuro) -> None:
        neuro.ach.fill_(self.ach_wake)
        neuro.set_ach.fill_(self.ach_wake)
        if self._ht_backup is not None:
            neuro.ht.copy_(self._ht_backup)
            self._ht_backup = None
        neuro.sleep_pressure = 0.0

    @torch.no_grad()
    def run_cycle(self, columns: Iterable, neuro, hippo,
                  prune_threshold: float = 0.004) -> Dict[str, float]:
        """Полный цикл: погружение -> реплей -> консолидация -> прунинг
        -> пробуждение."""
        cols = list(columns)
        self.enter(neuro, cols)
        rep = self.replay(cols, hippo, neuro)
        rep["consolidated"] = self.consolidate(cols)
        rep["pruned"] = float(self.prune(cols, prune_threshold))
        for col in cols:
            for syn in col.syn.values():
                syn.reset_state(1, syn.w_slow.device)
        self.wake(neuro)
        return rep
