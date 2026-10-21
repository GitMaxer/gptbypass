"""Кредит-ассайнмент Фазы 2: критик RPE + событийные DA-импульсы.

Наследие v7 (патчи канона v4.0.1):
  * RewardCritic: V — бегущая частота успеха; DA = max(0, R-V) (ур. 14);
  * СОБЫТИЙНАЯ семантика импульса (S4): observe_reward накапливает
    амплитуду в pending [Z], step_column раздаёт её СЛЕДУЮЩЕГО такта;
    удерживаемый уровень DA не мультиплицирует следы;
  * СИНАПСОСПЕЦИФИЧНАЯ нормировка (S6): импульс делится на средний
    масштаб живого следа КАЖДОГО синапса.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch


class RewardCritic:
    def __init__(self, tau_v: float = 150.0, dt: float = 1.0):
        self.v = 0.0
        self._rate = 1.0 - math.exp(-dt / tau_v)

    def rpe(self, reward: float) -> float:
        rpe = float(reward) - self.v
        self.v += self._rate * (float(reward) - self.v)
        return rpe

    @property
    def baseline(self) -> float:
        return self.v


class CreditPipeline:
    """Tag-and-Capture над StatefulSynapse с событийным DA."""

    def __init__(self, zone_names, device="cpu", lr_fast: float = 3e-3,
                 tau_v: float = 150.0, a_plus: float = 1.0, a_minus: float = 0.8):
        self.names = list(zone_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.critic = RewardCritic(tau_v=tau_v)
        self.lr_fast = float(lr_fast)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.pending = torch.zeros(len(self.names), device=device)
        self.stats = {"reward_events": 0.0, "da_total": 0.0, "rpe_last": 0.0}

    def reset_episode(self) -> None:
        self.pending.zero_()

    def observe_reward(self, reward: float,
                       weights: Optional[Dict[str, float]] = None) -> float:
        """Одно наблюдение критика -> импульсы в pending (следующий такт)."""
        rpe = self.critic.rpe(reward)
        da = max(0.0, rpe)
        if da > 0.0:
            for n in self.names:
                w = 1.0 if weights is None else float(weights.get(n, 1.0))
                if w > 0.0:
                    self.pending[self.index[n]] += da * w
        self.stats["reward_events"] += 1.0
        self.stats["da_total"] += da
        self.stats["rpe_last"] = rpe
        return da

    @torch.no_grad()
    def step_column(self, col, s_in: torch.Tensor, s_exc_post: Dict[str, torch.Tensor],
                    lr_scale: float = 1.0) -> torch.Tensor:
        """Раздать импульс зоны пластичности её синапсов (тензорно)."""
        i = self.index[col.name]
        imp = self.pending[i].detach()
        self.pending[i] = 0.0
        lr = self.lr_fast * lr_scale
        pairs = [("in_4", s_in, s_exc_post["l4"]),
                 ("4_23", col.prev["l4"], s_exc_post["l23"]),
                 ("rec", col.prev["l23"], s_exc_post["l23"]),
                 ("23_5", col.prev["l23"], s_exc_post["l5"]),
                 ("5_6", col.prev["l5"], s_exc_post["l6"]),
                 ("6_5", col.prev["l6"], s_exc_post["l5"])]
        # Следы и затухание w_fast обновляются ВСЕГДА (гомеостаз Y1);
        # imp=0 -> da=0 -> delta=0, остаётся только затухание.
        for key, pre, post in pairs:
            syn = col.syn[key]
            e = syn.e_pre
            mean_e = e.detach().abs().mean() if e is not None else torch.zeros_like(imp)
            da = (imp / (1.0 + mean_e)).clamp(max=1.0)
            syn.update_traces_and_plasticity(
                pre.detach(), post.detach(), da=da,
                lr=lr, a_plus=self.a_plus, a_minus=self.a_minus)
        return imp
