"""Global Workspace v8: конкуренция областей за выход + метакогнитивные заявки.

Наследие v7: латеральное GABA-торможение (ур. 9), Soft-WTA, storm-brake.

Усиление v8 — заявки компетентности (факт Этапа 6 v7: «уверенность !=
компетентность»): bid = confidence * (eps + track), где track — бегущий
EMA-протокол точности области, переживающий эпизоды. Уверенно-
ошибающийся эксперт больше не топит fused-выход.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspace(nn.Module):
    def __init__(self, n_areas: int, gamma_lat: float = 0.6,
                 beta_focus: float = 6.0, storm_rate: float = 0.35,
                 storm_gain: float = 2.5, competence_rho: float = 0.12,
                 competence_floor: float = 0.1, seed: int = 2027):
        super().__init__()
        if n_areas < 2:
            raise ValueError("конкуренция требует >= 2 областей")
        self.n = int(n_areas)
        self.gamma_lat = float(gamma_lat)
        self.beta_focus = float(beta_focus)
        self.storm_rate = float(storm_rate)
        self.storm_gain = float(storm_gain)
        self.rho = float(competence_rho)
        self.floor = float(competence_floor)
        gen = torch.Generator().manual_seed(seed)
        w = torch.rand(self.n, self.n, generator=gen)
        w.fill_diagonal_(0.0)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("w_gaba", w)
        self.register_buffer("track", torch.full((self.n,), 0.5))

    def lateral(self, bids: torch.Tensor) -> torch.Tensor:
        return self.gamma_lat * (bids @ self.w_gaba.T)

    def compete(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bids.shape[-1] != self.n:
            raise ValueError("bids != n_areas")
        eff = bids.clamp(min=0.0) - self.lateral(bids.clamp(min=0.0))
        alpha = F.softmax(self.beta_focus * eff, dim=-1)
        return alpha, eff.argmax(-1), eff

    def calibrated_bids(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Заявки = уверенность чтения x (floor + track-протокол)."""
        conf = torch.stack([torch.softmax(l, -1).max(-1).values for l in logits], -1)
        return conf * (self.floor + self.track.unsqueeze(0))

    @torch.no_grad()
    def update_track(self, logits: List[torch.Tensor], target: torch.Tensor) -> None:
        for i, l in enumerate(logits):
            corr = float((l.argmax(-1) == target).float().mean())
            self.track[i].mul_(1.0 - self.rho).add_(self.rho * corr)

    def fuse(self, alpha: torch.Tensor, logits: List[torch.Tensor]) -> torch.Tensor:
        return torch.einsum("bz,bzv->bv", alpha, torch.stack(logits, 1))

    def storm_brake(self, activity: torch.Tensor) -> torch.Tensor:
        cap = self.storm_rate
        total = activity.sum(-1, keepdim=True)
        brake = 1.0 / (1.0 + F.relu(total - cap) / cap)
        return activity * brake
