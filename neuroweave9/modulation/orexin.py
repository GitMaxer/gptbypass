"""Орексиновый интероцептивный контур — порт из Spiker v7.0.

Орексин (гипокретин) — сигнал энергетического дефицита и давления
незавершённых целей: метаболическая стоимость активности истощает energy;
при дефиците включается орексиновый всплеск, который (а) поднимает целевой
ACh (бодрствование), (б) в v9 — снижает порог базальных ганглиев, чтобы
«голодная» сеть решала быстрее (эффект голодания на скорость решений,
Gold & Shadlen 2007 в духе снижения bound).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class OrexinInteroceptiveLoop(nn.Module):
    def __init__(
        self,
        dt: float = 1.0,
        activity_cost: float = 0.001,
        recovery: float = 0.004,
        goal_gain: float = 0.02,
        goal_decay: float = 0.01,
        orexin_threshold: float = 0.35,
        orexin_amplitude: float = 1.0,
        orexin_ach_gain: float = 0.4,
    ):
        super().__init__()
        self.dt = float(dt)
        self.activity_cost = float(activity_cost)
        self.recovery = float(recovery)
        self.goal_gain = float(goal_gain)
        self.goal_decay = float(goal_decay)
        self.orexin_threshold = float(orexin_threshold)
        self.orexin_amplitude = float(orexin_amplitude)
        self.orexin_ach_gain = float(orexin_ach_gain)
        self.energy: Optional[torch.Tensor] = None
        self.goal_energy: Optional[torch.Tensor] = None
        self.orexin: Optional[torch.Tensor] = None
        self.goals: List[str] = []

    def reset_state(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu"),
                    dtype: torch.dtype = torch.float32) -> None:
        shape = (batch_size, 1)
        self.energy = torch.ones(shape, device=device, dtype=dtype)
        self.goal_energy = torch.zeros(shape, device=device, dtype=dtype)
        self.orexin = torch.zeros(shape, device=device, dtype=dtype)
        self.goals = []

    def detach_state(self) -> None:
        for attr in ("energy", "goal_energy", "orexin"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.detach())

    def register_goal(self, goal_id: str) -> None:
        if goal_id not in self.goals:
            self.goals.append(goal_id)

    def satisfy_goal(self, goal_id: str) -> None:
        if goal_id in self.goals:
            self.goals.remove(goal_id)

    @property
    def has_unfinished_goals(self) -> bool:
        return len(self.goals) > 0

    @property
    def unfinished_goals(self) -> List[str]:
        return list(self.goals)

    @torch.no_grad()
    def step(
        self,
        activity: Optional[torch.Tensor] = None,
        goal_pressure: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.energy is None:
            if activity is None:
                raise RuntimeError("OrexinInteroceptiveLoop: сначала reset_state()")
            self.reset_state(
                activity.shape[0] if activity.dim() > 0 else 1,
                activity.device, activity.dtype)

        act = (torch.zeros_like(self.energy) if activity is None
               else activity.reshape(-1, 1).to(self.energy.dtype))

        spend = self.dt * self.activity_cost * act
        rest = self.dt * self.recovery * (1.0 - act).clamp(min=0.0)
        self.energy = torch.clamp(self.energy - spend + rest, min=0.0, max=1.0)

        if self.has_unfinished_goals:
            growth = self.dt * self.goal_gain * (1.0 + act)
        else:
            growth = -self.dt * self.goal_decay * self.goal_energy
        if goal_pressure is not None:
            growth = growth + self.dt * self.goal_gain * goal_pressure.reshape(-1, 1)
        self.goal_energy = torch.clamp(self.goal_energy + growth, min=0.0, max=3.0)

        deficit = torch.clamp(
            (self.orexin_threshold - self.energy)
            / max(self.orexin_threshold, 1e-6), min=0.0, max=1.0)
        drive = torch.clamp(deficit + self.goal_energy, min=0.0, max=1.0)
        self.orexin = self.orexin_amplitude * drive

        return {"energy": self.energy, "goal_energy": self.goal_energy,
                "orexin": self.orexin,
                "ach_boost": self.orexin_ach_gain * self.orexin}

    def state_summary(self) -> Dict[str, float]:
        if self.energy is None:
            return {"energy": 1.0, "goal_energy": 0.0, "orexin": 0.0,
                    "unfinished_goals": 0.0}
        return {"energy": float(self.energy.mean()),
                "goal_energy": float(self.goal_energy.mean()),
                "orexin": float(self.orexin.mean()),
                "unfinished_goals": float(len(self.goals))}
