"""Активный вывод (Friston): внимание как выбор под expected free energy.

science_grounding §7: политика p(a) = softmax(-γ·G(a)), где
    G(a) = pragmatic + epistemic
    pragmatic(a) = -E[reward]   →  внимание к области, которая работает
    epistemic(a)  = -novelty    →  внимание к области, которая удивляет

В спайковой реализации «действие» = селективный прирост arousal
(неспецифическая L1-активация области): softmax(-γG) становится
коэффициентом усиления, γ связан с уровнем NA (точность политики).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ActiveInference:
    def __init__(self, n_areas: int, gamma: float = 4.0, gain: float = 0.5):
        self.n = int(n_areas)
        self.gamma = float(gamma)
        self.gain = float(gain)

    @torch.no_grad()
    def gains(self, competence: torch.Tensor, novelty: torch.Tensor) -> torch.Tensor:
        """Веса внимания [n] из компетентности (track) и новизны (surprise)."""
        n = min(self.n, competence.numel(), novelty.numel())
        g_prag = -competence[:n].float()
        g_epis = -novelty[:n].float()
        p = F.softmax(-self.gamma * (g_prag + g_epis), dim=-1)
        return self.gain * p * n  # среднее усиление = gain

    def policy(self, competence: torch.Tensor, novelty: torch.Tensor) -> torch.Tensor:
        """Дискретная политика: распределение внимания (телеметрия)."""
        n = min(self.n, competence.numel(), novelty.numel())
        return F.softmax(-self.gamma * (-competence[:n] - novelty[:n]), -1)
