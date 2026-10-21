"""Базальные ганглии: Drift-Diffusion гейт эмиссии токенов — порт из Spiker v6.0.

Прямой путь Go (D1) / непрямой No-Go (D2) + субталамическое ядро (STN):
накопитель свидетельств интегрирует margin(P_top1 - P_top2) с утечкой;
нейрохимический порог theta(t) растёт с 5-HT (терпение) и ACh (осторожность);
при достижении порога — дискретная эмиссия и сброс аккумулятора.

В v9 «Cortex» гейт задаёт ТЕМП решения (физиологическая цена ~2-5 мс/слово);
выбор токена остаётся за argmax декодера — ограничение задокументировано в
ARCHITECTURE_REVIEW §8.2 и сохранено здесь без изменений.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasalGangliaGate(nn.Module):
    """Дрифт-диффузионный гейт базальных ганглиев (Go/No-Go + STN)."""

    def __init__(
        self,
        dt: float = 1.0,
        base_threshold: float = 1.20,
        leak_rate: float = 0.05,
        alpha_5ht: float = 0.60,
        alpha_ach: float = 0.40,
        stn_conflict_gain: float = 0.35,
    ):
        super().__init__()
        self.dt = float(dt)
        self.base_threshold = float(base_threshold)
        self.leak_rate = float(leak_rate)
        self.alpha_5ht = float(alpha_5ht)
        self.alpha_ach = float(alpha_ach)
        self.stn_conflict_gain = float(stn_conflict_gain)
        self.evidence_acc: Optional[torch.Tensor] = None
        self.total_emissions: int = 0

    def reset_state(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu"),
                    dtype: torch.dtype = torch.float32) -> None:
        self.evidence_acc = torch.zeros((batch_size, 1), device=device, dtype=dtype)

    @torch.no_grad()
    def forward(
        self,
        workspace_logits: torch.Tensor,
        serotonin_5ht: Optional[torch.Tensor] = None,
        ach_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = workspace_logits.shape[0]
        device = workspace_logits.device
        dtype = workspace_logits.dtype

        if (self.evidence_acc is None
                or self.evidence_acc.shape[0] != batch_size
                or self.evidence_acc.device != device):
            self.reset_state(batch_size, device, dtype)

        probs = F.softmax(workspace_logits.detach(), dim=-1)
        if probs.shape[-1] >= 2:
            top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)
            margin = (top2_vals[:, 0:1] - top2_vals[:, 1:2]).clamp(min=0.0)
            selected_token = top2_idx[:, 0]
            stn_conflict = (1.0 - margin) * top2_vals[:, 1:2]
        else:
            margin = probs[:, 0:1]
            selected_token = torch.zeros(batch_size, dtype=torch.long, device=device)
            stn_conflict = torch.zeros((batch_size, 1), device=device, dtype=dtype)

        drift = margin - self.stn_conflict_gain * stn_conflict - self.leak_rate * self.evidence_acc
        self.evidence_acc = torch.clamp(self.evidence_acc + self.dt * drift, min=0.0, max=10.0)
        evidence_before_reset = self.evidence_acc.clone()

        mod_factor = torch.ones((batch_size, 1), device=device, dtype=dtype)
        if serotonin_5ht is not None:
            s5ht = serotonin_5ht.detach().mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
            mod_factor = mod_factor + self.alpha_5ht * (s5ht - 0.5)
        if ach_level is not None:
            ach_pos = torch.relu(ach_level.detach().mean(dim=-1, keepdim=True)).clamp(0.0, 1.0)
            mod_factor = mod_factor + self.alpha_ach * ach_pos

        threshold = (self.base_threshold * mod_factor).clamp(min=0.20, max=5.0)

        emit_mask = (self.evidence_acc >= threshold).to(dtype)
        emitted = float(emit_mask.sum().item())
        self.total_emissions += int(emitted)
        self.evidence_acc = self.evidence_acc * (1.0 - emit_mask)

        return {
            "emit_mask": emit_mask,
            "selected_token": selected_token,
            "evidence": evidence_before_reset,
            "threshold": threshold,
            "margin": margin,
        }
