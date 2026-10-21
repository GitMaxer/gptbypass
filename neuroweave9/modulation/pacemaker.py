"""Таламический пейсмейкер θ-γ — порт из Spiker v5.0/v6.0.

Кросс-частотное связывание (Phase-Amplitude Coupling, Lisman & Jensen 2013):
быстрый γ-ритм задаёт такты локальной обработки левого полушария, медленный
θ-ритм связывает γ-циклы в семантический фрейм правого и открывает фазовое
окно передачи Corpus Callosum. ``phase_reset`` — тета-мастер медиальной
перегородки: «один токен = один тета-цикл».
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class ThetaGammaOscillator(nn.Module):
    def __init__(
        self,
        dt: float = 1.0,
        theta_period_ms: float = 24.0,
        gamma_period_ms: float = 4.0,
        gamma_coupling_strength: float = 0.25,
        theta_coupling_strength: float = 0.30,
    ):
        super().__init__()
        self.dt = dt
        self.theta_period_ms = max(2.0, float(theta_period_ms))
        self.gamma_period_ms = max(1.0, float(gamma_period_ms))
        self.gamma_coupling_strength = gamma_coupling_strength
        self.theta_coupling_strength = theta_coupling_strength
        self.register_buffer("phase_theta", torch.tensor(0.0))
        self.register_buffer("phase_gamma", torch.tensor(0.0))

    def reset_state(self, device: torch.device = torch.device("cpu")) -> None:
        self.phase_theta = torch.zeros((), device=device)
        self.phase_gamma = torch.zeros((), device=device)

    @torch.no_grad()
    def phase_reset(self, target_phase: float = 0.0) -> None:
        self.phase_theta.fill_(float(target_phase))
        self.phase_gamma.fill_(float(target_phase))

    @torch.no_grad()
    def step(self) -> Dict[str, torch.Tensor]:
        two_pi = 2.0 * math.pi
        self.phase_theta = torch.remainder(
            self.phase_theta + two_pi * (self.dt / self.theta_period_ms), two_pi)
        self.phase_gamma = torch.remainder(
            self.phase_gamma + two_pi * (self.dt / self.gamma_period_ms), two_pi)

        theta_wave = torch.cos(self.phase_theta)
        gamma_wave = torch.cos(self.phase_gamma)
        pac_envelope = 0.5 * (1.0 + theta_wave)
        coupled_gamma = pac_envelope * gamma_wave

        return {
            "phase_theta": self.phase_theta,
            "phase_gamma": self.phase_gamma,
            "theta_wave": theta_wave,
            "gamma_wave": gamma_wave,
            "pac_envelope": pac_envelope,
            "left_gamma_gain": 1.0 + self.gamma_coupling_strength * coupled_gamma,
            "right_theta_gain": 1.0 + self.theta_coupling_strength * theta_wave,
            "callosum_phase_gate": 0.5 + 0.5 * pac_envelope,
        }
