"""Компартментный ламинарный ALIF-нейрон NEUROWEAVE 8.

Три сегмента: базальный дендрит (AMPA), апикальный дендрит (NMDA с Mg2+-
блоком), сома с адаптивным порогом. Проводимостная схема — см.
core/transmitters.py (инвариант V >= V_rest при E_Cl >= V_rest).

ДВА РЕЖИМА ПАМЯТИ (новое в v8, снимает разрыв v6 <-> v7):

  * ``step(..., sltt=True)``  — обучение (Spatio-Temporal Learning Through
    Time): состояние детачится на каждом такте, все операции out-of-place.
    Граф не растёт с длиной последовательности => память O(1) по времени,
    градиент течёт только через текущие синаптические токи.
  * ``step(..., sltt=False)`` — жизнь: состояние пишется in-place
    (``copy_``/``mul_``/``add_``). Это обязательное условие захвата шага в
    ``torch.cuda.CUDAGraph``: иначе граф читает «старую» память и счётчики
    не двигаются (анти-паттерн ``t = t + 1``).

Оба режима — ОДНА И ТА ЖЕ арифметика; их побитовая эквивалентность
проверяется тестом ``test_core_physics.py::test_sltt_matches_inplace``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import NeuronCfg
from .surrogate import spike_fn
from .transmitters import e_cl_effective, exact_step


class LaminarALIF(nn.Module):
    """Пул идентичных по параметрам, но независимых по состоянию нейронов."""

    def __init__(self, n: int, cfg: NeuronCfg, name: str = "pool"):
        super().__init__()
        self.n = int(n)
        self.cfg = cfg
        self.name = name
        self.e_cl_base = cfg.e_cl
        # Крутизна суррогата — медленный параметр (ставится мета-контроллером)
        self.k = float(cfg.k_surrogate)

        self.register_buffer("dec_ampa", torch.exp(torch.tensor(-cfg.dt / cfg.tau_ampa)))
        self.register_buffer("dec_nmda", torch.exp(torch.tensor(-cfg.dt / cfg.tau_nmda)))
        self.register_buffer("dec_gaba", torch.exp(torch.tensor(-cfg.dt / cfg.tau_gaba)))
        self.register_buffer("alpha_a", torch.exp(torch.tensor(-cfg.dt / cfg.tau_a)))

        self.v: Optional[torch.Tensor] = None
        self.a: Optional[torch.Tensor] = None
        self.refr: Optional[torch.Tensor] = None
        self.g_ampa: Optional[torch.Tensor] = None
        self.g_nmda: Optional[torch.Tensor] = None
        self.g_gaba: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        shp = (batch, self.n)
        self.v = torch.full(shp, self.cfg.v_rest, device=device, dtype=dtype)
        self.a = torch.zeros(shp, device=device, dtype=dtype)
        self.refr = torch.zeros(shp, device=device, dtype=torch.long)
        self.g_ampa = torch.zeros(shp, device=device, dtype=dtype)
        self.g_nmda = torch.zeros(shp, device=device, dtype=dtype)
        self.g_gaba = torch.zeros(shp, device=device, dtype=dtype)

    def _ensure(self, ref: torch.Tensor) -> None:
        if (self.v is None or self.v.shape != ref.shape
                or self.v.device != ref.device or self.v.dtype != ref.dtype):
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    def detach_state(self) -> None:
        """O(1)-память по времени: история предыдущих тактов вне графа."""
        for attr in ("v", "a", "g_ampa", "g_nmda", "g_gaba"):
            t = getattr(self, attr)
            if t is not None:
                setattr(self, attr, t.detach())

    # ------------------------------------------------------------------
    def step(
        self,
        raw_basal: torch.Tensor,
        raw_apical: Optional[torch.Tensor] = None,
        raw_gaba: Optional[torch.Tensor] = None,
        ach: Optional[torch.Tensor] = None,
        ht: Optional[torch.Tensor] = None,
        ne_rise: Optional[torch.Tensor] = None,
        theta_shift: Optional[torch.Tensor] = None,
        sltt: bool = True,
        k_spike: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Один такт dt.

        raw_* — ПРИРАЩЕНИЯ ПРОВОДИМОСТИ (спайки x веса), не токи.
        Возвращает dict: spikes, v, v_inf, theta, a.
        """
        self._ensure(raw_basal)
        if sltt:
            self.detach_state()
            inplace = False
        else:
            inplace = True

        cfg = self.cfg
        zero = torch.zeros_like(self.g_ampa)

        if inplace:
            self.g_ampa.mul_(self.dec_ampa).add_(raw_basal)
            self.g_nmda.mul_(self.dec_nmda)
            if raw_apical is not None:
                self.g_nmda.add_(raw_apical)
            self.g_gaba.mul_(self.dec_gaba)
            if raw_gaba is not None:
                self.g_gaba.add_(raw_gaba)
            g_ampa, g_nmda, g_gaba = self.g_ampa, self.g_nmda, self.g_gaba
        else:
            g_ampa = self.g_ampa * self.dec_ampa + raw_basal
            g_nmda = self.g_nmda * self.dec_nmda + (raw_apical if raw_apical is not None else zero)
            g_gaba = self.g_gaba * self.dec_gaba + (raw_gaba if raw_gaba is not None else zero)

        # --- проводимости неотрицательны по определению ---
        # Клэмп — ЧАСТЬ КОНТРАКТА, а не защита от ошибки: при отрицательной
        # суммарной проводимости denom = 1 + G_e + g_GABA может стать < 0,
        # lam < 0, exp(-lam dt) -> inf, и точный интегратор даёт NaN.
        g_ampa = g_ampa.clamp(min=0.0)
        g_nmda = g_nmda.clamp(min=0.0)
        g_gaba = g_gaba.clamp(min=0.0)
        if inplace:
            self.g_ampa.clamp_(min=0.0)
            self.g_nmda.clamp_(min=0.0)
            self.g_gaba.clamp_(min=0.0)

        # --- фронтальный сброс усталости норадреналином (событие, не уровень) ---
        if ne_rise is not None:
            keep = torch.clamp(1.0 - ne_rise, 0.0, 1.0)
            a_pre = self.a * keep
        else:
            a_pre = self.a

        # --- порог с ACh-возбудимостью и адаптацией ---
        theta = cfg.v_th + cfg.beta_adapt * a_pre
        if ach is not None:
            theta = theta - cfg.k_ach_th * torch.clamp(ach, -1.0, 1.0)
        if theta_shift is not None:
            # Дендритный кальциевый плато (Larkum et al.): апикальная
            # деполяризация СНИЖАЕТ порог сомы, разрешая пачечный ответ.
            # Это и есть AND-гейт «есть вход И есть контекст» на уровне
            # области, в отличие от простого суммирования токов.
            theta = theta - theta_shift

        # --- шум: свойство организма, не артефакт обучения ---
        sigma = None
        if cfg.stochastic and cfg.sigma > 0.0:
            base = torch.as_tensor(cfg.sigma, dtype=g_ampa.dtype, device=g_ampa.device)
            if ht is not None:
                h = torch.clamp(ht, 0.0, 1.0)
                base = base * (0.25 + 2.0 * (1.0 - h))
            sigma = base

        e_cl = e_cl_effective(
            torch.as_tensor(self.e_cl_base, dtype=g_ampa.dtype, device=g_ampa.device),
            cfg).expand_as(self.v)

        v_new, v_inf = exact_step(self.v, g_ampa, g_nmda, g_gaba, e_cl, cfg,
                                  sigma=sigma, generator=generator)

        # --- спайк, рефрактерность, сброс вычитанием ---
        cand = (v_new >= theta) & (self.refr == 0)
        if k_spike is not None and 0 < k_spike < self.n:
            # k-WTA ДО сброса: иначе отброшенный спайк всё равно сбросил бы
            # мембрану и «наказал» нейрон за спайк, которого не было.
            score = torch.where(cand, v_new - theta,
                                torch.full_like(v_new, -1e9))
            thr = torch.topk(score, k_spike, dim=-1).values[..., -1:]
            cand = cand & (score >= thr)
        s = spike_fn(v_new - theta, cfg.surrogate, self.k) * cand.to(v_new.dtype)
        v_new = v_new - s * (theta - cfg.v_rest)
        refr = torch.where(s > 0,
                           torch.full_like(self.refr, cfg.refractory),
                           torch.clamp(self.refr - 1, min=0))

        # --- адаптация порога ---
        a_new = self.alpha_a * a_pre + s.detach()

        if inplace:
            self.v.copy_(v_new)
            self.a.copy_(a_new)
            self.refr.copy_(refr)
        else:
            self.v, self.a, self.refr = v_new, a_new, refr
            self.g_ampa, self.g_nmda, self.g_gaba = g_ampa, g_nmda, g_gaba

        return {"spikes": s, "v": self.v, "v_inf": v_inf,
                "theta": theta, "a": self.a}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rate(self) -> float:
        return float(self.a.mean()) / max(self.cfg.beta_adapt, 1e-9)
