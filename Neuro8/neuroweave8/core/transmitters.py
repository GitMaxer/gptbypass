"""Ионотропная биофизика: AMPA / NMDA (Mg2+-блок) / GABA (с реверсией).

--------------------------------------------------------------------------------
МАТЕМАТИЧЕСКИЙ ИНВАРИАНТ (наследие v6, усилен в v8)

    tau_m dV/dt = (V_rest - V) + G_e (E_exc - V) + g_GABA (E_Cl - V)

Неподвижная точка при замороженном G_e:

    V_inf = (V_rest + G_e E_exc + g_GABA E_Cl) / (1 + G_e + g_GABA)

Это выпуклая комбинация (V_rest, E_exc, E_Cl) с неотрицательными весами,
сумма которых равна знаменателю. Следствия:

  * E_Cl >= V_rest  =>  V_inf >= V_rest при ЛЮБОМ g_GABA: торможение
    ШУНТИРУЕТ (tau_eff = tau_m/(1+G_e+g)), но не гиперполяризует;
  * при V < E_Cl ток GABA положителен (деполяризующий) — «тормоз» не
    является «толкателем вниз», он возвращает мембрану к E_Cl.

Численная гарантия: точный экспоненциальный интегратор
    V(t+dt) = V_inf + (V - V_inf) exp(-lam dt),  lam = (1+G_e+g_GABA)/tau_m
безусловно устойчив при любом dt. Явный Эйлер при глубоком торможении
(tau_eff << dt) обязан «провалиться» ниже V_rest — это демонстрируется
в тесте как контраст, доказывающий необходимость точной схемы.

ОТЛИЧИЕ v8: Mg2+-блок применяется к АПИКАЛЬНОЙ проводимости, а ACh
модулирует проводимости (а не ток) — канон (5) переписан в проводимостной
форме, поэтому инвариант V >= V_rest сохраняется при любом ACh.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import torch

from ..config import NeuronCfg


def mg_block(v: torch.Tensor, mg: float = 1.0) -> torch.Tensor:
    """Jahr-Stevens: доля открытых NMDA-каналов. B ~ 0 при V << -40 мВ."""
    return 1.0 / (1.0 + torch.exp(-0.062 * v) * (mg / 3.57))


def coincidence_gate(g_ampa: torch.Tensor, theta_mg: float = 0.25,
                     k_mg: float = 12.0) -> torch.Tensor:
    """Гейт совпадения по базальному драйву (канон v4.0, ур. 4).

    Физиология NMDA требует ДВУХ условий: связывания глутамата и локальной
    деполяризации. Уравнение (4) канона кодирует первое через
    sigma(I_basal - theta_Mg), а Jahr-Stevens — второе через V.

    Здесь сигмоида КАЛИБРОВАНА: при g_ampa = 0 гейт равен НУЛЮ точно, а не
    «маленькому остатку». Иначе детектор совпадения протекает и апикальный
    вход без базального способен запустить спайк — контракт v8 это запрещает.
    """
    zero = torch.sigmoid(torch.as_tensor(-k_mg * theta_mg, dtype=g_ampa.dtype,
                                         device=g_ampa.device))
    raw = torch.sigmoid(k_mg * (g_ampa - theta_mg))
    return torch.clamp((raw - zero) / (1.0 - zero), min=0.0, max=1.0)


def ach_gains(ach: torch.Tensor | None, gamma: float) -> tuple:
    """ACh балансирует Bottom-Up / Top-Down в ПРОВОДИМОСТЯХ.

    Возвращает (gain_basal, gain_apical), оба неотрицательны при
    |ach| <= 1, что сохраняет выпуклость V_inf. Клэмп — часть контракта,
    а не защита от ошибки: дрейф химии не должен ломать инвариант.
    """
    if ach is None:
        return 1.0, 1.0
    a = torch.clamp(ach, -1.0, 1.0)
    return 1.0 + gamma * a, 1.0 - gamma * a


def fixed_point(
    v: torch.Tensor,
    g_ampa: torch.Tensor,
    g_nmda: torch.Tensor,
    g_gaba: torch.Tensor,
    e_cl: torch.Tensor,
    cfg: NeuronCfg,
    ach: torch.Tensor | None = None,
) -> tuple:
    """V_inf, lam, denom для текущей проводимости."""
    b = mg_block(v, cfg.mg) * coincidence_gate(g_ampa, cfg.theta_mg, cfg.k_mg)
    gb, ga = ach_gains(ach, cfg.gamma_ach)
    g_e = gb * g_ampa + ga * cfg.alpha_nmda * g_nmda * b
    denom = 1.0 + g_e + g_gaba
    lam = denom / cfg.tau_m
    v_inf = (cfg.v_rest + g_e * cfg.e_exc + g_gaba * e_cl) / denom
    return v_inf, lam, denom


def e_cl_effective(e_cl_raw: torch.Tensor, cfg: NeuronCfg) -> torch.Tensor:
    """KCC2-подобный хлорный гомеостат: E_Cl не опускается ниже V_rest."""
    if cfg.ecl_mode == "raw":
        return e_cl_raw
    return torch.clamp(e_cl_raw, min=cfg.v_rest)


def exact_step(
    v: torch.Tensor,
    g_ampa: torch.Tensor,
    g_nmda: torch.Tensor,
    g_gaba: torch.Tensor,
    e_cl: torch.Tensor,
    cfg: NeuronCfg,
    sigma: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
) -> tuple:
    """Один точный шаг мембраны. Возвращает (V_new, V_inf)."""
    v_inf, lam, _ = fixed_point(v, g_ampa, g_nmda, g_gaba, e_cl, cfg)
    dec = torch.exp(-lam * cfg.dt)
    v_new = v_inf + (v - v_inf) * dec
    if sigma is not None:
        # Точный шум Орнштейна-Уленбека: std^2 = sigma^2 (1-e^{-2 lam dt})/(2 lam)
        std = sigma * torch.sqrt(
            torch.clamp(1.0 - dec * dec, min=1e-12) / (2.0 * lam))
        noise = torch.randn(std.shape, dtype=std.dtype, device=std.device,
                            generator=generator)
        v_new = v_new + std * noise
    return torch.clamp(v_new, min=cfg.v_floor), v_inf


def euler_step(
    v: torch.Tensor,
    g_ampa: torch.Tensor,
    g_nmda: torch.Tensor,
    g_gaba: torch.Tensor,
    e_cl: torch.Tensor,
    cfg: NeuronCfg,
) -> torch.Tensor:
    """Явный Эйлер. ТОЛЬКО для контраста в тестах — в рантайме не используется."""
    b = mg_block(v, cfg.mg) * coincidence_gate(g_ampa, cfg.theta_mg, cfg.k_mg)
    g_e = g_ampa + cfg.alpha_nmda * g_nmda * b
    dv = (cfg.v_rest - v) + g_e * (cfg.e_exc - v) + g_gaba * (e_cl - v)
    return v + (cfg.dt / cfg.tau_m) * dv


def asymptote(g_ampa, g_nmda, g_gaba, e_cl, cfg: NeuronCfg, v=-70.0) -> torch.Tensor:
    """Аналитическая неподвижная точка при заданных проводимостях."""
    v_inf, _, _ = fixed_point(torch.as_tensor(v), g_ampa, g_nmda, g_gaba,
                              e_cl, cfg)
    return v_inf
