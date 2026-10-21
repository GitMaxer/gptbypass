"""Ядро мембранной динамики NEUROWEAVE (Этап 6).

GABA С REVERSAL-ПОТЕНЦИАЛОМ:

    I_GABA = g_GABA * (V - E_Cl)

КЛЮЧЕВОЙ МАТЕМАТИЧЕСКИЙ ФАКТ (критично для протокола сна):

    tau_m dV/dt = (V_rest - V) + G_e*(E_exc - V) - g_GABA*(V - E_Cl) + sigma*xi
    где G_e = g_ampa + g_nmda*B_mg(V) >= 0.

    Неподвижная точка линеаризованного уравнения (B заморожен):

        V_inf = (V_rest + G_e*E_exc + g_GABA*E_Cl) / (1 + G_e + g_GABA)

    — это ВЫПУКЛАЯ КОМБИНАЦИЯ (V_rest, E_exc, E_Cl) с неотрицательными
    весами, в сумме равными знаменателю. Значит:

        E_Cl >= V_rest  и  G_e >= 0  =>  V_inf >= V_rest  при ЛЮБОМ g_GABA.

    При E_Cl = V_rest и G_e = 0:  V_inf = V_rest точно, для любого g_GABA:
    глубокое торможение ШУНТИРУЕТ (тау_eff = tau_m/(1+g_GABA) -> 0),
    но НЕ гиперполяризует. Торможение саморегулируется: при V < E_Cl
    ток I_GABA < 0 (деполяризующий!) — «тормоз», а не «толкатель вниз».

ЧИСЛЕННАЯ ГАРАНТИЯ: точный экспоненциальный интегратор
    V(t+dt) = V_inf + (V - V_inf)*exp(-lam*dt),  lam = (1+G_e+g_GABA)/tau_m
— безусловно устойчив при ЛЮБОМ dt (у явного Эйлера при глубоком
торможении tau_eff << dt и мембрана «проваливается» ниже V_rest —
см. демонстрацию в верификации, тест T2).

Полный вывод — MATH_GABA_REVERSAL.md.
"""
import torch

from .config import MembraneConfig


def mg_block(V: torch.Tensor, mg: float = 1.0) -> torch.Tensor:
    """Jahr–Stevens: блок Mg2+ NMDA-канала. B(V) ~ 0 при V << -40 мВ.

    Биологическое согласование с протоколом сна: в глубоком сне V низкий,
    NMDA автоматически закрыт Mg2+ -> G_e -> 0 -> V_inf -> V_rest.
    """
    return 1.0 / (1.0 + torch.exp(-0.062 * V) * (mg / 3.57))


def fixed_point(V, g_ampa, g_nmda, g_gaba, E_cl, mcfg: MembraneConfig):
    """V_inf, lam, denom — неподвижная точка и скорость релаксации."""
    B = mg_block(V, mcfg.mg)
    G_e = g_ampa + g_nmda * B
    denom = 1.0 + G_e + g_gaba
    lam = denom / mcfg.tau_m
    V_inf = (mcfg.v_rest + G_e * mcfg.e_exc + g_gaba * E_cl) / denom
    return V_inf, lam, denom


def membrane_exact_step(V, g_ampa, g_nmda, g_gaba, E_cl, mcfg: MembraneConfig,
                        sigma=None):
    """Один ТОЧНЫЙ шаг мембраны (conductance-based LIF, exact exponential).

    Возвращает (V_new, V_inf). Гарантирует V >= v_floor (по построению
    динамики V_inf >= V_rest при E_cl >= V_rest; v_floor — предохранитель).
    Все операции тензорные -> граф-безопасно (torch.cuda.CUDAGraph).
    """
    V_inf, lam, _ = fixed_point(V, g_ampa, g_nmda, g_gaba, E_cl, mcfg)
    dec = torch.exp(-lam * mcfg.dt)
    V_new = V_inf + (V - V_inf) * dec
    if sigma is not None:
        # точный член шума Орнштейна–Уленбека:
        # std = sigma * sqrt((1 - exp(-2*lam*dt)) / (2*lam))
        noise_std = sigma * torch.sqrt(
            torch.clamp(1.0 - dec * dec, min=1e-12) / (2.0 * lam))
        V_new = V_new + noise_std * torch.randn_like(V_new)
    V_new = torch.clamp(V_new, min=mcfg.v_floor)
    return V_new, V_inf


def membrane_euler_step(V, g_ampa, g_nmda, g_gaba, E_cl, mcfg: MembraneConfig):
    """Явный Эйлер — ТОЛЬКО ДЛЯ КОНТРАСТА в верификации (не используется
    в рантайме). При глубоком торможении tau_eff = tau_m/(1+g_GABA) << dt:
    численный офшут ниже V_rest и расходимость. Демонстрирует, почему нужен
    точный интегратор.
    """
    B = mg_block(V, mcfg.mg)
    G_e = g_ampa + g_nmda * B
    dV = (mcfg.v_rest - V) + G_e * (mcfg.e_exc - V) - g_gaba * (V - E_cl)
    return V + (mcfg.dt / mcfg.tau_m) * dV


def e_cl_effective(e_cl_raw: torch.Tensor, mcfg: MembraneConfig) -> torch.Tensor:
    """KCC2-подобный хлорный гомеостат.

    Биология: при коллапсе Cl- (интенсивная активность, отказ KCC2)
    E_cl может упасть ниже V_rest — и GABA станет гиперполяризующим ниже
    покоя. Гомеостат не опускает E_cl ниже V_rest:
        E_cl_eff = max(E_cl_raw, V_rest).
    Вместе с точным интегратором это даёт инвариант V >= V_rest по построению.
    """
    if mcfg.ecl_mode == "raw":
        return e_cl_raw  # патологический режим (для демонстрации в верификации)
    return torch.clamp(e_cl_raw, min=mcfg.v_rest)
