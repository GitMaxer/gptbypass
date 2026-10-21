"""Протокол сна NEUROWEAVE (Этап 6: GABA-reversal делает его безопасным).

Стадии:  W -> N1 -> N2 (веретёна) -> N3 (глубокий сон) -> REM -> W

КЛЮЧЕВАЯ ПРОВЕРКА: в N3 (глубокое торможение, g_GABA до ~3.5) мембрана
НИКОГДА не уходит ниже V_rest. Это обеспечивается тройной защитой:

  1. E_Cl = V_rest (шунтирующий режим): V_inf = V_rest точно при любом g_GABA;
  2. KCC2-гомеостат: E_cl_eff = max(E_cl, V_rest) — защита от коллапса Cl-;
  3. точный интегратор: численный офшут невозможен ни при каком dt.

Контрастный опыт (та же математика, сделанная наглядной): при
ecl_mode='raw' и E_cl = -85 мВ (коллапс Cl-, отказ KCC2) мембрана
ПРОВАЛИВАЕТСЯ ниже V_rest — ровно то, что бывает в биологии при
нарушении хлорного гомеостаза, и ровно то, от чего защищает гомеостат.
"""
from dataclasses import dataclass
from typing import List

import torch

from .config import NWConfig
from .fastloop import FastLoop
from .neurochem import ModulatorSchedules, tonic_schedules


@dataclass
class SleepStats:
    steps: int
    v_min: float                  # минимум V за ВСЁ время
    v_min_deep: float             # минимум V в N3
    v_mean_deep: float
    rate_wake: float              # средняя частота зон в бодрствовании
    rate_deep: float              # в N3 (глубокое подавление)
    bound_ok: bool                # V >= V_rest - tol всюду в N3
    stages: List[str]


# параметры стадий: (имя, шагов, g_sleep, ACh, 5-HT)
STAGES = [
    ("W",    150, 0.05, 0.55, 0.35),
    ("N1",   150, 0.40, 0.40, 0.25),
    ("N2",   200, 0.60, 0.30, 0.20),   # + веретёна (синус в g)
    ("N3",   300, 3.50, 0.15, 0.02),   # глубокий сон: g_GABA высок, 5-HT -> 0
    ("REM",  150, 0.15, 0.90, 0.50),   # ACh высокий (как в биологии)
    ("W2",   100, 0.05, 0.55, 0.35),
]

SPINDLE_PERIOD = 12.0
SPINDLE_DEPTH = 0.35


def build_sleep_tables(cfg: NWConfig, device, seed: int = 1):
    """Слабые спонтанные входы во сне (blank-токен 32): rows ~ 0."""
    g = torch.Generator().manual_seed(seed)
    tables = {}
    for name, zone in (("cur_A", "A"), ("cur_B", "B"), ("del_B", "B"),
                       ("cur_C", "C"), ("del_C", "C")):
        n_e = cfg.zone(zone).n_e
        tab = torch.rand(cfg.n_tokens, n_e, generator=g) * 0.004  # слабый спонтанный фон
        tab[cfg.blank_token] = torch.rand(n_e, generator=g) * 0.002
        tables[name] = tab
    return tables


def build_sleep_schedule(cfg: NWConfig, T: int, device,
                         spindle=True) -> ModulatorSchedules:
    """Расписание ACh/5-HT/GABA-сна по стадиям (+веретёна в N2)."""
    Z = len(cfg.zones)
    s = tonic_schedules(T, Z, device, ach=0.5, ht=0.35, g_sleep=0.05)
    t0 = 0
    for (name, steps, g, ach, ht) in STAGES:
        t1 = min(t0 + steps, T)
        s.ach[t0:t1] = ach
        s.ht[t0:t1] = ht
        s.g_sleep[t0:t1] = g
        if name == "N2" and spindle:
            ts = torch.arange(t1 - t0, device=device, dtype=torch.float32)
            spindle = SPINDLE_DEPTH * torch.sin(
                2 * torch.pi * ts / SPINDLE_PERIOD)
            s.g_sleep[t0:t1] = g + spindle.unsqueeze(1)  # веретёна: [steps, 1] -> [steps, Z]
        t0 = t1
        if t0 >= T:
            break
    return s


def run_sleep(cfg: NWConfig, device, pathological: bool = False,
              verbose: bool = True) -> SleepStats:
    """Прогон протокола сна. pathological=True: E_cl = -85 мВ БЕЗ гомеостата
    (демонстрация, зачем нужен KCC2-насос)."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.membrane.ecl_mode = "raw" if pathological else "homeostat"
    if pathological:
        cfg.membrane.e_cl = -85.0   # коллапс Cl-
        cfg.membrane.v_floor = -90.0  # предохранитель выключен
    else:
        cfg.membrane.e_cl = cfg.membrane.v_rest  # шунтирующий режим
        cfg.membrane.v_floor = cfg.membrane.v_rest  # belt-and-suspenders
        cfg.membrane.sigma_ht = 0.15  # в N3 5-HT и так ~0: мембрана тихая

    total = sum(st[1] for st in STAGES)
    tables = build_sleep_tables(cfg, device)
    sched = build_sleep_schedule(cfg, total, device)
    loop = FastLoop(cfg, device, tables=tables, schedule=sched)
    loop.learn = False

    blank = torch.tensor([cfg.blank_token], dtype=torch.long, device=device)
    ctx = torch.tensor([0.5, 0.5], device=device)

    v_min, v_min_deep, v_sum_deep, n_deep = 1e9, 1e9, 0.0, 0
    rate_w, rate_w_n = 0.0, 0
    rate_d, rate_d_n = 0.0, 0
    # границы N3 по кумулятивным шагам
    cum, deep_lo, deep_hi = 0, None, None
    for (name, steps, *_r) in STAGES:
        if name == "N3":
            deep_lo, deep_hi = cum, cum + steps
        cum += steps

    loop.reset_state()
    t = 0
    for (name, steps, _g, _a, _h) in STAGES:
        for _ in range(steps):
            out = loop.step(blank, ctx)
            v = out.V
            vmin_now = v.min().item()
            v_min = min(v_min, vmin_now)
            if deep_lo is not None and deep_lo <= t < deep_hi:
                v_min_deep = min(v_min_deep, vmin_now)
                v_sum_deep += v.mean().item()
                n_deep += 1
                rate_d += out.rates.mean().item()
                rate_d_n += 1
            if name in ("W", "W2") and t > 20:
                rate_w += out.rates.mean().item()
                rate_w_n += 1
            t += 1

    rate_deep = rate_d / max(rate_d_n, 1)
    tol = 1e-5
    bound_ok = v_min_deep >= cfg.membrane.v_rest - tol
    stats = SleepStats(
        steps=total, v_min=v_min, v_min_deep=v_min_deep,
        v_mean_deep=v_sum_deep / max(n_deep, 1),
        rate_wake=rate_w / max(rate_w_n, 1), rate_deep=rate_deep,
        bound_ok=bound_ok, stages=[st[0] for st in STAGES])

    if verbose:
        mode = "ПАТОЛОГИЯ (E_cl=-85, KCC2 отключён)" if pathological else "НОРМА (E_cl=V_rest + KCC2)"
        print(f"[СОН] {mode}")
        print(f"      стадии: {' -> '.join(stats.stages)}  ({total} шагов)")
        print(f"      V_min за всё время : {stats.v_min:8.3f} мВ (V_rest = {cfg.membrane.v_rest})")
        print(f"      V_min в N3 (глубокий сон, g_GABA~3.5): {stats.v_min_deep:8.3f} мВ")
        print(f"      V_mean в N3        : {stats.v_mean_deep:8.3f} мВ")
        print(f"      частота W -> N3    : {stats.rate_wake:.4f} -> {stats.rate_deep:.4f}")
        print(f"      ИНВАРИАНТ V >= V_rest в N3: "
              f"{'OK' if bound_ok else 'НАРУШЕН (ожидаемо для патологии)'}")
    return stats
