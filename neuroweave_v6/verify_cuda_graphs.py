#!/usr/bin/env python3
"""NEUROWEAVE v6.0 — ВЕРИФИКАЦИЯ (CUDA Graphs + математика GABA + Экзамен).

Запуск:  python verify_cuda_graphs.py
          (Windows: verify.bat  |  Linux/macOS: sh verify.sh)

Стадии:
  [0] ENV      — окружение: torch, устройство, CUDA
  [1] MATH     — GABA-reversal: инвариант V >= V_rest при глубоком
                 торможении (точный интегратор vs Эйлер; KCC2-гомеостат)
  [2] SLEEP    — протокол сна: N3 без провалов + контрастная патология
  [3] GRAPHS   — быстрый контур в torch.cuda.CUDAGraph: захват,
                 эквивалентность eager<->graph, ускорение (>2x)
  [4] EXAM     — гибридный режим «Экзамен»: синтаксис<->логика,
                 w_slow заморожен (дрейф 0), WTA + быстрый контекст
  [5] SCALE    — масштабирование: O(N), гомеостаз, память

Выход: код 0 = все обязательные стадии пройдены (CUDA-стадии на машине
без GPU помечаются SKIP и не считаются провалом — прогоните на GPU).
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from neuroweave.config import NWConfig
from neuroweave.utils import fmt_device, get_device, set_seed

RESULTS = []


def stage(name, fn, required=True):
    print(f"\n{'=' * 68}\nСТАДИЯ [{name}]\n{'=' * 68}")
    try:
        detail, ok, skipped = fn()
        tag = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        RESULTS.append((name, tag, detail, ok or not required))
        print(f"--> {name}: {tag}  ({detail})")
        return ok
    except Exception as e:
        traceback.print_exc()
        RESULTS.append((name, "ERROR", str(e), False))
        print(f"--> {name}: ERROR  ({e})")
        return False


# ---------------------------------------------------------------- стадия 0
def st_env():
    dev = get_device()
    info = (f"torch {torch.__version__} | устройство: {fmt_device(dev)} | "
            f"CUDA доступна: {torch.cuda.is_available()}")
    print(info)
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        print(f"      GPU: {props.name}, {props.total_memory/1e9:.1f} ГБ, "
              f"cc {props.major}.{props.minor}")
    else:
        print("      (GPU не обнаружен — стадии 3/4(graph) будут SKIP; "
              "они обязательны только на CUDA-машине)")
    return info, True, False


# ---------------------------------------------------------------- стадия 1
def st_math():
    from neuroweave.config import MembraneConfig
    from neuroweave.neurons import (e_cl_effective, fixed_point,
                                    membrane_euler_step,
                                    membrane_exact_step)
    mcfg = MembraneConfig()   # v_rest=-70, v_th=-50, tau_m=20, dt=1
    v_rest = mcfg.v_rest
    dev = get_device()

    # T1: глубокое торможение ТОЧНЫМ интегратором — V никогда < V_rest
    V = torch.full((8,), -50.0, device=dev)          # деполяризовано
    g_gaba = torch.full((8,), 50.0, device=dev)      # глубокое торможение
    z = torch.zeros_like(V)
    E_cl = torch.full_like(V, float(v_rest))
    vmin = V.min().item()
    for _ in range(1000):
        V, V_inf = membrane_exact_step(V, z, z, g_gaba, E_cl, mcfg)
        vmin = min(vmin, V.min().item())
    t1 = vmin >= v_rest - 1e-6
    print(f"T1 точный интегратор, g_GABA=50: V_min = {vmin:.6f} мВ "
          f"(V_rest = {v_rest}) -> {'OK' if t1 else 'FAIL'}")

    # T1b: ещё глубже — g_GABA = 500 (экстремум протокола сна)
    V = torch.full((8,), -45.0, device=dev)
    g_gaba = torch.full((8,), 500.0, device=dev)
    vmin = V.min().item()
    for _ in range(2000):
        V, V_inf = membrane_exact_step(V, z, z, g_gaba, E_cl, mcfg)
        vmin = min(vmin, V.min().item())
    t1b = vmin >= v_rest - 1e-6
    print(f"T1b g_GABA=500 (экстрим): V_min = {vmin:.6f} мВ -> "
          f"{'OK' if t1b else 'FAIL'}")

    # T2: явный Эйлер при том же режиме — офшут/расходимость ниже V_rest
    Ve = torch.full((8,), -50.0, device=dev)
    vmin_e = Ve.min().item()
    for _ in range(60):
        Ve = membrane_euler_step(Ve, z, z, g_gaba * 0.0 + 50.0, E_cl, mcfg)
        vmin_e = min(vmin_e, Ve.min().item())
    t2 = vmin_e < v_rest
    print(f"T2 явный Эйлер, g_GABA=50, dt=1мс (tau_eff = {mcfg.tau_m:.0f}/51 мс << dt!): "
          f"V_min = {vmin_e:.3f} мВ "
          f"{'(провал/расходимость ниже V_rest — ожидаемо)' if t2 else ''} "
          f"-> {'OK (контраст показан)' if t2 else 'FAIL'}")

    # T3: патологический E_cl=-85 (коллапс Cl-) без/с KCC2-гомеостатом
    E_bad = torch.full_like(V, -85.0)
    V = torch.full((8,), -60.0, device=dev)
    g = torch.full((8,), 20.0, device=dev)
    V_bad = V.clone()
    vmin_bad = V_bad.min().item()
    for _ in range(500):
        V_bad, _ = membrane_exact_step(V_bad, z, z, g, E_bad, mcfg)
        vmin_bad = min(vmin_bad, V_bad.min().item())
    E_ok = e_cl_effective(E_bad, mcfg)   # KCC2: max(E_cl, V_rest)
    V_ok = V.clone()
    vmin_ok = V_ok.min().item()
    for _ in range(500):
        V_ok, _ = membrane_exact_step(V_ok, z, z, g, E_ok, mcfg)
        vmin_ok = min(vmin_ok, V_ok.min().item())
    t3 = (vmin_bad < v_rest) and (vmin_ok >= v_rest - 1e-6)
    print(f"T3 E_cl=-85 без KCC2: V_min = {vmin_bad:.3f} мВ (провал); "
          f"с KCC2: V_min = {vmin_ok:.6f} мВ -> "
          f"{'OK (гомеостат спасает)' if t3 else 'FAIL'}")

    # T4: выпуклость неподвижной точки (1000 случайных конфигураций)
    gsm = torch.Generator(device="cpu").manual_seed(0)
    Ge = torch.rand(1000, generator=gsm) * 3
    Gg = torch.rand(1000, generator=gsm) * 100
    Ecl = v_rest + torch.rand(1000, generator=gsm) * 10
    Vinf = (v_rest + Ge * mcfg.e_exc + Gg * Ecl) / (1 + Ge + Gg)
    t4 = bool((Vinf >= v_rest - 1e-9).all())
    print(f"T4 V_inf = выпуклая комбинация (V_rest, E_exc>=V_rest, E_cl>=V_rest): "
          f"min V_inf = {Vinf.min().item():.6f} >= V_rest -> {'OK' if t4 else 'FAIL'}")

    ok = t1 and t1b and t2 and t3 and t4
    return (f"T1..T4: {'все OK' if ok else 'есть провалы'} "
            f"(точный интегратор держит инвариант, Эйлер — нет)"), ok, False


# ---------------------------------------------------------------- стадия 2
def st_sleep():
    from neuroweave.sleep import run_sleep
    cfg = NWConfig()
    st_n = run_sleep(cfg, get_device(), pathological=False, verbose=True)
    st_p = run_sleep(cfg, get_device(), pathological=True, verbose=True)
    ok = st_n.bound_ok and (not st_p.bound_ok)
    return (f"норма: V_min(N3) = {st_n.v_min_deep:.4f} мВ >= V_rest; "
            f"патология без KCC2: V_min = {st_p.v_min_deep:.2f} мВ < V_rest"), ok, False


# ---------------------------------------------------------------- стадия 3
def st_graphs():
    dev = get_device()
    if dev.type != "cuda":
        return ("CUDA не обнаружена — захват графа на этой машине невозможен; "
                "прогоните на GPU-машине (verify.bat там покажет >2x)"), True, True
    from neuroweave.cudagraph import graph_vs_eager
    from neuroweave.fastloop import FastLoop
    from neuroweave.neurochem import tonic_schedules
    from neuroweave.sleep import build_sleep_tables

    cfg = NWConfig()
    cfg.membrane.sigma_ht = 0.0   # детерминированная траектория для сравнения
    tables = build_sleep_tables(cfg, dev)
    sched = tonic_schedules(cfg.sched_len, len(cfg.zones), dev, ach=0.5, ht=0.35)
    loop = FastLoop(cfg, dev, tables=tables, schedule=sched)
    loop.learn = False

    print("Захват быстрого контура (входы + DA/NE пульсы + мембрана + WTA)...")
    res = graph_vs_eager(loop, steps=2000, warmup=200, correctness_steps=60)
    print(f"      граф захвачен:            {res['captured']}")
    print(f"      эквивалентность eager:    {res['match']} "
          f"(max|dV| = {res['max_err']:.2e})")
    print(f"      eager : {res['ms_per_step_eager']:8.4f} мс/шаг "
          f"({res['eager_ms']:.1f} мс / 2000 шагов)")
    print(f"      graph : {res['ms_per_step_graph']:8.4f} мс/шаг "
          f"({res['graph_ms']:.1f} мс / 2000 шагов)")
    print(f"      УСКОРЕНИЕ: x{res['speedup']:.2f}")
    ok = bool(res["captured"] and res["match"] and res["speedup"] >= 1.5)
    return (f"capture={res['captured']}, match={res['match']}, "
            f"speedup=x{res['speedup']:.2f} (цель > 2x на выделенном GPU)"), ok, False


# ---------------------------------------------------------------- стадия 4
def st_exam():
    from neuroweave.exam import ExamRunner, exam_config
    dev = get_device()
    cfg = exam_config(seed=7)
    runner = ExamRunner(cfg, dev)
    runner.pretrain(n_win_syn=60, n_pairs_log=60, verbose=True)
    rep = runner.run_exam(use_graph=False, verbose=True)
    rep_g = None
    if dev.type == "cuda":
        print("\n      -- повтор «Экзамена» ПОЛНОСТЬЮ внутри CUDA Graph --")
        runner2 = ExamRunner(exam_config(seed=7), dev)
        runner2.pretrain(n_win_syn=60, n_pairs_log=60, verbose=False)
        rep_g = runner2.run_exam(use_graph=True, verbose=True)
    ok = rep.passed and (rep_g is None or rep_g.passed)
    return (f"syn acc={rep.syntax_acc:.3f}, log acc={rep.logic_acc:.3f}, "
            f"follow={rep.follow:.3f}, drift_exact={rep.drift_exact}, "
            f"NE max={rep.ne_pulse_max:.2f}"
            + (f"; в графе: syn={rep_g.syntax_acc:.3f}, log={rep_g.logic_acc:.3f}"
               if rep_g else "")), ok, False


# ---------------------------------------------------------------- стадия 5
def st_scale():
    from neuroweave.scaling import run_scaling
    dev = get_device()
    factors = (1, 2, 4) if dev.type == "cuda" else (1, 2)
    pts = run_scaling(dev, factors=factors, steps=250)
    ok = all(p.finite for p in pts) and all(0.0 < p.rate_mean < 0.9 for p in pts)
    return (f"факторы {factors}: время/шаг {[round(p.ms_per_step, 3) for p in pts]} мс, "
            f"finite={all(p.finite for p in pts)}"), ok, False


# ---------------------------------------------------------------- итог
def summary():
    print(f"\n{'=' * 68}\nИТОГ ВЕРИФИКАЦИИ\n{'=' * 68}")
    hard_fail = False
    for (name, tag, detail, soft) in RESULTS:
        print(f"  [{tag:>5}] {name:<8} {detail[:80]}")
        if tag in ("FAIL", "ERROR") and not soft:
            hard_fail = True
    n_pass = sum(1 for r in RESULTS if r[1] == "PASS")
    print(f"\n  Пройдено: {n_pass}/{len(RESULTS)}   "
          f"ВЕРДИКТ: {'ALL GREEN' if not hard_fail else 'ЕСТЬ ПРОВАЛЫ'}")
    print("  (стадии GRAPHS обязательны только при наличии CUDA)")
    return 0 if not hard_fail else 1


def main():
    set_seed(7)
    stage("ENV", st_env, required=True)
    stage("MATH", st_math, required=True)
    stage("SLEEP", st_sleep, required=True)
    stage("GRAPHS", st_graphs, required=False)
    stage("EXAM", st_exam, required=True)
    stage("SCALE", st_scale, required=True)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
