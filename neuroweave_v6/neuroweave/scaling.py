"""Масштабирование NEUROWEAVE (Этап 6).

Принцип: размеры зон x factor при ПОСТОЯННОМ fan-in:
  * память  O(N * fan_in) — линейна;
  * время шага O(N * fan_in) — линейно (gather-связность);
  * локальная плотность связей неизменна -> динамика/гомеостаз
    масштаб-инвариантны (частоты зон держит S7).

Бенчмарк: время шага (eager и CUDA Graph), пиковая память, частоты,
NaN-скан, оценка показателя O(N^slope) по log-log регрессии.
"""
import math
import time
from dataclasses import dataclass
from typing import List

import torch

from .config import NWConfig
from .fastloop import FastLoop
from .neurochem import tonic_schedules
from .sleep import build_sleep_tables


@dataclass
class ScalePoint:
    factor: int
    n_total: int
    ms_per_step: float
    ms_per_step_graph: float   # None, если CUDA нет
    speedup: float
    peak_mem_mb: float
    rate_mean: float
    finite: bool


def scaling_tables(cfg: NWConfig, device, seed: int = 5):
    """Таблицы для «фоновой» активности (масштабный прогон без задачи)."""
    return build_sleep_tables(cfg, device, seed=seed)


def bench_scale(cfg: NWConfig, device, factor: int = 1, steps: int = 300,
                warmup: int = 50, use_graph_if_cuda: bool = True
                ) -> ScalePoint:
    loop = FastLoop(cfg, device, tables=scaling_tables(cfg, device),
                    schedule=tonic_schedules(cfg.sched_len, len(cfg.zones),
                                             device, ach=0.5, ht=0.35))
    loop.learn = False
    tok = torch.tensor([0], dtype=torch.long, device=device)
    ctx = torch.tensor([0.5, 0.5], device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(warmup):
            loop.step(tok, ctx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            loop.step(tok, ctx)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
    else:
        for _ in range(warmup):
            loop.step(tok, ctx)
        t0 = time.perf_counter()
        for _ in range(steps):
            loop.step(tok, ctx)
        ms = (time.perf_counter() - t0) * 1e3

    ms_graph, speedup = None, None
    if device.type == "cuda" and use_graph_if_cuda:
        from .cudagraph import FastLoopGraph, bench_step
        g = FastLoopGraph(loop)
        _m, ms_graph = bench_step(loop, steps=steps, use_graph=True, graph=g)
        speedup = ms / ms_graph if ms_graph else None
        # после реплеев граф работает с тем же состоянием — валидно
        out = loop.step(tok, ctx)
    else:
        out = loop.step(tok, ctx)

    rate = out.rates.mean().item()
    finite = bool(torch.isfinite(out.V).all().item() and
                  torch.isfinite(out.spikes).all().item())
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
    else:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        except Exception:
            peak = float("nan")
    return ScalePoint(factor, cfg.total_n(),
                      ms / steps, ms_graph, speedup, peak, rate, finite)


def run_scaling(device, factors: List[int] = (1, 2, 4), steps: int = 300,
                base_seed: int = 7, verbose=True) -> List[ScalePoint]:
    pts: List[ScalePoint] = []
    for f in factors:
        cfg = NWConfig(seed=base_seed).scaled(f)
        p = bench_scale(cfg, device, factor=f, steps=steps)
        pts.append(p)
        if verbose:
            g = (f"{p.ms_per_step_graph:7.3f} мс/шаг, x{p.speedup:.2f}"
                 if p.ms_per_step_graph else "—")
            print(f"[SCALE] x{f:<2d} N={p.n_total:5d} | eager {p.ms_per_step:7.3f} мс/шаг"
                  f" | graph {g} | peak {p.peak_mem_mb:7.1f} МБ | rate {p.rate_mean:.3f}"
                  f" | finite={p.finite}")
    if len(pts) >= 2:
        import math as _m
        xs = [_m.log(p.n_total) for p in pts]
        ys = [_m.log(p.ms_per_step) for p in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) /
                 max(1e-9, sum((x - mx) ** 2 for x in xs)))
        if verbose:
            print(f"[SCALE] log-log наклон времени шага: {slope:.3f} "
                  f"(1.0 = идеально линейно O(N); fan-in фиксирован)")
    return pts
