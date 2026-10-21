"""CUDA Graphs для быстрого контура NEUROWEAVE (Этап 6).

КАК ОБЕРНУТ БЫСТРЫЙ КОНТУР (с тензорными пульсами DA/NE) В
torch.cuda.CUDAGraph БЕЗ синхронизации на каждом шаге:

  1. СТАТИЧЕСКИЕ БУФЕРЫ: входы (токен, контекст) копируются в выделенные
     заранее тензоры; выходы — ссылки на пул памяти графа.
  2. ПРОГРЕВ на side-stream (3 итерации): аллокатор и автотюнинг ядер
     выходят на стационарный режим ДО захвата.
  3. ЗАХВАТ: g = torch.cuda.CUDAGraph(); with torch.cuda.graph(g): step(...)
     Каждое ядро шага (входы, связность, DA/NE-пульсы, мембрана, WTA,
     readout) записывается в граф ОДИН раз.
  4. РЕПЛЕЙ: static_in.copy_(x); g.replay(); читать static_out.
     Весь шаг = ОДИН запуск графа на GPU: НОЛЬ host<->device синхронизаций,
     ноль аллокаций, ноль запусков ядра из Python (~100 kernels/шаг у нас).

Почему пульсы DA/NE переживают захват:
  * DA живёт в ТЕНЗОРЕ расписания и читается по tensor-счётчику t,
    который продвигается in-place (t.add_(1)) ВНУТРИ графа;
  * NE-пульс детектируется ВНУТРИ контура (|ctx - ctx_ema|);
  * 5-HT-шум: RNG graph-safe (philox), каждый реплей — свежие числа;
  * ветвления в step() — только по python-константам (зоны, learn).

Классическая ловушка, которую мы обходим: «t = t + 1» создаёт новый тензор,
граф навсегда запоминает ЧТЕНИЕ старой памяти — счётчик не двигается.
Поэтому ВСЁ накапливаемое состояние обновляется строго in-place.
"""
import time
from typing import Callable, Optional

import torch


class FastLoopGraph:
    """Обёртка быстрого контура в CUDA Graph (захват один раз, реплей — шаг).

    Использование:
        gr = FastLoopGraph(loop, warmup_iters=3)
        out = gr(tok, ctx)      # == loop.step(tok, ctx), но одним replay
    """

    def __init__(self, loop, warmup_iters: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FastLoopGraph требует CUDA (torch.cuda.is_available() == False)")
        self.loop = loop
        dev = loop.device
        if dev.type != "cuda":
            raise RuntimeError(f"FastLoopGraph: loop на {dev}, нужен cuda")

        # статические входы (копируются ПЕРЕД каждым replay)
        self.static_tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.static_ctx = torch.tensor([0.5, 0.5], device=dev)

        # --- прогрев на side-stream ---
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                loop.step(self.static_tok, self.static_ctx)
        torch.cuda.current_stream().wait_stream(s)

        # --- захват ---
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = loop.step(self.static_tok, self.static_ctx)

    def __call__(self, tok: torch.Tensor, ctx: torch.Tensor):
        self.static_tok.copy_(tok, non_blocking=True)
        self.static_ctx.copy_(ctx, non_blocking=True)
        self.graph.replay()
        return self.static_out


@torch.no_grad()
def bench_step(loop, steps: int = 1000, use_graph: bool = False,
               graph: Optional[FastLoopGraph] = None, tok_id: int = 0):
    """Время steps шагов (мс) + мс/шаг. CUDA — по events, CPU — perf_counter."""
    dev = loop.device
    tok = torch.tensor([tok_id], dtype=torch.long, device=dev)
    ctx = torch.tensor([0.7, 0.3], device=dev)
    if dev.type == "cuda":
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(steps):
            (graph(tok, ctx) if use_graph else loop.step(tok, ctx))
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(steps):
            loop.step(tok, ctx)
        ms = (time.perf_counter() - t0) * 1e3
    return ms, ms / steps


@torch.no_grad()
def graph_vs_eager(loop, steps: int = 2000, warmup: int = 200,
                   correctness_steps: int = 60):
    """Полный тест: захват, эквивалентность eager-vs-graph, ускорение.

    Возвращает dict: {captured, match, max_err, eager_ms, graph_ms,
    speedup, ms_per_step_eager, ms_per_step_graph}.
    Эквивалентность проверяется при sigma=0 (выключите 5-HT-шум — RNG
    eager и графа идут разными потоками philox; при sigma=0 вклад шума
    ровно нулевой и траектории совпадают до машинной точности).
    """
    dev = loop.device
    res = dict(captured=False, match=None, max_err=None,
               eager_ms=None, graph_ms=None, speedup=None,
               ms_per_step_eager=None, ms_per_step_graph=None)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return res

    # --- корректность: детерминированная копия состояния ---
    loop.reset_state()
    g = FastLoopGraph(loop)
    toks = torch.randint(0, 32, (correctness_steps,), device=dev)
    ctxs = torch.rand(correctness_steps, 2, device=dev)
    # eager-эталон
    loop.reset_state()
    outs_e = [loop.step(toks[i:i + 1].clone(), ctxs[i].clone())
              for i in range(correctness_steps)]
    V_e = outs_e[-1].V.detach().clone()
    spk_e = outs_e[-1].spikes.detach().clone()
    # граф (сброс состояния и реплей той же последовательности)
    loop.reset_state()
    for i in range(correctness_steps):
        g(toks[i:i + 1].clone(), ctxs[i].clone())
    V_g = g.static_out.V.detach().clone()
    spk_g = g.static_out.spikes.detach().clone()
    res["captured"] = True
    res["max_err"] = ((V_e - V_g).abs().max().item())
    res["match"] = bool(res["max_err"] < 1e-4 and
                        torch.equal(spk_e, spk_g))

    # --- скорость ---
    loop.reset_state()
    bench_step(loop, steps=warmup)                    # прогрев eager
    e_ms, e_per = bench_step(loop, steps=steps)
    gr_ms, gr_per = bench_step(loop, steps=steps, use_graph=True, graph=g)
    res.update(eager_ms=e_ms, graph_ms=gr_ms,
               ms_per_step_eager=e_per, ms_per_step_graph=gr_per,
               speedup=e_ms / gr_ms if gr_ms > 0 else float("inf"))
    return res
