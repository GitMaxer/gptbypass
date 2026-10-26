"""Инженерия быстрого контура — порт runtime.py из NEUROWEAVE 8.1 / Spiker.

FastLoopProfiler: секционное таймирование такта (колонки / шина / память /
гейты) — без него следующий ablation вслепую. CUDAGraphRunner опускаем:
на CPU он noop, на GPU включается только при torch.cuda.is_available().
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch


class FastLoopProfiler:
    """Секционный профайлер такта: prof.section("columns") ... stat."""

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._starts: Dict[str, float] = {}
        self.host_syncs: int = 0

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def note_host_sync(self) -> None:
        """Отметить .item()/.tolist()-синхронизацию GPU->CPU."""
        self.host_syncs += 1

    def report(self) -> Dict[str, float]:
        rep: Dict[str, float] = {}
        total = sum(self.totals.values())
        for k, v in sorted(self.totals.items(),
                           key=lambda kv: -kv[1]):
            rep[f"{k}_ms"] = 1000.0 * v / max(1, self.counts[k])
            rep[f"{k}_pct"] = 100.0 * v / max(1e-9, total)
        rep["total_ms_per_tick"] = 1000.0 * total / max(
            1, max(self.counts.values(), default=0))
        rep["host_syncs"] = float(self.host_syncs)
        return rep

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.host_syncs = 0


class CUDAGraphRunner:
    """Захват такта в CUDA Graph (только если CUDA доступна; иначе passthrough)."""

    def __init__(self, enabled: Optional[bool] = None) -> None:
        self.enabled = (torch.cuda.is_available() if enabled is None
                        else bool(enabled and torch.cuda.is_available()))
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def capture(self, fn, *warmup_args):
        """Захватить fn в граф. На CPU просто вернуть fn без изменений."""
        if not self.enabled:
            return fn
        self.graph = torch.cuda.CUDAGraph()
        for _ in range(3):
            fn(*warmup_args)
        with torch.cuda.graph(self.graph):
            fn(*warmup_args)
        return lambda *a: self.graph.replay()
