"""Событийно-разреженный путь вычислений.

Зачем: канон обещает «спайки как фундамент вычислительного графа», но и v6,
и v7 в горячем такте считают ПОЛНЫЕ матричные произведения — плотность
топологии используется как маска, а не как экономия. Здесь появляется
второй, честно измеримый путь: стоимость пропорциональна числу АКТИВНЫХ
нейронов, а не размеру пула.

Инженерный компромисс формулируется явно и проверяется тестом:

  * ``dense_drive``  — [B,N_pre] @ W^T: 0 host-синхронизаций, дружит с
    ``torch.cuda.CUDAGraph``, но платит за все N_pre;
  * ``sparse_drive`` — индексация активных столбцов: платит за |active|,
    но требует ``nonzero()``, то есть host-синхронизации. Годится в eager-
    режиме и на CPU/edge, не годится внутри захваченного графа.

Числа обоих путей обязаны совпадать с точностью float32 — это контракт
``test_sparse_equiv.py``.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def active_indices(s_pre: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
    """Индексы активных входов (объединение по батчу)."""
    mask = (s_pre.abs() > thresh).any(dim=0)
    return torch.nonzero(mask, as_tuple=False).flatten()


def dense_drive(s_pre: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(s_pre, w)


def sparse_drive(
    s_pre: torch.Tensor,
    w: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Драйв по активному подмножеству входов и/или выходов.

    idx     — активные столбцы W (входы); out_idx — активные строки (выходы).
    Отсутствующий индекс означает «все».
    """
    if idx is None:
        idx = active_indices(s_pre)
    out_shape = (s_pre.shape[0], w.shape[0] if out_idx is None else out_idx.numel())
    if idx.numel() == 0:
        return torch.zeros(out_shape, dtype=s_pre.dtype, device=s_pre.device)

    w_sel = w.index_select(1, idx)
    if out_idx is not None:
        w_sel = w_sel.index_select(0, out_idx)
    s_sel = s_pre.index_select(1, idx)
    return F.linear(s_sel, w_sel)


def k_wta(x: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Оставить k наибольших значений в каждой строке (жёсткая разреженность)."""
    if k >= x.shape[-1]:
        return x if mask is None else x * mask
    vals, idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, vals)
    return out if mask is None else out * mask


def sparse_code(x: torch.Tensor, k: int, thresh: float = 0.0) -> torch.Tensor:
    """Топ-k по неотрицательной части + порог: код DG-типа."""
    pos = torch.clamp(x - thresh, min=0.0)
    return k_wta(pos, k)


class EventStats:
    """Счётчик реальной экономии: сколько драйва НЕ посчитано."""

    def __init__(self) -> None:
        self.dense_ops = 0
        self.sparse_ops = 0
        self.calls = 0

    def record(self, n_pre: int, n_active: int, n_out: int, batch: int) -> None:
        self.dense_ops += n_pre * n_out * batch
        self.sparse_ops += max(1, n_active) * n_out * batch
        self.calls += 1

    @property
    def saving(self) -> float:
        if self.dense_ops == 0:
            return 1.0
        return self.sparse_ops / self.dense_ops

    def reset(self) -> None:
        self.dense_ops = self.sparse_ops = self.calls = 0

    def __repr__(self) -> str:
        return (f"EventStats(calls={self.calls}, "
                f"work_ratio={self.saving:.3f})")


@torch.no_grad()
def benchmark_paths(
    n_in: int = 2048,
    n_out: int = 1024,
    batch: int = 4,
    activity: float = 0.03,
    repeats: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Сравнение плотного и событийного пути: время и доля работы."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_out, n_in, generator=g) / (n_in ** 0.5)
    s = (torch.rand(batch, n_in, generator=g) < activity).float()

    # прогрев
    for _ in range(3):
        dense_drive(s, w)
        sparse_drive(s, w)

    t0 = time.perf_counter()
    for _ in range(repeats):
        dense_drive(s, w)
    t_dense = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        sparse_drive(s, w)
    t_sparse = (time.perf_counter() - t0) / repeats

    d = dense_drive(s, w)
    sp = sparse_drive(s, w)
    return {
        "t_dense_ms": t_dense * 1e3,
        "t_sparse_ms": t_sparse * 1e3,
        "speedup": t_dense / max(t_sparse, 1e-12),
        "max_abs_diff": float((d - sp).abs().max()),
        "activity": float(s.mean()),
        "n_active": float(active_indices(s).numel()),
    }
