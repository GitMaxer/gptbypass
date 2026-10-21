"""Суррогатные градиенты спайка.

Две калиброванные псевдопроизводные:
    FastSigmoid: d/dx = 1 / (1 + k|x|)^2          (левое полушарие: точность)
    Arctan:      d/dx = (a/2) / (1 + (pi a x /2)^2)  (правое: тяжёлые хвосты)

Крутизна k — НЕ константа обучения, а медленный параметр организма:
мета-контроллер (modulation/controller.py) меняет её между эпизодами,
реализуя тезис канона «5-HT управляет резкостью обучения». Внутри такта
k постоянна, поэтому горячий контур остаётся чисто тензорным.
"""
from __future__ import annotations

import math

import torch


class _SpikeSigmoid(torch.autograd.Function):
    """Forward: ступенька Хевисайда. Backward: FastSigmoid-псевдопроизводная."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float):
        ctx.save_for_backward(x)
        ctx.k = float(k)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        denom = 1.0 + ctx.k * x.abs()
        return grad_out / (denom * denom), None


class _SpikeArctan(torch.autograd.Function):
    """Forward: ступенька. Backward: Arctan с тяжёлыми хвостами."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        f = (math.pi / 2.0) * a * x
        return grad_out * (a / 2.0) / (1.0 + f * f), None


def spike_fn(x: torch.Tensor, mode: str = "fast_sigmoid", k: float = 10.0) -> torch.Tensor:
    """Пороговая функция с суррогатным градиентом.

    x — «расстояние до порога» (V - theta), НЕ сам потенциал.
    """
    if mode == "fast_sigmoid":
        return _SpikeSigmoid.apply(x, k)
    if mode == "arctan":
        return _SpikeArctan.apply(x, k)
    raise ValueError(f"unknown surrogate mode: {mode!r}")


def hebbian_step(x: torch.Tensor, mode: str = "fast_sigmoid", k: float = 10.0) -> torch.Tensor:
    """Дифференцируемый «мягкий спайк» для локальных правил (без autograd-функции).

    Используется там, где нужен именно градиент, а не событие: локальное
    предсказание в колонке (predictive coding). Возвращает сигмоиду с
    температурой 1/k — монотонную аппроксимацию ступеньки.
    """
    if mode == "arctan":
        return 0.5 + torch.atan((math.pi / 2.0) * k * x) / math.pi
    return torch.sigmoid(k * x)
