# src/spiker/core/quant.py
# Этап 5 (канон v4.0/v4.0.1, разд. 8.3 + табл. 6): квантизация некритичных
# путей — int8-активности шины и тернарные веса (−1, 0, +1) проекций.
#
# Канон (разд. 8.3): «квантизация: int8-активности и тернарные веса
# (−1, 0, +1) в некритичных путях дают кратную экономию памяти и полосы
# пропускания». Некритичный путь организма — шина Corpus Callosum:
#   * КОЛЬЦО ЗАДЕРЖЕК — постоянная VRAM-память шины: int8-хранение
#     (масштаб на сообщение хранится рядом, [delay] скаляров) — 4x экономия
#     памяти против float32;
#   * ПРОЕКЦИИ W_l2r/W_r2l — QAT-схема: полный мастер-вес обучается
#     (Reward-STDP), эффективная матрица — ТЕРНАРНАЯ {−scale, 0, +scale}
#     (2 бита информации + 1 масштаб; storage int8 + знаковый бит — 4x).
#
# Тензоро-безопасность (разд. 8.3): все операции — плотные тензорные, без
# ветвлений по данным — совместимость с CUDA Graphs (Этап 5-b).
import torch


# ----------------------------------------------------------------------
# int8: симметричное per-tensor квантование
# ----------------------------------------------------------------------
def int8_scale(x: torch.Tensor) -> torch.Tensor:
    """Симметричный масштаб per-tensor: s = max(|x|) / 127 (>= eps)."""
    return (x.detach().abs().max() / 127.0).clamp(min=1e-12)


def quantize_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Округлить на int8-сетку: q = round(x/s)·s, |q/s| <= 127."""
    q = torch.round(x.detach() / scale).clamp_(-127.0, 127.0)
    return q * scale


def fake_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """Fake-quant: значения на int8-сетке (динамика проверяется,
    dtype остаётся float — STE-градиент сохранён при наличии графа)."""
    scale = int8_scale(x)
    q = quantize_int8(x, scale)
    # Straight-Through Estimator: градиент течёт сквозь квантование
    return x + (q - x).detach()


def dequantize(q_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Восстановить float из int8-хранилища и масштаба."""
    return q_int8.to(torch.float32) * scale


# ----------------------------------------------------------------------
# Тернарные веса: {−scale, 0, +scale}
# ----------------------------------------------------------------------
def ternarize(w: torch.Tensor, thr_frac: float = 0.7) -> torch.Tensor:
    """Тернаризация (Qiu et al., TWN-стиль, порог по среднему |w|).

        thr   = thr_frac · mean(|w|)            — порог отсечения мелочи;
        scale = mean(|w| где |w| > thr)         — амплитуда выживших;
        w_q   = scale · Sign(w) · 1[|w| > thr]  — ровно три значения.

    Мастер-вес остаётся полным (обучается Reward-STDP); тернарная —
    ЭФФЕКТИВНАЯ матрица прямого прохода шины.
    """
    w = w.detach()
    magnitude = w.abs()
    thr = thr_frac * magnitude.mean()
    survivors = magnitude > thr
    n_alive = survivors.sum()
    # scale = средний |w| выживших; при пустом множестве — 0 (вся матрица
    # нулевая — вырожденный, но корректный случай)
    scale = torch.where(
        n_alive > 0,
        magnitude[survivors].sum() / n_alive.clamp(min=1),
        torch.zeros((), device=w.device, dtype=w.dtype),
    )
    return scale * torch.sign(w) * survivors.to(w.dtype)


def quant_bandwidth_report(tensor: torch.Tensor, int8_dtype=torch.int8):
    """Учёт экономии памяти/полосы: float32 -> int8 (теоретический фактор)."""
    bytes_f32 = tensor.numel() * tensor.element_size()
    # int8_dtype — это torch.dtype (напр. torch.int8), у него нет метода
    # element_size(); корректно читаем размер через пустой тензор.
    bytes_int8 = tensor.numel() * torch.empty(0, dtype=int8_dtype).element_size()
    return {
        "bytes_float32": int(bytes_f32),
        "bytes_int8": int(bytes_int8),
        "reduction_x": float(bytes_f32) / float(bytes_int8),
    }