"""TemporalPhaseEncoder — порт из Spiker (сенсорный интерфейс, Этап 5).

Два режима кодирования токенов в спайковые поезда:
  * burst — каждый токен занимает окно burst_length тактов;
  * latency — один спайк на токен, латентность обратно пропорциональна
    частотности (салиентное приходит раньше — прямой механизм внимания
    на входе, не через обучение весов).

В v9 дополнительно используется `latency_salience()`: нормированный
приоритет токена [0..1] — скалярный гейт входа (прескалинг one-hot перед
таламусом), без изменения каденции тактов.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TemporalPhaseEncoder(nn.Module):
    """Токены -> спайковые поезда (burst/latency) + салиентность входа."""

    def __init__(
        self,
        vocab_size: int,
        burst_length: int = 5,
        max_latency: int = 8,
        mode: str = "burst",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.burst_length = max(1, int(burst_length))
        self.max_latency = max(1, int(max_latency))
        if mode not in ("burst", "latency", "both"):
            raise ValueError(f"Неизвестный режим кодирования: {mode}")
        self.mode = mode
        # EMA-частотность токенов для latency-салиентности
        self.register_buffer(
            "freq_ema", torch.ones(vocab_size) / max(1, vocab_size))

    # ------------------------------------------------------------------ burst
    def encode_burst(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[T, B] -> [T*burst_length, B, vocab]: спайк в каждом такте окна."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)
        t_len, batch = token_ids.shape
        out = torch.zeros(
            (t_len * self.burst_length, batch, self.vocab_size),
            device=token_ids.device)
        for t in range(t_len):
            base = t * self.burst_length
            for k in range(self.burst_length):
                out[base + k].scatter_(1, token_ids[t].unsqueeze(-1), 1.0)
        return out

    # ---------------------------------------------------------------- latency
    def encode_latency(
        self,
        token_ids: torch.Tensor,
        frequency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[T, B] -> [T*max_latency, B, vocab]: спайк в такте латентности."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)
        t_len, batch = token_ids.shape
        device = token_ids.device

        freq = (self.freq_ema if frequency is None
                else frequency.to(device=device, dtype=torch.float32))
        f_min, f_max = freq.min(), freq.max()
        if float(f_max - f_min) < 1e-9:
            norm = torch.zeros_like(freq)
        else:
            norm = (f_max - freq) / (f_max - f_min)
        latency = torch.round(norm * (self.max_latency - 1)).long().clamp(
            0, self.max_latency - 1)

        out = torch.zeros(
            (t_len * self.max_latency, batch, self.vocab_size),
            device=device)
        for t in range(t_len):
            base = t * self.max_latency
            for b in range(batch):
                tok = int(token_ids[t, b])
                out[base + int(latency[tok]), b, tok] = 1.0
        return out

    # ------------------------------------------------------------- salience
    def observe_tokens(self, token_ids: torch.Tensor, ema: float = 0.99) -> None:
        """Обновить EMA-частотность по потоку (для latency-кода)."""
        with torch.no_grad():
            counts = torch.bincount(
                token_ids.flatten(), minlength=self.vocab_size).float()
            freq = counts / counts.sum().clamp_min(1.0)
            self.freq_ema.mul_(ema).add_(freq * (1.0 - ema))

    def latency_salience(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Салиентность токена [B]: редкое = раньше = громче.

        Возвращает скалярный гейт в [0.5, 1.0] для прескалинга one-hot
        входа: салиентные (редкие) токены «приходят раньше» — в постоянной
        каденции это выражается как большая амплитуда входа.
        """
        f_min, f_max = self.freq_ema.min(), self.freq_ema.max()
        if float(f_max - f_min) < 1e-9:
            return torch.full(token_ids.shape, 0.75, device=token_ids.device)
        norm = (f_max - self.freq_ema) / (f_max - f_min)  # редкое -> 1
        sal = 0.5 + 0.5 * norm[token_ids.long()]
        return sal
