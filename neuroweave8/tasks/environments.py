"""Среды: марковский поток с инверсией (эксперимент M1_inv канона v4.0.1).

Эпизод = последовательность токенов; сенсорный вход — one-hot паттерн
[B, vocab]; цель — предсказание следующего токена. Инверсия на середине
эпизода (правило A->B меняется на B->A) — канонический тест быстрой
адаптации w_fast vs защиты w_slow.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class MarkovStream:
    def __init__(self, vocab: int = 8, seed: int = 8, inverted: bool = False):
        self.vocab = int(vocab)
        self.g = torch.Generator().manual_seed(seed)
        self.inverted = inverted

    def next_token(self, tok: torch.Tensor) -> torch.Tensor:
        rule = (tok + 1) % self.vocab
        if self.inverted:
            rule = (tok - 1) % self.vocab
        return rule

    def step(self, tok: torch.Tensor, invert_now: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (one-hot вход текущего токена, целевой next)."""
        if invert_now:
            self.inverted = not self.inverted
        x = torch.zeros(tok.shape[0], self.vocab)
        x.scatter_(1, tok.unsqueeze(1), 1.0)
        y = self.next_token(tok)
        return x, y


class BindingTask:
    """Роль-филлер через θ-γ: 'субъект S делает V объекту O' -> 3 слота."""
    def __init__(self, vocab: int = 12, seed: int = 33):
        self.vocab = int(vocab)
        self.g = torch.Generator().manual_seed(seed)

    def episode(self, n_steps: int, batch: int = 1):
        x = torch.randint(0, self.vocab, (n_steps, batch), generator=self.g)
        return x


class RuleShiftTask:
    """Задача смены правила в середине эпизода (M1_inv)."""
    def __init__(self, vocab: int = 8, switch_at: int = 40, seed: int = 8):
        self.env = MarkovStream(vocab=vocab, seed=seed)
        self.switch_at = int(switch_at)
        self.vocab = int(vocab)

    def run_episode(self, n_steps: int, batch: int = 1):
        """Генерирует (x_seq [T,B,V], y_seq [T,B]) с инверсией на switch_at."""
        tok = torch.randint(0, self.vocab, (batch,), generator=self.env.g)
        xs, ys = [], []
        for t in range(n_steps):
            x, y = self.env.step(tok, invert_now=(t == self.switch_at))
            xs.append(x)
            ys.append(y)
            tok = y
        return torch.stack(xs), torch.stack(ys)
