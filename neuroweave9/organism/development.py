"""Онтогенетический контроллер развития — порт из Spiker (DevelopmentalController).

Прогрессивное расписание обучения (Curriculum Learning): стадии развития
переключаются по мере того, как EMA точности превышает порог освоения,
расширяя активный словарь и длину последовательности.

В v9 структурный нейрогенез BatchedCorticalColumns заменён масштабированием
колонок через `scale` CortexOrganism и StructuralPlasticity из v8 —
онтогенез управляет РАСПИСАНИЕМ данных, не перестройкой топологии.
"""
from __future__ import annotations

from typing import Dict, List, Optional


class DevelopmentalController:
    """Контроллер онтогенеза: curriculum learning по EMA-мастерству."""

    def __init__(
        self,
        vocab_stages: Optional[List[int]] = None,
        seq_len_stages: Optional[List[int]] = None,
        mastery_accuracy_threshold: float = 0.80,
        ema_decay: float = 0.85,
        min_steps_per_stage: int = 5,
    ):
        self.vocab_stages = vocab_stages or [8, 32, 128, 1024, 8192]
        self.seq_len_stages = seq_len_stages or [4, 8, 16, 32, 64]
        assert len(self.vocab_stages) == len(self.seq_len_stages)

        self.mastery_accuracy_threshold = float(mastery_accuracy_threshold)
        self.ema_decay = float(ema_decay)
        self.min_steps_per_stage = max(1, int(min_steps_per_stage))

        self.current_stage: int = 0
        self.steps_in_stage: int = 0
        self.ema_accuracy: float = 0.0
        self.ema_loss: float = 10.0
        self.neurogenesis_events: int = 0

    @property
    def active_vocab_limit(self) -> int:
        return self.vocab_stages[self.current_stage]

    @property
    def active_seq_len(self) -> int:
        return self.seq_len_stages[self.current_stage]

    def clamp_tokens(self, token_ids):
        """Ограничить словарь текущей стадии (токены за пределами -> OOV=1)."""
        return token_ids.clamp(max=self.active_vocab_limit - 1)

    def observe_performance(
        self, accuracy: float, loss: float
    ) -> Dict[str, float]:
        """Обновляет EMA-метрики; при пороге мастерства — повышение стадии."""
        if self.steps_in_stage == 0 and self.current_stage == 0:
            self.ema_accuracy = float(accuracy)
            self.ema_loss = float(loss)
        else:
            self.ema_accuracy = (self.ema_decay * self.ema_accuracy
                                 + (1.0 - self.ema_decay) * float(accuracy))
            self.ema_loss = (self.ema_decay * self.ema_loss
                             + (1.0 - self.ema_decay) * float(loss))

        self.steps_in_stage += 1
        promoted = 0.0
        if (self.steps_in_stage >= self.min_steps_per_stage
                and self.ema_accuracy >= self.mastery_accuracy_threshold
                and self.current_stage < len(self.vocab_stages) - 1):
            self.current_stage += 1
            self.steps_in_stage = 0
            promoted = 1.0

        return {
            "stage": float(self.current_stage),
            "promoted": promoted,
            "active_vocab_limit": float(self.active_vocab_limit),
            "active_seq_len": float(self.active_seq_len),
            "ema_accuracy": float(self.ema_accuracy),
            "ema_loss": float(self.ema_loss),
        }
