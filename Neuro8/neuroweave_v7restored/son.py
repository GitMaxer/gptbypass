# src/spiker/modulation/son.py
# Этап 1 (канон v4.0/v4.0.1, разд. 6): Self-Observation Neurons —
# активная обучаемая межзональная шина само-наблюдения.
#
# На зону — ансамбль из 16 нейронов (канон). Четыре функции:
#   1. Компрессия: окно спайков зоны -> латентный спайковый вектор z (ур. 15);
#   2. AR-предиктор: локально рекуррентное предсказание z(t+1);
#   3. Surprise (ур. 17): ||z - z_hat||^2 — измеритель неожиданности;
#   4. Гейтинг Zero-Compute: низкий Surprise -> шина молчит (без спайков,
#      без трафика — событийная экономия энергии); высокий -> экспорт
#      латента, приоритетный пакет и триггер NE.
#
# Декодер замыкается на АПИКАЛЬНЫЕ дендриты зоны-получателя (ур. 16):
# пересланная информация модулирует, но не запускает реакцию сама по себе
# (NMDA-гейт, ур. 4) — межзонное влияние принципиально контекстное.
#
# Обучение Этапа 1: только AR-предиктор (локальный MSE, онлайн, без
# autograd-графа). Энкодер/декодер проекции обучаются в Фазе 2 (SON-to-SON).
import math
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from ..core.surrogate import spike_fn


class SONBus(nn.Module):
    """Шина само-наблюдения одной зоны."""

    def __init__(
        self,
        zone_name: str,
        n_zone_neurons: int,
        n_son: int = 16,          # канон: 16 нейронов на зону
        window: int = 8,          # окно сжатия состояния зоны (ур. 15)
        theta_son: float = 1.0,   # порог экспорта: Surprise-гейтинг
        theta_alarm: float = 6.0, # порог аларма: выброс NE
        pulse_max: float = 1.0,    # ограничение амплитуды NE-импульса (S3)
        lr_pred: float = 2.0e-2,  # темп онлайн-обучения AR-предиктора
        surrogate_mode: str = "fast_sigmoid",
        surrogate_slope: float = 10.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.zone_name = zone_name
        self.n_son = n_son
        self.window = window
        self.n_zone = n_zone_neurons
        self.theta_son = theta_son
        self.theta_alarm = theta_alarm
        self.pulse_max = float(pulse_max)
        self.lr_pred = lr_pred
        self.surrogate_mode = surrogate_mode
        self.surrogate_slope = surrogate_slope
        self.surrogate_alpha = surrogate_alpha

        # Энкодер: окно спайков [B, window*N] -> латентные логиты [B, 16]
        self.encoder = nn.Linear(window * n_zone_neurons, n_son, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))

        # AR-предиктор: z_hat(t+1) = z(t) @ W_pred^T — локально рекуррентный
        self.W_pred = nn.Parameter(torch.eye(n_son) * 0.5)

        # Декодер: латент -> ток на апикальные дендриты получателя (ур. 16)
        self.decoder = nn.Linear(n_son, n_zone_neurons, bias=False)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

        # Состояния: кольцевое окно спайков зоны и предыдущий латент
        self.spike_window: Optional[torch.Tensor] = None   # [B, window*N]
        self.z_prev: Optional[torch.Tensor] = None         # [B, n_son]
        # Амплитуда NE-аларма последнего такта (тензор, вычислена безветвево;
        # организм читает её для тензорного выброса NE — патч Этапа 5,
        # capture-ready)
        self.alarm_mag: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        self.spike_window = torch.zeros(
            (batch_size, self.window * self.n_zone), device=device, dtype=dtype
        )
        self.z_prev = torch.zeros((batch_size, self.n_son), device=device, dtype=dtype)

    def _predict(self, z_prev: torch.Tensor) -> torch.Tensor:
        return z_prev @ self.W_pred.T

    def step(
        self,
        s_zone: torch.Tensor,
        on_alarm: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Один такт шины.

        Параметры:
            s_zone:   [B, N] — текущие спайки зоны (следит за ними);
            on_alarm: колбэк (zone_name, surprise) для выброса NE
                      (проводится организмом; в тестах — напрямую).
        Возвращает:
            surprise: [B] — неожиданность (ур. 17);
            z_export: [B, n_son] — латент для шины (нули при низком Surprise);
            z:        [B, n_son] — «сырой» латент (для телеметрии/Фазы 2).
        """
        if (
            self.spike_window is None
            or self.spike_window.shape[0] != s_zone.shape[0]
            or self.spike_window.device != s_zone.device
        ):
            self.reset_state(s_zone.shape[0], s_zone.device, s_zone.dtype)

        # 1. Сдвиг кольцевого окна: [старое..новое]
        s_detached = s_zone.detach()
        self.spike_window = torch.cat(
            [self.spike_window[:, self.n_zone:], s_detached], dim=1
        )

        # 2. Кодирование: латентные спайки (суррогат проходим при Фазе 2)
        z_logits = self.encoder(self.spike_window)
        z = spike_fn(
            z_logits,
            mode=self.surrogate_mode,
            slope=self.surrogate_slope,
            alpha=self.surrogate_alpha,
        )

        # 3. Предсказание и Surprise (ур. 17) ДО онлайн-обновления предиктора
        z_hat = self._predict(self.z_prev)
        surprise = ((z - z_hat.detach()) ** 2).sum(dim=-1).detach()   # [B]

        # 4. Локальное онлайн-обучение предиктора: MSE ||W z_prev - z||^2.
        #    Ручной градиентный шаг без autograd-графа — O(1)-память,
        #    счёт на устройстве, learning-in-the-moment.
        with torch.no_grad():
            err = z_hat - z.detach()                          # [B, S]
            grad = 2.0 * (err.T @ self.z_prev) / max(1, z.shape[0])
            self.W_pred.data -= self.lr_pred * grad

        # 5. Zero-Compute гейтинг (разд. 6.3): тихий режим не занимает шину
        export_mask = (surprise > self.theta_son).to(z.dtype).unsqueeze(-1)
        z_export = z * export_mask

        # 6. Аларм -> фазический NE (сброс залипания адаптации в зонах).
        #    Патч Этапа 5 (capture-ready): амплитуда аларма — ТЕНЗОР,
        #    вычислена безветвево (клэмп S3 x маска события) и доступна
        #    организму как self.alarm_mag для тензорного выброса NE БЕЗ
        #    колбэка в горячем такте. Колбэк on_alarm сохранён для прямых
        #    вызовов (тесты, изолированные зоны) и стреляет ТОЛЬКО по
        #    событию — исходный контракт сохранён.
        peak = surprise.max()
        self.alarm_mag = torch.clamp(peak, min=0.0, max=self.pulse_max) \
            * (peak > self.theta_alarm).to(surprise.dtype)
        if on_alarm is not None and bool(self.alarm_mag.item() > 0.0):
            on_alarm(self.zone_name, self.alarm_mag)

        self.z_prev = z.detach()
        return surprise, z_export, z

    def decode(self, z_export: torch.Tensor) -> torch.Tensor:
        """Латент -> апикальный ток зоны-получателя (ур. 16)."""
        return self.decoder(z_export)
