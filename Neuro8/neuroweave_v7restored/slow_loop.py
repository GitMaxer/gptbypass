# src/spiker/organism/slow_loop.py
# Этап 4 (канон v4.0/v4.0.1, разд. 8 + табл. 5): асинхронный CPU-метаконтроллер.
#
# Dual-Channel Engine: быстрый контур (спайки, синапсы, нейрохимия) живёт
# на устройстве без хоста; SlowLoopController — редкий гость: каждые
# poll_interval тактов он забирает АГРЕГАТЫ (телеметрия) и возвращает
# МАКРО-КОМАНДЫ. Это архитектурная экономия PCIe (разд. 8): обмен —
# единожды за K ~ 100-500 шагов, а не каждый такт.
#
# Макро-решения канона (табл. 5):
#   * триггер сна: sleep_pressure > Theta_sleep (ур. 20) -> команда SLEEP;
#   * режимы ACh: бодрствование +0.3 / сон -0.8 (через NeuroChemistry);
#   * наблюдение за критиком: адаптация к успеху (привыкание DA);
#   * логирование телеметрии.
from typing import Dict, Optional


class SlowLoopController:
    """
    Метаконтроллер организма: редкий опрос, редкие решения.

    poll() вызывается организмом/демо на каждом такте, но РЕАЛЬНО работает
    раз в poll_interval тактов (счётчик тактов внутри): остальные вызовы —
    бесплатный no-op, быстрый контур не ждёт хост.
    """

    SLEEP = "sleep"
    NONE = None

    def __init__(
        self,
        theta_sleep: float = 1.0,
        poll_interval: int = 200,
        min_cycles_between_sleep: int = 1,
    ):
        self.theta_sleep = float(theta_sleep)
        self.poll_interval = max(1, int(poll_interval))
        self._since_poll = 0
        self._sleep_cycles_done = 0
        self._since_last_sleep = 0
        self.min_cycles_between_sleep = int(min_cycles_between_sleep)
        self.history: list = []

    # ------------------------------------------------------------------
    def poll(self, telemetry: Dict[str, float], step: Optional[int] = None):
        """Опрос агрегатов (вызывать на каждом такте; работает раз в K).

        Возвращает:
            None  — решений нет (бодрствуем);
            'sleep' — давление сна превысило порог, организму пора спать.
        """
        self._since_poll += 1
        self._since_last_sleep += 1
        if step is not None:
            self._since_poll = int(step) % self.poll_interval == 0 \
                and self.poll_interval or self._since_poll
        if self._since_poll < self.poll_interval:
            return self.NONE
        self._since_poll = 0

        p_sleep = float(telemetry.get("sleep_pressure", 0.0))
        self.history.append(dict(telemetry))
        if len(self.history) > 256:
            self.history.pop(0)

        if (
            p_sleep > self.theta_sleep
            and self._since_last_sleep >= self.poll_interval * self.min_cycles_between_sleep
        ):
            self._since_last_sleep = 0
            self._sleep_cycles_done += 1
            return self.SLEEP
        return self.NONE

    # ------------------------------------------------------------------
    def notify_slept(self):
        """Подтверждение: цикл сна выполнен (сброс интервала гистерезиса)."""
        self._since_last_sleep = 0

    @property
    def sleep_cycles_done(self) -> int:
        return self._sleep_cycles_done