"""Медленный CPU-метаконтур: редкий опрос, редкие решения (разд. 8)."""
from __future__ import annotations


class SlowLoopController:
    SLEEP = "sleep"
    NONE = None

    def __init__(self, theta_sleep: float = 1.0, poll_interval: int = 200,
                 min_cycles_between_sleep: int = 1):
        self.theta_sleep = float(theta_sleep)
        self.poll_interval = max(1, int(poll_interval))
        self.min_gap = max(0, int(min_cycles_between_sleep))
        self._since_poll = 0
        self._since_sleep = 0
        self.sleeps = 0

    def poll(self, sleep_pressure: float):
        """Вызывается каждый такт; реальная работа — раз в poll_interval."""
        self._since_poll += 1
        self._since_sleep += 1
        if self._since_poll < self.poll_interval:
            return self.NONE
        self._since_poll = 0
        rested = self.sleeps == 0 or self._since_sleep >= self.min_gap
        if rested and float(sleep_pressure) > self.theta_sleep:
            return self.SLEEP
        return self.NONE

    def register_sleep(self) -> None:
        self.sleeps += 1
        self._since_sleep = 0
