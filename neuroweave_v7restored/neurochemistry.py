# src/spiker/modulation/neurochemistry.py
# Этап 1 (канон v4.0/v4.0.1, разд. 4.2): векторная нейрохимия организма.
#
# Архитектурное решение канона: модуляторы — НЕ глобальные скаляры, а
# пространственные тензоры, привязанные к топологии зон. Реализация Этапа 1
# несёт пер-зонное разрешение (тензор [n_zones]); расширение до разрешения
# отдельных нейронов [n_zones, n_neurons] — Этап 2+ (хуки заложены).
#
# Dual-Channel Engine (разд. 8): вся динамика модуляторов — тензоры на
# устройстве в общем вычислительном графе, пошагово, без обмена с хостом.
# CPU-контур лишь изредка (каждые K ~ 100-500 шагов) забирает агрегаты
# (sleep_pressure, средние уровни) и возвращает редкие макро-команды.
import torch
import torch.nn as nn
from typing import Dict, Optional


class NeuroChemistry(nn.Module):
    """
    Метаботропный контур: DA, 5-HT, ACh, NE.

    Канонические диапазоны и роли (табл. 4):
      DA  >= 0       — RPE, разрешающий ключ закрепления следов (ур. 14);
      5-HT in [0, 1] — термодинамическая энтропия; базовый уровень задаётся
                       полушарием (высокий — левое, пониженный — правое);
      ACh in [-1, 1] — баланс Bottom-Up / Top-Down (ур. 5);
      NE  >= 0       — фазический аларм при Surprise; быстрый распад.

    Давление сна (ур. 20): dP/dt = kappa1 * SUM(s) - kappa2 * [5-HT],
    где SUM(s) — суммарная спайковая активность организма за такт (канон:
    интеграл нагрузки, а не средняя частота). Порог Theta_sleep и перевод
    организма в сон — прерогатива медленного CPU-контура (Этап 4).
    """

    AGENTS = ("da", "ht", "ach", "ne")

    def __init__(
        self,
        zone_names,
        dt: float = 1.0,
        tau_da: float = 100.0,
        tau_ht: float = 2000.0,
        tau_ach: float = 500.0,
        tau_ne: float = 20.0,
        ht_baseline=None,          # dict zone -> уровень 5-HT (или список)
        ach_target_default: float = 0.3,   # бодрствование: лёгкий приоритет входа
        kappa1: float = 5.0e-6,            # нагрузка: Σ спайков за такт (ур. 20)
        kappa2: float = 2.0e-4,            # серотониновое торможение давления сна
    ):
        super().__init__()
        self.zone_names = list(zone_names)
        self.n_zones = len(self.zone_names)
        self.zone_index = {name: i for i, name in enumerate(self.zone_names)}
        Z = self.n_zones

        # Тензоры модуляторов на зону: [Z] — векторная привязка к топологии
        self.register_buffer("da", torch.zeros(Z))
        self.register_buffer("ht", torch.full((Z,), 0.5))
        self.register_buffer("ach", torch.full((Z,), ach_target_default))
        self.register_buffer("ne", torch.zeros(Z))

        # Базовые уровни 5-HT по полушариям (табл. 2): левое ~0.8, правое ~0.3
        if ht_baseline is None:
            ht_baseline = {name: 0.5 for name in self.zone_names}
        if isinstance(ht_baseline, dict):
            base = torch.tensor(
                [float(ht_baseline.get(name, 0.5)) for name in self.zone_names],
                dtype=torch.float32,
            )
        else:
            base = torch.as_tensor(ht_baseline, dtype=torch.float32)
        self.register_buffer("ht_baseline", base)

        # Целевой режим ACh (управляется медленным контуром: бодрствование/сон)
        self.register_buffer("ach_target", torch.full((Z,), ach_target_default))

        # Коэффициенты динамики (Exact integration, на устройстве)
        self.register_buffer("decay_da", torch.exp(torch.tensor(-dt / tau_da)))
        self.register_buffer("decay_ht", torch.exp(torch.tensor(-dt / tau_ht)))
        self.register_buffer("decay_ach", torch.exp(torch.tensor(-dt / tau_ach)))
        self.register_buffer("decay_ne", torch.exp(torch.tensor(-dt / tau_ne)))

        # Давление сна (ур. 20) — тензор на устройстве; опрос — медленный контур
        self.register_buffer("sleep_pressure", torch.zeros(()))
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)

        # Отложенные DA-импульсы событий (патч S4): тензор [Z] на устройстве.
        # Патч Этапа 5 (capture-ready): импульс пластичности живёт на
        # устройстве — step_zone читает/обнуляет его ТЕНЗОРНЫМИ операциями,
        # без хост-словаря и синхронизаций (совместимость с CUDA Graphs).
        self.register_buffer("pending_da", torch.zeros(Z))

        # Инициализация 5-HT базовыми уровнями полушарий
        self.ht.copy_(self.ht_baseline)

    # ------------------------------------------------------------------
    # Источники (вызовы быстрого контура, без синхронизации с хостом)
    # ------------------------------------------------------------------
    def da_pulse(self, zone, rpe):
        """Положительный импульс дофамина: DA += max(0, RPE) (ур. 14)."""
        if isinstance(rpe, torch.Tensor):
            rpe = rpe.detach().float()
            if rpe.dim() > 0:
                rpe = rpe.mean()
        else:
            rpe = torch.tensor(float(rpe))
        with torch.no_grad():
            self.da[self.zone_index[zone]].add_(torch.clamp(rpe, min=0.0))

    def release_ne(self, zone, magnitude, pulse_max: float = 1.0):
        """Фазический выброс NE при детекции Surprise (разд. 4.2/6.3).

        Сброс залипания адаптации ALIF в зоне-источнике и соседях
        (межзонная диффузия — Этап 2, пока локально).

        Патч S3 (канон v4.0.1): амплитуда выброса ОГРАНИЧЕНА (pulse_max).
        Сырая величина Surprise амплитудой быть не может: при частых
        алармах NE насыщается до клэмпа и гасит адаптацию зон перманентно
        (шторм возбуждения, выявлено сборкой Этапа 2). Импульс — событие
        ограниченной силы; детекция фронта — в MultiCompartmentALIF.

        Патч Этапа 5 (capture-ready): magnitude — скалярный ТЕНЗОР (или
        float): клэмп и сложение — тензорные операции на устройстве,
        без приведения к хосту (совместимость с CUDA Graphs)."""
        with torch.no_grad():
            if isinstance(magnitude, torch.Tensor):
                mag = magnitude.detach().float()
            else:
                mag = torch.tensor(float(magnitude), device=self.ne.device)
            pulse = torch.clamp(mag, min=0.0, max=float(pulse_max))
            self.ne[self.zone_index[zone]].add_(pulse)
            self.ne.clamp_(min=0.0, max=2.0)

    # ------------------------------------------------------------------
    # Отложенные DA-импульсы (событийная семантика S4, тензорный контур)
    # ------------------------------------------------------------------
    def da_pending_add(self, zone, amount):
        """Накопить амплитуду DA-события зоны (тензорная операция)."""
        with torch.no_grad():
            if isinstance(amount, torch.Tensor):
                a = amount.detach().float()
            else:
                a = torch.tensor(float(amount), device=self.pending_da.device)
            self.pending_da[self.zone_index[zone]].add_(
                torch.clamp(a, min=0.0))

    def da_pending_consume(self, zone) -> torch.Tensor:
        """Считать и обнулить отложенный импульс зоны (0-dim тензор).

        Импульс расходуется РОВНО один раз — в такте, следующем за
        событием награды (биологическая задержка выброса, патч S4)."""
        with torch.no_grad():
            idx = self.zone_index[zone]
            value = self.pending_da[idx].clone()
            self.pending_da[idx].zero_()
            return value

    def reset_pending(self):
        """Новый эпизод: отложенные импульсы гаснут (S4)."""
        self.pending_da.zero_()

    def set_ach_target(self, zone, target: float):
        """Макро-команда медленного контура: целевой режим ACh зоны.

        Бодрствование: target ~ +0.3...+1.0 (приоритет сенсорного входа);
        сон (Этап 4): target ~ −0.8 (приоритет апикальных реплеев, ур. 5).
        """
        with torch.no_grad():
            self.ach_target[self.zone_index[zone]].fill_(
                min(1.0, max(-1.0, float(target)))
            )

    # ------------------------------------------------------------------
    # Пошаговая динамика быстрого контура
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        spike_sums: Optional[Dict[str, torch.Tensor]] = None,
        da_rpe: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Один такт dt. Все тензоры остаются на устройстве (разд. 8).

        spike_sums: dict zone -> скалярный тензор СУММЫ спайков зоны за такт
        (Σs в интеграле давления сна, ур. 20 — каноническая сумма, не среднее).
        da_rpe: dict zone -> скалярный тензор RPE (награда приходит событийно).
        """
        # DA: медленный распад (импульсы приходят через da_pulse/da_rpe)
        self.da.mul_(self.decay_da)
        if da_rpe:
            for zone, rpe in da_rpe.items():
                self.da_pulse(zone, rpe)

        # 5-HT: релаксация к базовому уровню полушария
        self.ht.add_((self.ht_baseline - self.ht) * (1.0 - self.decay_ht))

        # ACh: релаксация к целевому режиму (задаёт медленный контур)
        self.ach.add_((self.ach_target - self.ach) * (1.0 - self.decay_ach))

        # NE: быстрый фазический распад (~20 мс)
        self.ne.mul_(self.decay_ne)

        # Давление сна (ур. 20): интеграл нагрузки минус серотониновый запас
        if spike_sums is not None and len(spike_sums) > 0:
            total_spikes = torch.stack(
                [torch.as_tensor(s, device=self.da.device).float().sum()
                 for s in spike_sums.values()]
            ).sum()
            self.sleep_pressure.add_(
                self.kappa1 * total_spikes - self.kappa2 * self.ht.mean()
            )

        # Канонические диапазоны (патч S5: верхняя граница DA = 2.0, как у NE —
        # уровень не может расти неограниченно при частых событиях награды:
        # без клэмпа стационар уровня уходит за 4 при R ~ 0.6, вскрыто
        # зондом живого цикла Этапа 3)
        self.da.clamp_(min=0.0, max=2.0)
        self.ne.clamp_(min=0.0, max=2.0)
        self.ht.clamp_(0.0, 1.0)
        self.ach.clamp_(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Выдача модуляторов потребителям (зонам) — broadcast-тензоры [B, 1]
    # ------------------------------------------------------------------
    def get(self, agent: str, zone, batch_size: int) -> torch.Tensor:
        """Уровень агента для зоны, готовый к broadcast в [B, N]."""
        if agent not in self.AGENTS:
            raise ValueError(f"Unknown agent: {agent}")
        value = getattr(self, agent)[self.zone_index[zone]]
        return value.view(1, 1).expand(batch_size, 1)

    @property
    def telemetry(self) -> Dict[str, float]:
        """Агрегаты для медленного CPU-контура / логирования (вызывать редко)."""
        return {
            "da_mean": float(self.da.mean()),
            "ht_mean": float(self.ht.mean()),
            "ach_mean": float(self.ach.mean()),
            "ne_mean": float(self.ne.mean()),
            "sleep_pressure": float(self.sleep_pressure),
        }
