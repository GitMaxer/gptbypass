# src/spiker/core/neuron.py
# Канон v4.0/v4.0.1 + Этап 6 (GABA-реверс). Патчи: N1/N2 (тензоро-безопасный
# шум, флаг stochastic — шум есть свойство организма и в инференсе),
# N7 (guard сброса по устройству).
#
# ЭТАП 6 — МЕМБРАННОЕ УРАВНЕНИЕ С ПРОВОДИМОСТЬЮ GABA (ур. 6'):
#     tau_m · dV/dt = (v_rest − V) + r_m·i_exc − r_m·g_GABA·(V − E_Cl)
# G = r_m·g_GABA — безразмерная проводимость (g_gaba из transmitters).
# Exact integration при кусочно-постоянной G на такте (стандарт SNN-симуляции):
#     alpha_eff = exp(−dt·(1+G)/tau_m) = alpha_m^(1+G)      — ускоренный leak;
#     V_inf     = (v_rest + G·E_Cl + r_m·i_exc) / (1+G)     — смещённый стационар;
#     V(t+dt)   = alpha_eff·V(t) + (1 − alpha_eff)·V_inf.
# Это ЗАМКНУТАЯ форма: никакой явной синхронизации ток-напряжение не нужно,
# шаг остаётся тензорным и безветвевым (совместимость с CUDA Graphs).
# При G = 0 форма побитово совпадает с ур. 6 канона (alpha_eff = alpha_m,
# V_inf = v_rest + r_m·i_soma). ГАРАНТИЯ ПОЛА: при E_Cl >= v_rest
# стационар чистого торможения = (v_rest + G·E_Cl)/(1+G) >= v_rest —
# мембрана НЕ МОЖЕТ уйти ниже v_rest при любой глубине торможения
# (критично протоколу сна: реплей при глубоком GABA не умирает);
# при E_Cl < v_rest гиперполяризация ограничена полом НА E_Cl.
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .surrogate import spike_fn
from .transmitters import IonotropicTransmitters


class MultiCompartmentALIF(nn.Module):
    """
    Трёхсегментный нейрон: базальный дендрит (AMPA), апикальный дендрит (NMDA),
    сома ALIF с адаптивным порогом. Интегрирует ионотропные токи и управляется
    метаботропными модуляторами (ACh, NE, 5-HT).

    Патч N2 (канон v4.0.1): шум мембраны — конструктивная стохастика организма
    (правое полушарие σ ≈ 0.05 по табл. 2), поэтому он НЕ гейтируется режимом
    training. Управление — флагом `stochastic`; тесты задают σ = 0.
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,
        tau_m: float = 10.0,      # 5-10 мс — левое, 25-50 мс — правое полушарие
        tau_a: float = 100.0,     # константа времени адаптации порога
        v_rest: float = 0.0,
        theta_0: float = 1.0,     # базовый порог спайка
        beta_adapt: float = 0.15, # сила шага адаптации порога
        r_m: float = 1.0,
        sigma_noise: float = 0.001,  # базовый шум мембраны
        surrogate_mode: str = "fast_sigmoid",
        surrogate_slope: float = 10.0,
        surrogate_alpha: float = 2.0,
        stochastic: bool = True,  # патч N2: стохастика активна и в инференсе
        **transmitter_kwargs,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.v_rest = v_rest
        self.theta_0 = theta_0
        self.beta_adapt = beta_adapt
        self.r_m = r_m
        self.sigma_noise = sigma_noise
        self.surrogate_mode = surrogate_mode
        self.surrogate_slope = surrogate_slope
        self.surrogate_alpha = surrogate_alpha
        self.stochastic = stochastic

        # Экспоненциальные константы мембраны и адаптации
        self.register_buffer("alpha_m", torch.exp(torch.tensor(-dt / tau_m)))
        self.register_buffer("alpha_a", torch.exp(torch.tensor(-dt / tau_a)))
        # Этап 6: логарифм alpha_m — для alpha_eff = alpha_m^(1+G) одним exp
        # (стабильнее и быстрее поэлементного pow при тензорном показателе)
        self.register_buffer("log_alpha_m", torch.tensor(-dt / tau_m))

        # Ионотропный аппарат дендритов
        self.transmitters = IonotropicTransmitters(
            num_neurons, dt=dt, **transmitter_kwargs
        )

        # Динамические состояния сомы
        self.v_mem: Optional[torch.Tensor] = None
        self.adapt: Optional[torch.Tensor] = None
        # Уровень NE прошлого такта (для детекции ФРОНТА выброса, патч S3)
        self._ne_prev: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, self.num_neurons)
        self.v_mem = torch.full(shape, self.v_rest, device=device, dtype=dtype)
        self.adapt = torch.zeros(shape, device=device, dtype=dtype)
        self._ne_prev = None
        self.transmitters.reset_state(batch_size, device, dtype)

    def detach_state(self):
        """O(1)-потребление памяти по времени для SLTT Фазы 1 (канон v4.0)."""
        if self.v_mem is not None:
            self.v_mem = self.v_mem.detach()
            self.adapt = self.adapt.detach()
            self.transmitters.detach_state()

    def forward(
        self,
        raw_basal: torch.Tensor,
        raw_apical: torch.Tensor,
        raw_gaba: torch.Tensor,
        ach: Optional[torch.Tensor] = None,
        ne: Optional[torch.Tensor] = None,
        serotonin_5ht: Optional[torch.Tensor] = None,
        sltt_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Один шаг симуляции dt.
        Возвращает: (spikes, v_mem, theta_current)
        """
        if (
            self.v_mem is None
            or self.v_mem.shape != raw_basal.shape
            or self.v_mem.device != raw_basal.device  # патч N7
        ):
            self.reset_state(raw_basal.shape[0], raw_basal.device, raw_basal.dtype)

        if sltt_mode:
            # Отсекаем граф предыдущего шага для O(1)-памяти
            self.detach_state()

        # 1. Дендритные токи: AMPA + NMDA-коинциденция + GABA (ур. 3-5, 4.1')
        i_soma, g_gaba = self.transmitters(raw_basal, raw_apical, raw_gaba, ach_level=ach)

        # 2. Норадреналин (NE): фазический сброс залипания — гасит накопленную
        #    усталость adapt (разд. 3.4: сброс адаптации при surprise).
        #    Патч S3 (канон v4.0.1): сброс привязан к ФРОНТУ выброса NE —
        #    росту уровня за такт, а не к самому уровню. Удерживаемый высокий
        #    NE не гасит адаптацию перманентно: иначе частые алармы снимают
        #    усталость в каждый такт и организм гипервозбуждается (шторм
        #    возбуждения, выявлено сборкой Этапа 2; CHANGELOG_v4.0.1.md, S3).
        #    Физиология: всплеск — СОБЫТИЕ, а не состояние.
        if ne is not None:
            ne_level = torch.clamp(ne.detach(), min=0.0, max=1.0)
            if self._ne_prev is None:
                rise = ne_level                       # первый выброс — фронт
            else:
                rise = torch.clamp(ne_level - self._ne_prev, min=0.0, max=1.0)
            self.adapt = self.adapt * (1.0 - rise)
            self._ne_prev = ne_level

        # 3. Стохастический шум мембраны, модулируемый серотонином (5-HT):
        #    высокий 5-HT — подавление шума (эксплуатация); низкий — рост
        #    энтропии до трёхкратного (исследование). Патч N1: тензоро-безопасно.
        noise_scale = self.sigma_noise
        if serotonin_5ht is not None:
            noise_scale = self.sigma_noise * torch.clamp(
                1.0 - serotonin_5ht, min=0.0, max=3.0
            )
        if self.stochastic:
            noise = torch.randn_like(self.v_mem) * noise_scale
        else:
            noise = torch.zeros_like(self.v_mem)

        # 4. Интегрирование мембранного потенциала сомы: ур. 6 канона
        #    (вычитаемая форма, легаси v4.0) ИЛИ ур. 6' с хлорным реверсом
        #    GABA (Этап 6, conductance-форма I_GABA = g·(V − E_Cl)):
        #    проводимость сворачивается в leak, реверс — в стационар.
        #    Exact integration => безусловная устойчивость при любом G.
        if g_gaba is not None:
            G = g_gaba
            alpha_eff = torch.exp(self.log_alpha_m * (1.0 + G))
            v_inf = (
                self.v_rest + G * self.transmitters.e_cl
                + self.r_m * i_soma
            ) / (1.0 + G)
            v_candidate = (
                alpha_eff * self.v_mem
                + (1.0 - alpha_eff) * v_inf
                + noise
            )
        else:
            v_candidate = (
                self.alpha_m * (self.v_mem - self.v_rest)
                + self.v_rest
                + (1.0 - self.alpha_m) * self.r_m * i_soma
                + noise
            )

        # 5. Адаптивный порог и генерация спайка (ур. 7-8)
        theta_t = self.theta_0 + self.beta_adapt * self.adapt
        v_shifted = v_candidate - theta_t
        spikes = spike_fn(
            v_shifted,
            mode=self.surrogate_mode,
            slope=self.surrogate_slope,
            alpha=self.surrogate_alpha,
        )

        # 6. Сброс потенциала вычитанием (reset-by-subtraction):
        #    сохраняет градиентную информацию о превышении порога
        self.v_mem = v_candidate - spikes * (theta_t - self.v_rest)

        # 7. Обновление переменной адаптации порога
        self.adapt = self.alpha_a * self.adapt + spikes.detach()

        return spikes, self.v_mem, theta_t
