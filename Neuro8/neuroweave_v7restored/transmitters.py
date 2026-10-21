# src/spiker/core/transmitters.py
# Канон v4.0/v4.0.1 + Этап 6 (GABA-реверс). Патчи: T1 (клэмп ACh в [-1, 1]),
# T2 (mg_zero_baseline — буфер, без per-step аллокации),
# T5 (guard сброса проверяет устройство).
#
# ЭТАП 6 — ХЛОРНЫЙ РЕВЕРС GABA (разд. 4.1 канона, открытый вопрос Этапа 1):
# торможение — не вычитаемый ток, а ПРОВОДИМОСТЬ с потенциалом реверса:
#     I_GABA = g_GABA · (V − E_Cl)                        (ур. 4.1')
# Форма с реверсом решает проблему протокола сна: вычитаемый ток
# (v4.0) при глубоком торможении уводит мембрану СКЛЬЗЯЩЕ ниже V_rest
# (стационар V_rest − r_m·I_GABA не ограничен снизу), восстановление
# после снятия торможения занимает ~tau_m — реплей «умирает» на длинные
# окна. Проводимость с E_Cl = V_rest — ЧИСТЫЙ ШУНТ: гиперполяризации нет
# вовсе, вход возбуждения делится на (1+G); E_Cl < V_rest — ограниченная
# гиперполяризация с полом НА E_Cl (физиология хлора). Математика
# интегрирования — exact-integration свёртка в leak (см. neuron.py, ур. 6').
import torch
import torch.nn as nn
from typing import Optional


class IonotropicTransmitters(nn.Module):
    """
    Вычисляет ионотропные токи (AMPA, NMDA, GABA) и их фильтрацию во времени.
    Реализует физиологический детектор совпадений через снятие Mg2+-блока (ур. 4)
    и ацетилхолиновую модуляцию проводимости сомы (ур. 5, канон v4.0).

    Этап 6 — GABA-реверс (gaba_mode="conductance", дефолт):
    след i_gaba интерпретируется как безразмерная проводимость G = gaba_gain·i_gaba
    с потенциалом реверса e_cl; из тока сомы GABA ИСКЛЮЧАЕТСЯ — мембранное
    уравнение потребляет (i_soma, g_gaba) пару и сворачивает шунт в leak
    (exact integration, ур. 6' в neuron.py). Легаси-форма gaba_mode="subtractive"
    сохраняет арифметику v4.0 (− i_gaba в токе) для A/B-калибровки.

    ГАРАНТИЯ РЕВЕРСА: при e_cl >= v_rest торможение ЛЮБОЙ глубины не может
    увести мембрану ниже v_rest (стационар чистого торможения — выпуклая
    комбинация v_rest и e_cl); при e_cl < v_rest пол гиперполяризации — e_cl.

    Все состояния — GPU-тензоры в общем вычислительном графе (разд. 8):
    пошаговый счёт без возврата на хост.
    """

    GABA_MODES = ("subtractive", "conductance")

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,
        tau_ampa: float = 4.0,
        tau_nmda: float = 40.0,
        tau_gaba: float = 6.0,
        theta_mg: float = 0.25,
        k_mg: float = 12.0,
        alpha_nmda: float = 1.5,
        gamma_ach: float = 0.5,
        gaba_mode: str = "conductance",  # Этап 6: хлорный реверс — дефолт
        e_cl: float = 0.0,               # E_Cl; дефолт = v_rest ядра (чистый шунт)
        gaba_gain: float = 1.0,          # масштаб проводимости G = gaba_gain·i_gaba
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.theta_mg = theta_mg
        self.k_mg = k_mg
        self.alpha_nmda = alpha_nmda
        self.gamma_ach = gamma_ach

        # Этап 6: хлорный реверс GABA. e_cl в безразмерных единицах ядра
        # (v_rest = 0.0): e_cl >= v_rest — чистый шунт без гиперполяризации;
        # e_cl < v_rest — ограниченная гиперполяризация с полом на e_cl.
        if gaba_mode not in self.GABA_MODES:
            raise ValueError(
                f"gaba_mode={gaba_mode!r}: ожидается одно из {self.GABA_MODES}")
        self.gaba_mode = gaba_mode
        self.e_cl = float(e_cl)
        self.gaba_gain = float(gaba_gain)

        # Коэффициенты экспоненциального затухания проводимостей (Exact integration)
        self.register_buffer("decay_ampa", torch.exp(torch.tensor(-dt / tau_ampa)))
        self.register_buffer("decay_nmda", torch.exp(torch.tensor(-dt / tau_nmda)))
        self.register_buffer("decay_gaba", torch.exp(torch.tensor(-dt / tau_gaba)))

        # Патч T2: базовая линия магниевой сигмоиды — константа, а не per-step
        # аллокация. Гарантирует нулевой гейт при i_ampa == 0 (калибровка канона).
        self.register_buffer(
            "mg_zero_baseline", torch.sigmoid(torch.tensor(-k_mg * theta_mg))
        )

        # Состояния синаптических токов
        self.i_ampa: Optional[torch.Tensor] = None
        self.i_nmda_raw: Optional[torch.Tensor] = None
        self.i_gaba: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, self.num_neurons)
        self.i_ampa = torch.zeros(shape, device=device, dtype=dtype)
        self.i_nmda_raw = torch.zeros(shape, device=device, dtype=dtype)
        self.i_gaba = torch.zeros(shape, device=device, dtype=dtype)

    def detach_state(self):
        """Для O(1)-памяти SLTT в Фазе 1: отсекаем историю градиентов."""
        if self.i_ampa is not None:
            self.i_ampa = self.i_ampa.detach()
            self.i_nmda_raw = self.i_nmda_raw.detach()
            self.i_gaba = self.i_gaba.detach()

    def forward(
        self,
        raw_basal: torch.Tensor,
        raw_apical: torch.Tensor,
        raw_gaba: torch.Tensor,
        ach_level: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Параметры:
            raw_basal:  [Batch, N] — мгновенный входной ток W_AMPA * s_pre(t)
            raw_apical: [Batch, N] — мгновенный контекстный ток W_apical * s_context(t)
            raw_gaba:   [Batch, N] — мгновенный драйвер проводимости W_GABA * s_inh(t)
            ach_level:  [Batch, N] или [Batch, 1], диапазон [-1, 1] — уровень ACh
        Возвращает (Этап 6 — пара для мембранного уравнения):
            i_soma:  [Batch, N] — ток возбуждения в сому (AMPA + NMDA, ур. 5);
                      в conductance-режиме БЕЗ GABA-члена (шунт сворачивается
                      в leak мембраны, ур. 6');
            g_gaba:  Optional[torch.Tensor] — безразмерная проводимость
                      G = gaba_gain·i_gaba [Batch, N] (conductance-режим);
                      None в subtractive-режиме (GABA уже вычтен из i_soma).
        """
        if (
            self.i_ampa is None
            or self.i_ampa.shape != raw_basal.shape
            or self.i_ampa.device != raw_basal.device  # патч T5
        ):
            self.reset_state(raw_basal.shape[0], raw_basal.device, raw_basal.dtype)

        # 1. Кинетика синаптических фильтров
        self.i_ampa = self.i_ampa * self.decay_ampa + raw_basal
        self.i_nmda_raw = self.i_nmda_raw * self.decay_nmda + raw_apical
        self.i_gaba = self.i_gaba * self.decay_gaba + raw_gaba

        # 2. NMDA-гейтинг — снятие Mg2+-блока только при базальной
        #    деполяризации (ур. 4). Калиброванная сигмоида со сдвигом:
        #    при i_ampa == 0 гейт строго 0 (инвариант апикальной тишины).
        mg_unblock = torch.sigmoid(self.k_mg * (self.i_ampa - self.theta_mg))
        mg_gate = torch.clamp(
            (mg_unblock - self.mg_zero_baseline) / (1.0 - self.mg_zero_baseline),
            min=0.0,
            max=1.0,
        )
        i_nmda_effective = mg_gate * torch.relu(self.i_nmda_raw)

        # 3. Ацетилхолиновая модуляция проводимости (ур. 5, канон v4.0):
        #    высокий ACh — бдительность (усилен Bottom-Up, подавлен Top-Down);
        #    низкий ACh — эндогенные ассоциации и реплей.
        if ach_level is not None:
            ach = torch.clamp(ach_level, -1.0, 1.0)  # патч T1: диапазон канона
            basal_gain = 1.0 + self.gamma_ach * ach
            apical_gain = 1.0 - self.gamma_ach * ach
        else:
            basal_gain = 1.0
            apical_gain = 1.0

        # 4. Итог: пара (ток возбуждения, проводимость шунта) — ур. 5/6'
        #    (Этап 6). Conductance: GABA покидает ток сомы — мембрана получит
        #    его как проводимость с реверсом (neuron.py). Subtractive:legacy
        #    арифметика v4.0 — ток уже содержит − i_gaba, проводимость None.
        i_exc = (
            basal_gain * self.i_ampa
            + self.alpha_nmda * apical_gain * i_nmda_effective
        )
        if self.gaba_mode == "conductance":
            return i_exc, self.gaba_gain * self.i_gaba
        return i_exc - self.i_gaba, None
