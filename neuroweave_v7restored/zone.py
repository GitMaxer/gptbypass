# src/spiker/topology/zone.py
# Этап 1 (канон v4.0/v4.0.1, разд. 2.1): функциональная зона —
# замкнутый рекуррентный анклав.
#
# Три определяющих свойства канона:
#   1. Собственная локальная цель обучения (лосс задаётся снаружи —
#      зона есть субстрат, а не тренер);
#   2. Рекуррентность: состояние t зависит от истории самой зоны;
#   3. Инкапсуляция ресурсов: пул возбуждающих ALIF + пул GABA-интернейронов
#      + ансамбль SON (16 нейронов).
#
# Изоляция Фазы 1 (разд. 7.1): межзонные входы физически разорваны —
# апикальный вход замаскирован нулём (isolation_mask = 0). Включение шины —
# акт Фазы 2 (connect_apical), до того зона обязана научиться в одиночку.
#
# Каноническая семантика SON (разд. 6.1, ур. 15-16): экспорт SON уходит
# В ШИНУ — на апикальные дендриты ДРУГИХ зон (узел приёма декодирует
# сообщение своим W_dec). Самонаблюдение зоны = энкодер + AR-предиктор
# (Surprise), БЕЗ апикальной самопети: петля «спайки -> собственный
# апикальный ток -> NMDA-интегратор (tau=40) -> спайки» — положительная
# обратная связь, разгоняющая шторм при включённой шине (выявлено
# сборкой Этапа 2; см. CHANGELOG_v4.0.1.md, S2).
#
# Дендритная задержка 1 такт в рекуррентности и тормозном контуре —
# физиологична и развязывает алгебраический цикл.
#
# ЭТАП 6 — МАСШТАБИРОВАНИЕ (разд. 8.3 канона): все обучаемые матрицы
# ядра (syn_in/syn_rec/readout/SON) kaiming-нормированы по fan-in по
# построению — их синаптический драйв масштабно-инвариантен. ЕДИНСТВЕННОЕ
# исключение — фиксированные латеральные GABA-маски W_ei/W_ie с
# амплитудой scale: ожидаемая сумма строки растёт ЛИНЕЙНО с размером
# пула-источника (n_exc/n_inh) — без нормировки зона 512+ нейронов
# получает 8-30-кратное перекрытие торможения и коллапсирует в тишину.
# Закон масштабирования (решение Этапа 6): СОХРАНЕНИЕ СРЕДНЕГО ДРАЙВА ПРИ
# РОСТЕ — амплитуда нормируется на ожидаемую сумму строки референсного
# PoC-размера (64 exc / 16 inh) ТОЛЬКО при fan-in выше референсного:
# множитель min(1, ref/n) — зоны PoC-размеров и меньше сохраняют
# откалиброванную динамику побитово, апскейл получает E[Σ_j W_ij·s_j]
# референсного уровня (инвариант к масштабу).
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from ..core.neuron import MultiCompartmentALIF
from ..core.synapse import StatefulSynapse
from ..modulation.son import SONBus


# Референсные размеры PoC для закона сохранения драйва (Этап 6)
_SCALE_REF_EXC = 64.0   # fan-in W_ei = n_exc референса
_SCALE_REF_INH = 16.0   # fan-in W_ie = n_inh референса


# Дивергенция полушарий (табл. 2 канона)
HEMISPHERE_PRESETS = {
    "left": dict(
        tau_m=8.0, sigma_noise=0.001,
        surrogate_mode="fast_sigmoid", surrogate_slope=10.0,
        ht_baseline=0.8, lr_local=0.01,
    ),
    "right": dict(
        tau_m=35.0, sigma_noise=0.05,
        surrogate_mode="arctan", surrogate_alpha=2.0,
        ht_baseline=0.3, lr_local=0.002,
    ),
}


class Zone(nn.Module):
    """
    Изолированный рекуррентный анклав организма.

    Поток такта:
        s_input --syn_in--> базальные токи возб. пула
        s_exc(t-1) --syn_rec--> рекуррентные базальные токи
        s_inh(t-1) --W_ie--> латеральное GABA — проводимость с хлорным
            реверсом (ур. 9 + ур. 4.1' Этапа 6: I = g·(V − E_Cl),
            E_Cl = v_rest => чистый шунт, мембрана не ныряет под покой)
        s_apical (шина Callosum, ур. 16) => апикальные токи
            (замаскированы в Фазе 1; NMDA-гейт не пустит без базального)
        => MultiCompartmentALIF => s_exc(t)
        s_exc(t-1) --W_ei--> интернейроны => s_inh(t)
        s_exc(t) -> SON: окно, Surprise, экспорт В ШИНУ при новизне
        s_exc(t) -> интегратор чтения (SLTT, O(1)-память)
    """

    def __init__(
        self,
        name: str,
        side: str = "left",
        n_inputs: int = 32,
        n_exc: int = 64,
        n_inh: int = 16,
        n_readout: Optional[int] = None,
        tau_m: Optional[float] = None,
        sigma_noise: Optional[float] = None,
        surrogate_mode: Optional[str] = None,
        surrogate_slope: float = 10.0,
        surrogate_alpha: float = 2.0,
        n_son: int = 16,
        son_window: int = 8,
        readout_decay: float = 0.8,
        inhibitory_strength: float = 0.35,
        input_gain: float = 1.0,
        **neuron_kwargs,
    ):
        super().__init__()
        if side not in HEMISPHERE_PRESETS:
            raise ValueError(f"Unknown side: {side}")
        preset = HEMISPHERE_PRESETS[side]
        self.name = name
        self.side = side
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.ht_baseline = preset["ht_baseline"]
        self.lr_local = preset["lr_local"]
        # Гейн сенсорного входа: компенсация масштаба инициализации весов —
        # обеспечивает надпороговый базальный ток (детектор совпадений живой)
        self.input_gain = input_gain

        # ---- Пулы нейронов -------------------------------------------------
        exc_kwargs = dict(
            tau_m=tau_m if tau_m is not None else preset["tau_m"],
            sigma_noise=sigma_noise if sigma_noise is not None else preset["sigma_noise"],
            surrogate_mode=surrogate_mode if surrogate_mode is not None else preset["surrogate_mode"],
            surrogate_slope=surrogate_slope,
            surrogate_alpha=surrogate_alpha,
            **neuron_kwargs,
        )
        # Возбуждающий пул: полный компартментный ALIF
        self.exc = MultiCompartmentALIF(n_exc, **exc_kwargs)
        # Тормозной пул: короткая мембранная, без апикального входа,
        # вдвое меньший шум (интернейроны — быстрые и надёжные)
        inh_kwargs = dict(exc_kwargs)
        inh_kwargs["tau_m"] = 5.0
        inh_kwargs["sigma_noise"] = exc_kwargs["sigma_noise"] * 0.5
        self.inh = MultiCompartmentALIF(n_inh, **inh_kwargs)

        # ---- Синапсы -------------------------------------------------------
        self.syn_in = StatefulSynapse(n_inputs, n_exc)
        self.syn_rec = StatefulSynapse(n_exc, n_exc)

        # ---- Тормозной контур (ур. 9): exc -> inh -> exc -------------------
        # Фиксированные разреженные маски PoC; обучаемость W_ei/W_ie — Фаза 2
        # Ориентация по F.linear: W[out, in].
        # W_ei: s_exc [B, n_exc] -> basal inh [B, n_inh]  => [n_inh, n_exc]
        # W_ie: s_inh [B, n_inh] -> GABA exc  [B, n_exc]  => [n_exc, n_inh]
        # Этап 6: амплитуды нормированы законом сохранения драйва ПРИ РОСТЕ
        # (см. шапку модуля): E[сумма строки] не превосствует референс
        # density·ref·strength; на PoC-размерах — побитово прежние маски.
        ei_scale = inhibitory_strength * min(1.0, _SCALE_REF_EXC / n_exc)
        ie_scale = inhibitory_strength * min(1.0, _SCALE_REF_INH / n_inh)
        W_ei = self._sparse_mask(n_inh, n_exc, density=0.5, scale=ei_scale)
        W_ie = self._sparse_mask(n_exc, n_inh, density=0.5, scale=ie_scale)
        self.register_buffer("W_ei", W_ei)
        self.register_buffer("W_ie", W_ie)

        # ---- SON: шина само-наблюдения (16 нейронов, канон) -----------------
        self.son = SONBus(
            name, n_exc, n_son=n_son, window=son_window,
            surrogate_mode=exc_kwargs["surrogate_mode"],
            surrogate_slope=surrogate_slope, surrogate_alpha=surrogate_alpha,
        )

        # ---- Голова чтения локальной задачи (напр. следующий токен, ур. 2) --
        if n_readout is not None:
            self.readout = nn.Linear(n_exc, n_readout)
        else:
            self.readout = None

        # Интегратор чтения: экспоненциальное окно по спайкам.
        # Канон SLTT (разд. 7.1): история отсечена (detach), градиент течёт
        # только через спайки текущего такта — память O(1) по времени.
        self.readout_decay = readout_decay
        self.spike_accum: Optional[torch.Tensor] = None

        # ---- Изоляция Фазы 1 ------------------------------------------------
        # 0.0 = межзонные пути физически разорваны (апикальный вход занулён)
        self.register_buffer("isolation_mask", torch.zeros(()))

        # Отсроченные состояния (дендритная задержка 1 такт)
        self.s_exc_prev: Optional[torch.Tensor] = None
        self.s_inh_prev: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _sparse_mask(rows: int, cols: int, density: float, scale: float) -> torch.Tensor:
        """Разреженная маска с амплитудой scale (плотные тензоры + маски —
        стратегия Dense-Tensors канона, разд. 8.3)."""
        gen = torch.Generator().manual_seed(rows * 1000 + cols)
        m = (torch.rand(rows, cols, generator=gen) < density).float()
        return m * scale

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        self.exc.reset_state(batch_size, device, dtype)
        self.inh.reset_state(batch_size, device, dtype)
        self.syn_in.reset_state(batch_size, device, dtype)
        self.syn_rec.reset_state(batch_size, device, dtype)
        self.son.reset_state(batch_size, device, dtype)
        self.spike_accum = torch.zeros((batch_size, self.n_exc), device=device, dtype=dtype)
        self.s_exc_prev = torch.zeros((batch_size, self.n_exc), device=device, dtype=dtype)
        self.s_inh_prev = torch.zeros((batch_size, self.n_inh), device=device, dtype=dtype)

    def isolate(self):
        """Фаза 1: физически разорвать межзонные пути (разд. 7.1)."""
        self.isolation_mask.fill_(0.0)

    def connect_apical(self):
        """Фаза 2: активировать межзонные пути (вызывается сборкой организма)."""
        self.isolation_mask.fill_(1.0)

    # ------------------------------------------------------------------
    def step(
        self,
        s_input: torch.Tensor,
        s_apical: Optional[torch.Tensor] = None,
        ach: Optional[torch.Tensor] = None,
        ne: Optional[torch.Tensor] = None,
        serotonin_5ht: Optional[torch.Tensor] = None,
        on_son_alarm: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Один такт dt зоны.

        s_apical: [B, N_exc] — внешний апикальный ток (SON/Callosum другой
        зоны). В Фазе 1 физически занулён isolation_mask (канон разд. 7.1).
        on_son_alarm: колбэк шины SON (запуск NE при Surprise).
        """
        if (
            self.spike_accum is None
            or self.spike_accum.shape[0] != s_input.shape[0]
            or self.spike_accum.device != s_input.device
        ):
            self.reset_state(s_input.shape[0], s_input.device, s_input.dtype)

        batch = s_input.shape[0]
        zero_exc = torch.zeros_like(self.s_exc_prev)

        # Внешний апикальный вход: замаскирован в Фазе 1
        external_apical = s_apical if s_apical is not None else zero_exc
        external_apical = external_apical * self.isolation_mask

        # Апикальный путь (канон ур. 16): ТОЛЬКО внешние сообщения шины,
        # декодированные SON-узлом приёма (см. Hemisphere.callosum_apical).
        # В Фазе 1 физически занулён isolation_mask (разд. 7.1). Собственной
        # апикальной самопети у зоны НЕТ — экспорт SON уходит другим зонам.
        raw_apical = external_apical

        # Базальные токи: сенсорный вход (с гейном) + рекуррентность (задержка 1 такт)
        raw_basal = self.input_gain * self.syn_in(s_input) + self.syn_rec(self.s_exc_prev)

        # Латеральное торможение (ур. 9 + Этап 6): интернейроны прошлого
        # такта драйвят GABA-проводимость с хлорным реверсом — потребитель
        # (мембранное уравнение, ур. 6') сворачивает шунт в leak.
        raw_gaba = F.linear(self.s_inh_prev, self.W_ie)

        # Возбуждающий пул: полный компартментный проход
        s_exc, v_mem, theta = self.exc(
            raw_basal, raw_apical, raw_gaba,
            ach=ach, ne=ne, serotonin_5ht=serotonin_5ht,
        )

        # Интернейроны следуют за возбуждением (задержка 1 такт)
        raw_basal_inh = F.linear(self.s_exc_prev, self.W_ei)
        zero_inh = torch.zeros_like(self.s_inh_prev)
        s_inh, _, _ = self.inh(raw_basal_inh, zero_inh, zero_inh)

        # SON: наблюдаем свежие спайки зоны; экспорт уйдёт В ШИНУ другим
        # зонам (сжатое сообщение, ур. 15; Surprise-гейтинг — разд. 6.3)
        surprise, z_export, _z = self.son.step(s_exc, on_alarm=on_son_alarm)

        # Интегратор чтения (SLTT): история — константа, градиент — текущий такт
        self.spike_accum = self.spike_accum.detach() * self.readout_decay + s_exc

        # Обновление отложенных состояний
        self.s_exc_prev = s_exc.detach()
        self.s_inh_prev = s_inh.detach()

        return {
            "spikes": s_exc,
            "v_mem": v_mem,
            "theta": theta,
            "s_inh": s_inh,
            "surprise": surprise,
            "z_export": z_export,
        }

    def readout_logits(self) -> torch.Tensor:
        """Логиты локальной задачи по интегратору спайков (зона A, ур. 2)."""
        if self.readout is None:
            raise RuntimeError(f"Zone {self.name}: голова чтения не сконфигурирована")
        return self.readout(self.spike_accum)
