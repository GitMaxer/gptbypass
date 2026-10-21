# src/spiker/topology/callosum.py
# Этап 2 (канон v4.0/v4.0.1, разд. 2.3 + табл. 6): Corpus Callosum —
# асинхронная асимметричная шина сжатых сообщений между полушариями.
#
# Пять физиологических свойств канона (разд. 2.3):
#   1. ЗАДЕРЖКА: кольцевой буфер на 5-10 шагов разносит во времени источник
#      и приёмник — межполушарное взаимодействие принципиально асинхронно;
#   2. ТОРМОЖЕНИЕ: разреженный GABA-фильтр шины — слабые фоновые пересылки
#      подавляются, приоритетные пакеты проходят;
#   3. АСИММЕТРИЯ ВЕСОВ: проекции left->right и right->left — РАЗНЫЕ
#      обучаемые матрицы (различие ролей полушарий, табл. 2);
#   4. СЖАТЫЕ СООБЩЕНИЯ: шина несёт не данные, а латенты SON (ур. 15);
#      приёмный узел декодирует их на апикальные дендриты (ур. 16);
#   5. УЗКАЯ ПОЛОСА: пропускная способность ограничена снизу Surprise-
#      гейтингом SON (Zero-Compute) и сверху GABA-фильтром — намеренное
#      узкое место системы (разд. 2.3).
#
# Инженерия (разд. 8.3): плотные тензоры + маски разреженности (никаких
# разреженных форматов); GABA-фильтр — гладкий тензоро-безопасный гейт без
# ветвлений по данным (совместимость с CUDA Graphs, Этап 5); кольцевой буфер
# — предвыделенные тензоры на устройстве с ТЕНЗОРНЫМ указателем (index_select/
# index_copy_), пошагового обмена с хостом нет (кроме редкой телеметрии).
# Патч Этапа 5 (capture-ready): python-целочисленный указатель кольца
# заморожен бы графом CUDA — указатель стал тензором [1] и обновляется
# в графе (semantics побитово идентичны, tests/test_runtime.py).
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.quant import quantize_int8, dequantize, ternarize


class CorpusCallosum(nn.Module):
    """
    Мозолистое тело: буферизованная асимметричная шина латентов SON.

    Протокол такта:
        m_left [B, left_dim]  --(ring, delay)--> GABA-фильтр --> W_l2r --> to_right
        m_right [B, right_dim] --(ring, delay)--> GABA-фильтр --> W_r2l --> to_left

    Сообщения в шине живут в латентном пространстве полушария-ОТПРАВИТЕЛЯ;
    проекция W переводит их в латентное пространство получателя, где каждый
    чанк размером n_son раздаётся своей зоне и декодируется её SON-узлом
    приёма на апикальные дендриты (ур. 16; см. Hemisphere.callosum_apical).

    Пластичность (Фаза 2, Этап 3): W_l2r/W_r2l обучаются через 3-факторный
    Reward-STDP; структурные маски и коэффициенты асимметрии не обучаются.
    Шина НЕ участвует в SLTT-графе Фазы 1: сообщения отсоединяются (detach)
    при постановке в буфер — межзонных градиентов не существует по канону.
    """

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        delay: int = 7,             # канон: 5-10 шагов
        density: float = 0.3,       # разреженность проекций масками (разд. 8.3)
        theta_bus: float = 0.35,    # порог GABA-фильтра шины
        k_bus: float = 10.0,        # крутизна GABA-фильтра
        gain_l2r: float = 1.0,      # асимметрия: точные сообщения левого — полная
        gain_r2l: float = 0.6,      # асимметрия: контекст правого — ослабленный
        seed: int = 2026,
        int8_ring: bool = False,    # Этап 5: int8-хранение кольца задержек
        ternary_proj: bool = False, # Этап 5: тернарные эффективные проекции
        ternary_thr: float = 0.7,   # порог тернаризации (доля mean|w|)
    ):
        super().__init__()
        if left_dim <= 0 or right_dim <= 0:
            raise ValueError("Размерности шины должны быть положительными")
        if not (5 <= delay <= 10):
            raise ValueError(
                f"delay={delay} вне канонического диапазона 5-10 шагов (разд. 2.3)"
            )
        if not (0.0 < density <= 1.0):
            raise ValueError(f"density={density} должен быть в (0, 1]")

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.delay = int(delay)
        self.theta_bus = theta_bus
        self.k_bus = k_bus

        # Этап 5 (разд. 8.3): квантизация некритичного пути — шины.
        # int8_ring — ЧЕСТНОЕ int8-хранилище кольца (VRAM 4x); масштаб
        # per-message хранится рядом. ternary_proj — QAT-схема проекций:
        # мастер-вес полный (обучается Reward-STDP), эффективный — тернарный.
        self.int8_ring = bool(int8_ring)
        self.ternary_proj = bool(ternary_proj)
        self.ternary_thr = float(ternary_thr)

        gen = torch.Generator().manual_seed(seed)

        # Асимметричные обучаемые проекции (Фаза 2). Ориентация F.linear: W[out, in].
        # Инициализация — независимые выборки => матрицы различны и до обучения.
        self.W_l2r = nn.Parameter(self._init_proj(right_dim, left_dim, gen))
        self.W_r2l = nn.Parameter(self._init_proj(left_dim, right_dim, gen))

        # Структурная разреженность — масками поверх плотных тензоров (разд. 8.3)
        self.register_buffer("mask_l2r", self._mask(right_dim, left_dim, density, gen))
        self.register_buffer("mask_r2l", self._mask(left_dim, right_dim, density, gen))

        # Структурная асимметрия пропускания (не обучается): левое -> правое
        # несёт точные аналитические сообщения с полным весом; правое -> левое
        # — ассоциативный контекст с ослабленным влиянием (табл. 2).
        self.register_buffer("gain_l2r", torch.tensor(float(gain_l2r)))
        self.register_buffer("gain_r2l", torch.tensor(float(gain_r2l)))

        # Кольцевые буферы задержек: [delay, B, dim] на устройстве.
        # Указатель — ТЕНЗОР [1] long (capture-ready, Этап 5): продвигается
        # в вычислительном графе, без хост-ветвлений по данным.
        # int8_ring: хранилище — int8 ([delay, B, dim] байт вместо 4х),
        # масштаб сообщения — отдельный вектор [delay] (де/квантизация
        # на границе кольца — тензорные операции без ветвлений).
        self._buf_from_left: Optional[torch.Tensor] = None
        self._buf_from_right: Optional[torch.Tensor] = None
        self._scale_buf_left: Optional[torch.Tensor] = None
        self._scale_buf_right: Optional[torch.Tensor] = None
        self.register_buffer("_ptr_t", torch.zeros(1, dtype=torch.long))

        # Телеметрия (тензор, БЕЗ синхронизации с хостом; опрос — редкий)
        self._filtered_left: Optional[torch.Tensor] = None
        self._filtered_right: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------
    @staticmethod
    def _init_proj(out_dim: int, in_dim: int, gen: torch.Generator) -> torch.Tensor:
        """Kaiming-подобный масштаб U[-b, b], b = sqrt(1/fan_in) (a=sqrt(5))."""
        bound = 1.0 / math.sqrt(in_dim)
        return torch.empty(out_dim, in_dim).uniform_(-bound, bound, generator=gen)

    @staticmethod
    def _mask(rows: int, cols: int, density: float, gen: torch.Generator) -> torch.Tensor:
        return (torch.rand(rows, cols, generator=gen) < density).float()

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        """Предвыделить кольцевые буферы (ноль обмена с хостом в такте)."""
        ring_dtype = torch.int8 if self.int8_ring else dtype
        self._buf_from_left = torch.zeros(
            self.delay, batch_size, self.left_dim,
            device=device, dtype=ring_dtype)
        self._buf_from_right = torch.zeros(
            self.delay, batch_size, self.right_dim,
            device=device, dtype=ring_dtype)
        # Масштабы int8-сообщений кольца (float32, [delay])
        self._scale_buf_left = torch.zeros(
            self.delay, device=device, dtype=torch.float32)
        self._scale_buf_right = torch.zeros(
            self.delay, device=device, dtype=torch.float32)
        self._ptr_t.fill_(0)
        self._filtered_left = None
        self._filtered_right = None

    def _ensure_state(self, m_left: torch.Tensor, m_right: torch.Tensor):
        # dtype-проверка не применяется к int8-кольцу: хранилище НАМЕРЕННО
        # хранит int8 при float-сообщениях (масштаб живёт в _scale_buf) —
        # иначе состояние перезаводилось бы каждый такт (дефект Этапа 5-c,
        # вскрыт тестом доставки: буфер обнулялся, доставка была пустой)
        if (
            self._buf_from_left is None
            or self._buf_from_left.shape[1] != m_left.shape[0]
            or self._buf_from_left.device != m_left.device
            or (not self.int8_ring
                and self._buf_from_left.dtype != m_left.dtype)
        ):
            self.reset_state(m_left.shape[0], m_left.device, m_left.dtype)

    # ------------------------------------------------------------------
    # GABA-фильтр шины (разд. 2.3, свойство 2)
    # ------------------------------------------------------------------
    def _gaba_filter(self, m: torch.Tensor) -> torch.Tensor:
        """Разреженный GABA-контур: слабые фоновые пересылки подавляются,
        приоритетные пакеты проходят. Гладкий гейт амплитуды — тензоро-
        безопасный, без ветвлений по данным (CUDA-Graphs-совместимость)."""
        gate = torch.sigmoid(self.k_bus * (m.abs() - self.theta_bus))
        return m * gate

    def _project(self, m: torch.Tensor, direction: str) -> torch.Tensor:
        """Проекция сообщения в латентное пространство получателя.

        Этап 5 (ternary_proj): эффективная матрица — тернарная
        {−scale, 0, +scale} (мастер-вес остаётся полным для обучения
        Reward-STDP; тернарная пересобирается в такте — честная цена QAT,
        входит в профиль производительности)."""
        if direction == "l2r":
            w = self.W_l2r * self.mask_l2r
            if self.ternary_proj:
                w = ternarize(w, thr_frac=self.ternary_thr)
            return F.linear(m, w) * self.gain_l2r
        elif direction == "r2l":
            w = self.W_r2l * self.mask_r2l
            if self.ternary_proj:
                w = ternarize(w, thr_frac=self.ternary_thr)
            return F.linear(m, w) * self.gain_r2l
        raise ValueError(f"Unknown direction: {direction}")

    # ------------------------------------------------------------------
    # Такт шины
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        m_left: torch.Tensor,
        m_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Один такт шины.

        Параметры:
            m_left:  [B, left_dim]  — конкатенация SON-экспортов левого полушария
            m_right: [B, right_dim] — конкатенация SON-экспортов правого полушария
        Возвращает:
            to_right: [B, right_dim] — сообщение для правого полушария,
                      отправленное левым delay шагов назад (нули при прогреве);
            to_left:  [B, left_dim]  — сообщение для левого полушария.

        Сообщения отсоединяются от SLTT-графа: межзонных градиентов нет (канон
        разд. 7.1/7.2 — шина обучается локальным Reward-STDP в Фазе 2).
        """
        if m_left.shape[-1] != self.left_dim or m_right.shape[-1] != self.right_dim:
            raise ValueError(
                f"Размерности сообщений ({m_left.shape[-1]}, {m_right.shape[-1]}) "
                f"не соответствуют шине ({self.left_dim}, {self.right_dim})"
            )
        self._ensure_state(m_left, m_right)

        # 1. Снятие задержанных сообщений с кольцевого буфера (асинхронность);
        #    тензорная индексация — указатель живёт на устройстве (Этап 5).
        #    int8_ring: деквантизация на выходе кольца (масштаб сообщения
        #    хранится рядом с байтами)
        if self.int8_ring:
            raw_left = dequantize(
                self._buf_from_left.index_select(0, self._ptr_t)[0],
                self._scale_buf_left[self._ptr_t][0])
            raw_right = dequantize(
                self._buf_from_right.index_select(0, self._ptr_t)[0],
                self._scale_buf_right[self._ptr_t][0])
        else:
            raw_left = self._buf_from_left.index_select(0, self._ptr_t)[0]
            raw_right = self._buf_from_right.index_select(0, self._ptr_t)[0]

        # 2. Постановка текущих сообщений (detach: шина вне градиентного
        #    графа); int8_ring: квантизация на входе кольца
        if self.int8_ring:
            s_l = (m_left.detach().abs().max() / 127.0).clamp(min=1e-12)
            s_r = (m_right.detach().abs().max() / 127.0).clamp(min=1e-12)
            q_l = torch.round(m_left.detach() / s_l).clamp_(-127.0, 127.0)
            q_r = torch.round(m_right.detach() / s_r).clamp_(-127.0, 127.0)
            self._buf_from_left.index_copy_(
                0, self._ptr_t, q_l.to(torch.int8).unsqueeze(0))
            self._buf_from_right.index_copy_(
                0, self._ptr_t, q_r.to(torch.int8).unsqueeze(0))
            self._scale_buf_left.index_copy_(
                0, self._ptr_t, s_l.reshape(1))
            self._scale_buf_right.index_copy_(
                0, self._ptr_t, s_r.reshape(1))
        else:
            self._buf_from_left.index_copy_(
                0, self._ptr_t, m_left.detach().unsqueeze(0))
            self._buf_from_right.index_copy_(
                0, self._ptr_t, m_right.detach().unsqueeze(0))
        self._ptr_t.copy_((self._ptr_t + 1) % self.delay)

        # 3. GABA-фильтр транзита: слабые пересылки гаснут, пакеты проходят
        filtered_left = self._gaba_filter(raw_left)
        filtered_right = self._gaba_filter(raw_right)
        self._filtered_left = filtered_left
        self._filtered_right = filtered_right

        # 4. Асимметричные проекции в латентные пространства получателей
        to_right = self._project(filtered_left, "l2r")
        to_left = self._project(filtered_right, "r2l")
        return to_right, to_left

    # ------------------------------------------------------------------
    # Телеметрия (вызывать редко — медленный контур, разд. 8)
    # ------------------------------------------------------------------
    @property
    def traffic(self) -> float:
        """Средняя амплитуда сообщений, прошедших GABA-фильтр, за такт."""
        if self._filtered_left is None or self._filtered_right is None:
            return 0.0
        both = torch.stack([
            self._filtered_left.abs().mean(),
            self._filtered_right.abs().mean(),
        ])
        return float(both.mean())
