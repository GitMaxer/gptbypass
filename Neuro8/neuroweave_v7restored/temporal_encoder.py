# src/spiker/encoding/temporal_encoder.py
# Этап 5 (канон v4.0/v4.0.1, табл. 6): сенсорные интерфейсы организма.
#
# Два кодировщика:
#   * DelayLineTokenEncoder — МИНИМАЛЬНЫЙ интерфейс Этапа 1 (историческая
#     заглушка, закрывшая разрыв зависимости дорожной карты): линия задержки,
#     токен -> one-hot слоты «сейчас / k тактов назад». Оставлен как
#     нативный сенсорный формат зон, обученных в Этапах 1-4.
#   * TemporalPhaseEncoder — ПОЛНЫЙ временной фазовый кодировщик Этапа 5:
#     burst-coding и latency-coding (табл. 6: «временной фазовый кодировщик
#     токенов (burst-coding и latency-coding)»).
#
# Физиология кода (разд. 8.3 канона: спайки/задержки — фундамент графа):
#   LATENCY:  интенсивность признака s∈[0,1] кодируется МОМЕНТОМ спайка —
#             сильный (салиентный) признак стреляет РАНЬШЕ:
#             d = ⌊(1-s)·(W-1) + 1/2⌋ тактов после прихода стимула
#             (классический latency-код Thorpe: «кто первым — тот сильней»);
#   BURST:    интенсивность кодируется ДЛИНОЙ залпа:
#             L = 1 + ⌊s·(B_max-1) + 1/2⌋ тактов подряд.
# Гибрид burst_latency: салиентный токен — ранний И длинный залп; фоновый —
# поздний одиночный спайк. Информация живёт ВРЕМЕНЕМ, а не амплитудой:
# выход бинарный на каждом такте (спайки не имеют величины, канон).
#
# Канальный базис — ФАЗОВЫЕ ДЕТЕКТОРЫ (линия задержки, модель Джеффресса):
# канал (v, k) — периферийный нейрон, срабатывающий когда токен v находится
# в фазе k своего кодового окна. Выход такта — бинарный фазовый образ
# [B, vocab·span]: текущий токен виден в ранних слотах (салиентный — сразу
# несколькими), предыдущие токены — в поздних: кодовое окно ОДНОВРЕМЕННО
# каузальная история (зона читает (x_t, x_{t-1}) для ур. 2 без внешних
# механизмов памяти).
#
# Салиентность (что считать «интенсивностью» токена) — решение ВЫШЕ
# кодировщика: явный аргумент step(salience=...). В эксперименте инверсии
# правил салиентность = сюрприз замороженного эксперта (top-down усиление
# сенсорного входа — физиологичный механизм внимания, ур. 5).
#
# Инженерия (разд. 8.3): только тензорные операции, никаких ветвлений по
# данным — пайплайн тензоро-безопасен и CUDA-Graphs-совместим (Этап 5-b).
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DelayLineTokenEncoder(nn.Module):
    """Линия задержки токенов: поток id -> бинарные спайковые тензоры."""

    def __init__(self, vocab_size: int, n_channels: int = 2):
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels >= 1")
        self.vocab_size = vocab_size
        self.n_channels = n_channels
        self.n_out = vocab_size * n_channels
        # Регистр сдвига: последние n_channels-1 one-hot векторов
        self.shift: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        width = self.vocab_size * max(0, self.n_channels - 1)
        self.shift = torch.zeros((batch_size, width), device=device, dtype=dtype)

    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B] int64 -> спайки [B, vocab * n_channels] (бинарные).
        Канал 0 — текущий токен, канал k — токен k тактов назад.
        """
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).to(
            dtype=self.shift.dtype if self.shift is not None else torch.float32
        )
        if self.shift is None or self.shift.shape[0] != token_ids.shape[0] \
                or self.shift.device != one_hot.device:
            self.reset_state(token_ids.shape[0], one_hot.device, one_hot.dtype)
        if self.n_channels == 1:
            return one_hot
        s = torch.cat([one_hot, self.shift], dim=1)
        # сдвиг: регистр хранит историю без графа (чистая память)
        self.shift = s[:, : self.shift.shape[1]].detach()
        return s


class TemporalPhaseEncoder(nn.Module):
    """
    Полный временной фазовый кодировщик Этапа 5 (burst + latency).

    Поток токенов -> бинарный фазовый образ на span слотах задержки:
        канал (v, k) = 1 <=> токен v находится в фазе k СВОЕГО кодового
                        окна (пришёл k - d тактов назад, d <= k < d + L,
                        где d — латентность, L — длина залпа).
    Салиентный токен занимает РАННИЕ слоты и стреляет залпом; фоновый —
    поздний одиночный. Текущий + предыдущие токены видны одновременно
    в разных слотах: кодовое окно = каузальная история.

    Выход: [B, vocab * span] — бинарный (спайки, не амплитуды).

    Параметры:
        vocab_size:    размер словаря;
        coding_window: W — окно латентности (слот спайка ∈ [0, W-1]);
        max_burst:     B_max — максимальная длина залпа;
        mode:          "burst" | "latency" | "burst_latency" (гибрид);
        salience_default: s при явной салиентности нет. Дефолт 1.0 —
                       немедленный максимальный залп (калибровка зонда
                       probe_stage5_encoder_calib.py: W4/B3/s=1.0 даёт
                       обучаемость НЕ ХУЖЕ линии задержки; информативная
                       салиентность подаётся явно — см. эксперимент
                       инверсии правил);
        online_salience: оценивать s онлайн по частоте токенов (новизна
                       периферии: редкий токен — салиентный).
    """

    def __init__(
        self,
        vocab_size: int,
        coding_window: int = 4,
        max_burst: int = 3,
        mode: str = "burst_latency",
        salience_default: float = 1.0,
        online_salience: bool = False,
        salience_lr: float = 0.05,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size >= 1")
        if coding_window < 1:
            raise ValueError("coding_window >= 1")
        if max_burst < 1:
            raise ValueError("max_burst >= 1")
        if mode not in ("burst", "latency", "burst_latency"):
            raise ValueError(f"Неизвестный режим кодирования: {mode!r}")
        self.vocab_size = vocab_size
        self.coding_window = coding_window
        self.max_burst = max_burst
        self.mode = mode
        self.salience_default = float(salience_default)
        self.online_salience = bool(online_salience)
        self.salience_lr = float(salience_lr)

        # Полное кодовое окно: максимум задержки + максимум залпа
        self.span = (coding_window - 1) + max_burst
        self.n_out = vocab_size * self.span

        # Возрастные буферы кода (модель Джеффресса, [B, V, span]):
        #   _valid [B, V, span] — «токен v пришёл a тактов назад»;
        #   _d_store/_l_store    — его латентность d и длина залпа L.
        # Слот a = ВОЗРАСТ прибывшего токена; каждый такт буферы стареют
        # (сдвиг вправо), выход = валидность AND (a в [d, d+L)).
        self._valid: Optional[torch.Tensor] = None
        self._d_store: Optional[torch.Tensor] = None
        self._l_store: Optional[torch.Tensor] = None
        # Онлайн-оценка частот (салиентность новизны периферии)
        self._freq: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        shape = (batch_size, self.vocab_size, self.span)
        self._valid = torch.zeros(shape, device=device, dtype=dtype)
        self._d_store = torch.zeros(shape, device=device, dtype=dtype)
        self._l_store = torch.zeros(shape, device=device, dtype=dtype)
        self._freq = torch.full(
            (self.vocab_size,), 1.0 / self.vocab_size,
            device=device, dtype=dtype)

    def _ensure_state(self, batch: int, device: torch.device,
                      dtype: torch.dtype):
        if (self._valid is None
                or self._valid.shape[0] != batch
                or self._valid.device != device
                or self._valid.dtype != dtype):
            self.reset_state(batch, device, dtype)

    # ------------------------------------------------------------------
    def _burst_params(self, s: torch.Tensor):
        """Салиентность -> (d, L) — параметры залпа (чисто тензорно).

            d = ⌊(1-s)·(W-1) + 1/2⌋ — слот ПЕРВОГО спайка (latency);
            L = 1 + ⌊s·(B_max-1) + 1/2⌋ — длина залпа (burst).
        Режимы: "burst" фиксирует d=0 (информация только в длине),
        "latency" фиксирует L=1 (информация только в моменте), гибрид — оба.
        """
        if self.mode == "burst":
            d = torch.zeros_like(s)
        else:
            d = torch.floor((1.0 - s) * (self.coding_window - 1) + 0.5)
        if self.mode == "latency":
            length = torch.ones_like(s)
        else:
            length = 1.0 + torch.floor(s * (self.max_burst - 1) + 0.5)
        return d, length

    def _estimate_salience(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Онлайн-новизна периферии: редкий токен — салиентный.

        Частоты — EMA по батчу; s = 1 - freq/freq_max (нормировка по
        самому частому). На стационарно-равномерном потоке s ~ 0 —
        салиентность информативна только при НЕравномерной статистике.
        """
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).to(
            self._freq.dtype)
        batch_freq = one_hot.mean(dim=0)                    # [V]
        self._freq.mul_(1.0 - self.salience_lr).add_(
            self.salience_lr * batch_freq)
        freq_max = self._freq.max().clamp(min=1e-6)
        rel = (self._freq / freq_max)[token_ids]             # [B]
        return (1.0 - rel).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def step(
        self,
        token_ids: torch.Tensor,
        salience: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Такт кодировщика.

        token_ids: [B] int64 — токены, пришедшие СЕЙЧАС;
        salience:  [B] float в [0,1] — их интенсивность (решение верхнего
                   уровня; None -> дефолт или онлайн-оценка новизны).
        Возвращает: [B, vocab*span] — бинарный фазовый образ этого такта.
        """
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids должны быть [B], Got {token_ids.shape}")
        batch = int(token_ids.shape[0])
        device, dtype = token_ids.device, torch.float32
        self._ensure_state(batch, device, dtype)

        # 1. Салиентность потока
        if salience is None:
            if self.online_salience:
                s = self._estimate_salience(token_ids)
            else:
                s = torch.full((batch,), self.salience_default,
                               device=device, dtype=dtype)
        else:
            s = salience.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if s.shape != token_ids.shape:
                raise ValueError(
                    f"salience {tuple(s.shape)} не совпадает с токенами "
                    f"{tuple(token_ids.shape)}")

        # 2. Параметры залпа прибывшего токена и его канал
        d, length = self._burst_params(s)                  # [B], [B]
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).to(dtype)
        arr = one_hot.unsqueeze(-1)                         # [B, V, 1]

        # 3. Возрастные буферы: старение на такт (сдвиг вправо) + прибытие
        #    в слот возраста 0 своего канала (d/L прибытия пишутся и стареют
        #    ВМЕСТЕ с токеном — фаза не размазывается между токенами)
        valid = torch.cat([arr, self._valid[:, :, :-1]], dim=2)
        d_store = torch.cat([(d.unsqueeze(-1) * one_hot).unsqueeze(-1),
                             self._d_store[:, :, :-1]], dim=2)
        l_store = torch.cat([(length.unsqueeze(-1) * one_hot).unsqueeze(-1),
                             self._l_store[:, :, :-1]], dim=2)

        # 4. Фазовый образ такта: канал (v, a) активен <=> токен v возраста a
        #    в своём залпе (a в [d, d+L)) — детектор фазы сработал
        slots = torch.arange(self.span, device=device, dtype=dtype)
        in_burst = (slots >= d_store) & (slots < d_store + l_store)
        out = (in_burst & (valid > 0)).to(dtype)

        # 5. Сохранить возраслевые буферы (чистая память, без графа)
        self._valid = valid.detach()
        self._d_store = d_store.detach()
        self._l_store = l_store.detach()
        return out.reshape(batch, self.n_out)

    # ------------------------------------------------------------------
    def step_features(
        self,
        features: torch.Tensor,
        salience: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Аналоговый сенсорный вход: признаки [B, F] -> те же поезда.

        Непрерывный вариант интерфейса: каждый признак f с величиной x_f
        (нормируется в [0,1] по максимуму батча) кодируется латентностью
        и залпом на СВОЁМ канальном блоке из span слотов. Величина -> ВРЕМЯ:
        фундамент канона «спайки — не амплитуды».
        """
        if features.dim() != 2:
            raise ValueError(f"features должны быть [B, F], Got {features.shape}")
        if features.shape[1] != self.vocab_size:
            raise ValueError(
                f"Число признаков {features.shape[1]} != vocab_size "
                f"{self.vocab_size} (кодировщик сконфигурирован под словарь)")
        batch = features.shape[0]
        device = features.device
        dtype = features.float().dtype
        self._ensure_state(batch, device, dtype)

        x = features.to(dtype)
        if salience is None:
            # Величина признака -> салиентность НАПРЯМУЮ (признаки ожидаются
            # в [0,1]; клэмп страхует): сильный признак — ранний и залповый,
            # величина -> ВРЕМЯ (нормировка по батчу стирала бы абсолютную
            # величину: сильнейший всегда становился бы s=1)
            s = x.abs().clamp(0.0, 1.0)
        else:
            s = salience.to(dtype).clamp(0.0, 1.0)
            if s.shape != (batch, features.shape[1]):
                raise ValueError("salience должна быть [B, F] для step_features")

        # Аналоговое прибытие: каждый признак > 0 приходит КАЖДЫЙ такт —
        # залп признака перезаписывает слот возраста 0 своего канала,
        # залпы прошлых тактов стареют в поздних слотах (перекрытие честно)
        d, length = self._burst_params(s)                  # [B, F], [B, F]
        arr = (x.abs() > 0).to(dtype).unsqueeze(-1)        # [B, F, 1]

        valid = torch.cat([arr, self._valid[:, :, :-1]], dim=2)
        d_store = torch.cat([d.unsqueeze(-1), self._d_store[:, :, :-1]], dim=2)
        l_store = torch.cat([length.unsqueeze(-1), self._l_store[:, :, :-1]], dim=2)

        slots = torch.arange(self.span, device=device, dtype=dtype)
        in_burst = (slots >= d_store) & (slots < d_store + l_store)
        out = (in_burst & (valid > 0)).to(dtype)

        self._valid = valid.detach()
        self._d_store = d_store.detach()
        self._l_store = l_store.detach()
        return out.reshape(batch, self.n_out)
