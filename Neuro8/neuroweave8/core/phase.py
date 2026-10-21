"""Фазовое кодирование: тета-гамма связывание и фазовая VSA.

ЭТО НОВОЕ ЯДРО v8, которого не было ни в v6, ни в v7. Канон v4.0 умел
только rate-кодирование: «сколько нейронов выстрелило». Такая репрезентация
не различает «собака кусает человека» и «человек кусает собаку» — обе
ситуации дают одинаковый вектор активности. Связывание признаков с ролями
(role-filler binding) требует ВТОРОГО измерения кода — времени/фазы.

--------------------------------------------------------------------------------
1) ThetaGammaBinder — рабочая память на тета-гамма цикле (Lisman & Idiart)

    Тета-цикл (~6-8 Гц) делится на гамма-подциклы (~40 Гц). Каждый подцикл —
    один «слот» рабочей памяти: внутри него популяция выстреливает паттерн
    filler'а, а номер слота кодирует роль. Ёмкость тогда не статистическая,
    а АРХИТЕКТУРНАЯ:

        K = f_theta / f_gamma  ~  6..8 / 40  =>  5..7 слотов

    Ровно «7 +/- 2» Миллера, но с физическим механизмом: чтобы удержать
    восьмой элемент, нужен либо более длинный тета-цикл, либо более быстрый
    гамма. Переполнение даёт АЛИАСИНГ (два элемента в одном слоте), а не
    плавную деградацию — это проверяемый контракт.

2) PhaseVSA — фазовые вектор-символьные архитектуры (Plate, Kanerva)

    Символ = случайный вектор единичного модуля размерности d:
        x_j = exp(i * phi_j),  phi_j ~ U[0, 2pi)
    Связывание  (bind):   (x *_R y)_j = x_j y_j        (сложение фаз)
    Суперпозиция(bundle): b = sum_k (x_k *_R y_k)        (интерференция мала)
    Разыменование(unbind): y' = b *_R conj(x) = y + шум
    Очистка    (cleanup): argmax_s Re<b *_R conj(x), code_s>

    Математически это HRR/tensor-product representation с циркулярной
    свёрткой, реализованной в фазовом (комплексном) базисе: ассоциативность,
    дистрибутивность и обратимость выполняются ТОЧНО, а не приближённо.
    Ёмкость ~ d / (2 ln(1/eps)); при d=512 это десятки пар.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import PhaseCfg


def capacity_from_frequencies(theta_hz: float, gamma_hz: float) -> int:
    """Архитектурная ёмкость рабочей памяти: число гамма-циклов в тета-цикле."""
    if theta_hz <= 0 or gamma_hz <= 0:
        raise ValueError("частоты должны быть положительны")
    return max(1, int(round(gamma_hz / theta_hz)))


# ---------------------------------------------------------------------------
class ThetaGammaBinder(nn.Module):
    """Слотовая рабочая память: роль = номер гамма-подцикла."""

    def __init__(self, n_units: int, cfg: PhaseCfg):
        super().__init__()
        self.n_units = int(n_units)
        self.n_slots = int(cfg.n_slots)
        self.gamma_ticks = int(cfg.gamma_ticks)
        self.buf: Optional[torch.Tensor] = None      # [B, K, N]
        self.slot = 0
        self.theta_phase = 0.0
        self.alias_events = 0

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self.buf = torch.zeros(batch, self.n_slots, self.n_units,
                               device=device, dtype=dtype)
        self.slot = 0
        self.theta_phase = 0.0
        self.alias_events = 0

    def _ensure(self, ref: torch.Tensor) -> None:
        if (self.buf is None or self.buf.shape[0] != ref.shape[0]
                or self.buf.device != ref.device):
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    # ------------------------------------------------------------------
    def write(self, spikes: torch.Tensor, role: Optional[int] = None) -> None:
        """Записать паттерн в слот (роль). Переполнение = алиасинг."""
        self._ensure(spikes)
        k = self.slot if role is None else int(role) % self.n_slots
        if role is None and self.buf[:, k, :].abs().sum() > 0:
            self.alias_events += 1
        self.buf[:, k, :].add_(spikes.detach())
        if role is None:
            self.advance()

    def advance(self) -> None:
        self.slot = (self.slot + 1) % self.n_slots
        self.theta_phase = (self.theta_phase + 2 * math.pi / self.n_slots) % (
            2 * math.pi)

    def read(self, role: int) -> torch.Tensor:
        """Прочитать filler по роли (номеру слота)."""
        return self.buf[:, int(role) % self.n_slots, :]

    def read_soft(self, role: int, jitter: float = 0.35) -> torch.Tensor:
        """Чтение с фазовым джиттером: соседние слоты подмешиваются.

        Физиология: оценка фазы спайк-фазовая привязка неточна, поэтому
        чтение — не жёсткий селектор, а гауссово ядро по номеру слота.
        """
        k = int(role) % self.n_slots
        w = torch.zeros(self.n_slots, device=self.buf.device, dtype=self.buf.dtype)
        for i in range(self.n_slots):
            d = min(abs(i - k), self.n_slots - abs(i - k))
            w[i] = math.exp(-(d * d) / (2 * jitter * jitter))
        w = w / w.sum()
        return torch.einsum("k,bkn->bn", w, self.buf)

    def clear(self, role: int) -> None:
        self.buf[:, int(role) % self.n_slots, :] = 0.0

    def occupancy(self) -> float:
        """Доля заполненных слотов (телеметрия рабочей памяти)."""
        used = (self.buf.abs().sum(-1) > 0).float().mean()
        return float(used)

    def readout(self) -> torch.Tensor:
        """Развёрнутая репрезентация всех слотов: [B, K*N]."""
        b = self.buf.shape[0]
        return self.buf.reshape(b, -1)


# ---------------------------------------------------------------------------
class PhaseVSA(nn.Module):
    """Фазовые вектор-символьные операции для структурного связывания."""

    def __init__(self, cfg: PhaseCfg, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.dim = int(cfg.vsa_dim)
        self.n_sym = int(cfg.vsa_codebook)
        self.sharpen = float(cfg.cleanup_sharpen)
        gen = generator or torch.Generator().manual_seed(20260808)

        def unit_phase(rows: int) -> torch.Tensor:
            ang = torch.rand(rows, self.dim, generator=gen) * (2 * math.pi)
            return torch.polar(torch.ones(rows, self.dim), ang)

        self.register_buffer("codebook", unit_phase(self.n_sym))
        self.register_buffer("roles", unit_phase(32))

    # ---- базовые операции -------------------------------------------
    def symbol(self, idx: torch.Tensor) -> torch.Tensor:
        return self.codebook[idx]

    def role(self, idx: int) -> torch.Tensor:
        return self.roles[int(idx)]

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Связывание: поэлементное произведение = сложение фаз."""
        return a * b

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Развязывание: умножение на сопряжённый ключ (обратимо точно)."""
        return bound * torch.conj(key)

    def bundle(self, items: Sequence[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        out = torch.stack(list(items), dim=0).sum(0)
        if normalize:
            out = out / math.sqrt(max(1, len(items)))
        return out

    # ---- очистка памяти ---------------------------------------------
    def similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Косинус в комплексном пространстве: Re<x, c>/d."""
        return torch.einsum("...d,sd->...s", x, torch.conj(self.codebook)).real / self.dim

    def cleanup(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.similarity(x)
        if self.sharpen != 1.0:
            sim = sim * self.sharpen
        return sim.argmax(-1), sim.max(-1).values

    def cleanup_acc(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Мягкая очистка (дифференцируемая): взвешенная сумма кодов."""
        prob = torch.softmax(self.similarity(x) * self.sharpen, dim=-1)
        return prob @ self.codebook

    # ---- кодирование спайковых паттернов ----------------------------
    def encode_pattern(self, spikes: torch.Tensor, role_idx: int = 0) -> torch.Tensor:
        """Спайковый паттерн -> фазовый вектор (filler), связанный с ролью.

        Активный юнит = фаза 0 (+1), молчащий = фаза pi (-1): обе точки лежат
        на единичной окружности, поэтому вектор остаётся unit-modulus и
        связывание/развязывание точны. Информация о паттерне живёт в ЗНАКЕ
        вещественной части, а роль — в фазовом сдвиге.
        """
        phase = math.pi * (1.0 - spikes)
        if phase.shape[-1] != self.dim:
            raise ValueError(
                f"encode_pattern: длина паттерна {phase.shape[-1]} != vsa_dim {self.dim}. "
                "Фазовый вектор живёт в той же размерности, что и паттерн.")
        filler = torch.polar(torch.ones_like(phase), phase)
        return self.bind(filler, self.role(role_idx))

    def decode_pattern(self, bound: torch.Tensor, role_idx: int,
                       threshold: float = 0.0) -> torch.Tensor:
        """Обратное преобразование связанного вектора в спайковый паттерн."""
        filler = self.unbind(bound, self.role(role_idx))
        logits = filler.real
        return (logits > threshold).to(logits.dtype)

    # ---- диагностика -------------------------------------------------
    def capacity_probe(self, n_pairs: int, seed: int = 0) -> Dict[str, float]:
        """Эмпирическая ёмкость: точность разыменования при n_pairs парах.

        Символы и роли выбираются БЕЗ повторов (randperm): иначе одна и та же
        пара встречается дважды и метрика завышается тривиально.
        """
        g = torch.Generator().manual_seed(seed)
        n_pairs = min(n_pairs, self.n_sym, self.roles.shape[0])
        fill_idx = torch.randperm(self.n_sym, generator=g)[:n_pairs]
        role_idx = torch.randperm(self.roles.shape[0], generator=g)[:n_pairs]
        bound = [self.bind(self.symbol(fill_idx[i]), self.role(int(role_idx[i])))
                 for i in range(n_pairs)]
        mem = self.bundle(bound)
        ok = 0
        for i in range(n_pairs):
            rec = self.unbind(mem, self.role(int(role_idx[i])))
            ok += int(self.cleanup(rec)[0].item() == int(fill_idx[i]))
        return {"n_pairs": float(n_pairs), "accuracy": ok / max(1, n_pairs),
                "snr": float(self.cleanup(
                    self.unbind(mem, self.role(int(role_idx[0]))))[1])}
