# src/spiker/core/synapse.py
# Канон v4.0/v4.0.1. Патчи: Y1 (гомеостатическое затухание w_fast на каждом
# такте), Y5 (sync-free обновление без torch.any-ветвления), Y7 (метод
# consolidate_to_slow — исправленное ур. 21), Y2 (guard сброса по форме
# и устройству).
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class StatefulSynapse(nn.Module):
    """
    Синаптическая матрица с состоянием:
      - w_slow: долговременная память, обучается суррогатным градиентом в Фазе 1;
      - w_fast: быстрая пластичность, обучается 3-факторным Reward-STDP в Фазе 2;
      - честная O(N)-память следов: векторы [Batch, N_pre] / [Batch, N_post]
        вместо матрицы [Batch, N_post, N_pre] (разд. 5.2 канона).

    Канон v4.0.1: w_fast затухает на КАЖДОМ такте update_traces_and_plasticity()
    (гомеостаз по разд. 5.1), а захват дофамином вычисляется без условных
    ветвлений по значениям тензоров (совместимость с CUDA Graphs, Этап 5).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dt: float = 1.0,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_elig: float = 100.0,   # окно памяти следа для Tag-and-Capture
        w_fast_max: float = 0.5,
        w_fast_decay: float = 0.9995,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt
        self.w_fast_max = w_fast_max
        self.w_fast_decay = w_fast_decay

        # Долговременные веса (Фаза 1)
        self.w_slow = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.w_slow, a=math.sqrt(5))

        # Быстрые веса (Фаза 2) — не участвуют в стандартном autograd-оптимизаторе
        self.register_buffer("w_fast", torch.zeros(out_features, in_features))
        # Снапшот для EWC (ур. 13)
        self.register_buffer("w_snapshot", torch.zeros(out_features, in_features))

        # Константы затухания следов O(N)
        self.register_buffer("decay_pre", torch.exp(torch.tensor(-dt / tau_pre)))
        self.register_buffer("decay_post", torch.exp(torch.tensor(-dt / tau_post)))
        self.register_buffer("decay_elig", torch.exp(torch.tensor(-dt / tau_elig)))

        # Векторы следов размера O(N_pre) и O(N_post)
        self.z_pre: Optional[torch.Tensor] = None
        self.z_post: Optional[torch.Tensor] = None
        self.e_pre: Optional[torch.Tensor] = None
        self.e_post: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.z_pre = torch.zeros((batch_size, self.in_features), device=device, dtype=dtype)
        self.z_post = torch.zeros((batch_size, self.out_features), device=device, dtype=dtype)
        self.e_pre = torch.zeros((batch_size, self.in_features), device=device, dtype=dtype)
        self.e_post = torch.zeros((batch_size, self.out_features), device=device, dtype=dtype)

    def freeze_for_phase2(self):
        """Замораживает w_slow и снимает снапшот по окончании Фазы 1.

        Примечание (Этап 3): оптимизатор Фазы 1 должен быть пересобран после
        заморозки — накопленные моменты Adam по w_slow иначе продолжили бы
        сдвигать замороженный параметр.
        """
        self.w_slow.requires_grad_(False)
        self.w_snapshot.copy_(self.w_slow.data)

    def forward(self, s_pre: torch.Tensor) -> torch.Tensor:
        """Постсинаптический ток: (w_slow + w_fast) @ s_pre."""
        w_total = self.w_slow + self.w_fast
        return F.linear(s_pre, w_total)

    @torch.no_grad()
    def update_traces_and_plasticity(
        self,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
        da_reward: Optional[torch.Tensor] = None,
        lr_fast: float = 0.01,
        a_plus: float = 1.0,
        a_minus: float = 0.8,
    ):
        """
        Обновляет O(N)-следы и применяет факторное 3-правило Reward-STDP (ур. 12).

        da_reward: [Batch, 1] — ошибка предсказания награды; канон (ур. 14):
        DA = max(0, RPE) — только положительное подкрепление, неудачи не
        разрушают собранные связи.
        """
        if (
            self.z_pre is None
            or self.z_pre.shape != s_pre.shape
            or self.z_pre.device != s_pre.device  # патч Y2
        ):
            self.reset_state(s_pre.shape[0], s_pre.device, s_pre.dtype)

        # 1. Быстрые синаптические следы спайков
        self.z_pre = self.z_pre * self.decay_pre + s_pre
        self.z_post = self.z_post * self.decay_post + s_post

        # 2. Медленные следы пригодности (ур. 11) — память строго O(N)
        self.e_pre = self.e_pre * self.decay_elig + self.z_pre
        self.e_post = self.e_post * self.decay_elig + self.z_post

        # 3. Гомеостаз (патч Y1): затухание w_fast на каждом такте —
        #    оперативная память обязана медленно забываться сама по себе.
        self.w_fast.mul_(self.w_fast_decay)

        # 4. Tag-and-Capture (патч Y5): без ветвления по значениям тензоров —
        #    при DA <= 0 einsum-вклад естественно нулевой (da_pos — множитель),
        #    граф вычислений остаётся захватываемым в CUDA Graphs.
        if da_reward is not None:
            da_pos = torch.clamp(da_reward, min=0.0)  # ур. 14
            # Внешние произведения, усреднённые по батчу: [Out, In]
            # LTP: длинный пред-след встречает свежий пост-спайк;
            # LTD: свежий пре-спайк встречает затухающий пост-след.
            mod_post = s_post * da_pos      # [Batch, Out]
            mod_e_post = self.e_post * da_pos  # [Batch, Out]
            ltp = torch.einsum("bo,bi->oi", mod_post, self.e_pre)
            ltd = torch.einsum("bo,bi->oi", mod_e_post, s_pre)
            delta_w = lr_fast * (a_plus * ltp - a_minus * ltd) / float(s_pre.shape[0])
            self.w_fast.add_(delta_w)

        # 5. Амплитудный захват: |w_fast| <= w_fast_max (табл. 8 канона)
        self.w_fast.clamp_(-self.w_fast_max, self.w_fast_max)

    @torch.no_grad()
    def consolidate_to_slow(
        self,
        lam_consol: float = 0.1,
        theta_c: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        """
        Консолидация сна: перенос подтвердившихся быстрых весов в w_slow.

        Канон v4.0.1 (патч Y7) — исправленное ур. (21). Запись v4.0
        Sign(w)⊙|w| ≡ w вырождалась в простое копирование всего буфера без
        селекции. Новая каноническая форма — мягкий порог с переносом:
            g = σ((|w_fast| − θ_c) / T)     — доверие синапсу;
            w_slow += λ · g · w_fast         — перенос подтвердившегося;
            w_fast  *= (1 − g)               — снятие перенесённого с буфера.
        Слабые (шумовые) флуктуации ниже порога θ_c остаются в w_fast и
        затухают гомеостазом, а не кристаллизуются.
        """
        if theta_c is None:
            theta_c = 0.5 * self.w_fast_max
        if temperature is None:
            temperature = 0.05 * self.w_fast_max
        gate = torch.sigmoid((self.w_fast.abs() - theta_c) / temperature)
        self.w_slow.data.add_(lam_consol * gate * self.w_fast)
        self.w_fast.mul_(1.0 - gate)
