# src/spiker/topology/workspace.py
# Этап 2 (канон v4.0/v4.0.1, разд. 2.4 + табл. 6): Global Workspace —
# арена конкуренции зон за выходной токен (WTA / Soft-Attention).
#
# Выход организма — не сумма мнений зон, а результат конкуренции.
# Физическая основа (разд. 2.4): активация доминантной популяции
# возбуждает ГАМК-пул, который пропорционально подавляет мембранные
# потенциалы альтернативных путей (ур. 9):
#     I_GABA,i = gamma_lat * SUM_{j != i} W^GABA_ij s_j.
# Следствия: единый когерентный фокус внимания и защита от
# эпилептиформного спайкового шторма и «шизофренического» выхода.
#
# Уровень формализации Этапа 2: конкуренция ведётся по МАКРО-заявкам
# зон (доминирование ансамблей — активность/уверенность чтения), а
# мембранное подавление альтернатив живёт в двух нижних слоях:
#   * внутризонный контур W_ei/W_ie (zone.py, ур. 9, уже в Этапе 1);
#   * глобальный GABA-пул workspace (storm_brake): мягкое капирование
#     суммарной частоты — пропорциональное торможение всех путей при
#     превышении нормы (разд. 2.4, «защита от штормов»).
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspace(nn.Module):
    """
    Global Workspace: латеральное GABA-торможение (ур. 9) + Soft-WTA.

    Протокол такта:
        bids [B, Z] >= 0                      — заявки зон (активность или
                                                 уверенность чтения);
        lat  = gamma_lat * bids @ W_gaba^T     — латеральное торможение
                                                 (диагональ W строго 0);
        bids_eff = bids - lat                  — подавление альтернатив;
        alpha = softmax(beta_focus * bids_eff) — мягкий победитель;
        fused = SUM_z alpha_z * logits_z       — единый когерентный выход.
    """

    def __init__(
        self,
        n_zones: int,
        gamma_lat: float = 0.6,   # сила латерального торможения (ур. 9)
        beta_focus: float = 6.0,  # резкость Soft-WTA (температура обратная)
        seed: int = 2027,
    ):
        super().__init__()
        if n_zones < 2:
            raise ValueError("Конкуренция требует минимум двух зон")
        if not (0.0 < gamma_lat <= 1.0):
            raise ValueError(f"gamma_lat={gamma_lat} должен быть в (0, 1]")
        self.n_zones = n_zones
        self.gamma_lat = float(gamma_lat)
        self.beta_focus = float(beta_focus)

        # Латеральная GABA-матрица (ур. 9): неотрицательная, строки
        # нормированы (торможение пропорционально), диагональ строго
        # нулевая — популяция себя не тормозит. PoC: фиксированная.
        gen = torch.Generator().manual_seed(seed)
        W = torch.rand(n_zones, n_zones, generator=gen)
        W.fill_diagonal_(0.0)
        W = W / W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W_gaba", W)

    # ------------------------------------------------------------------
    def lateral_gaba(self, activity: torch.Tensor) -> torch.Tensor:
        """Ур. (9): I_GABA,i = gamma_lat * SUM_{j != i} W_ij s_j."""
        return self.gamma_lat * (activity @ self.W_gaba.T)

    def compete(
        self, bids: torch.Tensor
    ) -> tuple:
        """
        Конкуренция за фокус внимания.

        Параметры:
            bids: [B, Z] — неотрицательные заявки зон (активность
                          ансамблей или уверенность чтения, Soft-Attention).
        Возвращает:
            alpha:     [B, Z] — веса Soft-WTA (сумма 1);
            winner:    [B]    — индекс доминирующей зоны такта;
            bids_eff:  [B, Z] — заявки после латерального торможения.
        """
        if bids.shape[-1] != self.n_zones:
            raise ValueError(
                f"Число заявок {bids.shape[-1]} != число зон {self.n_zones}"
            )
        if (bids < 0).any():
            raise ValueError("Заявки Global Workspace должны быть неотрицательными")
        bids_eff = bids - self.lateral_gaba(bids)
        alpha = F.softmax(self.beta_focus * bids_eff, dim=-1)
        winner = bids_eff.argmax(dim=-1)
        return alpha, winner, bids_eff

    def fuse_logits(
        self,
        alpha: torch.Tensor,
        logits_per_zone: list,
    ) -> torch.Tensor:
        """Единый когерентный выход: взвешенное смесение мнений зон.

        Концентрированное alpha ~ трансляция победителя без примеси
        несогласованных гипотез — защита от «шизофренического» текста."""
        stack = torch.stack(logits_per_zone, dim=1)      # [B, Z, V]
        return torch.einsum("bz,bzv->bv", alpha, stack)

    # ------------------------------------------------------------------
    def storm_brake(
        self,
        activity: torch.Tensor,
        rate_cap: float,
    ) -> torch.Tensor:
        """Глобальный GABA-пул: мягкое капирование суммарной частоты.

        Гладкое пропорциональное торможение ВСЕХ путей при превышении
        нормы rate_cap (защита от эпилептиформных штормов, разд. 2.4):
            brake = 1 / (1 + relu(total - cap) / cap)
        Ниже нормы торможения нет (brake = 1); выше — суммарная частота
        точно возвращается к капу, доли зон сохраняются. Тензоро-безопасно,
        без ветвлений по данным (CUDA-Graphs-совместимость, Этап 5).
        """
        if rate_cap <= 0:
            raise ValueError(f"rate_cap={rate_cap} должен быть положительным")
        total = activity.sum(dim=-1, keepdim=True)
        brake = 1.0 / (1.0 + F.relu(total - rate_cap) / rate_cap)
        return activity * brake
