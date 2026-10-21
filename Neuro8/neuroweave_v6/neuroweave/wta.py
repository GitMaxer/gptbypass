"""WTA (winner-take-all) NEUROWEAVE: маршрутизация по зонам + k-WTA внутри зоны.

Всё — тензорные операции: мягкий WTA (softmax по энергиям зон с байесом
быстрого контекста), заостряемый пульсом NE, и жёсткий k-WTA внутри зоны
(top-k спайков по мембранному потенциалу). Оба граф-безопасны.
"""
from typing import List

import torch

from .config import NWConfig, WTACfg


def kwt_mask(V_spk: torch.Tensor, spikes: torch.Tensor, k: int) -> torch.Tensor:
    """k-WTA: из спайковнут зоны оставить top-k по V в момент спайка.

    Если спайков меньше k — проходят все. scores = V, где спайк; иначе -1e9.
    """
    scores = torch.where(spikes, V_spk, torch.full_like(V_spk, -1e9))
    k = min(k, scores.numel())
    if k <= 0:
        return torch.zeros_like(spikes)
    top = torch.topk(scores, k).values
    thr = top[k - 1]
    return spikes & (V_spk >= thr)


def wta_mix(rates: torch.Tensor, ctx_bias: torch.Tensor, ne_eff: torch.Tensor,
            wcfg: WTACfg) -> torch.Tensor:
    """Мягкий WTA по зонам задач: mix = softmax(logits), logits =
    rate/rate_ref + k_ctx*ctx_bias, заострённые пульсом NE: logits*(1+k_ne*NE).

    rates [Z], ctx_bias [Z] (только зоны задач ненулевые), ne_eff — скалярный
    тензор (пульс NE от детектора переключения контекста).
    """
    logits = rates / wcfg.rate_ref + wcfg.k_ctx * ctx_bias
    logits = logits * (1.0 + wcfg.k_ne * ne_eff)
    return torch.softmax(logits / wcfg.temp, dim=0)


def ctx_bias_from_ctx(ctx: torch.Tensor, cfg: NWConfig,
                      task_zone_ids: List[int]) -> torch.Tensor:
    """Быстрый контекст -> байес зон: ctx = one-hot [syntax, logic].
    Зона B (синтаксис) получает +ctx[0]-ctx[1], зона C (логика) — наоборот.
    """
    Z = len(cfg.zones)
    bias = torch.zeros(Z, device=ctx.device)
    bid, cid = task_zone_ids
    syntax, logic = ctx[0], ctx[1]
    bias[bid] = syntax - logic
    bias[cid] = logic - syntax
    return bias
