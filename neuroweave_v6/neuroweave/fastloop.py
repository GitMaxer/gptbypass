"""Быстрый контур NEUROWEAVE — ОДИН шаг сети, полностью граф-безопасный.

Почему это работает внутри torch.cuda.CUDAGraph:

  1. СОСТОЯНИЕ ТОЛЬКО IN-PLACE. Всё, что должно накапливаться между
     реплеями (t, ctx_ema, rate_ema, g_ampa/g_nmda/g_gaba, refr, буферы
     токенов), обновляется in-place (add_/mul_/copy_/index_add_).
     Анти-паттерн t = t + 1 создаёт НОВЫЙ тензор: граф запишет ядро,
     читающее СТАРУЮ память, и счётчик никогда не продвинется.
  2. ВЫХОДЫ пересчитываются каждый реплей из состояния — их пул памяти
     стабилен, ссылки остаются валидными.
  3. ВЕТВЛЕНИЕ — только по статическим (python) константам: списки зон,
     флаг self.learn. Никаких .item()/.cpu() по значениям тензоров.
  4. DA/NE — ТЕНЗОРНЫЕ пульсы: DA из расписания (индекс tensor-шагом t),
     NE рождается ВНУТРИ контура детектером |ctx - ctx_ema|.
  5. RNG (5-HT шум) — graph-safe philox: каждый реплей — свежие числа.

Порядок шага: t++ -> расписание модуляторов -> NE-детекция -> ctx_ema ->
WTA-стробы -> входы (токен + задержанная линия) -> связность (gather) ->
сон-GABA + S7-всплеск -> мембрана (ТОЧНЫЙ интегратор, GABA-reversal) ->
спайки/k-WTA -> частоты -> readout.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .config import NWConfig
from .neurochem import ModulatorSchedules, tonic_schedules
from .neurons import e_cl_effective, membrane_exact_step
from .s7 import S7
from .wta import ctx_bias_from_ctx, kwt_mask, wta_mix
from .zones import Topology, build_topology


@dataclass
class FastOutputs:
    """Ссылки на тензоры шага (при работе в графе — пул графа, валидны
    до следующего replay; копируйте, если нужно хранить)."""
    spikes: torch.Tensor     # [N] {0,1}
    V: torch.Tensor          # [N] мВ, пост-обновление (после сброса)
    V_inf: torch.Tensor      # [N] неподвижная точка (для диагностики)
    r_B: torch.Tensor        # [1] логит readout синтаксиса
    r_C: torch.Tensor        # [1] логит readout логики
    logit_mix: torch.Tensor  # [1] решение WTA-смеси
    mix: torch.Tensor        # [Z] softmax WTA по зонам
    rates: torch.Tensor      # [Z] EMA частот зон
    ne_dyn: torch.Tensor     # [1] пульс NE (детектор переключения)
    da_t: torch.Tensor       # [1] текущий DA (из расписания)


class FastLoop:
    def __init__(self, cfg: NWConfig, device,
                 tables: Optional[Dict[str, torch.Tensor]] = None,
                 schedule: Optional[ModulatorSchedules] = None,
                 seed: int = 0):
        self.cfg = cfg
        self.device = device
        self.mcfg = cfg.membrane
        gen = torch.Generator().manual_seed(cfg.seed + 31 * seed)

        self.topo = build_topology(cfg, device, tables=tables, seed=seed)
        self.topo_names = self.topo.zone_names
        Z = len(self.topo_names)
        self.Z = Z
        N = cfg.total_n()
        self.N = N

        # ---------- состояние (всё персистентно, обновляется in-place) ----------
        def st(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.t = st(1, dtype=torch.long)
        self.V = st(N)
        self.V.fill_(self.mcfg.v_rest)
        self.refr = st(N, dtype=torch.long)
        self.g_ampa, self.g_nmda, self.g_gaba = st(N), st(N), st(N)
        self.spikes = st(N)
        self.buf_tok = st(1, dtype=torch.long)          # текущий токен
        self.buf_prev_tok = st(1, dtype=torch.long)     # токен прошлого шага
        self.buf_prev_win = st(1, dtype=torch.long)     # токен ПРОШЛОГО ОКНА (линия задержки)
        self.ctx_ema = st(2)
        self.ctx_ema.fill_(0.5)
        self.rate_ema = st(Z)
        self.rate_ema.fill_(0.05)
        self.tr_unit = st(N)   # пер-юнитный след спайков (вход readout)

        # ---------- модуляторы ----------
        self.sched = schedule or tonic_schedules(
            cfg.sched_len, Z, device, ach=0.5, ht=0.35)
        self.ne_profile = torch.tensor(
            [1.0 if n in ("B", "C") else 0.2 for n in self.topo_names],
            device=device)
        self.e_cl_base = torch.full((Z,), self.mcfg.e_cl, device=device)
        self.th_base = torch.empty(N, device=device)
        for z in cfg.zones:
            self.th_base[self.topo.e_slices[z.name]] = self.mcfg.v_th
            self.th_base[self.topo.i_slices[z.name]] = (
                self.mcfg.v_th + self.mcfg.v_th_i_offset)
        self.zone_of = self.topo.zone_of

        # ---------- WTA ----------
        self.task_ids = [self.topo_names.index(n) for n in cfg.task_zones]
        self.task_ids_t = torch.tensor(self.task_ids, dtype=torch.long,
                                       device=device)

        # ---------- readout (w_slow-семейство, претрейн -> заморозка) ----------
        nB = cfg.zone("B").n_e
        nC = cfg.zone("C").n_e
        self.W_rB = (torch.randn(1, nB, generator=gen) * 0.01).to(device)
        self.b_B = torch.zeros(1, device=device)
        self.W_rC = (torch.randn(1, nC, generator=gen) * 0.01).to(device)
        self.b_C = torch.zeros(1, device=device)

        # ---------- S7 ----------
        self.s7 = S7(cfg.s7cfg, self.mcfg.v_rest, device)

        self.learn = False  # статический флаг пластичности (python-уровень)

        # кэш констант шага
        self._dec_ampa = torch.exp(torch.tensor(-self.mcfg.dt / self.mcfg.tau_ampa))
        self._dec_nmda = torch.exp(torch.tensor(-self.mcfg.dt / self.mcfg.tau_nmda))
        self._dec_gaba = torch.exp(torch.tensor(-self.mcfg.dt / self.mcfg.tau_gaba))
        self._kwt_k = {z.name: max(cfg.wta.kwt_k_min, int(z.n_e * cfg.wta.kwt_k_frac))
                       for z in cfg.zones}

    # ------------------------------------------------------------------
    def set_schedule(self, sched: ModulatorSchedules) -> None:
        self.sched = sched

    def reset_state(self) -> None:
        self.t.zero_()
        self.V.fill_(self.mcfg.v_rest)
        self.refr.zero_()
        self.g_ampa.zero_(); self.g_nmda.zero_(); self.g_gaba.zero_()
        self.spikes.zero_()
        self.buf_tok.zero_(); self.buf_prev_tok.zero_(); self.buf_prev_win.zero_()
        self.ctx_ema.fill_(0.5)
        self.rate_ema.fill_(0.05)
        self.tr_unit.zero_()

    def snapshot_wslow(self) -> Dict[str, torch.Tensor]:
        snap = self.topo.snapshot_wslow()
        snap["readout:W_rB"] = self.W_rB.detach().clone()
        snap["readout:b_B"] = self.b_B.detach().clone()
        snap["readout:W_rC"] = self.W_rC.detach().clone()
        snap["readout:b_C"] = self.b_C.detach().clone()
        return snap

    def wslow_drift(self, snap: Dict[str, torch.Tensor]):
        cur = self.snapshot_wslow()
        exact = all(torch.equal(cur[k], snap[k]) for k in snap)
        maxabs = max((cur[k] - snap[k]).abs().max().item() for k in snap)
        return exact, maxabs

    # ------------------------------------------------------------------
    def step(self, tok: torch.Tensor, ctx: torch.Tensor,
             target: Optional[torch.Tensor] = None) -> FastOutputs:
        """tok: int64 [1] (id токена), ctx: float [2] (быстрый контекст,
        one-hot задач), target: float [1] (метка — только в претрейне)."""
        cfg, mcfg = self.cfg, self.mcfg
        wcfg = cfg.wta

        # (1) счётчик шага — ТОЛЬКО in-place (иначе граф его «заморозит»)
        self.t.add_(1)
        idx = self.t % self.sched.ach.shape[0]           # tensor-индекс
        ach_row = self.sched.ach[idx].squeeze(0)          # [Z]
        ht_row = self.sched.ht[idx].squeeze(0)
        da_row = self.sched.da[idx].squeeze(0)
        ne_row = self.sched.ne[idx].squeeze(0)
        g_sleep_row = self.sched.g_sleep[idx].squeeze(0)
        da_t = da_row.mean()                              # [скаляр-тензор]

        # (2) S7: гомеостаз ДО эффектов (тензорные сигмоиды)
        ach_row = ach_row + self.s7.ach_boost(self.rate_ema)
        surge = self.s7.gaba_surge(self.rate_ema)         # [Z], шунтирующий

        # (3) NE-детектор переключения контекста (пульс рождается В КОНТУРЕ)
        ne_dyn = wcfg.ne_k_det * (ctx - self.ctx_ema).abs().sum()
        ne_row = ne_row + ne_dyn * self.ne_profile        # [Z]
        ne_eff_task = ne_row[self.task_ids_t].mean()      # скаляр

        # (4) быстрый след контекста (in-place EMA)
        self.ctx_ema.mul_(1.0 - wcfg.ctx_ema_rho).add_(ctx, alpha=wcfg.ctx_ema_rho)

        # (5) WTA-стробы маршрутизации (замкнутый контур: mix от частот+контекста)
        ctx_bias = ctx_bias_from_ctx(ctx, cfg, self.task_ids)
        rates_task = self.rate_ema[self.task_ids_t]
        bias_task = ctx_bias[self.task_ids_t]
        mix_task = wta_mix(rates_task, bias_task, ne_eff_task, wcfg)
        gate_vec = wcfg.gate_floor + (1.0 - wcfg.gate_floor) * mix_task  # [2]
        mix_full = torch.zeros(self.Z, device=ctx.device)
        mix_full = mix_full.index_copy(0, self.task_ids_t, mix_task)

        # (6) буферы токенов (in-place!): линия задержки держит токен ПРОШЛОГО
        #     ОКНА (K шагов назад), а не прошлого шага — иначе она дублирует
        #     текущий токен внутри окна и «совпадение» теряет смысл.
        is_new_win = tok != self.buf_tok                       # [1] bool
        new_prev_win = torch.where(is_new_win, self.buf_tok, self.buf_prev_win)
        self.buf_prev_win.copy_(new_prev_win)
        self.buf_prev_tok.copy_(self.buf_tok)
        self.buf_tok.copy_(tok)

        # (7) затухание проводимостей (in-place)
        self.g_ampa.mul_(self._dec_ampa)
        self.g_nmda.mul_(self._dec_nmda)
        self.g_gaba.mul_(self._dec_gaba)

        # (8) входные линии: текущий токен + линия задержки (прошлое ОКНО;
        #     совпадение классов на одних юнитах = надпороговый «И»)
        for line in self.topo.lines:
            gate = gate_vec[line.gate_id] if line.gate_id >= 0 else 1.0
            src_t = self.buf_tok if line.name.startswith("cur_") else self.buf_prev_win
            drive = line.table[src_t].squeeze(0) * (cfg.in_gain * gate)
            if line.kind == "ampa":
                self.g_ampa.index_add_(0, line.dst_idx, drive)
            else:
                self.g_nmda.index_add_(0, line.dst_idx, drive)

        # (9) связность: фиксированный fan-in, физическая изоляция по построению
        for g in self.topo.groups:
            gate = gate_vec[g.gate_id] if g.gate_id >= 0 else 1.0
            spk_g = self.spikes[g.src_idx]                      # [n_dst, fan_in]
            cur = (g.w * spk_g).sum(1) * gate
            if g.kind == "exc":
                self.g_ampa.index_add_(0, g.dst_idx, cur * g.f_ampa)
                self.g_nmda.index_add_(0, g.dst_idx, cur * (1.0 - g.f_ampa))
            else:
                self.g_gaba.index_add_(0, g.dst_idx, cur)

        # (10) сон-GABA (гуморальный, по зонам) + S7-всплеск — ШУНТИРУЮЩИЕ:
        #      обе добавки входят как g_GABA с E_cl >= V_rest => безопасны.
        g_gaba_extra = g_sleep_row[self.zone_of] + surge[self.zone_of]
        self.g_gaba.add_(g_gaba_extra)

        # (11) мембрана: ТОЧНЫЙ интегратор + GABA-reversal + KCC2-гомеостат
        e_cl_row = e_cl_effective(self.e_cl_base, mcfg)          # [Z]
        e_cl_n = e_cl_row[self.zone_of]                          # [N]
        v_th_eff = self.th_base - mcfg.k_ach_th * ach_row[self.zone_of]
        sigma_n = mcfg.sigma_ht * (0.35 + ht_row[self.zone_of])
        V_new, V_inf = membrane_exact_step(
            self.V, self.g_ampa, self.g_nmda, self.g_gaba, e_cl_n, mcfg,
            sigma=sigma_n)
        V_new = self.s7.scrub(V_new)

        # (12) спайки + рефрактерность + k-WTA внутри зон
        cand = (V_new >= v_th_eff) & (self.refr == 0)
        keep = torch.zeros_like(cand)
        for z in cfg.zones:
            keep[self.topo.e_slices[z.name]] = kwt_mask(
                V_new[self.topo.e_slices[z.name]],
                cand[self.topo.e_slices[z.name]], self._kwt_k[z.name])
            keep[self.topo.i_slices[z.name]] = cand[self.topo.i_slices[z.name]]
        V_final = torch.where(keep, torch.full_like(V_new, mcfg.v_reset), V_new)

        # in-place фиксация состояния
        self.V.copy_(V_final)
        refr_new = torch.where(keep, torch.full_like(self.refr, mcfg.refractory),
                               torch.clamp(self.refr - 1, min=0))
        self.refr.copy_(refr_new)
        self.spikes.copy_(keep.float())
        # пер-юнитный след спайков (in-place EMA, вход readout — «частотное» кодирование)
        self.tr_unit.mul_(0.75).add_(self.spikes, alpha=0.25)

        # (13) частоты зон (EMA, in-place)
        rates_new = torch.stack([keep[self.topo.e_slices[z.name]].float().mean()
                                 for z in cfg.zones])
        self.rate_ema.mul_(1.0 - wcfg.rate_alpha).add_(rates_new, alpha=wcfg.rate_alpha)

        # (14) readout зон задач — по СЛЕДУ спайков (устойчиво к пачечной динамике)
        tr_B = self.tr_unit[self.topo.e_slices["B"]]
        tr_C = self.tr_unit[self.topo.e_slices["C"]]
        r_B = self.W_rB @ tr_B + self.b_B
        r_C = self.W_rC @ tr_C + self.b_C
        logit_mix = mix_full[self.task_ids_t[0]] * r_B + \
            mix_full[self.task_ids_t[1]] * r_C

        # (15) пластичность (ТОЛЬКО на претрейне; на «Экзамене» learn=False,
        #      тензор w_slow не получает НИ ОДНОЙ записи -> дрейф строго 0)
        if self.learn:
            eta = cfg.eta_readout
            if target is not None:
                # задача-селективно: обновляем только readout АКТИВНОЙ зоны (по ctx)
                ind_B = (ctx[0] > ctx[1]).to(r_B.dtype)
                ind_C = 1.0 - ind_B
                err_B = (target - torch.tanh(r_B)) * ind_B
                self.W_rB = self.W_rB + eta * err_B.unsqueeze(1) * tr_B.unsqueeze(0)
                self.b_B = self.b_B + eta * err_B
                err_C = (target - torch.tanh(r_C)) * ind_C
                self.W_rC = self.W_rC + eta * err_C.unsqueeze(1) * tr_C.unsqueeze(0)
                self.b_C = self.b_C + eta * err_C
            # трёхфакторный Хебб w_slow: eta_h * DA * (пре-след * пост-спайк)
            eta_h = cfg.eta_hebb * da_t
            for g in self.topo.groups:
                if g.kind != "exc" or g.trace is None:
                    continue
                pre = self.spikes[g.src_idx]
                g.trace = g.trace * 0.9 + pre
                post = self.spikes[g.dst_idx].unsqueeze(1)
                upd = eta_h * g.trace * post
                g.w = torch.clamp(g.w * (1.0 - cfg.hebb_decay) + upd,
                                  min=0.0, max=cfg.w_max)

        return FastOutputs(self.spikes, self.V, V_inf, r_B, r_C, logit_mix,
                           mix_full, self.rate_ema, ne_dyn, da_t)
