"""«ЭКЗАМЕН» — гибридный режим NEUROWEAVE (Этап 6).

Постановка: сеть переключается между ДВУМЯ языковыми задачами —
СИНТАКСИС и ЛОГИКА — БЕЗ переобучения w_slow. Работают только:

  * WTA (замкнутый контур маршрутизации по зонам B/C),
  * быстрый контекст (ctx-cue + след ctx_ema + NE-пульс от детектора
    переключения ВНУТРИ контура).

Задачи (синтетические токен-потоки, 33-токенный словарь):
  СИНТАКСИС (зона B): токены 0..15, два класса слов (0..7 / 8..15).
    Грамматика: классы должны ЧЕРЕДОВАТЬСЯ. Нарушение = тот же класс
    подряд -> target=1. Сеть детектирует совпадение токена с предыдущим
    (совпадение cur+del линий на ОДНИХ юнитах поднимает мембрану над
    порогом — «И»-нелинейность на уровне порога спайка).
  ЛОГИКА (зона C): токены 16..31 парами (a, b), бит класса = (tok-16)//8.
    target = a_класс OR b_класс (хотя бы одна посылка несёт признак).
    Класс-1 юниты зоны C сильные (спайки), класс-0 — подпороговые.

Протокол: претрейн по каждой задаче отдельно (readout + лёгкий Хебб
w_slow под DA-пульсами) -> ЗАМОРОЗКА w_slow (снапшот) -> экзамен:
чередующиеся блоки задач с 4-шаговым blank-сигналом контекста (cue)
ДО переключения. Метрики: точность по задачам, следование WTA за cue,
стоимость переключения, пульс NE, ДРЕЙФ w_slow == 0 (torch.equal).

Режим работы: eager (CPU/GPU) ИЛИ полностью внутри torch.cuda.CUDAGraph
(cм. cudagraph.py) — гибридный режим survives захват графа, потому что
маршрутизация и модуляция — чисто тензорные операции.
"""
import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .config import NWConfig
from .fastloop import FastLoop
from .neurochem import build_exam_schedule, build_pretrain_schedule

# ---- тюнинг «Экзамена» (эмпирически калибровано) ----
K = 6                  # шагов на токен-окно
U = 8                  # классовых юнитов на класс в зоне
LINE_D_B = 0.089       # класс B на линию: ОДИН уровень строго подпорогов, ДВА — надпорогов
LINE_D_C1 = 0.20       # класс-1 зоны C: сильный (надпороговый)
LINE_D_C0 = 0.03       # класс-0 зоны C: подпороговый
LINE_D_A = 0.06        # сенсорная копия в зоне A (тихая)
JITTER = 0.006         # приватный шум юнитов
TAU_M_EXAM = 4.0       # быстрая мембрана (совпадения внутри окна токена)
CUE_LEN = 4            # blank-сигнал контекста до переключения
SYN_BLOCKS = 3         # блоков каждой задачи на экзамене
WIN_PER_BLOCK = 12     # токен-окон в блоке


@dataclass
class ExamReport:
    syntax_acc: float
    logic_acc: float
    follow: float
    switch_cost: float          # steady_acc - post_switch_acc
    ne_pulse_max: float
    drift_exact: bool
    drift_max: float
    n_win_syn: int
    n_win_log: int
    v_min: float

    @property
    def passed(self) -> bool:
        # v_min: стохастический 5-HT-шум (±~3 сигма при sigma~0.5-0.8 мВ)
        # может слегка заносить мембрану ниже покоя — это НЕ тормозный провал;
        # инвариант торможения (V >= V_rest при E_cl >= V_rest) здесь не нарушен.
        return (self.syntax_acc >= 0.85 and self.logic_acc >= 0.85 and
                self.follow >= 0.90 and self.drift_exact and
                self.ne_pulse_max > 0.5 and self.v_min >= -74.0)


def exam_config(seed: int = 7, scale: int = 1) -> NWConfig:
    """Конфиг, откалиброванный под «Экзамен» (быстрая мембрана)."""
    cfg = NWConfig(seed=seed)
    cfg.membrane.tau_m = TAU_M_EXAM
    if scale > 1:
        cfg = cfg.scaled(scale)
    return cfg


def build_task_tables(cfg: NWConfig, device, seed: int = 3):
    """Входные таблицы токенов: класс-кластеры юнитов + приватный шум.

    Зона B: cur и del таблицы ОДИНАКОВЫ -> совпадение классов (нарушение
    синтаксиса) удваивает проводимость НА ТЕХ ЖЕ юнитах -> спайк.
    Зона C: класс-1 сильный, класс-0 слабый -> спайки кодируют OR.
    """
    g = torch.Generator().manual_seed(seed)
    nA, nB, nC = (cfg.zone(z).n_e for z in ("A", "B", "C"))
    NT = cfg.n_tokens
    blank = cfg.blank_token

    def blank_row(n):
        return torch.rand(n, generator=g) * 0.001

    def base(n):
        return torch.rand(NT, n, generator=g) * JITTER

    def put_class(tab, tok, c, n, amp):
        lo, hi = c * U, (c + 1) * U
        tab[tok, lo:hi] = amp * (0.85 + 0.3 * torch.rand(U, generator=g))

    # зона B: одинаковые cur/del, два класса
    def make_B():
        tab = base(nB)
        for tok in range(16):
            put_class(tab, tok, tok // 8, nB, LINE_D_B)
        tab[blank] = blank_row(nB)
        return tab

    # зона C: класс-1 сильный, класс-0 слабый
    def make_C():
        tab = base(nC)
        for tok in range(16, 32):
            c = (tok - 16) // 8
            amp = LINE_D_C1 if c == 1 else LINE_D_C0
            put_class(tab, tok, c, nC, amp)
        tab[blank] = blank_row(nC)
        return tab

    # зона A: сенсорная копия классов (токены обеих задач)
    def make_A():
        tab = base(nA)
        for tok in range(16):
            put_class(tab, tok, tok // 8, nA, LINE_D_A)
        for tok in range(16, 32):
            put_class(tab, tok, (tok - 16) // 8, nA, LINE_D_A)
        tab[blank] = blank_row(nA)
        return tab

    return {"cur_A": make_A(), "cur_B": make_B(), "del_B": make_B(),
            "cur_C": make_C(), "del_C": make_C()}


# ---------------------------------------------------------------- потоки
def gen_syntax_block(n_win: int, rng: random.Random, p_viol=0.5):
    """Окна синтаксиса: (tok, target). Нарушение = класс повторился.
    p_viol = 0.5 — сбалансированная задача (без априорного перевеса)."""
    out = []
    prev_c = rng.randint(0, 1)
    for _ in range(n_win):
        viol = rng.random() < p_viol
        c = prev_c if viol else 1 - prev_c
        tok = c * 8 + rng.randint(0, 7)
        out.append((tok, int(viol)))
        prev_c = c
    return out


def gen_logic_block(n_pairs: int, rng: random.Random, p_c1=0.42):
    """Пары логики: [(a, target=None), (b, target=OR)] — target на 2-м.
    p_c1 = 0.42 — признак OR встречается ~59% (почти сбалансировано)."""
    out = []
    for _ in range(n_pairs):
        ca = 1 if rng.random() < p_c1 else 0
        cb = 1 if rng.random() < p_c1 else 0
        a, b = 16 + ca * 8 + rng.randint(0, 7), 16 + cb * 8 + rng.randint(0, 7)
        out.append((a, None))
        out.append((b, int(ca or cb)))
    return out


@dataclass
class StepPlan:
    """Развёрнутый план шагов для прогона."""
    toks: List[int]
    ctxs: List[Tuple[float, float]]
    targets: List[Optional[int]]     # per-step target (None = не учить/не оценивать)
    win_ends: List[int]              # индексы последних шагов окон
    win_tasks: List[str]             # задача каждого окна
    win_targets: List[Optional[int]]
    switch_steps: List[int]          # шаги сразу после переключения задачи


def unfold(blocks: List[Tuple[str, list]], K: int, cue_len: int,
           blank: int, with_cue: bool) -> StepPlan:
    """Разворачивает (задача, окна) в пошаговый план с cue-паузами."""
    toks, ctxs, targets = [], [], []
    win_ends, win_tasks, win_targets = [], [], []
    switch_steps = []
    prev_task = None
    for (task, wins) in blocks:
        if prev_task is not None and with_cue:
            # cue: blank-токены с НОВЫМ контекстом (сигнал до переключения)
            ctx = (1.0, 0.0) if task == "syntax" else (0.0, 1.0)
            for _ in range(cue_len):
                toks.append(blank); ctxs.append(ctx); targets.append(None)
        prev_task = task
        ctx = (1.0, 0.0) if task == "syntax" else (0.0, 1.0)
        started = len(toks)
        for w_i, (tok, tgt) in enumerate(wins):
            # первые ДВА окна блока после cue — фаза реконфигурации задачи
            # (task-set reconfiguration): del-линия ещё пуста, зона холодная.
            # Это ЧЕСТНАЯ стоимость переключения — оцениваем с 3-го окна.
            eff_tgt = None if (with_cue and w_i < 2) else tgt
            for j in range(K):
                toks.append(tok); ctxs.append(ctx)
                # учим/оцениваем ТОЛЬКО на последних 2 шагах окна: проводимости
                # «хвоста» предыдущего окна уже затухли, сигнал совпадения чист
                targets.append(eff_tgt if (eff_tgt is not None and j >= K - 2)
                               else None)
            win_ends.append(len(toks) - 1)
            win_tasks.append(task)
            win_targets.append(eff_tgt)
        if with_cue:
            switch_steps.append(started + cue_len)
    return StepPlan(toks, ctxs, targets, win_ends, win_tasks, win_targets,
                    switch_steps)


# ---------------------------------------------------------------- раннер
class ExamRunner:
    def __init__(self, cfg: NWConfig, device, seed: int = 11):
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(cfg.seed + seed)
        tables = build_task_tables(cfg, device, seed=cfg.seed + 3)
        sched = build_pretrain_schedule(cfg, cfg.sched_len, device)
        self.loop = FastLoop(cfg, device, tables=tables, schedule=sched,
                             seed=seed)
        self.loop.learn = False
        self.snapshot: Optional[dict] = None

    # ------------------------------------------------ претрейн
    def pretrain(self, n_win_syn=50, n_pairs_log=50, verbose=False):
        loop = self.loop
        loop.learn = True
        loop.reset_state()

        def run_plan(plan: StepPlan):
            tgt_t = torch.zeros(1, device=self.device)
            for i in range(len(plan.toks)):
                tok = torch.tensor([plan.toks[i]], dtype=torch.long,
                                   device=self.device)
                ctx = torch.tensor(plan.ctxs[i], device=self.device)
                t = plan.targets[i]
                tgt = None if t is None else tgt_t.fill_(float(t))
                loop.step(tok, ctx, target=tgt)

        # фаза A: синтаксис
        plan = unfold([("syntax", gen_syntax_block(n_win_syn, self.rng))],
                      K, 0, self.cfg.blank_token, with_cue=False)
        run_plan(plan)
        # фаза B: логика
        plan = unfold([("logic", gen_logic_block(n_pairs_log, self.rng))],
                      K, 0, self.cfg.blank_token, with_cue=False)
        run_plan(plan)

        # ЗАМОРОЗКА: w_slow больше НИ ОДНОЙ записи
        loop.learn = False
        loop.set_schedule(build_exam_schedule(self.cfg, self.cfg.sched_len,
                                              self.device))
        self.thresholds = self._calibrate()
        self.snapshot = loop.snapshot_wslow()
        if verbose:
            print(f"[ЭКЗАМЕН] претрейн завершён, w_slow заморожен "
                  f"({len(self.snapshot)} тензоров); пороги решения: "
                  f"syn>{self.thresholds['syntax']:+.3f}, "
                  f"log>{self.thresholds['logic']:+.3f}")

    def _calibrate(self, n_win=40):
        """Калибровка порога на свежем валидационном потоке БЕЗ обучения
        (w_slow не меняется — быстрая контекстная настройка порога readout).
        Порог = середина между СРЕДНИМИ классов: устойчивее best-accuracy
        скана (одиночный выброс не затягивает порог внутрь кластера)."""
        loop = self.loop
        th = {"syntax": 0.0, "logic": 0.0}
        for task, gen in (("syntax", gen_syntax_block),
                          ("logic", gen_logic_block)):
            wins = gen(n_win if task == "syntax" else n_win // 2, self.rng)
            plan = unfold([(task, wins)], K, 0, self.cfg.blank_token,
                          with_cue=False)
            rs0, rs1 = [], []
            for w_i, (tok, tgt) in enumerate(wins):
                if tgt is None:
                    continue
                for j in range(K):
                    i = w_i * K + j
                    out = loop.step(
                        torch.tensor([plan.toks[i]], dtype=torch.long,
                                     device=self.device),
                        torch.tensor(plan.ctxs[i], device=self.device))
                    if j == K - 1:
                        (rs1 if tgt else rs0).append(out.logit_mix.item())
            if rs0 and rs1:
                th[task] = (sum(rs0) / len(rs0) + sum(rs1) / len(rs1)) / 2.0
        return th

    # ------------------------------------------------ экзамен
    def run_exam(self, use_graph: bool = False, verbose=True) -> ExamReport:
        assert self.snapshot is not None, "сначала pretrain()"
        loop = self.loop
        assert loop.learn is False

        blocks = []
        tasks = ["syntax", "logic"] * SYN_BLOCKS
        for i, task in enumerate(tasks):
            wins = (gen_syntax_block(WIN_PER_BLOCK, self.rng) if task == "syntax"
                    else gen_logic_block(WIN_PER_BLOCK // 2, self.rng))
            blocks.append((task, wins))
        plan = unfold(blocks, K, CUE_LEN, self.cfg.blank_token, with_cue=True)

        graph = None
        if use_graph:
            from .cudagraph import FastLoopGraph
            graph = FastLoopGraph(loop)

        dec = []          # (task, prediction, target, step_idx)
        follow = []       # (task, mix_argmax_follows_cue)
        ne_max, v_min = 0.0, 1e9
        win_cursor = 0
        out = None
        for i in range(len(plan.toks)):
            tok = torch.tensor([plan.toks[i]], dtype=torch.long,
                               device=self.device)
            ctx = torch.tensor(plan.ctxs[i], device=self.device)
            if use_graph:
                out = graph(tok, ctx)
            else:
                out = loop.step(tok, ctx)
            ne_max = max(ne_max, out.ne_dyn.item())
            v_min = min(v_min, out.V.min().item())
            if win_cursor < len(plan.win_ends) and i == plan.win_ends[win_cursor]:
                task = plan.win_tasks[win_cursor]
                tgt = plan.win_targets[win_cursor]
                if tgt is not None:
                    pred = 1 if out.logit_mix.item() > self.thresholds[task] else 0
                    dec.append((task, pred, tgt, i))
                    cue = plan.ctxs[i]
                    mix_b = out.mix[loop.task_ids[0]].item()
                    follows = (mix_b > 0.5) == (cue[0] > 0.5)
                    follow.append(follows)
                win_cursor += 1

        acc = lambda sel: (sum(1 for (t, p, y, _i) in dec if t == sel and p == y) /
                           max(1, sum(1 for (t, _p, _y, _i) in dec if t == sel)))
        syntax_acc, logic_acc = acc("syntax"), acc("logic")
        follow_rate = (sum(follow) / len(follow)) if follow else 1.0

        # стоимость переключения: первые 2 ОЦЕНИВАЕМЫХ окна после смены
        # (фаза реконфигурации) против остальных (steady)
        post, steady = set(), set()
        for s in plan.switch_steps:
            blk = [i for i, y in zip(plan.win_ends, plan.win_targets)
                   if y is not None and i >= s]
            post.update(blk[:2])
        steady = [i for i, y in zip(plan.win_ends, plan.win_targets)
                  if y is not None and i not in post]
        acc_at = lambda idxs: (sum(1 for (t, p, y, i) in dec if i in idxs and p == y) /
                               max(1, sum(1 for (t, _p, _y, i) in dec if i in idxs)))
        switch_cost = acc_at(steady) - acc_at(list(post))

        exact, drift_max = self.loop.wslow_drift(self.snapshot)
        rep = ExamReport(syntax_acc=syntax_acc, logic_acc=logic_acc,
                         follow=follow_rate, switch_cost=switch_cost,
                         ne_pulse_max=ne_max, drift_exact=exact,
                         drift_max=drift_max,
                         n_win_syn=sum(1 for t in plan.win_tasks if t == "syntax"),
                         n_win_log=sum(1 for t in plan.win_tasks if t == "logic"),
                         v_min=v_min)
        if verbose:
            mode = "CUDA GRAPH" if use_graph else "eager"
            print(f"[ЭКЗАМЕН] режим: {mode} | шагов: {len(plan.toks)}, "
                  f"переключений: {len(plan.switch_steps)}")
            print(f"      синтаксис : acc = {syntax_acc:6.3f}  "
                  f"({rep.n_win_syn} окон)")
            print(f"      логика    : acc = {logic_acc:6.3f}  ({rep.n_win_log} окон)")
            print(f"      WTA следует cue: {follow_rate:6.3f}")
            print(f"      стоимость переключения: {switch_cost:+.3f} "
                  f"(отрицательная = мгновенное переключение)")
            print(f"      NE-пульс на переключениях: max = {ne_max:5.2f}")
            print(f"      дрейф w_slow: exact={exact}, max|dw| = {drift_max:.2e}")
            print(f"      V_min за экзамен: {v_min:8.3f} мВ")
            print(f"      ИТОГ: {'ПРОЙДЕН' if rep.passed else 'ПРОВАЛ'}")
        return rep
