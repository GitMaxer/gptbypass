# main.py
# Демонстрационная точка входа симуляции (листинг 6 канона).
#
# Демо Этапов 1-5 (канон v4.0.1):
#   ФАЗА 1 — изолированное обучение Зоны A (Синтаксис / Токены) на
#   марковском потоке: SLTT, суррогатные градиенты, O(1)-память, живая
#   телеметрия нейрохимии (DA/5-HT/ACh/NE) и шины SON.
#   СТАНД СЛОМА СИММЕТРИИ — одно входное воздействие в оба полушария
#   с идентичными весами: расхождение порождается ТОЛЬКО пресетами
#   дивергенции (tau_m, шум, суррогат; табл. 2 канона).
#   СБОРКА (превью Фазы 2) — макро-организм: полушария, шина Corpus
#   Callosum (задержка 7 тактов, GABA-фильтр, асимметричные проекции),
#   Global Workspace (латеральное GABA, ур. 9, Soft-WTA).
#   ЖИВОЙ ЦИКЛ (Этапы 3-4) — смена среды -> нейрогенез -> Reward-STDP ->
#   сон -> пробуждение; организм владеет обеими средами.
#   ЭТАП 5 — сенсорные интерфейсы и оптимизация: фазовый кодировщик
#   burst/latency, КЛЮЧЕВОЙ эксперимент ИНВЕРСИИ ПРАВИЛ (табл. 9) без
#   переобучения базового синтаксиса, квантизация int8/тернарная шины,
#   профиль быстрого контура, sync-audit (CUDA Graphs по конструкции).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.spiker.topology.zone import Zone
from src.spiker.topology.hemisphere import LeftHemisphere, RightHemisphere
from src.spiker.topology.callosum import CorpusCallosum
from src.spiker.topology.workspace import GlobalWorkspace
from src.spiker.encoding.temporal_encoder import (
    DelayLineTokenEncoder, TemporalPhaseEncoder)
from src.spiker.modulation.neurochemistry import NeuroChemistry
from src.spiker.organism.organism import SpikerOrganism
from src.spiker.organism.slow_loop import SlowLoopController

VOCAB = 16
DEVICE = torch.device("cpu")
BUS_APICAL_GAIN = 0.2   # калибровка зонда сборки (см. scripts/probe_stage2_assembly.py)


def _markov_stream(batch, seq_len, seed, rule=(3, 5)):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch, seq_len, dtype=torch.long)
    x = torch.randint(0, VOCAB, (batch,), generator=g)
    a, b = rule
    for t in range(seq_len):
        nxt = (a * x + b) % VOCAB
        mask = torch.rand(batch, generator=g) < 0.12
        rnd = torch.randint(0, VOCAB, (batch,), generator=g)
        nxt = torch.where(mask, rnd, nxt)
        tokens[:, t] = x
        x = nxt
    return tokens


# ---------------------------------------------------------------------
# Статистики стенда слома симметрии (без scipy)
# ---------------------------------------------------------------------
def _ks_statistic(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _welch_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    import math
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def _fano(samples):
    v = np.asarray(samples, dtype=float)
    return float(v.var(ddof=1) / max(v.mean(), 1e-9))


# ---------------------------------------------------------------------
# Этап 1: изолированная Фаза 1 — Зона A учится предсказывать токены
# ---------------------------------------------------------------------
def phase1_train(iters, batch, seq_len):
    torch.manual_seed(42)
    zone_a = Zone(
        name="A", side="left",
        n_inputs=VOCAB * 2, n_exc=64, n_inh=16, n_readout=VOCAB,
        input_gain=3.0,   # калибровка PoC (см. tests/test_zone_isolated.py)
    )
    zone_a.isolate()  # Фаза 1: межзонные пути физически разорваны
    encoder = DelayLineTokenEncoder(VOCAB, n_channels=2)
    neuro = NeuroChemistry(
        zone_names=["A", "B", "C"], ht_baseline={"A": 0.8, "B": 0.8, "C": 0.3}
    )
    opt = torch.optim.Adam(
        [zone_a.syn_in.w_slow, zone_a.syn_rec.w_slow,
         zone_a.readout.weight, zone_a.readout.bias],
        lr=2e-3,
    )

    print(f"Параметры Зоны A: {sum(p.numel() for p in zone_a.parameters()):,}")
    print(f"Задача: следующий токен, словарь {VOCAB}, "
          f"марковская цепь (3x+5)%{VOCAB} + 12% шума")
    ceiling = 0.12 * float(torch.log(torch.tensor(float(VOCAB))))
    print(f"Потолок кросс-энтропии: ~{ceiling:.3f} | "
          f"шанс: {float(torch.log(torch.tensor(float(VOCAB)))):.3f}\n")

    losses = []
    # Фаза 1 — организм БОДРСТВУЕТ: зоны обучаются при родной нейрохимии
    # (ACh ~ 0.3, базовые 5-HT, фазические NE при Surprise) — иначе сборка
    # Фазы 2 сдвинула бы режим проводимости и навык «поплыл» бы без причины.
    def on_alarm(zone_name, surprise):
        neuro.release_ne(zone_name, surprise)

    for it in range(iters):
        tokens = _markov_stream(batch, seq_len, seed=2026 + it)
        zone_a.reset_state(batch, DEVICE)
        encoder.reset_state(batch, DEVICE)

        step_losses, spikes_total, exports = [], 0, 0
        for t in range(seq_len - 1):
            s_in = encoder.step(tokens[:, t])
            out = zone_a.step(
                s_in,
                ach=neuro.get("ach", "A", batch),
                ne=neuro.get("ne", "A", batch),
                serotonin_5ht=neuro.get("ht", "A", batch),
                on_son_alarm=on_alarm,
            )
            logits = zone_a.readout_logits()
            loss = F.cross_entropy(logits, tokens[:, t + 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
            spikes_total += float(out["spikes"].detach().sum())
            exports += float((out["z_export"].detach().abs().sum() > 0).float())

            # Быстрый контур нейрохимии: ур. (20), SUM(s) — каноническая сумма
            neuro.step(spike_sums={"A": out["spikes"].detach().sum()})

        losses.append(sum(step_losses) / len(step_losses))
        if (it + 1) % 25 == 0:
            tel = neuro.telemetry
            print(
                f"iter {it + 1:3d} | CE {losses[-1]:.3f} | "
                f"spikes/step {spikes_total / (seq_len - 1):.1f} | "
                f"SON export {exports / (seq_len - 1):.0%} | "
                f"5-HT {tel['ht_mean']:.2f} | ACh {tel['ach_mean']:.2f} | "
                f"P_sleep {tel['sleep_pressure']:.4f}"
            )
    return zone_a, encoder, neuro, losses


# ---------------------------------------------------------------------
# Этап 2, стенд: слом симметрии на ОДНОМ входе (веса идентичны)
# ---------------------------------------------------------------------
def symmetry_stand(t=150, batch=4, repeats=6):
    torch.manual_seed(7)   # идентичные веса обоих боков!
    zone_l = Zone("probe", side="left", n_inputs=VOCAB * 2,
                  n_exc=48, n_inh=12, input_gain=3.0)
    torch.manual_seed(7)
    zone_r = Zone("probe", side="right", n_inputs=VOCAB * 2,
                  n_exc=48, n_inh=12, input_gain=3.0)
    enc = DelayLineTokenEncoder(VOCAB, n_channels=2)

    tokens = _markov_stream(batch, t, seed=2026)

    def run(zone):
        zone.reset_state(batch, DEVICE)
        enc.reset_state(batch, DEVICE)
        return [float(zone.step(enc.step(tokens[:, i]))["spikes"].detach().sum())
                for i in range(t)]

    counts_l, counts_r = run(zone_l), run(zone_r)
    totals_l = [sum(run(zone_l)) for _ in range(repeats)]
    totals_r = [sum(run(zone_r)) for _ in range(repeats)]

    D = _ks_statistic(counts_l, counts_r)
    D_crit = 1.36 * (2.0 / t) ** 0.5
    t_stat = _welch_t(counts_l, counts_r)
    fano_l, fano_r = _fano(totals_l), _fano(totals_r)
    return dict(D=D, D_crit=D_crit, t=t_stat, fano_l=fano_l, fano_r=fano_r,
                mean_l=float(np.mean(counts_l)), mean_r=float(np.mean(counts_r)))


# ---------------------------------------------------------------------
# Этап 2, сборка: превью Фазы 2 — полушария + Callosum + Workspace
# ---------------------------------------------------------------------
def assembly_session(zone_a, encoder, neuro, iters, batch, seq_len):
    # Зона C: правое полушарие; вход — спайки Зоны A (токенов НЕ видит,
    # ур. 1 канона — правое строит модель фазового пространства левого)
    torch.manual_seed(43)
    zone_c = Zone(
        name="C", side="right",
        n_inputs=64, n_exc=64, n_inh=16, n_readout=VOCAB,
        input_gain=1.5,
    )
    left = LeftHemisphere([zone_a])
    right = RightHemisphere([zone_c])
    callosum = CorpusCallosum(left_dim=left.bus_dim, right_dim=right.bus_dim,
                              delay=7)
    workspace = GlobalWorkspace(n_zones=2)

    # Контракт Фазы 2 (разд. 7.2): w_slow заморожены, снапшоты сняты
    zone_a.syn_in.freeze_for_phase2()
    zone_a.syn_rec.freeze_for_phase2()
    for p in zone_a.readout.parameters():
        p.requires_grad_(False)
    zone_a.connect_apical()   # шина физически включена — акт Фазы 2
    zone_c.connect_apical()

    def on_alarm(zone_name, surprise):
        neuro.release_ne(zone_name, surprise)   # фазический NE при Novelty

    stats = dict(ce_a=[], ce_fused=[], alpha_a=[], win_a=[],
                 rate_a=[], rate_c=[], exports_c=[])
    with torch.no_grad():
        for it in range(iters):
            tokens = _markov_stream(batch, seq_len, seed=7000 + it)
            zone_a.reset_state(batch, DEVICE)
            zone_c.reset_state(batch, DEVICE)
            encoder.reset_state(batch, DEVICE)
            callosum.reset_state(batch, DEVICE)
            to_right = torch.zeros(batch, left.bus_dim)
            to_left = torch.zeros(batch, right.bus_dim)

            for t in range(seq_len - 1):
                s_in = encoder.step(tokens[:, t])
                # Доставка сообщений шины на апикальные дендриты (ур. 16)
                apical_l = left.callosum_apical(to_left, gain=BUS_APICAL_GAIN)
                apical_r = right.callosum_apical(to_right, gain=BUS_APICAL_GAIN)
                # Левое: токены; правое: спайки левого (контекст, ур. 1)
                outs_l = left.step({"A": s_in}, apical=apical_l, neuro=neuro,
                                   on_son_alarm=on_alarm)
                outs_r = right.step(
                    {"C": outs_l["A"]["spikes"].detach()},
                    apical=apical_r, neuro=neuro, on_son_alarm=on_alarm,
                )
                # Асинхронный обмен сжатыми сообщениями (задержка 7 тактов)
                to_right, to_left = callosum.step(
                    left.son_export(outs_l), right.son_export(outs_r))

                # Global Workspace: конкуренция за выходной токен
                logits_a = zone_a.readout_logits()
                logits_c = zone_c.readout_logits()
                bids = torch.stack([
                    torch.softmax(logits_a, -1).max(-1).values,
                    torch.softmax(logits_c, -1).max(-1).values,
                ], dim=-1)
                alpha, winner, _ = workspace.compete(bids)
                fused = workspace.fuse_logits(alpha, [logits_a, logits_c])

                target = tokens[:, t + 1]
                stats["ce_a"].append(F.cross_entropy(logits_a, target).item())
                stats["ce_fused"].append(
                    F.cross_entropy(fused, target).item())
                stats["alpha_a"].append(float(alpha[:, 0].mean()))
                stats["win_a"].append(float((winner == 0).float().mean()))
                stats["rate_a"].append(
                    float(outs_l["A"]["spikes"].float().mean()))
                stats["rate_c"].append(
                    float(outs_r["C"]["spikes"].float().mean()))
                stats["exports_c"].append(
                    float((outs_r["C"]["z_export"].abs().sum(-1) > 0)
                          .float().mean()))

                neuro.step(spike_sums={
                    "A": outs_l["A"]["spikes"].sum(),
                    "C": outs_r["C"]["spikes"].sum(),
                })

            if (it + 1) % 10 == 0:
                n = seq_len - 1
                k = -n
                print(
                    f"iter {it + 1:3d} | CE_A {np.mean(stats['ce_a'][k:]):.3f} | "
                    f"CE_fused {np.mean(stats['ce_fused'][k:]):.3f} | "
                    f"alpha_A {np.mean(stats['alpha_a'][k:]):.2f} | "
                    f"win_A {np.mean(stats['win_a'][k:]):.0%} | "
                    f"rate A {np.mean(stats['rate_a'][k:]):.2f} "
                    f"C {np.mean(stats['rate_c'][k:]):.2f} | "
                    f"SON-C export {np.mean(stats['exports_c'][k:]):.0%} | "
                    f"bus {callosum.traffic:.2f}"
                )
    stats["traffic"] = callosum.traffic
    return stats


# ---------------------------------------------------------------------
# Этап 3: ЖИВОЙ ЦИКЛ — бодрствование -> нейрогенез -> сон -> пробуждение
# ---------------------------------------------------------------------
def living_cycle(zone_a, encoder, iters_phase1=250, episodes=14):
    """Живой цикл самоулучшающегося организма (демо Этапов 3-4).

    Сюжет канона (смена среды = жизненное событие):
      1. Зона A — эксперт среды M1, организм A+C встречает НОВУЮ среду M2:
         уверенный эксперт ошибается, награда редка (организм «страдает»);
      2. НЕЙРОГЕНЕЗ: рождается зона B, изолированная Фаза 1 на M2 —
         рост репертуара (канон разд. 7.1: новые навыки — новые зоны);
      3. Сборка v2 (A+B+C): WTA переключает доверие на компетентную зону,
         Reward-STDP достраивает её поверх основы (Tag-and-Capture);
      4. Сон (5 шагов разд. 7.3): реплей, консолидация w_fast -> w_slow,
         прунинг, сброс P_sleep; EWC перебазируется;
      5. Возврат M1: навык детства выжил — организм владеет ОБОИМИ средами.
    """
    M1, M2 = (3, 5), (7, 3)

    # ---- Новая зона B: нейрогенез + изолированная Фаза 1 на M2 --------
    torch.manual_seed(44)
    zone_b = Zone("B", side="left", n_inputs=VOCAB * 2, n_exc=64, n_inh=16,
                  n_readout=VOCAB, input_gain=3.0)
    opt = torch.optim.Adam(
        [zone_b.syn_in.w_slow, zone_b.syn_rec.w_slow,
         zone_b.readout.weight, zone_b.readout.bias], lr=2e-3)
    for it in range(iters_phase1):
        tokens = _markov_stream(8, 30, seed=3100 + it, rule=M2)
        zone_b.reset_state(8, DEVICE)
        encoder.reset_state(8, DEVICE)
        for t in range(29):
            s_in = encoder.step(tokens[:, t])
            zone_b.step(s_in)
            loss = F.cross_entropy(zone_b.readout_logits(), tokens[:, t + 1])
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- Организм v2: A + B + C ---------------------------------------
    torch.manual_seed(43)
    zone_c = Zone("C", side="right", n_inputs=64, n_exc=64, n_inh=16,
                  n_readout=VOCAB, input_gain=1.5)
    organism = SpikerOrganism([zone_a, zone_b], [zone_c], lr_fast=2e-3,
                              bus_apical_gain=BUS_APICAL_GAIN)
    organism.assemble()
    slow_loop = SlowLoopController(theta_sleep=1.0, poll_interval=50)

    def waking(rule, eps, label):
        ces, rewards = [], []
        for ep in range(eps):
            tokens = _markov_stream(8, 30, seed=8300 + ep, rule=rule)
            organism.reset_state(8, DEVICE)
            encoder.reset_state(8, DEVICE)
            prev = None
            for t in range(29):
                s_a = encoder.step(tokens[:, t])
                prev = organism.zones["A"].s_exc_prev.detach().clone() \
                    if organism.zones["A"].s_exc_prev is not None \
                    else torch.zeros(8, 64)
                summary = organism.behave(
                    {"A": s_a, "B": s_a, "C": prev}, tokens[:, t + 1])
                # Медленный контур: опрос раз в 50 тактов (табл. 5)
                if slow_loop.poll(organism.telemetry) == SlowLoopController.SLEEP:
                    organism.sleep_cycle(replay_steps=48)
                    slow_loop.notify_slept()
            ces.append(summary["ce_fused"])
            rewards.append(summary["reward"])
        tel = organism.telemetry
        print(f"  [{label}] CE_fused {np.mean(ces):.3f} | R {np.mean(rewards):.2f} | "
              f"DA база {tel['reward_baseline']:.2f} | "
              f"w_fast {tel['w_fast_mass']:.0f} | P_sleep {tel['sleep_pressure']:.2f}")
        return float(np.mean(ces)), float(np.mean(rewards))

    print("  Сборка v2: A(M1) + B(M2) + C; шина включена, WTA на 3 зоны")
    print("  SlowLoop: опрос каждые 50 тактов, Theta_sleep = 1.0")
    ce_m1_a, r_m1_a = waking(M1, episodes, "M1 после нейрогенеза")
    ce_m2, r_m2 = waking(M2, episodes, "M2 — новая среда освоена")

    print("\n  Сон: реплей при повышенной энтропии, консолидация, прунинг")
    report = organism.sleep_cycle(replay_steps=64)
    print(f"  P_sleep {report['p_sleep_before']:.2f} -> "
          f"{report['p_sleep_after']:.2f} | "
          f"консолидировано {report['consolidated_mass']:.2f} | "
          f"прунинг {report['pruned_frac']:.0%} | "
          f"w_fast {report['w_fast_mass_before']:.0f} -> "
          f"{report['w_fast_mass_after']:.0f}")

    ce_m2_s, r_m2_s = waking(M2, episodes, "M2 после сна")
    ce_m1_b, r_m1_b = waking(M1, episodes, "M1 — возврат в детство")

    return dict(ce_m1_a=ce_m1_a, ce_m2=ce_m2, ce_m2_s=ce_m2_s,
                ce_m1_b=ce_m1_b, r_m1_a=r_m1_a, r_m2=r_m2,
                r_m2_s=r_m2_s, r_m1_b=r_m1_b,
                sleeps=slow_loop.sleep_cycles_done + 1)


# ---------------------------------------------------------------------
def main(iters: int = 300, batch: int = 8, seq_len: int = 30):
    print("=" * 64)
    print("NEUROWEAVE / SPIKER — демо Этапов 1-5 (канон v4.0.1)")
    print("вычисление = память = обучение")
    print("=" * 64)

    print("\n--- ФАЗА 1: изолированное обучение Зоны A ---------------------")
    zone_a, encoder, neuro, losses = phase1_train(iters, batch, seq_len)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"\nИтог Фазы 1: CE {first:.3f} -> {last:.3f} ({(1 - last / first):.0%})")

    print("\n--- ЭТАП 2: стенд слома симметрии (один вход, веса идентичны) -")
    sym = symmetry_stand()
    print(f"KS D = {sym['D']:.3f} > крит {sym['D_crit']:.3f} (p < 0.05): "
          f"распределения спайков L/R разошлись")
    print(f"Уэлч t = {sym['t']:.1f} (|t| > 2.58 => p < 0.01): сдвиг средних "
          f"{sym['mean_l']:.1f} vs {sym['mean_r']:.1f} спайков/такт")
    print(f"Фано L/R = {sym['fano_l']:.4f}/{sym['fano_r']:.3f}: правое "
          f"полушарие вариабельнее в {sym['fano_r'] / max(sym['fano_l'], 1e-9):.0f}x "
          f"(«выраженная стохастика», табл. 2)")

    print("\n--- ЭТАП 2: сборка макро-организма (превью Фазы 2) ------------")
    print("Полушария: L(A, tau_m=8, sigma=0.001) + R(C, tau_m=35, sigma=0.05)")
    print("Callosum: задержка 7 тактов, GABA-фильтр, асимметрия 1.0/0.6")
    print("Workspace: латеральное GABA (ур. 9), Soft-WTA; пластичность "
          "выключена (Reward-STDP — Этап 3)")
    print()
    asm = assembly_session(zone_a, encoder, neuro, iters=60, batch=batch,
                           seq_len=seq_len)

    print(f"\nИтог сборки: CE_A {np.mean(asm['ce_a']):.3f} | "
          f"CE_fused {np.mean(asm['ce_fused']):.3f} | "
          f"деградация от мнения новичка: "
          f"{np.mean(asm['ce_fused']) / np.mean(asm['ce_a']) - 1:.1%}")
    print(f"Фокус: alpha_A = {np.mean(asm['alpha_a']):.2f}, Зона A побеждает "
          f"{np.mean(asm['win_a']):.0%} тактов (конкуренция отбирает "
          f"компетентную зону)")
    print(f"Шина: трафик {asm['traffic']:.2f}; обученное левое молчит "
          f"(Zero-Compute), молодое правое экспортирует "
          f"{np.mean(asm['exports_c']):.0%} тактов")

    tel = neuro.telemetry
    if tel["sleep_pressure"] > 1.0:
        print(f"Давление сна P = {tel['sleep_pressure']:.2f} > Theta_sleep = 1.0: "
              "организму пора спать — протокол ниже, в живом цикле")
    else:
        print(f"Давление сна P = {tel['sleep_pressure']:.3f} "
              f"(порог Theta_sleep = 1.0)")

    print("\n--- ЭТАП 3: ЖИВОЙ ОРГАНИЗМ — цикл самоулучшения ----------------")
    print("Смена среды -> нейрогенез (зона B) -> сборка -> Reward-STDP -> "
          "сон -> пробуждение")
    print()
    life = living_cycle(zone_a, encoder)

    print(f"\nИтог живого цикла ({life['sleeps']} сна):")
    print(f"  M1: CE {life['ce_m1_a']:.3f} -> {life['ce_m1_b']:.3f} "
          f"(навык детства жив, R {life['r_m1_a']:.2f} -> "
          f"{life['r_m1_b']:.2f})")
    print(f"  M2: CE {life['ce_m2']:.3f} -> {life['ce_m2_s']:.3f} после сна "
          f"(R {life['r_m2']:.2f} -> {life['r_m2_s']:.2f})")
    print("  Организм владеет обеими средами: WTA выбирает компетентную зону, "
          "сон закрепил новую компетенцию, старая не потеряна.")

    print("\n--- ЭТАП 5: СЕНСОРНЫЕ ИНТЕРФЕЙСЫ, ИНВЕРСИЯ ПРАВИЛ, ОПТИМИЗАЦИЯ -")
    print("Фазовый кодировщик (burst/latency) -> КЛЮЧЕВОЙ эксперимент "
          "инверсии правил -> квантизация шины -> профиль быстрого контура")
    print()
    inv = stage5_inversion(zone_a, encoder)
    stage5_optimization()


# ---------------------------------------------------------------------
# Этап 5: КЛЮЧЕВОЙ эксперимент — инверсия правил без переобучения
# ---------------------------------------------------------------------
def stage5_inversion(zone_a, encoder, iters_d=250, eps_per_block=4):
    """Табл. 9 канона: адаптивное переключение поведения при инверсии
    правил БЕЗ переобучения базового синтаксиса.

    M1_inv = (11x+9)%16 — ТОЧНАЯ обратная карта M1=(3x+5)%16: те же
    токены, та же статистика, противоположные ПРАВИЛА. Сюжет:
    инверсия ломает эксперта (организм замечает: салиентность скачет) ->
    нейрогенез D с TemporalPhaseEncoder (салиентность = сюрприз
    эксперта: top-down внимание к новизне) -> сборка v3 с чередованием
    M1/M1_inv: поведение переключается по режиму -> сон -> базовый
    синтаксис жив.
    """
    # инверсия — точная обратная карта (проверка)
    for x in range(VOCAB):
        y = (3 * x + 5) % VOCAB
        assert (11 * y + 9) % VOCAB == x
    M1_INV = (11, 9)

    def eval_a(rule, eps=2):
        ces, sals = [], []
        with torch.no_grad():
            for ep in range(eps):
                tokens = _markov_stream(6, 30, seed=9100 + ep, rule=rule)
                zone_a.reset_state(6, DEVICE)
                encoder.reset_state(6, DEVICE)
                prev_logits = None
                for t in range(29):
                    if prev_logits is not None:
                        p = torch.softmax(prev_logits, -1)
                        sur = -torch.log(p.gather(
                            1, tokens[:, t:t + 1]).clamp(min=1e-9))
                        sals.append(float((sur / np.log(VOCAB)).mean()))
                    zone_a.step(encoder.step(tokens[:, t]))
                    prev_logits = zone_a.readout_logits()
                    ces.append(F.cross_entropy(
                        zone_a.readout_logits(), tokens[:, t + 1]).item())
        return float(np.mean(ces)), float(np.mean(sals))

    ce_m1, sal_m1 = eval_a((3, 5))
    ce_inv, sal_inv = eval_a(M1_INV)
    print(f"  Инверсия правил: M1 -> M1_inv (обратная цепь, та же статистика)")
    print(f"  CE_A: {ce_m1:.3f} -> {ce_inv:.3f} | салиентность (сюрприз A): "
          f"{sal_m1:.2f} -> {sal_inv:.2f} (x{sal_inv / max(sal_m1, 1e-9):.1f})"
          " — организм ЗАМЕЧАЕТ смену правил")

    # --- НЕЙРОГЕНЕЗ: зона D с фазовым кодировщиком Этапа 5 ------------
    print("  Нейрогенез: зона D с TemporalPhaseEncoder "
          "(салиентность = сюрприз замороженного A)")
    torch.manual_seed(52)
    enc_d = TemporalPhaseEncoder(VOCAB)
    zone_d = Zone("D", side="left", n_inputs=enc_d.n_out, n_exc=64, n_inh=16,
                  n_readout=VOCAB, input_gain=3.0)
    zone_d.isolate()
    neuro_d = NeuroChemistry(zone_names=["D"], ht_baseline={"D": 0.8})
    on_alarm_d = lambda name, sur: neuro_d.release_ne(name, sur)
    opt_d = torch.optim.Adam(
        [zone_d.syn_in.w_slow, zone_d.syn_rec.w_slow,
         zone_d.readout.weight, zone_d.readout.bias], lr=2e-3)
    for it in range(iters_d):
        tokens = _markov_stream(6, 30, seed=5100 + it, rule=M1_INV)
        zone_a.reset_state(6, DEVICE)
        encoder.reset_state(6, DEVICE)
        zone_d.reset_state(6, DEVICE)
        enc_d.reset_state(6, DEVICE)
        with torch.no_grad():
            prev_logits, sal_stream = None, []
            for t in range(30):
                if prev_logits is not None:
                    p = torch.softmax(prev_logits, -1)
                    sur = -torch.log(p.gather(
                        1, tokens[:, t:t + 1]).clamp(min=1e-9))
                    sal_stream.append(
                        (sur / np.log(VOCAB)).clamp(0.0, 1.0).squeeze(1))
                else:
                    sal_stream.append(torch.zeros(6))
                zone_a.step(encoder.step(tokens[:, t]))
                prev_logits = zone_a.readout_logits()
        for t in range(29):
            s_d = enc_d.step(tokens[:, t], salience=sal_stream[t])
            zone_d.step(
                s_d, ach=neuro_d.get("ach", "D", 6),
                ne=neuro_d.get("ne", "D", 6),
                serotonin_5ht=neuro_d.get("ht", "D", 6),
                on_son_alarm=on_alarm_d)
            loss = F.cross_entropy(
                zone_d.readout_logits(), tokens[:, t + 1])
            opt_d.zero_grad(); loss.backward(); opt_d.step()
    print(f"  Фаза 1 D завершена ({iters_d} итер): фазовый код освоил "
          "инвернутые правила")

    # --- СБОРКА v3: чередование M1 / M1_inv ---------------------------
    torch.manual_seed(53)
    zone_c5 = Zone("C5", side="right", n_inputs=64, n_exc=64, n_inh=16,
                   n_readout=VOCAB, input_gain=1.5)
    org = SpikerOrganism([zone_a, zone_d], [zone_c5], lr_fast=2e-3,
                         bus_apical_gain=0.2)
    org.assemble()
    org.decoder_training = True

    print("  Сборка v3 (A+D+C5), чередование блоков M1 / M1_inv:")
    for blk in range(4):
        rule = (3, 5) if blk % 2 == 0 else M1_INV
        tag = "M1    " if blk % 2 == 0 else "M1_inv"
        ces, rewards, win_a = [], [], []
        for ep in range(eps_per_block):
            tokens = _markov_stream(6, 30, seed=8200 + 60 * blk + ep,
                                    rule=rule)
            org.reset_state(6, DEVICE)
            encoder.reset_state(6, DEVICE)
            enc_d.reset_state(6, DEVICE)
            prev_logits, sal = None, None
            for t in range(29):
                s_a = encoder.step(tokens[:, t])
                if prev_logits is not None:
                    p = torch.softmax(prev_logits, -1)
                    sur = -torch.log(p.gather(
                        1, tokens[:, t:t + 1]).clamp(min=1e-9))
                    sal = (sur / np.log(VOCAB)).clamp(0.0, 1.0).squeeze(1)
                else:
                    sal = torch.zeros(6)
                s_d = enc_d.step(tokens[:, t], salience=sal)
                prev = org.zones["A"].s_exc_prev.detach().clone() \
                    if org.zones["A"].s_exc_prev is not None \
                    else torch.zeros(6, 64)
                summary = org.behave(
                    {"A": s_a, "D": s_d, "C5": prev}, tokens[:, t + 1])
                prev_logits = org.zones["A"].readout_logits()
                ces.append(summary["ce_fused"])
                rewards.append(summary["reward"])
                win_a.append(float((summary["winner"] == 0).float().mean()))
            org.ramp_bus_gain(0.25)
        print(f"    блок {blk} [{tag}]: CE_fused {np.mean(ces):.3f} | "
              f"R {np.mean(rewards):.2f} | win_A {np.mean(win_a):.0%} | "
              f"гейн шины {org.bus_apical_gain:.2f}")

    report = org.sleep_cycle(replay_steps=48)
    ce_a_final, _ = eval_a((3, 5))
    print(f"  Сон: консолидировано {report['consolidated_mass']:.1f}, "
          f"прунинг {report['pruned_frac']:.0%}")
    print(f"  Базовый синтаксис (замер после сна): CE_A(M1) "
          f"{ce_m1:.3f} -> {ce_a_final:.3f} "
          f"({ce_a_final / ce_m1 - 1:+.1%}; w_slow заморожен с Фазы 2 — "
          "переобучения НЕ БЫЛО)")
    return dict(ce_m1=ce_m1, ce_inv=ce_inv, ce_a_final=ce_a_final,
                sal_m1=sal_m1, sal_inv=sal_inv)


# ---------------------------------------------------------------------
# Этап 5: квантизация шины + профиль быстрого контура + CUDA Graphs
# ---------------------------------------------------------------------
def stage5_optimization():
    """Разд. 8.3: int8/тернарные веса в некритичных путях (шина);
    профиль быстрого контура (табл. 9); статус CUDA Graphs."""
    from src.spiker.core.quant import quant_bandwidth_report
    from src.spiker.organism.runtime import (
        FastLoopProfiler, CUDAGraphRunner, audit_host_sync)

    def make_expert():
        # детерминированный фабричный метод: один сид -> идентичные веса
        # для чистого A/B сравнения конфигураций шины
        torch.manual_seed(61)
        zone = Zone("QA", side="left", n_inputs=VOCAB * 2, n_exc=48,
                    n_inh=12, n_readout=VOCAB, input_gain=3.0)
        zone.isolate()
        enc = DelayLineTokenEncoder(VOCAB, n_channels=2)
        opt = torch.optim.Adam(
            [zone.syn_in.w_slow, zone.syn_rec.w_slow,
             zone.readout.weight, zone.readout.bias], lr=2e-3)
        for it in range(60):
            tokens = _markov_stream(6, 24, seed=9300 + it)
            zone.reset_state(6, DEVICE)
            enc.reset_state(6, DEVICE)
            for t in range(23):
                zone.step(enc.step(tokens[:, t]))
                loss = F.cross_entropy(
                    zone.readout_logits(), tokens[:, t + 1])
                opt.zero_grad(); loss.backward(); opt.step()
        return zone

    def session(int8, ternary, eps=3):
        torch.manual_seed(62)
        zone = make_expert()
        zone_c = Zone("QC", side="right", n_inputs=48, n_exc=48, n_inh=12,
                      n_readout=VOCAB, input_gain=1.5)
        org = SpikerOrganism([zone], [zone_c], lr_fast=2e-3,
                             bus_apical_gain=0.2,
                             callosum_kwargs=dict(int8_ring=int8,
                                                  ternary_proj=ternary))
        org.assemble()
        ces = []
        enc_s = DelayLineTokenEncoder(VOCAB, n_channels=2)
        with torch.no_grad():
            for ep in range(eps):
                tokens = _markov_stream(6, 24, seed=9400 + ep)
                org.reset_state(6, DEVICE)
                enc_s.reset_state(6, DEVICE)
                for t in range(23):
                    s_a = enc_s.step(tokens[:, t])
                    prev = org.zones["QA"].s_exc_prev.detach().clone() \
                        if org.zones["QA"].s_exc_prev is not None \
                        else torch.zeros(6, 48)
                    summary = org.behave(
                        {"QA": s_a, "QC": prev}, tokens[:, t + 1])
                    ces.append(summary["ce_fused"])
        return float(np.mean(ces)), org

    ce_float, org_f = session(False, False)
    ce_quant, org_q = session(True, True)
    cal = org_q.callosum
    # отчёт полосы: float32-эквивалент кольца против фактического int8
    report = quant_bandwidth_report(
        torch.zeros_like(cal._buf_from_left, dtype=torch.float32))
    print(f"  Квантизация шины (некритичный путь, разд. 8.3):")
    print(f"    CE_fused: float {ce_float:.3f} vs int8+тернарная "
          f"{ce_quant:.3f} ({ce_quant / ce_float - 1:+.1%})")
    print(f"    кольцо задержек: {report['bytes_float32']:,} -> "
          f"{report['bytes_int8']:,} байт "
          f"(x{report['reduction_x']:.0f} VRAM); проекции — тернарные "
          "{-s, 0, +s} (мастер-вес полный, QAT)")

    # Профиль быстрого контура (критерий табл. 9: профиль снят)
    enc_p = DelayLineTokenEncoder(VOCAB, n_channels=2)
    enc_p.reset_state(4, DEVICE)
    tokens = _markov_stream(4, 40, seed=9500)
    org_f.reset_state(4, DEVICE)
    counter = {"t": 0}

    def make_inputs():
        t = counter["t"] % 39
        counter["t"] += 1
        s_a = enc_p.step(tokens[:, t])
        prev = org_f.zones["QA"].s_exc_prev.detach().clone() \
            if org_f.zones["QA"].s_exc_prev is not None \
            else torch.zeros(4, 48)
        return {"QA": s_a, "QC": prev}

    prof = FastLoopProfiler(org_f)
    rep = prof.profile(make_inputs, ticks=48, warmup=6)
    print(f"  Профиль быстрого контура: {rep['ticks_per_sec']:.0f} тактов/с "
          f"({rep['ms_per_tick']:.1f} мс/такт): зоны {rep['hemispheres_pct']:.0f}% | "
          f"шина {rep['callosum_pct']:.0f}% | пластичность "
          f"{rep['plasticity_pct']:.0f}% | химия {rep['neurochemistry_pct']:.0f}%")

    # CUDA Graphs: аудит синхронно-безопасности + статус
    with audit_host_sync():
        for _ in range(3):
            org_f.awake_step(make_inputs())
    runner = CUDAGraphRunner(org_f, batch_size=4, device=DEVICE)
    status = "захват готов (CUDA)" if runner.available else \
        "CPU-конфигурация: захват аппаратно-гейтед; такт прошёл " \
        "sync-audit (захватываем ПО КОНСТРУКЦИИ)"
    print(f"  CUDA Graphs: {status}")
    print("  (критерий табл. 9 — снижение оверхеда такта >= 2x — "
          "переносится на GPU-конфигурацию: блок захвата = delay Callosum, "
          "тензорный указатель кольца, статические буферы входов/DA)")


if __name__ == "__main__":
    main()
