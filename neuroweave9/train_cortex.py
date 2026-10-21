"""train_cortex.py — пайплайн обучения NEUROWEAVE v9.0 «Cortex».

Метод «одной большой партии» из Spiker (train_scale_original.py, доказан
в ARCHITECTURE_REVIEW §8): корпус целиком = один батч; Фаза 1 — изолированные
колонки с SLTT + EWC; Фаза 2 — ОДИН optimizer.step() декодера на эпоху
(backward() на каждом такте с весом 1/seq_len — тот же градиент, память
O(1)); сон с реплеем гиппокампа; генерация через BasalGangliaGate.

Отличие от Spiker: плоские зоны заменены ЛАМИНАРНЫМИ КОЛОНКАМИ v8, а
декодер дополнительно получает межполушарный VSA-фрейм (phase.py v8).

Запуск:
    python -m neuroweave9.train_cortex --max-stories 200 --holdout 40 \
        --phase1-epochs 4 --phase2-epochs 10 --max-vocab-size 128
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .cortex.organism import CortexOrganism
from .data import build_holdout, build_vocab_and_sequences, load_stories
from .learning.phase1 import Phase1ColumnTrainer
from .readout import LaplaceContextReadout


def column_a_logits(org: CortexOrganism, x: torch.Tensor) -> torch.Tensor:
    return org.step_isolated("A", x, learn=False)["logits"]


@torch.no_grad()
def evaluate_zone_a_loss(org: CortexOrganism, x_seq: torch.Tensor,
                         y_seq: torch.Tensor) -> float:
    """Контроль неразрушения памяти EWC после сна (лосс колонки A)."""
    org.set_phase(1)
    org.reset_state(x_seq.shape[1], x_seq.device)
    total = 0.0
    for t in range(x_seq.shape[0]):
        total += float(F.cross_entropy(column_a_logits(org, x_seq[t]),
                                       y_seq[t]).item())
    org.set_phase(2)
    return total / float(x_seq.shape[0])


@torch.no_grad()
def evaluate_split(org: CortexOrganism, readout: LaplaceContextReadout,
                   x_seq: torch.Tensor, y_seq: torch.Tensor) -> Tuple[float, float]:
    org.reset_state(x_seq.shape[1], x_seq.device)
    total, correct, count = 0.0, 0, 0
    for t in range(x_seq.shape[0]):
        org.oscillator.phase_reset(0.0)
        out = org.step(x_seq[t], learn=False, sync_metrics=False)
        cortical = torch.cat([out["left"]["zone_a"]["spikes"],
                              out["left"]["zone_b"]["spikes"],
                              out["right"]["zone_c"]["spikes"]], dim=-1)
        lap = org.laplace_memory.laplace_state.permute(1, 0, 2).reshape(
            x_seq.shape[1], -1)
        logits = readout(cortical, lap, out["left"]["zone_a"]["logits"],
                         out["sync"]["frame_real"])
        total += float(F.cross_entropy(logits, y_seq[t], reduction="sum").item())
        correct += int((logits.argmax(-1) == y_seq[t]).sum().item())
        count += int(y_seq[t].numel())
    return total / max(1, count), correct / max(1, count)


@torch.no_grad()
def generate_continuation(
    org: CortexOrganism,
    readout: LaplaceContextReadout,
    prompt_words: List[str],
    itos: List[str],
    stoi: Dict[str, int],
    num_tokens: int = 14,
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, Dict[str, float]]:
    """Оригинальный путь генерации: Лаплас + декодер, темп — базальные ганглии."""
    org.set_phase(2)
    org.reset_state(batch_size=1, device=device)
    vocab = len(itos)

    def make_input(word: str) -> torch.Tensor:
        x = torch.zeros((1, org.input_dim), device=device)
        x[0, stoi.get(word, 1)] = 1.0
        return x

    for w in prompt_words[:-1]:
        org.oscillator.phase_reset(0.0)
        org.step(make_input(w), learn=False, sync_metrics=False)

    out_words: List[str] = []
    bg_emissions, bg_ticks = 0, 0
    for _ in range(num_tokens):
        org.oscillator.phase_reset(0.0)
        last = prompt_words[-1] if not out_words else out_words[-1]
        out = org.step(make_input(last), learn=False, sync_metrics=False)
        cortical = torch.cat([out["left"]["zone_a"]["spikes"],
                              out["left"]["zone_b"]["spikes"],
                              out["right"]["zone_c"]["spikes"]], dim=-1)
        lap = org.laplace_memory.laplace_state.permute(1, 0, 2).reshape(1, -1)
        logits = readout(cortical, lap, out["left"]["zone_a"]["logits"],
                         out["sync"]["frame_real"])
        logits[:, :2] = -1e9

        org.basal_ganglia.reset_state(1, device=device)
        for _tick in range(5):
            bg_ticks += 1
            bg = org.basal_ganglia(
                logits,
                serotonin_5ht=out["left"]["zone_a"]["modulators"]["serotonin_5ht"],
                ach_level=out["left"]["zone_a"]["modulators"]["ach"])
            if float(bg["emit_mask"].item()) >= 1.0:
                bg_emissions += 1
                break
        out_words.append(itos[int(logits.argmax(-1).item())])

    text = ""
    for tok in [prompt_words[0]] + out_words:
        text += tok if tok in (".", ",", "!", "?", ";") else (" " if text else "") + tok
    return text, {"bg_emissions": float(bg_emissions),
                  "mean_ticks_to_decision": bg_ticks / max(1, num_tokens)}


def main() -> int:
    ap = argparse.ArgumentParser(description="NEUROWEAVE v9.0 Cortex — обучение")
    ap.add_argument("--data", default="data/TinyStories_train_4k.txt")
    ap.add_argument("--max-stories", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--max-vocab-size", type=int, default=128)
    ap.add_argument("--phase1-epochs", type=int, default=4)
    ap.add_argument("--phase2-epochs", type=int, default=10)
    ap.add_argument("--col-scale", type=float, default=1.0,
                    help="масштаб ламинарных колонок (1.0 ~ 64/48/64 зон Spiker)")
    ap.add_argument("--vsa-dim", type=int, default=512)
    ap.add_argument("--sleep-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--retain-graph", action="store_true",
                    help="буквальный оригинал: один backward на эпоху")
    ap.add_argument("--shuffle", action="store_true",
                    help="перемешивать истории в партии на каждой эпохе")
    ap.add_argument("--readout-chunks", type=int, default=1,
                    help="шагов оптимизатора декодера за эпоху (1 = оригинал)")
    ap.add_argument("--prior-weight", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    t_start = time.perf_counter()

    print("=" * 84)
    print("NEUROWEAVE v9.0 «CORTEX» — The Great Merge (одна большая партия)")
    print("=" * 84)
    print(f"[конфигурация] корпус={args.max_stories} | holdout={args.holdout} | "
          f"seq_len={args.seq_len} | словарь<={args.max_vocab_size} | "
          f"col_scale={args.col_scale} | Ф1x{args.phase1_epochs} Ф2x{args.phase2_epochs} "
          f"| {device}")

    stories = load_stories(args.data, args.max_stories + args.holdout)
    train_stories = stories[: args.max_stories]
    holdout_stories = stories[args.max_stories: args.max_stories + args.holdout]
    itos, stoi, x_seq, y_seq, chosen = build_vocab_and_sequences(
        train_stories, args.max_vocab_size, args.seq_len,
        batch_size=args.max_stories, device=device)
    vocab_size = len(itos)
    x_ho, y_ho, _chosen_ho = build_holdout(
        holdout_stories, stoi, itos, args.seq_len, args.holdout, device)
    print(f"[данные] партия {x_seq.shape[1]}x{x_seq.shape[0]} | holdout "
          f"{x_ho.shape[1]} | словарь {vocab_size} | топ-12: {itos[2:14]}")

    org = CortexOrganism(
        vocab_size=vocab_size, input_dim=vocab_size, scale=args.col_scale,
        vsa_dim=args.vsa_dim, seed=args.seed, device=device)
    n_neurons = sum(org.cfg.col(n).n_neurons() for n in org.names)
    print(f"[кора] колонки A/B/C (ламинарные): {n_neurons} нейронов, "
          f"кортикальная размерность {org.cortical_dim()}, "
          f"VSA-фрейм {args.vsa_dim}")

    target_counts = torch.bincount(y_seq.flatten(), minlength=vocab_size).float()
    log_prior = ((target_counts + 1.0) / (target_counts.sum() + vocab_size)).log().to(device)

    readout = LaplaceContextReadout(
        cortical_dim=org.cortical_dim(),
        laplace_dim=org.laplace_memory.num_scales * org.input_dim,
        vocab_size=vocab_size,
        sync_dim=args.vsa_dim,
        lr=0.025,
        log_prior=log_prior if args.prior_weight != 0.0 else None,
        prior_weight=args.prior_weight).to(device)

    # ---------------- Фаза 1 ----------------
    print(f"\n[Фаза 1] изолированные ламинарные колонки ({args.phase1_epochs} эпох)")
    trainer = Phase1ColumnTrainer(
        org, lr_left=0.020, lr_right=0.006,
        horizon_delta_t=3, lambda_ewc=100.0)
    t0 = time.perf_counter()
    for ep in range(1, args.phase1_epochs + 1):
        m = trainer.train_sequence_epoch(
            x_seq, y_seq, accumulate_fisher=(ep == args.phase1_epochs))
        if ep in (1, args.phase1_epochs // 2, args.phase1_epochs):
            print(f"    эпоха {ep:2d}: L_A={m['loss_zone_a']:.4f} "
                  f"L_B={m['loss_zone_b']:.4f} L_C={m['loss_zone_c']:.4f} "
                  f"({time.perf_counter() - t0:.1f} с)")
    trainer.ewc.finalize_phase1_and_freeze(num_samples=args.seq_len)
    org.set_phase(2)
    loss_a_before = evaluate_zone_a_loss(org, x_seq, y_seq)
    print(f"    EWC зафиксирован | эталонный L_A = {loss_a_before:.4f}")

    # ---------------- Фаза 2 (одна большая партия) ----------------
    print(f"\n[Фаза 2] сборка + Лаплас + VSA-синхронизация "
          f"({args.phase2_epochs} эпох, ОДИН step/эпоху)")
    org.basal_ganglia.base_threshold = 0.85
    batch_size = x_seq.shape[1]
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(args.seed)
    for ep in range(1, args.phase2_epochs + 1):
        org.reset_state(batch_size=batch_size, device=device)
        opt = readout.optimizer
        opt.zero_grad()
        if args.shuffle:
            perm = torch.randperm(batch_size, generator=gen).to(device)
            x_seq = x_seq[:, perm]
            y_seq = y_seq[:, perm]
        step_losses = []
        ep_acc = 0.0
        chunk = max(1, args.seq_len // max(1, args.readout_chunks))
        for t in range(args.seq_len):
            org.oscillator.phase_reset(0.0)
            reward_t = torch.full((batch_size, 1), 1.0, device=device)
            out = org.step(x_seq[t], reward=reward_t,
                           sync_metrics=(t == args.seq_len - 1))
            cortical = torch.cat([out["left"]["zone_a"]["spikes"],
                                  out["left"]["zone_b"]["spikes"],
                                  out["right"]["zone_c"]["spikes"]], dim=-1)
            lap = org.laplace_memory.laplace_state.permute(1, 0, 2).reshape(
                batch_size, -1)
            logits_t = readout(cortical, lap, out["left"]["zone_a"]["logits"],
                               out["sync"]["frame_real"])
            loss_t = F.cross_entropy(logits_t, y_seq[t])
            step_losses.append(loss_t)
            ep_acc += float((logits_t.argmax(-1) == y_seq[t]).float().mean())
            if not args.retain_graph:
                (loss_t / args.seq_len).backward()
                is_boundary = ((t + 1) % chunk == 0) or (t + 1 == args.seq_len)
                if is_boundary and args.readout_chunks > 1:
                    opt.step()
                    opt.zero_grad()
        total_ep_loss = torch.stack(step_losses).mean()
        if args.retain_graph:
            total_ep_loss.backward()
        total_ep_loss = float(total_ep_loss.item())
        if args.readout_chunks <= 1:
            opt.step()
        ep_acc /= float(args.seq_len)
        if ep in (1, args.phase2_epochs // 2, args.phase2_epochs) or ep % 5 == 0:
            ins_ce, ins_acc = evaluate_split(org, readout, x_seq, y_seq)
            ho_ce, ho_acc = evaluate_split(org, readout, x_ho, y_ho)
            print(f"    эпоха {ep:2d}: train CE={float(total_ep_loss):.4f} "
                  f"acc={ep_acc:.2%} | in-sample CE={ins_ce:.4f} acc={ins_acc:.2%} | "
                  f"HOLDOUT CE={ho_ce:.4f} acc={ho_acc:.2%} "
                  f"({time.perf_counter() - t0:.0f} с)")

    # ---------------- Сон и EWC ----------------
    print(f"\n[Сон] протокол v9 (реплей гиппокампа -> консолидация -> прунинг)")
    sleep_metrics = org.sleep_cycle()
    loss_a_after = evaluate_zone_a_loss(org, x_seq, y_seq)
    degradation = (loss_a_after - loss_a_before) / max(1e-6, abs(loss_a_before))
    print(f"    реплей={sleep_metrics.get('replay_steps', 0):.0f} "
          f"эпизодов={sleep_metrics.get('episodes', 0):.0f} "
          f"прунинг={sleep_metrics.get('pruned', 0):.0f} | "
          f"EWC: L_A {loss_a_before:.4f} -> {loss_a_after:.4f} "
          f"({degradation:+.2%})")
    trainer.ewc.rebaseline_after_sleep()

    # ---------------- Генерация ----------------
    print("\n[Генерация] Лаплас + декодер, темп — базальные ганглии")
    for tag, src in (("TRAIN", chosen), ("HOLDOUT", _chosen_ho)):
        if not src:
            continue
        prompt = src[0][:10]
        text, stats = generate_continuation(
            org, readout, prompt, itos, stoi,
            num_tokens=14, device=device)
        print(f"    [{tag:7s}] затравка: '{' '.join(prompt)}'")
        print(f"    [{tag:7s}] сеть:     '{text}'")
        print(f"    [{tag:7s}] эмиссий БГ={stats['bg_emissions']:.0f}, "
              f"решение={stats['mean_ticks_to_decision']:.1f} мс/слово")

    tel = org.telemetry()
    print(f"\n[телеметрия] sync_coherence={tel['sync_coherence']:.3f} "
          f"wm_occupancy={tel['wm_occupancy']:.3f} "
          f"hippo_stored={tel['hippo_stored']:.0f}")
    print("=" * 84)
    print(f"ГОТОВО за {time.perf_counter() - t_start:.1f} с ({device})")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
