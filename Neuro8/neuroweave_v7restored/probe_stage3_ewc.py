# Зонд Этапа 3: масштаб EWC. Диагональный эмпирический Фишер от CE-градиентов
# даёт F ~ 1e-6..1e-4: штраф lam*F*dw^2 микроскопичен против CE ~ 1.5 при
# любых разумных lam. Решение: нормировка F/max(F) (относительная важность
# весов) + умеренный lam. Зонд подбирает рабочую точку для живого теста
# «деградация задачи 1 <= 5% при дообучении задачи 2».
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "download" / "Spiker"))

import numpy as np
import torch
import torch.nn.functional as F

from src.spiker.topology.zone import Zone
from src.spiker.encoding.temporal_encoder import DelayLineTokenEncoder
from src.spiker.learning.ewc import EWCRegularizer

VOCAB = 8
DEVICE = torch.device("cpu")
M1 = (3, 5)
M2 = (3, 1)   # анти-задача: (3x+5) и (3x+1) не пересекаются mod 8 —
              # каждая пара переходов переворачивается, конфликт максимален


def markov_stream(batch, seq_len, seed, rule):
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


def make_zone(seed=42):
    torch.manual_seed(seed)
    # Критически малая ёмкость (8 нейронов, 8 классов): представления
    # анти-задачи ПЕРЕЗАПИСЫВАЮТ представления задачи 1
    return Zone("A", side="left", n_inputs=VOCAB * 2, n_exc=8, n_inh=2,
                n_readout=VOCAB, input_gain=4.0)


def make_perm(seed=77):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(VOCAB * 2, generator=g)


PERM = make_perm()


def run_stream(zone, encoder, tokens, train=False, opt=None, perm=None):
    batch, seq_len = tokens.shape
    zone.reset_state(batch, DEVICE)
    encoder.reset_state(batch, DEVICE)
    losses = []
    for t in range(seq_len - 1):
        s_in = encoder.step(tokens[:, t])
        if perm is not None:
            s_in = s_in[:, perm]
        zone.step(s_in)
        loss = F.cross_entropy(zone.readout_logits(), tokens[:, t + 1])
        if train and opt is not None:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def ewc_loss_fn(zone, encoder):
    def fn():
        tokens = markov_stream(6, 16, seed=int(torch.randint(0, 10_000, (1,))),
                               rule=M1)
        zone.reset_state(6, DEVICE)
        encoder.reset_state(6, DEVICE)
        total = 0.0
        for t in range(15):
            s_in = encoder.step(tokens[:, t])
            zone.step(s_in)
            total = total + F.cross_entropy(zone.readout_logits(),
                                            tokens[:, t + 1])
        return total / 15.0
    return fn


def train_task(zone, encoder, rule, iters=300, ewc=None, lam=0.0, perm=None,
              noise_target=False):
    params = [zone.syn_in.w_slow, zone.syn_rec.w_slow,
              zone.readout.weight, zone.readout.bias]
    if ewc is not None:
        for p in params:
            p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=5e-3)
    for it in range(iters):
        tokens = markov_stream(6, 16, seed=2026 + it, rule=rule)
        if noise_target:
            g = torch.Generator().manual_seed(5000 + it)
            tokens = torch.cat([
                tokens[:, :1],
                torch.randint(0, VOCAB, (6, 15), generator=g)],
                dim=1)
        zone.reset_state(6, DEVICE)
        encoder.reset_state(6, DEVICE)
        opt.zero_grad()
        total = 0.0
        for t in range(15):
            s_in = encoder.step(tokens[:, t])
            if perm is not None:
                s_in = s_in[:, perm]
            zone.step(s_in)
            total = total + F.cross_entropy(zone.readout_logits(),
                                            tokens[:, t + 1])
        loss = total / 15.0
        if ewc is not None and ewc.is_armed:
            loss = loss + lam * ewc.penalty()
        loss.backward()
        opt.step()
    return zone


def main():
    encoder = DelayLineTokenEncoder(VOCAB, n_channels=2)

    # База: задача 1
    zone = make_zone()
    train_task(zone, encoder, M1, iters=300)
    ce1_base = np.mean([run_stream(zone, encoder,
                                   markov_stream(6, 16, 9100 + i, M1))
                        for i in range(4)])
    print(f"CE(задача1) после Фазы 1: {ce1_base:.3f} (шанс {np.log(VOCAB):.3f})")

    # Контроль: шумовая перезапись (случайные цели — дрейф градиентов
    # вымывает структуру задачи 1) БЕЗ защиты
    zone_ctrl = make_zone()
    zone_ctrl.load_state_dict(zone.state_dict())
    train_task(zone_ctrl, encoder, M1, iters=500, noise_target=True)
    ce1_ctrl = np.mean([run_stream(zone_ctrl, encoder,
                                   markov_stream(6, 16, 9100 + i, M1))
                        for i in range(4)])
    print(f"КОНТРОЛЬ (шум без EWC): CE1 {ce1_base:.3f} -> {ce1_ctrl:.3f} "
          f"({ce1_ctrl / ce1_base - 1:+.1%})")

    # EWC с нормированным Фишером
    for lam in (10.0,):
        zone_ewc = make_zone()
        zone_ewc.load_state_dict(zone.state_dict())
        params = dict(
            syn_in=zone_ewc.syn_in.w_slow, syn_rec=zone_ewc.syn_rec.w_slow,
            readout_w=zone_ewc.readout.weight,
        )
        ewc = EWCRegularizer(params, lam=lam)
        mass = ewc.estimate_fisher(ewc_loss_fn(zone_ewc, encoder), n_samples=48)
        raw_max = max(f.max().item() for f in ewc.fisher.values())
        # нормировка: относительная важность
        for f in ewc.fisher.values():
            f.div_(raw_max + 1e-12)
        train_task(zone_ewc, encoder, M1, iters=500, ewc=ewc, lam=lam,
                   noise_target=True)
        ce1_ewc = np.mean([run_stream(zone_ewc, encoder,
                                      markov_stream(6, 16, 9100 + i, M1))
                           for i in range(4)])
        print(f"EWC lam={lam:5.1f}: CE1 {ce1_base:.3f} -> {ce1_ewc:.3f} "
              f"({ce1_ewc / ce1_base - 1:+.1%}) "
              f"| масса Фишера {mass:.2e}")


if __name__ == "__main__":
    main()
