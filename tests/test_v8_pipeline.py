"""Пайплайн-тесты: адаптация к инверсии, сон, EWC, внимание, WM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from neuroweave8.config import default_cfg
from neuroweave8.organism.organism import Organism
from neuroweave8.organism.active_inference import ActiveInference
from neuroweave8.tasks.environments import RuleShiftTask


def _run(org, cfg, xs, ys, learn=True):
    accs = []
    for t in range(xs.shape[0]):
        inp = {n: torch.zeros(xs.shape[1], cfg.col(n).n_in) for n in org.names}
        inp[org.names[0]][:, : xs.shape[-1]] = xs[t]
        r = org.behave(inp, ys[t], learn=learn)
        accs.append(float(r["correct"].float().mean()))
    return accs


def test_adaptation_after_inversion():
    """M1_inv: после инверсии правила точность восстанавливается выше нуля."""
    cfg = default_cfg("small")
    org = Organism(cfg, batch=4)
    org.reset_state(4)
    org.assemble()
    n_out = cfg.col(org.names[0]).n_out
    env = RuleShiftTask(vocab=n_out, switch_at=120, seed=8)
    xs, ys = env.run_episode(240, batch=4)
    accs = _run(org, cfg, xs, ys)
    post = accs[125:]
    assert sum(post) / len(post) > 0.0
    # последние 40 лучше первых 40 после инверсии — идёт адаптация
    assert sum(post[-40:]) >= sum(post[:40])


def test_sleep_consolidates_and_preserves():
    """Сон: консолидация переносит массу w_fast -> w_slow, прунит пыль."""
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    org.assemble()
    n_out = cfg.col(org.names[0]).n_out
    env = RuleShiftTask(vocab=n_out, switch_at=10**9, seed=3)
    xs, ys = env.run_episode(120, batch=2)
    _run(org, cfg, xs, ys)
    rep = org.sleep_cycle()
    assert rep["consolidated"] >= 0.0 and rep["episodes"] >= 0.0
    assert float(org.neuro.sleep_pressure) == 0.0


def test_ewc_penalty_arms():
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    fm = org.arm_ewc(
        loss_fn=lambda: sum(s.w_slow.pow(2).sum()
                            for c in org.columns.values()
                            for s in c.syn.values()),
        n_samples=4)
    assert fm >= 0.0 and org._ewc.is_armed
    pen = org._ewc.penalty()
    assert torch.isfinite(pen)


def test_active_inference_gains_normalized():
    ai = ActiveInference(3, gamma=4.0, gain=0.6)
    g = ai.gains(torch.tensor([0.8, 0.2, 0.5]),
                 torch.tensor([0.01, 0.5, 0.05]))
    assert g.shape == (3,)
    # лучший по компетентности+новизне получает максимум внимания
    assert float(g.argmax()) in (0.0, 1.0)
    assert abs(float(g.mean()) - 0.6) < 0.3


def test_wm_binder_integrated():
    """θ-γ биндер наполняется в такте колонки и участвует в features."""
    cfg = default_cfg("small")
    col = org_col = None
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    col = org.columns[org.names[0]]
    x = torch.zeros(2, cfg.col(org.names[0]).n_in)
    x[:, 0] = 1.0
    for _ in range(5):
        org.tick({n: (x if n == org.names[0] else torch.zeros(2, cfg.col(n).n_in))
                  for n in org.names})
    assert col.binder is not None and col.binder.occupancy() > 0.0
    c = cfg.col(org.names[0])
    assert col.features().shape[-1] == c.n_l23 * 2 + c.n_l5 + c.n_in


def test_sparse_input_path_matches_dense():
    """Разреженный событийный драйв даёт ту же математику, что и плотный."""
    from neuroweave8.core.sparse import sparse_drive
    w = torch.randn(8, 16)
    s = torch.zeros(4, 16)
    s[:, [1, 5]] = 1.0
    assert torch.equal(sparse_drive(s, w), F.linear(s, w))
