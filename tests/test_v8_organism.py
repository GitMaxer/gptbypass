"""Интеграционный тест цикла организма v8 (быстрый, small-пресет)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neuroweave8.config import default_cfg
from neuroweave8.organism.organism import Organism


def test_organism_tick_and_behave():
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    x = {n: torch.zeros(2, cfg.col(n).n_in) for n in org.names}
    n_out = cfg.col(org.names[0]).n_out
    x[org.names[0]][:, : 2] = torch.eye(2)
    outs = org.tick(x, learn=True)
    assert org.names[0] in outs
    r = org.behave(x, torch.zeros(2, dtype=torch.long), learn=True)
    assert r["fused"].shape == (2, n_out)
    org.assemble()
    r2 = org.behave(x, torch.zeros(2, dtype=torch.long), learn=True)
    assert 0.0 <= r2["reward"] <= 1.0


def test_sleep_cycle_runs():
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    x = {n: torch.zeros(2, cfg.col(n).n_in) for n in org.names}
    for _ in range(3):
        org.tick(x, learn=True)
    rep = org.sleep_cycle()
    assert "consolidated" in rep
