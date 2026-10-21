"""Тесты этапа оптимизация/масштабирование/квантизация: int8-паритет,
NeuroSystem-фасад, скейлинг-сконструированность wide-пресета."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from neuroweave8.config import default_cfg
from neuroweave8.core.quantize import QuantizedSynapse, quantize_organism
from neuroweave8.organism.organism import Organism
from neuroweave8.organism.system import NeuroSystem
from neuroweave8.tasks.environments import RuleShiftTask


def _feed(org, cfg, x):
    inp = {n: torch.zeros(x.shape[0], cfg.col(n).n_in) for n in org.names}
    inp[org.names[0]][:, : x.shape[-1]] = x
    return inp


def test_quantize_parity():
    """int8-снимок даёт почти тот же драйв, что и float32-оригинал."""
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    col = org.columns[org.names[0]]
    syn = col.syn["in_4"]
    w_ref = syn.w_eff.detach().clone()
    qs = QuantizedSynapse(syn, bits=8)
    # паритет самого веса
    assert float((qs.w_eff - w_ref).abs().max()) < 0.05
    # паритет прямого прохода на реальном спайк-входе
    s = torch.zeros(2, syn.n_in)
    s[:, :3] = 1.0
    out_ref = syn(s)
    out_q = qs(s)
    denom = float(out_ref.abs().max()) or 1.0
    assert float((out_q - out_ref).abs().max()) / denom < 0.05


def test_quantize_organism_compresses_and_ticks():
    """quantize_organism: память ~4x меньше, организм продолжает тактать."""
    cfg = default_cfg("small")
    org = Organism(cfg, batch=2)
    org.reset_state(2)
    stats = quantize_organism(org, bits=8)
    assert stats["compression"] > 4.0  # 24B/ребро (w_slow+w_fast+snap+mask+age) -> 2B
    assert stats["max_abs_err"] < 0.05
    # горячий контур работает на int8-снимках (learn=False: пластичность офф)
    x = torch.zeros(2, cfg.col(org.names[0]).n_in)
    x[:, 0] = 1.0
    outs = org.tick(_feed(org, cfg, x), learn=False)
    assert torch.isfinite(outs[org.names[0]].spikes["l23"]).all()


def test_neurosystem_lifecycle():
    """Фасад системы: boot -> эпизод -> отчёт -> save/load."""
    sys_ = NeuroSystem(preset="small", batch=2)
    sys_.boot(warmup_ticks=5)
    env = RuleShiftTask(vocab=cfg_n_out(sys_), switch_at=10**9, seed=5)
    res = sys_.run_episode(env, 40, auto_sleep=True)
    assert res["steps"] == 40.0 and 0.0 <= res["acc_mean"] <= 1.0
    rep = sys_.report()
    assert rep["neurons"] > 0 and rep["synapses"] > 0
    assert torch.isfinite(torch.tensor(rep["sleep_pressure"]))
    sys_.save("/tmp/nws_ckpt.pt")
    sys2 = NeuroSystem.load("/tmp/nws_ckpt.pt")
    assert sys2.org.cfg.seed == sys_.org.cfg.seed


def cfg_n_out(sys_):
    return sys_.cfg.col(sys_.org.names[0]).n_out


def test_wide_preset_constructs_and_ticks():
    """Масштабирование: wide (4x нейроны/колонку) собирается и тикает."""
    cfg = default_cfg("wide")
    org = Organism(cfg, batch=1)
    org.reset_state(1)
    x = torch.zeros(1, cfg.col(org.names[0]).n_in)
    x[:, 0] = 1.0
    outs = org.tick(_feed(org, cfg, x), learn=True)
    assert torch.isfinite(outs[org.names[0]].v["l5"]).all()
    assert cfg.total_neurons() > default_cfg("poc").total_neurons() * 3


def test_scaling_sweep_runs():
    from neuroweave8.bench import bench_scaling
    rows = bench_scaling(presets=("small",), batches=(1, 4), ticks=10)
    assert len(rows) == 2 and all(r["ticks_per_sec"] > 0 for r in rows)
