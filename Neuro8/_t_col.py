import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroweave8.config import default_cfg
from neuroweave8.microcircuit import LaminarColumn
cfg = default_cfg("poc"); B=2
col = LaminarColumn(cfg.col("A"), cfg.neuron, cfg.plasticity, seed=1)
def run(arousal, ticks=24, s_in=None):
    col.reset_state(B,"cpu")
    acc={k:0.0 for k in ("l4","l23","l5","l6","pv","sst")}
    for t in range(ticks):
        s = s_in if s_in is not None else None
        s = torch.zeros(B,32); s[:, t%16]=1; s[:,(t*3)%16]=1
        ar = None if arousal is None else torch.full((B, cfg.col("A").n_l5), arousal)
        o = col.step(s, arousal=ar, sltt=False)
        for k in acc: acc[k]+=float(o.spikes[k].mean())
    return {k: round(v/ticks,3) for k,v in acc.items()}
for a in (None, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
    print(a, run(a))
# drives
col.reset_state(B,"cpu")
o = col.step(torch.ones(B,32)*0.0+torch.eye(32)[0].repeat(B,1), sltt=False)
print("L23 drive mean", float(col.syn["4_23"](col.prev["l4"]).mean()), "L23 spikes", float(o.spikes["l23"].sum()))
print("L5 basal drive", float(col.syn["23_5"](col.prev["l23"]).mean()))
print("L6->L5 apical drive", float(col.syn["6_5"](col.prev["l6"]).mean()))
