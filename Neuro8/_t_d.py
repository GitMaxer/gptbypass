import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroweave8.config import default_cfg
from neuroweave8.microcircuit import LaminarColumn
cfg = default_cfg("poc"); B=1
col = LaminarColumn(cfg.col("A"), cfg.neuron, cfg.plasticity, seed=1)
col.reset_state(B,"cpu")
s = torch.zeros(B,32); s[:,0]=1; s[:,5]=1
for t in range(8):
    o = col.step(s, sltt=False)
    l4 = col.pools["l4"]
    v = l4.v[0]
    print(t, "spk", int(o.spikes["l4"].sum()), "| V mean %.1f max %.1f | g mean %.2f | th %.1f | refr %d" % (
        v.mean(), v.max(), l4.g_ampa.mean(), o.theta["l4"].mean(), int(l4.refr[0].sum())))
print("drive per neuron:", torch.round(col.syn["in_4"](s)[0][:8],3))
