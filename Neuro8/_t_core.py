import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroweave8.config import NeuronCfg
from neuroweave8.core import LaminarALIF
cfg = NeuronCfg(sigma=0.0, alpha_nmda=2.0)
z=torch.zeros(1,8)
def run(b,a,ticks=20):
    n = LaminarALIF(8, cfg); n.reset_state(1,"cpu"); t=0
    for _ in range(ticks):
        t += n.step(b,a,z,sltt=False)["spikes"].sum().item()
    return t
for amp in (0.08,0.10,0.12,0.15,0.20):
    print(f"basal {amp}: alone={run(torch.full((1,8),amp), z)}  +apical={run(torch.full((1,8),amp), torch.full((1,8),5.0))}")
print("apical only:", run(z, torch.full((1,8),5.0)))
print("apical 50:", run(z, torch.full((1,8),50.0)))
