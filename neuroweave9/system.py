"""CortexSystem — системный фасад v9 (аналог NeuroSystem для CortexOrganism)."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from neuroweave8.core.quantize import quantize_organism
from .cortex.organism import CortexOrganism


class CortexSystem:
    def __init__(self, vocab_size: int = 128, scale: float = 1.0,
                 batch: int = 1, seed: int = 42, vsa_dim: int = 512):
        self.batch = int(batch)
        self.seed = seed
        self.scale = float(scale)
        self.vsa_dim = int(vsa_dim)
        self.org = CortexOrganism(vocab_size=vocab_size, input_dim=vocab_size,
                                  scale=scale, vsa_dim=vsa_dim, seed=seed)
        self.booted = False
        self.quantized: Optional[Dict[str, float]] = None

    def boot(self, warmup_ticks: int = 30) -> "CortexSystem":
        self.org.reset_state(self.batch)
        x = torch.zeros(self.batch, self.org.input_dim)
        for _ in range(int(warmup_ticks)):
            self.org.step(x, learn=True, sync_metrics=False)
        self.booted = True
        return self

    def step(self, x: torch.Tensor, reward: Optional[torch.Tensor] = None):
        return self.org.step(x, reward=reward)

    def run_episode(self, token_seq: torch.Tensor,
                    target_seq: Optional[torch.Tensor] = None) -> Dict[str, float]:
        return self.org.run_episode(token_seq, target_seq=target_seq)

    def lifecycle(self) -> Optional[Dict[str, float]]:
        if self.org.slow.poll(float(self.org.neuro.sleep_pressure)) == 1:
            return self.org.sleep_cycle()
        return None

    def sleep(self) -> Dict[str, float]:
        return self.org.sleep_cycle()

    def quantize(self, bits: int = 8) -> Dict[str, float]:
        self.quantized = quantize_organism(self.org, bits=bits)
        return self.quantized

    def report(self) -> Dict[str, object]:
        syn = sum(s.edge_count() for c in self.org.columns.values()
                  for s in c.syn.values())
        bytes_ = sum(s.memory_bytes() for c in self.org.columns.values()
                     for s in c.syn.values()
                     if hasattr(s, "memory_bytes"))
        return {"version": "9.0-cortex",
                "neurons": sum(self.org.cfg.col(n).n_neurons()
                               for n in self.org.names),
                "synapses": syn, "synapse_bytes": bytes_,
                "quantized": self.quantized,
                "steps": self.org._steps,
                "telemetry": self.org.telemetry()}

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.org.state_dict(),
                    "vocab_size": self.org.vocab_size,
                    "input_dim": self.org.input_dim,
                    "scale": self.scale, "vsa_dim": self.vsa_dim,
                    "seed": self.seed, "batch": self.batch}, path)

    @classmethod
    def load(cls, path: str) -> "CortexSystem":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sys = cls(vocab_size=ckpt["vocab_size"], batch=ckpt["batch"],
                  seed=ckpt["seed"], scale=ckpt["scale"],
                  vsa_dim=ckpt["vsa_dim"])
        sys.org.load_state_dict(ckpt["state_dict"])
        sys.org.reset_state(sys.batch)
        sys.booted = True
        return sys
