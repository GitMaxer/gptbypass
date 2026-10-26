"""generate.py — генерация текста обученным NEUROWEAVE v9.0 «Cortex».

Загружает чекпоинт, сохранённый `train_cortex.py --save out.pt`
(организм + декодер + словарь TinyStories), прогоняет затравку через
таламус→колонки→SON→синхронизацию полушарий→Лаплас→БГ и генерирует
продолжение. Темп эмиссии задаёт дрифт-диффузия базальных ганглиев
(порог опускает орексин при дефиците энергии, повышает SON-surprise на
непредсказанном контексте).

Запуск:
    python -m neuroweave9.generate --ckpt cortex.pt \
        --prompt "once upon a time" --num-tokens 14
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from .cortex.organism import CortexOrganism
from .readout import LaplaceContextReadout


@torch.no_grad()
def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    org = CortexOrganism(
        vocab_size=cfg["vocab_size"], input_dim=cfg["input_dim"],
        scale=cfg["scale"], vsa_dim=cfg["vsa_dim"], seed=cfg["seed"],
        device=device)
    org.load_state_dict(ckpt["org"])
    readout = LaplaceContextReadout(
        cortical_dim=org.cortical_dim(),
        laplace_dim=org.laplace_memory.num_scales * org.input_dim,
        vocab_size=cfg["vocab_size"], sync_dim=cfg["vsa_dim"]).to(device)
    readout.load_state_dict(ckpt["readout"])
    return org, readout, ckpt["itos"], ckpt["stoi"]


@torch.no_grad()
def generate(
    org: CortexOrganism,
    readout: LaplaceContextReadout,
    prompt_words: List[str],
    itos: List[str],
    stoi: Dict[str, int],
    num_tokens: int = 14,
    max_bg_ticks: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, Dict[str, float]]:
    """Затравка через полный контур, затем автономная генерация.

    Каждый токен: такт коры → fused-логиты readout → БГ решает «когда
    сказать» (emit_mask) — если ни один тик не эмитировал, берём argmax.
    """
    org.set_phase(2)
    org.reset_state(batch_size=1, device=device)

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
        emitted = False
        for _tick in range(max_bg_ticks):
            bg_ticks += 1
            bg = org.basal_ganglia(
                logits,
                serotonin_5ht=out["left"]["zone_a"]["modulators"][
                    "serotonin_5ht"],
                ach_level=out["left"]["zone_a"]["modulators"]["ach"],
                orexin_level=out["orexin"]["orexin"],
                surprise_level=out["son"]["surprise"])
            if float(bg["emit_mask"].item()) >= 1.0:
                bg_emissions += 1
                emitted = True
                break
        out_words.append(itos[int(logits.argmax(-1).item())])

    text = ""
    for tok in [prompt_words[0]] + out_words:
        text += tok if tok in (".", ",", "!", "?", ";") \
            else (" " if text else "") + tok
    return text, {
        "bg_emissions": float(bg_emissions),
        "mean_ticks_to_decision": bg_ticks / max(1, num_tokens),
        "emit_rate": bg_emissions / max(1, num_tokens),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NEUROWEAVE v9.0 Cortex — генерация")
    ap.add_argument("--ckpt", required=True, help="чекпоинт train_cortex --save")
    ap.add_argument("--prompt", default="once upon a time")
    ap.add_argument("--num-tokens", type=int, default=14)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    org, readout, itos, stoi = load_checkpoint(args.ckpt, device)
    prompt = args.prompt.strip().split()
    print(f"[v9 генерация] словарь={len(itos)} | затравка: "
          f"'{' '.join(prompt)}' | {device}")
    text, stats = generate(org, readout, prompt, itos, stoi,
                           num_tokens=args.num_tokens, device=device)
    print(f"  сеть: '{text}'")
    print(f"  эмиссий БГ={stats['bg_emissions']:.0f} "
          f"({stats['emit_rate']:.0%} слов через гейт), "
          f"решение={stats['mean_ticks_to_decision']:.1f} мс/слово "
          f"({time.perf_counter() - t0:.1f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
