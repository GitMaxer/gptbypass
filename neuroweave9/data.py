"""Данные TinyStories для пайплайна v9 — порт загрузчиков Spiker."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import torch

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[.,!?;]")


def load_stories(data_path: str, max_stories: int = 32) -> List[str]:
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    if "<|endoftext|>" in text:
        raw = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    else:
        raw = [s.strip() for s in text.split("\n\n") if s.strip()]
    return raw[:max_stories]


def tokenize_text(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def build_vocab_and_sequences(
    stories: List[str],
    max_vocab_size: int = 96,
    seq_len: int = 20,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[str], Dict[str, int], torch.Tensor, torch.Tensor, List[List[str]]]:
    tokenized: List[List[str]] = []
    counter: Counter = Counter()
    for st in stories:
        toks = tokenize_text(st)[: seq_len + 4]
        if len(toks) >= seq_len + 1:
            tokenized.append(toks)
            counter.update(toks)

    special = ["<pad>", "<unk>"]
    most_common = [w for w, _ in counter.most_common(max_vocab_size - len(special))]
    itos = special + most_common
    stoi = {w: i for i, w in enumerate(itos)}
    vocab_size = len(itos)

    chosen = tokenized[:batch_size]
    while len(chosen) < batch_size:
        chosen.append(tokenized[len(chosen) % len(tokenized)])

    x_seq = torch.zeros((seq_len, batch_size, vocab_size), device=device)
    y_seq = torch.zeros((seq_len, batch_size), device=device, dtype=torch.long)
    for b, toks in enumerate(chosen):
        for t in range(seq_len):
            x_seq[t, b, stoi.get(toks[t], 1)] = 1.0
            y_seq[t, b] = stoi.get(toks[t + 1], 1)
    return itos, stoi, x_seq, y_seq, chosen


def build_holdout(
    stories: List[str],
    stoi: Dict[str, int],
    itos: List[str],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    chosen: List[List[str]] = []
    for st in stories:
        toks = [t.lower() for t in tokenize_text(st)][: seq_len + 4]
        if len(toks) >= seq_len + 1:
            chosen.append(toks)
    chosen = chosen[:batch_size]
    vocab_size = len(itos)
    x_seq = torch.zeros((seq_len, len(chosen), vocab_size), device=device)
    y_seq = torch.zeros((seq_len, len(chosen)), device=device, dtype=torch.long)
    for b, toks in enumerate(chosen):
        for t in range(seq_len):
            x_seq[t, b, stoi.get(toks[t], 1)] = 1.0
            y_seq[t, b] = stoi.get(toks[t + 1], 1)
    return x_seq, y_seq, chosen
