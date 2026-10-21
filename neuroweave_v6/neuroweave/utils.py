"""Утилиты: устройство, сиды, тензорные хелперы."""
import random

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def t(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def fmt_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device.type}:{device.index or 0} [{torch.cuda.get_device_name(device)}]"
    return "cpu"
