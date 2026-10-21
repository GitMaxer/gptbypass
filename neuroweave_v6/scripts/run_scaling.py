"""Масштабирование: время шага / память / гомеостаз по факторам роста."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroweave.scaling import run_scaling
from neuroweave.utils import fmt_device, get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description="Масштабирование NEUROWEAVE")
    ap.add_argument("--factors", default="1,2,4")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    dev = get_device()
    set_seed(7)
    print(f"Устройство: {fmt_device(dev)}")
    factors = [int(x) for x in args.factors.split(",") if x.strip()]
    pts = run_scaling(dev, factors=factors, steps=args.steps)
    ok = all(p.finite for p in pts)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
