"""Протокол сна: норма (E_cl=V_rest + KCC2) против патологии (E_cl=-85)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroweave.config import NWConfig
from neuroweave.sleep import run_sleep
from neuroweave.utils import fmt_device, get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description="Протокол сна NEUROWEAVE")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = get_device() if args.device == "auto" else __import__("torch").device(args.device)
    set_seed(7)
    print(f"Устройство: {fmt_device(dev)}")
    cfg = NWConfig()
    st_normal = run_sleep(cfg, dev, pathological=False)
    st_patho = run_sleep(cfg, dev, pathological=True)
    ok = st_normal.bound_ok and (not st_patho.bound_ok)
    print(f"\n[ИТОГ] норма: V >= V_rest в N3: {st_normal.bound_ok}; "
          f"патология без KCC2 проваливается: {not st_patho.bound_ok} -> "
          f"{'МАТЕМАТИКА РАБОТАЕТ' if ok else 'ПРОВЕРЬ КОНФИГ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
