"""«Экзамен»: смешанные режимы синтаксис<->логика, w_slow заморожен."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroweave.exam import ExamRunner, exam_config
from neuroweave.utils import fmt_device, get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description="Экзамен NEUROWEAVE")
    ap.add_argument("--graph", action="store_true",
                    help="прогнать экзамен внутри CUDA Graph (нужен GPU)")
    ap.add_argument("--pre-win", type=int, default=60)
    ap.add_argument("--scale", type=int, default=1)
    args = ap.parse_args()
    dev = get_device()
    set_seed(7)
    print(f"Устройство: {fmt_device(dev)}")
    cfg = exam_config(seed=7, scale=args.scale)
    runner = ExamRunner(cfg, dev)
    runner.pretrain(n_win_syn=args.pre_win, n_pairs_log=args.pre_win, verbose=True)
    rep = runner.run_exam(use_graph=args.graph and dev.type == "cuda")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
