#!/bin/sh
# NEUROWEAVE v6.0 — verification (CUDA Graphs + GABA math + Exam)
cd "$(dirname "$0")"

echo "============================================================"
echo "  NEUROWEAVE v6.0 - verification"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
    echo "[ERROR] Python not found. Install Python 3.9+ first."
    exit 1
fi
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "[INFO] torch not found - installing..."
    "$PY" -m pip install torch || exit 1
fi

"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo
"$PY" verify_cuda_graphs.py
exit $?
