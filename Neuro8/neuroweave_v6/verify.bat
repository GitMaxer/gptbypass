@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   NEUROWEAVE v6.0 - verification (CUDA Graphs + GABA + Exam)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.9+ first.
    pause
    exit /b 1
)

python --version

python -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo [INFO] torch not found - installing...
    python -m pip install torch
    if errorlevel 1 (
        echo [ERROR] torch installation failed.
        echo        For CUDA build use:
        echo        python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
        pause
        exit /b 1
    )
)

python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo.
echo Running full verification...
echo.
python verify_cuda_graphs.py
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo RESULT: ALL GREEN - exit code 0
) else (
    echo RESULT: FAILURES DETECTED - exit code %RC%
)
echo See VERIFY.md for expected outputs and troubleshooting.
pause
exit /b %RC%
