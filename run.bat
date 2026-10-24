@echo off
REM Windows: first run sets up .venv (about a minute), then starts teleop.
REM   run.bat              (robot "hrg" over Tailscale)
REM   run.bat --dry        (camera + preview only, nothing sent)
REM   run.bat --robot 100.97.141.88
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo setting up .venv ^(one time^)...
    py -3 -m venv .venv 2>nul || python -m venv .venv || (echo Python 3 not found -- install it from python.org & exit /b 1)
    .venv\Scripts\python -m pip install -q --upgrade pip
    .venv\Scripts\python -m pip install -q -r requirements.txt || exit /b 1
)
.venv\Scripts\python teleop.py %*
