@echo off
if not exist "input" mkdir input
if not exist "venv" (
    echo Creating venv...
    python -m venv venv
)
echo Installing dependencies...
venv\Scripts\pip install -r requirements.txt
echo Done.
pause
