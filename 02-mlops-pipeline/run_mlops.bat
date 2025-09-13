@echo off
title MLOps Pipeline Launcher
color 0a

echo ================================================
echo          MLOPS PIPELINE LAUNCHER
echo ================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Checking dependencies...
python -c "import torch, transformers, fastapi" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Dependencies not installed. Installing...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [3/3] Starting MLOps Pipeline...
python run_mlops_pipeline.py --mode full

echo.
echo Pipeline execution completed.
pause
