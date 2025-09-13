@echo off
title Object Detection Project Launcher
color 0a
echo.
echo ================================================
echo    OBJECT DETECTION PROJECT LAUNCHER
echo ================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run these commands first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [2/4] Checking if FastAPI is installed...
python -c "import fastapi" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] FastAPI not installed. Run: pip install fastapi uvicorn
    pause
    exit /b 1
)

echo [3/4] Starting FastAPI backend...
start "Object Detection API Server" cmd /c "cd src\api && python main.py && pause"

echo [4/4] Waiting for server startup...
timeout /t 4 /nobreak >nul

echo [5/4] Testing API connection...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] API server is running!
) else (
    echo [WARNING] API server may not be ready yet
)

echo.
echo Opening frontend...
start "" "src\frontend\app.html"

echo.
echo ================================================
echo           PROJECT STARTED SUCCESSFULLY!
echo ================================================
echo.
echo - API Server: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo - Frontend: opened in browser
echo.
echo Close the API server window to stop the backend.
echo.
pause