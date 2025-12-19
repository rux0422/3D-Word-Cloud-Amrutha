@echo off
REM ============================================
REM 3D Word Cloud - Start Script for Windows
REM ============================================

echo.
echo ======================================
echo   3D Word Cloud - Starting Servers
echo ======================================
echo.

REM Start Backend
echo Starting Backend Server on port 8001...
start "Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && python main.py"

timeout /t 3 /nobreak >nul

REM Start Frontend
echo Starting Frontend Server on port 3000...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo   Backend API: http://localhost:8001
echo   API Docs:    http://localhost:8001/docs
echo   Frontend:    http://localhost:3000
echo.

timeout /t 5 /nobreak >nul
start "" http://localhost:8001/docs
timeout /t 1 /nobreak >nul
start "" http://localhost:3000

pause
