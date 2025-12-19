@echo off
REM ============================================
REM 3D Word Cloud - Setup Script for Windows
REM Installs all dependencies and starts servers
REM ============================================

echo.
echo ======================================
echo   3D Word Cloud - Setup Script
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

echo [INFO] Python and Node.js detected.
echo.

REM Install Backend Dependencies
echo ======================================
echo   Installing Backend Dependencies
echo ======================================
echo.

cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing packages...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

REM Download NLTK data
echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('maxent_ne_chunker', quiet=True); nltk.download('maxent_ne_chunker_tab', quiet=True); nltk.download('words', quiet=True)"

cd ..

echo.
echo ======================================
echo   Installing Frontend Dependencies
echo ======================================
echo.

cd frontend
call npm install
cd ..

echo.
echo ======================================
echo   Setup Complete!
echo ======================================
echo.
echo To start the application, run: start.bat
echo.
echo Or manually:
echo   Backend:  cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo   Frontend: cd frontend ^&^& npm run dev
echo.

pause
