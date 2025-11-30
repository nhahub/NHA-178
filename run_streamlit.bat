@echo off
REM Streamlit Application Setup Script for Windows

echo ============================================
echo Pima Diabetes Streamlit App Setup
echo ============================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Install dependencies
echo [4/5] Installing dependencies...
echo This may take a few minutes...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Create necessary directories
echo [5/5] Setting up directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist mlruns mkdir mlruns
if not exist artifacts mkdir artifacts
echo Directories created!
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To run the Streamlit app:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run: streamlit run streamlit_app.py
echo   3. Open browser to: http://localhost:8501
echo.
echo Starting Streamlit app now...
echo.

REM Start Streamlit
streamlit run streamlit_app.py

pause
