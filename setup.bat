@echo off
REM Setup script for Pima MLflow Project (Windows)
REM This script sets up the environment and installs dependencies

echo ==========================================
echo Pima MLflow Project - Setup Script
echo ==========================================

REM Check Python version
echo.
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating project directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist artifacts mkdir artifacts
if not exist mlruns mkdir mlruns
if not exist logs mkdir logs

REM Copy environment template
echo.
echo Setting up environment configuration...
if not exist .env (
    copy .env.example .env
    echo .env file created from template
) else (
    echo .env file already exists
)

echo.
echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.
echo Next steps:
echo 1. Virtual environment is activated
echo.
echo 2. Run the pipeline:
echo    python main.py
echo.
echo 3. View results:
echo    mlflow ui --port 5000
echo.
echo ==========================================
pause
