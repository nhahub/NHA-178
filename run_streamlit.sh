#!/bin/bash

# Streamlit Application Setup Script for Linux/Mac

echo "============================================"
echo "Pima Diabetes Streamlit App Setup"
echo "============================================"
echo ""

# Check Python installation
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully!"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "Virtual environment activated!"
echo ""

# Install dependencies
echo "[4/5] Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

# Create necessary directories
echo "[5/5] Setting up directories..."
mkdir -p data models mlruns artifacts
echo "Directories created!"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To run the Streamlit app:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run streamlit_app.py"
echo "  3. Open browser to: http://localhost:8501"
echo ""
echo "Starting Streamlit app now..."
echo ""

# Start Streamlit
streamlit run streamlit_app.py
