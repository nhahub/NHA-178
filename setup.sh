#!/bin/bash

# Setup script for Pima MLflow Project
# This script sets up the environment and installs dependencies

echo "=========================================="
echo "Pima MLflow Project - Setup Script"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p artifacts
mkdir -p mlruns
mkdir -p logs

# Copy environment template
echo ""
echo "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created from template"
else
    echo ".env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   - Windows: venv\\Scripts\\activate"
echo "   - Linux/Mac: source venv/bin/activate"
echo ""
echo "2. Run the pipeline:"
echo "   python main.py"
echo ""
echo "3. View results:"
echo "   mlflow ui --port 5000"
echo ""
echo "=========================================="
