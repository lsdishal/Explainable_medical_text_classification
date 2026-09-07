#!/bin/bash
set -e

echo "=== Medical Text XAI Environment Setup ==="

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment $VENV_DIR already exists."
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "=== Setup Completed Successfully ==="
echo "To activate the environment in your terminal, run:"
echo "source .venv/bin/activate"
