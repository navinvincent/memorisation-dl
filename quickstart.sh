#!/bin/bash
# Quick start script for memorization experiments

echo "==================================================================="
echo "  Memorization in Deep Learning - Quick Start"
echo "==================================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==================================================================="
echo "  Setup Complete!"
echo "==================================================================="
echo ""
echo "You can now run experiments:"
echo ""
echo "  1. Train with clean labels:"
echo "     python train.py --noise_ratio 0.0 --max_epochs 100"
echo ""
echo "  2. Train with 100% label corruption (memorization):"
echo "     python train.py --noise_ratio 1.0 --max_epochs 100"
echo ""
echo "  3. Run noise ratio experiments:"
echo "     python experiments/run_noise_experiments.py"
echo ""
echo "  4. Run regularization experiments:"
echo "     python experiments/run_regularization_experiments.py"
echo ""
echo "==================================================================="
