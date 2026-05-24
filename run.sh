#!/bin/bash
set -e

echo "=============================="
echo " CWRU VAE Anomaly Detection"
echo "=============================="

# Create virtual environment (optional but recommended)
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[2/3] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "[3/3] Running pipeline..."
python main.py
