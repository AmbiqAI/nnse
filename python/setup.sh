#!/bin/bash

# Exit on any error
set -e

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete."
