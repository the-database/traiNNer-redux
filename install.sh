#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install .

echo "traiNNer-redux dependencies installed successfully!"
