#!/bin/bash

python3.12 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install .

echo "traiNNer-redux dependencies installed successfully!"
