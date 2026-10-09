#!/bin/bash

python3.14 -m venv venv
source venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install .

echo "traiNNer-redux dependencies installed successfully!"
