#!/bin/bash

apt install libvips-dev
python3.12 -m venv venv
source venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install .

echo "traiNNer-redux dependencies installed successfully!"
