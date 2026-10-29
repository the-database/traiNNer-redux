#!/bin/bash

# Prefer the newest supported Python, fall back to older supported versions
PYTHON=""
for candidate in python3.14 python3.13 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.12, 3.13, or 3.14 is required (3.14 recommended). Please install one and try again."
    exit 1
fi

echo "Using $PYTHON"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install .

echo "traiNNer-redux dependencies installed successfully!"
