@echo off
python -m venv venv
call venv\Scripts\activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install .
echo traiNNer-redux dependencies installed successfully!
pause
