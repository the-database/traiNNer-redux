# Contributing
## Getting Started
[Visual Studio Code](https://code.visualstudio.com/) is the recommended IDE, with the [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) and [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extensions.

Install dependencies:
- `pip install -e .[dev]`

Commands:
- Ruff format: `ruff format`
- Ruff lint: `ruff check --fix`
- Pyright: `pyright`
- pytest: `pytest tests` or use the [VS Code test runner](https://code.visualstudio.com/docs/python/testing#_run-tests).
