# Contributing
## Prerequisites
This project targets [Python 3.12](https://www.python.org/). [Visual Studio Code](https://code.visualstudio.com/) is the recommended IDE, with the [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance), [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), and [YAML](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) extensions.

This repo includes a tiny test dataset so it's ready to run the training pipeline out of the box, with no need to prepare your own dataset.

To set up your environment, do the following:
1. [Fork and clone the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
2. [Install PyTorch](https://pytorch.org/get-started/locally/) with CUDA or whatever is most suitable for your hardware.
3. Install other project dependencies including dev dependencies:
  `pip install -e .[dev]`

## Launch training code

1. To start training, do one of the following:
   - In VS Code's Run and Debug tab, select the `Train ESRGAN` launch configuration and launch it to start debugging the training code.
   - Run the command `python -opt ./options/train/ESRGAN/ESRGAN.yml` (modify the path to the yml file if you want to test a different architecture) to start training without debugging.
2. Verify that you see the following output within a few minutes, depending on your GPU speed:
   ```
   ...
   2024-07-02 21:40:56,593 INFO: Model [SRModel] is created.
   2024-07-02 21:40:56,668 INFO: Start training from epoch: 0, iter: 0
   2024-07-02 21:41:17,816 INFO: [4x_SP..][epoch:  0, iter:     100, lr:(1.000e-04,)] [performance: 4.729] [eta: 14:11:33] l_g_mssim: 1.0000e+00 l_g_percep: 3.5436e+00 l_g_hsluv: 4.3935e-01 l_g_gan: 2.4346e+00 l_g_total: 7.4175e+00 l_d_real: 2.4136e-01 out_d_real: 2.9309e+00 l_d_fake: 5.2773e-02 out_d_fake: -2.4104e+01
   ```
3. Press `ctrl+C` to end the training run.

## PR Requirements

All PRs have the following build checks. To save time, you can run these tests locally to ensure your code passes these tests.
- Formatting and linting checks with Ruff
  - `ruff format`
  - `ruff check --fix`
- Static type checks with Pyright
  - `pyright`
- Unit tests with pytest
  - `pytest tests/` or use the [VS Code test runner](https://code.visualstudio.com/docs/python/testing#_run-tests).
