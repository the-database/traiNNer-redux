"""Train a model: python train.py -opt options/train/<ARCH>/<config>.yml

This file stays small on purpose. On Windows every DataLoader worker is a new
process that re-imports it before it can receive its dataset, and the parent waits
for that, so heavy imports here would start the workers one after another.
The training loop is in traiNNer/training.py.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if __name__ == "__main__":
    from traiNNer.check.check_dependencies import check_dependencies

    check_dependencies()

    from traiNNer.training import train_pipeline

    train_pipeline(os.path.abspath(os.path.dirname(__file__)))
