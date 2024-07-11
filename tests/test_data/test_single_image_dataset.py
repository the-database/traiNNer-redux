from os import path as osp

import msgspec
from traiNNer.data.single_image_dataset import SingleImageDataset
from traiNNer.utils.redux_options import DatasetOptions


def test_singleimagedataset() -> None:
    """Test dataset: SingleImageDataset"""

    opt_str = r"""
name: Test
type: SingleImageDataset
dataroot_lq: datasets/val/dataset1/lr
io_backend:
    type: disk

mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
"""
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)

    dataset = SingleImageDataset(opt)

    # ------------------ test scan folder mode -------------------- #
    opt.io_backend = {"type": "disk"}
    dataset = SingleImageDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert len(dataset) == 3  # whether to correctly scan folders

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ["lq", "lq_path"]
    assert set(expected_keys).issubset(set(result.keys()))
    assert "lq" in result and "lq_path" in result
    # check shape and contents
    assert result["lq"].shape == (3, 128, 128)
    assert osp.normpath(result["lq_path"]) == osp.normpath(
        "datasets/val/dataset1/lr/0007.png"
    )

    # ------------------ test lmdb backend and with y channel-------------------- #
    # TODO
    # opt["dataroot_lq"] = "tests/data/lq.lmdb"
    # opt["io_backend"] = {"type": "lmdb"}
    # opt["color"] = "y"
    # opt["mean"] = [0.5]
    # opt["std"] = [0.5]

    # dataset = SingleImageDataset(opt)
    # assert dataset.io_backend_opt["type"] == "lmdb"  # io backend
    # assert len(dataset) == 2  # whether to read correct meta info
    # assert dataset.std == [0.5]

    # # test __getitem__
    # result = dataset.__getitem__(1)
    # # check returned keys
    # expected_keys = ["lq", "lq_path"]
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert "lq" in result and "lq_path" in result
    # assert result["lq"].shape == (1, 90, 60)
    # assert result["lq_path"] == "comic"
