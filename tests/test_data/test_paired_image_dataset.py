from os import path as osp

import msgspec
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.utils.redux_options import DatasetOptions


def test_pairedimagedataset() -> None:
    """Test dataset: PairedImageDataset"""

    opt_str = r"""
name: Test
type: PairedImageDataset
dataroot_gt: datasets/val/dataset1/hr
dataroot_lq: datasets/val/dataset1/lr
filename_tmpl: '{}'
io_backend:
    type: disk

scale: 4
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
gt_size: 128
use_hflip: true
use_rot: true

phase: train
"""
    image_names = (7, 8, 9)
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)

    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert len(dataset) == 3  # whether to read correct meta info
    assert dataset.mean == [0.5, 0.5, 0.5]

    # ------------------ test scan folder mode -------------------- #
    opt.io_backend = {"type": "disk"}
    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert len(dataset) == 3  # whether to correctly scan folders

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ["lq", "gt", "lq_path", "gt_path"]
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert (
        "gt" in result
        and "lq" in result
        and "lq_path" in result
        and "gt_path" in result
    )
    assert result["gt"].shape == (3, 128, 128)
    assert result["lq"].shape == (3, 32, 32)
    assert osp.normpath(result["lq_path"]) in {
        osp.normpath(f"datasets/val/dataset1/lr/{x:04d}.png") for x in image_names
    }
    assert osp.normpath(result["gt_path"]) in {
        osp.normpath(f"datasets/val/dataset1/hr/{x:04d}.png") for x in image_names
    }

    # ------------------ test lmdb backend and with y channel-------------------- #
    # TODO
    # opt["dataroot_gt"] = "tests/data/gt.lmdb"
    # opt["dataroot_lq"] = "tests/data/lq.lmdb"
    # opt["io_backend"] = {"type": "lmdb"}
    # opt["color"] = "y"
    # opt["mean"] = [0.5]
    # opt["std"] = [0.5]

    # dataset = PairedImageDataset(opt)
    # assert dataset.io_backend_opt["type"] == "lmdb"  # io backend
    # assert len(dataset) == 2  # whether to read correct meta info
    # assert dataset.std == [0.5]

    # # test __getitem__
    # result = dataset.__getitem__(1)
    # # check returned keys
    # expected_keys = ["lq", "gt", "lq_path", "gt_path"]
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert (
    #     "gt" in result
    #     and "lq" in result
    #     and "lq_path" in result
    #     and "gt_path" in result
    # )
    # assert result["gt"].shape == (1, 128, 128)
    # assert result["lq"].shape == (1, 32, 32)
    # assert result["lq_path"] == "comic"
    # assert result["gt_path"] == "comic"

    # ------------------ test case: val/test mode -------------------- #
    # TODO
    # opt["phase"] = "test"
    # opt["io_backend"] = {"type": "lmdb"}
    # dataset = PairedImageDataset(opt)

    # # test __getitem__
    # result = dataset.__getitem__(0)
    # # check returned keys
    # expected_keys = ["lq", "gt", "lq_path", "gt_path"]
    # assert set(expected_keys).issubset(set(result.keys()))
    # # check shape and contents
    # assert (
    #     "gt" in result
    #     and "lq" in result
    #     and "lq_path" in result
    #     and "gt_path" in result
    # )
    # assert result["gt"].shape == (1, 480, 492)
    # assert result["lq"].shape == (1, 120, 123)
    # assert result["lq_path"] == "baboon"
    # assert result["gt_path"] == "baboon"
