from os import path as osp

import msgspec
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.utils.redux_options import DatasetOptions


def test_pairedvideodataset() -> None:
    """Test dataset: PairedVideoDataset"""

    clip_size = 5
    gt_size = 128
    scale = 2

    opt_str = rf"""
name: Test
type: PairedVideoDataset
dataroot_gt: datasets/train/video/hr
dataroot_lq: datasets/train/video/lr
filename_tmpl: '{{}}'
io_backend:
    type: disk
clip_size: {clip_size}
scale: {scale}
gt_size: {gt_size}
use_hflip: true
use_rot: true

phase: train
"""
    image_names = [f"show1_Frame{i}" for i in range(200, 211)]
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)

    dataset = PairedVideoDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert (
        len(dataset) == len(image_names) - clip_size + 1
    )  # whether to read correct meta info

    # ------------------ test scan folder mode -------------------- #
    opt.io_backend = {"type": "disk"}
    dataset = PairedVideoDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert (
        len(dataset) == len(image_names) - clip_size + 1
    )  # whether to correctly scan folders

    # test __getitem__
    for i in range(7):
        result = dataset.__getitem__(i)
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
        assert result["gt"].shape == (3, gt_size, gt_size)
        assert result["lq"].shape == (clip_size, 3, gt_size // scale, gt_size // scale)
        print(i, result["lq_path"], result["gt_path"])
        assert osp.normpath(result["lq_path"]) == osp.normpath(
            f"datasets/train/video/lr/{image_names[i + clip_size // 2]}.png"
        )
        assert osp.normpath(result["gt_path"]) == osp.normpath(
            f"datasets/train/video/hr/{image_names[i + clip_size // 2]}.png"
        )

    # ------------------ test lmdb backend and with y channel-------------------- #
    # TODO
    # opt["dataroot_gt"] = "tests/data/gt.lmdb"
    # opt["dataroot_lq"] = "tests/data/lq.lmdb"
    # opt["io_backend"] = {"type": "lmdb"}
    # opt["color"] = "y"
    # opt["mean"] = [0.5]
    # opt["std"] = [0.5]

    # dataset = PairedVideoDataset(opt)
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
    # dataset = PairedVideoDataset(opt)

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
