from os import path as osp

import msgspec
from traiNNer.data.single_gt_dataset import SingleGtDataset
from traiNNer.utils.redux_options import DatasetOptions


def test_single_gt_dataset() -> None:
    opt_str = r"""
name: EcoTest
type: SingleGtDataset
dataroot_gt: [datasets/val/dataset1/hr]
io_backend:
    type: disk
scale: 4
gt_size: 128
use_hflip: true
use_rot: true
phase: train
"""
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)

    dataset = SingleGtDataset(opt)
    assert len(dataset) == 3
    assert dataset.io_backend_opt["type"] == "disk"

    result = dataset[0]
    assert set(result.keys()) == {"gt", "gt_path", "lq_path"}
    assert result["gt"].shape == (3, 128, 128)
    # lq_path is a harmless placeholder (same as gt_path) so downstream logging works
    assert result["lq_path"] == result["gt_path"]
    assert osp.normpath(result["gt_path"]).endswith(
        osp.normpath("datasets/val/dataset1/hr/0007.png")
    )


def test_single_gt_dataset_rejects_misaligned_gt_size() -> None:
    opt_str = r"""
name: EcoTestBad
type: SingleGtDataset
dataroot_gt: [datasets/val/dataset1/hr]
io_backend:
    type: disk
scale: 4
gt_size: 127
use_hflip: true
use_rot: true
phase: train
"""
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)
    try:
        SingleGtDataset(opt)
    except AssertionError as e:
        assert "divisible by scale" in str(e)
    else:
        raise AssertionError("expected AssertionError for gt_size=127 with scale=4")
