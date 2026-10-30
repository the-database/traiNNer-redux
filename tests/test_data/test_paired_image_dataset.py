from os import path as osp
from pathlib import Path

import cv2
import msgspec
import numpy as np
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.utils.options import derive_train_patch_size
from traiNNer.utils.redux_options import DatasetOptions


def test_pairedimagedataset() -> None:
    """Test dataset: PairedImageDataset"""

    opt_str = r"""
name: Test
type: PairedImageDataset
dataroot_gt: [
datasets/val/dataset1/hr,
datasets/val/dataset1/hr2,
]
dataroot_lq: [
datasets/val/dataset1/lr,
datasets/val/dataset1/lr2
]
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


def make_rect_dataset(
    tmp_path: Path, lq_size: tuple[int, int], use_rot: bool = True
) -> PairedImageDataset:
    """A 48x32 LR / 96x64 HR pair at scale 2, so 90 degree rotation changes the shape."""
    rng = np.random.default_rng(0)
    for folder, (w, h) in (("lr", (48, 32)), ("hr", (96, 64))):
        (tmp_path / folder).mkdir()
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / folder / "0001.png"), img)

    opt = DatasetOptions(
        name="Test",
        type="PairedImageDataset",
        dataroot_gt=[str(tmp_path / "hr")],
        dataroot_lq=[str(tmp_path / "lr")],
        scale=2,
        lq_size=lq_size,
        use_hflip=True,
        use_rot=use_rot,
        phase="train",
    )
    derive_train_patch_size(opt, 2, high_order_degradation=False)
    return PairedImageDataset(opt)


def test_rectangular_crop_is_width_by_height(tmp_path: Path) -> None:
    dataset = make_rect_dataset(tmp_path, (16, 8))
    for _ in range(20):
        result = dataset[0]
        assert result["lq"].shape == (3, 8, 16)
        assert result["gt"].shape == (3, 16, 32)


def test_full_frame_crop_skips_rotation(tmp_path: Path) -> None:
    dataset = make_rect_dataset(tmp_path, (48, 32))
    for _ in range(30):
        result = dataset[0]
        assert result["lq"].shape == (3, 32, 48)
        assert result["gt"].shape == (3, 64, 96)
