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
dataroot_gt: [datasets/train/video/hr]
dataroot_lq: [datasets/train/video/lr]
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


# def test_getitem() -> None:
#     opt = DatasetOptions(
#         name="train",
#         type="train",
#         clip_size=5,
#         num_worker_per_gpu=0,
#         persistent_workers=False,
#         prefetch_factor=None,
#         scale=2,
#         gt_size=128,
#         dataroot_gt=[
#             "datasets/train/send/HR1",
#             "datasets/train/send/HR2",
#             "datasets/train/video/hr",
#         ],
#         dataroot_lq=[
#             "datasets/train/send/LR1",
#             "datasets/train/send/LR2",
#             "datasets/train/video/lr",
#         ],
#     )

#     dataset = PairedVideoDataset(opt)

#     # test __getitem__
#     for i in range(len(dataset)):
#         result = dataset.__getitem__(i)
#         print(i, result["lq_path"])
