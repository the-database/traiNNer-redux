import math
import os

import torchvision.utils
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.utils.optionsfile import DatasetOptions


def main(mode: str = "folder") -> None:
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info'.
    """
    opt = DatasetOptions(
        phase="train",
        name="DIV2K",
        type="PairedImageDataset",
        gt_size=128,
        use_hflip=True,
        use_rot=True,
        num_worker_per_gpu=2,
        batch_size_per_gpu=16,
        scale=4,
        dataset_enlarge_ratio=1,
        dataroot_gt="datasets/DIV2K/DIV2K_train_HR_sub",
        dataroot_lq="datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub",
        filename_tmpl="{}",
        io_backend={"type": "disk"},
    )

    if mode == "meta_info":
        opt.dataroot_gt = "datasets/DIV2K/DIV2K_train_HR_sub"
        opt.dataroot_lq = "datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub"
        opt.meta_info = "traiNNer/data/meta_info/meta_info_DIV2K800sub_GT.txt"
        opt.filename_tmpl = "{}"
        opt.io_backend = {"type": "disk"}
    elif mode == "lmdb":
        opt.dataroot_gt = "datasets/DIV2K/DIV2K_train_HR_sub.lmdb"
        opt.dataroot_lq = "datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb"
        opt.io_backend = {"type": "lmdb"}

    os.makedirs("tmp", exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=False, sampler=None)

    nrow = int(math.sqrt(opt.batch_size_per_gpu))
    padding = 2 if opt.phase == "train" else 0

    print("start...")
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data["lq"]
        gt = data["gt"]
        lq_path = data["lq_path"]
        gt_path = data["gt_path"]
        print(lq_path, gt_path)
        torchvision.utils.save_image(
            lq, f"tmp/lq_{i:03d}.png", nrow=nrow, padding=padding, normalize=False
        )
        torchvision.utils.save_image(
            gt, f"tmp/gt_{i:03d}.png", nrow=nrow, padding=padding, normalize=False
        )


if __name__ == "__main__":
    main()
