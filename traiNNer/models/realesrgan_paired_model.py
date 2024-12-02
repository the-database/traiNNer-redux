import sys
from os import path as osp

import torch

from traiNNer.models.realesrgan_model import RealESRGANModel
from traiNNer.utils import RNG
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANPairedModel(RealESRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        self.dataroot_lq_prob = opt.dataroot_lq_prob
        self.force_high_order_degradation_filename_masks = (
            opt.force_high_order_degradation_filename_masks
        )
        self.force_dataroot_lq_filename_masks = opt.force_dataroot_lq_filename_masks

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        gt_path = data["otf_gt_path"]  # type: ignore

        force_otf = False
        force_paired = False

        for m in self.force_high_order_degradation_filename_masks:
            if m in gt_path:
                force_otf = True

        if not force_otf:
            for m in self.force_dataroot_lq_filename_masks:
                if m in gt_path:
                    force_paired = True

        if force_paired or (
            not force_otf and RNG.get_rng().uniform() < self.dataroot_lq_prob
        ):
            # paired feed data
            print("paired", gt_path, force_paired)
            new_data = {
                k.replace("paired_", ""): v
                for k, v in data.items()
                if k.startswith("paired_")
            }

            assert "lq" in new_data
            self.lq = new_data["lq"].to(
                self.device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            if "gt" in new_data:
                self.gt = new_data["gt"].to(
                    self.device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )

            # moa
            if self.is_train and self.batch_augment and self.gt is not None:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)

        else:
            # OTF feed data
            print("otf", gt_path, force_otf)
            new_data = {
                k.replace("otf_", ""): v
                for k, v in data.items()
                if k.startswith("otf_")
            }
            super().feed_data(new_data)  # type: ignore
