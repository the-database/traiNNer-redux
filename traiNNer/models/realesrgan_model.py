import os
import random
import sys
from os import path as osp

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    resize_pt,
)
from traiNNer.data.transforms import paired_random_crop
from traiNNer.models.sr_model import SRModel
from traiNNer.utils import RNG, DiffJPEG, get_root_logger
from traiNNer.utils.img_process_util import USMSharp, filter2d
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANModel(SRModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        self.queue_lr: Tensor | None = None
        self.queue_gt: Tensor | None = None
        self.queue_ptr = 0
        self.kernel1: Tensor | None = None
        self.kernel2: Tensor | None = None
        self.sinc_kernel: Tensor | None = None

        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.queue_size = opt.queue_size

        self.otf_debug = opt.high_order_degradations_debug
        self.otf_debug_limit = opt.high_order_degradations_debug_limit

        if self.otf_debug:
            logger = get_root_logger()
            logger.info(
                "OTF debugging enabled. LR tiles will be saved to: %s", OTF_DEBUG_PATH
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """

        assert self.lq is not None
        assert self.gt is not None

        # initialize
        b, c, h, w = self.lq.size()
        if self.queue_lr is None:
            assert self.queue_size % b == 0, (
                f"queue size {self.queue_size} should be divisible by batch size {b}"
            )
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            assert self.queue_lr is not None
            assert self.queue_gt is not None
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            assert self.queue_lr is not None
            assert self.queue_gt is not None

            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train:
            assert (
                "gt" in data
                and "kernel1" in data
                and "kernel2" in data
                and "sinc_kernel" in data
            )
            # training data synthesis
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            self.kernel1 = data["kernel1"].to(
                self.device,
                non_blocking=True,
            )
            self.kernel2 = data["kernel2"].to(
                self.device,
                non_blocking=True,
            )
            self.sinc_kernel = data["sinc_kernel"].to(
                self.device,
                non_blocking=True,
            )

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            if self.opt.lq_usm:
                usm_sharpener = USMSharp(
                    RNG.get_rng().integers(
                        *self.opt.lq_usm_radius_range, dtype=int, endpoint=True
                    )
                ).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765
                out = usm_sharpener(self.gt)
            else:
                out = self.gt

            # thick lines
            if RNG.get_rng().uniform() < self.opt.thicklines_prob:
                thick_lines = ThickLines().cuda()
                out = thick_lines(out)

            # blur
            if RNG.get_rng().uniform() < self.opt.blur_prob:
                out = filter2d(out, self.kernel1)
            # random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt.resize_prob)[
                0
            ]
            if updown_type == "up":
                scale = RNG.get_rng().uniform(1, self.opt.resize_range[1])
            elif updown_type == "down":
                scale = RNG.get_rng().uniform(self.opt.resize_range[0], 1)
            else:
                scale = 1

            if scale != 1:
                assert len(self.opt.resize_mode_list) == len(
                    self.opt.resize_mode_prob
                ), "resize_mode_list and resize_mode_prob must be the same length"
                mode = random.choices(
                    self.opt.resize_mode_list, weights=self.opt.resize_mode_prob
                )[0]

                out = resize_pt(
                    out,
                    scale_factor=scale,
                    mode=mode,
                )

            # add noise
            gray_noise_prob = self.opt.gray_noise_prob
            if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt.noise_range,
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # JPEG compression
            if RNG.get_rng().uniform() < self.opt.jpeg_prob:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range)
                out = torch.clamp(
                    out, 0, 1
                )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if RNG.get_rng().uniform() < self.opt.blur_prob2:
                out = filter2d(out, self.kernel2)
            # random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt.resize_prob2)[
                0
            ]
            if updown_type == "up":
                scale = RNG.get_rng().uniform(1, self.opt.resize_range2[1])
            elif updown_type == "down":
                scale = RNG.get_rng().uniform(self.opt.resize_range2[0], 1)
            else:
                scale = 1

            if scale != 1:
                assert len(self.opt.resize_mode_list2) == len(
                    self.opt.resize_mode_prob2
                ), "resize_mode_list2 and resize_mode_prob2 must be the same length"
                mode = random.choices(
                    self.opt.resize_mode_list2, weights=self.opt.resize_mode_prob2
                )[0]
                out = resize_pt(
                    out,
                    size=(
                        int(ori_h / self.opt.scale * scale),
                        int(ori_w / self.opt.scale * scale),
                    ),
                    mode=mode,
                )
            # add noise
            gray_noise_prob = self.opt.gray_noise_prob2
            if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt.noise_range2,
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt.poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.

            assert len(self.opt.resize_mode_list3) == len(self.opt.resize_mode_prob3), (
                "resize_mode_list3 and resize_mode_prob3 must be the same length"
            )

            mode = random.choices(
                self.opt.resize_mode_list3, weights=self.opt.resize_mode_prob3
            )[0]
            if RNG.get_rng().uniform() < 0.5:
                # resize back + the final sinc filter
                out = resize_pt(
                    out,
                    size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                    mode=mode,
                )
                out = filter2d(out, self.sinc_kernel)
                # JPEG compression
                if RNG.get_rng().uniform() < self.opt.jpeg_prob2:
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range2)
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                if RNG.get_rng().uniform() < self.opt.jpeg_prob2:
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range2)
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                out = resize_pt(
                    out,
                    size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                    mode=mode,
                )
                out = filter2d(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt.datasets["train"].gt_size
            assert gt_size is not None
            self.gt, self.lq = paired_random_crop(
                self.gt, self.lq, gt_size, self.opt.scale
            )

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            i = 1
            if self.otf_debug:
                os.makedirs(OTF_DEBUG_PATH, exist_ok=True)
                while os.path.exists(rf"{OTF_DEBUG_PATH}/{i:06d}_otf_lq.png"):
                    i += 1

                if i <= self.otf_debug_limit or self.otf_debug_limit == 0:
                    torchvision.utils.save_image(
                        self.lq,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_lq.png"),
                        padding=0,
                    )

                    torchvision.utils.save_image(
                        self.gt,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_gt.png"),
                        padding=0,
                    )

            # moa
            if self.is_train and self.batch_augment:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)
        else:
            # for paired training or validation
            assert "lq" in data
            self.lq = data["lq"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            if "gt" in data:
                self.gt = data["gt"].to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )


class ThickLines(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, original: Tensor) -> Tensor:
        # Calculate the padding needed for the convolution
        pad = self.kernel_size // 2

        # Apply a max pooling operation with the same kernel size to both images
        min_original = -F.max_pool2d(-original, self.kernel_size, padding=pad, stride=1)
        avg_original = original * 3 / 4 + min_original * 1 / 4

        return avg_original

    def blend(self, original: Tensor, usm: Tensor) -> Tensor:
        input = torch.cat((original, usm), dim=1)
        return self.conv(input)  # pyright: ignore[reportCallIssue]
