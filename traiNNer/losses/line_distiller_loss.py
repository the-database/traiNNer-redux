import os
import sys
from os import path as osp
from typing import Literal

import torch
import torchvision
from torch import Tensor, nn
from torchvision.transforms import Normalize

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(
        osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/linedistillerloss")
    )
)


@LOSS_REGISTRY.register()
class LineDistillerLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        criterion: Literal["l1", "charbonnier"] = "l1",
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.model = LineDistiller().eval()
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.debug = debug

        weights_path = osp.join(
            osp.dirname(osp.abspath(__file__)), r"line_distiller_weights.pth"
        )

        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights)
        self.loss_weight = loss_weight

        for param in self.model.parameters():
            param.requires_grad = False

        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        pred_lines = self.model(self.norm(x))
        gt_lines = self.model(self.norm(gt.detach()))
        i = 1
        if self.debug:
            os.makedirs(os.path.join(OTF_DEBUG_PATH, "pred_base"), exist_ok=True)
            os.makedirs(os.path.join(OTF_DEBUG_PATH, "pred_lines"), exist_ok=True)
            os.makedirs(os.path.join(OTF_DEBUG_PATH, "gt_base"), exist_ok=True)
            os.makedirs(os.path.join(OTF_DEBUG_PATH, "gt_lines"), exist_ok=True)

            while os.path.exists(rf"{OTF_DEBUG_PATH}/pred_base/{i:06d}.png"):
                i += 1

            torchvision.utils.save_image(
                x,
                os.path.join(OTF_DEBUG_PATH, f"pred_base/{i:06d}.png"),
                padding=0,
            )

            torchvision.utils.save_image(
                pred_lines,
                os.path.join(OTF_DEBUG_PATH, f"pred_lines/{i:06d}.png"),
                padding=0,
            )

            torchvision.utils.save_image(
                gt,
                os.path.join(OTF_DEBUG_PATH, f"gt_base/{i:06d}.png"),
                padding=0,
            )

            torchvision.utils.save_image(
                gt_lines,
                os.path.join(OTF_DEBUG_PATH, f"gt_lines/{i:06d}.png"),
                padding=0,
            )

        return self.criterion(pred_lines, gt_lines) * self.loss_weight


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.left(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.shortcut = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.left(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_shortcut: bool = False,
    ) -> None:
        super().__init__()

        self.use_shortcut = use_shortcut

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        if self.use_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        return self.left(x) + self.shortcut(x)


class LineDistiller(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channel = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
        )

        self.conv2 = nn.Sequential(
            ResidualBlockDown(64, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
        )

        self.conv3 = nn.Sequential(
            ResidualBlockDown(128, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
        )

        self.conv4 = nn.Sequential(
            ResidualBlockDown(256, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
        )

        self.conv5 = nn.Sequential(
            ResidualBlockUp(512, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
        )

        self.conv6 = nn.Sequential(
            ResidualBlockUp(256, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
        )

        self.conv7 = nn.Sequential(
            ResidualBlockUp(128, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
        )

        self.conv8 = nn.Sequential(
            ResidualBlockUp(64, 16, 32),
            ResidualBlock(32, 8, 32),
            ResidualBlock(32, 8, 32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)

        u1 = d3 + self.conv5(d4)
        u2 = d2 + self.conv6(u1)
        u3 = d1 + self.conv7(u2)
        u4 = self.conv8(u3)

        return u4
