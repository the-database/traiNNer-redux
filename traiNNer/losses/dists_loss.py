from os import path as osp

import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import models

from traiNNer.losses.perceptual_loss import VGG_PATCH_SIZE
from traiNNer.utils.registry import LOSS_REGISTRY

###################################################
# DISTS loss
# https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py
###################################################


class L2pooling(nn.Module):
    def __init__(
        self,
        channels: int,
        filter_size: int = 5,
        stride: int = 2,
        as_loss: bool = True,
        pad_off: int = 0,
    ) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, input: Tensor) -> Tensor:
        input = input**2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()


@LOSS_REGISTRY.register()
class DISTSLoss(nn.Module):
    r"""DISTS. "Image Quality Assessment: Unifying Structure and Texture Similarity":
    https://arxiv.org/abs/2004.07728

    Args:
        as_loss (bool): True to use as loss, False for metric.
            Default: True.
        loss_weight (float).
            Default: 1.0.
        load_weights (bool): loads pretrained weights for DISTS.
            Default: False.
    """

    def __init__(
        self,
        as_loss: bool = True,
        loss_weight: float = 1.0,
        load_weights: bool = True,
        use_input_norm: bool = True,
        resize_input: bool = False,
        clip_min: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.resize_input = resize_input
        self.clip_min = clip_min

        vgg_pretrained_features = models.vgg16(weights="DEFAULT").features
        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64, as_loss=as_loss))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128, as_loss=as_loss))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256, as_loss=as_loss))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512, as_loss=as_loss))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        if use_input_norm:
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            )

        self.chns = [3, 64, 128, 256, 512, 512]

        alpha_param = nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        beta_param = nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        assert isinstance(alpha_param, nn.Parameter) and isinstance(
            beta_param, nn.Parameter
        )
        self.register_parameter("alpha", alpha_param)
        self.register_parameter("beta", beta_param)
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        weights_path = osp.join(osp.dirname(osp.abspath(__file__)), r"dists_weights.pt")

        if load_weights:
            if osp.exists(weights_path):
                weights = torch.load(weights_path)
            else:
                raise FileNotFoundError(weights_path)

            self.alpha.data = weights["alpha"]
            self.beta.data = weights["beta"]

    def forward_once(self, x: Tensor) -> list[Tensor]:
        if (
            self.resize_input
            and x.shape[2] != VGG_PATCH_SIZE
            or x.shape[3] != VGG_PATCH_SIZE
        ):
            # skip resize if dimensions already match
            h = tf.resize(
                x,
                [VGG_PATCH_SIZE],
                interpolation=tf.InterpolationMode.BICUBIC,
                antialias=True,
            )
        else:
            h = x

        if self.use_input_norm:
            h = (h - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = torch.tensor(0, device=x.device)
        dist2 = torch.tensor(0, device=x.device)
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            s1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * s1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * s2).sum(1, keepdim=True)

        if self.as_loss:
            out = (
                torch.clamp(1 - (dist1 + dist2).mean(), self.clip_min)
                * self.loss_weight
            )
        else:
            out = 1 - (dist1 + dist2).squeeze()

        return out
