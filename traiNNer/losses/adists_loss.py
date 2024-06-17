import math

import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch import nn
from torch.nn import functional as F
from torchvision import models
from traiNNer.losses.perceptual_loss import VGG_PATCH_SIZE
from traiNNer.utils.registry import LOSS_REGISTRY


class Downsample(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
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

    def forward(self, input):
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
# https://github.com/dingkeyan93/A-DISTS
class ADISTSLoss(torch.nn.Module):
    def __init__(self, window_size=21, resize_input=False, loss_weight=1.0):
        super().__init__()
        self.resize_input = resize_input
        self.loss_weight = loss_weight
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), Downsample(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), Downsample(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), Downsample(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), Downsample(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [3, 64, 128, 256, 512, 512]
        self.windows = nn.ParameterList()
        self.window_size = window_size
        for k in range(len(self.chns)):
            self.windows.append(
                self.create_window(self.window_size, self.window_size / 3, self.chns[k])
            )

    def compute_prob(self, feats):
        ps_list = []
        x = feats[0]
        pad = nn.ReflectionPad2d(0)
        ps_prod = torch.ones_like(x[:, 0:1, :, :])
        c0 = 1e-12
        for k in range(len(feats) - 1, -1, -1):
            try:
                x_mean = F.conv2d(
                    pad(feats[k]),
                    self.windows[k],
                    stride=1,
                    padding=0,
                    groups=feats[k].shape[1],
                )
                x_var = (
                    F.conv2d(
                        pad(feats[k] ** 2),
                        self.windows[k],
                        stride=1,
                        padding=0,
                        groups=feats[k].shape[1],
                    )
                    - x_mean**2
                )
                h, w = x_mean.shape[2], x_mean.shape[3]
                gamma = torch.mean(x_var / (x_mean + c0), dim=1, keepdim=True)
                exponent = -(gamma - gamma.mean(dim=(2, 3), keepdim=True)) / (
                    gamma.std(dim=(2, 3), keepdim=True) + c0
                )
                exponent = torch.clamp(exponent, None, 50)
                ps = 1 / (1 + torch.exp(exponent))
                ps_min, _ = ps.flatten(2).min(dim=-1, keepdim=True)
                ps_max, _ = ps.flatten(2).max(dim=-1, keepdim=True)
                ps = (ps - ps_min.unsqueeze(-1)) / (
                    ps_max.unsqueeze(-1) - ps_min.unsqueeze(-1) + c0
                )
                ps_prod = ps * F.interpolate(
                    ps_prod, size=(h, w), mode="bilinear", align_corners=True
                )
                psd_min, _ = ps_prod.flatten(2).min(dim=-1, keepdim=True)
                psd_max, _ = ps_prod.flatten(2).max(dim=-1, keepdim=True)
                ps_prod = (ps_prod - psd_min.unsqueeze(-1)) / (
                    psd_max.unsqueeze(-1) - psd_min.unsqueeze(-1) + c0
                )
            except:
                x_mean = feats[k].mean([2, 3], keepdim=True)
                x_var = ((feats[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
                h, w = x_mean.shape[2], x_mean.shape[3]
                gamma = torch.mean(x_var / (x_mean + c0), dim=1, keepdim=True)
                ps = 1 / (1 + torch.exp(-gamma))
                ps_prod = ps * F.interpolate(
                    ps_prod, size=(h, w), mode="bilinear", align_corners=True
                )

            ps_list.append(ps_prod)
        return ps_list[::-1]

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, window_sigma, channel):
        _1D_window = self.gaussian(window_size, window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def forward_once(self, x):
        if self.resize_input:
            # skip resize if dimensions already match
            if x.shape[2] != VGG_PATCH_SIZE or x.shape[3] != VGG_PATCH_SIZE:
                h = tf.resize(
                    x,
                    [VGG_PATCH_SIZE],
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                )
        else:
            h = x

        h = (h - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        if len(self.chns) == 6:
            h = self.stage5(h)
            h_relu5_3 = h
            outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        else:
            outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        return outs

    def entropy(self, feat):
        c0 = 1e-12
        b, c, h, w = feat.shape
        feat = F.normalize(F.relu(feat), dim=(2, 3))
        feat = feat.reshape(b, c, -1)
        feat = feat / (torch.sum(feat, dim=2, keepdim=True) + c0)
        weight = torch.sum(-feat * torch.log2(feat + c0), dim=2, keepdim=True)
        weight = weight / (weight.sum(dim=1, keepdim=True) + c0)
        return weight * c

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x, y):
        assert x.shape == y.shape

        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        ps_x = self.compute_prob(feats_x)

        pad = nn.ReflectionPad2d(0)
        D = 0
        weight = []
        for k in range(len(self.chns)):
            weight.append(self.entropy(feats_x[k]))
        weight = torch.concat(weight, dim=1)

        weight = weight / weight.sum(dim=(1, 2), keepdim=True)
        weight_mean = weight.mean(dim=(1, 2), keepdim=True)
        weight_std = torch.sqrt(
            ((weight - weight_mean) ** 2).mean(dim=(1, 2), keepdim=True)
        )
        weight = weight.clamp(
            min=weight_mean - 0.5 * weight_std, max=weight_mean + 0.5 * weight_std
        )
        weight = weight / weight.sum(dim=(1, 2), keepdim=True)
        weight_list = torch.split(weight, self.chns, dim=1)

        for k in range(len(self.chns) - 1, -1, -1):
            feat_x = F.normalize(feats_x[k], dim=(2, 3))
            feat_y = F.normalize(feats_y[k], dim=(2, 3))
            try:
                x_mean = F.conv2d(
                    pad(feat_x),
                    self.windows[k],
                    stride=1,
                    padding=0,
                    groups=self.chns[k],
                )
                y_mean = F.conv2d(
                    pad(feat_y),
                    self.windows[k],
                    stride=1,
                    padding=0,
                    groups=self.chns[k],
                )
                x_var = (
                    F.conv2d(
                        pad(feat_x**2),
                        self.windows[k],
                        stride=1,
                        padding=0,
                        groups=self.chns[k],
                    )
                    - x_mean**2
                )
                y_var = (
                    F.conv2d(
                        pad(feat_y**2),
                        self.windows[k],
                        stride=1,
                        padding=0,
                        groups=self.chns[k],
                    )
                    - y_mean**2
                )
                xy_cov = (
                    F.conv2d(
                        pad(feat_x * feat_y),
                        self.windows[k],
                        stride=1,
                        padding=0,
                        groups=self.chns[k],
                    )
                    - x_mean * y_mean
                )
            except:
                x_mean = feat_x.mean([2, 3], keepdim=True)
                y_mean = feat_y.mean([2, 3], keepdim=True)
                x_var = ((feat_x - x_mean) ** 2).mean([2, 3], keepdim=True)
                y_var = ((feat_y - y_mean) ** 2).mean([2, 3], keepdim=True)
                xy_cov = (feat_x * feat_y).mean([2, 3], keepdim=True) - x_mean * y_mean

            T = (2 * x_mean * y_mean + 1e-6) / (x_mean**2 + y_mean**2 + 1e-6)
            S = (2 * xy_cov + 1e-6) / (x_var + y_var + 1e-6)

            ps = ps_x[k].expand(x_mean.shape[0], x_mean.shape[1], -1, -1)
            pt = 1 - ps
            D_map = (pt * T + ps * S) * weight_list[k].unsqueeze(3)
            D = D + D_map.mean([2, 3]).sum(1)
        return (1 - D.mean()) * self.loss_weight
