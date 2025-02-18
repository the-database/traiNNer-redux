import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


def abs(x):
    return torch.sqrt(x[:, :, :, :, 0] ** 2 + x[:, :, :, :, 1] ** 2 + 1e-12)


def real(x):
    return x[:, :, :, :, 0]


def imag(x):
    return x[:, :, :, :, 1]


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def preprocess_lab(lab):
    L_chan, a_chan, b_chan = torch.unbind(lab, dim=2)
    # L_chan: black and white with input range [0, 100]
    # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
    # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
    return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
    # TODO This is axis=3 instead of axis=2 when deprocessing batch of images
    # ( we process individual images but deprocess batches)
    # return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
    return torch.stack(
        [(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2
    )


def rgb_to_lab(srgb):
    srgb = srgb / 255
    srgb_pixels = torch.reshape(srgb, [-1, 3])
    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
        ((srgb_pixels + 0.055) / 1.055) ** 2.4
    ) * exponential_mask

    rgb_to_xyz = (
        torch.tensor(
            [
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754])
        .type(torch.FloatTensor)
        .to(device),
    )

    epsilon = 6.0 / 29.0
    linear_mask = (
        (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    )
    exponential_mask = (
        (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)
    )
    fxfyfz_pixels = (
        xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0
    ) * linear_mask + (
        (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)
    ) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = (
        torch.tensor(
            [
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor(
        [-16.0, 0.0, 0.0]
    ).type(torch.FloatTensor).to(device)
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab):
    lab_pixels = torch.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = (
        torch.tensor(
            [
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )
    fxfyfz_pixels = torch.mm(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device),
        lab_to_fxfyfz,
    )

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)

    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
        (fxfyfz_pixels + 0.000001) ** 3
    ) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device),
    )

    xyz_to_rgb = (
        torch.tensor(
            [
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
        ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055
    ) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


def spatial_normalize(x):
    min_v = torch.min(x.view(x.shape[0], 1, -1), dim=2)[0]
    range_v = torch.max(x.view(x.shape[0], 1, -1), dim=2)[0] - min_v
    return (x - min_v.unsqueeze(2).unsqueeze(3)) / (
        range_v.unsqueeze(2).unsqueeze(3) + 1e-12
    )


def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = torch.from_numpy(g / g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels, 1, 1, 1)


def downsample(img1, img2, maxSize=256):
    _, channels, H, W = img1.shape
    f = int(max(1, np.round(min(H, W) / maxSize)))
    if f > 1:
        aveKernel = (torch.ones(channels, 1, f, f) / f**2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding=0, groups=channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding=0, groups=channels)
    return img1, img2


def extract_patches_2d(
    img, patch_shape=None, step=None, batch_first=True, keep_last_patch=False
):
    if step is None:
        step = [27, 27]
    if patch_shape is None:
        patch_shape = [64, 64]
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if img.size(2) < patch_H:
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if img.size(3) < patch_W:
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if (isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if ((img.size(2) - patch_H) % step_int[0] != 0) and keep_last_patch:
        patches_fold_H = torch.cat(
            (
                patches_fold_H,
                img[
                    :,
                    :,
                    -patch_H:,
                ]
                .permute(0, 1, 3, 2)
                .unsqueeze(2),
            ),
            dim=2,
        )
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if ((img.size(3) - patch_W) % step_int[1] != 0) and keep_last_patch:
        patches_fold_HW = torch.cat(
            (
                patches_fold_HW,
                patches_fold_H[:, :, :, -patch_W:, :]
                .permute(0, 1, 2, 4, 3)
                .unsqueeze(3),
            ),
            dim=3,
        )
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if batch_first:
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches.reshape(-1, 3, patch_H, patch_W)


def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out


def ssim(X, Y, win, get_ssim_map=False, get_cs=False, get_weight=False):
    C1 = 0.01**2
    C2 = 0.03**2

    win = win.to(X.device)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(
        cs_map
    )  # force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val


class SSIM(torch.nn.Module):
    def __init__(self, channels=3) -> None:
        super().__init__()
        self.win = fspecial_gauss(11, 1.5, channels)

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = ssim(X, Y, win=self.win)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = ssim(X, Y, win=self.win)
            return score


def ms_ssim(X, Y, win):
    if not X.shape == Y.shape:
        raise ValueError("Input images must have the same dimensions.")

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(
        X.device, dtype=X.dtype
    )

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = ssim(X, Y, win=win, get_cs=True)
        mcs.append(cs)
        padding = (X.shape[2] % 2, X.shape[3] % 2)  # TODO
        X = F.interpolate(X, scale_factor=1/2, mode="bicubic", antialias=True).clamp(0, 1)
        Y = F.interpolate(Y, scale_factor=1/2, mode="bicubic", antialias=True).clamp(0, 1)

    mcs = torch.stack(mcs, dim=0)
    msssim_val = torch.prod(
        (mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0
    )
    return msssim_val

@LOSS_REGISTRY.register()
class MSSSIMLoss(torch.nn.Module):
    def __init__(self, loss_weight: float, channels=3) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.win = fspecial_gauss(11, 1.5, channels)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = ms_ssim(X, Y, win=self.win)
            return self.loss_weight * (1 - score.mean())
        else:
            with torch.no_grad():
                score = ms_ssim(X, Y, win=self.win)
            return score
