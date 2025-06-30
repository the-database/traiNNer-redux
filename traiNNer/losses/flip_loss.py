# ruff: noqa
# type: ignore
"""FLIP metric functions"""
#################################################################################
# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################

# Visualizing and Communicating Errors in Rendered Images
# Ray Tracing Gems II, 2021,
# by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
# Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

# Visualizing Errors in Rendered High Dynamic Range Images
# Eurographics 2021,
# by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
# Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics 2020,
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
# Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
# Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

# Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

import sys

import numpy as np
import torch
from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FLIPLoss(nn.Module):
    """Class for computing LDR FLIP loss"""

    def __init__(
        self,
        loss_weight: float = 1.0,
        pixels_per_degree: float = (0.7 * 3840 / 0.7) * np.pi / 180,
    ) -> None:
        """Init"""
        super().__init__()
        self.qc = 0.7
        self.qf = 0.5
        self.pc = 0.4
        self.pt = 0.95
        self.eps = 1e-15
        self.pixels_per_degree = pixels_per_degree
        self.loss_weight = loss_weight

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(
        self,
        test,
        reference,
    ):
        """
        Computes the LDR-FLIP error map between two LDR images,
        assuming the images are observed at a certain number of
        pixels per degree of visual angle

        :param test: test tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
        :param reference: reference tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
        :param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer,
                                                          default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
        :return: float containing the mean FLIP error (in the range [0,1]) between the LDR reference and test images in the batch
        """
        # LDR-FLIP expects non-NaN values in [0,1] as input
        reference = torch.clamp(reference, 0, 1)
        test = torch.clamp(test, 0, 1)

        # Transform reference and test to opponent color space
        reference_opponent = color_space_transform(reference, "srgb2ycxcz")
        test_opponent = color_space_transform(test, "srgb2ycxcz")

        deltaE = compute_ldrflip(
            test_opponent,
            reference_opponent,
            self.pixels_per_degree,
            self.qc,
            self.qf,
            self.pc,
            self.pt,
            self.eps,
        )

        return torch.mean(deltaE)


def compute_ldrflip(test, reference, pixels_per_degree, qc, qf, pc, pt, eps):
    """
    Computes the LDR-FLIP error map between two LDR images,
    assuming the images are observed at a certain number of
    pixels per degree of visual angle

    :param reference: reference tensor (with NxCxHxW layout with values in the YCxCz color space)
    :param test: test tensor (with NxCxHxW layout with values in the YCxCz color space)
    :param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer,
                                                      default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
    :param qc: float describing the q_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
    :param qf: float describing the q_f exponent in the LDR-FLIP feature pipeline (see FLIP paper for details)
    :param pc: float describing the p_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
    :param pt: float describing the p_t exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
    :param eps: float containing a small value used to improve training stability
    :return: tensor containing the per-pixel FLIP errors (with Nx1xHxW layout and values in the range [0, 1]) between LDR reference and test images
    """
    # --- Color pipeline ---
    # Spatial filtering
    s_a, radius_a = generate_spatial_filter(pixels_per_degree, "A")
    s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, "RG")
    s_by, radius_by = generate_spatial_filter(pixels_per_degree, "BY")
    radius = max(radius_a, radius_rg, radius_by)
    filtered_reference = spatial_filter(reference, s_a, s_rg, s_by, radius)
    filtered_test = spatial_filter(test, s_a, s_rg, s_by, radius)

    # Perceptually Uniform Color Space
    preprocessed_reference = hunt_adjustment(
        color_space_transform(filtered_reference, "linrgb2lab")
    )
    preprocessed_test = hunt_adjustment(
        color_space_transform(filtered_test, "linrgb2lab")
    )

    # Color metric
    deltaE_hyab = hyab(preprocessed_reference, preprocessed_test, eps)
    power_deltaE_hyab = torch.pow(deltaE_hyab, qc)
    hunt_adjusted_green = hunt_adjustment(
        color_space_transform(
            torch.tensor([[[0.0]], [[1.0]], [[0.0]]]).unsqueeze(0), "linrgb2lab"
        )
    )
    hunt_adjusted_blue = hunt_adjustment(
        color_space_transform(
            torch.tensor([[[0.0]], [[0.0]], [[1.0]]]).unsqueeze(0), "linrgb2lab"
        )
    )
    cmax = torch.pow(hyab(hunt_adjusted_green, hunt_adjusted_blue, eps), qc).item()
    deltaE_c = redistribute_errors(power_deltaE_hyab, cmax, pc, pt)

    # --- Feature pipeline ---
    # Extract and normalize Yy component
    ref_y = (reference[:, 0:1, :, :] + 16) / 116
    test_y = (test[:, 0:1, :, :] + 16) / 116

    # Edge and point detection
    edges_reference = feature_detection(ref_y, pixels_per_degree, "edge")
    points_reference = feature_detection(ref_y, pixels_per_degree, "point")
    edges_test = feature_detection(test_y, pixels_per_degree, "edge")
    points_test = feature_detection(test_y, pixels_per_degree, "point")

    # Feature metric
    deltaE_f = torch.max(
        torch.abs(
            torch.norm(edges_reference, dim=1, keepdim=True)
            - torch.norm(edges_test, dim=1, keepdim=True)
        ),
        torch.abs(
            torch.norm(points_test, dim=1, keepdim=True)
            - torch.norm(points_reference, dim=1, keepdim=True)
        ),
    )
    deltaE_f = torch.clamp(deltaE_f, min=eps)  # clamp to stabilize training
    deltaE_f = torch.pow(((1 / np.sqrt(2)) * deltaE_f), qf)

    # --- Final error ---
    return torch.pow(deltaE_c, 1 - deltaE_f)


def generate_spatial_filter(pixels_per_degree, channel):
    """
    Generates spatial contrast sensitivity filters with width depending on
    the number of pixels per degree of visual angle of the observer

    :param pixels_per_degree: float indicating number of pixels per degree of visual angle
    :param channel: string describing what filter should be generated
    :yield: Filter kernel corresponding to the spatial contrast sensitivity function of the given channel and kernel's radius
    """
    a1_A = 1
    b1_A = 0.0047
    a2_A = 0
    b2_A = 1e-5  # avoid division by 0
    a1_rg = 1
    b1_rg = 0.0053
    a2_rg = 0
    b2_rg = 1e-5  # avoid division by 0
    a1_by = 34.1
    b1_by = 0.04
    a2_by = 13.5
    b2_by = 0.025
    if channel == "A":  # Achromatic CSF
        a1 = a1_A
        b1 = b1_A
        a2 = a2_A
        b2 = b2_A
    elif channel == "RG":  # Red-Green CSF
        a1 = a1_rg
        b1 = b1_rg
        a2 = a2_rg
        b2 = b2_rg
    elif channel == "BY":  # Blue-Yellow CSF
        a1 = a1_by
        b1 = b1_by
        a2 = a2_by
        b2 = b2_by

    # Determine evaluation domain
    max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
    r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
    r = int(r)
    deltaX = 1.0 / pixels_per_degree
    x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    z = (x * deltaX) ** 2 + (y * deltaX) ** 2

    # Generate weights
    g = a1 * np.sqrt(np.pi / b1) * np.exp(-(np.pi**2) * z / b1) + a2 * np.sqrt(
        np.pi / b2
    ) * np.exp(-(np.pi**2) * z / b2)
    g = g / np.sum(g)
    g = torch.Tensor(g).unsqueeze(0).unsqueeze(0).cuda()

    return g, r


def spatial_filter(img, s_a, s_rg, s_by, radius):
    """
    Filters an image with channel specific spatial contrast sensitivity functions
    and clips result to the unit cube in linear RGB

    :param img: image tensor to filter (with NxCxHxW layout in the YCxCz color space)
    :param s_a: spatial filter matrix for the achromatic channel
    :param s_rg: spatial filter matrix for the red-green channel
    :param s_by: spatial filter matrix for the blue-yellow channel
    :return: input image (with NxCxHxW layout) transformed to linear RGB after filtering with spatial contrast sensitivity functions
    """
    dim = img.size()
    # Prepare image for convolution
    img_pad = torch.zeros(
        (dim[0], dim[1], dim[2] + 2 * radius, dim[3] + 2 * radius), device="cuda"
    )
    img_pad[:, 0:1, :, :] = nn.functional.pad(
        img[:, 0:1, :, :], (radius, radius, radius, radius), mode="replicate"
    )
    img_pad[:, 1:2, :, :] = nn.functional.pad(
        img[:, 1:2, :, :], (radius, radius, radius, radius), mode="replicate"
    )
    img_pad[:, 2:3, :, :] = nn.functional.pad(
        img[:, 2:3, :, :], (radius, radius, radius, radius), mode="replicate"
    )

    # Apply Gaussian filters
    img_tilde_opponent = torch.zeros((dim[0], dim[1], dim[2], dim[3]), device="cuda")
    img_tilde_opponent[:, 0:1, :, :] = nn.functional.conv2d(
        img_pad[:, 0:1, :, :], s_a.cuda(), padding=0
    )
    img_tilde_opponent[:, 1:2, :, :] = nn.functional.conv2d(
        img_pad[:, 1:2, :, :], s_rg.cuda(), padding=0
    )
    img_tilde_opponent[:, 2:3, :, :] = nn.functional.conv2d(
        img_pad[:, 2:3, :, :], s_by.cuda(), padding=0
    )

    # Transform to linear RGB for clamp
    img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, "ycxcz2linrgb")

    # Clamp to RGB box
    return torch.clamp(img_tilde_linear_rgb, 0.0, 1.0)


def hunt_adjustment(img):
    """
    Applies Hunt-adjustment to an image

    :param img: image tensor to adjust (with NxCxHxW layout in the L*a*b* color space)
    :return: Hunt-adjusted image tensor (with NxCxHxW layout in the Hunt-adjusted L*A*B* color space)
    """
    # Extract luminance component
    L = img[:, 0:1, :, :]

    # Apply Hunt adjustment
    img_h = torch.zeros(img.size(), device="cuda")
    img_h[:, 0:1, :, :] = L
    img_h[:, 1:2, :, :] = torch.mul((0.01 * L), img[:, 1:2, :, :])
    img_h[:, 2:3, :, :] = torch.mul((0.01 * L), img[:, 2:3, :, :])

    return img_h


def hyab(reference, test, eps):
    """
    Computes the HyAB distance between reference and test images

    :param reference: reference image tensor (with NxCxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
    :param test: test image tensor (with NxCxHxW layout in the standard or Hunt-adjusted L*a*b* color space)
    :param eps: float containing a small value used to improve training stability
    :return: image tensor (with Nx1xHxW layout) containing the per-pixel HyAB distances between reference and test images
    """
    delta = reference - test
    root = torch.sqrt(torch.clamp(torch.pow(delta[:, 0:1, :, :], 2), min=eps))
    delta_norm = torch.norm(delta[:, 1:3, :, :], dim=1, keepdim=True)
    return root + delta_norm  # alternative abs to stabilize training


def redistribute_errors(power_deltaE_hyab, cmax, pc, pt):
    """
    Redistributes exponentiated HyAB errors to the [0,1] range

    :param power_deltaE_hyab: float tensor (with Nx1xHxW layout) containing the exponentiated HyAb distance
    :param cmax: float containing the exponentiated, maximum HyAB difference between two colors in Hunt-adjusted L*A*B* space
    :param pc: float containing the cmax multiplier p_c (see FLIP paper)
    :param pt: float containing the target value, p_t, for p_c * cmax (see FLIP paper)
    :return: image tensor (with Nx1xHxW layout) containing redistributed per-pixel HyAB distances (in range [0,1])
    """
    # Re-map error to 0-1 range. Values between 0 and
    # pccmax are mapped to the range [0, pt],
    # while the rest are mapped to the range (pt, 1]
    deltaE_c = torch.zeros(power_deltaE_hyab.size(), device="cuda")
    pccmax = pc * cmax
    deltaE_c = torch.where(
        power_deltaE_hyab < pccmax,
        (pt / pccmax) * power_deltaE_hyab,
        pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt),
    )

    return deltaE_c


def feature_detection(img_y, pixels_per_degree, feature_type):
    """
    Detects edges and points (features) in the achromatic image

    :param imgy: achromatic image tensor (with Nx1xHxW layout, containing normalized Y-values from YCxCz)
    :param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer
    :param feature_type: string indicating the type of feature to detect
    :return: image tensor (with Nx2xHxW layout, with values in range [0,1]) containing large values where features were detected
    """
    # Set peak to trough value (2x standard deviations) of human edge
    # detection filter
    w = 0.082

    # Compute filter radius
    sd = 0.5 * w * pixels_per_degree
    radius = int(np.ceil(3 * sd))

    # Compute 2D Gaussian
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    g = np.exp(-(x**2 + y**2) / (2 * sd * sd))

    if feature_type == "edge":  # Edge detector
        # Compute partial derivative in x-direction
        Gx = np.multiply(-x, g)
    else:  # Point detector
        # Compute second partial derivative in x-direction
        Gx = np.multiply(x**2 / (sd * sd) - 1, g)

    # Normalize positive weights to sum to 1 and negative weights to sum to -1
    negative_weights_sum = -np.sum(Gx[Gx < 0])
    positive_weights_sum = np.sum(Gx[Gx > 0])
    Gx = torch.Tensor(Gx)
    Gx = torch.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
    Gx = Gx.unsqueeze(0).unsqueeze(0).cuda()

    # Detect features
    featuresX = nn.functional.conv2d(
        nn.functional.pad(img_y, (radius, radius, radius, radius), mode="replicate"),
        Gx,
        padding=0,
    )
    featuresY = nn.functional.conv2d(
        nn.functional.pad(img_y, (radius, radius, radius, radius), mode="replicate"),
        torch.transpose(Gx, 2, 3),
        padding=0,
    )
    return torch.cat((featuresX, featuresY), dim=1)


def color_space_transform(input_color, fromSpace2toSpace):
    """
    Transforms inputs between different color spaces

    :param input_color: tensor of colors to transform (with NxCxHxW layout)
    :param fromSpace2toSpace: string describing transform
    :return: transformed tensor (with NxCxHxW layout)
    """
    dim = input_color.size()

    # Assume D65 standard illuminant
    reference_illuminant = torch.tensor(
        [[[0.950428545]], [[1.000000000]], [[1.088900371]]]
    ).cuda()
    inv_reference_illuminant = torch.tensor(
        [[[1.052156925]], [[1.000000000]], [[0.918357670]]]
    ).cuda()

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = torch.where(
            input_color > limit,
            torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
            input_color / 12.92,
        )  # clamp to stabilize training

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = torch.where(
            input_color > limit,
            1.055 * torch.pow(torch.clamp(input_color, min=limit), (1.0 / 2.4)) - 0.055,
            12.92 * input_color,
        )

    elif fromSpace2toSpace in ["linrgb2xyz", "xyz2linrgb"]:
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        if fromSpace2toSpace == "linrgb2xyz":
            a11 = 10135552 / 24577794
            a12 = 8788810 / 24577794
            a13 = 4435075 / 24577794
            a21 = 2613072 / 12288897
            a22 = 8788810 / 12288897
            a23 = 887015 / 12288897
            a31 = 1425312 / 73733382
            a32 = 8788810 / 73733382
            a33 = 70074185 / 73733382
        else:
            # Constants found by taking the inverse of the matrix
            # defined by the constants for linrgb2xyz
            a11 = 3.241003275
            a12 = -1.537398934
            a13 = -0.498615861
            a21 = -0.969224334
            a22 = 1.875930071
            a23 = 0.041554224
            a31 = 0.055639423
            a32 = -0.204011202
            a33 = 1.057148933
        A = torch.Tensor([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

        input_color = input_color.view(dim[0], dim[1], dim[2] * dim[3]).cuda()  # NC(HW)

        transformed_color = torch.matmul(A.cuda(), input_color)
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])

    elif fromSpace2toSpace == "xyz2ycxcz":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        y = 116 * input_color[:, 1:2, :, :] - 16
        cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((y, cx, cz), 1)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        cx = input_color[:, 1:2, :, :] / 500
        cz = input_color[:, 2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = torch.cat((x, y, z), 1)

        transformed_color = torch.mul(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)

        clamped_term = torch.pow(
            torch.clamp(input_color, min=delta_cube), 1.0 / 3.0
        ).to(dtype=input_color.dtype)
        div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
        input_color = torch.where(
            input_color > delta_cube, clamped_term, div
        )  # clamp to stabilize training

        L = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

        transformed_color = torch.cat((L, a, b), 1)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        a = input_color[:, 1:2, :, :] / 500
        b = input_color[:, 2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = torch.cat((x, y, z), 1)
        delta = 6 / 29
        delta_square = delta * delta
        factor = 3 * delta_square
        xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))

        transformed_color = torch.mul(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, "srgb2linrgb")
        transformed_color = color_space_transform(transformed_color, "linrgb2xyz")
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, "srgb2linrgb")
        transformed_color = color_space_transform(transformed_color, "linrgb2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2ycxcz")
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, "linrgb2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2ycxcz")
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, "srgb2linrgb")
        transformed_color = color_space_transform(transformed_color, "linrgb2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2lab")
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, "linrgb2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2lab")
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, "ycxcz2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2linrgb")
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, "lab2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2linrgb")
        transformed_color = color_space_transform(transformed_color, "linrgb2srgb")
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, "ycxcz2xyz")
        transformed_color = color_space_transform(transformed_color, "xyz2lab")
    else:
        sys.exit(f"Error: The color transform {fromSpace2toSpace} is not defined!")

    return transformed_color
